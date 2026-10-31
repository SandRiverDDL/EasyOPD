# Copyright 2026 EasyOPD Contributors
"""Counterfactual fork probe: at a student "two-way" position, does the teacher-preferred token
change the final accuracy compared with the other top-2 candidate?

New script. Reused pieces: prompt tokenization (chat template text -> tokenizer(add_special_tokens=False)
-> prompt_token_ids) and the vLLM engine settings from gen_offline_rollouts.py / evaluate.py; the
verifier is reward_fn.score_answer (same as training). New: the 3 per-shard stages below and the stats.
Each shard is self-contained (prompts are sharded, every fork of a prompt lives in its shard), so the
16 shards on 2 nodes never synchronise; `stats` merges them at the end.

  gen    student (vLLM) samples k rollouts / prompt at the training sampling (T=1, top_p=1, 8192),
         logprobs=2 per position.
  score  keep prompts with lo <= #correct <= hi; candidate positions = student top-2 mass >= M,
         p2 >= P2, neither candidate a stop token; teacher (HF, bf16) gives log q at those positions;
         pick per prompt up to F forks (alternating "disagree" = teacher prefers student's 2nd,
         and "agree"), each from a different rollout.
  cont   for each fork and each branch token (teacher-preferred T / other O): prompt + prefix[:t] + [tok],
         n continuations, budget 8192 - t - 1, score.
  stats  merge shards; per-fork dacc = acc_T - acc_O; split-half null; true-variance estimate.

    python fork_probe.py gen   --out_dir D --shard_id i --num_shards 16 ...
    python fork_probe.py score --out_dir D --shard_id i ...
    python fork_probe.py cont  --out_dir D --shard_id i ...
    python fork_probe.py stats --out_dir D
"""

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]


def shard_path(args, stage):
    return os.path.join(args.out_dir, "shards", f"{stage}_{args.shard_id:02d}.jsonl")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic: a half-written shard never looks finished


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def make_llm(args, model_path):
    from vllm import LLM

    return LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True,
               max_model_len=args.max_prompt_len + args.max_tokens, gpu_memory_utilization=args.gpu_memory_utilization,
               seed=args.seed + args.shard_id, dtype="bfloat16", enable_prefix_caching=True)


def stop_ids_of(tok):
    return sorted({tok.eos_token_id, *tok.convert_tokens_to_ids(STOP_TOKENS)} - {None})


# ---------------------------------------------------------------------------
# gen
# ---------------------------------------------------------------------------


def stage_gen(args):
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    sys.path.insert(0, os.path.dirname(HERE))  # reward_fn.py lives in scripts/, one level up
    from reward_fn import score_answer

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    df = pd.read_parquet(args.data)
    rows_all = sorted(random.Random(args.seed).sample(range(len(df)), args.n_prompts))  # same list on every shard
    rows = rows_all[args.shard_id::args.num_shards]
    tmpl = json.loads(args.chat_template_kwargs)
    prompt_ids = [tok(tok.apply_chat_template([dict(m) for m in df.iloc[i]["prompt"]], add_generation_prompt=True,
                                              tokenize=False, **tmpl), add_special_tokens=False)["input_ids"] for i in rows]
    llm = make_llm(args, args.student)
    sp = SamplingParams(n=args.k, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.max_tokens,
                        stop_token_ids=stop_ids_of(tok), logprobs=2, detokenize=False)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sp, use_tqdm=False)
    out = []
    for i, p, o in zip(rows, prompt_ids, outs):
        gt = df.iloc[i]["reward_model"]["ground_truth"]
        rolls = []
        for c in o.outputs:
            ids = list(c.token_ids)
            # per position: [[id, logp], ...] for the student top-2 (rank order); only these are needed to pick
            # forks, and keeping more made gen shards ~3 MB / prompt on NFS
            top = [[[tid, round(lp.logprob, 4)] for tid, lp in sorted(d.items(), key=lambda kv: -kv[1].logprob)[:2]]
                   for d in c.logprobs]
            rolls.append({"ids": ids, "top": top, "finish": c.finish_reason,
                          "score": score_answer(tok.decode(ids, skip_special_tokens=True), gt)})
        out.append({"row": i, "gt": gt, "prompt_ids": p, "rollouts": rolls})
    print(f"[gen {args.shard_id}] {len(rows)} prompts x {args.k} in {time.time() - t0:.0f}s", flush=True)
    write_jsonl(shard_path(args, "gen"), out)


# ---------------------------------------------------------------------------
# score: teacher log q at candidate positions + fork selection
# ---------------------------------------------------------------------------


def candidates(roll, stop_set, args):
    """Student two-way positions: (t, a, b, pa, pb) with a = student top-1, b = top-2."""
    out = []
    for t, top in enumerate(roll["top"][:-1]):  # never the last token (nothing to continue after)
        if len(top) < 2:
            continue
        (a, la), (b, lb) = top[0], top[1]
        pa, pb = math.exp(la), math.exp(lb)
        if pa + pb >= args.min_top2_mass and pb >= args.min_p2 and a not in stop_set and b not in stop_set:
            out.append((t, a, b, pa, pb))
    return out


def teacher_logq(model, prompt_ids, resp_ids, positions):
    """log q(. | prompt + resp[:t]) for t in positions -> (logq rows [len(positions), V]) on GPU, fp32."""
    import torch

    ids = torch.tensor([prompt_ids + resp_ids], device="cuda")
    with torch.no_grad():
        h = model.model(input_ids=ids).last_hidden_state[0]  # [L, H]
        idx = torch.tensor([len(prompt_ids) + t - 1 for t in positions], device="cuda")  # hidden that predicts resp[t]
        return torch.log_softmax(model.lm_head(h[idx]).float(), dim=-1)


def stage_score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    stop_set = set(stop_ids_of(tok))
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda().eval()
    rng = random.Random(args.seed * 1000 + args.shard_id)
    gen = read_jsonl(shard_path(args, "gen"))
    forks, n_keep, n_cand, t0 = [], 0, 0, time.time()
    for g in gen:
        n_ok = sum(r["score"] > 0 for r in g["rollouts"])
        if not (args.acc_lo <= n_ok <= args.acc_hi):
            continue
        n_keep += 1
        pool = {"disagree": [], "agree": []}  # (rollout idx, cand dict)
        for ri, roll in enumerate(g["rollouts"]):
            cs = candidates(roll, stop_set, args)
            n_cand += len(cs)
            if not cs:
                continue
            lq = teacher_logq(model, g["prompt_ids"], roll["ids"], [c[0] for c in cs])
            qa = lq[torch.arange(len(cs)), torch.tensor([c[1] for c in cs])].tolist()
            qb = lq[torch.arange(len(cs)), torch.tensor([c[2] for c in cs])].tolist()
            t1v, t1i = lq.max(dim=-1)
            for (t, a, b, pa, pb), la, lb, tv, ti in zip(cs, qa, qb, t1v.tolist(), t1i.tolist()):
                m = la - lb
                kind = "agree" if m >= args.min_tmargin else "disagree" if m <= -args.min_tmargin else None
                if kind:
                    pool[kind].append((ri, {"t": t, "a": a, "b": b, "pa": pa, "pb": pb, "qa": math.exp(la),
                                            "qb": math.exp(lb), "t_top1": ti, "q_top1": math.exp(tv)}))
        used, order = set(), ["disagree", "agree"] * args.forks_per_prompt
        for kind in order:
            if sum(f["row"] == g["row"] for f in forks) >= args.forks_per_prompt:
                break
            cands = [x for x in pool[kind] if x[0] not in used]
            if not cands:
                continue
            ri, c = rng.choice(cands)
            used.add(ri)
            roll = g["rollouts"][ri]
            tok_t, tok_o = (c["a"], c["b"]) if kind == "agree" else (c["b"], c["a"])
            forks.append({"row": g["row"], "gt": g["gt"], "prompt_ids": g["prompt_ids"], "rollout": ri, "kind": kind,
                          "n_ok_of_k": n_ok, "roll_score": roll["score"], "roll_len": len(roll["ids"]),
                          "prefix": roll["ids"][:c["t"]], "tok_T": tok_t, "tok_O": tok_o, **c,
                          "str_a": tok.decode([c["a"]]), "str_b": tok.decode([c["b"]]),
                          "ctx": tok.decode(roll["ids"][max(0, c["t"] - 40):c["t"]])})
    print(f"[score {args.shard_id}] prompts kept {n_keep}/{len(gen)}, candidates {n_cand}, forks {len(forks)} "
          f"({sum(f['kind'] == 'disagree' for f in forks)} disagree) in {time.time() - t0:.0f}s", flush=True)
    write_jsonl(shard_path(args, "forks"), forks)


# ---------------------------------------------------------------------------
# cont: n continuations per branch
# ---------------------------------------------------------------------------


def stage_cont(args):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    sys.path.insert(0, os.path.dirname(HERE))  # reward_fn.py lives in scripts/, one level up
    from reward_fn import score_answer

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    forks = read_jsonl(shard_path(args, "forks"))
    llm = make_llm(args, args.student)
    reqs, sps = [], []
    for f in forks:
        budget = args.max_tokens - len(f["prefix"]) - 1
        sp = SamplingParams(n=args.n_cont, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=budget,
                            stop_token_ids=stop_ids_of(tok), detokenize=False)
        for br in ("T", "O"):
            reqs.append({"prompt_token_ids": f["prompt_ids"] + f["prefix"] + [f[f"tok_{br}"]]})
            sps.append(sp)
    t0 = time.time()
    outs = llm.generate(reqs, sps, use_tqdm=False) if reqs else []
    out = []
    for j, f in enumerate(forks):
        r = {k: v for k, v in f.items() if k not in ("prompt_ids", "prefix")}
        r["prefix_len"] = len(f["prefix"])
        for b, br in enumerate(("T", "O")):
            head = f["prefix"] + [f[f"tok_{br}"]]
            cs = outs[2 * j + b].outputs
            r[f"sc_{br}"] = [score_answer(tok.decode(head + list(c.token_ids), skip_special_tokens=True), f["gt"])
                             for c in cs]
            r[f"len_{br}"] = [len(c.token_ids) for c in cs]
            r[f"trunc_{br}"] = [c.finish_reason == "length" for c in cs]
            r[f"head32_{br}"] = [list(c.token_ids[:32]) for c in cs]  # for a later re-convergence check
        out.append(r)
    print(f"[cont {args.shard_id}] {len(forks)} forks x 2 x {args.n_cont} in {time.time() - t0:.0f}s", flush=True)
    write_jsonl(shard_path(args, "cont"), out)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def perm_p(sT, sO, rng, n_perm=2000):
    """two-sided permutation p for |mean(sT) - mean(sO)| (exchangeable under 'branch token does not matter')."""
    obs, pool, n = abs(sum(sT) / len(sT) - sum(sO) / len(sO)), list(sT) + list(sO), len(sT)
    hit = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        hit += abs(sum(pool[:n]) / n - sum(pool[n:]) / (len(pool) - n)) >= obs - 1e-12
    return (hit + 1) / (n_perm + 1)


def stage_stats(args):
    import glob
    import statistics as st

    rows = [r for p in sorted(glob.glob(os.path.join(args.out_dir, "shards", "cont_*.jsonl"))) for r in read_jsonl(p)]
    rng = random.Random(0)
    for r in rows:
        r["accT"], r["accO"] = (sum(x > 0 for x in r["sc_T"]) / len(r["sc_T"]), sum(x > 0 for x in r["sc_O"]) / len(r["sc_O"]))
        r["d"] = r["accT"] - r["accO"]
        r["p"] = perm_p([x > 0 for x in r["sc_T"]], [x > 0 for x in r["sc_O"]], rng)
        n = len(r["sc_T"])
        r["sv"] = (r["accT"] * (1 - r["accT"]) + r["accO"] * (1 - r["accO"])) / (n - 1)  # unbiased sampling var of d
    lines = [f"forks={len(rows)}  n_cont/branch={len(rows[0]['sc_T']) if rows else 0}"]
    for kind in ("all", "disagree", "agree"):
        rs = [r for r in rows if kind == "all" or r["kind"] == kind]
        if len(rs) < 2:
            continue
        ds = [r["d"] for r in rs]
        var_obs, var_samp = st.variance(ds), st.mean(r["sv"] for r in rs)
        lines.append(
            f"{kind:9s} n={len(rs):4d}  mean dacc(T-O)={st.mean(ds):+.3f}±{st.stdev(ds) / math.sqrt(len(rs)):.3f}  "
            f"Var(d)={var_obs:.4f} sampling={var_samp:.4f} -> true sd={math.sqrt(max(var_obs - var_samp, 0)):.3f}  "
            f"p<0.05: {sum(r['p'] < 0.05 for r in rs) / len(rs):.1%}  p<0.01: {sum(r['p'] < 0.01 for r in rs) / len(rs):.1%}  "
            f"|d|>=0.5: {sum(abs(d) >= 0.5 for d in ds) / len(rs):.1%}  base acc={st.mean((r['accT'] + r['accO']) / 2 for r in rs):.2f}")
    top = sorted(rows, key=lambda r: r["p"])[:15]
    lines.append("most significant forks: kind  d  p  pa/pb  qa/qb  'a'|'b'  ctx")
    for r in top:
        lines.append(f"  {r['kind']:8s} {r['d']:+.2f} {r['p']:.3f} {r['pa']:.2f}/{r['pb']:.2f} {r['qa']:.2f}/{r['qb']:.2f} "
                     f"{r['str_a']!r}|{r['str_b']!r}  ...{r['ctx'][-80:]!r}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "stats.txt"), "w") as f:
        f.write(text + "\n")
    write_jsonl(os.path.join(args.out_dir, "forks_merged.jsonl"),
                [{k: v for k, v in r.items() if not k.startswith("head32")} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "score", "cont", "stats"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--student", default="/mnt/NLU_disk17/chenhaoyu02/runs/train/07_opd_repro/vanilla_opd/"
                                          "1030_0745_opd100lr1_vanilla/hf/global_step_200")
    ap.add_argument("--teacher", default="/mnt/NLU_disk17/chenhaoyu02/models/Qwen3-4B")
    ap.add_argument("--data", default="/mnt/NLU_disk17/chenhaoyu02/data/opd_repro/train_en.parquet")
    ap.add_argument("--n_prompts", type=int, default=640, help="total over all shards")
    ap.add_argument("--k", type=int, default=8, help="rollouts per prompt in gen")
    ap.add_argument("--acc_lo", type=int, default=2, help="keep prompts with acc_lo <= #correct of k <= acc_hi")
    ap.add_argument("--acc_hi", type=int, default=6)
    ap.add_argument("--min_top2_mass", type=float, default=0.9)
    ap.add_argument("--min_p2", type=float, default=0.1)
    ap.add_argument("--min_tmargin", type=float, default=0.693, help="|log qa - log qb| (default log 2)")
    ap.add_argument("--forks_per_prompt", type=int, default=2)
    ap.add_argument("--n_cont", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--chat_template_kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()
    {"gen": stage_gen, "score": stage_score, "cont": stage_cont, "stats": stage_stats}[args.stage](args)


if __name__ == "__main__":
    main()
