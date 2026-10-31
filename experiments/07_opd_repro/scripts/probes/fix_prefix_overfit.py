# Copyright 2026 EasyOPD Contributors
"""Fixed-prefix overfit probe: why does the student/teacher gap at "two-way" positions stop shrinking?

  H-a  the prefix does not predict which of the two the teacher prefers -> train KL falls, held-out does not
  H-c  capacity / optimisation -> even a tiny train set cannot be fitted
  H-b  online noise floor -> train and held-out both fall once sampling noise is gone

New script. Reused: fork_probe.py gen shards (step-200 rollouts, student top-2 per position), its candidate
rule (`candidates`) and the "hidden at len(prompt)+t-1 predicts resp[t]" indexing of `teacher_logq`.
New: exact full-vocab reverse KL KL(p||q) training on fixed prefixes (single GPU, fp32 master + bf16 autocast,
AdamW like the verl actor), periodic train/held-out eval, and a linear probe on frozen student hiddens.

  python fix_prefix_overfit.py prep  --gen_dir D --out_dir O                 # CPU: compact data + split
  python fix_prefix_overfit.py train --out_dir O --tag T --pool big|small --positions two|all --lr 1e-5
  python fix_prefix_overfit.py probe --out_dir O
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fork_probe import candidates, read_jsonl, stop_ids_of  # noqa: E402

STUDENT = "/mnt/NLU_disk17/chenhaoyu02/runs/train/07_opd_repro/vanilla_opd/1030_0745_opd100lr1_vanilla/hf/global_step_200"
TEACHER = "/mnt/NLU_disk17/chenhaoyu02/models/Qwen3-4B"


# ---------------------------------------------------------------------------
# prep: compact data (prompt_ids, rollout ids, two-way positions) + prompt-level split
# ---------------------------------------------------------------------------


def stage_prep(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    stop_set = set(stop_ids_of(tok))
    rows = []
    for f in sorted(glob.glob(os.path.join(args.gen_dir, "gen_*.jsonl"))):
        for g in read_jsonl(f):
            seqs = []
            for r in g["rollouts"]:
                cs = candidates(r, stop_set, args)
                seqs.append({"ids": r["ids"], "score": r["score"], "two": [[t, a, b] for t, a, b, _, _ in cs]})
            rows.append({"row": g["row"], "prompt_ids": g["prompt_ids"], "seqs": seqs})
    rows.sort(key=lambda x: x["row"])
    random.Random(args.seed).shuffle(rows)
    n_ho, n_big, n_small = args.n_heldout, args.n_big, args.n_small
    split = {"heldout": [r["row"] for r in rows[:n_ho]],
             "big": [r["row"] for r in rows[n_ho:n_ho + n_big]]}
    split["small"] = split["big"][:n_small]  # small is a subset of big -> same train-eval set for both
    os.makedirs(args.out_dir, exist_ok=True)
    keep = set(split["heldout"]) | set(split["big"])
    with open(os.path.join(args.out_dir, "data.jsonl"), "w") as f:
        for r in rows:
            if r["row"] in keep:
                f.write(json.dumps(r) + "\n")
    json.dump(split, open(os.path.join(args.out_dir, "split.json"), "w"))
    for k, v in split.items():
        sub = [r for r in rows if r["row"] in set(v)]
        n_seq = sum(len(r["seqs"]) for r in sub)
        n_two = sum(len(s["two"]) for r in sub for s in r["seqs"])
        n_tok = sum(len(s["ids"]) for r in sub for s in r["seqs"])
        print(f"[prep] {k}: prompts {len(v)} seqs {n_seq} two-way positions {n_two} tokens {n_tok}")


def load_data(out_dir):
    split = json.load(open(os.path.join(out_dir, "split.json")))
    data = {r["row"]: r for r in read_jsonl(os.path.join(out_dir, "data.jsonl"))}
    return data, split


def seqs_of(data, rows):
    return [(data[r]["prompt_ids"], s) for r in rows for s in data[r]["seqs"]]


# ---------------------------------------------------------------------------
# shared: forward + KL at given positions
# ---------------------------------------------------------------------------


def load_models(args, train):
    import torch
    from transformers import AutoModelForCausalLM

    st = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.float32 if train else torch.bfloat16,
                                              attn_implementation="flash_attention_2").cuda()
    te = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                              attn_implementation="flash_attention_2").cuda().eval()
    assert st.lm_head.weight.shape[0] == te.lm_head.weight.shape[0], "vocab mismatch"
    for p in te.parameters():
        p.requires_grad_(False)
    return st, te


def hidden_idx(prompt_ids, positions):
    return [len(prompt_ids) + t - 1 for t in positions]  # hidden that predicts resp[t]


def kl_rows(st_logits, te_logits):
    """per-row exact reverse KL(p||q) over the full vocab, fp32."""
    import torch

    lp = torch.log_softmax(st_logits.float(), -1)
    lq = torch.log_softmax(te_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1), lp, lq


def fwd_at(model, prompt_ids, ids, positions):
    """(logits [n, V], hidden [n, H]) at the positions that predict resp[t], t in positions."""
    import torch

    x = torch.tensor([prompt_ids + ids], device="cuda")
    h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
    h = h[torch.tensor(hidden_idx(prompt_ids, positions), device="cuda")]
    return model.lm_head(h), h


def make_items(data, rows, max_seqs, n_all, seed):
    """fixed eval items: (prompt_ids, ids, two-way [[t,a,b]], sampled all-positions)."""
    rng = random.Random(seed)
    seqs = [x for x in seqs_of(data, rows) if x[1]["two"]]
    if max_seqs and len(seqs) > max_seqs:
        seqs = rng.sample(seqs, max_seqs)
    return [(p, s["ids"], s["two"], sorted(rng.sample(range(len(s["ids"])), min(n_all, len(s["ids"])))))
            for p, s in seqs]


def evaluate(st, te, items, margin):
    import torch

    st.eval()
    tot = {"kl_two": 0.0, "n_two": 0, "pair_kl": 0.0, "agree": 0, "n_clear": 0, "kl_all": 0.0, "n_all": 0}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for p, ids, two, allp in items:
            pos = [t for t, _, _ in two] + allp
            ls, _ = fwd_at(st, p, ids, pos)
            lt, _ = fwd_at(te, p, ids, pos)
            kl, lp, lq = kl_rows(ls, lt)
            n2 = len(two)
            tot["kl_two"] += kl[:n2].sum().item(); tot["n_two"] += n2
            tot["kl_all"] += kl[n2:].sum().item(); tot["n_all"] += len(allp)
            r = torch.arange(n2, device="cuda")
            a = torch.tensor([x[1] for x in two], device="cuda"); b = torch.tensor([x[2] for x in two], device="cuda")
            dp = lp[r, a] - lp[r, b]; dq = lq[r, a] - lq[r, b]
            # binary KL on the {a, b} pair renormalised: the "two-way gap" itself
            pa, qa = torch.sigmoid(dp), torch.sigmoid(dq)
            pk = pa * (torch.log(pa + 1e-12) - torch.log(qa + 1e-12)) + (1 - pa) * (torch.log(1 - pa + 1e-12) - torch.log(1 - qa + 1e-12))
            tot["pair_kl"] += pk.sum().item()
            clear = dq.abs() >= margin
            tot["agree"] += ((dp > 0) == (dq > 0))[clear].sum().item(); tot["n_clear"] += clear.sum().item()
    st.train()
    return {"kl_two": tot["kl_two"] / max(tot["n_two"], 1), "pair_kl": tot["pair_kl"] / max(tot["n_two"], 1),
            "pair_agree": tot["agree"] / max(tot["n_clear"], 1), "kl_all": tot["kl_all"] / max(tot["n_all"], 1),
            "n_two": tot["n_two"], "n_clear": tot["n_clear"]}


def stage_train(args):
    import torch

    torch.manual_seed(args.seed)
    data, split = load_data(args.out_dir)
    run_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "args.json"), "w"), indent=1)
    st, te = load_models(args, train=True)
    st.gradient_checkpointing_enable()
    st.train()
    opt = torch.optim.AdamW(st.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)  # verl actor defaults
    train_rows = split[args.pool]
    train_seqs = [x for x in seqs_of(data, train_rows) if x[1]["two"]]
    ev = {"train": make_items(data, train_rows, args.max_eval_seqs, args.n_all_eval, args.seed + 1),
          "heldout": make_items(data, split["heldout"], args.max_eval_seqs, args.n_all_eval, args.seed + 2)}
    print(f"[train {args.tag}] pool={args.pool} seqs={len(train_seqs)} positions={args.positions} lr={args.lr} "
          f"bs={args.bs} steps={args.steps} eval seqs {len(ev['train'])}/{len(ev['heldout'])}", flush=True)
    mf = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    rng, order, t0 = random.Random(args.seed), [], time.time()

    def log_eval(step, extra):
        rec = {"step": step, "epoch": step * args.bs / len(train_seqs), "time": round(time.time() - t0), **extra}
        for k, items in ev.items():
            for m, v in evaluate(st, te, items, args.min_tmargin).items():
                rec[f"{k}/{m}"] = v
        mf.write(json.dumps(rec) + "\n"); mf.flush()
        print(f"[eval {args.tag}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)

    log_eval(0, {})
    for step in range(1, args.steps + 1):
        batch = []
        while len(batch) < args.bs:
            if not order:
                order = list(range(len(train_seqs))); rng.shuffle(order)
            batch.append(train_seqs[order.pop()])
        work = []
        for p, s in batch:
            pos = ([t for t, _, _ in s["two"]] if args.positions == "two"
                   else sorted(rng.sample(range(len(s["ids"])), min(args.n_all_train, len(s["ids"])))))
            work.append((p, s["ids"], pos))
        n_pos = sum(len(w[2]) for w in work)
        loss_sum = 0.0
        for p, ids, pos in work:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    lt, _ = fwd_at(te, p, ids, pos)
                ls, _ = fwd_at(st, p, ids, pos)
                kl, _, _ = kl_rows(ls, lt)
            loss = kl.sum() / n_pos  # token-mean over the batch
            loss.backward()
            loss_sum += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0).item()
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            print(f"[step {args.tag}] {step} loss={loss_sum:.4f} grad_norm={gn:.3f} n_pos={n_pos} "
                  f"{(time.time() - t0) / step:.1f}s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            log_eval(step, {"loss": loss_sum, "grad_norm": gn})
    print(f"[train {args.tag}] DONE in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# probe: can a linear read-out of the frozen student hidden predict which of {a, b} the teacher prefers?
# ---------------------------------------------------------------------------


def probe_feats(st, te, data, rows, max_seqs, margin, seed):
    import torch

    rng = random.Random(seed)
    seqs = [(r, data[r]["prompt_ids"], s) for r in rows for s in data[r]["seqs"] if s["two"]]
    if len(seqs) > max_seqs:
        seqs = rng.sample(seqs, max_seqs)
    H, DP, Y, G = [], [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for r, p, s in seqs:
            pos = [t for t, _, _ in s["two"]]
            ls, h = fwd_at(st, p, s["ids"], pos)
            lt, _ = fwd_at(te, p, s["ids"], pos)
            lp, lq = torch.log_softmax(ls.float(), -1), torch.log_softmax(lt.float(), -1)
            i = torch.arange(len(pos), device="cuda")
            a = torch.tensor([x[1] for x in s["two"]], device="cuda"); b = torch.tensor([x[2] for x in s["two"]], device="cuda")
            dp, dq = lp[i, a] - lp[i, b], lq[i, a] - lq[i, b]
            H.append(h.float().cpu()); DP.append(dp.cpu()); Y.append(dq.cpu())
            G += [r] * len(pos)
    return torch.cat(H), torch.cat(DP), torch.cat(Y), G


def fit_logreg(X, y, l2, iters=200):
    import torch

    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros((), device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    pw = ((1 - y).sum() / y.sum().clamp(min=1)).detach()  # class-balanced: labels are ~83/17 for "teacher picks a"

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y, pos_weight=pw) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def acc_bal(logit, y):
    pred = (logit > 0).float()
    acc = (pred == y).float().mean().item()
    bal = 0.5 * (pred[y == 1].mean().item() + (1 - pred[y == 0]).mean().item())
    return acc, bal


def stage_probe(args):
    import torch

    data, split = load_data(args.out_dir)
    st, te = load_models(args, train=False)
    st.eval()
    tr_rows = split[args.pool]
    Htr_all, DPtr_all, DQtr_all, Gtr_all = probe_feats(st, te, data, tr_rows, args.max_probe_seqs, args.min_tmargin, args.seed)
    Hte_all, DPte_all, DQte_all, _ = probe_feats(st, te, data, split["heldout"], args.max_probe_seqs, args.min_tmargin, args.seed + 1)
    del st, te
    torch.cuda.empty_cache()
    mu, sd = Htr_all.mean(0), Htr_all.std(0) + 1e-4
    out = {}
    # label "pick": teacher prefers a (student top-1) over b, at |dq| >= margin
    # label "dir":  direction of the needed update, sign(dq - dp), at |dq - dp| >= margin (the gradient sign)
    for label in ["pick", "dir"]:
        ytr_s = DQtr_all if label == "pick" else DQtr_all - DPtr_all
        yte_s = DQte_all if label == "pick" else DQte_all - DPte_all
        mtr, mte = ytr_s.abs() >= args.min_tmargin, yte_s.abs() >= args.min_tmargin
        out[label] = run_probe(args, (Htr_all[mtr] - mu) / sd, DPtr_all[mtr], (ytr_s[mtr] > 0).float(),
                               [g for g, k in zip(Gtr_all, mtr.tolist()) if k],
                               (Hte_all[mte] - mu) / sd, DPte_all[mte], (yte_s[mte] > 0).float(), label)
    print("[probe] " + json.dumps(out), flush=True)
    os.makedirs(os.path.join(args.out_dir, "probe"), exist_ok=True)
    json.dump(out, open(os.path.join(args.out_dir, "probe", "result.json"), "w"), indent=1)


def run_probe(args, Htr, DPtr, Ytr, Gtr, Hte, DPte, Yte, label):
    import torch

    dev = "cuda"
    feats = {
        "dp_only": (DPtr[:, None], DPte[:, None]),
        "hidden": (Htr, Hte),
        "hidden+dp": (torch.cat([Htr, DPtr[:, None]], 1), torch.cat([Hte, DPte[:, None]], 1)),
    }
    # l2 chosen on a prompt-level 80/20 split of the train pool (positions of one prompt are correlated)
    rows = sorted(set(Gtr)); random.Random(args.seed).shuffle(rows)
    val_rows = set(rows[: len(rows) // 5])
    vm = torch.tensor([g in val_rows for g in Gtr])
    out = {"n_train": len(Ytr), "n_heldout": len(Yte), "train_pos_rate": Ytr.mean().item(), "heldout_pos_rate": Yte.mean().item()}
    for name, (Xtr, Xte) in feats.items():
        Xtr, Xte, ytr, yte = Xtr.to(dev), Xte.to(dev), Ytr.to(dev), Yte.to(dev)
        best = None
        for l2 in ([0.0] if name == "dp_only" else [1e-4, 1e-3, 1e-2, 1e-1]):
            w, b = fit_logreg(Xtr[~vm.to(dev)], ytr[~vm.to(dev)], l2)
            _, bal = acc_bal(Xtr[vm.to(dev)] @ w + b, ytr[vm.to(dev)])
            if best is None or bal > best[0]:
                best = (bal, l2)
        w, b = fit_logreg(Xtr, ytr, best[1])
        tr_acc, tr_bal = acc_bal(Xtr @ w + b, ytr)
        te_acc, te_bal = acc_bal(Xte @ w + b, yte)
        out[name] = {"l2": best[1], "train_acc": tr_acc, "train_bal": tr_bal, "heldout_acc": te_acc, "heldout_bal": te_bal}
        print(f"[probe {label}] {name}: l2={best[1]} n={len(Ytr)}/{len(Yte)} pos={Ytr.mean().item():.3f} "
              f"train acc {tr_acc:.3f} bal {tr_bal:.3f} | heldout acc {te_acc:.3f} bal {te_bal:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "train", "probe"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gen_dir", default="/mnt/NLU_disk17/chenhaoyu02/runs/train/07_opd_repro/fork_probe/1031_0548_fork300/shards")
    ap.add_argument("--student", default=STUDENT)
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--min_top2_mass", type=float, default=0.9)  # same two-way rule as fork_probe
    ap.add_argument("--min_p2", type=float, default=0.1)
    ap.add_argument("--min_tmargin", type=float, default=0.693, help="'clear teacher preference' for pair_agree/probe")
    ap.add_argument("--n_heldout", type=int, default=128)
    ap.add_argument("--n_big", type=int, default=256)
    ap.add_argument("--n_small", type=int, default=32)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--pool", default="big", choices=["big", "small"])
    ap.add_argument("--positions", default="two", choices=["two", "all"])
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8, help="sequences per optimizer step")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--n_all_train", type=int, default=512, help="positions=all: positions sampled per seq per step")
    ap.add_argument("--n_all_eval", type=int, default=128)
    ap.add_argument("--max_eval_seqs", type=int, default=160)
    ap.add_argument("--max_probe_seqs", type=int, default=1200)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"prep": stage_prep, "train": stage_train, "probe": stage_probe}[args.stage](args)


if __name__ == "__main__":
    main()
