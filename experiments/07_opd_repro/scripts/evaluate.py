# Copyright 2026 EasyOPD Contributors
"""Single-node 8-GPU math eval with avg@k / pass@k (07_opd_repro).

New script. Process layout copied from ``experiments/_shared/scripts/evaluate_model.py``
(one vLLM engine per GPU, each on a shard of the prompts, results merged by the
parent); added on top: several benchmarks per engine load, k samples per prompt,
configurable temperature / top_p / max_tokens, chat-template kwargs, and the
``reward_fn.score_answer`` scorer used in training.

    avg@k  = mean over prompts of (correct samples / k)
    pass@k = fraction of prompts with at least one correct sample

Usage::

    python evaluate.py --model_path <hf_dir> --data_dir /mnt/NLU_disk17/chenhaoyu02/data/opd_repro \
        --output_dir <dir> --benchmarks aime24:32,aime25:32,amc23:32,math500:8 \
        --temperature 0.6 --top_p 0.95 --max_tokens 8192 --dp_size 8
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_benchmarks(spec: str):
    out = []
    for item in spec.split(","):
        name, _, k = item.strip().partition(":")
        out.append((name, int(k or 1)))
    return out


# ---------------------------------------------------------------------------
# worker: one GPU, all benchmarks
# ---------------------------------------------------------------------------


def worker(args):
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, HERE)
    from reward_fn import score_answer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tmpl_kwargs = json.loads(args.chat_template_kwargs)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_prompt_len + args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed + args.shard_id,
        dtype="bfloat16",
    )
    stop_ids = sorted({tok.eos_token_id, *tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])} - {None})
    # All benchmarks of this shard go into ONE llm.generate call, each prompt with its own SamplingParams
    # (its benchmark's n=k). Calling generate once per benchmark made every benchmark wait for its own
    # longest response before the next could start; one call lets continuous batching fill that tail
    # with other benchmarks' requests, so the shard pays the long-tail wait once instead of per benchmark.
    jobs = []  # (bench, k, df, idx) in request order
    prompts, sps = [], []
    for bench, k in parse_benchmarks(args.benchmarks):
        df = pd.read_parquet(os.path.join(args.data_dir, f"{bench}.parquet"))
        idx = list(range(args.shard_id, len(df), args.num_shards))
        sp = SamplingParams(
            n=k, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens,
            stop_token_ids=stop_ids,  # engine-level seed only; per-request seeds force vLLM's slow sampler
        )
        prompts += [
            tok.apply_chat_template([dict(m) for m in df.iloc[i]["prompt"]], tokenize=False, add_generation_prompt=True, **tmpl_kwargs)
            for i in idx
        ]
        sps += [sp] * len(idx)
        jobs.append((bench, k, df, idx))
    t0 = time.time()
    outs = llm.generate(prompts, sps, use_tqdm=False) if prompts else []  # returned in input order
    assert len(outs) == len(prompts), f"got {len(outs)} outputs for {len(prompts)} prompts"
    gen_s = time.time() - t0
    results, pos = {}, 0
    for bench, k, df, idx in jobs:
        rows = []
        for i, o in zip(idx, outs[pos:pos + len(idx)]):
            assert len(o.outputs) == k, f"{bench}[{i}]: {len(o.outputs)} samples, expected {k}"
            gt = df.iloc[i]["reward_model"]["ground_truth"]
            samples = [
                {"score": score_answer(c.text, gt), "len": len(c.token_ids), "truncated": c.finish_reason == "length",
                 "text": c.text[-args.keep_chars:]}
                for c in o.outputs
            ]
            rows.append({"index": i, "ground_truth": gt, "samples": samples})
        pos += len(idx)
        results[bench] = {"rows": rows}
        print(f"[shard {args.shard_id}] {bench}: {len(rows)} prompts x {k}", flush=True)
    print(f"[shard {args.shard_id}] all benchmarks: {len(prompts)} prompts in one generate, {gen_s:.0f}s", flush=True)
    with open(args.shard_out, "w") as f:
        json.dump(results, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# parent: launch shards, merge, summarise
# ---------------------------------------------------------------------------


def summarise(rows, k):
    n = len(rows)
    sc = [[s["score"] for s in r["samples"]] for r in rows]
    lens = [s["len"] for r in rows for s in r["samples"]]
    trunc = [s["truncated"] for r in rows for s in r["samples"]]
    return {
        "n_prompts": n, "k": k,
        f"avg@{k}": 100.0 * sum(sum(x) / len(x) for x in sc) / max(n, 1),
        f"pass@{k}": 100.0 * sum(max(x) > 0 for x in sc) / max(n, 1),
        "mean_len": sum(lens) / max(len(lens), 1),
        "trunc_frac": sum(trunc) / max(len(trunc), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_dir", default="/mnt/NLU_disk17/chenhaoyu02/data/opd_repro")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag", default=None, help="row name in results.jsonl (default: model dir name)")
    ap.add_argument("--benchmarks", default="aime24:32,aime25:32,amc23:32,math500:8", help="name:k,...")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--chat_template_kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--dp_size", type=int, default=8)
    ap.add_argument("--gpu_ids", default=None,
                    help="comma-separated physical GPU ids, one per shard (default 0..dp_size-1); "
                         "needed when another job shares the node")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_chars", type=int, default=600, help="chars of each response tail kept in details")
    # worker-only
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_out", default=None)
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or os.path.basename(os.path.normpath(args.model_path))
    shard_dir = os.path.join(args.output_dir, f".shards_{tag}")
    os.makedirs(shard_dir, exist_ok=True)
    gpu_ids = args.gpu_ids.split(",") if args.gpu_ids else [str(i) for i in range(args.dp_size)]
    assert len(gpu_ids) == args.dp_size, f"--gpu_ids has {len(gpu_ids)} ids but --dp_size={args.dp_size}"
    procs = []
    for g in range(args.dp_size):
        out = os.path.join(shard_dir, f"{g}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--shard_id", str(g), "--num_shards", str(args.dp_size),
               "--shard_out", out] + [a for a in sys.argv[1:]]
        # VLLM_PORT: a per-GPU port base. Without it, concurrent evals on one node race for the same
        # random port (EADDRINUSE in TCPStore). A GPU is never shared, so gpu id makes the base unique.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_ids[g], VLLM_USE_V1="0", VLLM_WORKER_MULTIPROC_METHOD="spawn",
                   VLLM_PORT=str(31000 + 50 * int(gpu_ids[g])))
        log = open(os.path.join(shard_dir, f"{g}.log"), "w")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), out, log))
    t0 = time.time()
    failed = []
    for g, (p, out, log) in enumerate(procs):
        rc = p.wait()
        log.close()
        if rc != 0 or not os.path.exists(out):
            failed.append(g)
    if failed:
        raise SystemExit(f"[eval] shards {failed} failed, see {shard_dir}/<id>.log")

    merged = {}
    for _, out, _ in procs:
        with open(out) as f:
            for bench, r in json.load(f).items():
                merged.setdefault(bench, []).extend(r["rows"])
    summary = {"tag": tag, "model_path": args.model_path, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "wall_s": round(time.time() - t0), "sampling": {
                   "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                   "max_tokens": args.max_tokens, "chat_template_kwargs": json.loads(args.chat_template_kwargs),
                   "seed": args.seed}}
    for bench, k in parse_benchmarks(args.benchmarks):
        rows = sorted(merged[bench], key=lambda r: r["index"])
        summary[bench] = summarise(rows, k)
        with open(os.path.join(args.output_dir, f"{tag}_{bench}_details.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summary[bench]
        print(f"[eval] {tag} {bench}: avg@{k}={s[f'avg@{k}']:.2f} pass@{k}={s[f'pass@{k}']:.2f} "
              f"len={s['mean_len']:.0f} trunc={s['trunc_frac']:.3f} (n={s['n_prompts']})")
    with open(os.path.join(args.output_dir, "results.jsonl"), "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    for _, out, _ in procs:
        os.remove(out)


if __name__ == "__main__":
    main()
