"""Semi-OPD stage 1: sample one fixed response per training prompt from the *initial* student.

New script; the vLLM data-parallel sharding (one process per GPU, CUDA_VISIBLE_DEVICES + VLLM_PORT)
is copied from scripts/evaluate.py. Differences from evaluate.py:
  * prompts are tokenized exactly as verl's RLHFDataset does (chat template text ->
    tokenizer(add_special_tokens=False)), and passed to vLLM as token ids, so the training-time
    prompt ids match the ids the response was sampled after;
  * the raw sampled token ids (incl. the stop token) are stored, never re-tokenized text;
  * sampling = the training rollout's (temperature 1.0, top_p 1.0, top_k -1).
Output: the input parquet with extra columns `offline_response_ids` (list[int]) and
`offline_finish_reason`, row order unchanged. Generate once at the largest response length you
need; a shorter training MAX_RESPONSE_LEN truncates (= the prefix an online rollout would give).

    python gen_offline_rollouts.py --model_path .../Qwen3-1.7B --input .../train_en.parquet \
        --output .../semi_opd/train_en__Qwen3-1.7B_nothink_t1_r4096.parquet --max_tokens 4096 \
        --dp_size 4 --gpu_ids 4,5,6,7
"""

import argparse
import json
import os
import subprocess
import sys
import time


def worker(args):
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tmpl_kwargs = json.loads(args.chat_template_kwargs)
    df = pd.read_parquet(args.input)
    idx = list(range(args.shard_id, len(df), args.num_shards))
    prompt_ids = []
    for i in idx:
        text = tok.apply_chat_template(
            [dict(m) for m in df.iloc[i][args.prompt_key]], add_generation_prompt=True, tokenize=False, **tmpl_kwargs
        )
        prompt_ids.append(tok(text, add_special_tokens=False)["input_ids"])
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
    sp = SamplingParams(
        n=1, temperature=args.temperature, top_p=args.top_p, top_k=-1, max_tokens=args.max_tokens,
        stop_token_ids=stop_ids,
    )
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sp, use_tqdm=False)
    rows = [
        {"index": i, "ids": list(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason}
        for i, o in zip(idx, outs)
    ]
    print(f"[shard {args.shard_id}] {len(rows)} prompts in {time.time() - t0:.0f}s", flush=True)
    with open(args.shard_out, "w") as f:
        json.dump(rows, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--chat_template_kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--dp_size", type=int, default=4)
    ap.add_argument("--gpu_ids", default=None)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_out", default=None)
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    import pandas as pd

    shard_dir = args.output + ".shards"
    os.makedirs(shard_dir, exist_ok=True)
    gpu_ids = args.gpu_ids.split(",") if args.gpu_ids else [str(i) for i in range(args.dp_size)]
    assert len(gpu_ids) == args.dp_size, f"--gpu_ids has {len(gpu_ids)} ids but --dp_size={args.dp_size}"
    procs = []
    for g in range(args.dp_size):
        out = os.path.join(shard_dir, f"{g}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--shard_id", str(g),
               "--num_shards", str(args.dp_size), "--shard_out", out] + sys.argv[1:]
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
        raise SystemExit(f"[gen] shards {failed} failed, see {shard_dir}/<id>.log")

    by_index = {}
    for _, out, _ in procs:
        with open(out) as f:
            for r in json.load(f):
                by_index[r["index"]] = r
    df = pd.read_parquet(args.input)
    assert len(by_index) == len(df), f"{len(by_index)} responses for {len(df)} prompts"
    df["offline_response_ids"] = [by_index[i]["ids"] for i in range(len(df))]
    df["offline_finish_reason"] = [by_index[i]["finish"] for i in range(len(df))]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_parquet(args.output, index=False)
    lens = df["offline_response_ids"].map(len)
    meta = {"model_path": args.model_path, "input": args.input, "n": len(df), "wall_s": round(time.time() - t0),
            "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": -1,
                         "max_tokens": args.max_tokens, "chat_template_kwargs": json.loads(args.chat_template_kwargs),
                         "seed": args.seed, "dp_size": args.dp_size},
            "mean_len": float(lens.mean()), "trunc_frac": float((df["offline_finish_reason"] == "length").mean())}
    with open(args.output + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[gen] wrote {args.output}: {json.dumps(meta)}")
    for _, out, _ in procs:
        os.remove(out)


if __name__ == "__main__":
    main()
