# Copyright 2026 EasyOPD Contributors
"""Train parquets for the 07_opd_repro data ablations (few-prompt / template / WildChat).

New script; reuses ``prepare_data.row`` / ``INSTRUCTION`` so the schema and the
\\boxed{} instruction match the main DAPO train set.

- ``dapo{1,4,16}``: questions picked from ``dapo_probe512`` (512 DAPO prompts, student_init
  sampled 8x at T=1.0 / 2048 tokens, see ``_data_probe/``). The student is weak on DAPO
  (446/512 prompts 0/8), so "medium" = the highest pass counts available: ranked by
  pass count desc, ties broken by a seed-0 shuffle; the sets are nested (1 c 4 c 16).
- ``semi{1,4,16}`` (``--which semi``): few-trajectory Semi-OPD, N correct offline responses
  of the initial student with length in [1024, 2048] (``build_semi``), offline ids kept.
- ``template``: Rethinking OPD II (2609.04172 §7.1) system-hint variant: system = the
  \\boxed{} instruction, user turn empty. No ground truth (reward logged as 0).
- ``wildchat``: first user turn of English, non-toxic WildChat-1M conversations
  (one shard is enough), <= 3000 chars, de-duplicated, seed-0 sample. No ground truth.

Each set is replicated / sampled to ``--rows`` rows (>= steps * batch) so the
dataloader never runs out within one epoch.
"""

import argparse
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import INSTRUCTION, row  # noqa: E402

ROOT = "/mnt/NLU_disk17/chenhaoyu02"
PROBE_DIR = f"{ROOT}/runs/train/07_opd_repro/_data_probe"
PROBE_DETAILS = f"{PROBE_DIR}/student_init_dapo_probe_dapo_probe512_details.jsonl"
WILDCHAT_SHARD = f"{ROOT}/data/WildChat-1M/data/train-00000-of-00014.parquet"


def replicate(rows, n):
    return pd.DataFrame([dict(rows[i % len(rows)], extra_info={**rows[i % len(rows)]["extra_info"], "rep": i})
                         for i in range(n)])


def build_dapo(data_dir, sizes, n_rows):
    probe = pd.read_parquet(f"{data_dir}/dapo_probe512.parquet")
    with open(PROBE_DETAILS) as f:
        det = {d["index"]: d for d in map(json.loads, f)}
    order = list(range(len(probe)))
    random.Random(0).shuffle(order)
    passes = {i: sum(s["score"] > 0 for s in det[i]["samples"]) for i in order}
    ranked = sorted(order, key=lambda i: -passes[i])  # stable sort keeps the shuffled tie order
    out = {}
    for k in sizes:
        picked = ranked[:k]
        rows = []
        for i in picked:
            r = probe.iloc[i].to_dict()
            r["prompt"] = list(r["prompt"])
            r["extra_info"] = {**r["extra_info"], "probe_pass8": passes[i]}
            rows.append(r)
        print(f"[dapo{k}] pass@8 counts={[passes[i] for i in picked]}")
        out[f"train_dapo{k}"] = replicate(rows, n_rows)
    return out


def build_semi(src, student, sizes, n_rows, min_len, max_len):
    """Few-trajectory Semi-OPD: N fixed (prompt, offline response) pairs from ``src``
    (gen_offline_rollouts.py output), replicated to ``n_rows``. Candidates: response
    length in [min_len, max_len] tokens, finished with a stop token, and correct
    (reward_fn.score_answer); picked in a seed-0 shuffle order, so the sets are nested."""
    from transformers import AutoTokenizer

    from reward_fn import score_answer

    tok = AutoTokenizer.from_pretrained(student)
    df = pd.read_parquet(src)
    lens = df["offline_response_ids"].map(len)
    cand = df.index[(lens >= min_len) & (lens <= max_len) & (df["offline_finish_reason"] == "stop")].tolist()
    random.Random(0).shuffle(cand)
    picked = []
    for i in cand:
        r = df.loc[i]
        if score_answer(tok.decode(r["offline_response_ids"]), r["reward_model"]["ground_truth"]) >= 1.0:
            picked.append(i)
            if len(picked) == max(sizes):
                break
    assert len(picked) == max(sizes), f"only {len(picked)} correct candidates"
    print(f"[semi] window [{min_len},{max_len}] stop-finished candidates={len(cand)}; picked rows={picked}")
    out = {}
    for k in sizes:
        rows = []
        for i in picked[:k]:
            r = df.loc[i].to_dict()
            r["prompt"] = list(r["prompt"])
            r["extra_info"] = {**r["extra_info"], "src_row": int(i)}
            rows.append(r)
        print(f"[semi{k}] response lens={[len(r['offline_response_ids']) for r in rows]}")
        out[f"semi_opd/train_semi{k}_len{min_len}-{max_len}"] = replicate(rows, n_rows)
    return out


def build_template(n_rows):
    r = {"prompt": [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": ""}],
         "data_source": "template", "ability": "math",
         "reward_model": {"style": "rule", "ground_truth": ""}, "extra_info": {"index": 0}}
    return replicate([r], n_rows)


def build_wildchat(n_rows):
    df = pd.read_parquet(WILDCHAT_SHARD, columns=["conversation", "language", "toxic", "redacted"])
    df = df[(df["language"] == "English") & (~df["toxic"]) & (~df["redacted"])]
    seen, cands = set(), []
    for conv in df["conversation"]:
        if len(conv) == 0 or conv[0]["role"] != "user":
            continue
        q = conv[0]["content"].strip()
        if 10 <= len(q) <= 3000 and q not in seen:
            seen.add(q)
            cands.append(q)
    print(f"[wildchat] english non-toxic first turns: {len(cands)} unique")
    random.Random(0).shuffle(cands)
    rows = [{"prompt": [{"role": "user", "content": q}], "data_source": "wildchat", "ability": "chat",
             "reward_model": {"style": "rule", "ground_truth": ""}, "extra_info": {"index": i}}
            for i, q in enumerate(cands[:n_rows])]
    assert len(rows) == n_rows, len(rows)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=f"{ROOT}/data/opd_repro")
    ap.add_argument("--rows", type=int, default=3200)
    ap.add_argument("--which", default="dapo,template,wildchat")  # "semi" is opt-in (needs the offline parquet)
    ap.add_argument("--semi_src", default=f"{ROOT}/data/opd_repro/semi_opd/train_en__Qwen3-1.7B_nothink_t1_r4096.parquet")
    ap.add_argument("--semi_student", default=f"{ROOT}/models/Qwen3-1.7B")
    ap.add_argument("--semi_min_len", type=int, default=1024)
    ap.add_argument("--semi_max_len", type=int, default=2048)
    args = ap.parse_args()
    which = args.which.split(",")
    out = {}
    if "dapo" in which:
        out.update(build_dapo(args.data_dir, [1, 4, 16], args.rows))
    if "semi" in which:
        out.update(build_semi(args.semi_src, args.semi_student, [1, 4, 16], args.rows,
                              args.semi_min_len, args.semi_max_len))
    if "template" in which:
        out["train_template_sys"] = build_template(args.rows)
    if "wildchat" in which:
        out["train_wildchat"] = build_wildchat(args.rows)
    for name, df in out.items():
        path = f"{args.data_dir}/{name}.parquet"
        df.to_parquet(path)
        print(f"[write] {path} rows={len(df)} uniq_prompts={df['prompt'].map(lambda p: json.dumps(list(p))).nunique()}")


if __name__ == "__main__":
    main()
