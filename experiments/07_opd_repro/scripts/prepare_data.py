# Copyright 2026 EasyOPD Contributors
"""Build the train / eval parquets for 07_opd_repro.

New script (the shared ``experiments/_shared/scripts/prepare_data.py`` only knows
mixed_math_code_10k / math500 / gsm8k and has placeholder paths). Output schema
follows verl's RLHFDataset: ``prompt`` (chat list), ``data_source``,
``reward_model.ground_truth``, ``extra_info``.

- train: DAPO-Math-17k, de-duplicated by prompt (1.79M rows -> 17,398, see
  ``easyopd/docs/pitfalls.md``); the DAPO "Answer:" instructions are replaced by
  the same ``\\boxed{}`` instruction used at eval time. ``--english_only`` drops
  prompts with CJK characters (vOPD / Revisiting OPD use the English subset).
- eval: MATH-500, AMC23 (40), AIME24 (30), AIME25 (30).

Usage::

    python prepare_data.py --out_dir /mnt/NLU_disk17/chenhaoyu02/data/opd_repro [--english_only]
"""

import argparse
import json
import os
import re

import pandas as pd

ROOT = "/mnt/NLU_disk17/chenhaoyu02"
DAPO = f"{ROOT}/data/DAPO-Math-17k/data/dapo-math-17k.parquet"
BENCH = f"{ROOT}/benchmarks"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

_DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
_DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
_CJK = re.compile(r"[\u4e00-\u9fff]")


def make_prompt(question: str):
    return [{"role": "user", "content": f"{question.strip()}\n{INSTRUCTION}"}]


def row(question, answer, source, idx, **extra):
    return {
        "prompt": make_prompt(question),
        "data_source": source,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer).strip()},
        "extra_info": {"index": idx, **extra},
    }


def strip_dapo(text: str) -> str:
    if text.startswith(_DAPO_PREFIX):
        text = text[len(_DAPO_PREFIX):]
    if text.endswith(_DAPO_SUFFIX):
        text = text[: -len(_DAPO_SUFFIX)]
    return text


def build_train(english_only: bool):
    df = pd.read_parquet(DAPO)
    seen, rows, n_cjk = set(), [], 0
    for prompt, rm in zip(df["prompt"], df["reward_model"]):
        content = prompt[0]["content"]
        if content in seen:
            continue
        seen.add(content)
        q = strip_dapo(content)
        if english_only and _CJK.search(q):
            n_cjk += 1
            continue
        rows.append(row(q, rm["ground_truth"], "dapo_math", len(rows)))
    print(f"[train] DAPO rows={len(df)} unique={len(seen)} dropped_cjk={n_cjk} kept={len(rows)}")
    return pd.DataFrame(rows)


def _amc_answer(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else str(x)


def build_eval():
    out = {}
    with open(f"{BENCH}/MATH-500/test.jsonl") as f:
        items = [json.loads(line) for line in f]
    out["math500"] = [row(d["problem"], d["answer"], "math500", i, level=d.get("level")) for i, d in enumerate(items)]
    amc = pd.read_parquet(f"{BENCH}/amc23/test-00000-of-00001.parquet")
    out["amc23"] = [row(q, _amc_answer(a), "amc23", i) for i, (q, a) in enumerate(zip(amc["question"], amc["answer"]))]
    a24 = pd.read_parquet(f"{BENCH}/aime24/test-00000-of-00001.parquet")
    boxed = re.compile(r"\\boxed\{(.*)\}")
    out["aime24"] = [
        row(q, boxed.search(s).group(1), "aime24", i) for i, (q, s) in enumerate(zip(a24["problem"], a24["solution"]))
    ]
    with open(f"{BENCH}/aime25/test.jsonl") as f:
        items = [json.loads(line) for line in f]
    out["aime25"] = [row(d["problem"], d["answer"], "aime25", i) for i, d in enumerate(items)]
    expected = {"math500": 500, "amc23": 40, "aime24": 30, "aime25": 30}
    for k, v in out.items():
        assert len(v) == expected[k], (k, len(v))
    return {k: pd.DataFrame(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--english_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    name = "train_en.parquet" if args.english_only else "train.parquet"
    build_train(args.english_only).to_parquet(os.path.join(args.out_dir, name))
    for k, df in build_eval().items():
        df.to_parquet(os.path.join(args.out_dir, f"{k}.parquet"))
        print(f"[eval] {k}: {len(df)} sample answer={df.iloc[0]['reward_model']['ground_truth']!r}")
    print(f"[done] -> {args.out_dir}")


if __name__ == "__main__":
    main()
