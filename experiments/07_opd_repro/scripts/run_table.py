#!/usr/bin/env python3
"""Generate <RUNS>/RUNS.md: one row per training run plus eval scores, rebuilt from what is on disk.

New script. The train.log config parser comes from the one-off run inventory of 10-31; no other
reference. Sources, in order of trust:
  <run>/resolved.yaml        scheduler runs: source yaml (-> batch), method
  <run>/logs/train.log       the config dict verl pprints at start (last dump wins, so resumed runs
                             show the extended step count) and the `step:N` lines
  <run>/hf/global_step_*     merged checkpoints
  */eval/results.jsonl       every evaluate.py row under RUNS, joined to runs by model_path
  configs/changes.yaml       comparability rules -> 当前 / 过时(<id>)
  configs/run_notes.yaml     hand-written batch questions, doc links, notes

Stdlib + PyYAML only, so the driver's system python3 can import it from train_scheduler.py.

  python3 scripts/run_table.py [--out PATH]
"""
import argparse
import fnmatch
import glob
import json
import os
import re
import time
from collections import defaultdict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)  # experiments/07_opd_repro
REPO = os.path.dirname(os.path.dirname(EXP))
NLU = "/mnt/NLU_disk17/chenhaoyu02"
RUNS = f"{NLU}/runs/train/07_opd_repro"
RL_METHODS = {"grpo", "dapo"}  # no teacher, so no EOS issue
BASE_EOS = 151643  # <|endoftext|>
REPORT_BENCH = ["aime24", "aime25", "math500"]  # amc23 is contaminated (eval-protocol.md), not shown
RUNNING_S = 15 * 60  # train.log touched this recently and step < target -> 在跑

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PFX = re.compile(r"^\((TaskRunner|main_task)[^)]*\)\s?")
STEP = re.compile(r"(?:^|\s)step:(\d+) ")


def read_train_log(path):
    """Return (raw text of every pprinted config dict, last step seen). Resumed runs print one per start."""
    dumps, buf, depth, last = [], None, 0, None
    with open(path, errors="ignore") as f:
        for line in f:
            m = STEP.search(line)
            if m:
                last = int(m.group(1))
            line = PFX.sub("", ANSI.sub("", line.rstrip("\n")))
            if buf is None:
                i = line.find("{'actor_rollout_ref'")
                if i < 0:
                    continue
                line, buf, depth = line[i:], [], 0
            else:
                j = line.find("(TaskRunner")  # other processes' lines interleave with the dump
                if j > 0:
                    line = line[:j]
                # Ray splits long lines, so a dump line can start with "{" / ":" or be a bare
                # fragment like "8}," ; anything else unindented (warnings, progress bars) is not dump.
                elif buf and not (line.startswith((" ", "{", "}", "[", "]", ":", "'"))
                                  or (line and " " not in line and len(line) < 40)):
                    continue
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                dumps.append("\n".join(buf))
                buf = None
    return dumps, last


def _f(pat, text):
    m = re.search(pat, text, re.S)
    return m.group(1) if m else None


def _section(text, key):
    i = text.find(f"'{key}': {{")
    return text[i:] if i >= 0 else ""


def parse_cfgs(dumps):
    """Merge the dumps: later values win, but a field another process's output garbled (None) in a
    later dump does not erase an earlier value."""
    out = {}
    for t in dumps:
        for k, v in parse_cfg(t).items():
            if v not in (None, [], "None") or k not in out:
                out[k] = v
    return out


def parse_cfg(t):
    """Pull the fields RUNS.md needs out of the pprinted config. Key order is alphabetical in
    pprint, so within actor_rollout_ref the actor block comes first, then hybrid_engine/model/ref/rollout."""
    if not t:
        return {}
    cut = t.find("'hybrid_engine'")
    actor, rest = t[:cut], t[cut:]
    opd = _f(r"'opd': (\{[^}]*\})", actor) or ""
    ref = _section(rest, "ref")
    rollout = _section(rest, "rollout")
    trainer = _section(t, "trainer")
    eos_ids = _f(r"'terminal_eos_ids': \[([^\]]*)\]", opd)
    return {
        "lr": _f(r"'lr': ([^,]+),", actor),
        "dtype": _f(r"'model_dtype': '([^']+)'", actor),
        "student": _f(r"'path': '([^']+)'", _section(rest, "model")),
        "teacher": _f(r"'model': \{'path': '([^']+)'", ref) or _f(r"'path': '([^']+)'", ref),
        "n": _f(r"'n': (\d+)", rollout),
        "resp": _f(r"'max_response_length': (\d+)", t),
        "bsz": _f(r"'train_batch_size': (\d+)", t),
        "target": _f(r"'total_training_steps': (\d+)", trainer),
        "test_freq": _f(r"'test_freq': (-?\d+)", trainer),
        "train": os.path.basename(_f(r"'train_files':\s*\[\s*'([^']+)'", t) or ""),
        "val": os.path.basename(_f(r"'val_files':\s*\[\s*'([^']+)'", t) or ""),
        "terminal_stop": (_f(r"'terminal_stop': '?([^,}']+)", opd) or "None"),
        "eos_ids": [int(x) for x in re.findall(r"\d+", eos_ids or "")],
    }
def load_notes():
    p = f"{EXP}/configs/run_notes.yaml"
    d = yaml.safe_load(open(p)) if os.path.exists(p) else {}
    return d.get("batches") or {}, d.get("runs") or {}


def note_for(tag, run_notes):
    out = {}
    for pat, v in run_notes.items():
        if fnmatch.fnmatch(tag, pat):
            out.update(v or {})
    return out


def find_runs():
    """A run dir is <RUNS>/<method or batch>/<tag>/ with resolved.yaml or logs/train.log."""
    dirs = set()
    for pat in ("*/*/resolved.yaml", "*/*/logs/train.log"):
        for p in glob.glob(f"{RUNS}/{pat}"):
            d = p.split("/logs/")[0] if "/logs/" in p else os.path.dirname(p)
            if not os.path.basename(os.path.dirname(d)).startswith("_"):
                dirs.add(d)
    return sorted(dirs)


def collect_runs():
    _, run_notes = load_notes()
    runs = []
    for d in find_runs():
        tag, parent = os.path.basename(d), os.path.basename(os.path.dirname(d))
        res = {}
        if os.path.exists(f"{d}/resolved.yaml"):
            res = yaml.safe_load(open(f"{d}/resolved.yaml")) or {}
        dumps, last = [], None
        if os.path.exists(f"{d}/logs/train.log"):
            dumps, last = read_train_log(f"{d}/logs/train.log")
        c = parse_cfgs(dumps)
        method = (res.get("cfg") or {}).get("method") or parent
        src = res.get("source") or ""
        batch = os.path.basename(os.path.dirname(src)) if "/configs/exps/" in src else None
        steps = sorted(int(x.rsplit("_", 1)[1]) for x in glob.glob(f"{d}/hf/global_step_*"))
        log = f"{d}/logs/train.log"
        r = dict(dir=d, rel=os.path.relpath(d, RUNS), tag=tag, method=method, batch=batch,
                 last=last, hf=steps, mtime=os.path.getmtime(log) if os.path.exists(log) else 0, **c)
        r.update({k: v for k, v in note_for(tag, run_notes).items() if k in ("batch", "note") and v})
        if r["target"] is None:  # no config dump (crashed before printing) -> fall back to resolved.yaml
            r["target"] = str(((res.get("cfg") or {}).get("env") or {}).get("TOTAL_STEPS") or "") or None
        runs.append(r)
    for r in runs:
        if not r["batch"]:
            r["batch"] = r["tag"]  # legacy runs: the shared tag of a multi-method round is the batch
    return runs


def run_fields(r):
    """Values that configs/changes.yaml can test."""
    student = os.path.basename(r.get("student") or "")
    eos_ok = (r["method"] in RL_METHODS or "Base" not in student
              or r.get("terminal_stop") == "cross_eos" or BASE_EOS in r.get("eos_ids", []))
    try:
        lr = float(r["lr"]) if r.get("lr") else None
    except ValueError:
        lr = None
    return {"lr": lr, "dtype": r.get("dtype"), "eos_ok": eos_ok, "val": r.get("val")}


def status(r, changes, now):
    fields, stale, val_only = run_fields(r), [], []
    for ch in changes:
        v = fields.get(ch["field"])
        exp = ch["expect"]
        if v is None or (v not in exp and not (isinstance(v, float) and any(
                isinstance(e, float) and abs(v - e) < 1e-12 for e in exp))):
            (stale if ch["scope"] == "train" else val_only).append(ch["id"])
    target = int(r["target"]) if r.get("target") else None
    done = target is not None and r["last"] is not None and r["last"] >= target
    if r["batch"] == "smoke":
        s = "冒烟"
    elif not done and now - r["mtime"] < RUNNING_S:
        s = "在跑"
    elif not done:
        s = "中断"
    elif stale:
        s = "过时"
    else:
        s = "当前"
    return s, stale, val_only
def collect_evals():
    """{(model_path, max_tokens, T, top_p, results dir): {bench: (avg, k, trunc)}}. Rows of one model +
    sampling in the same results.jsonl are merged (older runs evaluated MATH-500 and AIME in separate
    calls); the same model evaluated from two places (baselines re-run per batch) stays two rows."""
    out = defaultdict(dict)
    files = set()
    for pat in ("*/eval/results.jsonl", "*/*/eval/results.jsonl", "*/*/*/eval/results.jsonl"):
        files.update(glob.glob(f"{RUNS}/{pat}"))
    for p in sorted(files):
        for line in open(p, errors="ignore"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            s = row.get("sampling") or {}
            src = os.path.relpath(os.path.dirname(os.path.dirname(p)), RUNS)
            key = (os.path.normpath(row.get("model_path", "")), s.get("max_tokens"), s.get("temperature"), s.get("top_p"), src)
            for b, v in row.items():
                if isinstance(v, dict) and "n_prompts" in v:
                    k = v.get("k")
                    out[key][b] = (v.get(f"avg@{k}"), k, v.get("trunc_frac"))
    return out


def cell(v):
    if not v:
        return ""
    avg, k, tr = v
    s = f"{avg:.1f}" if avg is not None else "?"
    return s + (f" ({tr * 100:.0f}%)" if tr is not None else "")


def md_row(xs):
    return "| " + " | ".join(str(x).replace("|", "\\|") for x in xs) + " |"


def render(runs, evals, batches, changes):
    now = time.time()
    lines = [
        "# 07_opd_repro 训练 run 总表",
        "",
        f"> 自动生成，不要手改。生成于 {time.strftime('%Y-%m-%d %H:%M')}，"
        "脚本 `experiments/07_opd_repro/scripts/run_table.py`；可比性规则 `configs/changes.yaml`，"
        "备注 `configs/run_notes.yaml`。改备注后重跑脚本。",
        "",
        "状态：**当前** = 满足 changes.yaml 里全部 train 级条目；**过时** = 训练配置和当前不同，"
        "括号里是不满足的条目 id；**中断** = 没跑到目标步数且日志 15 分钟没更新；**在跑**；**冒烟**。"
        "「验证」列标出只影响训练时验证曲线的差异（scope: val）。",
        "",
        "## 配置变更",
        "",
        md_row(["id", "日期", "影响", "内容"]), md_row(["---"] * 4),
    ]
    for ch in changes:
        lines.append(md_row([ch["id"], ch["date"], ch["scope"], " ".join(str(ch["change"]).split())]))
    groups = defaultdict(list)
    for r in runs:
        r["status"], r["stale"], r["val_only"] = status(r, changes, now)
        groups[r["batch"]].append(r)
    order = sorted(groups, key=lambda b: (b != "smoke", max(x["tag"] for x in groups[b])), reverse=True)
    lines += ["", "## 批次", "", md_row(["批次", "要回答的问题", "结论文档", "run（状态）"]), md_row(["---"] * 4)]
    for b in order:
        meta = batches.get(b) or {}
        doc = meta.get("doc") or (f"旧 README：`{meta['readme']}`" if meta.get("readme") else "无")
        stat = defaultdict(int)
        for r in groups[b]:
            stat[r["status"]] += 1
        lines.append(md_row([b, meta.get("question", ""), doc, "，".join(f"{k} {v}" for k, v in stat.items())]))
    head = ["状态", "目录", "方法", "学生 ← 教师", "题×n", "长度", "lr", "主权重", "结尾", "步数", "hf ckpt", "验证", "备注"]
    lines += ["", "## 全部 run（按批次，新在前）", "", md_row(head), md_row(["---"] * len(head))]
    for b in order:
        for r in sorted(groups[b], key=lambda x: x["tag"], reverse=True):
            st = r["status"] + (f"({','.join(r['stale'])})" if r["stale"] and r["status"] != "冒烟" else "")
            stu = os.path.basename(r.get("student") or "?")
            tea = "—" if r["method"] in RL_METHODS else os.path.basename(r.get("teacher") or "?")
            hf = f"{len(r['hf'])} 个，{r['hf'][0]}–{r['hf'][-1]}" if r["hf"] else ""
            val = (r.get("val") or "").replace(".parquet", "") + ("（旧）" if r["val_only"] else "")
            term = r.get("terminal_stop")
            if term in (None, "None"):
                term = f"mask {r['eos_ids']}" if r.get("eos_ids") else "—"
            lines.append(md_row([st, f"`{r['rel']}`", r["method"], f"{stu} ← {tea}", f"{r.get('bsz')}×{r.get('n')}",
                                 r.get("resp"), r.get("lr"), r.get("dtype"), term,
                                 f"{r['last']}/{r['target']}", hf, val, r.get("note", "")]))
    lines += render_scores(runs, evals, order, groups)
    return "\n".join(lines) + "\n"
def render_scores(runs, evals, order, groups):
    """Scores per batch; one row per (run, step, sampling). Different max_tokens never share a row."""
    by_model = defaultdict(list)
    for (mp, mt, t, tp, src), v in evals.items():
        by_model[mp].append(((mt, t, tp), src, v))
    head = ["run", "step", "max_tokens / T / top_p"] + [f"{b} avg@k (截断)" for b in REPORT_BENCH] + ["其他"]
    lines = ["", "## 评测分数", "",
             "单元格是 avg@k，括号里是截断比例。AMC23 被训练集污染，不列（见 `easyopd/docs/reference/eval-protocol.md`）。"
             "max_tokens 不同的数字不可比。"]
    used = set()
    for b in order:
        rows = []
        for r in sorted(groups[b], key=lambda x: x["tag"]):
            for s in r["hf"]:
                mp = os.path.normpath(f"{r['dir']}/hf/global_step_{s}")
                for samp, _, v in sorted(by_model.get(mp, []), key=lambda x: (x[0][0] or 0, x[1])):
                    used.add(mp)
                    other = "，".join(f"{k} {cell(x)}" for k, x in v.items() if k not in REPORT_BENCH + ["amc23"])
                    rows.append(md_row([f"{r['method']}/{r['tag']}", s, "/".join(map(str, samp))]
                                       + [cell(v.get(k)) for k in REPORT_BENCH] + [other]))
        if rows:
            lines += ["", f"### {b}", "", md_row(head), md_row(["---"] * len(head))] + rows
    base = sorted(mp for mp in by_model if mp not in used and "/runs/" not in mp)
    if base:
        bhead = ["模型", "评测目录"] + head[2:-1]
        lines += ["", "### 基线（未训练的模型）", "", md_row(bhead), md_row(["---"] * len(bhead))]
        for mp in base:
            for samp, src, v in sorted(by_model[mp], key=lambda x: (x[0][0] or 0, x[1])):
                lines.append(md_row([os.path.basename(mp), f"`{src}`", "/".join(map(str, samp))]
                                    + [cell(v.get(k)) for k in REPORT_BENCH]))
    return lines


def build(out=None):
    out = out or f"{RUNS}/RUNS.md"
    batches, _ = load_notes()
    changes = yaml.safe_load(open(f"{EXP}/configs/changes.yaml")) or []
    text = render(collect_runs(), collect_evals(), batches, changes)
    tmp = f"{out}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help=f"default {RUNS}/RUNS.md")
    print(build(ap.parse_args().out))


if __name__ == "__main__":
    main()
