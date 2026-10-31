#!/usr/bin/env python
# Copyright 2026 EasyOPD Contributors
"""Browse EasyOPD token dumps (``easyopd/opd/analysis.py``) in a local web viewer.

New file, no prior implementation in the repo (dump reading follows
``experiments/07_opd_repro/scripts/analyze_rock_tokens.py``).

    python easyopd/viz/token_view.py [--root <runs/train>] [--port 8765] [--tokenizer <hf dir>]

Runs are discovered as ``<root>/<exp>/<method>/<tag>/token_dump`` (fixed depth, no
recursive scan). The page picks a run and steps, lists trajectories with sortable
summary columns, and loads one trajectory's tokens on demand.

Background colour = log ratio (log p_T - log p_S), student / teacher log p, or the
training advantage. Tokens are grouped into display units so a character whose
UTF-8 bytes span several tokens is shown once; a unit's (or word's) value is the
sum over its tokens, exact for these log-domain metrics. Dumps written with
``opd.diag_teacher_entropy`` / ``opd.diag_topk`` also offer teacher entropy and the
sampled token's rank in the teacher top-k (-1 = outside); these are shown per unit as
the max over its tokens (rank: -1 wins), since a sum has no meaning for them.
"""

import argparse
import glob
import json
import os
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token_view.html")
DEFAULT_ROOT = "/mnt/NLU_disk17/chenhaoyu02/runs/train"
DEFAULT_TOKENIZER = "/mnt/NLU_disk17/chenhaoyu02/models/Qwen3-1.7B-Base-imend"
# name -> (label, kind): "div" = signed, diverging colours; "seq" = log p <= 0, darker = lower
METRICS = {
    "logratio": ("log p_T - log p_S", "div"),
    "student_logp": ("log p_S", "seq"),
    "teacher_logp": ("log p_T", "seq"),
    "adv": ("adv (训练用)", "div"),
}
# Only present in dumps written with the matching diag switch. "pos" = >= 0, darker =
# higher; "rank" = like "pos" but -1 (sampled token outside the teacher top-k) is its own colour.
OPTIONAL_METRICS = {
    "teacher_entropy": ("H_T (teacher 熵)", "pos"),
    "student_entropy": ("H_S (student 熵)", "pos"),
    "teacher_rank": ("teacher top-k 名次 (0=top1, -1=不在 top-k)", "rank"),
}
ALL_METRICS = {**METRICS, **OPTIONAL_METRICS}
TOP_MIN_COUNT = 5
TOP_N = 40
WIN = 32  # window for the degeneration column: max over windows of mean -log p_S
LOWP = np.log(0.01)

RUNS: dict[str, str] = {}  # run id "<exp>/<method>/<tag>" -> token_dump dir
TOK = None
_DEC: dict[int, str] = {}


def discover(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "*", "*", "token_dump"))):
        out[os.path.relpath(os.path.dirname(d), root)] = d
    return out


def steps_of(run):
    fs = glob.glob(os.path.join(RUNS[run], "step_*.parquet"))
    return sorted(int(re.findall(r"step_(\d+)\.parquet$", f)[0]) for f in fs)


@lru_cache(maxsize=64)
def load_step(run, step):
    path = os.path.join(RUNS[run], f"step_{step}.parquet")
    # skip the flattened top-k blocks (large, unused by the viewer)
    names = pq.read_schema(path).names
    return pd.read_parquet(path, columns=[c for c in names if "_topk" not in c])


def _arr(r):
    s = np.asarray(r.student_logp, dtype=np.float64)
    t = np.asarray(r.teacher_logp, dtype=np.float64)
    out = {"logratio": t - s, "student_logp": s, "teacher_logp": t, "adv": np.asarray(r.adv, dtype=np.float64)}
    for m in OPTIONAL_METRICS:
        if hasattr(r, m):
            out[m] = np.asarray(getattr(r, m), dtype=np.float64)
    return out


def _max_win(nll):
    if len(nll) <= WIN:
        return float(nll.mean()) if len(nll) else 0.0
    c = np.concatenate([[0.0], np.cumsum(nll)])
    return float(((c[WIN:] - c[:-WIN]) / WIN).max())


@lru_cache(maxsize=64)
def summarize_step(run, step):
    """Per-trajectory summary rows plus pooled values for colour scales / top tokens."""
    d = load_step(run, step)
    pooled = {m: [] for m in ALL_METRICS if m in METRICS or m in d.columns}
    rows, ids = [], []
    for r in d.itertuples():
        v, n = _arr(r), int(r.length)
        lr, s = v["logratio"], v["student_logp"]
        for m in pooled:
            pooled[m].append(v[m])
        ids.append(np.asarray(r.token_ids, dtype=np.int64))
        score = float(r.score)
        rows.append({
            "step": step, "row": int(r.row), "uid": str(r.uid), "length": n,
            "score": None if np.isnan(score) else score, "truncated": bool(r.truncated),
            "mean_lr": float(lr.mean()) if n else 0.0,
            "neg_sum": float(np.minimum(lr, 0).sum()),
            "neg_frac": float((lr < -1).mean()) if n else 0.0,
            "win_nll": _max_win(-s),
            "lowp_frac": float((s < LOWP).mean()) if n else 0.0,
        })
    return rows, np.concatenate(ids), {m: np.concatenate(p) for m, p in pooled.items()}


def decode1(i):
    if i not in _DEC:
        _DEC[i] = TOK.decode([i])
    return _DEC[i]


def units(ids):
    """Split ids into display units [text, n_tokens]; n > 1 only when a char's bytes span tokens."""
    out, buf = [], []
    for i in ids:
        buf.append(i)
        piece = decode1(i) if len(buf) == 1 else TOK.decode(buf)
        if not piece.endswith("\ufffd"):
            out.append([piece, len(buf)])
            buf = []
    if buf:
        out.append([TOK.decode(buf), len(buf)])
    return out


def api_runs(_q):
    return [{"run": r, "steps": steps_of(r)} for r in RUNS]


def _run_steps(q):
    run = q.get("run", [""])[0]
    if run not in RUNS:  # only discovered runs, so the query cannot point outside --root
        raise ValueError(f"unknown run {run!r}")
    avail = set(steps_of(run))
    steps = [int(s) for s in q.get("steps", [""])[0].split(",") if s.strip()]
    bad = [s for s in steps if s not in avail]
    if bad or not steps:
        raise ValueError(f"steps {steps} not all in {sorted(avail)}")
    return run, steps


def api_trajs(q):
    run, steps = _run_steps(q)
    parts = [summarize_step(run, s) for s in steps]
    rows = [r for p in parts for r in p[0]]
    ids = np.concatenate([p[1] for p in parts])
    metrics = {}
    for m, (label, kind) in ALL_METRICS.items():
        if not all(m in p[2] for p in parts):  # optional metric missing from some selected step
            continue
        v = np.concatenate([p[2][m] for p in parts])
        x = np.abs(v) if kind == "div" else -v if kind == "seq" else v[v >= 0]
        q = float(np.quantile(x, 0.99)) if len(x) else 0.0
        metrics[m] = {"label": label, "kind": kind, "scale": max(q, 1e-3)}
    lr = np.concatenate([p[2]["logratio"] for p in parts])
    g = pd.DataFrame({"id": ids, "lr": lr}).groupby("id")["lr"].agg(["count", "sum", "mean"])
    g = g[g["count"] >= TOP_MIN_COUNT].sort_values("sum").head(TOP_N)
    top = [[int(i), decode1(int(i)), int(c), float(s), float(mu)] for i, (c, s, mu) in g.iterrows()]
    return {"rows": rows, "metrics": metrics, "top": top}


def api_traj(q):
    run, (step,) = _run_steps(q)
    row = int(q.get("row", ["-1"])[0])
    d = load_step(run, step)
    hit = d[d["row"] == row]
    if hit.empty:
        raise ValueError(f"row {row} not in step {step}")
    r = next(hit.itertuples())
    ids = [int(i) for i in r.token_ids]
    return {"ids": ids, "units": units(ids),
            "v": {m: np.round(x, 3).tolist() for m, x in _arr(r).items()}}


API = {"/api/runs": api_runs, "/api/trajs": api_trajs, "/api/traj": api_traj}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            with open(TEMPLATE, encoding="utf-8") as f:  # re-read so template edits show on reload
                return self._send(200, f.read(), "text/html; charset=utf-8")
        fn = API.get(u.path)
        if fn is None:
            return self._send(404, "not found", "text/plain")
        try:
            out = json.dumps(fn(parse_qs(u.query)), ensure_ascii=False, allow_nan=False)
        except Exception as e:  # noqa: BLE001  report to the page instead of a dropped connection
            return self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}), "application/json")
        self._send(200, out, "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        pass


def main():
    global TOK, RUNS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    RUNS = discover(a.root)
    if not RUNS:
        raise SystemExit(f"no <exp>/<method>/<tag>/token_dump under {a.root}")
    TOK = AutoTokenizer.from_pretrained(a.tokenizer)
    print(f"[token_view] {len(RUNS)} runs under {a.root}; http://{a.host}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
