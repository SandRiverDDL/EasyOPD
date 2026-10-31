"""Do large-gradient tokens cancel each other? CPU probe on token_dump parquet.

New script; loading / gradient proxy follow grad_proxy_share.py (same dump columns).
Sampled-token PG: d(-A log p_S(x))/d logit_x = -A(1-p), so the update on the sampled
token's own logit is u = A(1-p) (A = log p_T - log p_S here, vanilla OPD).
Per token id v, C(v) = |sum u| / sum |u| over its occurrences: 1 = every occurrence pushes
the same way, C0(v) = sqrt(sum u^2) / sum |u| is what random signs would give. C ~ C0 means
the own-logit pushes on v cancel like noise. This is a proxy: the real lm_head-row gradient
is sum u_t h_t with different hidden states h_t, and the -A p_S(v) terms from positions
where v was not sampled are not in the dump. Parameter-space cancellation needs a GPU run.

Usage: python cancel_probe.py <token_dump_dir> [<out_dir>]   (one worker process per step)
"""
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

WIN = 32
BINS = [("lr<-1", -np.inf, -1.0), ("-1<=lr<0", -1.0, 0.0), ("lr>=0", 0.0, np.inf)]
MIN_N = 5


def win_nll(lp):
    if len(lp) < WIN:
        return float(-lp.mean())
    c = np.concatenate([[0.0], np.cumsum(-lp)])
    return float(((c[WIN:] - c[:-WIN]) / WIN).max())


def load(path):
    df = pd.read_parquet(path)
    has = "teacher_entropy" in df.columns and "teacher_rank" in df.columns
    parts = []
    for r in df.itertuples():
        s = np.asarray(r.student_logp, dtype=np.float64)
        n = len(s)
        parts.append(pd.DataFrame(dict(
            id=np.asarray(r.token_ids), s=s, a=np.asarray(r.adv, dtype=np.float64),
            lr=np.asarray(r.teacher_logp, dtype=np.float64) - s,
            H=np.asarray(r.teacher_entropy, dtype=np.float64) if has else np.full(n, np.nan),
            rk=np.asarray(r.teacher_rank) if has else np.full(n, -2),
            degen=np.full(n, win_nll(s) > 5))))
    d = pd.concat(parts, ignore_index=True)
    d["p"] = np.exp(d.s)
    d["u"] = d.a * (1 - d.p)
    d["g2"] = d.u ** 2
    return d, int(df.row.nunique())


def bin_stats(d, big):
    out = []
    for name, lo, hi in BINS:
        m = (d.lr >= lo) & (d.lr < hi)
        x = d[m]
        rk = x.rk
        out.append(dict(
            bin=name, tok=m.mean(), g2=x.g2.sum() / d.g2.sum(), big_tok=(m & big).sum() / big.sum(),
            pS_med=x.p.median(), H_med=x.H.median(), rank0=(rk == 0).mean(),
            rank1_4=((rk >= 1) & (rk <= 4)).mean(), rank5_19=(rk >= 5).mean(), out_topk=(rk == -1).mean()))
    return pd.DataFrame(out)


def per_id(d):
    d = d.assign(abs_u=d.u.abs(), neg=(d.lr < -1).astype(float), out=(d.rk == -1).astype(float))
    g = d.groupby("id").agg(n=("u", "size"), sum_u=("u", "sum"), sum_abs=("abs_u", "sum"),
                            g2=("g2", "sum"), lr=("lr", "mean"), neg=("neg", "mean"),
                            pS=("p", "median"), H=("H", "median"), out=("out", "mean"))
    g["C"] = g.sum_u.abs() / g.sum_abs.clip(lower=1e-12)
    g["C0"] = np.sqrt(g.g2) / g.sum_abs.clip(lower=1e-12)
    return g


def work(path):
    step = int(re.findall(r"step_(\d+)\.parquet$", path)[0])
    d, ntraj = load(path)
    ndeg = int(d.groupby(d.degen).size().get(True, 0))
    d = d[~d.degen]
    big = d.g2 >= d.g2.quantile(0.9)
    return step, ntraj, ndeg, len(d), bin_stats(d, big), per_id(d)


def cancel_summary(g):
    g = g[g.n >= MIN_N]
    w = g.g2 / g.g2.sum()
    top = g.sort_values("g2", ascending=False)
    head = top[top.g2.cumsum() <= 0.5 * g.g2.sum() + top.g2.iloc[0]]  # ids carrying half of sum u^2
    return dict(ids=len(g), wC=float((w * g.C).sum()), wC0=float((w * g.C0).sum()),
                noise_like_g2=float(g.g2[g.C < 2 * g.C0].sum() / g.g2.sum()),
                head_ids=len(head), head_C=float(np.average(head.C, weights=head.g2)),
                head_C0=float(np.average(head.C0, weights=head.g2)))


def main():
    root = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    files = sorted(glob.glob(os.path.join(root, "step_*.parquet")),
                   key=lambda f: int(re.findall(r"step_(\d+)", f)[0]))
    with Pool(min(len(files), os.cpu_count() or 1)) as pool:
        res = pool.map(work, files)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/NLU_disk17/chenhaoyu02/models/Qwen3-1.7B-Base-imend")
    pd.set_option("display.width", 220)
    rows = []
    for step, ntraj, ndeg, ntok, bs, g in res:
        print(f"\n== step {step}: {ntraj} traj, {ntok} tok after dropping {ndeg} degenerate-traj tok ==")
        print(bs.round(3).to_string(index=False))
        cs = cancel_summary(g)
        rows.append(dict(step=step, **cs))
    print("\n== own-logit cancellation per token id (n>=5), weighted by sum u^2 ==")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    s0, g0, s1, g1 = res[0][0], res[0][-1], res[-1][0], res[-1][-1]
    j = g1.join(g0[["n", "lr", "neg"]], rsuffix="_0", how="inner")
    j = j[(j.n >= 20) & (j.n_0 >= 20)].sort_values("g2", ascending=False).head(20)
    j.insert(0, "tok", [repr(tok.decode([int(i)])) for i in j.index])
    print(f"\n== top-20 ids by sum u^2 at step {s1} (n>=20 at steps {s0} and {s1}) ==")
    cols = ["tok", "n_0", "n", "lr_0", "lr", "neg_0", "neg", "pS", "H", "out", "C", "C0"]
    print(j[cols].round(3).to_string())
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for step, *_, g in res:
            g.to_csv(os.path.join(out_dir, f"ids_step_{step}.csv"))
        print(f"\nper-id tables -> {out_dir}")


if __name__ == "__main__":
    main()
