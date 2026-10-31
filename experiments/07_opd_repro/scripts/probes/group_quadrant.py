"""Two follow-ups to position_profile.py on the same token dumps.

New script; tokens come from position_profile.load (same filters: L >= 64, final stop
token and EOS ids dropped).

[Q] teacher-entropy x student-entropy quadrants (dumps with student entropy / top-k).
    Tokens pooled over the step (shares of total mass are what the gradient sees):
    token share, share of the top-k RKL mass (kl) and of the exact top-k RKL gradient^2
    (sig2), mean lr / kl, and how often teacher top1 != student top1. Threshold tau on both
    entropies (nats); main tau = 0.5, also 0.2 / 1.0.
[G] correct vs wrong inside the same prompt: groups with >= 1 correct and >= 1 wrong,
    truncated rollouts left out. Per-trajectory features; per group
    diff = mean(wrong) - mean(correct), then mean over groups (SE over groups, which are
    independent prompts). AUC = P(feature of a wrong rollout > feature of a correct one)
    inside a group, averaged over groups. "_lres" = feature residualised on log length over
    all non-truncated trajectories of the step, since wrong rollouts are longer.
    Events: big_neg = lr < log(0.05) (teacher gives the sampled token < 1/20 of the
    student's prob); out_t = sampled token outside the teacher top-20.
    Per decile: group-mean wrong - correct of the event rate / kl.
The dump has no error location, so [G] can say whether the teacher signal separates wrong
from correct rollouts and where along the response the excess sits, not whether it comes
before the error.

Usage: python group_quadrant.py <token_dump_dir> <step> [<step> ...]
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import position_profile as P

TAUS = (0.2, 0.5, 1.0)
BIG_NEG = np.log(0.05)
FEATS = ["L", "lr", "abs_lr", "kl", "ent_t", "nll_t", "big_neg", "out_t", "top1_diff", "first_big_neg", "any_big_neg"]
DEC_FEATS = ["big_neg", "out_t", "kl", "ent_t"]
QUADS = ["T-lo S-lo", "T-lo S-hi", "T-hi S-lo", "T-hi S-hi"]


def ms(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return f"{'-':>13s}"
    return f"{v.mean():+7.3f}±{v.std(ddof=1) / np.sqrt(len(v)):.3f}"


def quadrants(x, tau):
    hi_t, hi_s = (x.ent_t >= tau).values, (x.ent_s >= tau).values
    q = pd.Series(np.select([~hi_t & ~hi_s, ~hi_t & hi_s, hi_t & ~hi_s], QUADS[:3], QUADS[3]), index=x.index)
    g = x.groupby(q)
    out = pd.DataFrame({
        "tok": g.size() / len(x),
        "tok_correct": q[x.grp == "correct"].value_counts(normalize=True),
        "tok_wrong": q[x.grp == "wrong"].value_counts(normalize=True),
        "kl_share": g.kl.sum() / x.kl.sum(),
        "sig2_share": g.sig2.sum() / x.sig2.sum(),
        "lr": g.lr.mean(), "abs_lr": g.abs_lr.mean(), "kl": g.kl.mean(),
        "top1_diff": g.top1_diff.mean(),
    }).reindex(QUADS)
    return out


def traj_feats(x):
    f = x.groupby("traj").agg(uid=("uid", "first"), grp=("grp", "first"), L=("L", "first"),
                              **{m: (m, "mean") for m in FEATS[1:9]})
    ev = x[x.big_neg > 0]
    f["first_big_neg"] = (ev.groupby("traj").pos.min() / f.L).reindex(f.index)
    f["any_big_neg"] = f.first_big_neg.notna().astype(float)
    nt = f.grp != "trunc"
    logl = np.log(f.L[nt])
    for c in FEATS[1:9]:
        y = f.loc[nt, c]
        ok = y.notna()
        if ok.sum() < 10:
            continue
        b, a = np.polyfit(logl[ok], y[ok], 1)
        f.loc[nt, c + "_lres"] = y - (a + b * logl)
    return f


def mixed_groups(f):
    f = f[f.grp != "trunc"]
    has = f.groupby("uid").grp.agg(lambda g: ("correct" in set(g)) and ("wrong" in set(g)))
    return f[f.uid.isin(has.index[has])]


def group_table(f, cols):
    rows = []
    for c in cols:
        diffs, aucs = [], []
        for _, g in f.groupby("uid"):
            w = g.loc[g.grp == "wrong", c].dropna().values
            r = g.loc[g.grp == "correct", c].dropna().values
            if len(w) == 0 or len(r) == 0:
                continue
            diffs.append(w.mean() - r.mean())
            aucs.append(((w[:, None] > r[None, :]) + 0.5 * (w[:, None] == r[None, :])).mean())
        if len(diffs) >= 3:
            rows.append((c, len(diffs), ms(diffs), ms(np.array(aucs))))
    return rows
def decile_excess(x, f, m):
    """n_groups x 10: per group, mean over wrong - mean over correct of the per-trajectory decile mean."""
    t = P.per_traj(x[x.traj.isin(f.index)], m, "rel").reindex(columns=range(10)).join(f[["uid", "grp"]])
    out = []
    for _, g in t.groupby("uid"):
        w, r = g[g.grp == "wrong"][list(range(10))], g[g.grp == "correct"][list(range(10))]
        out.append((w.mean(axis=0) - r.mean(axis=0)).values)
    return np.array(out)


def report(st, x):
    x = x.assign(big_neg=(x.lr < BIG_NEG).astype(float))
    summ = {"step": st}
    print(f"\n######## step {st}: {x.traj.nunique()} traj, {len(x)} tokens")
    if x.ent_s.notna().any():
        for tau in TAUS:
            print(f"\n[Q] entropy quadrants, tau = {tau} nats (T = teacher, S = student; shares pooled over tokens)")
            q = quadrants(x, tau)
            print(q.to_string(float_format=lambda v: f"{v:.3f}"))
            if tau == 0.5:
                for k in QUADS:
                    summ[f"q_{k}"] = f"tok {q.tok[k]:.0%} / sig2 {q.sig2_share[k]:.0%}"

    f = traj_feats(x)
    fm = mixed_groups(f)
    nw, nc = (fm.grp == "wrong").sum(), (fm.grp == "correct").sum()
    print(f"\n[G] mixed groups: {fm.uid.nunique()} of {f.uid.nunique()} prompts, {nw} wrong / {nc} correct rollouts")
    print(f"  base rates over all tokens: big_neg {x.big_neg.mean():.4f}, out_t {x.out_t.mean():.4f}")
    cols = [c for c in FEATS + [c + "_lres" for c in FEATS[1:9]] if c in fm.columns and fm[c].notna().any()]
    print(f"  {'feature':18s} {'n_grp':>5s} {'wrong - correct':>16s} {'AUC(wrong>correct)':>19s}")
    for c, n, d, a in group_table(fm, cols):
        print(f"  {c:18s} {n:5d} {d:>16s} {a:>19s}")
        if c in ("L", "lr", "kl", "big_neg", "out_t", "lr_lres", "kl_lres", "big_neg_lres"):
            summ[f"auc_{c}"] = a

    print("\n[G] per decile, wrong - correct (group mean), then tail - early = mean(d7..d9) - mean(d1..d3)")
    print(f"  {'metric':9s} " + " ".join(f"{'d' + str(i):>7s}" for i in range(10)) + f" {'tail-early':>14s}")
    for m in DEC_FEATS:
        if not x[m].notna().any():
            continue
        e = decile_excess(x, fm, m)
        te = np.nanmean(e[:, 7:10], axis=1) - np.nanmean(e[:, 1:4], axis=1)
        print(f"  {m:9s} " + " ".join(f"{v:+7.3f}" for v in np.nanmean(e, axis=0)) + f" {ms(te):>14s}")
    return summ


def main():
    from transformers import AutoTokenizer
    P.ROOT, steps = sys.argv[1], [int(s) for s in sys.argv[2:]]
    P.F = P.id_flags(AutoTokenizer.from_pretrained(P.TOK))
    with Pool(min(len(steps), 8)) as pool:
        ds = pool.map(P.load, steps)
    summ = pd.DataFrame([report(st, x) for st, x in zip(steps, ds)]).set_index("step")
    pd.set_option("display.width", 250, "display.max_columns", 50)
    for pre, title in (("auc_", "[G] AUC(wrong > correct) inside a prompt"), ("q_", "[Q] tau=0.5 token / sig2 share")):
        print(f"\n== summary over steps: {title}")
        print(summ[[c for c in summ.columns if c.startswith(pre)]].rename(columns=lambda c: c[len(pre):]).fillna("-"))


if __name__ == "__main__":
    main()
