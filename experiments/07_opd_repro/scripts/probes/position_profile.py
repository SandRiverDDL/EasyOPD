"""Position profiles of token-dump metrics: along the trajectory and within a sentence.

New script; dump loading and the top-k RKL follow grad_token_stability.py.
Q1 trajectory position: do teacher entropy / teacher NLL rise and teacher margin /
   student-teacher divergence fall toward the end of a response?
Q2 sentence position: are the first tokens of a sentence more divergent?

Controls against fake effects:
  - every trajectory has weight 1 inside a bin (per-trajectory mean first, then mean over
    trajectories), so long trajectories do not dominate;
  - relative position (deciles) reported per length bucket, and the head table only uses
    trajectories of length >= 2048, so late bins are not filled by a different population;
  - correct / wrong / truncated split;
  - "_res" = metric minus the mean of that token id in the same step (ids seen >= MIN_ID
    times), i.e. what is left after "which token it is";
  - SE = bootstrap over prompts (uid), since the 8 rollouts of a prompt are correlated.
The final stop token and <|endoftext|>/<|im_end|> are dropped.

Metrics (sampled token y unless noted): lr = log p_T(y) - log p_S(y); abs_lr = |lr|;
kl = sum_j p_S(j)(log p_S(j) - log p_T(j)) over student top-k (top-k dumps only);
ent_t / ent_s = full-vocab entropies; nll_t = -log p_T(y); margin_t = teacher p(top1) - p(top2)
log p (top-k dumps only); out_t = 1 if y is outside the teacher top-k.

Sentence segmentation (heuristic): a sentence ends after a token ending in a newline, or
ending in . ! ? when the next token starts with whitespace. Whitespace-only tokens get no
position. "e.g. x" gives a false split, "3.5" does not split.

Usage: python position_profile.py <token_dump_dir> <step> [<step> ...]
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

TOK = "/mnt/NLU_disk17/chenhaoyu02/models/Qwen3-1.7B-Base-imend"
EOS_IDS = [151643, 151645]
LEN_EDGES = [64, 1024, 2048, 4096, 8193]
ABS_EDGES = [0, 8, 32, 128, 512, 2048]
ABS_LAB = ["0-7", "8-31", "32-127", "128-511", "512-2047"]
SENT_EDGES = [0, 1, 2, 3, 5, 10, 20, 10**9]
SENT_LAB = ["0", "1", "2", "3-4", "5-9", "10-19", "20+"]
METRICS = ["lr", "abs_lr", "kl", "ent_t", "ent_s", "nll_t", "margin_t", "out_t"]
RESID = ["abs_lr", "kl", "ent_t", "nll_t"]
MIN_ID = 20
NBOOT = 300
ROOT = None
F = {}  # per-id string flags, filled in main before the pool forks


def id_flags(tok):
    # pad past len(tok): the model vocab (151936) is larger than the tokenizer's
    strs = tok.batch_decode([[i] for i in range(len(tok))]) + ["\x00"] * (152064 - len(tok))
    ws = np.array([s.strip() == "" for s in strs])
    f = {
        "ws": ws,
        "nl_any": np.array(["\n" in s for s in strs]),
        "nl_start": np.array([s.lstrip(" \t")[:1] == "\n" for s in strs]) & ~ws,
        "nl_end": np.array([s.rstrip(" \t")[-1:] == "\n" for s in strs]),
        "punct": np.array([s.rstrip()[-1:] in (".", "!", "?") and s.strip() != "" for s in strs]),
        "sp_start": np.array([s[:1].isspace() for s in strs]),
    }
    return f


def sent_k(ids):
    ws, nla, nls, nle, pu, sp = (F[k][ids] for k in ("ws", "nl_any", "nl_start", "nl_end", "punct", "sp_start"))
    k = np.full(len(ids), -1, np.int64)
    cur, n = 0, len(ids)
    for i in range(n):
        if ws[i]:
            if nla[i]:
                cur = 0
            continue
        if nls[i]:
            cur = 0
        k[i] = cur
        cur += 1
        if nle[i] or (pu[i] and (i + 1 == n or sp[i + 1])):
            cur = 0
    return k


def load(st):
    df = pd.read_parquet(f"{ROOT}/step_{st}.parquet")
    has_k = "student_topk_logp" in df.columns
    has_es = "student_entropy" in df.columns
    out = []
    for r in df.itertuples():
        ids = np.asarray(r.token_ids, dtype=np.int64)
        n = len(ids)
        if n < LEN_EDGES[0]:
            continue
        nan = np.full(n, np.nan)
        d = {"id": ids, "pos": np.arange(n),
             "s": np.asarray(r.student_logp, np.float64), "t": np.asarray(r.teacher_logp, np.float64),
             "ent_t": np.asarray(r.teacher_entropy, np.float64),
             "ent_s": np.asarray(r.student_entropy, np.float64) if has_es else nan,
             "out_t": (np.asarray(r.teacher_rank) == -1).astype(np.float64),
             "sk": sent_k(ids), "kl": nan, "margin_t": nan, "sig2": nan, "top1_diff": nan}
        if has_k:
            k = int(r.student_topk_k)
            ls = np.asarray(r.student_topk_logp, np.float64).reshape(n, k)
            lt = np.asarray(r.teacher_logp_at_s_topk, np.float64).reshape(n, k)
            p = np.exp(ls)
            d["kl"] = (p * (ls - lt)).sum(1)
            # exact RKL gradient restricted to student top-k (same as grad_token_stability.py)
            a = lt - ls
            d["sig2"] = ((p * (a - (p * a).sum(1, keepdims=True))) ** 2).sum(1)
            kt = int(r.teacher_topk_k)
            tlr = np.asarray(r.teacher_topk_logp, np.float64).reshape(n, kt)
            s_top1 = np.asarray(r.student_topk_ids).reshape(n, k)[np.arange(n), ls.argmax(1)]
            t_top1 = np.asarray(r.teacher_topk_ids).reshape(n, kt)[np.arange(n), tlr.argmax(1)]
            d["top1_diff"] = (s_top1 != t_top1).astype(np.float64)
            tl = np.sort(tlr, 1)
            # probability margin p1 - p2 (bounded); the log-p margin is heavy-tailed (top2 down to -20)
            d["margin_t"] = np.exp(tl[:, -1]) - np.exp(tl[:, -2])
        x = pd.DataFrame(d)
        trunc = bool(r.truncated)
        n_eff = n if trunc else n - 1
        x = x.iloc[:n_eff].copy()
        x["rel"] = np.minimum(x.pos.values * 10 // n_eff, 9)
        x["traj"], x["uid"], x["L"] = int(r.row), str(r.uid), n
        x["grp"] = "trunc" if trunc else ("correct" if r.score > 0.5 else "wrong")
        out.append(x)
    x = pd.concat(out, ignore_index=True)
    x = x[~x.id.isin(EOS_IDS)].copy()
    x["lr"] = x.t - x.s
    x["abs_lr"] = x.lr.abs()
    x["nll_t"] = -x.t
    g = x.groupby("id")
    for m in RESID:
        cnt = g[m].transform("count")
        x[m + "_res"] = np.where(cnt >= MIN_ID, x[m] - g[m].transform("mean"), np.nan)
    x["abs_bin"] = pd.cut(x.pos, ABS_EDGES, right=False, labels=ABS_LAB)
    x["sk_bin"] = pd.cut(x.sk, SENT_EDGES, right=False, labels=SENT_LAB)
    x["step"] = st
    return x
RNG = np.random.default_rng(0)


def boot(v, uids):
    """Mean over trajectories, SE by bootstrap over prompts (uid)."""
    v, uids = np.asarray(v, np.float64), np.asarray(uids)
    m = np.isfinite(v)
    if m.sum() < 3:
        return np.nan, np.nan
    v, u = v[m], pd.factorize(uids[m])[0]
    nu = u.max() + 1
    w = RNG.multinomial(nu, np.full(nu, 1 / nu), size=NBOOT)[:, u]
    return v.mean(), ((w * v).sum(1) / w.sum(1)).std()


def fmt(m, se):
    return f"{'-':>13s}" if not np.isfinite(m) else f"{m:+7.3f}±{se:.3f}"


def per_traj(x, col, by):
    return x.groupby(["traj", by], observed=False)[col].mean().unstack()


def report(st, x):
    meta = x.groupby("traj")[["uid", "grp", "L"]].first()
    ms = [m for m in METRICS if x[m].notna().any()]
    summ = {"step": st}
    print(f"\n######## step {st}: {len(meta)} traj ({meta.grp.value_counts().to_dict()}), {len(x)} tokens,"
          f" metrics {ms}")
    if x.margin_t.notna().any():
        print(f"  sanity: margin_t<0 frac {(x.margin_t < 0).mean():.4f}")

    print("\n[A] relative-position deciles (d0 = first 10% of tokens), mean over trajectories")
    print(f"  {'metric':9s} {'grp':8s} {'n':>4s} " + " ".join(f"{'d' + str(i):>6s}" for i in range(10)))
    for m in ms:
        for gname in ("correct", "wrong", "trunc"):
            t = per_traj(x[x.grp == gname], m, "rel")
            if len(t):
                print(f"  {m:9s} {gname:8s} {len(t):4d} " + " ".join(f"{v:6.3f}" for v in t.mean(axis=0)))

    print("\n[B] tail - early = mean(d7..d9) - mean(d1..d3) per trajectory; d0 left out (see [C])")
    rows = [(f"L {a}-{b - 1}", (meta.grp != "trunc") & (meta.L >= a) & (meta.L < b))
            for a, b in zip(LEN_EDGES, LEN_EDGES[1:])]
    rows += [(g, meta.grp == g) for g in ("correct", "wrong", "trunc")] + [("non-trunc", meta.grp != "trunc")]
    print(f"  {'traj set':12s} {'n':>4s} " + " ".join(f"{m:>13s}" for m in ms))
    dlts = {}
    for m in ms:
        t = per_traj(x, m, "rel").reindex(meta.index)
        dlts[m] = t[[7, 8, 9]].mean(axis=1) - t[[1, 2, 3]].mean(axis=1)
    for name, sel in rows:
        cells = []
        for m in ms:
            mu, se = boot(dlts[m][sel], meta.uid[sel])
            cells.append(fmt(mu, se))
            if name == "non-trunc":
                summ[f"tail_{m}"] = fmt(mu, se)
        print(f"  {name:12s} {int(sel.sum()):4d} " + " ".join(cells))

    print("\n[C] absolute head positions, trajectories with L >= 2048 only (same population in every bin)")
    sel = meta.L >= 2048
    xh = x[x.traj.isin(meta.index[sel])]
    print(f"  n traj {int(sel.sum())}")
    print(f"  {'metric':9s} " + " ".join(f"{b:>9s}" for b in ABS_LAB) + f" {'0-7 - 512-2047':>16s}")
    for m in ms:
        t = per_traj(xh, m, "abs_bin")
        mu, se = boot(t["0-7"] - t["512-2047"], meta.uid.reindex(t.index))
        summ[f"head_{m}"] = fmt(mu, se)
        print(f"  {m:9s} " + " ".join(f"{v:9.3f}" for v in t.mean(axis=0)) + f" {fmt(mu, se):>16s}")

    xs = x[x.sk >= 0].copy()
    print(f"\n[D] position k inside a sentence; {len(xs) / len(x):.0%} of tokens have a position,"
          f" {(xs.sk == 0).sum() / len(meta):.0f} sentences per trajectory")
    share = xs.sk_bin.value_counts(normalize=True).reindex(SENT_LAB)
    print(f"  {'token share':13s} " + " ".join(f"{v:7.1%}" for v in share) + f" {'k0 - k>=10':>15s}")
    xs["kc"] = np.select([xs.sk == 0, xs.sk >= 10], ["k0", "k10"], "mid")
    for m in ms + [r + "_res" for r in RESID if x[r].notna().any()]:
        t = per_traj(xs, m, "sk_bin")
        c = per_traj(xs, m, "kc")
        mu, se = boot(c["k0"] - c["k10"], meta.uid.reindex(c.index))
        summ[f"sent_{m}"] = fmt(mu, se)
        print(f"  {m:13s} " + " ".join(f"{v:7.3f}" for v in t.mean(axis=0)) + f" {fmt(mu, se):>15s}")
    return summ


def main():
    global ROOT, F
    from transformers import AutoTokenizer
    ROOT, steps = sys.argv[1], [int(s) for s in sys.argv[2:]]
    tok = AutoTokenizer.from_pretrained(TOK)
    F = id_flags(tok)
    with Pool(min(len(steps), 8)) as pool:
        ds = pool.map(load, steps)
    summ = pd.DataFrame([report(st, x) for st, x in zip(steps, ds)]).set_index("step")
    pd.set_option("display.width", 250, "display.max_columns", 50)
    for pre, title in (("tail_", "[B] tail - early, non-truncated"), ("head_", "[C] 0-7 - 512-2047"),
                       ("sent_", "[D] k0 - k>=10")):
        print(f"\n== summary over steps: {title}")
        print(summ[[c for c in summ.columns if c.startswith(pre)]].rename(columns=lambda c: c[len(pre):]).fillna("-"))


if __name__ == "__main__":
    main()
