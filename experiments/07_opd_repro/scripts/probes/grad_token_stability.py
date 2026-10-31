"""Which tokens carry the late-stage OPD gradient, and is that set stable across steps?

New script. Loading and the sampled-token proxy follow grad_proxy_share.py; the rock score
follows analyze_rock_tokens.py (R(v) = sum of loss over occurrences / N). Reads token_dump parquet.

Per position (logits space, different positions are separate variables):
  sampled proxy  g2  = (A (1 - p_y))^2,  A = log p_T(y) - log p_S(y)      (all dumps)
  expected grad  sig2 = || p ⊙ (A_j - Abar) ||^2 over student top-k        (dumps with top-k)
  topk RKL       kl  = sum_j p_j (log p_S(j) - log p_T(j)) over student top-k
sig2 is the exact gradient of KL(p_S || p_T) restricted to student top-k (mass ~0.997), i.e. the
"signal" with sampling noise removed. Parameter-space cancellation is NOT measured.

Usage: python grad_token_stability.py <token_dump_dir> <step> [<step> ...]
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

TOK = "/mnt/NLU_disk17/chenhaoyu02/models/Qwen3-1.7B-Base-imend"
ROCK = "/mnt/NLU_disk17/chenhaoyu02/oss_refs/Rock-Token/stumbling/rock.json"
EOT, IM_END = 151643, 151645
TOPN = 50  # size of the "top gradient tokens" set compared across steps


def load(args):
    root, st = args
    df = pd.read_parquet(f"{root}/step_{st}.parquet")
    has_k = "student_topk_logp" in df.columns
    cols = dict(id=[], s=[], t=[], ent_t=[], ent_s=[], sig2=[], kl=[], pos=[])
    for r in df.itertuples():
        n = len(r.token_ids)
        cols["id"].append(np.asarray(r.token_ids))
        cols["s"].append(np.asarray(r.student_logp, dtype=np.float64))
        cols["t"].append(np.asarray(r.teacher_logp, dtype=np.float64))
        cols["pos"].append(np.arange(n))
        cols["ent_t"].append(np.asarray(r.teacher_entropy, dtype=np.float64))
        es = getattr(r, "student_entropy", None) if "student_entropy" in df.columns else None
        cols["ent_s"].append(np.asarray(es, dtype=np.float64) if es is not None else np.full(n, np.nan))
        if has_k:
            k = int(r.student_topk_k)
            ls = np.asarray(r.student_topk_logp, dtype=np.float64).reshape(n, k)
            lt = np.asarray(r.teacher_logp_at_s_topk, dtype=np.float64).reshape(n, k)
            p = np.exp(ls)
            a = lt - ls
            abar = (p * a).sum(1, keepdims=True)
            cols["sig2"].append(((p * (a - abar)) ** 2).sum(1))
            cols["kl"].append(-(p * a).sum(1))
        else:
            cols["sig2"].append(np.full(n, np.nan))
            cols["kl"].append(np.full(n, np.nan))
    d = pd.DataFrame({c: np.concatenate(v) for c, v in cols.items()})
    d["lr"] = d.t - d.s
    d["g2"] = (d.lr * (1 - np.exp(d.s))) ** 2
    d["step"] = st
    return d


def conc(x, fracs=(0.001, 0.01, 0.1)):
    x = np.sort(x[np.isfinite(x)])[::-1]
    c = np.cumsum(x) / x.sum()
    return " ".join(f"top{f*100:g}%={c[max(int(len(x)*f),1)-1]:.2f}" for f in fracs)


def ent_buckets(d, col):
    b = pd.cut(d.ent_t, [-1, 0.1, 0.5, 1.0, 2.0, 99], labels=["<0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
    g = d.groupby(b, observed=False)[col].sum() / d[col].sum()
    f = b.value_counts(normalize=True).reindex(g.index)
    return " ".join(f"{k}:tok{f[k]:.0%}/{g[k]:.0%}" for k in g.index)


def per_id(d, col):
    t = d.groupby("id").agg(n=(col, "size"), w=(col, "sum"), lr=("lr", "mean"), ent_t=("ent_t", "mean"))
    t["share"] = t.w / d[col].sum()
    return t.sort_values("share", ascending=False)


def main():
    import json
    from transformers import AutoTokenizer
    root, steps = sys.argv[1], [int(s) for s in sys.argv[2:]]
    with Pool(min(len(steps), 32)) as pool:
        ds = pool.map(load, [(root, s) for s in steps])
    tok = AutoTokenizer.from_pretrained(TOK)
    rock = set(json.load(open(ROCK)))
    tops = {}
    for st, d in zip(steps, ds):
        col = "sig2" if d.sig2.notna().all() else "g2"
        print(f"\n== step {st}: {len(d)} tok, metric={col}")
        print("  conc g2 :", conc(d.g2.values), "| eos+eot share", round(d.g2[d.id.isin([EOT, IM_END])].sum() / d.g2.sum(), 3))
        if col == "sig2":
            print("  conc sig2:", conc(d.sig2.values), "| conc kl:", conc(d.kl.values))
            print("  sig2/g2 total ratio (signal fraction of sampled grad^2):", round(d.sig2.sum() / d.g2.sum(), 3))
            print("  kl by teacher-entropy bucket:", ent_buckets(d, "kl"))
            print("  sig2 by bucket:", ent_buckets(d, "sig2"))
        print("  g2 by bucket:", ent_buckets(d, "g2"))
        t = per_id(d, col)
        tops[st] = set(t.index[:TOPN])
        print(f"  top{TOPN} ids hold {t.share.iloc[:TOPN].sum():.0%} of {col}, {t.n.iloc[:TOPN].sum()/len(d):.0%} of tokens,"
              f" rock.json overlap {len(tops[st] & rock)}/{TOPN}")
    print("\n== stability of top ids (Jaccard)")
    for a, b in zip(steps, steps[1:]):
        print(f"  {a}->{b}: {len(tops[a]&tops[b])/len(tops[a]|tops[b]):.2f}")
    allD = pd.concat(ds, ignore_index=True)
    sk = allD[allD.sig2.notna()]
    t = per_id(sk, "sig2")
    core = set.intersection(*tops.values())
    print(f"  in top{TOPN} at every step: {len(core)} ids")
    kl_by = sk.groupby(["id", "step"]).kl.mean().unstack()
    ks = [s for s in steps if s in kl_by.columns]
    print(f"\n== pooled top 30 ids by sig2 share; per-occ = sig2 per occurrence / global mean; kl trend over {ks}")
    mean_sig = sk.sig2.mean()
    for v, r in t.head(30).iterrows():
        tr = kl_by.loc[v, ks]
        print(f"  {repr(tok.decode([int(v)])):14s} n={int(r.n):<6d} share={r.share:.3f} per-occ={r.w/r.n/mean_sig:5.1f}"
              f" ent_t={r.ent_t:.2f} rock={'Y' if v in rock else '-'} core={'Y' if v in core else '-'}"
              f" kl={' '.join(f'{x:.2f}' for x in tr)}")
    # top 1% positions by sig2: which ids, and is the id mix stable?
    print("\n== top-1% positions by sig2 per step: id mix")
    prev = None
    for s in ks:
        x = sk[sk.step == s]
        hi = x[x.sig2 >= x.sig2.quantile(0.99)]
        vc = hi.id.value_counts(normalize=True)
        top = set(vc.index[:TOPN])
        jac = f"jac_prev={len(top&prev)/len(top|prev):.2f}" if prev else ""
        prev = top
        print(f"  step {s}: {len(hi)} pos, {hi.id.nunique()} distinct ids, top20 ids cover {vc.iloc[:20].sum():.0%},"
              f" rock {hi.id.isin(rock).mean():.0%} (all tok {x.id.isin(rock).mean():.0%}), {jac}")
    hi = sk[sk.sig2 >= sk.groupby("step").sig2.transform(lambda z: z.quantile(0.99))]
    vc = hi.id.value_counts().head(30)
    base = sk.id.value_counts()
    print("  pooled top ids in top-1% positions (count, enrichment = share in hi / share overall):")
    print("  " + " ".join(f"{tok.decode([int(v)])!r}:{c}x{(c/len(hi))/(base[v]/len(sk)):.0f}" for v, c in vc.items()))


if __name__ == "__main__":
    main()
