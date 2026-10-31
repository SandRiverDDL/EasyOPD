"""Per-step on-policy top-k RKL, split into student two-way positions vs the rest.

New script (descriptive statistics only, no new method). Loading and the top-k RKL follow
grad_token_stability.py; the two-way rule is the one in fork_probe.py `candidates()`:
student top-2 mass >= 0.9, p2 >= 0.1, neither candidate a stop token.

Question: in the late stage, does the KL at two-way positions stay flat while the KL elsewhere
keeps falling (consistent with "unlearnable at two-way", H-a), or do both stay flat?
Each dump is the rollout of that step, scored before the update -> on-policy per-step KL.

Usage: python kl_trend_by_group.py <token_dump_dir> <step> [<step> ...]
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

STOP = {151643, 151645}  # <|endoftext|>, <|im_end|>


def load(args):
    root, st = args
    df = pd.read_parquet(f"{root}/step_{st}.parquet")
    if "student_topk_logp" not in df.columns:
        return None
    kl, two, ent = [], [], []
    for r in df.itertuples():
        n, k = len(r.token_ids), int(r.student_topk_k)
        ls = np.asarray(r.student_topk_logp, dtype=np.float64).reshape(n, k)
        lt = np.asarray(r.teacher_logp_at_s_topk, dtype=np.float64).reshape(n, k)
        ids = np.asarray(r.student_topk_ids).reshape(n, k)
        p = np.exp(ls)
        kl.append((p * (ls - lt)).sum(1))
        stop = np.isin(ids[:, :2], list(STOP)).any(1)
        two.append((p[:, 0] + p[:, 1] >= 0.9) & (p[:, 1] >= 0.1) & ~stop)
        ent.append(np.asarray(r.student_entropy, dtype=np.float64))
    kl, two = np.concatenate(kl), np.concatenate(two)
    # per-response mean is not used: token-mean matches the training loss aggregation (token-mean)
    return dict(step=st, n=len(kl), frac_two=two.mean(), kl_all=kl.mean(), kl_two=kl[two].mean(),
                kl_rest=kl[~two].mean(), share_two=kl[two].sum() / kl.sum(), ent_s=np.concatenate(ent).mean())


def slope(x, y):
    """OLS slope per 100 steps; NOTE: ignores step-to-step autocorrelation (SE optimistic)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = np.polyfit(x, y, 1)[0]
    res = y - np.polyval(np.polyfit(x, y, 1), x)
    se = np.sqrt(res.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    return 100 * b, 100 * se


def main():
    root, steps = sys.argv[1], [int(s) for s in sys.argv[2:]]
    with Pool(min(len(steps), 32)) as pool:
        rows = [r for r in pool.map(load, [(root, s) for s in steps]) if r is not None]
    t = pd.DataFrame(rows).sort_values("step")
    pd.set_option("display.width", 200)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nslope per 100 steps (+-SE, autocorrelation ignored), relative = slope / mean")
    for c in ["kl_all", "kl_two", "kl_rest", "frac_two", "ent_s"]:
        b, se = slope(t.step, t[c])
        print(f"  {c:9s} {b:+.4f} +- {se:.4f}   rel {b / t[c].mean():+.1%}")
    # first half vs second half of the window, a cruder check robust to a few outlier steps
    h = len(t) // 2
    for c in ["kl_two", "kl_rest"]:
        a, b = t[c].iloc[:h], t[c].iloc[h:]
        print(f"  {c:9s} first half {a.mean():.4f} -> second half {b.mean():.4f} ({b.mean() / a.mean() - 1:+.1%})")


if __name__ == "__main__":
    main()
