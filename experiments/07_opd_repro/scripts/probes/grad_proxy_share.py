"""Share of the logits-space gradient carried by token groups, from token_dump parquet.

New script (no existing equivalent; reads the same dump columns as analyze_rock_tokens.py).
For one sampled token x with advantage A, d(-A*log p_S(x))/d logits = -A*(e_x - p_S).
Its L2 norm is between |A|(1-p) and sqrt(2)|A|(1-p), p = p_S(x); we use g = |A|(1-p).
Logits of different positions are separate variables, so in logits space per-token
gradients are orthogonal and shares of sum(g^2) are exact up to the sqrt(2) factor.
Parameter-space direction (cancellation through shared weights) is NOT measured here.

Usage: python grad_proxy_share.py <token_dump_dir> <step> [<step> ...]
"""
import sys

import numpy as np
import pandas as pd

EOT = 151643  # <|endoftext|> in Qwen3 tokenizer
IM_END = 151645  # <|im_end|>
WIN = 32


def win_nll(lp):
    if len(lp) < WIN:
        return float(-lp.mean())
    c = np.concatenate([[0.0], np.cumsum(-lp)])
    return float(((c[WIN:] - c[:-WIN]) / WIN).max())


def load(path):
    df = pd.read_parquet(path)
    rows = []
    for r in df.itertuples():
        ids = np.asarray(r.token_ids)
        s = np.asarray(r.student_logp, dtype=np.float64)
        t = np.asarray(r.teacher_logp, dtype=np.float64)
        a = np.asarray(r.adv, dtype=np.float64)
        n = len(ids)
        rows.append(pd.DataFrame(dict(
            row=r.row, pos=np.arange(n), id=ids, s=s, t=t, a=a,
            degen=win_nll(s) > 5, score=r.score)))
    d = pd.concat(rows, ignore_index=True)
    d["g"] = np.abs(d.a) * (1 - np.exp(d.s))
    d["g2"] = d.g ** 2
    d["lr"] = d.t - d.s
    return d


def share(d, name, mask):
    tot_g, tot_g2, n = d.g.sum(), d.g2.sum(), len(d)
    m = d[mask]
    return dict(group=name, tok=len(m) / n, g=m.g.sum() / tot_g, g2=m.g2.sum() / tot_g2,
                lr_mean=m.lr.mean() if len(m) else np.nan)


def groups(d):
    pos = d.pos
    lr = d.lr
    p = np.exp(d.s)
    out = [
        share(d, "pos<16", pos < 16),
        share(d, "16<=pos<128", (pos >= 16) & (pos < 128)),
        share(d, "pos>=128", pos >= 128),
        share(d, "endoftext", d.id == EOT),
        share(d, "im_end", d.id == IM_END),
        share(d, "A>0 (boost)", d.a > 0),
        share(d, "A<0 (suppress)", d.a < 0),
        share(d, "|lr|<1", lr.abs() < 1),
        share(d, "1<=|lr|<5", (lr.abs() >= 1) & (lr.abs() < 5)),
        share(d, "|lr|>=5", lr.abs() >= 5),
        share(d, "p_S<0.01", p < 0.01),
        share(d, "0.01<=p_S<0.5", (p >= 0.01) & (p < 0.5)),
        share(d, "p_S>=0.5", p >= 0.5),
        share(d, "degen traj (win>5)", d.degen),
        share(d, "score==1 traj", d.score == 1),
    ]
    return pd.DataFrame(out)


def topk_share(d, fracs=(0.001, 0.01, 0.1)):
    g2 = np.sort(d.g2.values)[::-1]
    c = np.cumsum(g2) / g2.sum()
    return {f"top{f*100:g}%tok": float(c[max(int(len(g2) * f), 1) - 1]) for f in fracs}


def main():
    root, steps = sys.argv[1], sys.argv[2:]
    pd.set_option("display.width", 200)
    for st in steps:
        d = load(f"{root}/step_{st}.parquet")
        print(f"\n== step {st}: {d.row.nunique()} traj, {len(d)} tok ==")
        print("concentration of sum(g^2):", {k: round(v, 3) for k, v in topk_share(d).items()})
        print(groups(d).round(3).to_string(index=False))
        top = d.groupby("id").agg(n=("g2", "size"), g2=("g2", "sum")).sort_values("g2", ascending=False)
        top["g2"] /= d.g2.sum()
        print("top token ids by g^2 share:", top.head(8).round(3).to_dict("index"))


if __name__ == "__main__":
    main()
