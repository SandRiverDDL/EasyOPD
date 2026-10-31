"""Follow-up stats for fork_probe.py output (new script): bootstrap CI of the true sd of dacc, calibrated
null for the permutation-p fraction, acc(student 2nd) - acc(student 1st), and splits by token class and
teacher margin.   python fork_probe_analyze.py <out_dir>   (needs forks_merged.jsonl from `fork_probe.py stats`)
"""
import json, random, math, re, statistics as st, glob, sys
D = sys.argv[1]
rows = [json.loads(l) for l in open(f"{D}/forks_merged.jsonl")]
gen = {}
for p in glob.glob(f"{D}/shards/gen_*.jsonl"):
    for l in open(p):
        g = json.loads(l); gen[g["row"]] = [r["ids"] for r in g["rollouts"]]
rng = random.Random(0)
def tv(rs):  # observed var of d minus mean unbiased sampling var
    return st.variance([r["d"] for r in rs]) - st.mean(r["sv"] for r in rs)
def boot(rs, f, B=2000):
    v = sorted(f([rs[rng.randrange(len(rs))] for _ in rs]) for _ in range(B)); return v[int(.025*B)], v[int(.975*B)]
for kind in ("all", "disagree", "agree"):
    rs = [r for r in rows if kind == "all" or r["kind"] == kind]
    lo, hi = boot(rs, tv)
    print(f"{kind:8s} true var {tv(rs):.4f} CI95 [{lo:.4f},{hi:.4f}] -> sd {math.sqrt(max(tv(rs),0)):.3f} [{math.sqrt(max(lo,0)):.3f},{math.sqrt(max(hi,0)):.3f}]")
# calibrated null for 'p<0.05' fraction: both branches ~ Binom(16, pooled acc)
def perm_p(a, b, n_perm=400):
    obs = abs(sum(a)/16 - sum(b)/16); pool = a + b; h = 0
    for _ in range(n_perm):
        rng.shuffle(pool); h += abs(sum(pool[:16]) - sum(pool[16:]))/16 >= obs - 1e-12
    return (h+1)/(n_perm+1)
null = []
for r in rows:
    m = (r["accT"] + r["accO"]) / 2
    a = [rng.random() < m for _ in range(16)]; b = [rng.random() < m for _ in range(16)]
    null.append(perm_p(a, b))
print(f"null p<0.05 frac {sum(p<0.05 for p in null)/len(null):.1%}  observed {sum(r['p']<0.05 for r in rows)/len(rows):.1%};"
      f" null p<0.01 {sum(p<0.01 for p in null)/len(null):.1%} observed {sum(r['p']<0.01 for r in rows)/len(rows):.1%}")
# b (student 2nd) minus a (student 1st), and teacher effect
for r in rows:
    r["dba"] = r["d"] if r["kind"] == "disagree" else -r["d"]
    r["samp"] = "a" if gen[r["row"]][r["rollout"]][r["t"]] == r["a"] else "b" if gen[r["row"]][r["rollout"]][r["t"]] == r["b"] else "other"
x = [r["dba"] for r in rows]
print(f"acc(b)-acc(a) all {st.mean(x):+.4f}±{st.stdev(x)/math.sqrt(len(x)):.4f}")
for s in ("a", "b", "other"):
    x = [r["dba"] for r in rows if r["samp"] == s]
    if len(x) > 1: print(f"  original rollout sampled {s}: n={len(x)} {st.mean(x):+.4f}±{st.stdev(x)/math.sqrt(len(x)):.4f}")
def cls(s):
    s = s.strip()
    if re.fullmatch(r"\d+", s): return "digit"
    if s == "" : return "space"
    if re.fullmatch(r"[A-Za-z]+", s): return "word"
    return "symbol/latex"
for r in rows: r["cls"] = cls(r["str_a"]) if cls(r["str_a"]) == cls(r["str_b"]) else "mixed"
for c in sorted(set(r["cls"] for r in rows)):
    rs = [r for r in rows if r["cls"] == c]
    if len(rs) < 5: continue
    sg = [r for r in rs if r["kind"] == "disagree"]
    print(f"cls {c:13s} n={len(rs):3d} true sd {math.sqrt(max(tv(rs),0)):.3f}  |d|>=0.375 {sum(abs(r['d'])>=0.375 for r in rs)/len(rs):.1%}"
          f"  disagree mean d {st.mean(r['d'] for r in sg) if sg else float('nan'):+.3f} (n={len(sg)})")
# teacher margin bins
for lo, hi in ((0.69, 1.5), (1.5, 3), (3, 99)):
    rs = [r for r in rows if lo <= abs(math.log(r["qa"]/r["qb"])) < hi]
    sg = [r["d"] for r in rs if r["kind"] == "disagree"]
    print(f"|log qa/qb| in [{lo},{hi}) n={len(rs)} true sd {math.sqrt(max(tv(rs),0)):.3f} disagree mean d {st.mean(sg):+.3f}±{st.stdev(sg)/math.sqrt(len(sg)):.3f} (n={len(sg)})")
tr = [x for r in rows for x in r["trunc_T"] + r["trunc_O"]]
print(f"continuation trunc frac {sum(tr)/len(tr):.3f}, mean prefix len {st.mean(r['prefix_len'] for r in rows):.0f}, rel pos {st.mean(r['prefix_len']/r['roll_len'] for r in rows):.2f}")
