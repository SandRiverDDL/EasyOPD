# Quick rock-token probe on 07 ablation token dumps (sampled-token RKL estimate, not full-vocab KL).
import json, sys, numpy as np, pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer

B = "/mnt/NLU_disk17/chenhaoyu02/runs/train/07_opd_repro/vanilla_opd/1029_2323_abl80v10_"
tok = AutoTokenizer.from_pretrained("/mnt/NLU_disk17/chenhaoyu02/models/Qwen3-1.7B-Base-imend")
paper = set(json.load(open("/mnt/NLU_disk17/chenhaoyu02/oss_refs/Rock-Token/stumbling/rock.json")))
K = 100


def load(run, steps):
    ids, loss = [], []
    for s in steps:
        d = pd.read_parquet(f"{B}{run}/token_dump/step_{s}.parquet")
        for r in d.itertuples():
            ids.append(np.asarray(r.token_ids)); loss.append(np.asarray(r.student_logp) - np.asarray(r.teacher_logp))
    return np.concatenate(ids), np.concatenate(loss)


def rock(ids, loss):
    S, C = defaultdict(float), defaultdict(int)
    for i, l in zip(ids.tolist(), loss.tolist()):
        S[i] += l; C[i] += 1
    # R(v) = mean(l_v) * Freq(v) = sum(l_v) / N
    top = sorted(S, key=lambda v: -S[v])[:K]
    pos_total = loss.clip(min=0).sum()
    cov = sum(max(S[v], 0) for v in top) / pos_total
    dens = np.isin(ids, top).mean()
    return top, cov, dens, S, C


for run in sys.argv[1:]:
    ids, loss = load(run, range(10, 81, 10))
    top, cov, dens, S, C = rock(ids, loss)
    e = rock(*load(run, [10, 20, 30, 40]))[0]; l = rock(*load(run, [50, 60, 70, 80]))[0]
    jac = len(set(e) & set(l)) / len(set(e) | set(l))
    print(f"== {run}: tokens={len(ids)} vocab_seen={len(C)} top{K} pos-loss coverage={cov:.2f} density={dens:.2f} "
          f"jaccard(early,late)={jac:.2f} overlap_with_paper={len(set(top) & paper)}/{K}")
    print("   top20:", [(tok.decode([v]), C[v], round(S[v] / C[v], 2)) for v in top[:20]])
    m = np.isin(ids, top)
    print(f"   mean l: rock={loss[m].mean():.3f} other={loss[~m].mean():.3f} | |l|>2 share in rock={np.isin(ids[np.abs(loss)>2], top).mean():.2f}")
