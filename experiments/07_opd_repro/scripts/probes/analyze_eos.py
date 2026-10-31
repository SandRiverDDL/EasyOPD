# How do responses end in the 07 ablation token dumps, and what does the teacher say about that last token?
import sys, numpy as np, pandas as pd

B = "/mnt/NLU_disk17/chenhaoyu02/runs/train/07_opd_repro/vanilla_opd/1029_2323_abl80v10_"
IM_END, EOT = 151645, 151643
for run in sys.argv[1:]:
    rows = []
    for s in range(10, 81, 10):
        try:
            d = pd.read_parquet(f"{B}{run}/token_dump/step_{s}.parquet")
        except FileNotFoundError:
            continue
        for r in d.itertuples():
            ids = np.asarray(r.token_ids); sl = np.asarray(r.student_logp); tl = np.asarray(r.teacher_logp)
            eot_pos = np.where(ids == EOT)[0]
            rows.append(dict(step=s, n=len(ids), last=int(ids[-1]), trunc=bool(r.truncated),
                             n_eot=len(eot_pos), eot_last=bool(len(eot_pos) and eot_pos[-1] == len(ids) - 1),
                             last_s=sl[-1], last_t=tl[-1]))
    df = pd.DataFrame(rows)
    end = np.where(df["last"] == IM_END, "im_end", np.where(df["last"] == EOT, "endoftext", "other(trunc)"))
    df["end"] = end
    print(f"== {run}: responses={len(df)} | endoftext occurrences={df.n_eot.sum()}, of which at last position={df.eot_last.sum()}")
    print(df.groupby("end").agg(cnt=("n", "size"), student_logp=("last_s", "mean"), teacher_logp=("last_t", "mean")).round(2).to_string())
    print("   by step (endoftext-ended share):", df.groupby("step").apply(lambda g: round((g["end"] == "endoftext").mean(), 2)).to_dict())
