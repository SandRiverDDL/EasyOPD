"""Per-token absolute KL / grad^2, two-way vs rest, per dump step (descriptive statistics, new script).

k1 = log p - log q at the sampled token (single-sample KL estimate); g2 = ((log q - log p)(1 - p))^2.
Proxy for steps < 110 (no top-k in dump): sampled-token p in [0.1, 0.9]; overlaps 97% of the exact two-way
positions (fork_probe.candidates rule) on steps >= 110. Usage: python abs_by_group.py <token_dump_dir>
"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
D=sys.argv[1]; STOP={151643,151645}
def load(st):
    df=pd.read_parquet(f"{D}/step_{st}.parquet"); has='student_topk_logp' in df.columns
    out=dict(k1=[],g2=[],px=[],two=[],kl=[],sig2=[])
    for r in df.itertuples():
        s=np.asarray(r.student_logp,float); t=np.asarray(r.teacher_logp,float); p=np.exp(s)
        out['k1'].append(s-t); out['g2'].append(((t-s)*(1-p))**2); out['px'].append((p>=0.1)&(p<=0.9))
        if has:
            n,k=len(s),int(r.student_topk_k); ls=np.asarray(r.student_topk_logp,float).reshape(n,k)
            lt=np.asarray(r.teacher_logp_at_s_topk,float).reshape(n,k); ids=np.asarray(r.student_topk_ids).reshape(n,k)
            q=np.exp(ls); a=lt-ls; ab=(q*a).sum(1,keepdims=True)
            out['kl'].append(-(q*a).sum(1)); out['sig2'].append(((q*(a-ab))**2).sum(1))
            out['two'].append((q[:,0]+q[:,1]>=.9)&(q[:,1]>=.1)&~np.isin(ids[:,:2],list(STOP)).any(1))
    o={k:np.concatenate(v) for k,v in out.items() if v}
    res=dict(step=st,px_frac=o['px'].mean(),k1_px=o['k1'][o['px']].mean(),k1_rest=o['k1'][~o['px']].mean(),
             g2_px=o['g2'][o['px']].mean(),g2_rest=o['g2'][~o['px']].mean())
    if has:
        tw=o['two']; res.update(two_frac=tw.mean(),overlap=(o['px']&tw).sum()/tw.sum(),
            kl_two=o['kl'][tw].mean(),kl_rest=o['kl'][~tw].mean(),sig2_two=o['sig2'][tw].mean(),sig2_rest=o['sig2'][~tw].mean(),
            g2_two=o['g2'][tw].mean(),g2_rest2=o['g2'][~tw].mean())
    return res
with Pool(20) as P: R=P.map(load,range(10,201,10))
pd.set_option('display.width',250); print(pd.DataFrame(R).to_string(index=False,float_format=lambda v:f'{v:.4f}'))
