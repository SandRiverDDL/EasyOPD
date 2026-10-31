"""Gradient noise scale B_simple of the vanilla-OPD loss at a fixed checkpoint.

Method: McCandlish et al. 2018, "An Empirical Model of Large-Batch Training" (arXiv 1812.06162),
B_simple = tr(Sigma) / |G|^2, with the unbiased small/big-batch estimators of their Appendix A.2
taken at B_small = 1 unit and B_big = N units:
  |G|^2     ~ (|sum_j G_j|^2 - sum_j |G_j|^2) / (N (N - 1))
  tr(Sigma) ~ (sum_j |G_j|^2 - |sum_j G_j|^2 / N) / (N - 1)
New code (not from the paper's repo); what is specific to us:
- unit = one prompt with its k rollouts, because training samples prompts (32 prompts x 8 per step),
  so B_simple is in prompts and is compared with the training batch of 32 prompts;
- G_j = - sum over the prompt's rollouts and response tokens of A_t * grad log p(y_t) / (k * L_bar),
  A_t = log q(y_t) - log p(y_t) detached (vanilla_opd, on-policy so PPO ratio = 1 and clip inactive,
  entropy_coeff 0, no KL loss). L_bar is a constant (token-mean up to a constant factor; B is scale-free).
- rollouts: fork_probe gen shards (step-200 student, vLLM, T=1, 8192), i.e. the step-201 on-policy data.
Sanity check: sqrt(|G|^2 + tr(Sigma)/32) should be close to the logged actor/grad_norm (~0.56 at step 200).

Stages:
  acc   one GPU: N_shard prompts -> saves S = sum_j G_j (fp32, per-parameter) and sum_j |G_j|^2
  stats Gram matrix of the shard sums (mmap, per parameter) -> estimates + leave-one-shard-out jackknife SE
"""
import argparse
import glob
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fork_probe import STOP_TOKENS, candidates, read_jsonl  # noqa: E402

STUDENT = "/mnt/NLU_disk17/chenhaoyu02/runs/train/07_opd_repro/vanilla_opd/1030_0745_opd100lr1_vanilla/hf/global_step_200"
TEACHER = "/mnt/NLU_disk17/chenhaoyu02/models/Qwen3-4B"
GEN = "/mnt/NLU_disk17/chenhaoyu02/runs/train/07_opd_repro/fork_probe/1031_0548_fork300/shards"


def full_logsm(model, prompt_ids, ids, pos=None):
    """log_softmax over the vocab at response positions `pos` (all if None); lm_head only on those rows, so
    full-vocab rows for long responses do not OOM (the first est=exact run OOM'd at 4.6 GiB per extra row block)."""
    import torch

    x = torch.tensor([prompt_ids + ids], device="cuda")
    h = model.model(input_ids=x, use_cache=False).last_hidden_state[0, len(prompt_ids) - 1:-1]
    if pos is not None:
        h = h[pos]
    return torch.log_softmax(model.lm_head(h).float(), -1)


def token_logp(model, prompt_ids, ids):
    import torch

    y = torch.tensor(ids, device="cuda")
    return full_logsm(model, prompt_ids, ids).gather(1, y[:, None])[:, 0]


def stage_acc(args):
    import torch
    from transformers import AutoModelForCausalLM

    prompts = [r for f in sorted(glob.glob(f"{args.gen_dir}/gen_*.jsonl")) for r in read_jsonl(f)]
    prompts.sort(key=lambda r: r["row"])
    pick = random.Random(args.seed).sample(range(len(prompts)), args.n_prompts)  # same list on every shard
    mine = [prompts[i] for i in pick[args.shard_id::args.num_shards]]

    st = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.float32,
                                              attn_implementation="flash_attention_2").cuda()
    st.gradient_checkpointing_enable()
    st.config.use_cache = False
    te = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                              attn_implementation="flash_attention_2").cuda().eval()
    # --mask two/rest: keep only student two-way positions (fork_probe.candidates rule, vLLM top-2 at gen) or the
    # complement. Same normaliser k * l_bar for every mask, so G_two + G_rest = G_all exactly. (Position mask is our
    # extension of McCandlish et al.; the estimator itself is unchanged.)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.student)
    stop_set = set(tk.convert_tokens_to_ids(STOP_TOKENS)) | {tk.eos_token_id}
    n_sel = 0
    params = [p for p in st.parameters() if p.requires_grad]
    names = [n for n, p in st.named_parameters() if p.requires_grad]
    S = [torch.zeros_like(p) for p in params]
    sq, n_tok, adv_abs, t0 = 0.0, 0, 0.0, time.time()
    for j, r in enumerate(mine):
        st.zero_grad(set_to_none=False)
        for roll in r["rollouts"][: args.k]:
            ids = roll["ids"]
            if args.est == "exact":
                # --est exact (needs --mask two): at each two-way position replace the sampled-token term
                # -(lq(y) - lp(y)) grad lp(y) by its expectation over y ~ p(.|prefix), i.e. the per-position reverse-KL
                # gradient sum_v p_v (lp_v - lq_v) grad lp_v over the full vocab (Rao-Blackwellised over y_t only;
                # prefixes and the other rollouts are unchanged). Same normaliser as the sampled estimator.
                tw = [c[0] for c in candidates(roll, stop_set, args)]
                n_sel += len(tw)
                n_tok += len(ids)
                if not tw:
                    continue
                ti = torch.tensor(tw, device="cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lt = full_logsm(te, r["prompt_ids"], ids, ti)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ls = full_logsm(st, r["prompt_ids"], ids, ti)
                w = (ls.exp() * (ls - lt)).detach()
                adv_abs += (ls - lt).detach().gather(1, torch.tensor(ids, device="cuda")[ti][:, None]).abs().sum().item()
                loss = (w * ls).sum() / (args.k * args.l_bar)
                loss.backward()
                del lt, ls, w, loss
                continue
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lq = token_logp(te, r["prompt_ids"], ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = token_logp(st, r["prompt_ids"], ids)
            a = (lq - lp).detach()
            if args.mask != "all":
                m = torch.zeros(len(ids), device="cuda")
                tw = [c[0] for c in candidates(roll, stop_set, args)]
                if tw:
                    m[torch.tensor(tw, device="cuda")] = 1.0
                a = a * (m if args.mask == "two" else 1.0 - m)
                n_sel += int(m.sum().item()) if args.mask == "two" else len(ids) - int(m.sum().item())
            else:
                n_sel += len(ids)
            loss = -(a * lp).sum() / (args.k * args.l_bar)
            loss.backward()
            n_tok += len(ids)
            adv_abs += a.abs().sum().item()
        with torch.no_grad():
            g2 = 0.0
            for s, p in zip(S, params):
                if p.grad is None:  # est=exact and no two-way position seen yet on this shard
                    continue
                s.add_(p.grad)
                g2 += p.grad.double().pow(2).sum().item()
        sq += g2
        if j % 8 == 0:
            print(f"[acc {args.shard_id}] {j + 1}/{len(mine)} |G_j|^2={g2:.4e} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    torch.save({n: s.cpu() for n, s in zip(names, S)}, f"{args.tmp_dir}/S_{args.shard_id:02d}.pt")
    meta = dict(shard=args.shard_id, mask=args.mask, est=args.est, n_sel=n_sel, n=len(mine), sq=sq, n_tok=n_tok, mean_abs_adv=adv_abs / max(n_tok, 1),
                rows=[r["row"] for r in mine], sec=time.time() - t0)
    os.makedirs(f"{args.out_dir}/shards", exist_ok=True)
    json.dump(meta, open(f"{args.out_dir}/shards/meta_{args.shard_id:02d}.json", "w"))
    print(f"[acc {args.shard_id}] done n={len(mine)} sq={sq:.4e} tokens={n_tok} in {time.time() - t0:.0f}s", flush=True)


def estimates(N, sq, gsum2):
    tr = (sq - gsum2 / N) / (N - 1)
    g2 = (gsum2 - sq) / (N * (N - 1))
    return tr, g2


def stage_stats(args):
    import numpy as np
    import torch

    metas = [json.load(open(f)) for f in sorted(glob.glob(f"{args.out_dir}/shards/meta_*.json"))]
    files = [f"{args.tmp_dir}/S_{m['shard']:02d}.pt" for m in metas]
    sds = [torch.load(f, mmap=True, map_location="cpu") for f in files]
    K = len(sds)
    M = torch.zeros(K, K, dtype=torch.float64, device="cuda")
    chunk = 1 << 25
    for name in sds[0]:
        flat = [sd[name].reshape(-1) for sd in sds]
        for i in range(0, flat[0].numel(), chunk):
            X = torch.stack([f[i:i + chunk] for f in flat]).cuda().double()
            M += X @ X.T
    M = M.cpu().numpy()
    n = np.array([m["n"] for m in metas], float)
    sqk = np.array([m["sq"] for m in metas])
    N, sq = n.sum(), sqk.sum()
    assert N == args.n_prompts, f"only {int(N)}/{args.n_prompts} prompts in shard metas (a shard failed?)"
    tr, g2 = estimates(N, sq, M.sum())
    # rescale from the constant l_bar to the actual mean response length (token-mean normaliser)
    L = sum(m["n_tok"] for m in metas) / (N * args.k)
    c = (args.l_bar / L) ** 2
    tr, g2 = tr * c, g2 * c
    jk = []
    for m in range(K):
        keep = [k for k in range(K) if k != m]
        t_, g_ = estimates(N - n[m], sq - sqk[m], M[np.ix_(keep, keep)].sum())
        jk.append((t_ * c, g_ * c, t_ / g_))
    jk = np.array(jk)
    se = np.sqrt((K - 1) / K * ((jk - jk.mean(0)) ** 2).sum(0))
    B = tr / g2
    pred = {b: float(np.sqrt(g2 + tr / b)) for b in (8, 16, 32, 64, 128, 256)}
    out = dict(N_prompts=int(N), k=args.k, shards=K, mean_resp_len=L, tr_sigma=tr, tr_sigma_se=se[0], G2=g2,
               G2_se=se[1], B_simple_prompts=B, B_simple_se=se[2], B_simple_jackknife=jk[:, 2].tolist(),
               pred_grad_norm_by_batch=pred, signal_frac_at_32=g2 / (g2 + tr / 32),
               mean_abs_adv=float(np.mean([m["mean_abs_adv"] for m in metas])),
               mask=metas[0].get("mask", "all"), est=metas[0].get("est", "sampled"),
               frac_tokens_selected=sum(m.get("n_sel", m["n_tok"]) for m in metas) / sum(m["n_tok"] for m in metas),
               mean_cos_between_prompts=g2 / (g2 + tr))  # E[g_i.g_j]/E|g|^2 = |G|^2/(|G|^2+tr) = 1/(1+B)
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{args.out_dir}/result.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["acc", "stats"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tmp_dir", required=True, help="local disk for the fp32 shard sums (~7 GB each)")
    ap.add_argument("--gen_dir", default=GEN)
    ap.add_argument("--student", default=STUDENT)
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--n_prompts", type=int, default=512, help="total over shards")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--l_bar", type=float, default=2400.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--mask", default="all", choices=["all", "two", "rest"])
    ap.add_argument("--min_top2_mass", type=float, default=0.9)  # same two-way rule as fork_probe
    ap.add_argument("--min_p2", type=float, default=0.1)
    ap.add_argument("--est", default="sampled", choices=["sampled", "exact"],
                    help="exact: full-vocab expectation over y_t at two-way positions (requires --mask two)")
    args = ap.parse_args()
    assert args.est == "sampled" or args.mask == "two", "--est exact is only implemented for --mask two"
    {"acc": stage_acc, "stats": stage_stats}[args.stage](args)


if __name__ == "__main__":
    main()
