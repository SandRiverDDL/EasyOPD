# Copyright 2026 EasyOPD Contributors
"""IER (Information-to-Error Ratio) on the candidate set C (pure torch).

Vectorised port of official ``ier_metrics.py`` ``compute_position``:

- C = student top-K  U  teacher top-K  U  {sampled token}
- each side's value at a candidate: its own top-K value if present, else the
  sampled-token logprob if the candidate is the sampled token, else ``floor``
- renormalise each side over C, rho = log p - log q clipped to [-clip, clip]
- signal = Var_p(rho); L = 1/max(p, eps) - 1; b* = E_p[rho L] / E_p[L]
- noise = max(E_p[(rho - b*)^2 L] - signal, 0); IER = signal / max(noise, eps)
"""

from __future__ import annotations

import math

import torch

NEG = -1e30


def _lookup(cand_ids, ids, vals):
    """value of ``cand_ids`` inside (ids, vals) top-k lists; NaN when absent. Shapes (N,M),(N,K),(N,K)."""
    hit = (cand_ids[:, :, None] == ids[:, None, :]) & torch.isfinite(vals)[:, None, :]
    found = hit.any(-1)
    v = (hit.to(vals.dtype) * torch.nan_to_num(vals, neginf=0.0, posinf=0.0)[:, None, :]).sum(-1)
    return torch.where(found, v, torch.full_like(v, float("nan")))


def ier_flat(
    s_ids: torch.Tensor,
    s_lp: torch.Tensor,
    t_ids: torch.Tensor,
    t_lp: torch.Tensor,
    a_id: torch.Tensor,
    a_s_lp: torch.Tensor,
    a_t_lp: torch.Tensor,
    floor: float = -12.0,
    clip: float = 30.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """IER for N flattened positions. ids (N,K), lps (N,K), sampled id / lps (N,)."""
    dt = torch.float64
    cand = torch.cat([s_ids, t_ids, a_id[:, None]], dim=-1).long()  # (N, M)
    m = cand.shape[-1]
    # first occurrence only (dedupe)
    eq = cand[:, :, None] == cand[:, None, :]
    earlier = torch.tril(torch.ones(m, m, dtype=torch.bool, device=cand.device), diagonal=-1)
    dup = (eq & earlier[None]).any(-1)

    is_a = cand == a_id[:, None]

    def side(ids, lps, a_lp):
        v = _lookup(cand, ids.long(), lps.to(dt))
        a_ok = torch.isfinite(a_lp.to(dt))[:, None]
        v = torch.where(torch.isnan(v) & is_a & a_ok, a_lp.to(dt)[:, None].expand_as(v), v)
        v = torch.nan_to_num(v, nan=floor)
        v = v.masked_fill(dup, NEG)
        return v - torch.logsumexp(v, dim=-1, keepdim=True)

    lp_s = side(s_ids, s_lp, a_s_lp)
    lp_t = side(t_ids, t_lp, a_t_lp)
    p = lp_s.exp().masked_fill(dup, 0.0)
    rho = (lp_s - lp_t).clamp(-clip, clip).masked_fill(dup, 0.0)
    mean_r = (p * rho).sum(-1, keepdim=True)
    signal = (p * (rho - mean_r) ** 2).sum(-1)
    lev = (1.0 / p.clamp(min=eps) - 1.0).masked_fill(dup, 0.0)
    e_l = (p * lev).sum(-1)
    b_star = torch.where(
        e_l.abs() > eps, (p * rho * lev).sum(-1) / e_l.clamp(min=eps), mean_r.squeeze(-1)
    )
    second = (p * (rho - b_star[:, None]) ** 2 * lev).sum(-1)
    noise = (second - signal).clamp(min=0.0)
    return signal / noise.clamp(min=eps)


def ier_tokens(td: dict, floor: float = -12.0, clip: float = 30.0, chunk: int = 65536) -> torch.Tensor:
    """(B, R) IER for every valid token; 0 elsewhere. Needs student/teacher top-k + sampled lps."""
    mask = td["response_mask"].bool()
    out = torch.zeros(mask.shape, dtype=torch.float64)
    idx = mask.nonzero(as_tuple=True)
    if idx[0].numel() == 0:
        return out
    flat = {
        "s_ids": td["old_topk_indices"][idx],
        "s_lp": td["old_topk_log_probs"][idx],
        "t_ids": td["ref_topk_indices"][idx],
        "t_lp": td["ref_topk_log_probs"][idx],
        "a_id": td["responses"][idx],
        "a_s_lp": td["old_log_probs"][idx],
        "a_t_lp": td["ref_log_prob"][idx],
    }
    vals = []
    n = flat["a_id"].shape[0]
    for s in range(0, n, chunk):
        part = {k: v[s : s + chunk].cpu() for k, v in flat.items()}
        vals.append(ier_flat(**part, floor=floor, clip=clip))
    out[idx] = torch.cat(vals).to(out.dtype)
    return out


def budget_select(
    scores: torch.Tensor, mask: torch.Tensor, ratio: float, min_keep: int = 1, random_seed=None
) -> torch.Tensor:
    """Global top-ceil(ratio*N) over the batch + per-response min_keep (official _apply_budget_mask)."""
    m = mask.bool()
    n_valid = int(m.sum().item())
    sel = torch.zeros_like(m)
    if n_valid == 0:
        return sel.to(mask.dtype)
    budget = min(n_valid, int(math.ceil(ratio * n_valid)))
    flat_idx = m.flatten().nonzero(as_tuple=True)[0]
    if random_seed is None:
        s = scores.double().flatten()[flat_idx]
        order = torch.sort(s, descending=True, stable=True).indices
    else:
        g = torch.Generator(device="cpu").manual_seed(int(random_seed))
        order = torch.randperm(flat_idx.numel(), generator=g)
        s_mat = torch.rand(m.shape, generator=g, dtype=torch.float64)
        scores = s_mat
    chosen = flat_idx[order[:budget].to(flat_idx.device)]
    sel.view(-1)[chosen] = True
    if min_keep > 0:
        have = sel.sum(-1)
        need = (have < min_keep) & (m.sum(-1) > 0)
        if need.any():
            s = scores.double().masked_fill(~m | sel, float("-inf"))
            for row in need.nonzero(as_tuple=True)[0].tolist():
                k = int(min(min_keep - int(have[row]), int(m[row].sum()) - int(have[row])))
                top = torch.topk(s[row], k).indices
                sel[row, top] = True
    return sel.to(mask.dtype)
