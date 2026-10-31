# Copyright 2026 EasyOPD Contributors
"""Sampled-token OPD/advantage transforms that need only sampled-token logprobs.

Pure torch, CPU-testable. All inputs are (B, R) tensors; ``mask`` is the
response mask. Outputs are already multiplied by ``mask``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def vanilla_advantage(ref_lp: torch.Tensor, old_lp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (ref_lp - old_lp) * mask


def power_advantage(ref_lp: torch.Tensor, old_lp: torch.Tensor, mask: torch.Tensor, alpha: float) -> torch.Tensor:
    """PowerOPD Eq. 4: r = pi_T^alpha - pi_theta^alpha (no 1/alpha, as official code)."""
    if alpha <= 0:
        raise ValueError(f"PowerOPD alpha must be > 0, got {alpha}")
    r = torch.exp(alpha * ref_lp.double()) - torch.exp(alpha * old_lp.double())
    return r.to(ref_lp.dtype) * mask


def zscore_tokens(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = mask.bool()
    if not m.any():
        return x * mask
    v = x[m]
    return ((x - v.mean()) / (v.std(unbiased=False) + eps)) * mask


def topd_immediate(ref_lp: torch.Tensor, old_lp: torch.Tensor, mask: torch.Tensor, alpha: float) -> torch.Tensor:
    """TOP-D component A: log(alpha * pi*/pi + 1 - alpha) (official logaddexp form)."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"TOP-D alpha must be in (0, 1], got {alpha}")
    log_ratio = ref_lp - old_lp
    if alpha == 1.0:
        r = log_ratio
    else:
        r = torch.logaddexp(log_ratio + math.log(alpha), torch.full_like(log_ratio, math.log1p(-alpha)))
    return r * mask


def topd_returns(r: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """TOP-D component B: R_k = r_k + mean_{j>k} r_j (0 when no future token)."""
    r = r * mask
    suffix = torch.flip(torch.cumsum(torch.flip(r, dims=[-1]), dim=-1), dims=[-1])
    future_sum = suffix - r
    cnt = torch.flip(torch.cumsum(torch.flip(mask, dims=[-1]), dim=-1), dims=[-1]) - mask
    future_mean = torch.where(cnt > 0, future_sum / cnt.clamp(min=1), torch.zeros_like(r))
    return (r + future_mean) * mask


def group_token_zscore(x: torch.Tensor, mask: torch.Tensor, group: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """TOP-D component C: z-score over all valid tokens of the responses sharing a prompt."""
    out = torch.zeros_like(x)
    m = mask.bool()
    for g in torch.unique(group):
        rows = group == g
        sel = m & rows[:, None]
        if not sel.any():
            continue
        v = x[sel]
        out[rows] = (x[rows] - v.mean()) / (v.std(unbiased=False) + eps)
    return out * mask


def terminal_mask(responses: torch.Tensor, mask: torch.Tensor, eos_ids: Optional[list[int]]) -> torch.Tensor:
    """Drop the last valid token when it is one of ``eos_ids`` (DIAL/PowerOPD terminal handling)."""
    if not eos_ids:
        return mask
    mask = mask.clone()
    n = mask.sum(-1).long()
    rows = torch.nonzero(n > 0, as_tuple=False).flatten()
    if rows.numel() == 0:
        return mask
    last = n[rows] - 1
    last_tok = responses[rows, last]
    is_eos = torch.zeros_like(last_tok, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= last_tok == int(e)
    mask[rows[is_eos], last[is_eos]] = 0
    return mask
