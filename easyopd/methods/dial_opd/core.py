# Copyright 2026 EasyOPD Contributors
"""DIAL-OPD token scoring / selection (pure torch, CPU-testable).

Mirrors official ``selection.py`` ``select_score_tokens``: scores in float64 log
space, per-response top-k with k_i = max(1, round(ratio * n_valid_i)).
"""

from __future__ import annotations

from typing import Optional

import torch


def log_scores(student_lp: torch.Tensor, teacher_lp: torch.Tensor, beta: float) -> torch.Tensor:
    """log s_beta = (1-beta) log|r| + beta log|q-p|, r = log q - log p (float64)."""
    p = student_lp.double()
    q = teacher_lp.double()
    r_abs = (q - p).abs()
    log_r = torch.log(r_abs)
    if beta == 0.0:
        return log_r
    # log|e^q - e^p| = max(p,q) + log(1 - e^{-|q-p|})
    log_diff = torch.maximum(p, q) + torch.log(-torch.expm1(-r_abs))
    if beta == 1.0:
        return log_diff
    return (1.0 - beta) * log_r + beta * log_diff


def per_row_k(mask: torch.Tensor, ratio: float) -> torch.Tensor:
    n = mask.sum(-1)
    k = torch.round(n.double() * ratio).long().clamp(min=1)
    return torch.where(n > 0, k, torch.zeros_like(k))


def select_topk_per_row(scores: torch.Tensor, mask: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pick the k_i highest-scoring valid positions of each row (stable order on ties)."""
    s = scores.double().masked_fill(~mask.bool(), float("-inf"))
    s = torch.nan_to_num(s, nan=float("-inf"))
    order = torch.sort(s, dim=-1, descending=True, stable=True).indices
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(s.shape[-1], device=s.device).expand_as(order).contiguous())
    sel = (rank < k[:, None]) & mask.bool()
    return sel.to(mask.dtype)


def select_random_per_row(mask: torch.Tensor, k: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.rand(mask.shape, generator=g, dtype=torch.float64).to(mask.device)
    return select_topk_per_row(noise, mask, k)


def dial_select(
    student_lp: torch.Tensor,
    teacher_lp: torch.Tensor,
    mask: torch.Tensor,
    ratio: float,
    beta: float,
    selector: str = "dial",
    seed: Optional[int] = None,
) -> torch.Tensor:
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"DIAL keep_ratio must be in (0, 1], got {ratio}")
    k = per_row_k(mask, ratio)
    if selector == "dial":
        return select_topk_per_row(log_scores(student_lp, teacher_lp, beta), mask, k)
    if selector == "random":
        return select_random_per_row(mask, k, 0 if seed is None else seed)
    raise ValueError(f"DIAL selector must be dial|random, got {selector}")


def low_low_frac(student_lp, teacher_lp, sel, threshold: float = 0.01) -> float:
    """Diagnostic: fraction of selected tokens with p <= thr and q <= thr."""
    lt = torch.log(torch.tensor(threshold, dtype=torch.float64))
    ll = (student_lp.double() <= lt) & (teacher_lp.double() <= lt) & sel.bool()
    return float(ll.sum().item() / sel.sum().clamp(min=1).item())
