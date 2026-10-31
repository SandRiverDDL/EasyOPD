# Copyright 2026 EasyOPD Contributors
"""ResOPD estimator on gathered log-probs (pure torch, CPU-testable).

Port of official ``verl/trainer/distillation/resopd.py`` ``resopd_loss``.
Every quantity there depends on the student only through full-vocab-normalised
log-probs at the teacher support S and at the sampled token A, so it can be
evaluated from ``student_gathered_log_probs`` (at ``ref_topk_indices``) and
``log_prob`` (at ``responses``) without materialising (B, R, V) logits.
The gradient is identical to the official one; ``test_opd_methods`` checks it.
"""

from __future__ import annotations

import torch


def resopd_terms(
    support_log_p: torch.Tensor,
    sampled_log_p: torch.Tensor,
    teacher_topk_lp: torch.Tensor,
    teacher_sampled_lp: torch.Tensor,
    head_hit: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Shapes: support (..., K), sampled (...), head_hit bool (...)."""
    dtype = torch.float64 if support_log_p.dtype == torch.float64 else torch.float32
    support_log_p = support_log_p.to(dtype)
    sampled_log_p = sampled_log_p.to(dtype)
    support_log_q = teacher_topk_lp.detach().to(dtype)
    support_p = support_log_p.exp()
    conditional_p = support_log_p.softmax(dim=-1)
    support_ratio = support_log_p - support_log_q

    head_value = (conditional_p * support_ratio).sum(dim=-1)
    head_surrogate = ((conditional_p * support_ratio).detach() * support_log_p).sum(dim=-1)

    sampled_ratio = sampled_log_p - teacher_sampled_lp.detach().to(dtype)
    sampled_surrogate = sampled_ratio.detach() * sampled_log_p
    student_mass = support_p.sum(dim=-1)
    teacher_mass = support_log_q.exp().sum(dim=-1)

    eps = torch.finfo(dtype).eps
    student_tail = (1.0 - student_mass).clamp_min(eps)
    teacher_tail = (1.0 - teacher_mass).clamp_min(eps)
    tail_log_ratio = student_tail.log() - teacher_tail.log()
    support_score = (support_p.detach() * support_log_p).sum(dim=-1)
    bucket_surrogate = -tail_log_ratio.detach() * support_score / student_tail.detach()

    hit = head_hit.bool()
    centered = hit.to(dtype) - student_mass.detach()
    surrogate = torch.where(hit, head_surrogate, sampled_surrogate)
    surrogate = surrogate - centered.detach() * (head_surrogate - bucket_surrogate)
    value = torch.where(hit, head_value, sampled_ratio)
    losses = value.detach() + (surrogate - surrogate.detach())
    return {
        "losses": losses,
        "student_mass": student_mass.detach(),
        "teacher_mass": teacher_mass.detach(),
        "head_hit": hit.to(dtype),
        "tail_log_ratio": tail_log_ratio.detach(),
    }
