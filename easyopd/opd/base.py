# Copyright 2026 EasyOPD Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Unified same-tokenizer OPD method interface.

A method is a class registered with ``@register_method(name)`` that subclasses
:class:`OPDMethod`. verl calls it at exactly two places (see
``easyopd/docs/reference/opd-infra.md``):

1. ``batch_stage`` -- on the driver, once per training step, over the whole
   rollout batch, after ``old_log_probs`` / ``ref_log_prob`` are known and after
   verl's own ``compute_advantage``. Use it for everything that needs the full
   batch: advantages, group / batch normalisation, token selection with a
   global budget, random controls. No gradient.
2. ``actor_loss`` -- inside ``DataParallelPPOActor.update_policy`` for every
   micro-batch. Returns the scalar loss *before* verl's ``loss_scale_factor``.

Tensor keys (all shaped ``(B, R)`` or ``(B, R, K)``, R = response length):

=========================  ==================================================
``response_mask``          valid response tokens
``responses``              sampled token ids
``old_log_probs``          student (rollout-time policy) logprob of sampled token
``ref_log_prob``           teacher logprob of sampled token
``ref_topk_log_probs``     teacher top-k, full-vocab-normalised (``teacher_topk``)
``ref_topk_indices``       their vocab ids
``ref_entropy``            teacher full-vocab entropy (``teacher_entropy``)
``old_topk_log_probs``     student old-policy top-k (``student_topk``)
``old_topk_indices``       their vocab ids
``ref_gathered_log_probs`` teacher logprob at ``old_topk_indices``
                           (``teacher_at_student_topk``)
=========================  ==================================================

In ``actor_loss`` additionally:

=============================  ==============================================
``log_prob``                   current student logprob of sampled token (grad)
``student_gathered_log_probs`` current student full-vocab-normalised logprob
                               at ``ref_topk_indices`` (grad; only when the
                               method requests ``teacher_topk`` and
                               ``student_gather_teacher_topk``)
``entropy``                    student entropy (only if entropy_coeff != 0)
``advantages``                 output of ``batch_stage``
``opd_*``                      any extra tensor written by ``batch_stage``
=============================  ==============================================

All log-probs are computed at the rollout temperature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch

from easyopd.opd import common


@dataclass
class OPDSignals:
    """Which teacher / student signals a method needs beyond sampled logprobs."""

    teacher_topk: int = 0
    teacher_entropy: bool = False
    student_topk: int = 0
    teacher_at_student_topk: bool = False
    # Gather the *current* student (with grad) at ``ref_topk_indices`` in the
    # actor forward. Only meaningful with teacher_topk > 0.
    student_gather_teacher_topk: bool = False


class OPDMethod:
    """Base class. Default behaviour = vanilla sampled-token OPD with PPO clip."""

    name: str = ""
    description: str = ""
    paper_url: str = ""
    code_url: str = ""
    # Every OPD method plugs into the same touchpoints (see opd-infra.md).
    verl_modified_files = list(common.OPD_VERL_FILES)
    integration = "opd"

    # ---- signals ---------------------------------------------------------
    @classmethod
    def signals(cls, cfg: Mapping[str, Any]) -> OPDSignals:
        return OPDSignals()

    # ---- driver side -----------------------------------------------------
    @classmethod
    def batch_stage(
        cls,
        td: dict[str, torch.Tensor],
        uid: Optional[np.ndarray],
        cfg: Mapping[str, Any],
        step: int = 0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Return tensors to write into the batch (must contain ``advantages``)."""
        mask = td["response_mask"].float()
        adv = (td["ref_log_prob"] - td["old_log_probs"]) * mask
        return {"advantages": adv}, common.reward_metrics(td, adv)

    # ---- actor side ------------------------------------------------------
    @classmethod
    def actor_loss(
        cls, mb: dict[str, Any], cfg: Mapping[str, Any], actor_config: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return common.pg_loss_from_mb(mb, actor_config)
