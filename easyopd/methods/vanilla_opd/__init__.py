# Copyright 2026 EasyOPD Contributors
"""vanilla_opd: sampled-token on-policy distillation baseline.

advantage_t = log pi_T(y_t) - log pi_old(y_t), PPO-clip policy gradient, no
task reward. This is the shared baseline every other OPD method is compared to
(same as ``OPDMethod`` defaults). Launch with
``+actor_rollout_ref.actor.opd.method=vanilla_opd``.

Ablation switch ``+actor_rollout_ref.actor.opd.adv_transform`` (07_opd_repro
signal ablations; default ``none`` = the baseline above):

* ``sign``: A = sign(log pi_T - log pi_old), i.e. only the direction of the
  per-token signal is kept (+1 / -1, 0 on exact ties).
* ``fixed_neg_lowp``: teacher-free control after OPSA (arXiv 2608.31046):
  tokens whose student logprob ``log pi_old`` is in the lowest ``lowp_frac``
  (default 0.2) of all valid tokens in the batch get A = ``neg_value``
  (default -0.5); every other token gets 0. The quantile is batch-level.
"""

import torch

from easyopd.opd import OPDMethod, common
from easyopd.registry import register_method

__all__ = ["VanillaOPDMethod"]

ADV_TRANSFORMS = ("none", "sign", "fixed_neg_lowp")


@register_method("vanilla_opd")
class VanillaOPDMethod(OPDMethod):
    name = "vanilla_opd"
    description = "Sampled-token OPD baseline: A = log pi_T - log pi_old, PPO clip."
    paper_url = "https://arxiv.org/abs/2306.13649"
    code_url = ""

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        mode = str(common.cfg_get(cfg, "adv_transform", "none"))
        if mode not in ADV_TRANSFORMS:
            raise ValueError(f"[EasyOPD:vanilla_opd] adv_transform={mode!r} not in {ADV_TRANSFORMS}")
        mask = td["response_mask"].float()
        raw = (td["ref_log_prob"] - td["old_log_probs"]) * mask
        if mode == "none":
            adv = raw
        elif mode == "sign":
            adv = torch.sign(raw) * mask
        else:
            frac = float(common.cfg_get(cfg, "lowp_frac", 0.2))
            neg = float(common.cfg_get(cfg, "neg_value", -0.5))
            valid = mask.bool()
            lp = td["old_log_probs"].float()
            thr = torch.quantile(lp[valid], frac) if valid.any() else lp.new_tensor(0.0)
            sel = valid & (lp <= thr)
            adv = sel.float() * neg
        metrics = common.reward_metrics(td, adv)
        if mode != "none":
            metrics[f"opd/adv_transform_{mode}"] = 1.0  # adv sign fractions: ana/adv_{pos,neg}_frac
        return {"advantages": adv}, metrics
