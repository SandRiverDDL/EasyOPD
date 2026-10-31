# Copyright 2026 EasyOPD Contributors
"""distillm2: DistiLLM-2 skew reverse KL, on-policy, teacher top-K approximation.

See ``easyopd/docs/methods/distillm2.md``. This is a *deviation* from the paper:

- only the SRKL term on the student rollout y_s is used (no teacher-response
  SKL term, no alpha curriculum, no beta schedule);
- the full-vocab sum is approximated on the teacher top-K support with both
  distributions renormalised on it (like ``revisit_opd``).

Config keys: ``topk`` K (default 32), ``skew_alpha`` (default 0.1, official
``base_alpha_2``), ``kl_coef`` (default 1.0).

SRKL_t = sum_x q^(x) (log q^(x) - log m(x)), m = (1 - a) p^ + a sg(q^), q^ / p^
the student / teacher renormalised on the teacher top-K; the student inside the
mixture is detached as in official ``distillm_trainer.py:1196``. Aggregated as a
per-sequence token mean then batch mean (official ``average_log_prob=True``).
"""

import math

import torch

from easyopd.opd import OPDMethod, OPDSignals, common
from easyopd.registry import register_method

__all__ = ["DistiLLM2Method", "skew_rkl"]


def skew_rkl(student_lp, teacher_lp, alpha: float):
    """KL(q || (1-a) p + a sg(q)) over the last dim, both sides renormalised on it."""
    q = common.renorm(student_lp.float())
    p = common.renorm(teacher_lp.float())
    if alpha <= 0.0:
        return common.kl_from_logs(q, p)
    log_m = torch.logaddexp(p + math.log1p(-alpha), q.detach() + math.log(alpha))
    return common.kl_from_logs(q, log_m)


@register_method("distillm2")
class DistiLLM2Method(OPDMethod):
    name = "distillm2"
    description = "DistiLLM-2 (on-policy SRKL only): skew reverse KL on the teacher top-K support."
    paper_url = "https://arxiv.org/abs/2503.07067"
    code_url = "https://github.com/jongwooko/distillm-2"

    @classmethod
    def signals(cls, cfg):
        k = int(common.cfg_get(cfg, "topk", 32))
        return OPDSignals(teacher_topk=k, student_gather_teacher_topk=True)

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        a = float(common.cfg_get(cfg, "skew_alpha", 0.1))
        if not (0.0 <= a < 1.0):
            raise ValueError(f"[EasyOPD:distillm2] skew_alpha must be in [0, 1), got {a}")
        adv = torch.zeros_like(td["response_mask"].float())
        return {"advantages": adv}, common.reward_metrics(td, adv)

    @classmethod
    def actor_loss(cls, mb, cfg, actor_config):
        common.require(mb, ("ref_topk_log_probs", "student_gathered_log_probs"), "distillm2")
        a = float(common.cfg_get(cfg, "skew_alpha", 0.1))
        kl = skew_rkl(mb["student_gathered_log_probs"], mb["ref_topk_log_probs"], a)
        mask = mb["response_mask"]
        loss = common.agg(kl, mask, "seq-mean-token-mean") * float(common.cfg_get(cfg, "kl_coef", 1.0))
        metrics = {
            "opd/srkl": float(common.masked_mean(kl.detach(), mask).item()),
            "opd/ppo_kl": float(common.masked_mean(mb["old_log_prob"] - mb["log_prob"].detach(), mask).item()),
        }
        return loss, metrics
