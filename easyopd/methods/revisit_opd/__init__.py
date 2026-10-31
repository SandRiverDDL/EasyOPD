# Copyright 2026 EasyOPD Contributors
"""revisit_opd: Revisiting OPD, teacher-top-K renormalised reverse KL as a direct loss.

See ``easyopd/docs/methods/revisit_opd.md``. Config keys:

- ``topk`` K (default 32): teacher top-K support
- ``kl_coef`` (default 1.0)
- ``mask_token_ids`` (list[int], default []): per sequence, drop the *first*
  occurrence of each listed token from the loss (official
  ``opd_mask_special_tokens``; paper default tokens ``<``, ``think``,
  ``<|im_end|>``, pass their ids). Empty = "Ours w/o mask".

loss = coef * token-mean_t sum_{x in TopK_q} pi^(x) (log pi^(x) - log q^(x)),
pi^ / q^ = student / teacher softmax-renormalised over the teacher top-K
(official ``compute_memory_efficient_kl`` with ``norm_to_one``). No policy
gradient term (advantage = 0). Rollout top_p=0.9 is a launcher setting.
"""

import torch

from easyopd.opd import OPDMethod, OPDSignals, common
from easyopd.registry import register_method

__all__ = ["RevisitOPDMethod", "first_occurrence_mask", "topk_renorm_rkl"]


def first_occurrence_mask(responses, mask, token_ids):
    """mask with the first occurrence (per row) of every id in ``token_ids`` removed."""
    out = mask.clone()
    m = mask.bool()
    for tid in token_ids:
        hit = (responses == int(tid)) & m
        first = hit & (hit.long().cumsum(-1) == 1)
        out = out.masked_fill(first, 0)
    return out


def topk_renorm_rkl(student_lp, teacher_lp):
    """sum pi^(log pi^ - log q^) over the last dim, both renormalised on that support."""
    s = common.renorm(student_lp.float())
    t = common.renorm(teacher_lp.float())
    return common.kl_from_logs(s, t)


@register_method("revisit_opd")
class RevisitOPDMethod(OPDMethod):
    name = "revisit_opd"
    description = "Revisiting OPD: teacher top-K renormalised reverse KL loss + optional special-token mask."
    paper_url = "https://arxiv.org/abs/2603.25562"
    code_url = "https://github.com/hhh675597/revisiting_opd"

    @classmethod
    def signals(cls, cfg):
        k = int(common.cfg_get(cfg, "topk", 32))
        return OPDSignals(teacher_topk=k, student_gather_teacher_topk=True)

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        mask = td["response_mask"].float()
        ids = list(common.cfg_get(cfg, "mask_token_ids", []))
        kl_mask = first_occurrence_mask(td["responses"], mask, ids) if ids else mask
        adv = torch.zeros_like(mask)
        metrics = common.reward_metrics(td, adv)
        metrics["opd/masked_special_tokens"] = float((mask - kl_mask).sum().item())
        return {"advantages": adv, "opd_mask": kl_mask}, metrics

    @classmethod
    def actor_loss(cls, mb, cfg, actor_config):
        common.require(mb, ("ref_topk_log_probs", "student_gathered_log_probs", "opd_mask"), "revisit_opd")
        coef = float(common.cfg_get(cfg, "kl_coef", 1.0))
        kl = topk_renorm_rkl(mb["student_gathered_log_probs"], mb["ref_topk_log_probs"])
        loss = common.agg(kl, mb["opd_mask"], "token-mean") * coef
        metrics = {
            "opd/topk_rkl": float(common.masked_mean(kl.detach(), mb["opd_mask"]).item()),
            "opd/ppo_kl": float(common.masked_mean(mb["old_log_prob"] - mb["log_prob"].detach(), mb["response_mask"]).item()),
        }
        return loss, metrics
