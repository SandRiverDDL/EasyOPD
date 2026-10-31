# Copyright 2026 EasyOPD Contributors
"""res_opd: ResOPD, head-exact + tail-bucket control variate + sampled residual.

See ``easyopd/docs/methods/res_opd.md``. Config keys:

- ``topk`` k (default 4): teacher support size

Direct supervision (no PPO ratio; official ``use_policy_gradient=False``),
aggregated with the actor ``loss_agg_mode`` (official default token-mean).
Unbiasedness assumes one on-policy update per rollout: launch with
``ppo_epochs=1`` and ``ppo_mini_batch_size == train_batch_size`` and a
full-support rollout (top_p=1, top_k=-1) unless reproducing the paper's
truncated-rollout table.
"""

from easyopd.opd import OPDMethod, OPDSignals, common
from easyopd.registry import register_method

from . import core

__all__ = ["ResOPDMethod"]


@register_method("res_opd")
class ResOPDMethod(OPDMethod):
    name = "res_opd"
    description = "ResOPD: unbiased sparse-teacher reverse-KL gradient with tail-bucket control variate."
    paper_url = "https://arxiv.org/abs/2610.04882"
    code_url = "https://github.com/InternLM/ResOPD"

    @classmethod
    def signals(cls, cfg):
        k = int(common.cfg_get(cfg, "topk", 4))
        return OPDSignals(teacher_topk=k, student_gather_teacher_topk=True)

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        common.require(td, ("ref_topk_indices",), "res_opd")
        out, metrics = super().batch_stage(td, uid, cfg, step)
        mask = td["response_mask"].float()
        hit = (td["ref_topk_indices"].long() == td["responses"].long()[..., None]).any(-1).float() * mask
        out["opd_head_hit"] = hit
        metrics["opd/head_hit_frac"] = float(hit.sum().item() / mask.sum().clamp(min=1).item())
        return out, metrics

    @classmethod
    def actor_loss(cls, mb, cfg, actor_config):
        common.require(
            mb, ("ref_topk_log_probs", "student_gathered_log_probs", "ref_log_prob", "opd_head_hit"), "res_opd"
        )
        t = core.resopd_terms(
            mb["student_gathered_log_probs"],
            mb["log_prob"],
            mb["ref_topk_log_probs"],
            mb["ref_log_prob"],
            mb["opd_head_hit"] > 0.5,
        )
        mask = mb["response_mask"]
        mode = common.cfg_get(actor_config, "loss_agg_mode", "token-mean")
        loss = common.agg(t["losses"], mask, mode)
        metrics = {
            "opd/resopd_loss": float(loss.detach().item()),
            "opd/student_mass": float(common.masked_mean(t["student_mass"], mask).item()),
            "opd/teacher_mass": float(common.masked_mean(t["teacher_mass"], mask).item()),
            "opd/head_hit": float(common.masked_mean(t["head_hit"], mask).item()),
            "opd/ppo_kl": float(common.masked_mean(mb["old_log_prob"] - mb["log_prob"].detach(), mask).item()),
        }
        return loss.to(mb["log_prob"].dtype), metrics
