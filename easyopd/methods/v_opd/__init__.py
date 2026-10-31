# Copyright 2026 EasyOPD Contributors
"""v_opd: vOPD, sampled-token OPD with a detached -KL value baseline.

See ``easyopd/docs/methods/v_opd.md`` (registered as ``v_opd`` because
``vopd`` is taken by vision_opd). Config keys:

- ``kl_top_k`` (default 20): support = student top-k at rollout policy
- ``use_baseline`` (default True): False = plain sampled-token OPD

a_t = (log q(y_t) - log p(y_t)) + KL(p_bar || q_bar), with p_bar / q_bar the
student / teacher renormalised over the student top-k (official
``opd_trainer.py:164-183``). The baseline is computed from ``old_topk_*`` /
``ref_gathered_log_probs`` on the driver, i.e. from the rollout-time policy;
with one PPO epoch over one mini-batch this equals the official on-the-fly value.
"""

from easyopd.opd import OPDMethod, OPDSignals, common, rewards
from easyopd.registry import register_method

__all__ = ["VOPDMethod", "topk_rkl_baseline"]


def topk_rkl_baseline(student_topk_lp, teacher_at_student_lp):
    """KL(p_bar || q_bar) over the last dim after renormalising both sides (float32+)."""
    s = common.renorm(student_topk_lp.float())
    t = common.renorm(teacher_at_student_lp.float())
    return common.kl_from_logs(s, t)


@register_method("v_opd")
class VOPDMethod(OPDMethod):
    name = "v_opd"
    description = "vOPD: sampled-token OPD + detached top-k reverse-KL control-variate baseline."
    paper_url = "https://arxiv.org/abs/2605.07865"
    code_url = "https://github.com/holi-lab/vOPD"

    @classmethod
    def signals(cls, cfg):
        k = int(common.cfg_get(cfg, "kl_top_k", 20))
        return OPDSignals(student_topk=k, teacher_at_student_topk=True)

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        mask = td["response_mask"].float()
        adv = rewards.vanilla_advantage(td["ref_log_prob"], td["old_log_probs"], mask)
        metrics = {}
        if bool(common.cfg_get(cfg, "use_baseline", True)):
            common.require(td, ("old_topk_log_probs", "ref_gathered_log_probs"), "v_opd")
            base = topk_rkl_baseline(td["old_topk_log_probs"], td["ref_gathered_log_probs"]).to(adv.dtype)
            adv = adv + base * mask
            metrics["opd/vopd_baseline_mean"] = float(common.masked_mean(base, mask).item())
        metrics.update(common.reward_metrics(td, adv))
        return {"advantages": adv}, metrics
