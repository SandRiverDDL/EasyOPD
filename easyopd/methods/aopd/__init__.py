# Copyright 2026 EasyOPD Contributors
"""aopd: Asymmetric OPD, OPD where teacher beats student, top-K forward KL elsewhere.

See ``easyopd/docs/methods/aopd.md``. Config keys:

- ``topk`` K (default 32)
- ``threshold`` tau (default 0.0): gate G = 1[P_T(y) - P_S(y) < tau] (official
  ``prob_diff = P_S - P_T > -tau``; the paper writes <=, differs only on ties)
- ``opd_weight`` (default 1.0), ``fkl_weight`` (default 1.0)
- ``fkl_divide_k`` (default False): paper Eq. 7 1/K factor; official code omits it

Advantage = opd_weight * (log P_T - log P_S) on non-gated tokens, 0 on gated
ones (PPO-clip). FKL_t = sum_{v in topK_T} P_T(v)(log P_T(v) - log P_S(v)) with
P_T the full-vocab teacher top-K probs (no renormalisation) and P_S the
full-vocab student. Both terms aggregated with ``loss_agg_mode`` over all valid
tokens and summed (official ``dp_actor.py:457-523``).
"""

from easyopd.opd import OPDMethod, OPDSignals, common
from easyopd.registry import register_method

__all__ = ["AOPDMethod", "aopd_gate", "raw_topk_fkl"]


def aopd_gate(ref_lp, old_lp, mask, tau: float):
    """1 where the token goes to FKL: P_S - P_T > -tau."""
    return ((old_lp.double().exp() - ref_lp.double().exp()) > -tau).to(mask.dtype) * mask


def raw_topk_fkl(teacher_topk_lp, student_lp):
    """sum_v exp(t)(t - s) without renormalisation (official ``compute_gkd_loss`` beta=1)."""
    t = teacher_topk_lp.float()
    return (t.exp() * (t - student_lp.float())).sum(-1)


@register_method("aopd")
class AOPDMethod(OPDMethod):
    name = "aopd"
    description = "AOPD: probability-gap gate, OPD on positive tokens, teacher top-K forward KL on the rest."
    paper_url = "https://arxiv.org/abs/2605.06387"
    code_url = "https://github.com/stargazer319/AOPD"

    @classmethod
    def signals(cls, cfg):
        k = int(common.cfg_get(cfg, "topk", 32))
        return OPDSignals(teacher_topk=k, student_gather_teacher_topk=True)

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        mask = td["response_mask"].float()
        tau = float(common.cfg_get(cfg, "threshold", 0.0))
        w = float(common.cfg_get(cfg, "opd_weight", 1.0))
        gate = aopd_gate(td["ref_log_prob"], td["old_log_probs"], mask, tau)
        adv = w * (td["ref_log_prob"] - td["old_log_probs"]) * mask * (1.0 - gate)
        metrics = common.reward_metrics(td, adv)
        metrics["opd/gkd_token_ratio"] = float(gate.sum().item() / mask.sum().clamp(min=1).item())
        return {"advantages": adv, "opd_fkl_gate": gate}, metrics

    @classmethod
    def actor_loss(cls, mb, cfg, actor_config):
        common.require(mb, ("ref_topk_log_probs", "student_gathered_log_probs", "opd_fkl_gate"), "aopd")
        pg, metrics = common.pg_loss_from_mb(mb, actor_config)
        mode = common.cfg_get(actor_config, "loss_agg_mode", "token-mean")
        fkl = raw_topk_fkl(mb["ref_topk_log_probs"], mb["student_gathered_log_probs"])
        if bool(common.cfg_get(cfg, "fkl_divide_k", False)):
            fkl = fkl / mb["ref_topk_log_probs"].shape[-1]
        gate = mb["opd_fkl_gate"].float()
        fkl_loss = common.agg(fkl * gate, mb["response_mask"], mode)
        metrics["opd/fkl_loss"] = float(fkl_loss.detach().item())
        metrics["opd/gkd_token_ratio_mb"] = float(common.masked_mean(gate, mb["response_mask"]).item())
        return pg + float(common.cfg_get(cfg, "fkl_weight", 1.0)) * fkl_loss, metrics
