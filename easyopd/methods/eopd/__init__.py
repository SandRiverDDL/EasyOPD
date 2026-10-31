# Copyright 2026 EasyOPD Contributors
"""eopd: Entropy-gated OPD, PPO sampled-token OPD + teacher-top-k forward KL on high-entropy tokens.

See ``easyopd/docs/methods/eopd.md``. Config keys:

- ``topk`` (default 16): teacher top-k support
- ``entropy_threshold`` tau (default 0.8): gate 1[H_T > tau] (paper; official uses >=)
- ``fkl_coef`` alpha (default 1.0)
- ``student_full_vocab`` (default True): student log-prob is full-vocab normalised
  (paper Eq. 10 / README); False renormalises it over the top-k (official code default)

L = L_PPO(A = log q - log p_old) + alpha * sum_t 1[H_t > tau] FKL_t / N_valid with
FKL_t = sum_{x in topk} q~(x) (log q~(x) - log p(x)); N_valid counts all valid
tokens of the micro-batch (official ``core_algos.py:1277-1279``).
"""

from easyopd.opd import OPDMethod, OPDSignals, common
from easyopd.registry import register_method

__all__ = ["EOPDMethod", "topk_fkl"]


def topk_fkl(teacher_topk_lp, student_lp, student_full_vocab: bool = True):
    """sum_x q~(x) (log q~(x) - log p(x)) over the last dim; q~ = teacher renormalised on top-k."""
    t = common.renorm(teacher_topk_lp.float())
    s = student_lp.float() if student_full_vocab else common.renorm(student_lp.float())
    return common.kl_from_logs(t, s)


@register_method("eopd")
class EOPDMethod(OPDMethod):
    name = "eopd"
    description = "EOPD: sampled-token OPD + teacher-entropy-gated top-k forward KL."
    paper_url = "https://arxiv.org/abs/2603.07079"
    code_url = "https://github.com/WLS04/EOPD"

    @classmethod
    def signals(cls, cfg):
        k = int(common.cfg_get(cfg, "topk", 16))
        return OPDSignals(teacher_topk=k, teacher_entropy=True, student_gather_teacher_topk=True)

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        out, metrics = super().batch_stage(td, uid, cfg, step)
        common.require(td, ("ref_entropy",), "eopd")
        mask = td["response_mask"].float()
        tau = float(common.cfg_get(cfg, "entropy_threshold", 0.8))
        gate = (td["ref_entropy"].float() > tau).float() * mask
        out["opd_fkl_gate"] = gate
        metrics["opd/high_entropy_frac"] = float(gate.sum().item() / mask.sum().clamp(min=1).item())
        metrics["opd/teacher_entropy_mean"] = float(common.masked_mean(td["ref_entropy"].float(), mask).item())
        return out, metrics

    @classmethod
    def actor_loss(cls, mb, cfg, actor_config):
        common.require(mb, ("ref_topk_log_probs", "student_gathered_log_probs", "opd_fkl_gate"), "eopd")
        pg, metrics = common.pg_loss_from_mb(mb, actor_config)
        alpha = float(common.cfg_get(cfg, "fkl_coef", 1.0))
        full = bool(common.cfg_get(cfg, "student_full_vocab", True))
        fkl = topk_fkl(mb["ref_topk_log_probs"], mb["student_gathered_log_probs"], full)
        mask = mb["response_mask"].float()
        gate = mb["opd_fkl_gate"].float() * mask
        fkl_loss = (fkl * gate).sum() / mask.sum().clamp(min=1.0)
        metrics["opd/fkl_loss"] = float(fkl_loss.detach().item())
        metrics["opd/fkl_gated_mean"] = float(
            ((fkl.detach() * gate).sum() / gate.sum().clamp(min=1.0)).item()
        )
        return pg + alpha * fkl_loss, metrics
