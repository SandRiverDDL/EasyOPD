# Copyright 2026 EasyOPD Contributors
"""topd: TOP-D (proximal teacher reward + future-mean returns + per-prompt z-score).

See ``easyopd/docs/methods/topd.md``. Config keys:

- ``alpha`` (float in (0, 1], default 0.1): component A interpolation
- ``variant`` (``full`` | ``naive``, default ``full``): ``naive`` = only A
  (official ``compute_naive_topd_advantage``)
- ``group_norm`` (bool, default True): component C (only for ``full``)

Component D (PPO-clip over several mini-batches) is a launcher setting:
``ppo_mini_batch_size < train_batch_size`` with
``loss_agg_mode=seq-mean-token-mean`` as in the official scripts.
"""

from easyopd.opd import OPDMethod, common, rewards
from easyopd.registry import register_method

__all__ = ["TOPDMethod"]


@register_method("topd")
class TOPDMethod(OPDMethod):
    name = "topd"
    description = "TOP-D: proximal teacher reward log(a*pi_T/pi + 1-a), future-mean returns, group z-score."
    paper_url = "https://arxiv.org/abs/2607.04751"
    code_url = "https://github.com/xie-lab-ml/TOP-D"

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        alpha = float(common.cfg_get(cfg, "alpha", 0.1))
        variant = str(common.cfg_get(cfg, "variant", "full"))
        if variant not in ("full", "naive"):
            raise ValueError(f"[EasyOPD:topd] variant must be full|naive, got {variant}")
        mask = td["response_mask"].float()
        r = rewards.topd_immediate(td["ref_log_prob"], td["old_log_probs"], mask, alpha)
        adv = r
        if variant == "full":
            adv = rewards.topd_returns(r, mask)
            if bool(common.cfg_get(cfg, "group_norm", True)):
                group = common.group_index(uid, mask.shape[0])
                adv = rewards.group_token_zscore(adv, mask, group)
        metrics = common.reward_metrics(td, adv)
        metrics["opd/topd_immediate_mean"] = float(common.masked_mean(r, mask).item())
        return {"advantages": adv}, metrics
