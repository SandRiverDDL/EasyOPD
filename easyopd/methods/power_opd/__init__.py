# Copyright 2026 EasyOPD Contributors
"""power_opd: PowerOPD, reward r_t = pi_T(y_t)^alpha - pi_old(y_t)^alpha.

See ``easyopd/docs/methods/power_opd.md``. Config keys
(``actor_rollout_ref.actor.opd.*``):

- ``alpha`` (float, default 5.0; official script value, paper uses 500)
- ``zscore`` (bool, default False): token z-score over the batch
- ``terminal_eos_ids`` (list[int], default []): drop the last token when it is
  one of these ids (official ``TERMINAL_STOP_TARGET=mask``)

Official training is REINFORCE; with ``ppo_epochs=1`` and
``ppo_mini_batch_size == train_batch_size`` the PPO ratio is 1 and the
gradient is identical.
"""

from easyopd.opd import OPDMethod, common, rewards
from easyopd.registry import register_method

__all__ = ["PowerOPDMethod"]


@register_method("power_opd")
class PowerOPDMethod(OPDMethod):
    name = "power_opd"
    description = "PowerOPD: power-transformed teacher/student probability reward."
    paper_url = "https://arxiv.org/abs/2606.17199"
    code_url = "https://github.com/EIT-NLP/PowerOPD"

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        alpha = float(common.cfg_get(cfg, "alpha", 5.0))
        mask = rewards.terminal_mask(
            td["responses"], td["response_mask"].float(), list(common.cfg_get(cfg, "terminal_eos_ids", []))
        )
        adv = rewards.power_advantage(td["ref_log_prob"], td["old_log_probs"], mask, alpha)
        if bool(common.cfg_get(cfg, "zscore", False)):
            adv = rewards.zscore_tokens(adv, mask)
        metrics = common.reward_metrics(td, adv)
        metrics["opd/kept_token_frac"] = float(mask.sum().item() / td["response_mask"].float().sum().clamp(min=1).item())
        return {"advantages": adv, "opd_mask": mask}, metrics

    @classmethod
    def actor_loss(cls, mb, cfg, actor_config):
        return common.pg_loss_from_mb(mb, actor_config, mask=mb["opd_mask"])
