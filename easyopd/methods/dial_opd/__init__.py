# Copyright 2026 EasyOPD Contributors
"""dial_opd: DIAL-OPD, select tokens by s = |r|^{1-beta} |q-p|^beta per response.

See ``easyopd/docs/methods/dial_opd.md``. Config keys:

- ``keep_ratio`` (default 0.2), ``score_beta`` (default 1.0)
- ``selector`` (``dial`` | ``random``): ``random`` = uniform same-k control,
  re-drawn every step with seed ``seed + step``
- ``seed`` (default 42)
- ``terminal_eos_ids`` (list[int]): excluded from V_i (official
  ``mask_terminal_eos``); set to the tokenizer eos id(s)

Loss = -(1/M_B) sum_sel sg(r) log pi with M_B = selected tokens of the whole
batch (``common.global_token_weight``). Matches official REINFORCE when
ppo_epochs=1 and mini-batch = batch.
"""

from easyopd.opd import OPDMethod, common, rewards
from easyopd.registry import register_method

from . import core

__all__ = ["DIALOPDMethod"]


@register_method("dial_opd")
class DIALOPDMethod(OPDMethod):
    name = "dial_opd"
    description = "DIAL-OPD: logarithmic-mean token scoring, per-response top-k selection."
    paper_url = "https://arxiv.org/abs/2610.11659"
    code_url = "https://github.com/EIT-NLP/DIAL-OPD"

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        ratio = float(common.cfg_get(cfg, "keep_ratio", 0.2))
        beta = float(common.cfg_get(cfg, "score_beta", 1.0))
        selector = str(common.cfg_get(cfg, "selector", "dial"))
        seed = int(common.cfg_get(cfg, "seed", 42)) + int(step)
        eff = rewards.terminal_mask(
            td["responses"], td["response_mask"].float(), list(common.cfg_get(cfg, "terminal_eos_ids", []))
        )
        sel = core.dial_select(td["old_log_probs"], td["ref_log_prob"], eff, ratio, beta, selector, seed)
        adv = rewards.vanilla_advantage(td["ref_log_prob"], td["old_log_probs"], sel)
        metrics = common.reward_metrics(td, adv)
        n_valid = eff.sum().clamp(min=1).item()
        metrics.update(
            {
                "opd/selected_frac": float(sel.sum().item() / n_valid),
                "opd/selected_tokens": float(sel.sum().item()),
                "opd/low_low_frac": core.low_low_frac(td["old_log_probs"], td["ref_log_prob"], sel),
            }
        )
        # Batch-level denominator M_B (official: selected count of the whole window).
        return {"advantages": adv, "opd_weight": common.global_token_weight(sel)}, metrics

    @classmethod
    def actor_loss(cls, mb, cfg, actor_config):
        return common.pg_loss_from_mb(mb, actor_config, mask=mb["opd_weight"], loss_agg_mode="seq-mean-token-sum")
