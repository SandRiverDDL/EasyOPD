# Copyright 2026 EasyOPD Contributors
"""ier: IER sparse OPD, keep the top ceil(ratio*N) tokens of the batch by IER.

See ``easyopd/docs/methods/ier.md``. Config keys:

- ``budget_ratio`` (default 0.01), ``min_keep`` (default 1 per response)
- ``selector`` (``ier`` | ``random``): ``random`` = same-budget control with
  seed ``seed + step`` (official ``selector_seed + rollout_id``)
- ``seed`` (default 42), ``topk`` (default 16, both sides), ``floor_logp``
  (default -12), ``ratio_clip`` (default 30)

Advantage = (log q - log p_old) on selected tokens, PPO-clip, aggregated with
``seq-mean-token-mean`` over the selected tokens. Usefulness fusion (IER-OR /
IER-AND) is not implemented.
"""

from easyopd.opd import OPDMethod, OPDSignals, common, rewards
from easyopd.registry import register_method

from . import core

__all__ = ["IERMethod"]


@register_method("ier")
class IERMethod(OPDMethod):
    name = "ier"
    description = "IER: information-to-error-ratio token scoring, global-budget sparse OPD."
    paper_url = "https://arxiv.org/abs/2609.24432"
    code_url = "https://github.com/BruceSheng1202/IER-OPD"

    @classmethod
    def signals(cls, cfg):
        k = int(common.cfg_get(cfg, "topk", 16))
        return OPDSignals(teacher_topk=k, student_topk=k)

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        ratio = float(common.cfg_get(cfg, "budget_ratio", 0.01))
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"[EasyOPD:ier] budget_ratio must be in (0, 1], got {ratio}")
        min_keep = int(common.cfg_get(cfg, "min_keep", 1))
        selector = str(common.cfg_get(cfg, "selector", "ier"))
        mask = td["response_mask"].float()
        if selector == "ier":
            common.require(td, ("old_topk_indices", "old_topk_log_probs", "ref_topk_indices", "ref_topk_log_probs"), "ier")
            scores = core.ier_tokens(
                td,
                floor=float(common.cfg_get(cfg, "floor_logp", -12.0)),
                clip=float(common.cfg_get(cfg, "ratio_clip", 30.0)),
            )
            sel = core.budget_select(scores, mask.cpu(), ratio, min_keep)
        elif selector == "random":
            scores = None
            seed = int(common.cfg_get(cfg, "seed", 42)) + int(step)
            sel = core.budget_select(mask.cpu(), mask.cpu(), ratio, min_keep, random_seed=seed)
        else:
            raise ValueError(f"[EasyOPD:ier] selector must be ier|random, got {selector}")
        sel = sel.to(mask.device).float()
        adv = rewards.vanilla_advantage(td["ref_log_prob"], td["old_log_probs"], sel)
        metrics = common.reward_metrics(td, adv)
        n_valid = mask.sum().clamp(min=1).item()
        metrics.update(
            {
                "opd/selected_frac": float(sel.sum().item() / n_valid),
                "opd/selected_tokens": float(sel.sum().item()),
            }
        )
        if scores is not None:
            m = mask.cpu().bool()
            metrics["opd/ier_mean"] = float(scores[m].mean().item()) if m.any() else 0.0
            metrics["opd/ier_selected_mean"] = float(scores[sel.cpu().bool()].mean().item()) if sel.any() else 0.0
        return {"advantages": adv, "opd_mask": sel}, metrics

    @classmethod
    def actor_loss(cls, mb, cfg, actor_config):
        return common.pg_loss_from_mb(mb, actor_config, mask=mb["opd_mask"], loss_agg_mode="seq-mean-token-mean")
