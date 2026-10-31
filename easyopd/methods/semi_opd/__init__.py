# Copyright 2026 EasyOPD Contributors
"""semi_opd: Semi-OPD (arXiv 2610.11291) -- OPD on *fixed* offline student rollouts.

Before training, the initial student samples one response per training prompt
(``experiments/07_opd_repro/scripts/gen_offline_rollouts.py`` writes them to the
parquet column ``offline_response_ids``). During training no rollout is generated:
each step replays the stored responses of its prompts (``replay.py``, hooked at
``[EasyOPD:semi_opd]`` in ``ray_trainer.fit``), and the loss is exactly
vanilla OPD's -- A_t = log pi_T(y_t) - log pi_theta(y_t) with pi_theta the
*current* student (verl recomputes ``old_log_probs`` with the actor every step),
PPO clip with ratio 1 at one update per step. So the only difference from
``vanilla_opd`` is where the response comes from.

Deviation from the paper: the teacher logprob is recomputed online by the ref
worker each step instead of being cached once (same numbers, more compute);
no official code was found (docs/methods/semi_opd.md).
"""

from easyopd.methods.vanilla_opd import VanillaOPDMethod
from easyopd.registry import register_method

__all__ = ["SemiOPDMethod", "OFFLINE_RESPONSE_KEY"]

OFFLINE_RESPONSE_KEY = "offline_response_ids"


@register_method("semi_opd")
class SemiOPDMethod(VanillaOPDMethod):
    name = "semi_opd"
    description = "OPD on fixed offline rollouts of the initial student (replayed each step), vanilla OPD loss."
    paper_url = "https://arxiv.org/abs/2610.11291"
    code_url = ""
    verl_modified_files = ("verl/trainer/ppo/ray_trainer.py",)
