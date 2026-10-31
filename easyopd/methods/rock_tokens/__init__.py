# Copyright 2026 EasyOPD Contributors
"""rock_tokens: Rock-Freeze, reweight the OPD loss on a static list of "rock" token ids.

See ``easyopd/docs/methods/rock_tokens.md``. Config keys:

- ``token_list`` (``rock`` | ``random`` | path to a JSON list of ids, default
  ``rock``): ``rock`` / ``random`` are the official Qwen3 lists
  (``lists/*.json``, copied from YuxuanJiang1/Rock-Token ``stumbling/``, MIT)
- ``token_ids`` (list[int]): overrides ``token_list`` when non-empty
- ``freeze_weight`` lambda (default 0.0): weight of tokens in the list
- ``normalize`` (``sum_w`` | ``valid``, default ``sum_w``): ``sum_w`` divides by
  sum of weights (official ``token_freeze_kd.py:95``), ``valid`` by the number
  of valid tokens (same scale as the unfrozen baseline)

Base objective = sampled-token OPD (A = log q - log p_old, PPO-clip,
token-mean). The window W_R of paper Eq. 18 is not implemented (the official
code has none either). A token is matched on the sampled id ``responses[t]``,
i.e. the token being predicted at t.
"""

import json
import os
from functools import lru_cache

import torch

from easyopd.opd import OPDMethod, common
from easyopd.registry import register_method

__all__ = ["RockTokensMethod", "load_token_ids", "rock_weights"]

_LIST_DIR = os.path.join(os.path.dirname(__file__), "lists")


@lru_cache(maxsize=8)
def _load(path: str) -> tuple[int, ...]:
    with open(path) as f:
        ids = json.load(f)
    return tuple(int(i) for i in ids)


def load_token_ids(cfg) -> list[int]:
    ids = list(common.cfg_get(cfg, "token_ids", []))
    if ids:
        return [int(i) for i in ids]
    name = str(common.cfg_get(cfg, "token_list", "rock"))
    path = os.path.join(_LIST_DIR, f"{name}.json") if name in ("rock", "random") else name
    return list(_load(path))


def rock_weights(responses, mask, token_ids, lam: float):
    """w = lam on listed tokens, 1 elsewhere, 0 on padding."""
    ids = torch.as_tensor(token_ids, dtype=responses.dtype, device=responses.device)
    hit = torch.isin(responses, ids)
    w = torch.where(hit, torch.full_like(mask, lam), torch.ones_like(mask))
    return w * mask, hit & mask.bool()


@register_method("rock_tokens")
class RockTokensMethod(OPDMethod):
    name = "rock_tokens"
    description = "Rock Tokens: freeze / down-weight a static token-id list in sampled-token OPD."
    paper_url = "https://arxiv.org/abs/2605.09253"
    code_url = "https://github.com/YuxuanJiang1/Rock-Token"

    @classmethod
    def batch_stage(cls, td, uid, cfg, step=0):
        out, metrics = super().batch_stage(td, uid, cfg, step)
        mask = td["response_mask"].float()
        lam = float(common.cfg_get(cfg, "freeze_weight", 0.0))
        if lam < 0:
            raise ValueError(f"[EasyOPD:rock_tokens] freeze_weight must be >= 0, got {lam}")
        w, hit = rock_weights(td["responses"], mask, load_token_ids(cfg), lam)
        out["opd_weight"] = w
        metrics["opd/rock_token_frac"] = float(hit.sum().item() / mask.sum().clamp(min=1).item())
        return out, metrics

    @classmethod
    def actor_loss(cls, mb, cfg, actor_config):
        norm = str(common.cfg_get(cfg, "normalize", "sum_w"))
        w = mb["opd_weight"].to(mb["advantages"].dtype)
        if norm == "sum_w":
            # sum(w * l) / sum(w)
            return common.pg_loss_from_mb(mb, actor_config, mask=w, loss_agg_mode="token-mean")
        if norm == "valid":
            # PPO dual-clip loss is positively homogeneous in A, so w*A == w*loss.
            return common.pg_loss_from_mb(mb, actor_config, advantages=mb["advantages"] * w, loss_agg_mode="token-mean")
        raise ValueError(f"[EasyOPD:rock_tokens] normalize must be sum_w|valid, got {norm}")
