# Copyright 2026 EasyOPD Contributors
"""Shared tensor helpers for OPD methods (pure torch, CPU-testable)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch

# verl files touched by the unified OPD integration (also listed in
# easyopd/docs/verl-touchpoints.md under [EasyOPD:OPD]).
OPD_VERL_FILES = (
    "verl/workers/actor/dp_actor.py",
    "verl/workers/fsdp_workers.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/trainer/main_ppo.py",
    "verl/workers/config/actor.py",
    "verl/trainer/config/actor/actor.yaml",
)

NEG_INF_FILL = -1e4


def cfg_get(cfg: Optional[Mapping[str, Any]], key: str, default: Any) -> Any:
    if cfg is None:
        return default
    val = cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)
    return default if val is None else val


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def renorm(log_p: torch.Tensor) -> torch.Tensor:
    """Renormalise log-probs over the last (support) dimension."""
    return log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)


def kl_from_logs(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """sum_v p (log p - log q) over the last dim."""
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def agg(loss_mat: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """verl ``agg_loss`` with empty rows tolerated in seq-mean-token-mean."""
    mask = mask.to(loss_mat.dtype)
    if mode == "token-mean":
        return (loss_mat * mask).sum() / mask.sum().clamp(min=1.0)
    if mode == "seq-mean-token-sum":
        return (loss_mat * mask).sum(-1).mean()
    if mode == "seq-mean-token-mean":
        return ((loss_mat * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    if mode == "seq-mean-token-sum-norm":
        return (loss_mat * mask).sum() / mask.shape[-1]
    raise ValueError(f"Invalid loss_agg_mode: {mode}")


def ppo_clip_loss_mat(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float,
    clip_high: float,
    clip_c: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dual-clip PPO loss (same math as verl compute_policy_loss_vanilla).

    Returns (loss_mat, clipped_indicator).
    """
    neg_kl = (log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = torch.exp(neg_kl)
    l1 = -advantages * ratio
    l2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    clip1 = torch.maximum(l1, l2)
    l3 = -advantages * clip_c
    clip2 = torch.min(l3, clip1)
    loss = torch.where(advantages < 0, clip2, clip1)
    return loss, (l2 > l1).float()


def clip_params(actor_config: Any) -> tuple[float, float, float]:
    clip = float(cfg_get(actor_config, "clip_ratio", 0.2))
    lo = cfg_get(actor_config, "clip_ratio_low", None)
    hi = cfg_get(actor_config, "clip_ratio_high", None)
    c = float(cfg_get(actor_config, "clip_ratio_c", 3.0))
    return float(lo if lo is not None else clip), float(hi if hi is not None else clip), c


def pg_loss_from_mb(
    mb: Mapping[str, Any],
    actor_config: Any,
    advantages: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    loss_agg_mode: Optional[str] = None,
    prefix: str = "opd",
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO-clip policy-gradient loss on (possibly re-masked) advantages."""
    adv = mb["advantages"] if advantages is None else advantages
    m = mb["response_mask"] if mask is None else mask
    mode = loss_agg_mode or cfg_get(actor_config, "loss_agg_mode", "token-mean")
    lo, hi, c = clip_params(actor_config)
    loss_mat, clipped = ppo_clip_loss_mat(mb["log_prob"], mb["old_log_prob"], adv, lo, hi, c)
    loss = agg(loss_mat, m, mode)
    metrics = {
        f"{prefix}/pg_loss": float(loss.detach().item()),
        f"{prefix}/pg_clipfrac": float(masked_mean(clipped, m).item()),
        f"{prefix}/ppo_kl": float(masked_mean(mb["old_log_prob"] - mb["log_prob"].detach(), m).item()),
    }
    return loss, metrics


def global_token_weight(mask: torch.Tensor) -> torch.Tensor:
    """Per-token weight w = mask * B / sum(mask), computed on the full driver batch.

    With ``loss_agg_mode="seq-mean-token-sum"`` in the actor, verl's micro-batch
    scaling (rows / mini_batch_rows) and the data-parallel gradient average turn
    ``sum(w * loss) / rows`` into ``sum(mask * loss) / sum(mask)`` over the whole
    batch (exact when one mini-batch covers the batch; otherwise normalised by the
    expected per-mini-batch count). Used when the paper's denominator is a
    batch-level token count (e.g. DIAL's M_B) rather than verl's per-micro-batch one.
    """
    mask = mask.float()
    return mask * (mask.shape[0] / mask.sum().clamp(min=1.0))


def reward_metrics(td: Mapping[str, torch.Tensor], adv: torch.Tensor, prefix: str = "opd") -> dict[str, float]:
    mask = td["response_mask"].float()
    lr = td["ref_log_prob"] - td["old_log_probs"]
    out = {
        f"{prefix}/log_ratio_mean": float(masked_mean(lr, mask).item()),
        f"{prefix}/neg_rkl_sampled": float(masked_mean(lr, mask).item()),
        f"{prefix}/adv_mean": float(masked_mean(adv, mask).item()),
        f"{prefix}/adv_abs_max": float((adv.abs() * mask).max().item()) if mask.any() else 0.0,
        f"{prefix}/teacher_logp_mean": float(masked_mean(td["ref_log_prob"], mask).item()),
    }
    return out


def group_index(uid: Optional[np.ndarray], n: int) -> torch.Tensor:
    """Map uid strings to contiguous group ids; whole batch = one group if None."""
    if uid is None:
        return torch.zeros(n, dtype=torch.long)
    _, inv = np.unique(np.asarray(uid), return_inverse=True)
    return torch.as_tensor(inv, dtype=torch.long)


def require(mb: Mapping[str, Any], keys: tuple[str, ...], method: str) -> None:
    missing = [k for k in keys if k not in mb or mb[k] is None]
    if missing:
        raise RuntimeError(
            f"[EasyOPD:{method}] missing tensors {missing}. They are produced only by the "
            "rmpad / non-fused / non-ulysses forward; check use_remove_padding, "
            "use_fused_kernels=False, ulysses_sequence_parallel_size=1 and the method's signals()."
        )
