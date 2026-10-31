# Copyright 2026 EasyOPD Contributors
"""Unified same-tokenizer OPD integration (driver + actor entry points).

verl only imports this module; per-method logic lives in
``easyopd/methods/<name>/``. See ``easyopd/opd/base.py`` for the contract.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from easyopd.opd.base import OPDMethod, OPDSignals

__all__ = [
    "OPDMethod",
    "OPDSignals",
    "SIGNAL_KEYS",
    "get_opd_method",
    "trainer_signals",
    "swap_terminal_for_teacher",
    "restore_terminal",
    "run_batch_stage",
    "actor_keys",
]

# Batch keys produced by the OPD signal plumbing that the actor may need.
SIGNAL_KEYS = (
    "ref_log_prob",
    "ref_topk_log_probs",
    "ref_topk_indices",
    "ref_entropy",
    "ref_gathered_log_probs",
    "old_topk_log_probs",
    "old_topk_indices",
)


def _opd_cfg(actor_cfg: Any) -> Optional[Mapping[str, Any]]:
    if actor_cfg is None:
        return None
    opd = actor_cfg.get("opd", None) if hasattr(actor_cfg, "get") else getattr(actor_cfg, "opd", None)
    if not opd:
        return None
    method = opd.get("method", None) if hasattr(opd, "get") else None
    return opd if method else None


def get_opd_method(actor_cfg: Any) -> tuple[Optional[type], Optional[Mapping[str, Any]]]:
    """Return (method class, opd cfg) if ``actor.opd.method`` is set, else (None, None)."""
    cfg = _opd_cfg(actor_cfg)
    if cfg is None:
        return None, None
    from easyopd.registry import ensure_discovered, get_method

    ensure_discovered()
    cls = get_method(str(cfg.get("method")))
    if not (isinstance(cls, type) and issubclass(cls, OPDMethod)):
        raise ValueError(
            f"[EasyOPD:OPD] actor.opd.method={cfg.get('method')!r} is registered but is not an "
            "OPDMethod; it must be launched through its own loss_mode instead."
        )
    return cls, cfg


def trainer_signals(method: Optional[type], cfg: Optional[Mapping[str, Any]]) -> Optional[OPDSignals]:
    """Signals the *trainer* requests from the workers: the method's own, plus what
    the method-agnostic diagnostics (``opd.diag_topk`` / ``opd.diag_teacher_entropy``)
    need. A method's own top-k size is never changed (diagnostics reuse it); only
    signals the method did not ask for are switched on. The actor side keeps calling
    ``method.signals`` so the loss path is unaffected.
    """
    if method is None:
        return None
    from dataclasses import replace

    sig = method.signals(cfg)
    k = int(cfg.get("diag_topk", 0) or 0)
    if k > 0:
        sig = replace(
            sig,
            teacher_topk=sig.teacher_topk or k,
            student_topk=sig.student_topk or k,
            teacher_at_student_topk=True,
        )
    if bool(cfg.get("diag_teacher_entropy", False)):
        sig = replace(sig, teacher_entropy=True)
    return sig


# Qwen3 Base students end with <|endoftext|>, instruct teachers with <|im_end|>.
_CROSS_EOS = (151643, 151645)


def swap_terminal_for_teacher(batch: Any, cfg: Optional[Mapping[str, Any]]) -> Optional[tuple]:
    """``opd.terminal_stop=cross_eos`` (official PowerOPD ``TERMINAL_STOP_TARGET=cross_eos``):
    when a response's last valid token is ``<|endoftext|>``, the teacher scores
    ``<|im_end|>`` at that position instead; the student's action, mask and loss labels
    are unchanged. Only the last valid token is rewritten, so no other position's
    context changes. Returns restore info for :func:`restore_terminal`, or None.
    """
    mode = str(cfg.get("terminal_stop", "raw")) if cfg else "raw"
    if mode == "raw":
        return None
    if mode != "cross_eos":
        raise ValueError(f"[EasyOPD:OPD] opd.terminal_stop={mode!r} not in ('raw', 'cross_eos')")
    src, dst = (int(x) for x in cfg.get("cross_eos_ids", _CROSS_EOS))
    resp = batch.batch["responses"]
    ids = batch.batch["input_ids"]
    mask = batch.batch["response_mask"]
    R = resp.shape[1]
    last = mask.long().sum(-1) - 1
    rows = torch.nonzero(last >= 0, as_tuple=True)[0]
    cols = last[rows]
    rows = rows[resp[rows, cols] == src]
    cols = last[rows]
    resp[rows, cols] = dst
    ids[rows, ids.shape[1] - R + cols] = dst
    return rows, cols, src, int(resp.shape[0])


def restore_terminal(batch: Any, info: Optional[tuple]) -> dict[str, float]:
    if info is None:
        return {}
    rows, cols, src, n = info
    R = batch.batch["responses"].shape[1]
    batch.batch["responses"][rows, cols] = src
    batch.batch["input_ids"][rows, batch.batch["input_ids"].shape[1] - R + cols] = src
    return {"opd/cross_eos_frac": float(len(rows)) / max(n, 1)}


def run_batch_stage(batch: Any, method: type, cfg: Mapping[str, Any], step: int = 0) -> dict[str, float]:
    """Driver-side hook: update ``batch.batch`` in place, return metrics."""
    keys = set(batch.batch.keys())
    td = {k: batch.batch[k] for k in keys if k in SIGNAL_KEYS or k.startswith("opd_")}
    for k in ("response_mask", "responses", "old_log_probs", "advantages", "token_level_scores", "old_entropy"):
        if k in keys:
            td[k] = batch.batch[k]
    if "ref_log_prob" not in td:
        raise RuntimeError(
            "[EasyOPD:OPD] ref_log_prob missing from batch: the teacher (ref worker) did not run. "
            "main_ppo creates it automatically when actor.opd.method is set."
        )
    uid = batch.non_tensor_batch.get("uid", None) if batch.non_tensor_batch is not None else None
    with torch.no_grad():
        out, metrics = method.batch_stage(td, uid, cfg, step=step)
    if "advantages" not in out:
        raise RuntimeError(f"[EasyOPD:{method.name}] batch_stage must return 'advantages'")
    for k, v in out.items():
        if k != "advantages" and not k.startswith("opd_"):
            raise ValueError(f"[EasyOPD:{method.name}] batch_stage may only write 'advantages' or 'opd_*', got {k}")
        batch.batch[k] = v.to(batch.batch["response_mask"].device)
    metrics = dict(metrics)
    metrics[f"opd/method_{method.name}"] = 1.0
    from easyopd.opd import analysis

    with torch.no_grad():
        metrics.update(analysis.step_scalars(td, out["advantages"], cfg))
        analysis.maybe_dump(td, out["advantages"], uid, cfg, step)
    if "old_entropy" in keys:  # only the dump needs it; keep it out of the actor update
        batch.batch.pop("old_entropy")
    return metrics


def actor_keys(available: set[str]) -> list[str]:
    """Keys update_policy must keep for the OPD loss."""
    return sorted(k for k in available if k in SIGNAL_KEYS or k.startswith("opd_"))
