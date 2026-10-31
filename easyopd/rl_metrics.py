"""Group-level outcome metrics for group-based RL (GRPO / DAPO, rollout.n > 1).

New file (10-30), no upstream reference. Called from ray_trainer.fit right after compute_advantage and
before filter_groups, so it sees every generated group. A group is one prompt (same `uid`) with n rollouts.

Correctness per rollout: the reward manager's `acc` extra info when present (DAPO manager stores the raw
verifier score there, before the overlong penalty is added), otherwise the summed token_level_scores
(naive manager with our 0/1 reward_fn). A rollout counts as correct when that value > 0.5.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def group_outcome_metrics(batch: Any, n: int) -> dict[str, float]:
    if n <= 1 or "uid" not in batch.non_tensor_batch:
        return {}
    acc = batch.non_tensor_batch.get("acc")
    if acc is None:
        acc = batch.batch["token_level_scores"].sum(-1).float().cpu().numpy()
    correct = np.asarray(acc, dtype=np.float64) > 0.5
    uid = np.asarray(batch.non_tensor_batch["uid"])

    _, inv = np.unique(uid, return_inverse=True)
    n_groups = int(inv.max()) + 1
    n_correct = np.bincount(inv, weights=correct, minlength=n_groups)
    n_total = np.bincount(inv, minlength=n_groups)
    all_correct = int((n_correct == n_total).sum())
    all_wrong = int((n_correct == 0).sum())
    mixed = n_groups - all_correct - all_wrong
    return {
        "rl/group_all_correct": float(all_correct),
        "rl/group_all_wrong": float(all_wrong),
        "rl/group_mixed_frac": mixed / n_groups,
    }
