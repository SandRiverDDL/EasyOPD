# Copyright 2026 EasyOPD Contributors
"""Math answer scoring shared by 07_opd_repro training (monitoring only) and eval.

New file. Reuses verl's scorers: ``verl.utils.reward_score.math.compute_score``
(last ``\\boxed{}`` + string normalisation, the scorer of
``experiments/_shared/scripts/reward_fn.py``) first, and falls back to
``math_verify`` (symbolic equivalence) when the string check fails.
OPD methods ignore this reward for the gradient; it is logged as
``critic/score/mean`` so the training curve stays interpretable.
"""

from verl.utils.reward_score.math import compute_score as _string_score

try:
    from verl.utils.reward_score.math_verify import compute_score as _verify_score
except Exception:  # math_verify not importable -> string check only
    _verify_score = None


def score_answer(solution_str: str, ground_truth: str) -> float:
    # A \boxed{} answer is required (math_verify alone would also accept a bare last number).
    if not solution_str or "\\boxed" not in solution_str:
        return 0.0
    tail = solution_str[solution_str.rfind("\\boxed"):]  # keeps math_verify on the final answer only
    try:
        if _string_score(tail, ground_truth) >= 1.0:
            return 1.0
    except Exception:
        pass
    if _verify_score is None:
        return 0.0
    try:
        return float(_verify_score(tail, ground_truth) >= 1.0)
    except Exception:
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return score_answer(solution_str, ground_truth)
