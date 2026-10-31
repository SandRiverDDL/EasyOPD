# Copyright 2026 EasyOPD Contributors
"""Method-agnostic OPD analysis logging, called from ``run_batch_stage`` on the driver.

New file (07_opd_repro analysis needs; no prior implementation in the repo).

1. Per-step scalars (always on, a few masked reductions): ``ana/*`` -- student
   -log p mean / quantiles, fraction of tokens the student / teacher gives p < 0.01,
   fraction of positive advantages. With T=1, top_p=1 rollouts the sampled-token
   -log p mean is an unbiased estimate of the student's entropy.
2. Token-level dump every ``opd.dump_freq`` steps (0 = off) into
   ``opd.dump_dir/step_<N>.parquet``: one row per response with the valid tokens'
   ids, student old log p, teacher log p and final advantage, plus length, score
   and a truncated flag. Also, when present in the batch: ``student_entropy`` (full-vocab,
   kept by ray_trainer on dump steps as ``old_entropy``), ``teacher_entropy``,
   ``teacher_rank``, and the top-k blocks ``{student,teacher}_topk_{ids,logp}`` /
   ``teacher_logp_at_s_topk``, flattened per row to n_tokens * K (row-major; K in ``{student,teacher}_topk_k``).
3. Extra scalars (10-30 additions, all from tensors already in the batch):

   - ``ana/adv_conc_top{1,5,10,20,50}``: share of sum|A| carried by the top-q% tokens by |A|.
   - ``ana/grp_<g>/{tok_frac,adv_mean,adv_abs_share,logratio_mean}`` per structural token
     group (eos / ws / math_delim / markdown / discourse / other), from a vocab lookup
     table built once from ``opd.diag_tokenizer`` (skipped when unset).
   - ``ana/teacher_entropy_mean`` when ``ref_entropy`` is in the batch
     (``opd.diag_teacher_entropy=True``).
   - Top-k overlap when ``opd.diag_topk=K`` (teacher top-K, student top-K and teacher
     logp at the student's top-K are then requested): ``ana/ov_top1_agree``,
     ``ana/ov_jaccard``, ``ana/ov_sampled_in_t_top{1,5,K}``,
     ``ana/ov_t_mass_on_s_topk`` (sum of p_T over the student's top-K),
     ``ana/ov_s_mass_on_t_topk_lb`` (sum of p_S over the student top-K ids that are also in
     the teacher top-K: a lower bound), ``ana/ov_{s,t}_topk_mass`` (own top-K coverage),
     ``ana/ov_kl_s_t_topk`` (sum over the student top-K of p_S (log p_S - log p_T)).
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Mapping, Optional

import numpy as np
import torch

_LOW = math.log(0.01)
_CONC_Q = (1, 5, 10, 20, 50)
GROUPS = ("eos", "ws", "math_delim", "markdown", "discourse", "other")
_EOS_IDS = (151643, 151645)
_MATH = re.compile(r"^(\\[()\[\]]|\$+|\\boxed\{?|[{}]+|\\\\)$")
_MD = re.compile(r"^(#+|\*+|-|>|\d+\.|\d+\))$")
_DISC = {"To", "Let", "First", "Firstly", "So", "Therefore", "Thus", "Hence", "Wait", "Now", "Next",
         "Finally", "However", "But", "Given", "Since", "Step", "Alternatively", "Hmm", "Okay", "Then"}
_GROUP_LUT: dict[str, torch.Tensor] = {}


def token_group(s: str, tid: int) -> int:
    if tid in _EOS_IDS:
        return 0
    t = s.strip()
    if t == "":
        return 1
    if _MATH.match(t):
        return 2
    if _MD.match(t):
        return 3
    if t.rstrip(",:") in _DISC:
        return 4
    return 5


def group_lut(tokenizer_path: str) -> torch.Tensor:
    """(vocab,) int8 group id per token id, built once per tokenizer path."""
    if tokenizer_path not in _GROUP_LUT:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        n = max(len(tok), max(_EOS_IDS) + 1)
        strs = tok.batch_decode([[i] for i in range(n)])
        _GROUP_LUT[tokenizer_path] = torch.tensor([token_group(s, i) for i, s in enumerate(strs)], dtype=torch.int8)
    return _GROUP_LUT[tokenizer_path]


def _concentration(a_abs: torch.Tensor) -> dict[str, float]:
    tot = float(a_abs.sum())
    if tot <= 0:
        return {}
    srt = torch.sort(a_abs, descending=True).values
    cs = torch.cumsum(srt, 0)
    n = srt.numel()
    return {f"ana/adv_conc_top{q}": float(cs[max(1, int(math.ceil(n * q / 100))) - 1]) / tot for q in _CONC_Q}


def _groups(td, adv, mask, lut) -> dict[str, float]:
    ids = td["responses"][mask].long().clamp(0, lut.numel() - 1)
    g = lut.to(ids.device)[ids].long()
    a = adv.float()[mask]
    lr = (td["ref_log_prob"].float() - td["old_log_probs"].float())[mask]
    tot_abs = float(a.abs().sum()) or 1.0
    out = {}
    for gi, name in enumerate(GROUPS):
        m = g == gi
        c = int(m.sum())
        out[f"ana/grp_{name}/tok_frac"] = c / max(g.numel(), 1)
        if c:
            out[f"ana/grp_{name}/adv_mean"] = float(a[m].mean())
            out[f"ana/grp_{name}/adv_abs_share"] = float(a[m].abs().sum()) / tot_abs
            out[f"ana/grp_{name}/logratio_mean"] = float(lr[m].mean())
    return out


def _overlap(td, mask) -> dict[str, float]:
    t_idx, t_lp = td["ref_topk_indices"][mask].long(), td["ref_topk_log_probs"][mask].float()
    s_idx, s_lp = td["old_topk_indices"][mask].long(), td["old_topk_log_probs"][mask].float()
    k = min(t_idx.shape[-1], s_idx.shape[-1])
    t_idx, t_lp, s_idx, s_lp = t_idx[:, :k], t_lp[:, :k], s_idx[:, :k], s_lp[:, :k]
    y = td["responses"][mask].long().unsqueeze(-1)
    eq = s_idx.unsqueeze(-1) == t_idx.unsqueeze(-2)  # (N, k_s, k_t)
    inter = eq.any(-1)  # student top-k id also in teacher top-k
    n_int = inter.sum(-1).float()
    hit = t_idx == y
    out = {
        "ana/ov_top1_agree": float((s_idx[:, 0] == t_idx[:, 0]).float().mean()),
        "ana/ov_jaccard": float((n_int / (2 * k - n_int)).mean()),
        "ana/ov_sampled_in_t_top1": float(hit[:, :1].any(-1).float().mean()),
        f"ana/ov_sampled_in_t_top{k}": float(hit.any(-1).float().mean()),
        "ana/ov_s_topk_mass": float(s_lp.exp().sum(-1).mean()),
        "ana/ov_t_topk_mass": float(t_lp.exp().sum(-1).mean()),
        "ana/ov_s_mass_on_t_topk_lb": float((s_lp.exp() * inter).sum(-1).mean()),
    }
    if k >= 5:
        out["ana/ov_sampled_in_t_top5"] = float(hit[:, :5].any(-1).float().mean())
    if "ref_gathered_log_probs" in td:
        g = td["ref_gathered_log_probs"][mask].float()[:, :k]
        out["ana/ov_t_mass_on_s_topk"] = float(g.exp().sum(-1).mean())
        out["ana/ov_kl_s_t_topk"] = float((s_lp.exp() * (s_lp - g)).sum(-1).mean())
    return out


def step_scalars(td: Mapping[str, torch.Tensor], adv: torch.Tensor,
                 cfg: Optional[Mapping[str, Any]] = None) -> dict[str, float]:
    mask = td["response_mask"].bool()
    if not mask.any():
        return {}
    s = td["old_log_probs"].float()[mask]
    t = td["ref_log_prob"].float()[mask]
    a = adv.float()[mask]
    q = torch.quantile(-s, torch.tensor([0.5, 0.9, 0.99], device=s.device))
    out = {
        "ana/student_nlogp_mean": float((-s).mean()),
        "ana/student_nlogp_p50": float(q[0]),
        "ana/student_nlogp_p90": float(q[1]),
        "ana/student_nlogp_p99": float(q[2]),
        "ana/teacher_nlogp_mean": float((-t).mean()),
        "ana/student_lowp_frac": float((s < _LOW).float().mean()),
        "ana/teacher_lowp_frac": float((t < _LOW).float().mean()),
        "ana/adv_pos_frac": float((a > 0).float().mean()),
        "ana/adv_neg_frac": float((a < 0).float().mean()),
    }
    out.update(_concentration(a.abs()))
    cfg = cfg or {}
    tok_path = cfg.get("diag_tokenizer", None)
    if tok_path:
        out.update(_groups(td, adv, mask, group_lut(str(tok_path))))
    if "ref_entropy" in td:
        out["ana/teacher_entropy_mean"] = float(td["ref_entropy"].float()[mask].mean())
    if all(k in td for k in ("ref_topk_indices", "ref_topk_log_probs", "old_topk_indices", "old_topk_log_probs")):
        out.update(_overlap(td, mask))
    return out


def maybe_dump(td: Mapping[str, torch.Tensor], adv: torch.Tensor, uid: Optional[Any],
               cfg: Mapping[str, Any], step: int, eos_ids: tuple[int, ...] = (151645, 151643)) -> Optional[str]:
    freq = int(cfg.get("dump_freq", 0) or 0)
    out_dir = cfg.get("dump_dir", None)
    if freq <= 0 or not out_dir or step % freq != 0:
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq

    mask = td["response_mask"].bool().cpu()
    resp = td["responses"].cpu()
    s = td["old_log_probs"].float().cpu()
    t = td["ref_log_prob"].float().cpu()
    a = adv.float().cpu()
    score = td["token_level_scores"].float().sum(-1).cpu() if "token_level_scores" in td else None
    t_ent = td["ref_entropy"].float().cpu() if "ref_entropy" in td else None
    s_ent = td["old_entropy"].float().cpu() if "old_entropy" in td else None
    # top-k blocks, stored flat per row (n_tokens * K, row-major); K per side in columns
    # student_topk_k / teacher_topk_k; ids int32, log p float16 (full-vocab normalised)
    blocks = {}
    for name, key in (("student_topk_ids", "old_topk_indices"), ("student_topk_logp", "old_topk_log_probs"),
                      ("teacher_topk_ids", "ref_topk_indices"), ("teacher_topk_logp", "ref_topk_log_probs"),
                      ("teacher_logp_at_s_topk", "ref_gathered_log_probs")):
        if key in td:
            blocks[name] = td[key].cpu()
    ks = {f"{n.split('_')[0]}_topk_k": int(v.shape[-1]) for n, v in blocks.items() if not n.startswith("teacher_logp_at")}
    t_rank = None
    if "ref_topk_indices" in td:  # rank of the sampled token in the teacher top-k, -1 = outside
        hit = td["ref_topk_indices"].long().cpu() == resp.long().unsqueeze(-1)
        t_rank = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full_like(resp, -1))
    max_len = resp.shape[1]
    cols = {k: [] for k in ("step", "row", "uid", "length", "truncated", "score",
                            "token_ids", "student_logp", "teacher_logp", "adv")}
    if t_ent is not None:
        cols["teacher_entropy"] = []
    if s_ent is not None:
        cols["student_entropy"] = []
    if t_rank is not None:
        cols["teacher_rank"] = []
    if blocks:
        for name in [*ks, *blocks]:
            cols[name] = []
    for i in range(resp.shape[0]):
        m = mask[i]
        n = int(m.sum())
        ids = resp[i][m]
        cols["step"].append(step)
        cols["row"].append(i)
        cols["uid"].append(str(uid[i]) if uid is not None else "")
        cols["length"].append(n)
        cols["truncated"].append(bool(n >= max_len and (n == 0 or int(ids[-1]) not in eos_ids)))
        cols["score"].append(float(score[i]) if score is not None else float("nan"))
        cols["token_ids"].append(ids.numpy().astype(np.int32))
        cols["student_logp"].append(s[i][m].numpy().astype(np.float32))
        cols["teacher_logp"].append(t[i][m].numpy().astype(np.float32))
        cols["adv"].append(a[i][m].numpy().astype(np.float32))
        if t_ent is not None:
            cols["teacher_entropy"].append(t_ent[i][m].numpy().astype(np.float32))
        if s_ent is not None:
            cols["student_entropy"].append(s_ent[i][m].numpy().astype(np.float32))
        if t_rank is not None:
            cols["teacher_rank"].append(t_rank[i][m].numpy().astype(np.int8))
        if blocks:
            for name, k in ks.items():
                cols[name].append(k)
            for name, v in blocks.items():
                dt = np.int32 if name.endswith("_ids") else np.float16
                cols[name].append(v[i][m].numpy().astype(dt).reshape(-1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step_{step}.parquet")
    pq.write_table(pa.table(cols), path)
    return path
