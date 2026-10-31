# Copyright 2026 EasyOPD Contributors
"""CPU tests for the same-tokenizer OPD methods (easyopd/opd + easyopd/methods/*).

Synthetic batches are built from random full-vocab logits so every top-k /
gathered signal is exactly what the verl plumbing would produce. Where the
official code is pure python / torch (IER ``ier_metrics.py``, ResOPD
``resopd.py``) it is imported from the local clone and compared directly.
"""

from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import pytest
import torch

from easyopd.opd import common, get_opd_method, rewards
from easyopd.registry import ensure_discovered, get_method

OSS = "/mnt/NLU_disk17/chenhaoyu02/oss_refs"
OPD_METHODS = [
    "vanilla_opd", "power_opd", "topd", "dial_opd", "ier", "v_opd",
    "eopd", "revisit_opd", "aopd", "res_opd", "rock_tokens", "distillm2",
]
ACTOR_CFG = {"clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.28, "clip_ratio_c": 3.0,
             "loss_agg_mode": "token-mean"}
FP32 = {"rtol": 1e-5, "atol": 1e-6}  # methods compute KL terms in float32 (bf16 safety)


def _load(path, name):
    if not os.path.exists(path):
        pytest.skip(f"official reference not available: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_batch(sig, B=4, R=12, V=48, seed=0, eos=None):
    """td (driver view) + mb (actor view, with grad through ``s_logits``)."""
    g = torch.Generator().manual_seed(seed)
    s_logits = (torch.randn(B, R, V, generator=g, dtype=torch.float64) * 2).requires_grad_(True)
    t_logits = torch.randn(B, R, V, generator=g, dtype=torch.float64) * 2
    s_lp_full = s_logits.log_softmax(-1)
    t_lp_full = t_logits.log_softmax(-1)
    # sample with the seeded generator (Categorical.sample() would use the global RNG)
    resp = torch.multinomial(s_lp_full.detach().exp().view(B * R, V), 1, generator=g).view(B, R)
    lens = torch.tensor([R, R - 3, R - 5, 1][:B])
    mask = (torch.arange(R)[None] < lens[:, None]).float()
    if eos is not None:
        resp[0, R - 1] = eos
    td = {
        "response_mask": mask,
        "responses": resp,
        "old_log_probs": s_lp_full.detach().gather(-1, resp[..., None]).squeeze(-1),
        "ref_log_prob": t_lp_full.gather(-1, resp[..., None]).squeeze(-1),
    }
    if sig.teacher_topk:
        v, i = t_lp_full.topk(sig.teacher_topk, dim=-1)
        td["ref_topk_log_probs"], td["ref_topk_indices"] = v, i
    if sig.teacher_entropy:
        td["ref_entropy"] = -(t_lp_full.exp() * t_lp_full).sum(-1)
    if sig.student_topk:
        v, i = s_lp_full.detach().topk(sig.student_topk, dim=-1)
        td["old_topk_log_probs"], td["old_topk_indices"] = v, i
        if sig.teacher_at_student_topk:
            td["ref_gathered_log_probs"] = t_lp_full.gather(-1, i)
    mb_extra = {
        "log_prob": s_lp_full.gather(-1, resp[..., None]).squeeze(-1),
        "old_log_prob": td["old_log_probs"],
        "student_gathered_log_probs": (
            s_lp_full.gather(-1, td["ref_topk_indices"]) if sig.student_gather_teacher_topk else None
        ),
        "entropy": None,
    }
    return td, mb_extra, s_logits, t_lp_full


def run_method(name, cfg=None, seed=0, uid=None):
    ensure_discovered()
    cls = get_method(name)
    cfg = dict(cfg or {})
    td, extra, s_logits, t_lp = make_batch(cls.signals(cfg), seed=seed)
    out, metrics = cls.batch_stage(dict(td), uid, cfg, step=3)
    assert "advantages" in out and all(k == "advantages" or k.startswith("opd_") for k in out)
    mb = dict(td)
    mb.update(out)
    mb.update(extra)
    loss, lm = cls.actor_loss(mb, cfg, ACTOR_CFG)
    return dict(td=td, out=out, metrics=metrics, loss=loss, lm=lm, s_logits=s_logits, t_lp=t_lp, mb=mb)


# ---------------------------------------------------------------------------
# contract / smoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", OPD_METHODS)
def test_method_smoke_backward(name):
    r = run_method(name, uid=np.array(["a", "a", "b", "b"]))
    assert torch.isfinite(r["loss"]).all()
    r["loss"].backward()
    grad = r["s_logits"].grad
    assert grad is not None and torch.isfinite(grad).all()
    # padding positions never receive gradient
    pad = r["td"]["response_mask"] == 0
    assert grad[pad].abs().max().item() == 0.0
    assert all(math.isfinite(v) for v in r["metrics"].values())


def test_get_opd_method_dispatch():
    for name in OPD_METHODS:
        cls, cfg = get_opd_method({"opd": {"method": name}})
        assert cls is get_method(name) and cfg["method"] == name
    assert get_opd_method({"opd": {}}) == (None, None)
    with pytest.raises(ValueError):
        get_opd_method({"opd": {"method": "gkd"}})  # registered but not an OPDMethod


def test_vanilla_matches_sampled_rkl_gradient():
    """With ratio = 1 (on-policy) the PPO surrogate gradient is -A * grad log p."""
    r = run_method("vanilla_opd")
    r["loss"].backward()
    g1 = r["s_logits"].grad.clone()
    td = r["td"]
    s = r["s_logits"].detach().clone().requires_grad_(True)
    lp = s.log_softmax(-1).gather(-1, td["responses"][..., None]).squeeze(-1)
    adv = (td["ref_log_prob"] - td["old_log_probs"]) * td["response_mask"]
    ref = -(adv * lp * td["response_mask"]).sum() / td["response_mask"].sum()
    ref.backward()
    torch.testing.assert_close(g1, s.grad)


# ---------------------------------------------------------------------------
# sampled-token reward transforms
# ---------------------------------------------------------------------------


def test_power_opd_advantage_and_terminal_mask():
    ensure_discovered()
    cls = get_method("power_opd")
    td, _, _, _ = make_batch(cls.signals({}), eos=7)
    out, _ = cls.batch_stage(dict(td), None, {"alpha": 2.0, "terminal_eos_ids": [7]})
    m = td["response_mask"].clone()
    m[0, -1] = 0
    exp = (td["ref_log_prob"].exp() ** 2 - td["old_log_probs"].exp() ** 2) * m
    torch.testing.assert_close(out["opd_mask"], m)
    torch.testing.assert_close(out["advantages"] * m, exp)


def test_topd_returns_and_group_norm():
    torch.manual_seed(0)
    r = torch.randn(3, 6)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]]).float()
    got = rewards.topd_returns(r, mask)
    for b in range(3):
        n = int(mask[b].sum())
        for k in range(n):
            fut = r[b, k + 1 : n]
            want = r[b, k] + (fut.mean() if fut.numel() else 0.0)
            assert abs(float(got[b, k]) - float(want)) < 1e-6
    imm = rewards.topd_immediate(torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1), 0.3)
    assert abs(float(imm)) < 1e-7  # log(a*1 + 1-a) = 0
    z = rewards.group_token_zscore(got, mask, torch.tensor([0, 0, 1]))
    sel = mask.bool() & torch.tensor([True, True, False])[:, None]
    assert abs(float(z[sel].mean())) < 1e-5


# ---------------------------------------------------------------------------
# DIAL-OPD
# ---------------------------------------------------------------------------


def test_dial_scores_selection_and_global_weight():
    from easyopd.methods.dial_opd import core as dial

    p = torch.log(torch.tensor([[0.5, 0.01, 0.2, 0.9]]))
    q = torch.log(torch.tensor([[0.1, 0.02, 0.6, 0.9]]))
    for beta in (0.0, 0.5, 1.0):
        want = (1 - beta) * (q - p).abs().log() + beta * (q.exp() - p.exp()).abs().log()
        got = dial.log_scores(p, q, beta)
        torch.testing.assert_close(got[:, :3], want[:, :3].double())
    assert torch.isneginf(dial.log_scores(p, q, 1.0)[0, 3])  # p == q never selected first
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1] * 10]).float()
    k = dial.per_row_k(mask, 0.2)
    assert k.tolist() == [1, 2]
    sel = dial.dial_select(torch.randn(2, 10), torch.randn(2, 10), mask, 0.2, 1.0)
    assert sel.sum(-1).tolist() == [1, 2] and (sel * (1 - mask)).sum() == 0
    r1 = dial.dial_select(torch.randn(2, 10), torch.randn(2, 10), mask, 0.2, 1.0, "random", 5)
    r2 = dial.dial_select(torch.randn(2, 10), torch.randn(2, 10), mask, 0.2, 1.0, "random", 5)
    torch.testing.assert_close(r1, r2)
    # global weight: sum(w * l) / B == sum(sel * l) / sum(sel)
    w = common.global_token_weight(sel)
    l = torch.randn(2, 10)
    torch.testing.assert_close((w * l).sum(-1).mean(), (sel * l).sum() / sel.sum())


def test_dial_loss_denominator_is_batch_selected_count():
    r = run_method("dial_opd", {"keep_ratio": 0.3})
    sel = (r["out"]["opd_weight"] > 0).float()
    mb = r["mb"]
    adv = r["out"]["advantages"]
    want = -(adv * sel).sum() / sel.sum()  # ratio = 1 on-policy
    torch.testing.assert_close(r["loss"], want)


# ---------------------------------------------------------------------------
# IER
# ---------------------------------------------------------------------------


def test_ier_matches_official_compute_position():
    from easyopd.methods.ier import core as ier

    off = _load(f"{OSS}/IER-OPD/slime/rollout/ier_opd/ier_metrics.py", "ier_metrics_official")
    ensure_discovered()
    sig = get_method("ier").signals({"topk": 6})
    for seed in range(4):
        td, _, _, _ = make_batch(sig, V=20, seed=seed)  # small V -> overlaps + sampled-in-topk
        got = ier.ier_tokens(td)
        m = td["response_mask"].bool()
        for b, t in m.nonzero().tolist():
            s_items = list(zip(td["old_topk_indices"][b, t].tolist(), td["old_topk_log_probs"][b, t].tolist()))
            t_items = list(zip(td["ref_topk_indices"][b, t].tolist(), td["ref_topk_log_probs"][b, t].tolist()))
            want = off.compute_position(
                s_items, t_items, sampled_id=int(td["responses"][b, t]),
                sampled_s_logp=float(td["old_log_probs"][b, t]), sampled_t_logp=float(td["ref_log_prob"][b, t]),
            )
            w = want["ier_r"] if "ier_r" in want else want["ier"]
            assert abs(float(got[b, t]) - w) <= 1e-6 * max(1.0, abs(w)), (seed, b, t, float(got[b, t]), w)
        assert (got[~m] == 0).all()


def test_ier_budget_select():
    from easyopd.methods.ier import core as ier

    mask = torch.tensor([[1] * 10, [1] * 5 + [0] * 5, [1] * 3 + [0] * 7]).float()
    scores = torch.arange(30, dtype=torch.float64).view(3, 10)
    sel = ier.budget_select(scores, mask, 0.1, min_keep=1)  # ceil(1.8) = 2 globally -> both in row 2
    assert sel[2].tolist()[:3] == [0, 1, 1] and sel[0].sum() == 1 and sel[1].sum() == 1
    assert sel[0, 9] == 1 and sel[1, 4] == 1  # min_keep picks each row's best valid token
    assert (sel * (1 - mask)).sum() == 0
    a = ier.budget_select(mask, mask, 0.2, random_seed=43)
    b = ier.budget_select(mask, mask, 0.2, random_seed=43)
    torch.testing.assert_close(a, b)
    assert (a * (1 - mask)).sum() == 0 and (a.sum(-1) >= 1).all()


# ---------------------------------------------------------------------------
# losses that read teacher top-k / student gathered log-probs
# ---------------------------------------------------------------------------


def _fresh_student(r):
    s = r["s_logits"].detach().clone().requires_grad_(True)
    return s, s.log_softmax(-1)


def test_v_opd_baseline_is_topk_rkl():
    r = run_method("v_opd", {"kl_top_k": 5})
    td = r["td"]
    s = td["old_topk_log_probs"] - td["old_topk_log_probs"].logsumexp(-1, keepdim=True)
    t = td["ref_gathered_log_probs"] - td["ref_gathered_log_probs"].logsumexp(-1, keepdim=True)
    kl = (s.exp() * (s - t)).sum(-1)
    want = (td["ref_log_prob"] - td["old_log_probs"] + kl) * td["response_mask"]
    torch.testing.assert_close(r["out"]["advantages"], want.to(r["out"]["advantages"].dtype), **FP32)
    assert (kl >= -1e-9).all()


def test_eopd_fkl_gradient():
    cfg = {"topk": 4, "entropy_threshold": 3.0}
    r = run_method("eopd", cfg)
    r["loss"].backward()
    td = r["td"]
    s, lp = _fresh_student(r)
    m = td["response_mask"]
    pg = -((td["ref_log_prob"] - td["old_log_probs"]) * m * lp.gather(-1, td["responses"][..., None]).squeeze(-1)).sum() / m.sum()
    tt = td["ref_topk_log_probs"].log_softmax(-1)
    fkl = (tt.exp() * (tt - lp.gather(-1, td["ref_topk_indices"]))).sum(-1)
    gate = (td["ref_entropy"] > 3.0).float() * m
    (pg + (fkl * gate).sum() / m.sum()).backward()
    torch.testing.assert_close(r["s_logits"].grad, s.grad)
    assert 0 < gate.sum() < m.sum()  # gate is non-trivial on this batch


def test_revisit_loss_and_first_occurrence_mask():
    from easyopd.methods.revisit_opd import first_occurrence_mask

    resp = torch.tensor([[5, 3, 5, 3, 9], [3, 3, 1, 5, 2]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]]).float()
    got = first_occurrence_mask(resp, mask, [3, 5])
    assert got.tolist() == [[0, 0, 1, 1, 0], [0, 0, 1, 0, 1]]
    r = run_method("revisit_opd", {"topk": 6})
    td = r["td"]
    s = r["mb"]["student_gathered_log_probs"].log_softmax(-1)
    t = td["ref_topk_log_probs"].log_softmax(-1)
    kl = (s.exp() * (s - t)).sum(-1)
    m = td["response_mask"]
    torch.testing.assert_close(r["loss"], (kl * m).sum() / m.sum(), check_dtype=False, **FP32)
    assert (r["out"]["advantages"] == 0).all()


def test_aopd_gate_advantage_and_fkl():
    r = run_method("aopd", {"topk": 5})
    td, out = r["td"], r["out"]
    m = td["response_mask"]
    gate = ((td["old_log_probs"].exp() - td["ref_log_prob"].exp()) > 0).float() * m
    torch.testing.assert_close(out["opd_fkl_gate"], gate)
    torch.testing.assert_close(out["advantages"], (td["ref_log_prob"] - td["old_log_probs"]) * m * (1 - gate))
    assert (out["advantages"] >= 0).all()  # tau = 0 -> only positive reinforcement
    tk = td["ref_topk_log_probs"]
    fkl = (tk.exp() * (tk - r["mb"]["student_gathered_log_probs"])).sum(-1)
    want = -(out["advantages"] * m).sum() / m.sum() + (fkl * gate * m).sum() / m.sum()
    torch.testing.assert_close(r["loss"], want, **FP32)


def test_res_opd_matches_official_gradient():
    off = _load(f"{OSS}/ResOPD/verl/trainer/distillation/resopd.py", "resopd_official")
    r = run_method("res_opd", {"topk": 4})
    m = r["td"]["response_mask"]
    r["loss"].backward()
    td = r["td"]
    s, _ = _fresh_student(r)
    o = off.resopd_loss(
        s, td["ref_topk_log_probs"], td["ref_topk_indices"],
        td["ref_log_prob"][..., None], td["responses"][..., None],
    )
    want = (o["distillation_losses"] * m).sum() / m.sum()
    torch.testing.assert_close(r["loss"].detach(), want.detach())
    want.backward()
    torch.testing.assert_close(r["s_logits"].grad, s.grad)
    hit = (td["ref_topk_indices"] == td["responses"][..., None]).any(-1)
    assert hit[m.bool()].any() and (~hit[m.bool()]).any()  # both branches exercised


def test_rock_tokens_weights_and_normalisation():
    from easyopd.methods.rock_tokens import load_token_ids

    assert len(load_token_ids({})) == 100 and len(load_token_ids({"token_list": "random"})) == 100
    r0 = run_method("rock_tokens", {"token_ids": [1, 2, 3, 4, 5, 6, 7, 8], "freeze_weight": 0.0})
    td, m = r0["td"], r0["td"]["response_mask"]
    hit = torch.isin(td["responses"], torch.arange(1, 9)) & m.bool()
    w = (~hit).float() * m
    torch.testing.assert_close(r0["out"]["opd_weight"], w)
    adv = r0["out"]["advantages"]
    torch.testing.assert_close(r0["loss"], -(adv * w).sum() / w.sum())
    r1 = run_method("rock_tokens", {"token_ids": [1, 2, 3, 4, 5, 6, 7, 8], "normalize": "valid"})
    torch.testing.assert_close(r1["loss"], -(adv * w).sum() / m.sum())
    r2 = run_method("rock_tokens", {"token_ids": [1, 2, 3, 4, 5, 6, 7, 8], "freeze_weight": 1.0})
    torch.testing.assert_close(r2["loss"], run_method("vanilla_opd")["loss"])


def test_distillm2_skew_rkl():
    from easyopd.methods.distillm2 import skew_rkl
    from easyopd.methods.revisit_opd import topk_renorm_rkl

    torch.manual_seed(0)
    q, p = torch.randn(3, 7).log_softmax(-1), torch.randn(3, 7).log_softmax(-1)
    torch.testing.assert_close(skew_rkl(q, p, 0.0), topk_renorm_rkl(q, p))
    a = 0.1
    mix = torch.log((1 - a) * p.exp() + a * q.exp())
    torch.testing.assert_close(skew_rkl(q, p, a), (q.exp() * (q - mix)).sum(-1))
    assert (skew_rkl(q, p, a) <= topk_renorm_rkl(q, p) + 1e-6).all()  # skewing shrinks the KL


def test_cross_eos_swaps_only_terminal_endoftext():
    from types import SimpleNamespace

    from easyopd.opd import restore_terminal, swap_terminal_for_teacher

    P, R = 3, 5
    resp = torch.tensor([[5, 6, 151643, 0, 0], [5, 151643, 7, 8, 9], [5, 6, 151645, 0, 0], [0, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
    ids = torch.cat([torch.full((4, P), 9), resp.clone()], 1)
    b = SimpleNamespace(batch={"responses": resp.clone(), "input_ids": ids.clone(), "response_mask": mask})
    assert swap_terminal_for_teacher(b, {"terminal_stop": "raw"}) is None
    info = swap_terminal_for_teacher(b, {"terminal_stop": "cross_eos"})
    exp = resp.clone()
    exp[0, 2] = 151645  # only row 0: mid-sequence endoftext (row 1), im_end (row 2), empty (row 3) untouched
    assert torch.equal(b.batch["responses"], exp) and torch.equal(b.batch["input_ids"][:, P:], exp)
    m = restore_terminal(b, info)
    assert torch.equal(b.batch["responses"], resp) and torch.equal(b.batch["input_ids"], ids)
    assert m["opd/cross_eos_frac"] == 0.25


def test_trainer_signals_keep_method_topk():
    from easyopd.opd import trainer_signals

    van, _ = get_opd_method({"opd": {"method": "vanilla_opd"}})
    s = trainer_signals(van, {"method": "vanilla_opd", "diag_topk": 20, "diag_teacher_entropy": True})
    assert (s.teacher_topk, s.student_topk, s.teacher_at_student_topk, s.teacher_entropy) == (20, 20, True, True)
    assert not s.student_gather_teacher_topk
    assert trainer_signals(van, {"method": "vanilla_opd"}) == van.signals({})
    v, cfg = get_opd_method({"opd": {"method": "v_opd"}})
    own = v.signals(cfg)
    s2 = trainer_signals(v, {**cfg, "diag_topk": 20})
    assert s2.student_topk == (own.student_topk or 20) and s2.teacher_topk == (own.teacher_topk or 20)


def test_analysis_scalars_overlap_and_groups():
    from easyopd.opd import analysis

    torch.manual_seed(0)
    B, R, V, K = 2, 6, 50, 4
    s_full, t_full = torch.randn(B, R, V).log_softmax(-1), torch.randn(B, R, V).log_softmax(-1)
    resp = torch.randint(0, V, (B, R))
    mask = torch.ones(B, R)
    s_lp, s_idx = s_full.topk(K, -1)
    t_lp, t_idx = t_full.topk(K, -1)
    td = {"response_mask": mask, "responses": resp, "old_log_probs": s_full.gather(-1, resp[..., None])[..., 0],
          "ref_log_prob": t_full.gather(-1, resp[..., None])[..., 0], "ref_entropy": torch.ones(B, R),
          "old_topk_log_probs": s_lp, "old_topk_indices": s_idx, "ref_topk_log_probs": t_lp, "ref_topk_indices": t_idx,
          "ref_gathered_log_probs": t_full.gather(-1, s_idx)}
    adv = td["ref_log_prob"] - td["old_log_probs"]
    out = analysis.step_scalars(td, adv, {})
    agree = (s_idx[..., 0] == t_idx[..., 0]).float().mean()
    assert math.isclose(out["ana/ov_top1_agree"], float(agree), abs_tol=1e-6)
    assert math.isclose(out["ana/ov_t_mass_on_s_topk"], float(t_full.gather(-1, s_idx).exp().sum(-1).mean()), rel_tol=1e-5)
    assert out["ana/ov_s_mass_on_t_topk_lb"] <= out["ana/ov_s_topk_mass"] + 1e-6
    assert out["ana/teacher_entropy_mean"] == 1.0
    c = [out[f"ana/adv_conc_top{q}"] for q in (1, 5, 10, 20, 50)]
    assert all(0 < x <= 1 + 1e-6 for x in c) and c == sorted(c)
    assert analysis.token_group("x", 151643) == 0 and analysis.token_group(" \\(", 1) == 2
    assert analysis.token_group("\n\n", 1) == 1 and analysis.token_group("###", 1) == 3
    assert analysis.token_group(" Let", 1) == 4 and analysis.token_group(" apple", 1) == 5
    lut = torch.full((V,), 5, dtype=torch.int8)
    g = analysis._groups(td, adv, mask.bool(), lut)
    assert g["ana/grp_other/tok_frac"] == 1.0 and math.isclose(g["ana/grp_other/adv_abs_share"], 1.0, rel_tol=1e-5)
