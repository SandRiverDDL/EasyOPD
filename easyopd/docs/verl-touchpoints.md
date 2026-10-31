# verl Touchpoints — EasyOPD Hook Unification

This document tracks every `[EasyOPD:*]` marker that remains in `verl/`
after the Hook Unification work. Each entry must fall into **exactly one**
of the three permitted categories:

1. **5 hook dispatch points** (loss / rollout / reward / alignment / teacher_sidecar).
2. **Configuration field passthrough** (`easyopd.method` config wiring).
3. **Cross-worker rollout scheduling** (Ray actor orchestration the Hook
   layer cannot yet abstract — explicit TODO required).

Anything outside these three reasons must be moved out of `verl/` and into
`easyopd/methods/<method>/hooks.py`.

## Snapshot

Run the auditor to refresh:

```bash
grep -rE 'loss_mode\s*==\s*"' verl/        # MUST be 0
grep -rc '\[EasyOPD' verl/ | awk -F: '{s+=$2} END{print s}'  # target ≤ 8
```

## Marker Inventory

| File:Line | Hook Category | Reason | TODO? |
|---|---|---|---|
| `verl/workers/actor/dp_actor.py:~96` | dispatcher init | Eager `HookDispatcher.from_config(self.config)` so actor processes can route loss hooks in-place. | — |
| `verl/workers/actor/dp_actor.py:~545` | config passthrough | Reads `policy_loss.loss_mode` and resolves canonical method name via `registry.resolve_method_name`. | — |
| `verl/workers/actor/dp_actor.py:~700` | loss hook | Dispatches into `easyopd.methods.vision_opd.core.compute_self_distillation_loss` for vopd. Future: route via `dispatcher.compute_loss_with_context`. | TODO: full migration to dispatcher.compute_loss |
| `verl/trainer/ppo/ray_trainer.py:~480` | dispatcher init | Trainer-side `HookDispatcher.from_config(self.config)` for build_teacher_batch. | — |
| `verl/trainer/ppo/ray_trainer.py:~1505` | rollout/advantage hook | SOD step-wise advantage path. Currently goes through `dispatcher` for advantage correction. | — |
| `verl/trainer/ppo/ray_trainer.py:~2356` | teacher_batch builder | Vision-OPD `_maybe_build_vision_opd_batch`. | TODO: lift into `dispatcher.build_teacher_batch` |
| `verl/trainer/ppo/ray_trainer.py:~2380` | teacher_batch builder | OPSA `_maybe_build_opsa_batch`. | TODO: lift into `dispatcher.build_teacher_batch` |
| `verl/trainer/config/algorithm.py:*` | config passthrough | `easyopd.method` generic config field declaration. | — |
| `verl/workers/config/rollout.py` `SamplingConfig`, `verl/trainer/config/rollout/rollout.yaml` `val_kwargs` (+ both `_generated_*.yaml`) | config passthrough | `[EasyOPD:VAL_LEN]` `val_kwargs.response_length` (default null = rollout.response_length). Not method-specific; lets in-training validation generate longer than training rollouts. | — |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` (`__init__` max_model_len; `generate_sequences` validate branch + padding) | rollout (framework) | `[EasyOPD:VAL_LEN]` engine `max_model_len` covers prompt+val length; validate calls set `max_tokens` and pad responses to the val length. Only the sync vLLM path; sglang / agent_loop ignore the field. | — |
| `verl/trainer/ppo/ray_trainer.py` (`fit` save condition, `_rotate_ckpts`, `_save_checkpoint(kind=)`), `verl/workers/fsdp_workers.py` actor `save_checkpoint(save_contents=)` | config passthrough (framework) | `[EasyOPD:CKPT]` two ckpt kinds: full (`trainer.save_freq` / `max_actor_ckpt_to_keep`, resumable) and weights-only (`+trainer.save_weights_freq` / `+trainer.max_weights_ckpt_to_keep`, under `<default_local_dir>/weights/`, never resumed from). `save_weights_freq <= 0` = original behaviour. In weights mode both kinds rotate by directory scan on the driver. FSDP actor only; megatron workers do not take `save_contents`. | — |
| `verl/trainer/ppo/ray_trainer.py` (`fit`, right after `compute_advantage`, before `_apply_filter_groups`) | metrics (framework) | `[EasyOPD:RL_METRICS]` `easyopd/rl_metrics.py` logs `rl/group_all_correct`, `rl/group_all_wrong` (counts of prompt groups) and `rl/group_mixed_frac` when `rollout.n > 1`. Correctness = reward manager `acc` if present, else summed `token_level_scores` > 0.5. | — |
| `verl/trainer/ppo/ray_trainer.py` (`fit`, gen block) | rollout replay (method) | `[EasyOPD:semi_opd]` if the train parquet has `offline_response_ids`, `easyopd/methods/semi_opd/replay.py` builds the rollout DataProto from the stored ids (cut to `data.max_response_length`) instead of calling `generate_sequences`; logs `semi_opd/replayed=1`. Validation still generates (and syncs weights to vLLM). | semi_opd |
| `verl/trainer/ppo/ray_trainer.py` (`fit`, old_log_prob block, before `pop("entropys")`) | diagnostics (framework) | `[EasyOPD:OPD]` on steps where `opd.dump_freq` divides the step, the per-token full-vocab student entropy is kept as `old_entropy`; `easyopd/opd/__init__.py:run_batch_stage` writes it to the token dump (`student_entropy`) and pops it before the actor update. | all OPD methods |

## Cross-Worker Rollout Scheduling Exemption

The OPCD/OPSA teacher rollout requires Ray actors that cross worker process
boundaries. Until `RolloutHook` is upgraded to support cross-process Ray
calls (planned future work), the trainer keeps these dispatch sites as
explicit method calls. They are tagged with the marker `[EasyOPD:OPSA]` /
`[EasyOPD]` and must remain in `ray_trainer.py` only because dispatcher
cannot serialize a remote Ray handle today.

**Future work**: upgrade `RolloutHook` Protocol to accept
`teacher_actor_handle: Optional[ray.ObjectRef]` and route the
trainer-to-worker call through dispatcher.

## Adding a New Method — Quick Reference

If you find yourself wanting to add a marker to verl/ in order to wire in
a new method, **STOP**. Instead:

1. Create `easyopd/methods/<your_method>/`.
2. Implement `hooks.py` with the appropriate `*LossHook` / `*RolloutHook` etc.
3. Register via `@register_method("your_method")` in `__init__.py`.
4. Add yaml at `easyopd/config/<your_method>.yaml`.

If your method genuinely cannot be expressed via one of the 5 hook
categories above, **add a new hook type to `easyopd/hook_dispatch.py`**
(not to `verl/`). See `add-new-method.md` for the full guide.
