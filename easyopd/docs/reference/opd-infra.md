# 同 tokenizer OPD 的现有基础设施（挂点地图）

> 常青文档。2026-10-29 只读梳理所得，**全部来自读代码，没有实际跑过**；跑通或改代码后要更新这里。
> 涉及代码：`verl/workers/fsdp_workers.py`、`verl/workers/actor/dp_actor.py`、`verl/trainer/ppo/ray_trainer.py`、`verl/trainer/main_ppo.py`、`verl/workers/config/actor.py`、`verl/trainer/distillation/losses.py`、`easyopd/methods/{g_opd,gkd,opsa}/`、`experiments/_shared/scripts/`。

结论：本仓同 tokenizer 下唯一接通的 teacher 信号是「teacher 对 student 采样 token 的 logprob」（`ref_log_prob`），teacher 就是和 actor 同卡的 FSDP ref worker。要 teacher top-k / 熵 / 在指定 indices 取值的方法，都要先扩展 ref worker 的输出；只要采样 token logprob 的方法（PowerOPD、TOP-D、DIAL-OPD）可以直接在现有 advantage 覆盖点上实现。verl 版本是 `0.5.0.dev`，rollout 用 vLLM 0.8.5。

## teacher 从哪来

- teacher = RefPolicy worker，挂在 `global_pool`，和 actor / rollout 共用同 8 张卡（`main_ppo.py:183,204-213`）。只有设了 `use_kl_in_reward`、`use_kl_loss`、`token_kl_reg.enable` 或 `opsa_enable` 之一才会创建 ref worker。
- 模型路径取 `actor_rollout_ref.ref.model.path`（缺省回落到 actor 路径），g_opd 的启动脚本用 `+actor_rollout_ref.ref.model.path=$TEACHER` 指向 teacher（`fsdp_workers.py:672-696`）。
- `compute_ref_log_prob`（`fsdp_workers.py:890-937`）返回 `ref_log_prob`，形状 (B, resp_len)，只是采样 token 的 logprob。
- 只有 OPSA 会在 `meta_info["opsa_topk_k"]` 里请求 top-k（k=512），此时额外返回 `ref_topk_log_probs` / `ref_topk_indices`（`ray_trainer.py:1438-1488`）。**全词表 teacher logits 从不返回。**

## 信号进入训练的三条路

| 路径 | 位置 | 现状 |
|---|---|---|
| advantage 覆盖（vanilla OPD） | `dp_actor.py:931-973`：`only_reverse_kl_advantages=True` 时用 `ref_log_prob − old_log_prob` 整体替换 GRPO advantage，之后走标准 PG loss（`dp_actor.py:1361`） | 已接通，g_opd 用 |
| advantage 叠加 | `ray_trainer.py:1622-1734` `_apply_token_kl_regularizer`：`grpo_adv + coef·w·(ref−old)` | 已接通，sod 用 |
| 新 adv estimator | `core_algos` 注册，`ray_trainer.py:322-328` 透传 | lightning_opd 用（teacher logprob 预先算在 parquet 里） |
| 直接 loss | `dp_actor.py` 里按 `loss_mode` 分支（vision_opd 979+、sdpo、opcd、opsa 1168-1316、跨 tokenizer 1319-1358），其余落到 `get_policy_loss_fn` | 每个方法自己加分支 |

注意：`easyopd/hooks.py` 的 `LossHook.compute_loss` 在 verl 里**没有调用点**，`hook_dispatch` 只用于判断是否启用（`ray_trainer.py:1642`）；`DISTILLATION_LOSS_REGISTRY` 里只有跨 tokenizer 的 5 个 loss。`gkd` 的 launch 传的 `+distillation.*` 参数在 `verl/` 里没有读取方。

## vanilla OPD 怎么配

`policy_loss.only_reverse_kl_advantages=True`、`lambda_vals=1.0`、不配 base model，`A = ref_log_prob − old_log_prob`，不用 task reward。需要同时让 ref worker 被创建（例如 `use_kl_loss=True` + `kl_loss_coef=0`）。`experiments/06_general_opd/methods/g_opd/launch.sh` 没设任何创建 ref 的开关，按代码推断会在 `dp_actor.py:970` 报 KeyError，未实测。

## 命名冲突

- 已注册方法：alm、dskd、echo_kd、g_opd、gad、gkd、lightning_opd、opcd、opsa、ropd、sdpo、sft、simct、simple、sod、uld、vision_opd；**`vopd` 是 vision_opd 的别名**（`vision_opd/__init__.py:55`），所以方差缩减的 vOPD 注册为 `v_opd`。
- policy loss：vanilla、gspo、gpg、clip_cov、kl_cov、geo_mean；adv estimator：gae、grpo、grpo_passk、reinforce_plus_plus(_baseline)、rloo、opo、remax、gpg、on_policy_distillation。重名在 `easyopd/registry.py:75-131` 报错。

## 各类改动的挂点

| 改动 | 挂点 |
|---|---|
| 只用采样 token logprob 的 reward 变换 / advantage（PowerOPD、TOP-D、AOPD 的门控） | `dp_actor.py:931-973` 的 advantage 覆盖，或新 adv estimator |
| token 选择 / 重加权（DIAL-OPD、IER、Rock Tokens） | `policy_loss_fn` 之前改 `response_mask` / advantages，注意分母 |
| 需要 teacher top-k、熵（EOPD、Revisiting OPD、AOPD 的 FKL、ResOPD、IER） | 泛化 OPSA 的 `opsa_topk_k` 请求（`ray_trainer.py:1438-1488`、`fsdp_workers.py:890-937`），新 loss_mode 分支放在 `dp_actor.py:1361` 之前 |
| 需要 teacher 在 student top-k indices 处取值（vOPD） | ref worker 增加「按给定 indices gather」的输出，比 top-k 多一次数据交换 |
| 新配置字段 | `PolicyLossConfig`（`verl/workers/config/actor.py:109-141`） |

## 评测与 reward

- `experiments/_shared/scripts/evaluate_model.py`：vLLM，每卡一个子进程（`--dp_size 8`），只支持 math500 / gsm8k 等从 `<data_dir>/<bench>_eval.parquet` 读；温度写死 0、`max_tokens` 2048、`max_model_len` 4096、每题 1 个样本。**没有 AIME / AMC，也没有 avg@k。**
- `evaluate_model_sglang.py`：连 SGLang 服务，温度 0.6，同样没有 AIME / AMC / 多样本。
- `reward_fn.py`：`\boxed{}` 抽取 + verl `math.compute_score` 字符串等价，没有用 math_verify。parquet 列要求 `prompt`（chat list）、`data_source`、`reward_model.ground_truth`、`extra_info`。
- 两个脚本和 launch.sh 都有 `/path/to/EasyOPD`、`/opt/conda/envs/OpenAgentRL-sj` 这类占位路径。

## 回归脚本

`scripts/test_all_methods.py` 只做 CPU 契约检查（注册、配置、hook 协议、数据 recipe），不需要 GPU。`EXPECTED_METHODS` 少了 alm、dskd、sft、uld，按代码推断 `:98` 的相等断言会失败，未实跑。新方法要加进这个列表。
