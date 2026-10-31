# Trust Region Policy Distillation（`topd`）

涉及代码：`oss_refs/TOP-D/verl/verl/trainer/ppo/core_algos.py`、`oss_refs/TOP-D/verl/verl/trainer/ppo/ray_trainer.py`、`oss_refs/TOP-D/recipe/opd/`（均在 `/mnt/NLU_disk17/chenhaoyu02/` 下）。

- 论文：Trust Region Policy Distillation，arXiv [2607.04751](https://arxiv.org/abs/2607.04751)。作者 Zhengpeng Xie、Li Lyna Zhang（project leader）、Zeke Xie（通讯）、Mao Yang；单位 HKUST(GZ)、HKUST、Shanghai Xingyunzhili AI Institute。v1 发布于 2026-07-06，v2 发布于 2026-10-26。
- 会议：v2 PDF 首页页脚印有「40th Conference on Neural Information Processing Systems (NeurIPS 2026)」（camera-ready 格式），仓库 README 写「Accepted to NeurIPS 2026」。arXiv comments 字段为空，v1 无会议信息；在 OpenReview / neurips.cc 上没有搜到对应条目。结论：NeurIPS 2026 有作者侧 camera-ready 证据，但未在会议官方页面核实。
- 官方实现：有。[xie-lab-ml/TOP-D](https://github.com/xie-lab-ml/TOP-D)，v2 摘要给出链接（v1 没有）。本地浅克隆 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/TOP-D`，commit `9a2716cba22cab508825ad3be3f78327bb8139e5`（2026-10-27）。框架是内置的 verl `0.8.0.dev`，Apache-2.0。「无官方实现」的说法不成立。
- 本仓实现：未实现（计划注册名 `topd`）
- 状态：调研中
- 结论（一句话）：TOP-D 是同 tokenizer、只需 teacher 采样 token logprob 的 OPD 改进，核心是「近端 teacher 奖励变换 + 平均未来回报 + 组内 token z-score + PPO mini-batch 复用」，在 verl 里只是一个新的 advantage estimator，实现成本低。

## 方法要点

标准 OPD 把每个采样 token 的奖励设为 r_k = log π\*(y_k|·) − log π_θ(y_k|·)，即采样 token 上的 reverse KL 负值，直接当 advantage 用（官方代码 `compute_opd_advantage`，`core_algos.py:339`：`advantages = (ref_log_prob - old_log_probs) * response_mask`）。论文认为当 teacher 与 student 差距大时，ρ_k = π\*/π_θ 的对数会出现极大的负值，方差大、训练不稳。TOP-D 由四个可拆开消融的组件构成：

- 组件 A，近端 teacher 奖励变换（论文 Eq 5）。把目标换成概率空间插值的 π̃\* = α·π\* + (1−α)·π_θ，于是奖励变为 r̃_k = log(α·ρ_k + 1 − α)。它有下界 log(1−α)，把极负的奖励截住；α=1 退化为原 OPD。论文 Eq 6 指出在 log 空间插值只得到 α·r_k，等价于缩放，不起作用。代码 `_compute_topd_immediate_rewards`（`core_algos.py:373`）用 `torch.logaddexp(log_ratio + log(α), log1p(-α))` 数值稳定地计算，其中 log_ratio = ref_log_prob − old_log_probs；α 由 `_get_topd_alpha`（`core_algos.py:360`）读取，默认 0.1，要求 0 < α ≤ 1。
- 组件 B，回报处理（论文 Eq 8）。R̃_k = r̃_k + (Σ_{j>k} r̃_j) / max(|y|−k, 1)，即当前奖励加未来奖励的平均（v2 写明空和为 0；v1 的 Eq 8 分母是 |y|−k，并引用 MiniLLM 的长度归一化未来回报）。代码 `_compute_topd_token_returns`（`core_algos.py:392`）用反向 cumsum 减自身得到未来和，未来 token 数为 0 时未来项取 0。
- 组件 C，归一化。对同一 prompt 的 G 条回答，把全部有效 token 的 R̃ 合在一起做 z-score（均值、总体标准差 `unbiased=False`，eps 1e-6）。代码 `_normalize_topd_token_returns_by_prompt`（`core_algos.py:409`）按 `uid` 分组；拿不到 index 时退回全 batch 归一化。
- 组件 D，PPO-clip 与 mini-batch 离策略复用（论文 Eq 7、Algorithm 1，论文称「internal trust region iterations」）。一个 rollout batch 切成多个 mini-batch 依次更新，后面的 mini-batch 相对 old policy 已经离策略，由 ratio clip 约束。论文 Table 5 是 512 prompts 切 16 个 mini-batch（每个 32 prompts），而 OPD baseline 是 1 个 mini-batch。论文附录 C.2 提到 SPO，但实际使用的目标是 PPO clip。

总入口 `compute_topd_advantage`（`core_algos.py:443`）在 no_grad 下依次做 A→B→C，返回 (advantages, token_returns)，不使用 `token_level_rewards`（任务奖励不参与训练）。代码另有 `compute_naive_topd_advantage`（`core_algos.py:476`），只做组件 A，不做 B、C，论文正文里没有这一变体，可直接作为消融。

理论部分：Theorem 4.2 给出奖励方差上界 tr(Var) ≤ M²(log²(1−α) + C\*·α)，α 越小方差越小；Theorem 4.4 给出与 teacher 的距离按 (1−α)^{k+1} 收缩，极限误差 ≤ ε∞/α；Theorem 4.9 给出单调改进保证；附录 C.1 说明某一界对 α 不变。这些是论文的理论陈述，本仓未验证。

## 适用条件

- 同 tokenizer：必须。官方启动脚本 `run_qwen3_1p7b_base_qwen3_teacher_dapo17k_common.sh:243-266` 在启动前断言 student 与 teacher tokenizer 的 EOS、PAD、词表长度一致，并让 ref worker 使用 student tokenizer（`REF_TOKENIZER_PATH` 默认等于 `MODEL_PATH`；`fsdp_workers.py:377-389` 读取 `ref.model.tokenizer_path`）。
- teacher 信息：只需要 teacher 在 student 采样序列上、采样 token 的 logprob，不需要 full-vocab logits，也不需要 teacher rollout。官方实现把 teacher 当作 verl 的 ref model 加载（`+actor_rollout_ref.ref.model.path=${TEACHER_MODEL_PATH}`，`ref.fsdp_config.param_offload=True`）。
- 任务奖励：不参与训练。`math_verify_reward.py` 只用于验证和日志。
- 规模：论文用 1.7B/8B student 与 14B/30B-A3B teacher。README 提醒用 dense teacher（Qwen3-8B/14B）时 TOP-D 和 OPD 都可能出现回答长度爆炸。

## 论文的实验设置

以下来自论文 §5.1 与 Table 5（v2）。

| 项 | 论文设置 |
|---|---|
| 训练数据 | DAPO-Math-17k |
| student / teacher | Qwen3-1.7B-Base、Qwen3-8B-Base；Qwen3-14B、Qwen3-30B-A3B-Instruct-2507 |
| baseline | GRPO、DAPO、OPD |
| G（每 prompt 采样数） | 8 |
| batch | 全局 512 prompts（4096 条样本）；TOP-D mini-batch 32 prompts（16 个）；OPD mini-batch 512（1 个） |
| off-policy epoch | 1 |
| 优化器 | AdamW，lr 1e-6；warmup GRPO/DAPO 为 10，OPD/TOP-D 为「-」 |
| 长度 | prompt 2048；rollout 与验证最大回答 16384 |
| 采样 | rollout temperature 1.0、top-p 1.0；验证 temperature 1.0、top-p 0.7 |
| clip | ε_low = ε_high = 0.2（DAPO 的 ε_high 为 0.28） |
| α | 0.1 / 0.2 |
| 硬件 | 4 节点 × 8 H200（32 卡），论文称 8 卡可复现 |
| 训练长度 | 约 200–400 次更新 |
| 评测 | AIME24/25/26 用 avg@32；AMC23、MATH-500、Olympiad 用 avg@8 |

Table 2（8B student，30B-A3B teacher），括号内为相对最强 baseline 的提升：

| 方法 | AIME24 | AIME25 | AIME26 | AMC23 | MATH-500 | Olympiad |
|---|---|---|---|---|---|---|
| Base | 9.38 | 8.02 | 6.15 | 53.13 | 75.23 | 40.08 |
| GRPO | 30.10 | 22.08 | 21.67 | 56.33 | 76.83 | 44.92 |
| DAPO | 32.92 | 27.81 | 32.29 | 65.39 | 81.65 | 44.23 |
| OPD | 24.58 | 23.33 | 25.42 | 76.88 | 87.98 | 59.29 |
| TOP-D | 50.42 (+25.84) | 34.06 (+10.73) | 44.06 (+18.64) | 88.13 (+11.25) | 91.23 (+3.25) | 64.67 (+5.38) |

Table 3（1.7B student，AIME24 / AIME25 / AIME26）：

| 方法 | 30B-A3B teacher | 14B teacher |
|---|---|---|
| Base | 3.44 / 1.98 / 1.88 | 同左 |
| GRPO | 10.52 / 7.19 / 4.69 | 同左 |
| DAPO | 12.29 / 11.56 / 8.54 | 同左 |
| OPD | 8.96 / 7.50 / 5.94 | 7.81 / 8.33 / 5.63 |
| TOP-D | 20.31 / 17.71 / 13.75 | 12.81 / 14.69 / 9.69 |

Table 4（1.7B student，30B-A3B teacher，AMC23 / MATH-500 / Olympiad）：Base 27.27 / 56.80 / 24.00；GRPO 35.55 / 54.73 / 18.47；DAPO 37.50 / 61.88 / 26.63；OPD 45.70 / 71.13 / 35.83；TOP-D 54.38 / 77.33 / 43.06。

消融（Fig 5，1.7B）只给曲线、没有数值表：α=1.0（即去掉组件 A）不稳定；「w/o off-policy」（mini-batch = batch，去掉组件 D 的复用）样本效率低；α ∈ {0.1, 0.2, 0.3} 表现稳健。调研时读到的消融里没有看到对组件 B、C 的单独消融，复现时需要自己补。

## 与本仓实现的差异

本仓尚未实现。下面列官方代码与论文的差异，以及接入 EasyOPD/verl 的要点。

官方代码 vs 论文：

- loss 聚合：论文 Eq 7 是全局 token-mean（除以 Σ|y^i|）；官方启动脚本用 `actor.loss_agg_mode=seq-mean-token-mean`（`run_..._common.sh:343`），两者在长度差异大时权重不同。
- batch 与评测：官方单机默认 `TRAIN_BATCH_SIZE=128`、`PPO_MINI_BATCH_SIZE=32`（4 个 mini-batch），验证 `VAL_N=16`；论文是 512 / 32（16 个 mini-batch）与 avg@32。OPD 启动脚本是 128 / 128。README 自己说明默认值小于论文，且端到端 GPU 复现未经验证。
- KL：脚本打开 `use_kl_in_reward` 与 `use_kl_loss`，但系数都是 0.0，实际不起作用。
- `naive_topd`（只有组件 A）只在代码里出现，论文没有。
- 其他脚本设置：clip 0.2/0.2，`clip_ratio_c=3.0`，entropy_coeff 0，`total_epochs=10`，`test_freq=10`，vllm rollout，teacher 的 FSDP wrap 类默认 `Qwen3MoeDecoderLayer`。
- trainer 挂接：`ray_trainer.py:261-296` 把 old_log_probs、ref_log_prob 和 index=uid 传给新 estimator；`ray_trainer.py:1503-1519` 对 opd/topd/naive_topd 强制重算 old log prob（`force_logprob_compute`）。官方带了 `recipe/verification/check_core.py` 做 CPU 级数值检查。

接入 EasyOPD 的成本：低。组件 A–C 是对 `old_log_probs`、`ref_log_prob`（teacher 采样 token logprob）和 `response_mask` 的纯张量运算，约 150 行，可以作为 `easyopd/methods/topd/` 里注册的 advantage estimator；组件 D 只是配置（`ppo_mini_batch_size` 小于 `train_batch_size`、clip 0.2）。需要确认的挂接点只有两处：estimator 能拿到 uid 分组和 teacher logprob，以及 old log prob 在 mini-batch 复用时是 rollout 时刻的值。动手前先对照本仓 `gkd` 等同 tokenizer 方法已有的 teacher logprob 通路，避免另起一套。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| 1.7B-Base ← 30B-A3B，AIME24/25/26 | 20.31 / 17.71 / 13.75（Table 3） | 未跑 | [repro/](../repro/) |
| 8B-Base ← 30B-A3B，AIME24/25/26 | 50.42 / 34.06 / 44.06（Table 2） | 未跑 | [repro/](../repro/) |

## 待解决

- 确认会议：在 OpenReview / neurips.cc 正式列表出来后核对。
- 决定用论文的 512/32/avg@32 还是官方单机默认 128/32/avg@16；两者不可混用来对照论文数字。
- loss 聚合选 token-mean（论文）还是 seq-mean-token-mean（代码），并作为一项消融。
- 补组件 B、C 的单独消融（论文没给），以及 `naive_topd`。
- 红旗：8B 的 OPD baseline 偏弱（AIME24 为 24.58，低于 GRPO 的 30.10），可能是 OPD 只用 1 个 mini-batch、更新次数少导致，对照不完全公平；GRPO/DAPO 与蒸馏方法训练长度不同；α 是手选的；消融只有曲线；dense teacher 下有长度爆炸。
- 复现前先跑 `check_core.py` 级别的 CPU 数值对拍，再上单卡小规模。

## 来源

- https://arxiv.org/abs/2607.04751
- https://arxiv.org/html/2607.04751v1
- https://arxiv.org/html/2607.04751v2
- https://arxiv.org/pdf/2607.04751v1
- https://arxiv.org/pdf/2607.04751v2
- https://github.com/xie-lab-ml/TOP-D（本地克隆 commit `9a2716c`，读了 README、`core_algos.py`、`ray_trainer.py`、`fsdp_workers.py`、`recipe/opd/*.sh`）
- OpenReview / neurips.cc：只做了搜索，未找到条目。
