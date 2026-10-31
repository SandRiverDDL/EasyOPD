# Revisiting OPD：teacher top-K 局部支撑匹配（`revisit_opd`）

> 文件名：`methods/revisit_opd.md`。一个方法一篇，常青：理解或实现变了就改，不按日期追加。
> 读之前先看 [INDEX](../INDEX.md)；时序细节放 `repro/` 和 `experiments/`，这里只放链接。

- 论文：Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes，arXiv [2603.25562](https://arxiv.org/abs/2603.25562)（v1 2026-03-26，v2 2026-04-27），作者 Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, Dongbin Zhao。会议：COLM 2026，已核实（OpenReview note `uZbZb058bA` 的 venue 字段为 "COLM 2026"；GitHub 仓库标题也标 "[COLM 2026]"；arXiv comments 字段为空）。
- 易混淆的论文：Rethinking OPD（arXiv 2604.13016，thunlp/OPD）及其 part II（2609.04172）是另一组作者的工作，不是本文。
- 官方实现：<https://github.com/hhh675597/revisiting_opd>，commit `f32f284f25bae5b16d2d44ee336b52851dccc736`（2026-05-20），基于 verl-agent（verl fork），入口 `verl.trainer.main_ppo_multitask`。本地浅克隆在 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/revisiting_opd`。
- 本仓实现：未实现（计划注册名 `revisit_opd`）
- 状态：调研中
- 结论（一句话）：把 sampled-token OPD 的单 token log-ratio 奖励换成「在 teacher top-K 上重归一化后的截断 reverse KL」，并配合 top-p rollout 与特殊 token 屏蔽，在数学单任务上把 OPD 的 5 集平均 pass@1 从 36.4 提到 41.7（论文 Table 1）。

## 方法要点

论文先分析 sampled-token OPD 的三个失败模式：单 token 信号方差大且不稳定、student 生成的前缀让 teacher 给出不可靠的低概率、以及 tokenizer / 特殊 token 约定不一致造成的假负信号（论文 §2.2，例子是 `<think>` 被切成 `<`,`think`,`>` 时 teacher 给 `<` 低概率）。论文还比较了 token 级和序列级估计：token 级有偏但方差上界是 O(T²)，序列级无偏但方差上界是 O(T⁴)，γ 折扣估计量在两者之间插值（论文 §2.1 与附录 D）。

基线 sampled-token OPD 的逐 token 奖励是 r_t = −(log π_θ(o_t) − log q(o_t))（论文 Eq. 2、5），代码中对应 `compute_opd_advantage`（`verl/trainer/ppo/core_algos.py:757`）：advantage = −(log π − log q)·mask，可选 γ 折扣、可选加 `reward_weight × outcome`、可选裁到 ±5，不做归一化。

本文方法（论文称 local support matching，LSM）的目标是论文 Eq. 8：

L = E[ (1/Σ_i|o_i|) Σ_i Σ_t Σ_{v∈TopK_q(t)} π̂(v) · (log π̂(v) − log q̂(v)) ]

其中 TopK_q 是 teacher 在该位置概率最高的 K 个 token，π̂ 与 q̂ 是 student 与 teacher 在这 K 个 token 上各自做 softmax 重归一化后的分布（论文 Eq. 6–7）。因此它是 student→teacher 方向的截断 reverse KL，作为可微 loss 直接反传，不再走 policy-gradient 的 advantage。消融显示重归一化是必须的，去掉会崩溃；K 太小会掉点；不加 top-p 的无约束采样会让训练变差（论文 §4.4 Ablations）。

两个配套的实用修补：rollout 时使用 top-p=0.9（只用于 LSM，基线不用，论文 §3.2 与 Table A1），以及屏蔽出问题的特殊 token（论文 §3.2：这对 sampled-token OPD 帮助大，对 LSM 影响小）。

代码中的对应位置（commit `f32f284`）：

| 作用 | 位置 |
|---|---|
| top-K 截断 KL（含 `norm_to_one`：对 `ref_logits_k`、`actor_logits_k` 做 log_softmax 后算 Σ p̂(log p̂ − log q̂)；可选 tail sampling 把 top-K 外的采样 token 追加进支撑集） | `verl/trainer/ppo/core_algos.py:867-1050` `compute_memory_efficient_kl` |
| 旧版 top-K KL 与 not-in-topK mask | `core_algos.py:721` `compute_topk_kl_divergence`、`core_algos.py:845` `compute_not_in_topk_mask` |
| k1 / k3 KL（k3 截断在 ±10） | `core_algos.py:615` `kl_penalty` |
| teacher 前向：返回 `ref_log_prob`、`ref_logits_k`、`ref_logsumexp`、`ref_topk_indices`，用 rollout 温度 | `verl/workers/fsdp_workers.py:821` `compute_ref_log_prob_with_logits` |
| actor 在 `ref_topk_indices` 处 gather 自身 logits，`kl_loss = agg_loss(kld, kl_mask, loss_agg_mode) × kl_loss_coef`，pg 项的 advantage 为 0 | `verl/workers/actor/dp_actor.py:1163-1392` `update_policy`；`dp_actor.py:408` `_forward_micro_batch_with_logits` |
| 特殊 token 屏蔽：默认 `["<","think","<|im_end|>"]`，每条序列只屏蔽每种 token 的第一次出现 | `dp_actor.py:85-111` `_init_opd_mask_tokens`、`dp_actor.py:113` `_compute_opd_kl_mask` |
| PLACE_HOLDER advantage（全 0）与 OPD advantage 分支 | `verl/trainer/ppo/ray_trainer_multitask.py:129`、`:395`、`:~408` |
| 熵 top-ratio 掩码（每条样本按熵保留前 ratio 的 token，再乘进 response_mask） | `ray_trainer_multitask.py:~80-98` `_compute_entropy_top_mask`，应用于 `:1391-1394` |

## 适用条件

本文按同 tokenizer 设计：teacher top-K 的 token id 直接用来 gather student 的 logits，跨 tokenizer 不能直接用。teacher 需要白盒 logit，具体是每个 response 位置的 top-K（K=32）logits 及其索引；官方实现还额外取了 `ref_logsumexp`，但 `norm_to_one` 路径只用 top-K logits 做重归一化。它不需要完整词表分布，也不需要 task reward：官方 launcher 中 outcome reward 不进 loss（`adv_estimator=placeholder`，advantage 全 0）。数学实验的 teacher 和 student 都是 7B（论文 Table A1）。

## 论文的实验设置

来源是论文 §4.1 与附录 F 的 Table A1，括号内是官方 launcher `examples/opd/opd_math_qwen2.5-7b_it.sh` 的取值。

| 项 | 论文 | 代码 launcher |
|---|---|---|
| 硬件 | 8×H100 | `n_gpus_per_node=8, nnodes=1` |
| 数学 student / teacher | Qwen2.5-7B-Instruct / OpenThinker3-7B | 同 |
| 训练数据 | DAPO-Math-17K 英文部分 | 同 |
| 长度 | max prompt 2048；max response 16384（数学），512（ALFWorld / WebShop） | max response 16384 |
| rollout | group size 8，T=1.0，top-p 0.9（只用于 LSM） | `env.rollout.n=8`，`rollout.top_p=0.9` |
| teacher top-K | 32 | `kl_topk_tokens=32` |
| 优化 | AdamW，lr 2e-6，warmup 0 | lr 2e-6 |
| batch | batch size 128，mini batch 64 | `train_batch_size=16` 个 prompt × 8 = 128 条轨迹，`ppo_mini_batch_size=64`，每步 2 次更新（推断，见差异一节） |
| 步数 | 数学 / 多任务 400 步（多任务 = 200 数学 + 200 ALFWorld），WebShop 60 步 | `total_epochs=1` |
| 评测采样 | top-p 0.9，T=1.0 | val T=1.0，top_p=0.9 |

多任务实验的 ALFWorld teacher 是 GiGPO-Qwen2.5-7B-It-ALFWorld，WebShop 实验用 1.5B student（Table A1）。数学评测是 Math500、AIME24、AIME25、Minerva、OlympiadBench 上的 pass@1，部分表格用 avg@32。论文主实验没有报告 seed 数或多 seed 方差（多 seed 只出现在附录 E 的 toy 实验），AIME 类小集的单次差异需按噪声看待。

主要数字（5 集平均 pass@1，除非另注）：

| 设置 | 数字 | 来源 |
|---|---|---|
| student / teacher | 28.2 / 56.0 | Table 1 |
| sampled-token OPD / +mask | 36.4 / 40.7 | Table 1 |
| Ours w/o mask（Math500 80.4，AIME24 23.3，AIME25 26.7，Minerva 34.2，Olympiad 43.9） | 41.7 | Table 1 |
| Ours w/ mask | 41.5 | Table 1 |
| 多任务 ALFWorld 成功率 / 数学平均：OPD | 90.6 / 34.8 | Table 2 |
| 多任务：OPD+mask | 93.8 / 36.6 | Table 2 |
| 多任务：Ours w/o mask | 95.3 / 41.7 | Table 2 |
| 多任务：Ours w/ mask | 97.7 / 38.6 | Table 2 |
| AIME24 avg@32：student 10.0，teacher 63.3，OPD 20.4，OPD+top-p 21.6，teacher top-K 17.7，top-K+top-p 23.6 | — | Table 3 |
| top-K 变体（单任务 avg@32 / pass@1 平均）：teacher top-K 23.6/41.7；student top-K w/ sampled 22.3/41.9；teacher top-K w/ sampled 22.4/42.9 | — | Table 4 |
| top-K 变体（多任务数学平均）：41.7 / 28.4 / 26.9 | — | Table 4 |
| WebShop 成功率：OPD 50.0 → Ours 57.8 | — | Table A2 |

§4.3 正文的 "+19.8%" 指多任务数学平均从 34.8 到 41.7（Table 2），是相对提升。Table 1 中 mask 对本方法几乎没有影响（41.7 vs 41.5），但对 OPD 有 +4.3 的提升，这与 §3.2 的说法一致。

## 与本仓实现的差异

本仓尚未实现，下面先列官方实现内部「论文 vs 代码」的差异，再列在 EasyOPD 中实现时的差异与成本。

论文与官方代码的差异和需要确认的点：

- 熵掩码只在代码里有。仓库 README 的推荐命令带 `entropy_top_ratio=0.2`，代码在 `ray_trainer_multitask.py:~80-98` 实现；但论文正文与 Table A1 都没有提到按熵筛 token，而 `opd_math_qwen2.5-7b_it.sh` 也没有设置这个参数（即默认不开）。复现论文数字应按 launcher 走、不开熵掩码，并把它作为待确认项。
- KL 方向的描述不一致。README 把 `full_reverse` 写成 KL(teacher‖student)，但 `compute_memory_efficient_kl` 实际算的是 Σ π̂(log π̂ − log q̂)，即 KL(student‖teacher)，与论文 Eq. 8 一致。以代码和论文为准。
- batch 的计数口径。Table A1 写 batch 128、mini batch 64；launcher 是 `train_batch_size=16` 个 prompt，经 `env.rollout.n=8` 在 `agent_system/multi_turn_rollout/rollout_loop.py:533` 复制成 128 条轨迹；actor 的 `rollout.n` 保持默认 1（`ppo_trainer.yaml:138`），所以 `fsdp_workers.py:150` 的 `ppo_mini_batch_size *= rollout.n` 不放大，mini batch 就是 64 条轨迹，每步 2 次 optimizer 更新。这一点是读代码推出的，未实际运行验证。
- 归一化方式。论文 Eq. 8 的 1/Σ|o_i| 对应 verl 的 `loss_agg_mode=token-mean`（`ppo_trainer.yaml:57`）；`kl_loss_coef=1`，`entropy_coeff=0`，`clip_ratio` 0.2 但 advantage 为 0，所以 PPO clip 实际不起作用。
- 特殊 token 屏蔽的范围。论文说屏蔽「有问题的特殊 token」，代码默认列表是 `["<","think","<|im_end|>"]`，且每条序列只屏蔽每种 token 的第一次出现（`dp_actor.py:113`）；主 launcher 里 `opd_mask_special_tokens=False`，即 "Ours w/o mask" 配置。
- 训推不一致未修正。附录 A 承认 rollout（vLLM + top-p）与训练前向之间的 mismatch 没有做重要性修正；launcher 里 `use_kl_iw=False`。
- teacher logits 用 rollout 温度计算（`fsdp_workers.py:821`），数学实验温度为 1.0，故不影响。
- 基线 launcher `opd_original_math_qwen2.5_7b_it.sh` 与主 launcher 的差异是：`adv_estimator=opd`、`kl_loss_type=k1`、`top_p=1.0`、`use_kl_loss=False`、`use_kl_in_reward=True`、seed 21。

在 EasyOPD（同 tokenizer teacher）中实现的估计：成本低到中。仓库里 OPSA 已经有一条 teacher top-K 通路：trainer 把 ref worker 输出的 `ref_topk_log_probs` / `ref_topk_indices` 作为 `opsa_teacher_topk_*` 传给 actor（`verl/trainer/ppo/ray_trainer.py:1472-1488`），actor 按这些 index gather student log-prob，并在 K 上做 logsumexp 重归一化后算 forward / reverse KL（`verl/workers/actor/dp_actor.py:1195-1225`）。`revisit_opd` 只需要复用这条通路，取重归一化后的 reverse KL、K=32、token-mean、advantage 置 0，并在 rollout 配置里打开 top-p 0.9。还需要确认 ref worker 输出 top-K 的逻辑是否被 OPSA 的开关门控，以及 OPSA 路径里的 `clamp(min=0)` 和温度缩放是否与官方实现等价（官方实现没有 clamp）。特殊 token 屏蔽属于可选项，按方法 hook 实现即可，不需要动 `verl/`。

## 复现 recipe 清单（数学单任务，"Ours w/o mask"）

以下每一项都来自 Table A1 或官方 launcher `examples/opd/opd_math_qwen2.5-7b_it.sh`；两者冲突时以 launcher 为准并在括号里注明。

1. 模型：student Qwen2.5-7B-Instruct，teacher OpenThinker3-7B（同 tokenizer）。
2. 数据：DAPO-Math-17K 的英文部分；max prompt 2048，max response 16384。
3. rollout：每个 prompt 采 8 条，温度 1.0，top-p 0.9（基线 OPD 用 top-p 1.0）。
4. teacher 信号：在每个 response 位置取 teacher top-32 的 logits 与索引，按 rollout 温度计算。
5. 损失：在 teacher top-32 上分别对 student 与 teacher 做 log_softmax 重归一化，loss = Σ π̂(log π̂ − log q̂)（`kl_loss_type=full_reverse`，`norm_to_one_for_kl=True`，`kl_topk_source=ref`），`kl_loss_coef=1`，`loss_agg_mode=token-mean`。
6. 关掉其他信号：`adv_estimator=placeholder`（advantage 全 0），`use_kl_in_reward=False`，`entropy_coeff=0`，`clip_log_ratio=False`，`kl_use_tail_sampling=False`，`use_kl_iw=False`，`opd_mask_special_tokens=False`；不设 `entropy_top_ratio`。
7. 优化：AdamW，lr 2e-6，warmup 0；每步 16 个 prompt × 8 = 128 条轨迹，mini batch 64 条（每步 2 次更新），PPO epoch 1。
8. 步数：400 步，8×H100。
9. 评测：Math500、AIME24、AIME25、Minerva、OlympiadBench，T=1.0、top-p 0.9 的 pass@1，AIME24 另报 avg@32；目标数字是 Table 1 的平均 41.7（OPD 基线 36.4）。
10. 可选变体："w/ mask" 打开 `opd_mask_special_tokens`（默认屏蔽 `<`、`think`、`<|im_end|>` 的首次出现），Table 1 目标 41.5。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| Qwen2.5-7B-It ← OpenThinker3-7B，DAPO-Math-17K，400 步，5 集平均 pass@1 | 41.7（Table 1，Ours w/o mask；OPD 36.4） | 未跑 | — |
| 同上，AIME24 avg@32 | 23.6（Table 3，top-K + top-p；OPD 20.4） | 未跑 | — |

## 待解决

- `entropy_top_ratio=0.2` 是否参与了论文数字：论文与主 launcher 都没有，README 有；需要问作者或在小规模上对照。
- 确认 EasyOPD 的 OPSA top-K 通路是否能在 `revisit_opd` 下复用（是否被 OPSA 开关门控、`clamp(min=0)` 是否需要去掉）。
- 论文主表没有 seed 数与方差，AIME24/25 只有 30 题，复现时至少报 avg@32，并说明差异是否在噪声内。
- 多任务（ALFWorld）设置依赖 verl-agent 的环境组件，本仓暂不考虑。

## 来源

- https://export.arxiv.org/abs/2603.25562
- https://arxiv.org/html/2603.25562v2
- https://api2.openreview.net/notes/search?term=Revisiting%20On-Policy%20Distillation%20Empirical%20Failure%20Modes&limit=3
- https://github.com/hhh675597/revisiting_opd
