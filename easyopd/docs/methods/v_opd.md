# vOPD：以 −KL 为 value baseline 的 OPD（`v_opd`）

> 文件名：`methods/v_opd.md`。一个方法一篇，常青：理解或实现变了就改，不按日期追加。
> 读之前先看 [INDEX](../INDEX.md)；时序细节放 `repro/` 和 `experiments/`，这里只放链接。

- 注册名说明：本仓的 `vopd` 已经被 `easyopd/methods/vision_opd/`（视觉 OPD）占用，所以本方法计划注册为 `v_opd`，文档名同步用 `v_opd.md`。
- 论文：KL for a KL: On-Policy Distillation with Control Variate Baseline，arXiv [2605.07865](https://arxiv.org/abs/2605.07865)（v1 2026-05-08），作者 Minjae Oh, Sangjun Song, Gyubin Choi, Yunho Choi, Yohan Jo（SNU GSDS）。
- 会议：未核实（仅作者 README 声明）。官方仓库 README 写 "Accepted to NeurIPS 2026."，但 arXiv comments 字段为空，README 里的 bibtex 仍是 `@misc` arXiv 条目，搜索也没有找到 NeurIPS 或 OpenReview 的官方条目。
- 官方实现：<https://github.com/holi-lab/vOPD>，commit `d4146d5015216c6f8e89a7243c24912b634f531a`（2026-10-08，"docs: add NeurIPS acceptance, vOPD results and citation"）。基于 TRL 1.2.0 的 `trl.experimental.minillm.MiniLLMTrainer`（GRPO 子类），不是 verl。本地浅克隆在 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/vOPD`。
- 本仓实现：未实现（计划注册名 `v_opd`）
- 状态：调研中
- 结论（一句话）：给 sampled-token OPD 的逐 token reward 减去一个 detached 的 baseline，即当前位置 student 对 teacher 的 reverse KL（可只在 student top-k 上算）；论文报告 Qwen3-1.7B 上 4 个数学评测集的 avg 平均从 OPD 的 29.8 提到 33.0–33.1，与 full-vocab OPD（33.0）持平，4B 上 top-k 版每步比 full-vocab OPD 快 57.7%（论文 Table 2、Fig. 3）。

## 方法要点

vanilla OPD 的逐 token reward 是 r_t = log π_T(y_t) − log π_θ(y_t)，它是 reverse KL 的单样本 MC 估计，无偏但方差大。论文指出，这个 MDP 中 state c_t 的 value 有闭式解（论文 Eq. 9）：

V(c_t) = −KL(π_θ(·|c_t) ‖ π_T(·|c_t))

把它当作 control variate baseline，得到 advantage（论文 Eq. 10）：

a_t = r_t + KL(π_θ ‖ π_T)(c_t)

baseline 是 detached 的，只依赖 state，不依赖采样到的 token，所以梯度估计仍然无偏；梯度只通过采样 token 的 log π_θ(y_t) 回传。论文 Eq. 11 给出方差降低量约为 KL² · E‖∇log π‖²：KL 越大的位置，降方差越多。

full-vocab 版本（vOPD_fullV）在整个词表上算 KL。top-k 版本（vOPD_topk，论文 Eq. 14–15）取 student 概率最高的 k 个 token 作为支撑集，分别把 student 和 teacher 在这 k 个 token 上重新归一化，再算 KL 作为 baseline。默认 k = 20。因为 baseline 不影响无偏性，top-k 近似带来的误差只影响降方差的效果，不会让梯度有偏。论文 Table 1 对比了各变体的计算量。

与其他方法的区别：OPD_fullV（论文的 full-vocab OPD 基线）直接对全词表 reverse KL 求导，需要全词表 teacher 分布参与反传；vOPD 仍然是 policy gradient，只是 advantage 多了一个 baseline 项，teacher 信息只用在 detached 的标量上。

### 官方代码中的对应位置（commit `d4146d5`）

| 作用 | 位置 |
|---|---|
| `CustomMiniLLMTrainer(MiniLLMTrainer)`，新增 `use_baseline`、`kl_top_k`、`opd_top_k`、`reward_clip_lambda` | `src/opd_trainer.py:15-43` |
| `_topk_log_softmax`：取 indices 处 logits，减去全词表 logsumexp（得到全词表归一化 log-prob） | `src/opd_trainer.py:46-48` |
| teacher 对整条序列做前向，得到全词表 logits | `src/opd_trainer.py:117` |
| 采样 token 上的 student / teacher log-prob | `src/opd_trainer.py:127-143` |
| top-k 支撑集：`kl_indices = torch.topk(student_logits, k)`（按 student 选） | `src/opd_trainer.py:151` |
| 在 top-k 上分别重新归一化 student 和 teacher | `src/opd_trainer.py:164-171` |
| `baseline = Σ p̄_s (log p̄_s − log p̄_t)`；`centered_gap = s_lab − t_lab − baseline`；`reverse_kl_advantage = −centered_gap.detach()` | `src/opd_trainer.py:178-183` |
| 把 advantage 加到 GRPO 的 advantage 上，再调 TRL 的 GRPO `_compute_loss` | `src/opd_trainer.py:211`、`:214` |
| `reward_clip_lambda`：把 reward 用 clamp_min 截断在 log λ / (1 − λ)（论文没有这一项，默认关） | `src/opd_trainer.py:50-70` |
| top-k OPD 基线（`opd_top_k`，直接对 top-k 上的 reverse KL 求导） | `src/opd_trainer.py:83-105` |
| 方法名到 trainer 开关的映射：`tml` 为 rkl_advantage=True、ssd=False；`gkd_rkl` 为 ssd=True | `src/util.py:632-640` |
| 数据集：dapo14k 对应 HF 数据集 `guanning-ai/dapo14k`；prompt 构造时 `enable_thinking=False` | `src/util.py:104` |

TRL 1.2.0 `MiniLLMTrainer` 的相关行为（读的是 v1.2.0 的 `trl/experimental/minillm/minillm_trainer.py` 与 `minillm_config.py`）：

- 默认 reward 函数 `dummy_reward_func` 对所有样本返回 1.0（`minillm_trainer.py:43`、`:182`）。`num_generations=1` 时 config 把 `scale_rewards` 强制设为 "none"（`minillm_config.py:92-93`）。GRPO 的组内中心化在只有 1 条样本时会把这个常数 reward 的 advantage 变成 0，所以 vOPD 实际不使用 task reward。这一点是推断，没有逐行读 GRPOTrainer 的中心化代码。
- `_compute_advantage` 在 gamma = 0 时直接返回逐 token reward，不做折扣累加（`minillm_trainer.py:292-345`）。
- MiniLLMConfig 默认值：`rkl_advantage=True`、`single_step_decomposition=True`、`kd_temperature=1.0`、`gamma=0.0`、`length_normalization=True`。

官方配置（`configs/training/`）：

| 文件 | 内容 |
|---|---|
| `default.yaml` | lr 1e-5，cosine，1 epoch，num_generations 1，per-device batch 1 × grad accum 64，prompt 1024，completion 2048，温度 1.0，top_p 1.0，LoRA r64 / α128 / dropout 0.05（全部 proj），vLLM colocate 0.6，bf16，flash_attention_2 |
| `vopd.yaml` | `use_baseline=true`，`kl_top_k=0`（即 full-vocab baseline） |
| `vopd-topk.yaml` | `kl_top_k=100`；论文主实验用 k = 20，README 说要手动 override |
| `sample-opd.yaml` | vanilla OPD |
| `full-opd.yaml` | `distill_mode=gkd_rkl`（full-vocab OPD）；加 `opd_top_k=20` 即为 top-k OPD |

依赖版本：torch 2.8.0、trl 1.2.0、transformers 4.57.1、vllm 0.11.0、peft 0.17.1。

## 适用条件

vOPD 只适用于同 tokenizer：baseline 需要 teacher 在 student 的 top-k token id 上的 log-prob。teacher 必须是白盒。注意 teacher 只返回自己的 top-k 是不够的，必须按 student 给出的 indices 取值（full-vocab 版本则需要 teacher 全词表分布）。不使用 task reward（见上一节）。论文的 teacher 与 student 规模接近，主设置是同规模的 Instruct → Base（Qwen3-1.7B → Qwen3-1.7B-Base）。

## 论文的实验设置

来源是论文 §4.1 与附录 B Table 4。

| 项 | 论文 |
|---|---|
| teacher → student | 主设置 Qwen3-1.7B → Qwen3-1.7B-Base；另有 Qwen3-4B → Qwen3-4B-Base、Qwen3-1.7B → Qwen3-0.6B-Base、Olmo-3-7B-Think → Olmo-3-7B-Base |
| 训练数据 | DAPO-Math-17K 的英文子集，共 14K 条，1 epoch |
| 长度 | prompt 1024，response 2048 |
| 优化 | batch 64，lr 1e-5（Qwen）/ 2e-5（Olmo），LoRA r64 / α128，bf16 |
| 硬件 | 单张 H200 或 A100，每次训练 2–6 小时 |
| top-k | k = 20 |
| 评测 | 温度 0.6，top-p 0.9，response 4096；MATH500、Minerva 报 avg@8 / pass@8，AMC23、AIME24/25 报 avg@32 / pass@32 |
| 科学推理 | 训练用 SciKnowEval 化学子集（train/eval/test 75/5/20），评测 SciKnowEval test 与 GPQA-Diamond |

表中 "Avg." 列只平均 4 个评测集的 avg 指标（README 说明，按 1.7B OPD 一行验算 (58.7 + 22.2 + 33.4 + 4.8) / 4 = 29.8，一致）。AIME 一列是 AIME24 与 AIME25 合并（共 60 题）。

论文 Table 2，每格为 avg / pass：

| 模型 | 方法 | MATH500 | Minerva | AMC23 | AIME24/25 | Avg. |
|---|---|---|---|---|---|---|
| 1.7B | Student | 42.3 / 79.8 | 13.8 / 37.1 | 23.2 / 85.0 | 3.0 / 20.0 | 20.6 |
| 1.7B | OPD | 58.7 / 82.6 | 22.2 / 40.8 | 33.4 / 77.5 | 4.8 / 28.3 | 29.8 |
| 1.7B | OPD_topk | 58.0 / 84.0 | 23.4 / 44.5 | 35.5 / 85.0 | 4.0 / 30.0 | 30.2 |
| 1.7B | OPD_fullV | 64.6 / 85.4 | 25.0 / 44.5 | 36.5 / 87.5 | 5.8 / 25.0 | 33.0 |
| 1.7B | vOPD_topk | 64.9 / 84.8 | 25.2 / 44.5 | 36.1 / 82.5 | 5.6 / 26.7 | 33.0 |
| 1.7B | vOPD_fullV | 64.0 / 84.6 | 26.2 / 48.9 | 36.1 / 80.0 | 6.2 / 25.0 | 33.1 |
| 4B | OPD | 75.2 / 90.8 | 35.3 / 54.0 | 50.5 / 90.0 | 10.4 / 38.3 | 42.9 |
| 4B | OPD_fullV | 78.6 / 91.8 | 36.3 / 53.3 | 51.0 / 87.5 | 14.3 / 40.0 | 45.1 |
| 4B | vOPD_topk | 79.3 / 93.0 | 37.6 / 54.4 | 51.2 / 87.5 | 13.2 / 41.7 | 45.3 |
| 4B | vOPD_fullV | 78.9 / 92.4 | 37.2 / 54.0 | 52.0 / 92.5 | 13.5 / 40.0 | 45.4 |

Table 2 其余的 Avg.：4B Student 28.4，4B OPD_topk 42.8；Olmo 上 OPD 29.9、OPD_fullV 31.8、vOPD_topk 33.1、vOPD_fullV 32.9。

其他结果：

- Table 5（Qwen3-1.7B → 0.6B-Base 的 Avg.）：OPD 17.5，OPD_topk 19.9，OPD_fullV 20.6，vOPD_topk 20.0，vOPD_fullV 21.1。teacher 自身的 Avg.：Qwen3-1.7B 40.2，Qwen3-4B 53.4，Olmo 58.4。k 消融（同在 Table 5，Qwen3-1.7B → 1.7B-Base 的 Avg.）：top5 33.4，top20 33.0，top50 32.6，top100 32.3。
- Table 3（SciKnowEval / GPQA-D）：Student 26.1 / 20.4，OPD 29.3 / 24.7，OPD_topk 29.7 / 24.1，OPD_fullV 35.1 / 28.7，vOPD_topk 33.2 / 28.6，vOPD_fullV 34.7 / 28.4。科学推理上 vOPD 没有超过 OPD_fullV。
- 速度（Fig. 3，README）：4B 上 vOPD_topk 每步 32.88s，OPD_fullV 每步 77.66s，少 57.7%。
- 梯度范数（Fig. 4）：vOPD 比 OPD 低 1–2 个数量级。

证据边界：vOPD 与 OPD_fullV 在 1.7B 和 4B 上的 Avg. 差距只有 0–0.3，在 AIME（60 题、avg@32）这类小集上的单格差异很可能在噪声内；本次阅读没有在论文里找到 seed 数或方差。论文更强的论点是「以 top-k 的成本达到 full-vocab 的效果」，而不是「超过 full-vocab」。

## 与本仓实现的差异

本仓还没有实现，下面列的是官方代码与论文的差异，以及在本仓 verl 中复现要做的事。

官方代码与论文的差异：

- top-k 预设不一致：论文主实验 k = 20，`configs/training/vopd-topk.yaml` 是 `kl_top_k=100`，README 要求手动 override。论文 Table 5 显示 k = 100 时 Avg. 为 32.3，比 k = 20 的 33.0 低。
- `reward_clip_lambda`（`src/opd_trainer.py:50-70`）是论文没有的选项，默认关闭，复现时不用。
- 官方 trainer 是 TRL 的 GRPO 子类，最终 loss 走 TRL 的 GRPO `_compute_loss`。advantage 按 GRPO 的方式外加了一个常数 reward 的组内 advantage（在 num_generations = 1 时为 0，见上文的推断）。

本仓与官方实现的结构性差异：

- 官方用 LoRA（r64 / α128）、单卡、lr 1e-5。本仓 verl 默认是 FSDP 全参训练。如果用全参训练，lr 1e-5 很可能过大，需要重新调；如果要与论文对齐，就在 verl 里开 LoRA（verl 的 actor 有 `lora_rank` 配置，本仓是否跑通过需要查 `experiments/`）。
- 官方 baseline 在 loss 计算时用当前 π_θ 的 logits 算（`src/opd_trainer.py:151-181`）。官方每批 rollout 只更新一次，所以 π_θ 等于 rollout 时的 π_old。本仓 verl 若一批 rollout 分多个 mini-batch 更新，baseline 用 π_old 算会与官方有细微差异；baseline 只依赖 state，换成 π_old 的 KL 仍是合法的 baseline，梯度依旧无偏，只是降方差的效果可能略有变化（这是推断）。

在本仓 verl 中复现 vOPD_topk 要做的事：

1. student 侧：在 old_log_prob 前向时输出 student 的 top-k indices 和对应的全词表归一化 log-prob。本仓 `verl/workers/actor/dp_actor.py:662-664` 的 `compute_log_prob` 已经能通过 `meta_info["opsa_topk_k"]` 返回 top-k（OPSA 加的挂接点），可以复用。
2. teacher 侧：teacher 要在 student 给出的 indices 上取 log-prob。`_forward_micro_batch` 已有 `opsa_gather_indices` 参数（`dp_actor.py:123`、`:450-460`、`:516-517`），但 `compute_log_prob` 目前只透传 `opsa_topk_k`，不透传 gather indices，`ray_trainer.py` 的 teacher 请求也只挂在 OPSA 的 `_maybe_build_opsa_batch`（`ray_trainer.py:1290`）里。这两处需要改 `verl/`，并按规范加 `[EasyOPD:V_OPD]` 标记、登记 touchpoint。
3. advantage：注册一个新的 adv_est，按官方 `src/opd_trainer.py:164-183` 的写法，在 top-k 上对 student 和 teacher 分别重新归一化，算 baseline = Σ p̄_s (log p̄_s − log p̄_t)，advantage = (log π_T(y_t) − log π_old(y_t)) + baseline。可以参考 `easyopd/methods/lightning_opd/advantage_estimator.py:19` 的 `on_policy_distillation` 估计器结构。
4. loss：直接用 verl 原生 PPO loss，不需要新的 LossHook。
5. 生效证据：记录 baseline 的均值、advantage 的方差和梯度范数，与 vanilla OPD 对比；论文 Fig. 4 报告梯度范数低 1–2 个数量级，可作为判据之一。

成本估计：vOPD_topk 为中，vOPD_fullV 为高。top-k 版的数学部分是几十行的 adv_est，主要成本在 `verl/` 的两处挂接（student top-k 传给 teacher，teacher 按 indices gather）。full-V 版需要同一位置同时拿到 student 和 teacher 的全词表分布；在 verl 里 teacher 和 student 在不同 worker 上，按 Qwen3 词表约 15 万计，64 条 × 2048 token 的全词表 fp32 log-prob 约 80GB，不能在 worker 之间传，只能把 teacher 放进 actor 进程里前向（类似 SDPO 的 `teacher_module` 用法），改动大。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| Qwen3-1.7B → Qwen3-1.7B-Base，DAPO 14K，vOPD_topk（k = 20）Avg.（Table 2） | 33.0 | 未跑 | — |
| 同上，vanilla OPD Avg.（Table 2） | 29.8 | 未跑 | — |

## 待解决

- NeurIPS 2026 录用尚未从官方渠道核实。
- 确认 TRL GRPOTrainer 在 num_generations = 1 时的 advantage 确实为 0（逐行读 `_generate_and_score_completions` 的中心化代码）。
- 确认本仓 verl 是否跑通过 LoRA；不跑 LoRA 时 lr 的取值需要先做小规模扫描。
- 论文的 "DAPO-Math-17K 英文子集 14K" 与代码的 `guanning-ai/dapo14k` 是否同一份数据，以及本仓是否已有（见 `reference/environment.md`）。
- 本次没有在论文中找到 seed 数，0–0.3 的 Avg. 差距是否显著无法判断。

## 来源

以下是本次调研实际打开过的页面：

- arXiv 元数据页：<https://export.arxiv.org/abs/2605.07865>
- 论文全文 HTML：<https://arxiv.org/html/2605.07865v1>
- 官方仓库：<https://github.com/holi-lab/vOPD>（浅克隆，读了 `README.md`、`src/opd_trainer.py`、`src/util.py`、`configs/training/*.yaml`）
- TRL v1.2.0 源码：<https://raw.githubusercontent.com/huggingface/trl/v1.2.0/trl/experimental/minillm/minillm_trainer.py> 与同目录的 `minillm_config.py`

搜索 NeurIPS / OpenReview 条目没有找到结果，所以没有可列的会议页面。
