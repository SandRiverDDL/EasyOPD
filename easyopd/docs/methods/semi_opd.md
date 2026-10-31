# Semi-OPD（`semi_opd`）

> 文件名：`methods/semi_opd.md`。一个方法一篇，常青：理解或实现变了就改，不按日期追加。
> 读之前先看 [INDEX](../INDEX.md)；时序细节放 `repro/` 和 `experiments/`，这里只放链接。

- 论文：When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better，arXiv [2610.11291](https://arxiv.org/abs/2610.11291)（2026-10，NVIDIA，Siyan Zhao、Yonggan Fu 等）
- 官方实现：2026-10-30 没找到。`NVIDIA-NeMo/Semi-OPD` 返回 404，WebSearch 只搜到论文和 [NVIDIA 研究页](https://research.nvidia.com/publication/2026-12_when-do-we-need-policy-distillation-distilling-offline-student-rollouts-often)。本仓实现按论文正文自行编写。相关项目 Lightning-OPD（slime/Megatron）已 clone 到 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/Lightning-OPD`（commit cd6b559），只作参考。
- 本仓实现：
  - 方法包 `easyopd/methods/semi_opd/`：`__init__.py` 复用 vanilla_opd 的损失，`replay.py` 负责回放。
  - verl 挂点：`[EasyOPD:semi_opd]`，位于 `ray_trainer.fit` 的 gen 段。
  - 离线采样脚本：`experiments/07_opd_repro/scripts/gen_offline_rollouts.py`。
  - 配置：`experiments/07_opd_repro/configs/methods/semi_opd.yaml` 和 `configs/exps/1030_semi/`。
- 状态：已实现，冒烟中。
- 结论（一句话）：待复现。论文的主张是：当 teacher 与 student 的 top-64 overlap 低时，在初始 student 的固定离线 rollout 上做 OPD，效果优于 OPD，而且快约 11 倍。

## 方法要点

训练前，先用初始 student π₀ 对每道训练 prompt 采 1 条回答。采样参数和长度上限与 OPD 相同。整个训练只用这批固定数据，不再在线 rollout。

目标函数与 OPD 完全相同，都是 sampled-token reverse KL，逐 token 的信号为 A_t = log π_T(y_t) − log π_θ(y_t)。其中 π_θ 是当前的 student，只有回答的来源由当前 π_θ 换成了固定的 π₀。由于回答固定，论文中 teacher logp 只需算一次，训练阶段既不生成也不跑 teacher。

论文对效果的解释是：好的蒸馏数据要同时满足两点，一是对 student on-policy，避免 exposure bias；二是 teacher 的监督质量高，这大体取决于数据对 teacher 是否 on-policy。在 overlap 低的组合上，OPD 的 student 会漂到 teacher 不擅长评分的区域。Semi-OPD 放弃了一部分 student on-policy 性，换来更有效的监督。

选择依据是 overlap：Semi-OPD 相对 OPD 的 AUC 增益与初始 overlap 的 Pearson 相关系数为 r = −0.73（17 对模型）。overlap 高时用 OPD，低时用 Semi-OPD。

与 Lightning OPD 的区别：Lightning OPD 从 base 模型出发，先在 teacher rollout 上做 SFT，再在 student 离线 rollout 上蒸馏；Semi-OPD 去掉 SFT，直接从 instruct student 开始。

## 适用条件

- 同 tokenizer，需要 teacher 给出 sampled token 的 logprob。
- 论文共测 17 对模型（3 个模型族，thinking 与 non-thinking 都有）。Semi-OPD 在 14 对上胜出，最高 +13.6%；其余 3 对差距在 0.1%–2.6% 之内。
- overlap 高的组合，例如 Qwen3-1.7B ← Qwen3-4B，overlap 70.0%，增益只有 +1.1，此时没必要用 Semi-OPD。

### 全部 17 对组合（论文 Table 3，2026-10-31 从 arXiv HTML 抄录）

增益 = Semi-OPD − OPD，Overlap = 初始 student 与 teacher 的 top-64 overlap（%）。NT = non-thinking（训练 4K），T = thinking（16K）。

| # | Student | Teacher | Mode | Overlap | Last-step | AUC |
|---|---|---|---|---|---|---|
| n | Qwen3-1.7B | Qwen3-30B-A3B-Thinking | T | 58.9 | +9.4 | +7.57 |
| a | Qwen3-1.7B | Qwen3-4B-Instruct-2507 | NT | 62.5 | +9.5 | +7.09 |
| p | Qwen3-4B | Qwen3-30B-A3B-Thinking | T | 63.8 | +6.2 | +6.22 |
| o | Qwen3-14B | Qwen3-235B-A22B-Thinking | T | 63.4 | +13.1 | +5.85 |
| b | Qwen3-1.7B | Qwen3-30B-A3B-Instruct-2507 | NT | 59.8 | +6.8 | +5.18 |
| f | Qwen3-4B | Qwen3-30B-A3B-Instruct-2507 | NT | 62.5 | +4.2 | +4.61 |
| c | Qwen3-1.7B | Qwen3-235B-A22B-Instruct-2507 | NT | 56.9 | +2.6 | +3.27 |
| g | Qwen3-1.7B (SFT) | Qwen3-4B-Instruct-2507 | NT | 62.9 | +2.4 | +2.05 |
| h | Qwen3-4B (SFT) | Qwen3-30B-A3B-Instruct-2507 | NT | 63.1 | +0.9 | +1.50 |
| d | Qwen3-1.7B | Qwen3-4B | NT | 70.0 | +1.1 | +1.35 |
| m | R1-Distill-7B | DeepCoder-14B | T | 73.9 | +1.5 | +0.97 |
| e | Qwen3-1.7B | Qwen3-8B | NT | 69.4 | +0.8 | +0.54 |
| j | R1-Distill-1.5B | Polaris-7B | T | 69.7 | +0.1 | +0.24 |
| q | Cascade-2-30B-A3B | Cascade-2-Math-Expert | T | 89.3 | −0.1 | +0.09 |
| l | R1-Distill-7B | Light-R1-14B | T | 72.0 | −1.7 | −0.30 |
| k | R1-Distill-7B | Polaris-7B | T | 86.8 | −2.6 | −0.68 |
| i | R1-Distill-1.5B | JustRL-1.5B | T | 79.7 | +1.4 | −1.36 |

本仓的推断（论文没有这样分析）：
- 增益 ≥ +3 的 7 对，student 都是未经额外训练的 Qwen3 原版小模型，teacher 都是 2507 代或 235B/30B 大模型。teacher 换成同代的 Qwen3-4B / 8B（d、e）后，overlap 升到约 70%，增益降到 ≤ +1.35。在 Qwen NT 组合里，「overlap 低」和「teacher 是 2507 代」两个因素同时出现，论文的 r = −0.73 分不开这两者。
- student 先在 teacher 数据上 SFT 过（g、h），overlap 只升了约 0.5 个点，但增益从 +7.09 掉到 +2.05、从 +4.61 掉到 +1.50。所以光看 overlap 解释不了这个变化，student 的起点本身就有影响。

### 和 Lightning OPD（arXiv 2604.13010）的组合差异

Lightning OPD 的结论是「离线 OPD ≈ 在线 OPD」，Semi-OPD 的结论是「离线明显更好」。两边的组合不一样：
- Lightning OPD 的 student 是 Base 模型（Qwen3-4B-Base ← Qwen3-8B、Qwen3-8B-Base ← Qwen3-32B、Qwen3-30B-A3B-Base ← Qwen3-30B-A3B-Thinking-2507），先用**同一个 teacher** 的数据 SFT 3000 步，再做 OPD（150 步，batch 256，4K，lr 2e-6，T=0.8）。论文 Table 4 说明 SFT teacher 和 OPD teacher 不一致时分数会下降（例如 4B 从 65.4 降到 62.4）。
- 所以 Lightning OPD 的设置，相当于 Semi-OPD 里先 SFT 的 g、h 那一类。在这一类上，Semi-OPD 论文自己测到的增益也只有 +1.5 到 +2.05，两篇的结论并不冲突。
- Semi-OPD 的 Table 1 在 a 这一对上也跑了 Lightning OPD（30.0），Semi-OPD 是 39.8。Semi-OPD 正文说两者的区别是 Lightning OPD 多了 SFT 这一步，这里的 Lightning OPD 具体怎么配置的，没有细查。

## 论文的实验设置（Table 2 和附录 A）

| 项 | 值 |
|---|---|
| 优化器 | AdamW，β=(0.9, 0.999)，weight decay 0.01，梯度裁剪 1.0 |
| 学习率 | 1e-6，constant，无 warmup |
| 精度 | bf16（大概率指计算精度，见 [pitfalls](../pitfalls.md) 的 bf16 主权重条目） |
| 每步数据量 | 128 题 × 1 条 |
| 训练长度上限 | prompt 2048；response 4096（non-thinking）/ 16384（thinking） |
| 采样 | temperature 1.0，top_p 1.0 |
| 数据 | DAPO-Math-17K 去重后 14,116 条英文 prompt，1 epoch 约 110 步；OPD 读 prompt 的顺序与 Semi-OPD 相同 |
| 评测 | AIME24/25/26 和 HMMT Feb26，avg@32，temperature 1.0，长度 32K；Table 1 取第 60/80/100 步三个 ckpt 的平均 |
| 框架 | NeMo-RL（DTensor），vLLM，H100 |

Table 1 中本仓选做的一对：Qwen3-1.7B（non-thinking）← Qwen3-4B-Instruct-2507，overlap 62.5%。

| 方法 | Avg@32 |
|---|---|
| Teacher | 47.5 |
| 未训练的 student | 9.0 |
| Standard OPD | 26.2 |
| FastOPD / Lightning OPD / Relay-OPD | 30.7 / 30.0 / 32.7 |
| Semi-OPD | 39.8（AIME24 50.3、AIME25 38.9、AIME26 41.8、HMMT 28.0） |

注意 Table 1 的 Standard OPD 26.2 引自文献 [13]，且 Table 1 的 OPD 用 16K 训练长度（论文附录原文："In Table 1, however, OPD is trained with a 16K rollout limit"）；Semi-OPD 的离线回答用 4K 上限生成。两者同为 4K 的对照是 Table 3：这一对 Semi-OPD 相对 OPD 增益 last-step +9.5，AUC +7.09，这才是同长度复现的目标。论文没提过滤截断的离线回答（只说"same sampling parameters and response cap as OPD"），本仓也不过滤。训练开销：OPD 56.8 GPU 小时，Semi-OPD 5.0 GPU 小时。

## 与本仓实现的差异

- teacher logp：本仓由 ref worker 每步在线重算，没有预先缓存。数值相同，只是计算量更大。所以本仓的速度对比不能代表论文的加速比，只用来比较效果。
- 回放方式：
  - `offline_response_ids` 存的是 vLLM 采样得到的原始 token id，包含停止 token，不经过文本重新分词。
  - prompt 的分词方式与 RLHFDataset 一致：先 chat template（`enable_thinking=False`），再用 `add_special_tokens=False` 分词。
  - 训练时 `replay.py` 把回答截到 `MAX_RESPONSE_LEN`，截出来的前缀等价于在线 rollout 在该长度上限下的输出。
  - attention mask 按存储长度生成。
- 优势与更新：verl 每步用当前 actor 重算 `old_log_probs`，所以 A_t 用的是当前 θ。每步只更新一次，PPO ratio 恒为 1，梯度等于 A·∇log π_θ，与论文的目标一致。
- 校验：每步都会写 `semi_opd/replayed=1`，作为确实走了回放路径的证据。rollout.n 必须为 1，否则会报错。
- 模板差异（2026-10-30 已用两边 tokenizer 的 `apply_chat_template` 核实）：Qwen3-1.7B 在 `enable_thinking=False` 时，assistant 前缀是 `<|im_start|>assistant\n<think>\n\n</think>\n\n`；Qwen3-4B-Instruct-2507 的模板里没有 think 相关逻辑，传不传 `enable_thinking` 都只生成 `<|im_start|>assistant\n`。本仓让 teacher 直接看 student 的 prompt id，所以 teacher 的输入里多了一个它原生模板没有的空 think 块。论文没有说明怎么处理。这个前缀对 teacher logp 的影响还没量化。
- 第一轮用本仓 OPD 协议（b32n1r2k），即 32 题 × 1、response 2048、lr 1e-6、fp32 主权重，跑 100 步；wd 0.01、grad clip 1.0、constant lr 无 warmup 已与论文一致（verl 默认值，train.log 核对过）。与论文的差异是每步题数（32 对 128）、训练 response 长度（2048 对 4096；离线回答按 4096 生成，训练时截到 2048）、总步数（100 步相当于 3200 题，论文约 110 步、14K 题）和评测集。如果这一轮复现不出"Semi-OPD 优于 OPD"，再改用论文配置。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| 1.7B(NT) ← 4B-Instruct-2507，本仓协议 b32n1r2k，100 步 | OPD 26.2 / Semi 39.8（论文配置） | 未复现：MATH-500 上 Semi 不优于 OPD（step 40–100 低 2.5–5.0） | [experiments/2026-10-30-semi-vs-opd.md](../experiments/2026-10-30-semi-vs-opd.md) |

## 待解决

- 冒烟和本仓协议的第一轮已完成。下一步按论文配置（128 × 1、4096）重跑，并在 32K 下评测。
- 评测：本仓评测集（aime24/25、amc23、math500）与论文（AIME24/25/26、HMMT26，32K 长度）不同。要和论文数字对齐，需要补 AIME26 和 HMMT Feb26 两个评测集。
- 用 diag 的 top-20 overlap 估一下这一对的 overlap，与论文的 62.5%（top-64）对照，两者口径不同。
