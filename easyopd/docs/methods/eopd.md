# EOPD：按 teacher 熵门控的 OPD + top-k forward KL（`eopd`）

> 文件名：`methods/eopd.md`。一个方法一篇，常青：理解或实现变了就改，不按日期追加。
> 读之前先看 [INDEX](../INDEX.md)；时序细节放 `repro/` 和 `experiments/`，这里只放链接。

- 论文：Entropy-Aware On-Policy Distillation of Language Models，arXiv [2603.07079](https://arxiv.org/abs/2603.07079)（v1 2026-03-07，v2 2026-05-22，v3 2026-06-12），作者 Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo, Kimin Lee（KAIST AI、U Toronto / Vector、IBM Research）。本文数字均出自 v3。
- 会议：ICML 2026。证据有两条：arXiv comments 字段为 "18 pages, 11 figures, ICML 2026"；WebSearch 结果里有 ICML 海报页 `icml.cc/virtual/2026/poster/64855`，标题与本文一致。海报页本身没能抓取成功，所以这一条只来自搜索结果摘要。
- 官方实现：<https://github.com/WLS04/EOPD>，commit `a09cc1fa3a457b61515ca1b5f3c3fff2252eccbb`（2026-05-18，提交信息 "init"）。这是一个 verl fork，本地浅克隆在 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/EOPD`。
- 本仓实现：未实现（计划注册名 `eopd`）
- 状态：调研中
- 结论（一句话）：在 PPO 式 sampled-token OPD 之上，只对 teacher 熵大于 τ 的 token 额外加一项 teacher top-k 上的 forward KL；论文报告 0.6B / 1.7B / 4B 三个 student 的 6 集平均 Avg@8 比 OPD 高 1.16 / 0.99 / 1.80（论文 §5.2）。

## 方法要点

论文的出发点是：sampled-token OPD 优化的是 reverse KL，它偏向 mode-seeking。在 teacher 熵高的位置（teacher 认为有多个合理续写），reverse KL 会让 student 只学到一个模式，丢掉 teacher 的多样性。forward KL 是 mode-covering 的，正好补这一块，但全词表 forward KL 的显存开销大。EOPD 的做法是只在高熵位置加 forward KL，并且只在 teacher 的 top-k 上算。

逐 token loss 是论文 Eq. 9：

L_t^EOPD = L_t^OPD + α · 𝕀[H_t^te > τ] · L_t^FKL

- L_t^OPD 是论文 Eq. 8 的 PPO clip 形式。advantage 为 Â_t = log π_te(y_t) − log π_old(y_t)，即 teacher 与 rollout 时 student 的 log-prob 之差，不用 GAE、不用 critic。
- H_t^te 是 teacher 在该位置的全词表熵。
- L_t^FKL 是论文 Eq. 10：Σ_{x∈S_t^k} π̃_te(x) · log(π̃_te(x) / π_θ(x))。S_t^k 是 teacher 概率最高的 k 个 token，π̃_te 是 teacher 在 S_t^k 内重新归一化后的分布。π_θ 用全词表归一化的 student 概率，不在 top-k 内重新归一化。
- 论文取值：τ = 0.8、α = 1.0（论文 §4），k = 16（附录 A Table 9）。τ = 0.8 时高熵 token 约占 15–20%（附录 E）。

与其他方法的区别：vanilla OPD 只有 L^OPD；GKD 用 β 在 forward / reverse KL 之间对所有 token 插值；EOPD 只在高熵 token 上加 forward KL。论文 Table 5 中 GKD（λ = 1.0）的 MATH500 Avg@8 为 67.20，EOPD 为 68.73（1.7B）。Table 7 的消融把门控换成「所有 token 都加 FKL」（Full FKL，67.58）或「随机选同样比例的 token」（Random FKL，67.33），都不如按熵选（68.73）。

### 官方代码中的对应位置（commit `a09cc1f`）

| 作用 | 位置 |
|---|---|
| advantage 估计器 `on_policy`：返回全 0，真正的 advantage 在 loss 里算 | `verl/trainer/ppo/core_algos.py:718` `compute_on_policy_dummy_advantage` |
| loss 注册 `on_policy_distill` | `core_algos.py:1025-1026` `compute_policy_loss_on_policy_distill` |
| `teacher_advantage = (teacher_log_prob - old_log_prob).detach()`，随后做 PPO clip / max | `core_algos.py:1161`、`:1168-1170` |
| top-k forward KL：`sum(exp(t_topk) * (t_topk - s_topk))` | `core_algos.py:1189` |
| `soft_kd_loss_coef`（即 α，默认 1.0） | `core_algos.py:1196` |
| `soft_kd_entropy_threshold`（即 τ，默认 0.8），比较符是 `>=` | `core_algos.py:1199-1210` |
| 每次调用都把高熵 token 写成每个 rank 一个 jsonl 的 debug 日志，并且必须传 tokenizer | `core_algos.py:1216-1273` |
| `soft_kd_loss = masked_sum(kl, soft_kd_mask) / total_valid_tokens * coef` | `core_algos.py:1277-1279` |
| 传入 `rollout_is_weights` 时直接报错 | `core_algos.py:1334` |
| `pg_loss = agg_loss(...) + soft_kd_loss` | `core_algos.py:1342-1347` |
| teacher top-k：先 `torch.topk(logits)`，再在 top-k 内 `log_softmax`（即 π̃_te） | `verl/workers/actor/dp_actor.py:264-266` |
| student 在 teacher 的 indices 处取值；`soft_kd_student_full_vocab=True` 时先全词表 log_softmax 再 gather（与 Eq. 10 一致），False 时在 top-k 内重新归一化 | `dp_actor.py:349-377`（无 rmpad 分支 `:427-449`） |
| ref_entropy、ref_topk 传进 loss | `dp_actor.py:625-634`、`:769-800` |
| ref（即 teacher）的 `compute_log_prob(calculate_entropy=True, topk=topk)`，输出 `ref_log_prob`、`ref_entropy`、`ref_topk_log_probs`、`ref_topk_indices` | `verl/workers/fsdp_workers.py:1036-1075` |
| `soft_kd_student_full_vocab: bool = False`（代码默认 False，README 设为 True） | `verl/workers/config/actor.py:93` |
| trainer `OnPolicyDistillTrainer`，在 `main_ppo.py:346` 注册；ref policy 的路径被替换为 `teacher_model.path` | `verl/trainer/ppo/on_policy_distill_trainer.py:14`、`ray_trainer.py:924` |

## 适用条件

EOPD 只适用于同 tokenizer：teacher top-k 的 token id 直接拿来 gather student 的 logits。teacher 必须是白盒，每个 response 位置要提供三样东西（论文 Algorithm 1）：采样 token 的 log π_te(y_t)、全词表熵 H_t、top-k 的 indices 和 log-prob。不需要完整词表分布落盘，只需要 k 个值加一个熵。EOPD 不使用 task reward，官方的 advantage 估计器直接返回 0。论文中 teacher 是 Qwen3-8B，student 是 0.6B 到 4B 的 Base 模型；附录 G 另有 Llama-3.1-8B-Instruct → Llama-3.2-3B-Instruct。

## 论文的实验设置

来源是论文 §5.1、附录 A Table 9 和附录 C；括号里是官方 README 的设置（与论文不同时标出）。

| 项 | 论文 |
|---|---|
| Student / 训练数据 | Qwen3-0.6B-Base、Qwen3-1.7B-Base 用 MATH 训练；Qwen3-4B-Base 用 DAPO（Table 2 标为 DAPO-Math-14k） |
| Teacher | Qwen3-8B，关闭 thinking（README：`teacher_model.path=Qwen/Qwen3-8B`） |
| 框架 / 硬件 | verl，4×A100 80GB |
| batch | B = 128，mini-batch 32，每轮 4 个梯度步 |
| 优化 | AdamW，lr 3e-6，cosine |
| rollout | 每个 prompt 采样 1 条（GRPO 基线 8 条），温度 1.0，top-p 1.0，最大 response 4096 |
| epoch | MATH 3 个，DAPO 2 个 |
| EOPD 超参 | τ = 0.8，α = 1.0，k = 16（README：`ref.topk_logits=32`） |
| chat template | `<\|im_start\|>user\n{query}<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>` |
| 评测 | Qwen2.5-Math 评测管线，温度 1.0，top-p 0.8，最大长度 8192，每题 8 个样本，报告 Avg@8 / Pass@8 |
| 评测集 | MATH500、AMC23、Minerva、OlympiadBench、AIME24、AIME25 |
| 开销 | 每步多 2.16s（平均步时 47.8s 的 4.5%），额外显存 219.3MB，全量 FKL 需要 1617.5MB（附录 C） |

论文 Table 2 的主结果如下，每格为 Avg@8 / Pass@8，列顺序为 KD / GRPO / OPD / EOPD。

| Student | 评测集 | KD | GRPO | OPD | EOPD |
|---|---|---|---|---|---|
| 0.6B | MATH500 | 47.80 / 69.60 | 51.83 / 74.40 | 50.09 / 73.20 | 52.02 / 76.00 |
| 0.6B | AMC23 | 23.43 / 52.50 | 25.25 / 55.00 | 24.69 / 57.50 | 27.81 / 55.00 |
| 0.6B | AIME24 | 2.19 / 6.67 | 4.58 / 10.00 | 2.50 / 10.00 | 4.17 / 13.33 |
| 1.7B | MATH500 | 63.80 / 84.20 | 68.83 / 84.60 | 67.76 / 84.80 | 68.73 / 87.60 |
| 1.7B | AMC23 | 38.11 / 70.00 | 40.62 / 70.00 | 39.06 / 70.00 | 41.88 / 75.00 |
| 1.7B | AIME24 | 10.06 / 20.00 | 9.17 / 20.00 | 8.33 / 20.00 | 10.42 / 23.33 |
| 4B | MATH500 | 74.73 / 92.20 | 80.47 / 91.00 | 78.81 / 90.80 | 80.20 / 93.00 |
| 4B | AMC23 | 48.13 / 85.00 | 58.44 / 75.00 | 57.33 / 80.00 | 60.94 / 87.50 |
| 4B | AIME24 | 12.50 / 30.00 | 14.85 / 30.00 | 18.33 / 26.67 | 17.92 / 36.67 |
| 4B | AIME25 | 12.08 / 23.33 | 12.66 / 26.67 | 12.08 / 30.00 | 17.50 / 33.33 |
| 0.6B | Minerva | 18.61 / 36.03 | 20.08 / 35.66 | 19.98 / 34.93 | 20.82 / 37.13 |
| 0.6B | OlympiadBench | 18.15 / 36.59 | 18.91 / 34.96 | 20.15 / 37.19 | 19.52 / 39.56 |
| 0.6B | AIME25 | 0.83 / 6.67 | 0.83 / 10.00 | 1.25 / 6.67 | 1.25 / 6.67 |
| 1.7B | Minerva | 28.14 / 44.10 | 29.46 / 48.16 | 29.83 / 47.06 | 30.15 / 50.74 |
| 1.7B | OlympiadBench | 27.66 / 48.40 | 29.89 / 50.81 | 30.09 / 51.56 | 30.28 / 51.11 |
| 1.7B | AIME25 | 3.36 / 16.67 | 4.58 / 16.67 | 6.25 / 16.67 | 5.83 / 16.67 |
| 4B | Minerva | 34.65 / 55.15 | 40.81 / 55.51 | 40.08 / 54.00 | 39.71 / 56.99 |
| 4B | OlympiadBench | 37.33 / 62.52 | 43.37 / 60.74 | 42.10 / 58.80 | 43.24 / 63.11 |

按论文 §5.2，6 个评测集平均下 EOPD 相对 OPD 的提升（Avg@8 / Pass@8）为：0.6B +1.16 / +1.37，1.7B +0.99 / +2.39，4B +1.80 / +5.05。需要注意，EOPD 并不是每一格都赢：例如 4B 的 MATH500 Avg@8 低于 GRPO（80.20 对 80.47），0.6B 的 AMC23 Pass@8 低于 OPD（55.00 对 57.50）。AIME 每集只有 30 题，单格差 3.33 分就是 1 道题；本次阅读只在 Figure 1 的玩具实验里看到 "3 seeds"，主表没有找到 seed 数或方差。

论文 Table 3 是 1.7B 的分布外评测，列顺序为 KD / GRPO / OPD / EOPD：GPQA-D Avg@8 为 27.01 / 27.86 / 30.08 / 31.50，MMLU-Pro 为 37.54 / 41.46 / 42.26 / 43.20，AlpacaEval LC-WR 为 19.30 / 16.86 / 22.86 / 23.83。

## 与本仓实现的差异

本仓还没有实现，下面列的是在本仓 verl 中复现要做的事，以及官方代码与论文之间的差异。

官方代码与论文的差异：

- top-k 的取值不一致：论文附录 A Table 9 是 k = 16，官方 README 的命令是 `actor_rollout_ref.ref.topk_logits=32`。复现以论文为准，并在 repro 报告里写明。
- 熵门控的比较符不一致：论文 Eq. 9 是 H > τ，代码 `core_algos.py:1199-1210` 是 `>=`。熵是连续值，实际影响可以忽略，但实现时按论文写。
- student 侧是否重新归一化由开关决定：`soft_kd_student_full_vocab` 的代码默认是 False（在 top-k 内重新归一化，与 Eq. 10 不一致），README 设为 True（与 Eq. 10 一致）。复现时用 True 的语义。
- README 引用的 launcher `examples/on_policy_distillation/on_policy_it.sh` 不在仓库里，该目录只有 README.md。完整的超参只能从 README 片段和论文 Table 9 拼出来。
- loss 每次调用都会对每个 rank 写高熵 token 的 jsonl 日志（`core_algos.py:1216-1273`），这是调试代码，复现时不要搬过来。
- 传入 rollout importance sampling 权重时直接报错（`core_algos.py:1334`），说明官方路径没有考虑 rollout / 训练引擎之间的 off-policy 修正。

在本仓 verl 中复现要做的事：

1. advantage：rollout 侧需要 teacher 在采样 token 上的 log-prob，advantage = log π_te − log π_old。本仓 `easyopd/methods/lightning_opd/advantage_estimator.py:19-20` 已注册 adv_est `on_policy_distillation`，计算的正是 teacher_log_prob − old_log_prob，可以直接复用，再走 verl 原生 PPO clip loss。
2. teacher 侧输出：需要 teacher 的 top-k log-prob、indices 和全词表熵。本仓 OPSA 已经在 `verl/trainer/ppo/ray_trainer.py:1438-1490`（`[EasyOPD:OPSA]` 标记）通过 `opsa_topk_logits_k` 向 teacher 请求 `ref_topk_log_probs` / `ref_topk_indices`，并把它们传进 actor batch。缺的是 teacher 熵：本仓 `verl/workers/fsdp_workers.py:916` 的 ref 前向写死了 `calculate_entropy=False`，需要在这里加一个开关并登记 touchpoint。OPSA 返回的 teacher top-k 是全词表归一化的 log-prob（`dp_actor.py:430-434`、`:514`，即 logits − logsumexp），EOPD 需要的 π̃_te 在 loss 里对这 k 个值再做一次 log_softmax 即可。
3. student 侧：需要 student 在 teacher indices 处、全词表归一化的 log-prob。`verl/workers/actor/dp_actor.py:120-175` 的 `opsa_gather_indices` 返回 `gathered_log_probs`，注释写明是 log-softmax 后 gather，语义与 Eq. 10 一致。该路径在 Ulysses SP 下会退化（`dp_actor.py:160-173`），复现时不要开 SP。
4. loss：新写一个 LossHook，计算 α · 𝕀[H > τ] · Σ π̃_te (log π̃_te − log π_θ)，按全部有效 token 数归一化（与 `core_algos.py:1277-1279` 一致），加到 PPO loss 上。
5. 配置：τ、α、k 三个超参；rollout 每个 prompt 1 条，lr 3e-6，B = 128，mini-batch 32。

成本估计为中。理由是 advantage、teacher top-k、student gather 三条数据通路本仓都已经有（lightning_opd 与 OPSA），新代码主要是一个几十行的 LossHook 和配置；需要改 `verl/` 的地方是 ref 前向返回熵（`fsdp_workers.py:916`），以及把 OPSA 专用的 top-k 请求（`ray_trainer.py:1438-1490`，目前挂在 OPSA 的 teacher 分支里）做成 EOPD 也能触发。以上判断基于读到的代码和注释，没有实际跑通这些路径。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| Qwen3-8B → Qwen3-1.7B-Base，MATH，MATH500 Avg@8（Table 2） | 68.73 | 未跑 | — |
| Qwen3-8B → Qwen3-4B-Base，DAPO，AIME25 Avg@8（Table 2） | 17.50 | 未跑 | — |

## 待解决

- 给本仓 ref 前向加熵输出的具体改法，以及 OPSA 的 top-k 请求能否脱离 OPSA 分支复用。
- 4B 的训练数据：正文 §5.1 只写 "DAPO (Yu et al., 2025)"，Table 2 表头写 "DAPO-Math-14k"，没有给出具体的数据来源；复现前要确认用哪个版本（本仓已有数据见 `reference/environment.md`）。
- 本次没有在论文里找到 teacher 打分时用的温度；EOPD 代码中 teacher log-prob 用的温度需要在 `fsdp_workers.py` 里再确认。
- 小评测集（AIME 30 题、AMC 40 题）差异需多 seed 才能下结论。

## 来源

以下是本次调研实际打开过的页面：

- arXiv 元数据页：<https://export.arxiv.org/abs/2603.07079>
- 论文全文 HTML：<https://arxiv.org/html/2603.07079v3>
- 官方仓库：<https://github.com/WLS04/EOPD>（浅克隆，读了 `README.md`、`examples/on_policy_distillation/README.md`、上表列出的源文件）

只在 WebSearch 结果里出现、没有成功打开的：ICML 海报页 `https://icml.cc/virtual/2026/poster/64855`（curl 抓取失败），OpenReview forum `WSRQ37tzk1`。
