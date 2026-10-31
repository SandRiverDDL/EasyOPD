# ResOPD：Tail Residualization for Sparse OPD（`res_opd`）

涉及代码：`oss_refs/ResOPD/verl/trainer/distillation/resopd.py`、`oss_refs/ResOPD/verl/trainer/distillation/losses.py`、`oss_refs/ResOPD/verl/trainer/distillation/fsdp/losses.py`、`oss_refs/ResOPD/verl/workers/rollout/vllm_rollout/prompt_logprobs.py`、`oss_refs/ResOPD/examples/resopd/run_resopd.sh`（均在 `/mnt/NLU_disk17/chenhaoyu02/` 下）。

- 论文：ResOPD: Tail Residualization for Sparse On-Policy Distillation，arXiv [2610.04882](https://arxiv.org/abs/2610.04882) v1，2026-10-04 提交。作者 Penghui Yang、Long Xing、Xuanlang Dai、Ziyu Liu、Kai Chen、Yuhang Zang（通讯，pjlab）；单位 SJTU、Shanghai AI Lab、USTC、Fudan。
- 会议：arXiv comments 字段只有「25 pages, 9 figures」，没有会议信息；README 也未声明录用。结论：预印本，未见录用证据。
- 官方实现：有。[InternLM/ResOPD](https://github.com/InternLM/ResOPD)，本地浅克隆 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/ResOPD`，commit `3bdf019daf9818419fe38dbc6516e41ca5a70dcb`（2026-10-09）。框架是 verl（`UPSTREAM.md` 记录基于 verl `1dda039b9c6311889a9c9a26040677e1254814f9`），FSDP2 + vLLM。README:252 写明「A full GPU training run of this extracted port is outside the recorded release validation」，即这份开源移植没有完整 GPU 训练验证。
- 本仓实现：未实现（计划注册名 `res_opd`）
- 状态：调研中
- 结论（一句话）：ResOPD 是同 tokenizer 的稀疏 teacher（Top-k）OPD 梯度估计器：head 精确求和、tail 用 tail bucket 做控制变量后残差采样，保持无偏并降方差；需要 teacher Top-k 加采样 token 的 logprob 和 student full-vocab logits，在 verl 里实现成本中等。

## 方法要点

记号（论文 Sec 3）：对固定前缀，student 分布 p、teacher 分布 q，r_i = log(p_i/q_i)，s_i = ∇log p_i，G_i = r_i·s_i，reverse-KL 的精确梯度 g = E_{A~p}[G_A]。sampled-token OPD 用单个采样 token 的 G_A 估计 g，无偏但方差大；Top-k fKL/rKL 只在 teacher Top-k 上算 KL，低方差但有偏。ResOPD 要在只拿到 teacher Top-k（S，|S|=k）的条件下做到无偏且低方差。

- HETS（Head-Exact Tail-Sampled，Eq 6）：ĝ = h + J·G_A，其中 h = Σ_{i∈S} p_i G_i 是 head 精确项，J = 1[A∉S] 表示采样 token 落在 tail。无偏，但 tail 项方差未处理。
- Tail bucket（Eq 8–9）：把 tail 合成一个「鬼 token」，τ_p、τ_q 是 student / teacher 的 tail 总质量，b = log(τ_p/τ_q)，μ_B = b·∇log τ_p；粗粒度梯度 g_coarse = h + τ_p·μ_B，这正是 TA-OPD 一类 tail-bucket 方法的梯度（有偏）。
- 通用无偏族（Eq 10）：ĝ(B) = h + τ_p·B + J·(G_A − B)，对任意控制变量 B 都无偏；方差（Eq 12）V = τ_p·V_tail + P·τ_p·‖μ_tail − B‖²，B 越接近 tail 条件均值 μ_tail 方差越小。
- ResOPD（Eq 13）：取 B = μ_B，ĝ = g_coarse + J·(G_A − μ_B)。直观上「粗粒度 tail-bucket 梯度 + 采样残差修正」：tail bucket 提供低方差主项，采样残差把偏差修回来。
- SC-TA（Eq 25，消融）：系数 γ=1 的变体，ĝ^SC = ĝ^ResOPD − J·C_A，论文 Table 10 显示效果更差。

保留比例与分母：ResOPD 不做 token 选择，所有 response token 都参与，保留比例 100%；稀疏的是每个位置的 teacher 支撑（k=4）。损失按 verl `loss_agg_mode` 聚合，默认 `token-mean`（`verl/trainer/config/actor/actor.yaml:85`），分母是 batch 内全部有效 response token 数（代码推断，论文未单独写分母）。

代码实现（`resopd.py:26` `resopd_loss(student_logits, teacher_topk_log_probs, teacher_topk_ids, teacher_sampled_log_probs, teacher_sampled_ids)`）：

- 在 FP32 里对 student logits 做 log_softmax，得到 full-vocab p。
- `head_surrogate`（:76）用 detach 的条件概率 × ratio × support_log_p 构造 head 项；`sampled_surrogate`（:81）对应采样 token。
- `bucket_surrogate = -tail_log_ratio.detach() * support_score / student_tail_mass.detach()`（:94），tail 质量在 :90–91 用 eps clamp。
- `centered_event = head_hit - student_mass.detach()`（:98）；`surrogate = where(head_hit, head, sampled) - centered_event * (head - bucket)`（:99–100），这是把 Eq 13 写成可反传的 surrogate。
- `losses = value.detach() + (surrogate - surrogate.detach())`（:102）：前向数值等于 reverse-KL 估计值，反向梯度等于 ResOPD 估计器（straight-through 写法）。
- 分发：`losses.py:124` `compute_topk_loss` 在 :138 分发到 resopd（仅 FSDP）；`collect_resopd` 在 :183 注册，`use_topk=True`；`distillation_loss`（:282）用 `agg_loss` + `config.loss_agg_mode` 聚合。`use_policy_gradient=False` 时是直接监督反传，不走 PPO ratio。
- FSDP 入口 `fsdp/losses.py:26` `compute_resopd` 解包 nested tensor，并为 Ulysses SP 切片。

## 适用条件

- 同 tokenizer：必须。README 写明 student 与 teacher 必须共享 token-ID 词表。
- teacher 信息：每个位置需要 teacher 的 full-normalized Top-k logprob（ids + 值）以及 student 采样 token 的 teacher logprob。官方用 vLLM `prompt_logprobs`，它返回 top-k 加实际 token；`prompt_logprobs.py:19` `extract_prompt_logprobs` 额外抽出 `sampled_prompt_ids/logprobs`（:63–64）。不需要 teacher full-vocab logits。
- student 信息：需要 full-vocab logits（算 tail 质量 τ_p 和 ∇log τ_p），显存与 GKD 类方法同量级。
- 任务奖励：不使用。launcher `use_task_rewards=False`。
- 无偏性前提：要求 rollout 从 student 完整分布采样。论文自己承认主数学实验用 top-p 0.95 / top-k 20 截断 rollout，违背了这一假设（Table 7 是只训第一个 epoch 的 full-support 对照；论文 Sec 4.6 指出去掉截断后轨迹更长、更常撞到 response 上限）。
- 不支持（README）：Megatron、SGLang teacher、fused kernels、chunked top-k。

## 论文的实验设置

数学主实验（论文 Table 11 超参表）：

- 模型：student Qwen3.5-4B（thinking），teacher Qwen3.5-27B。
- 数据：从 DeepMath-103K 过滤「teacher 两次都对、student 至少错一次」的题，共 1276 题，切成 1024 训练 / 252 held-out；HF 上发布为 `ygyjrc/ResOPD-deepmath`（README）。
- 训练：3 epoch（192 step），global batch 16，AdamW，lr 1e-6，梯度裁剪 1.0，max prompt 2048，max response 32,768（训练与评测相同）。
- rollout 采样：temperature 1.0、top-p 0.95、top-k 20。论文未写 rollout n（launcher 是 n=1）。
- teacher 支撑：默认 k=4。
- 硬件：16×H200（8 rollout、4 训练、4 teacher），verl + vLLM。
- 评测：DeepMath held-out 252 题（avg@2）、AIME26 30 题（avg@16）、HMMT26 33 题（avg@16）、OlymMATH 英文子集 200 题（avg@2）；另报 step 6/32/64/128/192 checkpoint 的 AUC。

跨领域实验（Table 12）：student Qwen3-4B，teacher Qwen3-30B-A3B-Instruct-2507，30k 混合 prompt，训练 response 8192、评测 32768，100 step，batch 288，lr 5e-6（10 step warmup，cosine 衰减到 0.1×），top-p 1.0 / top-k -1，关闭 thinking。

主要数字（均来自论文）：

| 方法 | 数学 Avg Score（Table 3） | 数学 Avg AUC（Table 3） |
|---|---|---|
| Student | 42.85 | – |
| Teacher | 74.01 | – |
| Top-k fKL | 43.58 | 42.80 |
| Top-k rKL | 44.76 | 44.63 |
| TA-OPD | 48.17 | 47.94 |
| Sampled-token OPD | 44.03 | 46.69 |
| vOPD | 48.85 | 47.49 |
| HETS | 45.72 | 46.18 |
| ResOPD | 50.12 | 47.96 |

- ResOPD 分项（Table 3）：DeepMath 60.91、AIME26 59.38、HMMT26 39.96、OlymMATH 40.25。Avg AUC 47.96 与 TA-OPD 的 47.94 几乎相同，优势主要在最终分数。
- 固定前缀梯度审计（Table 2，k=4）：协方差迹比 sampled-token OPD 低 52–75%；例如 DeepMath 4B/27B 面板 sampled 0.183 vs ResOPD 0.047。
- 跨领域（Table 5）Avg：sampled-token 42.48、HETS 43.63、ResOPD 44.15。
- k 消融（Table 6）ResOPD − HETS：k=1 +0.11、k=4 +1.01、k=16 +0.70、k=32 +1.56。
- lr 扫描（Table 8）ResOPD − HETS：1e-6 +1.16、2e-6 +0.67、5e-6 +3.81；1e-5 时两者都发散。
- SC-TA 消融（Table 10）：SC-TA 46.33 vs ResOPD 50.12。
- 同保留比例下的 random 对照：不适用。ResOPD 不做 token 选择，对照是同 k 的 Top-k / HETS / TA-OPD。

证据边界：AIME26、HMMT26 只有 30/33 题；论文全文（HTML 转文本后检索 seed / standard error / ±）没有出现 seed 数和误差棒，所以 ResOPD 相对 vOPD（+1.27）、TA-OPD（+1.95）的差距是否超出噪声无法从论文判断。

## 与本仓实现的差异

本仓尚未实现。官方 launcher 与论文的差异如下（`examples/resopd/run_resopd.sh`）：

| 项 | 论文（Table 11） | launcher 默认 | 位置 |
|---|---|---|---|
| max response | 32,768 | 2048 | :20 |
| max prompt | 2048 | 1024 | :19 |
| batch | 16 | 8（mini-batch = train batch） | :18、:43 |
| epoch | 3（192 step） | 1 | :81 |
| rollout top-p / top-k | 0.95 / 20 | 1.0 / -1 | :54–55 |
| lr | 1e-6 | 1e-6 | :42 |
| k | 4 | 4 | :17、:68 |

- README 的 defaults 表（README:130）写「Response budget 32,768 tokens」，与 launcher 实际的 2048 矛盾；README:65 又说 launcher「a shorter response budget for the integration example」。以 launcher 为准是 2048，复现论文必须手动改回 32,768。
- launcher 的采样是 full-support（top-p 1 / top-k -1），符合无偏假设，但不同于论文主表的截断 rollout；用它跑出的数字应对照论文 Table 7 的 Full 行（只训第一个 epoch、取 step 64：ResOPD Avg 47.86 / AUC 46.79，HETS 45.17 / 45.68，Naive Top-k rKL 42.72 / 44.96；Default 截断 rollout 下 ResOPD 为 47.10 / 47.57），而不是 Table 3。
- 其他 launcher 设置：`rollout.n=1`（:52）、`ppo_epochs=1`（:45）、micro batch 1（:44）、FSDP2（:38）、student 4 卡 + teacher 4 卡 TP4（:14–16）、`loss_mode=resopd`（:67）、`use_task_rewards=False`、`use_policy_gradient=False`、`loss_max_clamp=null`、`log_prob_min_clamp=null`（:69–72）；`algorithm.adv_estimator=grpo`（:25）只是占位，任务奖励关闭后不参与损失。
- 上游 verl `distillation.yaml` 的默认是 `loss_mode=k3`、`topk=32`、`use_task_rewards=true`，必须靠 launcher 覆盖。
- 测试：`tests/resopd/`（test_estimator、test_teacher_payload、test_verl_integration），`VALIDATION.md` 记录 22 项独立检查 + 13 项集成检查，但没有完整 GPU 训练（README:252）。

在本仓（同 tokenizer teacher）实现的成本：中等。

- EasyOPD 的 verl 没有上游的 `verl/trainer/config/distillation/distillation.yaml` 与 Top-k distillation loss 通路；本仓 `verl/trainer/distillation/losses.py` 的注册表（`DistillationLossSettings`）面向跨 tokenizer 方法（simple/simct）。
- 需要新增：teacher 侧返回 Top-k ids/logprobs + 采样 token logprob 的通道（可复用 `gkd` 的 teacher 前向，或参考官方 vLLM `prompt_logprobs` 抽取）；student 侧在 actor 前向里保留 full-vocab logits 算 τ_p（`gkd/core.py` 已经在用 full-vocab teacher log_probs，可参考其显存处理）；`resopd_loss` 本体约 100 行，可几乎原样移植。
- 风险点：32k response 下 student full-vocab FP32 log_softmax 的显存；Ulysses SP 切片；on-policy 单次更新（ppo_epochs=1、mini-batch = batch）是无偏前提，不能照搬多 mini-batch 设置。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| Qwen3.5-27B → 4B，DeepMath，k=4，数学 Avg（Table 3） | 50.12 | 未跑 | [repro/](../repro/) |
| 同上，full-support rollout，1 epoch step 64 Avg（Table 7） | 47.86 | 未跑 | [repro/](../repro/) |
| Qwen3-30B-A3B → Qwen3-4B，跨领域 Avg（Table 5） | 44.15 | 未跑 | [repro/](../repro/) |
| 固定前缀协方差迹降幅（Table 2，k=4） | 52–75% | 未跑 | [repro/](../repro/) |

## 待解决

- 确认 EasyOPD 当前 verl 能否以最小挂接点拿到 teacher Top-k + 采样 token logprob；若不能，评估是否移植上游 Top-k distillation 通路。
- 论文主表用截断 rollout，launcher 用 full-support；复现时先定以哪张表为目标（Table 3 还是 Table 7）。
- 32k response 下 student full-vocab logits 的显存方案（分块 log_softmax 或只在 head ∪ {A} 上保留梯度）。
- 成本阶梯最低一级可先做 CPU 级检查：复用官方 `tests/resopd/test_estimator` 的有限动作精确梯度检查，验证移植后的无偏性。

## 来源

- https://export.arxiv.org/abs/2610.04882（标题、作者、日期、comments）
- https://arxiv.org/html/2610.04882v1（正文、公式、Table 2/3/5/6/8/10/11/12）
- https://github.com/InternLM/ResOPD（浅克隆 commit `3bdf019`，README、launcher、代码、tests）
