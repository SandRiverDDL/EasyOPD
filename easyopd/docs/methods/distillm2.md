# DistiLLM-2：对比式蒸馏（`distillm2`）

> 文件名：`methods/distillm2.md`。一个方法一篇，常青：理解或实现变了就改，不按日期追加。
> 读之前先看 [INDEX](../INDEX.md)；时序细节放 `repro/` 和 `experiments/`，这里只放链接。

- 论文：DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs，arXiv [2503.07067](https://arxiv.org/abs/2503.07067)（v1 2025-03-10，v2 2025-05-30），作者 Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, Se-Young Yun。会议：ICML 2025，已核实（OpenReview note `rc65N9xIrY` 的 venue 为 \"ICML 2025 oral\"；ICML 官网有 oral 页面 `icml.cc/virtual/2025/oral/47185`）。注意 arXiv comments 写的是 \"Spotlight\"，与 OpenReview 的 oral 不一致，以 OpenReview 和 ICML 官网为准。
- 前作：DistiLLM（Ko et al., 2024），SKL / SRKL 和 adaptive off-policy 都来自它；本文称其为 v1。
- 官方实现：<https://github.com/jongwooko/distillm-2>，commit `fe4cf9bfbb4f83219dd1d69219800164b59685fa`（2025-06-27），基于 TRL 0.9.6 的 DPOTrainer 改写，依赖 transformers 4.43.2、torch 2.4.0、vllm 0.5.4、deepspeed 0.15.0、peft 0.9.0（`setup.py`），DeepSpeed ZeRO-3 + LoRA 训练。本地浅克隆在 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/distillm-2`。
- 本仓实现：未实现（计划注册名 `distillm2`）
- 状态：调研中
- 结论（一句话）：每个 prompt 同时用一条 teacher 回答（做 skew forward KL，把 student 概率往上拉）和一条 student 回答（做 skew reverse KL，把 student 多给的概率压下去），每个 epoch 前离线重新采一次 student 回答；在三组指令跟随模型对上胜率平均比次优方法高 1.95–4.53（论文 Table 2）。它本质是 off-policy / batched on-policy 方法，搬到 verl 需要改造。

## 方法要点

论文的出发点是观察到 forward KL 和 reverse KL 在不同来源的回答上作用不同：在 teacher 生成的回答（TGO）上，KL 主要提高 student 的似然；在 student 生成的回答（SGO）上，RKL 主要压低 student 过高的似然（论文 §3.1，Figure 1a）。直接把 DPO 套到 KD 上（DPKD，Eq. 4）会 reward hacking：Figure 1(b) 中 student 回答上的 NLL 被压到 91.25，teacher 回答上只有 20.29，student 丢掉了预训练知识。本文改为对两类回答用两种 divergence（CALD，Eq. 5），再加上数据构造和课程式系数，得到 DistiLLM-2（Eq. 2）：

L = 1/(2|D|) Σ_{(x, y_t, y_s)} [ (1−β)·D_SKL^(α_t)(x, y_t) + β·D_SRKL^(α_s)(x, y_s) ]

- SKL（skew KL）：D_SKL^(α) = KL(p ‖ αp + (1−α)q_θ)，在 teacher 回答 y_t 的每个位置上对全词表求和（Eq. 1 的逐 token 分解）。
- SRKL（skew reverse KL）：D_SRKL^(α) = KL(q_θ ‖ (1−α)p + αq_θ)，在 student 回答 y_s 上同样逐 token、全词表求和。
- p 是 teacher，q_θ 是 student；Eq. 5 的 CALD 等价于 β=1/2 的特例。
- α 的课程式更新（§3.3，Algorithm 1 第 11 行）：α_t ← 1 − (1−α0)·m / (p(y_s|x) − q_θ(y_s|x))，α_s ← 1 − (1−α0)·m / (p(y_t|x) − q_θ(y_t|x))。按字面看 α_t 用的是 y_s 上的差、α_s 用的是 y_t 上的差，像是记号互换了；官方代码里 α_1（SKL）用的是 chosen（teacher 回答）上的差，见下文「差异」。
- β 逐步增大（Algorithm 1 第 13 行）：β ← clip(e/E + τ/T, β0, 1)，β0=0.5。按字面计算，E=3 时第一个 epoch 内 β 就会被截到 1，即后期只剩 SRKL。
- 数据构造（§3.2）：每个 epoch 开始前用 vLLM 为所有 prompt 采一条 teacher 回答和一条当前 student 的回答，论文称 \"batched on-policy\"（Algorithm 1 第 5 行）。§3.2 还比较了用推测解码回答或更强 LLM 的回答替代 teacher 回答，结论是没有帮助（Figure 2）。
- α 课程的依据是 Eq. 7：log(p/q̃^(α)) ≈ (1−α)(p−q)，其中 q̃^(α) = αp + (1−α)q。论文据此为每个样本闭式求 α，使 (1−α)(p−q) 在整个训练中保持一致：p 与 q 差得大的「难」样本取较大的 α，差得小的取较小的 α（§3.3）。Eq. 6 的 Remark 把本文目标和 DPO 式的对比目标联系起来。

官方代码的实现路径（commit `fe4cf9b`）：

| 作用 | 位置 |
|---|---|
| 训练流程：可选 SFT → vLLM 分别生成 teacher / student 回答 → 拼成 chosen/rejected 偏好格式 → 必要时对齐词表大小 → 蒸馏 | `README.md:49-81` |
| 生成：每个 prompt 一条，默认 T=0.8、top-p 0.95、max_tokens 1024、seed 42 | `generate/generate_vllm.py:16-38` |
| 拼数据：chosen = teacher 回答，rejected = student 回答 | `generate/reformat.py:18-29` |
| 系数的硬编码默认值：`base_alpha_1 = base_alpha_2 = 0.1`，`update_alpha=False`，`gradual_beta=False` | `src/distillm_trainer.py:506-508` |
| SKL / SRKL 本体（全词表 log_softmax，混合分布用 logsumexp 计算；SRKL 混合分布里的 student 项 `.detach()`），以及 α 课程 | `src/distillm_trainer.py:1102-1242` `get_batch_logps` |
| 两项的加权：`(2−beta)·chosen_kl + beta·rejected_kl` | `src/distillm_trainer.py:1073-1078` `kd_loss` |
| 序列内 token 平均（`average_log_prob=True`），batch 内取均值 | `src/distillm_trainer.py:1298`、`:1347` |
| α 课程的锚点：在第一次 `log()` 时把存下的指标设成 anchor | `src/distillm_trainer.py:1528-1532` |
| 词表大小不同时扩 embedding | `utils/resize_embedding.py` |

## 适用条件

DistiLLM-2 要求 teacher 与 student 同 tokenizer（只允许词表大小不同，用 `resize_embedding.py` 补齐），因为 SKL / SRKL 在两者的全词表分布上逐 token 求和。teacher 必须是白盒，而且需要完整词表 logits，不是 top-K 也不是采样 token 的 logprob：官方代码把 teacher 加载进训练进程，在 teacher 回答和 student 回答两条序列上都做全词表前向。不使用 task reward，也没有 verifier。论文的模型对都是 7B/9B → 1.3B–2B，student 先在 ground-truth 回答上 SFT（附录 C.2）。训练数据需要预先生成好的 teacher 回答，这一点和 verl 只 rollout student 的模式不同。

## 论文的实验设置

超参来自附录 C.2 的 Table 11，「代码 config」一列是官方 `training_configs/qwen2.5-1.5b-distillm2.yaml` 的取值。

| 项 | 论文（Table 11 / 附录 C.2） | 代码 config |
|---|---|---|
| 硬件 | 4×A100 80GB | `--num_processes=4`（README） |
| 微调方式 | LoRA r=16，所有 attention 与 MLP 线性层 | r=16，alpha 128，dropout 0.05，q/k/v/o/gate/up/down |
| lr | 5e-5 | 5e-5，cosine，warmup ratio 0.1 |
| 有效 batch | 128 | per-device 1 × grad accum 8 × 4 卡 = 32 |
| epoch | 指令跟随 3，数学 2，代码 2 | `num_train_epochs: 1` |
| α0 / β0 | α0=0.1，第一个 epoch 不做 α 课程；β0=0.5 | α 固定 0.1；`gradual_beta=False` |
| 长度 | 未在 Table 11 中列出 | max_length 1024，max_prompt_length 512 |
| 预训练 LM loss | 不用（附录 C.2） | 不用 |

数据与评测（§4）：

- 指令跟随：UltraChat200k 中随机 50k 个 prompt。模型对是 Qwen2-7B-Inst → Qwen2-1.5B、Mistral-7B-Inst → Danube2-1.8B、Gemma-2-9B-Inst → Gemma-2-2B。评测 AlpacaEval、Evol-Instruct、UltraFeedback 的 pairwise 胜率，裁判是 GPT-4o（前两个）和 GPT-4o-mini（UltraFeedback），交换回答顺序后取平均；生成用 T=0.8、top-p 0.95、最长 512。
- 数学：MetaMathQA 中随机 50k 条，student 先在完整 MetaMathQA 上 SFT 1 个 epoch；模型对是 Qwen2-Math-7B-Inst → Qwen2-Math-1.5B、Qwen2.5-Math-7B-Inst → Qwen2.5-Math-1.5B；GSM8K 与 MATH 的 pass@1，greedy，最长 1024。
- 代码：WizardCoder 的 prompt；模型对是 DeepSeek-Coder-6.7B-Inst → 1.3B、Qwen2.5-Coder-7B-Inst → 1.5B；HumanEval 与 MBPP（EvalPlus）的 pass@1，greedy。

论文没有报告 seed 数或方差。

主要数字（AVG 列）：

| 设置 | teacher | student | GKD | DistiLLM | DistiLLM-2 | 来源 |
|---|---|---|---|---|---|---|
| Qwen2-7B → Qwen2-1.5B，三集平均胜率 | 76.12 | 30.33 | 56.14 | 56.35 | 58.69 | Table 2 |
| Mistral-7B → Danube2-1.8B | 83.01 | 27.02 | 50.68 | 52.37 | 56.45 | Table 2 |
| Gemma-2-9B → Gemma-2-2B | 90.15 | 28.62 | 69.73 | 70.30 | 74.83 | Table 2 |
| Qwen2-Math-7B → 1.5B，GSM8K/MATH 平均 pass@1 | 62.61 | 50.05 | 54.80 | 55.07 | 55.93 | Table 3 |
| Qwen2.5-Math-7B → 1.5B | 67.07 | 52.24 | 60.38 | 61.10 | 62.07 | Table 3 |
| DeepSeek-Coder 6.7B → 1.3B，HumanEval/MBPP 平均 | 83.96 | 61.42 | 64.61 | 64.00 | 67.79 | Table 4 |
| Qwen2.5-Coder-7B → 1.5B | 75.61 | 45.79 | 51.38 | 50.90 | 52.47 | Table 4 |

- Table 2 的 Qwen2 一行中，DistiLLM-2 的三个分项是 AlpacaEval 69.88、Evol-Instruct 47.13、UltraFeedback 59.05；同表 SpecKD 在三组上分别是 54.43、54.50、69.26。正文称比次优方法平均高 +2.34、+1.95、+4.53（§4.1）。
- 组件消融（Table 5，三组模型的平均）：DistiLLM v1 41.55，加对比损失 43.61，再加 β 递增 44.80，对比损失加 α 课程 45.22，三者全加 46.50。
- batched on-policy 的代价（附录 D.1，Table 12，GPT-2 在 dolly 上的 ROUGE-L，on-policy / batched / off-policy）：DistiLLM-2 26.37 / 26.20 / 26.13，DistiLLM 26.12 / 26.11 / 26.12，GKD 23.75 / 23.21 / 22.89，DPKD 6.85 / 6.43 / 6.43。三种数据方式之间的差距不到 0.3。
- 数学提升较小（Table 3 中比 DistiLLM 高 0.86 和 0.97），而 GSM8K / MATH 是单次 greedy、无 seed 报告，这类差异的置信度有限。

## 与本仓实现的差异

本仓尚未实现。下面先列官方实现内部「论文 vs 代码」的差异（均为读代码所得，未实际运行），再给出在 verl / EasyOPD 中实现的成本。

论文与官方代码的差异：

- 课程 α 和 β 递增在发布代码里默认关闭。`distillm_trainer.py:506-508` 硬编码 `update_alpha=False`、`gradual_beta=False`，没有从 config 读取，也没有命令行开关。配合 config 里 `num_train_epochs: 1`，发布代码跑的是「α 固定 0.1、两项等权」的单 epoch 训练，接近论文 Table 11 里「第一个 epoch 不做 α 课程」的设置，但缺少论文后续 epoch 的 β 递增和 α 课程，也就复现不了 Table 5 里后两个组件的增益。
- β 的形式不同。代码在 `gradual_beta=True` 时用 β_code = 1 + 0.5·min(1, 2·step/max_steps)，loss = (2−β_code)·SKL + β_code·SRKL（`:1073-1078`）。换算成论文的写法，SRKL 的占比从 0.5 增到 0.75，在训练过半时到顶；论文 Algorithm 1 是增到 1。两者都乘了常数因子，和 Eq. 2 的 1/2 不同，相当于改了学习率。
- α 课程的实现与 Algorithm 1 字面不一致。代码在 SKL 上用 chosen（teacher 回答）上的差 Δ = exp(token 平均 log p) − exp(token 平均 log q)，在 SRKL 上用 rejected（student 回答）上的反向差，α = clip(1 − anchor/(Δ+1e-5), 0.01, base_alpha)（`:1145-1186`）。所以 Δ 越大 α 越大，方向和论文「难样本取较大 α」一致，但 α 被限制在 [0.01, base_alpha=0.1]，只能比 α0 小，与论文 Table 1「后续 epoch 的最优 α 不大于第一个 epoch」的观察相符。anchor 取自第一次 `log()` 时存下的指标（`:1528-1532`）；`_stored_metrics` 是按 train/eval 分层的 defaultdict（`:511`），按指标名直接索引外层拿到的是空字典，之后 `(1−base_alpha)·{}` 会抛 TypeError，被 `:1145-1153` 的裸 `except` 吞掉，所以即使打开开关也很可能静默退回固定 α。这一点是读代码推出的，未运行验证。
- SRKL 的混合分布里 student 一项做了 `.detach()`（`:1196`、`:1208`），论文公式没有写这个 stop-gradient。
- 有效 batch 是 32（1 × 8 × 4 卡），论文是 128。
- 发布流程只生成一次 student 回答（README 的 generate → reformat → train），不是每个 epoch 重新生成；多 epoch 的 batched on-policy 需要用户自己循环执行 generate 和训练。
- 生成默认 T=0.8、top-p 0.95、每个 prompt 一条回答（`generate_vllm.py`），论文没有单独写训练数据的采样参数。

论文本身的可疑点：

- arXiv comments 写 Spotlight，OpenReview 和 ICML 官网是 oral。
- Table 4 表头把 DeepSeek-Coder teacher 写成 6.9B，正文写 6.7B（§4.3）。
- Algorithm 1 中 α_t / α_s 的下标看起来互换了（见「方法要点」）。

在 verl / EasyOPD（同 tokenizer teacher）中实现的估计：

- 照原样做（off-policy / batched on-policy）：成本中到高。verl 的训练循环只 rollout student，DistiLLM-2 每个 prompt 还需要一条 teacher 回答。可以离线用 vLLM 预生成 teacher 回答放进数据集，训练时作为额外字段读入；这需要改数据管线，并让 actor 对每个 prompt 多做一次前向（teacher 回答一条、student 回答一条）。「每个 epoch 重新生成 student 回答」可以直接用 verl 的在线 rollout 替代，这已经是比论文更 on-policy 的设置。
- teacher 信号是主要成本。SKL / SRKL 需要 teacher 在两条序列上的完整词表分布。以 Qwen2.5 的约 152k 词表、1024 长度计，一条序列的 bf16 logits 约 0.3 GB，ref worker 必须把 teacher logits 和 student logits 放在同一处计算，不能只传 top-K。若改用 top-K 近似（复用 OPSA 的 `ref_topk_*` 通路，见 [revisit_opd.md](revisit_opd.md) 的成本一节），就偏离了论文，混合分布 αp + (1−α)q 在 K 之外的质量也要另做处理，复现报告必须写明。
- on-policy 改造可行。只用 verl 的 student rollout 计算 SRKL 项（即在 y_s 上逐 token 的 skew reverse KL，与 GKD 在 student rollout 上算 JSD 的结构相同，可以参考本仓 `easyopd/methods/gkd/`），再把预生成的 teacher 回答上的 SKL 作为辅助 loss。附录 D.1 的 Table 12 显示 on-policy 与 batched 的差距只有 0.17 ROUGE-L（26.37 vs 26.20，GPT-2 小规模），说明 on-policy 化本身不太会损失效果，但这个证据来自 GPT-2 与 dolly，推到 7B → 1.5B 的指令跟随或数学只是推断。
- 其他偏差：官方用 LoRA r=16，本仓是否用 LoRA 需按所用 launcher 确认；指令跟随一组的评测是 GPT-4o 裁判的 pairwise 胜率，本仓 `experiments/_shared/scripts/` 是否支持未核对，数学和代码两组（greedy pass@1）更容易对齐。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| Qwen2.5-Math-7B-Inst → Qwen2.5-Math-1.5B，MetaMathQA 50k，2 epoch，GSM8K/MATH 平均 pass@1 | 62.07（Table 3；DistiLLM 61.10，GKD 60.38） | 未跑 | — |
| Qwen2-7B-Inst → Qwen2-1.5B，UltraChat 50k，3 epoch，三集平均胜率 | 58.69（Table 2；GKD 56.14） | 未跑 | — |

数学一行的评测是 greedy pass@1，不依赖 GPT-4o 裁判，最适合作为本仓的第一个复现目标。

## 待解决

- 发布代码的课程 α / β 递增默认关闭，打开后 α 课程很可能因 `_stored_metrics` 的索引问题静默失效；需要在小规模上打印 α 确认，或者问作者论文数字是用哪个版本的代码跑的。
- 论文 β 递增到 1（后期只剩 SRKL），代码只到 0.75；复现时选哪一个需要定下来，并在报告里写明。
- 在 verl 中是否接受 top-K 近似全词表 KL；不接受的话，需要评估 ref worker 输出全词表 logits 的显存和通信开销。
- 论文没有报告 seed 数和方差，数学一组和 DistiLLM 的差距不到 1 分，复现时至少跑多个 seed。
- 训练数据的生成参数（温度、top-p、每个 prompt 的回答数）论文未写，只能按代码默认 T=0.8、top-p 0.95、每个 prompt 一条。

## 来源

- https://export.arxiv.org/abs/2503.07067
- https://arxiv.org/html/2503.07067v2
- https://api2.openreview.net/notes/search?term=DistiLLM-2%20Contrastive%20Approach&limit=3
- https://icml.cc/virtual/2025/oral/47185
- https://github.com/jongwooko/distillm-2
