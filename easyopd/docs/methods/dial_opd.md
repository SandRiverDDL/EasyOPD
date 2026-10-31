# DIAL-OPD：概率尺度感知的 token 选择（`dial_opd`）

涉及代码：`oss_refs/DIAL-OPD/dial_opd/selection.py`、`oss_refs/DIAL-OPD/dial_opd/trainer.py`、`oss_refs/DIAL-OPD/dial_opd/config.py`、`oss_refs/DIAL-OPD/dial_opd/train.py`、`oss_refs/DIAL-OPD/configs/train_qwen3_06b.json`（均在 `/mnt/NLU_disk17/chenhaoyu02/` 下）。

- 论文：DIAL-OPD: Learning More from Fewer Tokens in On-Policy Distillation，arXiv [2610.11659](https://arxiv.org/abs/2610.11659) v1，2026-10-08 提交。作者 Anhao Zhao、Haoran Xin、Junlong Tong、Yingqi Fan、Xuan Lu、Ping Nie、Wenjie Li、Xiaoyu Shen（通讯）；单位 EIT Ningbo、PolyU、HKUST(GZ)、SJTU、HKU、Waterloo。
- 会议：arXiv comments 字段为空，未见录用信息。结论：预印本。
- 与 PowerOPD 的作者重叠（已核实）：PowerOPD 是 arXiv [2606.17199](https://arxiv.org/abs/2606.17199)（2026-06-15），作者 Anhao Zhao、Junlong Tong、Yingqi Fan、Ping Nie、Wenjie Li、Xiaoyu Shen，6 人全部是 DIAL-OPD 作者；DIAL-OPD 只多了 Haoran Xin、Xuan Lu。PowerOPD 在 DIAL-OPD 里是 baseline（参考文献 [43]），代码也基于 PowerOPD 的 TRL fork，属于同一团队自比，读 baseline 数字时要注意。
- 官方实现：有。[EIT-NLP/DIAL-OPD](https://github.com/EIT-NLP/DIAL-OPD)，本地浅克隆 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/DIAL-OPD`，commit `11ac9a3187e8b0b2899405705bc269b78f86c4e2`（2026-10-28）。框架是 Hugging Face TRL（PowerOPD 维护的 pinned TRL 源码包，实验性 `GOLDTrainer`，由 `scripts/install_trl.py` 安装），不是 verl；论文 HTML 也引用了 slime [47]，但训练代码是 TRL。运行时 Python 3.10、torch 2.10、vLLM 0.19.1、transformers 5.14.1，单卡 vLLM colocate（`docs/RUNNING.md:58`）。
- 本仓实现：未实现（计划注册名 `dial_opd`）
- 状态：调研中
- 结论（一句话）：DIAL-OPD 在 sampled-token reverse-KL OPD 上按 s_β = |r|^{1−β}·|q−p|^β 在每条 response 内保留前 ρ 的 token，只需 teacher 采样 token 的 logprob，在 verl 里是「一个 token mask + 改分母」，实现成本低。

## 方法要点

标准 sampled-token OPD 的每 token 奖励是 r_t = log(q_t/p_t)（q 为 teacher、p 为 student 在采样 token 上的概率），损失是 −sg[r_t]·log π_θ。论文的观察是：按 |r| 选 token 会偏向「两边概率都很低」的 token（low-low，论文定义为 p、q 都 ≤ 1%），这些 token 的 |log 比值| 大但概率差很小。论文统计：在 10/15/20% 保留率下，low-low token 只占所选 token 的 20.59/13.49/10.26%，却贡献了 49.80/41.43/38.01% 的绝对奖励。

- 对数平均（Eq 3）：L(p,q) = ∫₀¹ p^{1−u} q^u du = (q−p)/(log q − log p)，即把 log 比值换算回概率差的尺度因子。
- 打分（Eq 4/5）：s_β = L^β·|r| = |r|^{1−β}·|q−p|^β。β=0 退化为按 |r| 选；β=1 等价于按 |q−p| 选；β>1 进一步偏向高概率 token。
- 选择（Eq 8）：每条 response i 内取 K_i = max(1, round(ρ·|V_i|)) 个分数最高的 token，V_i 是该 response 的有效 token（排除 prompt、padding、末尾 EOS）。选择在 response 内做，不跨 response 竞争。
- 损失（Eq 9）：L = −(1/M_B)·Σ m_{i,t}·sg[r_{i,t}]·log π_θ(y_{i,t})，M_B 是 batch 内被选中 token 总数。分母是选中 token 数，不是全部 token 数；未选中 token 仍留在上下文里，只是不进损失。
- 奖励本身不做变换，仍是原始有符号 log 比值；DIAL 只改「哪些 token 进损失」。

代码实现：

- `selection.py:62` `select_score_tokens(student_logp, teacher_logp, effective_mask, keep_ratio=0.2, score_beta=0.5, probability_threshold=0.01)`：`_log_scores`（:25）在 float64 的 log 空间算分；β=0 时按原始 |r| 排序，β=1 时按 log|q−p| 排序。每行 `k = max(1, int(round(ratio * n_valid)))`（:127），稳定 argsort 取前 k。`probability_threshold` 只用于 low-low 诊断统计，不影响选择（:82 docstring）。注意函数签名默认 β=0.5，但训练入口会传入 config 的 `score_beta`。
- `trainer.py:138` `_build_reward_components`：student、teacher 都在 FP32 下对完整 logits 做 log_softmax 后 gather 采样 token，`reward_full = teacher_logp - student_logp`；`mask_terminal_eos` 去掉末尾 EOS。
- `trainer.py:175` `_cache_window_rewards`：对整个 optimizer window（梯度累积窗口）先做无梯度的 student 前向 + teacher 前向，构造 mask，并缓存 `denominator = selected_count`（:210）。
- `trainer.py:218` `_compute_loss_from_window_cache`：`loss = -(reward_full[mask] * logp[mask]).sum() / denominator`（:242），再乘 `ga_steps`（:243–244）抵消 HF Trainer 对梯度累积的平均，所以等效分母就是整个 32 条 response 窗口的选中 token 数，与 Eq 9 一致。
- `config.py:7` `DIALConfig(GOLDConfig)`：强制 `beta=1`、`lmbda=1`（全部 on-policy）、`num_generations=1`、不开 ULD / seq_kd / liger（:47–49）。
- `train.py:50–51`：student 以 float32 加载、teacher 以 bfloat16 加载；`train.py:88` 把 `probability_threshold` 固定为 0.01。

## 适用条件

- 同 tokenizer：必须（直接对同一 token id 取 teacher logprob）。
- teacher 信息：只需要 teacher 在 student 采样 token 上的 logprob（log q_t）。官方代码跑完整 teacher 前向再 gather，但算法本身不需要 Top-k 或 full-vocab。
- student 信息：只需采样 token 的 logprob。
- 任务奖励：不使用，纯蒸馏信号。
- 规模：论文只在 0.6B/1.7B student、4B/8B teacher 上验证，另有 Sec 3.1 用 Qwen3-4B-Base 作 student 的同尺寸检查（Fig. 2）；更大规模未验证。

## 论文的实验设置

来自论文 Sec 5 与附录 B：

- 模型：student Qwen3-0.6B-Base、Qwen3-1.7B-Base；teacher Qwen3-4B、Qwen3-8B（instruct）。
- 数据：DeepScaleR（官方数据文件 40,313 条，`docs/RUNNING.md:42`）。
- 训练：800 step，AdamW，BF16 混合精度，lr 5e-7，batch 32，每条 rollout 最多 1024 response token，rollout temperature 1.0、top-p 0.95、top-k 20。论文未写硬件和 seed 数。
- 评测：GSM8K、MATH-500、AMC23、AIME24、AIME25、Minerva Math、OlympiadBench；temperature 0.6、top-p 0.95；GSM8K/MATH-500/Olympiad 采 4 条、Minerva 8 条、AMC23/AIME24/AIME25 16 条。表格报 Avg（平均正确率）、Pass（至少一条正确）、Len（平均长度）；Mean 是 7 个 benchmark Avg 的不加权平均。主评测的最大生成长度论文没写，但表中 Len 有 8192 的封顶值（如 Table 1 CC-OPD 的 AIME24/25），推测上限是 8192。
- β 扫描：β ∈ {0, 0.25, …, 2}，40% 保留率。0.6B student 的 MATH-500：β=0 为 35.2、β=1 为 44.6、β=1.2 为 44.8；1.7B student 在 β=1 达到峰值 61.8。

同保留比例下的 random 对照：有。Table 1、2 在每个保留率（20%、40%）都列了 uniform Random selection；Sec 3.1（Fig. 1）还做了监督密度实验：random 在 40–90% 保留率下都比全 token OPD 好，80% 时 MATH-500 为 38.4 vs 30.2（附录 B.1：每步重新均匀采样 K = max(1, round(ρN)) 个位置）。

主要数字（Mean，单位 %，来自论文 Table 1 / Table 2）：

| 设置 | Vanilla OPD（100%） | PowerOPD（100%） | Random | CC-OPD | DIAL β=1 | DIAL β=1.2 |
|---|---|---|---|---|---|---|
| 4B → 0.6B，20% | 19.4 | 23.8 | 9.4 | 12.8 | 19.9 | 20.9 |
| 4B → 0.6B，40% | 19.4 | 23.8 | 17.5 | 22.7 | 24.7 | 23.8 |
| 4B → 1.7B，20% | 32.1 | 34.1 | 10.6 | 28.9 | 34.1 | – |
| 4B → 1.7B，40% | 32.1 | 34.1 | 27.5 | 33.9 | 35.4 | 34.9 |
| 8B → 0.6B，40% | 20.0 | 23.7 | 19.1 | – | 24.6 | – |
| 8B → 1.7B，20% | 32.9 | 33.5 | 27.3 | – | 33.8 | – |
| 8B → 1.7B，40% | 32.9 | 33.5 | 31.3 | – | 34.6 | 34.1 |

- 「–」表示本文档未摘录该格，不代表论文没有。Table 1 中 4B → 0.6B 的 ExOPD 为 17.9。
- Table 2：1.7B student、40%、β=1.2 时 AIME25 Pass@16 为 26.7，Vanilla OPD 为 13.3。
- Table 3 是 β=0 vs β=1 的消融。
- 读数注意：DIAL 相对 PowerOPD 的提升大多在 1 个百分点左右（例如 4B → 0.6B 40%：24.7 vs 23.8），论文没有 seed 数和误差棒；AIME 每年 30 题、AMC23 40 题（公开 benchmark 规模，非论文给出）。Random 在 20% 保留率下长度明显变长（Table 1 中 0.6B Random 的 GSM8K Len 1880，Vanilla 506），说明低保留率下 random 的退化部分来自长度失控。

## 与本仓实现的差异

本仓尚未实现。官方实现与论文之间的差异：

- 默认保留率：`configs/train_qwen3_06b.json` 用 `keep_ratio=0.2`、`score_beta=1.0`；论文多数亮点数字在 40%。`configs/smoke_vllm.json` 是 0.4 / 1.2。
- 训练配置（`train_qwen3_06b.json`）：0.6B-Base / 4B，per-device batch 1 × 梯度累积 32，800 step，lr 5e-7 constant，`max_completion_length=1024`，temperature 1.0 / top-p 0.95 / top-k 20，vLLM colocate，seed 42。与论文一致。
- 精度：论文写 BF16 mixed precision；代码 student 以 float32 加载、teacher bfloat16（`train.py:50–51`），配置开 `bf16: true`，即 FP32 master 权重 + BF16 autocast。
- 评测：仓库只有训练期 MATH 监控（默认完整 MATH test，`eval_max_new_tokens=4096`，`eval_n=1`，`docs/RUNNING.md:74–75` 提示需要自己换成 MATH-500 子集）；没有复现 7 个 benchmark 主表的评测脚本。4096 对应附录 B.1 监督密度实验的设置，不是主表设置。
- baseline 改造：附录 B.2 说明 TA-OPD、SA-OPD、Prefix OPD 都被改成「每条 response 内匹配保留率」；CC-OPD 因拿不到 teacher 的 RL 前 checkpoint，用同家族更小的 instruct 模型作 reference。这些 baseline 不是原论文设置，TA-OPD 在 20% 保留率下 Mean 只有 2.7（Table 1，0.6B），明显异常偏低。
- 舍入：代码用 Python `round()`（银行家舍入），论文写 round，边界情况相差一个 token，可忽略。

在本仓（同 tokenizer teacher）实现的成本：低。

- 所需信号与现有 sampled-token OPD 相同：student 与 teacher 在采样 token 上的 logprob。EasyOPD 的同 tokenizer 方法可参考 `easyopd/methods/gkd/`（teacher 前向）和最小示例 `echo_kd`。
- 新增逻辑只有两处：在算损失前，按 s_β 对每条 response 求 top-ρ mask（约 30 行，`select_score_tokens` 可近乎原样移植）；把损失聚合分母改成 batch（或 mini-batch）内选中 token 数。verl 的 `token-mean` 聚合在把 mask 乘进 response_mask 后，天然就是除以选中 token 数，只要 mask 在聚合前生效即可。
- 注意 verl 的 mini-batch / micro-batch 切分：官方分母是整个梯度累积窗口的选中数，verl 的 `token-mean` 按 mini-batch 全局 token 数归一，二者在一个 mini-batch = 一个窗口时一致。
- 生效证据：日志里打出每步选中比例、low-low 占比（官方在 `selection.py:131–136` 统计 `low_low_valid` / `low_low_selected`），确认 mask 真的进了损失。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| 4B → 0.6B，40%，β=1，Mean（Table 1） | 24.7 | 未跑 | [repro/](../repro/) |
| 4B → 1.7B，40%，β=1，Mean（Table 1） | 35.4 | 未跑 | [repro/](../repro/) |
| 4B → 0.6B，β 扫描 MATH-500，β=1（40%） | 44.6 | 未跑 | [repro/](../repro/) |

## 待解决

- 主表评测最大长度需向作者确认或按 8192 处理；需要自写 7 个 benchmark 的评测（复用 `experiments/_shared/scripts/`）。
- 论文没有 seed 数，复现时至少跑 2 个 seed 才能判断 ~1pp 的差距。
- 在本仓与 sampled-token OPD 基线用同一数据和步数对照，同时加同保留率 random 对照（论文自己的 random 在 20% 下崩得很厉害，值得复核）。

## 来源

- https://export.arxiv.org/abs/2610.11659（标题、作者、日期、comments）
- https://arxiv.org/html/2610.11659v1（正文、公式、Table 1–3、附录 B）
- https://export.arxiv.org/abs/2606.17199（PowerOPD 作者列表与日期）
- https://github.com/EIT-NLP/DIAL-OPD（浅克隆 commit `11ac9a3`，README、docs/RUNNING.md、configs、代码）
