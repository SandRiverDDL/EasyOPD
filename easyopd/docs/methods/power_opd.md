# PowerOPD：Box-Cox 有界 reward 的 OPD（`power_opd`）

> 文件名：`methods/power_opd.md`。一个方法一篇，常青：理解或实现变了就改，不按日期追加。
> 读之前先看 [INDEX](../INDEX.md)；时序细节放 `repro/` 和 `experiments/`，这里只放链接。

- 论文：arXiv [2606.17199](https://arxiv.org/abs/2606.17199)（v1 2026-06-15），PowerOPD: Stabilizing On-Policy Distillation with Bounded Power Transformation，作者 Anhao Zhao, Junlong Tong, Yingqi Fan, Ping Nie, Wenjie Li, Xiaoyu Shen（作者与标题取自 arXiv 元数据页的 citation 字段）。
- 会议：EMNLP 2026 main，已核实。ACL Anthology 条目 `2026.emnlp-main.2129`，pages 48414–48427。
- 官方实现：<https://github.com/EIT-NLP/PowerOPD>，commit `65819e14a49c940fa7b2977f07872cabd6229129`（2026-08-14，"Rename run_power_diff -> run_poweropd, update README"）。基于仓库内置的 TRL checkout（`trl/VERSION` 为 1.1.0.dev0），继承链是 `GOLDTrainer` → `FactorizedGOLDTrainer` → `OPDDynamicsTrainer`，不是 verl。本地浅克隆在 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/PowerOPD`。
- 本仓实现：未实现（计划注册名 `power_opd`）
- 状态：调研中
- 结论（一句话）：把 sampled-token OPD 的 log-ratio reward 换成有界的 π_T(y_t)^α − π_θ(y_t)^α（取值在 [−1, 1]）；论文报告 4 组 teacher/student 上 Mean Avg@8 比 vanilla OPD 平均高 4.47（论文 §6.1，按每个指标各自取最优 α 计），且 1.7B←4B 时 α=500 的 Mean Avg@8 为 36.02，高于 full-vocab OPD 的 34.16（Table 3）。

## 方法要点

vanilla OPD 的逐 token reward 是 r_t = log(π_T(y_t) / π_θ(y_t))（论文 Eq. 1）。当 student 给采样 token 的概率很小时，log π_θ 趋向 −∞，reward 无界，梯度范数出现尖峰。论文把 reward 写成一般形式 r = f(π_T, π_θ)（Eq. 2），并提出 reward 应满足两条性质（§4.2）：

- P1（有界）：reward 的取值有上下界。
- P2（符号一致）：reward 的符号与 log(π_T / π_θ) 一致，即 teacher 概率更高时 reward 为正。

论文用 Box-Cox 变换 h_α(x) = (x^α − 1) / α（Eq. 3）分别作用在 π_T 和 π_θ 上再相减，并把 1/α 并入学习率，得到（Eq. 4）：

r_t^α = sg[ π_T(y_t)^α − π_θ(y_t)^α ]

其中 sg 表示 stop-gradient。概率在 [0, 1] 内，所以 r_t^α ∈ [−1, 1]，满足 P1；x^α 单调递增，所以满足 P2。α → 0 时 h_α 退化为 log，回到 vanilla OPD。

论文比较的其他 reward 变换：Clip（把 log-ratio 截断到 ±1，阈值经网格搜索选出）、Tanh、Z-Score（按批次标准化）。论文指出 Z-Score 会改变 reward 符号，违反 P2（§4.2）。另一个基线是对全词表 KL 直接求导的 Full-vocab KL OPD（论文引用 Zhao et al., 2026a）。

与其他方法的区别：vOPD 是给 log-ratio reward 减一个 baseline 来降方差，不改变 reward 的期望；PowerOPD 直接换掉 reward 函数，优化目标不再是 reverse KL，所以梯度相对 reverse KL 是有偏的。这一点论文没有讨论无偏性，是按公式推断。

### 官方代码中的对应位置（commit `65819e1`）

| 作用 | 位置 |
|---|---|
| `OPDDynamicsTrainer(FactorizedGOLDTrainer)` | `opd_dynamics/trainer.py:15`；父类在 `factorized_gold/trainer.py:34` |
| `power_reward_alpha` 读取，getattr 默认 0.1 | `opd_dynamics/trainer.py:24` |
| student 和 teacher 都做 float32 全词表 log_softmax，再 gather 采样 token | `opd_dynamics/trainer.py:109-114` |
| 各种 reward：log_ratio（:116）、clamp（:117）、tanh(log_ratio / T)（:118）、prob_diff（:119）、sqrt_diff（:120） | `opd_dynamics/trainer.py:116-120` |
| `power_diff_full = exp(α·teacher_logp) − exp(α·student_logp)`，没有 1/α | `opd_dynamics/trainer.py:121-123` |
| 按 `reward_mode` 选 reward | `opd_dynamics/trainer.py:65-77` `_select_reward_full` |
| 用 `window_token_mean` 或 `reward_normalization != none` 时启用窗口缓存 | `opd_dynamics/trainer.py:33-34` |
| 窗口预计算：在 no_grad + eval 模式下，用 GA 窗口开始时的 student 对整个 buffer 前向 student 和 teacher 算 reward，再对整个窗口做 zscore 或 sign_max_abs 归一化 | `opd_dynamics/trainer.py:181-184`、`:195-360` |
| 窗口路径的 loss：−Σ(reward · logp) / window_token_count × ga_steps | `opd_dynamics/trainer.py:362-404` |
| 非窗口路径的 loss：−mean(reward · logp) | `opd_dynamics/trainer.py:406-488` |
| config：α 默认 0.1，要求 α > 0；`token_loss_normalization` 默认 microbatch_mean；`reward_normalization` 默认 none；`terminal_stop_target` 默认 im_end | `opd_dynamics/config.py:33-36`、`:102-103` |
| 终止 token 处理：`mask` 把最后一个 `<\|endoftext\|>` / `<\|im_end\|>` 从 loss 中去掉；`im_end` 把它改写成 `<\|im_end\|>`；`cross_eos` 只改 teacher 侧 label | `opd_dynamics/terminal_stop.py` |

loss 是纯 REINFORCE 形式，没有 PPO 的 ratio 和 clip。窗口路径下每个样本的 student 要前向两次：一次在预计算 reward 时（no_grad），一次在求梯度时。

## 适用条件

PowerOPD 只适用于同 tokenizer：reward 只需要 teacher 在 student 采样 token 上的 log-prob，按 token id 直接对齐。teacher 需要能给出逐 token log-prob（白盒或能返回 prompt logprob 的服务都行），不需要 top-k 或全词表分布。PowerOPD 不使用 task reward，reward 完全由 teacher 和 student 的概率决定。论文只在数学任务和 Qwen3 系列上做了实验（Limitations 一节自述）。

## 论文的实验设置

来源是论文 §5 与附录 B；右列是官方 `scripts/common.sh` 与 `scripts/run_poweropd.sh` 的默认值，与论文不同或论文没写的地方标出。

| 项 | 论文 | 官方脚本 |
|---|---|---|
| teacher → student | 4 组：Qwen3-4B / Qwen3-8B → Qwen3-0.6B-Base / Qwen3-1.7B-Base | 默认 Qwen3-4B → Qwen3-1.7B-Base |
| 训练数据 | DeepScaleR | `data/train/deepscaler_conversation.jsonl` |
| 步数 | 1.5k 步 | `MAX_STEPS=1500` |
| batch | 32（附录 B：B = 32，prompt 128，T = 1024） | per-device 1 × GA 32 |
| lr | 5e-7 | 5e-7，constant scheduler，warmup 0.05 |
| 精度 | bf16 | `BF16=True`，但 `STUDENT_DTYPE=float32`（`common.sh:18`），teacher bf16 |
| rollout | 最长 1024 | `MAX_COMPLETION=1024`；温度 1.0、top_p 0.95、top_k 20（`common.sh:173-175`），论文未写 |
| α | 0.1–500 扫描 | `run_poweropd.sh` 设 `POWER_REWARD_ALPHA=5`；`common.sh:56` 与 config 默认 0.1 |
| loss 归一化 | 未写 | `run_poweropd.sh` 设 `TOKEN_LOSS_NORMALIZATION=window_token_mean`、`TERMINAL_STOP_TARGET=mask` |
| Clip 基线阈值 | ±1（网格搜索） | `common.sh:53-54` 默认 ±5 |
| 评测集 | GSM8K、MATH500、AMC23、AIME24、AIME25、Minerva、OlympiadBench | 训练中评测 `data/eval/math_test.jsonl`，每 30 步、100 条 |
| 评测指标 | Avg@8 / Pass@8 / 平均长度，3 次运行取平均 | 训练中评测温度 0.6、n = 1、最大生成 4096 |
| 硬件 | 本次没有在论文中找到 | vLLM colocate，显存占比 0.5 |

论文的评测温度本次没有找到。`factorized_gold/eval_math.py` 的独立评测脚本默认温度 0.0、`max_new_tokens=5000`，与训练中评测的设置不同。

## 主要结果

论文 Table 2（student 为 Qwen3-0.6B-Base）与 Table 3（student 为 Qwen3-1.7B-Base），下面只列 7 个评测集的 Mean，每格为 Avg@8 / Pass@8 / 平均长度：

| 方法 | 0.6B ← 4B | 0.6B ← 8B | 1.7B ← 4B | 1.7B ← 8B |
|---|---|---|---|---|
| Full-vocab | 22.91 / 33.72 / 4092 | 24.49 / 32.47 / 4095 | 34.16 / 49.76 / 4021 | 33.23 / 49.12 / 4093 |
| Vanilla | 19.13 / 36.18 / 4084 | 21.18 / 37.75 / 2209 | 30.98 / 47.66 / 1975 | 31.85 / 49.57 / 1755 |
| Clip | 22.49 / 38.35 / 2343 | 24.12 / 39.77 / 2220 | 32.06 / 50.93 / 1986 | 32.68 / 51.67 / 1808 |
| Tanh | 21.06 / 35.17 / 2496 | 23.79 / 40.50 / 2203 | 33.94 / 50.26 / 1942 | 32.47 / 50.83 / 1807 |
| Z-Score | 18.25 / 36.00 / 2316 | 22.87 / 38.68 / 2102 | 16.95 / 41.71 / 2000 | 33.00 / 50.76 / 1838 |
| α = 0.1 | 24.16 / 39.61 / 2291 | 23.69 / 40.85 / 2157 | 33.89 / 51.26 / 1900 | 33.42 / 50.29 / 1835 |
| α = 0.5 | 24.60 / 40.54 / 2241 | 24.06 / 40.59 / 2127 | 表中无此行 | 33.37 / 52.10 / 1807 |
| α = 1 | 24.37 / 39.62 / 2243 | 23.53 / 40.15 / 2186 | 34.42 / 50.66 / 1910 | 32.91 / 50.60 / 1835 |
| α = 5 | 25.48 / 40.27 / 2185 | 24.60 / 41.21 / 2120 | 34.23 / 50.10 / 1918 | 33.82 / 50.52 / 1800 |
| α = 10 | 25.20 / 39.68 / 2172 | 23.96 / 41.26 / 2043 | 34.01 / 50.73 / 1888 | 33.86 / 50.60 / 1779 |
| α = 50 | 24.47 / 41.26 / 2088 | 24.64 / 40.74 / 1982 | 34.82 / 51.18 / 1770 | 表中无此行 |
| α = 100 | 24.64 / 40.86 / 2043 | 25.05 / 41.37 / 1958 | 34.31 / 50.93 / 1683 | 34.21 / 51.25 / 1681 |
| α = 500 | 25.50 / 41.89 / 1875 | 24.67 / 41.30 / 1859 | 36.02 / 52.05 / 1579 | 34.45 / 51.87 / 1567 |

「表中无此行」指 arXiv HTML v1 的 Table 3 里本来就没有这一行（逐行核对过 HTML 转出的文本），不是抄漏；PDF 版是否一致没有核对。

其他结果：

- §6.1 的提升：相对 vanilla，四组平均 Avg@8 +4.47、Pass@8 +4.06，单集最大是 AMC23（Avg +16.75，Pass +15.00）。相对 Clip 等 post-hoc 变换 +3.01 / +3.54。相对 full-vocab OPD，0.6B 的 Avg@8 高 +2.59 / +0.56（4B / 8B teacher），1.7B 高 +1.86 / +1.22；Pass@8 的差距更大（0.6B 为 +8.17 / +8.90）。这些数字是每个指标各自取最优 α 后算的，同一个 α 不一定在所有指标上都最优。
- full-vocab OPD 的回答长度几乎都顶到 4096（上表 Len 列），vanilla 在 0.6B ← 4B 时也是 4084。PowerOPD 的回答长度大体随 α 增大而缩短（例如 0.6B ← 4B 从 α = 0.1 的 2291 降到 α = 500 的 1875）。
- Table 4（0.6B ← 4B）：full-vocab OPD 每次更新 402.7 TFLOPs、每步 22.14s、峰值显存 78.99GiB；PowerOPD 为 346.6 TFLOPs、9.03s、60.72GiB，分别少 13.9%、59.2%、23.1%。峰值显存是在 batch 8 下测的。
- Fig. 6（梯度范数）：vanilla 初期尖峰约 10^3，后期仍高于 20；Clip、Tanh 高于 10；Z-Score 从 3 涨到 10 以上；PowerOPD 稳定在 0.25–0.35。
- Fig. 1 与 §3.1：1.7B ← 4B 时 MATH500 Avg@8，vanilla 54.93，full-vocab 63.12，α = 500 为 64.53。

证据边界：结果是 3 次运行的平均，但正文表格没有给方差。各 α 之间的 Mean Avg@8 差距多在 1 分以内（例如 0.6B ← 8B 的 23.53 到 25.05），α 的选择接近噪声量级。AIME24/25 每集 30 题，单格差异尤其不可靠。Z-Score 在 1.7B ← 4B 上只有 16.95，与 8B teacher 下的 33.00 差别很大，论文 §4.2 指出 Z-Score 可能违反 P2，但单组结果不足以排除训练崩溃等偶然因素。

## 与本仓实现的差异

本仓还没有实现，下面列的是官方代码与论文的差异、复现时的风险点，以及在本仓 verl 中复现要做的事。

官方代码与论文的差异和风险点：

- α 的默认值不一致：论文主表多数设置下 α = 500 最好（Table 2/3 的 Mean Avg@8），README 和 `run_poweropd.sh` 用 α = 5，`common.sh:56` 与 config 默认 0.1。复现以论文表格的 α 为准，并在 repro 报告里写明用的是哪个。
- 名为 vanilla 的脚本带了 z-score：`scripts/run_log_ratio.sh` 默认设 `REWARD_NORMALIZATION=zscore`，与 `run_log_zscore.sh` 的设置相同。所以用官方脚本跑出来的「vanilla」不是纯 log-ratio。论文表里的 Vanilla 用的是哪套设置无法从代码确认。
- Clip 基线的阈值：论文是 ±1，`common.sh:53-54` 默认 ±5。
- 1/α：论文 Eq. 4 说把 1/α 并入学习率，代码 `trainer.py:121-123` 直接用 π_T^α − π_θ^α，两者一致；但这意味着不同 α 的有效步长不同，论文用同一个 lr 5e-7 扫 α，α 的效果和有效学习率的效果是混在一起的（按公式推断）。
- 大 α 下的数值：α = 500 时，π_θ(y_t) = 0.9 对应 0.9^500 ≈ 1.3e-23，所以 reward 实际上只在两边概率都非常接近 1 的 token 上明显不为 0，其余 token 的 reward 接近 0。这是按公式推算的，论文没有讨论这一点。student 用 fp32（`common.sh:18`），在 bf16 下 exp 的下溢行为可能不同。
- 优化器形式：官方是纯 REINFORCE，一个 GA 窗口（32 条）只更新一次，reward 在窗口开始时用当时的 student 预计算（`trainer.py:195-360`），没有 PPO ratio 和 clip。
- 终止 token：`run_poweropd.sh` 用 `TERMINAL_STOP_TARGET=mask`，最后一个停止 token 不进 loss；论文没写这一点。
- 其他：README 的 News 日期写成 "[2025-06]"，与 arXiv 日期 2026-06 不符；`requirements` 只固定了 `vllm==0.19.0`。

在本仓 verl 中复现要做的事：

1. teacher 信息：只需要 teacher 在采样 token 上的 log-prob。本仓已有两条通路。一条是 G-OPD / SOD 的用法：把 ref 换成 teacher，`ray_trainer.py:2389-2392` 算出的 `batch["ref_log_prob"]` 就是 teacher 在采样 token 上的 log-prob（`ray_trainer.py:1645` 的 SOD 正是用 `ref_log_prob − old_log_probs` 当 OPD advantage）。另一条是 lightning_opd 的 `teacher_log_probs`，但它来自离线 parquet（`ray_trainer.py:2426-2431`），不适合 on-policy 训练。复现应走第一条。
2. advantage：在 `easyopd/methods/power_opd/` 注册一个 adv_est，计算 `(exp(α · ref_log_prob) − exp(α · old_log_probs)) * response_mask`，exp 前先转 fp32。结构参考 `easyopd/methods/lightning_opd/advantage_estimator.py:19-66`。
3. `verl/` 挂接：`ray_trainer.py:322-328` 的 `compute_advantage` 只在 adv_est 名为 `on_policy_distillation` 时才把 `old_log_probs` 和 `teacher_log_probs` 传进估计器，新估计器需要在这里加一个分支，透传 `old_log_probs` 与 `ref_log_prob`，用 `# [EasyOPD:POWER_OPD]` 包住，并登记到 `verl_modified_files` 和 `easyopd/docs/verl-touchpoints.md`。
4. loss：直接用 verl 原生 PPO loss。要与官方的单次 REINFORCE 对齐，就设 `ppo_epochs=1`、`ppo_mini_batch_size` 等于 train batch（每批 rollout 只更新一次，此时 ratio 恒为 1，clip 不起作用），`loss_agg_mode=token-mean`（对应 `window_token_mean`）；关闭 `use_kl_in_reward` 和 actor 的 KL loss，否则 ref（即 teacher）会被当成 KL 正则。
5. 需要在 repro 报告里写明的差异：终止 token 是否屏蔽（verl 的 `response_mask` 默认包含 EOS）；student 精度（verl FSDP 常用 bf16 计算，官方 student 是 fp32）；rollout 采样参数（官方 top_p 0.95、top_k 20）。
6. 生效证据：日志里记录 reward 的均值、绝对值最大值（应不超过 1）、非零 token 占比和梯度范数；论文 Fig. 6 报告 PowerOPD 的梯度范数稳定在 0.25–0.35，可作为判据之一。

成本估计为低。理由是 teacher 在采样 token 上的 log-prob 本仓已有现成通路（ref 换 teacher），数学部分是一行公式；需要改 `verl/` 的只有 `compute_advantage` 的参数透传一处。主要工作量在对齐官方的优化器设置和终止 token 处理，而不是写代码。以上判断基于读到的代码，没有实际跑通。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| Qwen3-4B → Qwen3-1.7B-Base，DeepScaleR 1.5k 步，α = 500，Mean Avg@8（Table 3） | 36.02 | 未跑 | — |
| 同上，Vanilla，Mean Avg@8（Table 3） | 30.98 | 未跑 | — |
| 同上，α = 500，MATH500 Avg@8（Table 3） | 64.53 | 未跑 | — |

## 待解决

- Table 3 中 1.7B ← 4B 的 α = 0.5 和 1.7B ← 8B 的 α = 50 两行在 HTML v1 中不存在，需要对照 PDF 或 ACL Anthology 版本确认。
- 论文 Vanilla 基线是否带了 z-score（见上文 `run_log_ratio.sh`），需要问作者或对比 issue。
- 论文的评测温度、top-p 和硬件本次没有找到；`eval_math.py` 默认温度 0.0，训练中评测温度 0.6，正式评测用哪个未知。
- 本仓是否已有 DeepScaleR 数据，见 `reference/environment.md`。
- 正文表格没有方差，各 α 之间不到 1 分的差距是否显著无法判断。

## 来源

以下是本次调研实际打开过的页面：

- arXiv 元数据页：<https://export.arxiv.org/abs/2606.17199>
- 论文全文 HTML：<https://arxiv.org/html/2606.17199v1>
- ACL Anthology 条目：<https://aclanthology.org/2026.emnlp-main.2129/>
- 官方仓库：<https://github.com/EIT-NLP/PowerOPD>（浅克隆，读了 `README.md`、`opd_dynamics/`、`factorized_gold/trainer.py`、`factorized_gold/eval_math.py`、`scripts/common.sh`、`scripts/run_*.sh`）
