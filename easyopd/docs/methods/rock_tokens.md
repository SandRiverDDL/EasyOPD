# Rock Tokens / Window-Aware Reweighting（`rock_tokens`）

涉及代码：`oss_refs/Rock-Token/stumbling/kdflow/algorithms/token_freeze_kd.py`、`oss_refs/Rock-Token/stumbling/run_stumb_random.sh`、`oss_refs/Rock-Token/stumbling/{rock,random}.json`、`oss_refs/Rock-Token/rock_detection/`（均在 `/mnt/NLU_disk17/chenhaoyu02/` 下）。

- 论文：Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation，arXiv [2605.09253](https://arxiv.org/abs/2605.09253)。作者 Yuxuan Jiang（UMBC）、Runchao Li（CWRU）、Shubhashis Roy Dipta（UMBC）、Dawei Li（ASU）、Zhao Yang（VU Amsterdam，通讯）。v1 2026-05-10，v2 05-29，v3 06-29，v4 07-01。
- 会议：未核实。arXiv v4 PDF 页脚是「Preprint.」，comments 字段为空；「NeurIPS 2026」只出现在仓库 README 的 badge（「accepted to NeurIPS 2026」），README 里的引用仍是 arXiv 预印本；OpenReview / neurips.cc 搜索无结果。目前只能算作者自述。
- 官方实现：有。[YuxuanJiang1/Rock-Token](https://github.com/YuxuanJiang1/Rock-Token)，本地浅克隆 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/Rock-Token`，commit `ed88d18d68cdae78d4cf6c2cfafc3b481656f3cc`（2026-10-01）。框架是 KDFlow（songmzhang/KDFlow，arXiv 2603.01875：SGLang 跑 teacher、FSDP2 训 student、teacher 传 hidden states 再过 lm_head 得 full-vocab logits），不是 verl。USAGE.md 说明未发布 checkpoint、原始张量和训练 JSONL。
- 本仓实现：未实现（计划注册名 `rock_tokens`）
- 状态：调研中
- 结论（一句话）：这是一篇分析型论文，主张 OPD 中少数高频高损失的「rock token」几乎不贡献有效梯度，冻结它们可以省算力而精度基本不掉；它是效率取舍而不是精度提升，官方代码只实现了按 token ID 冻结，没有实现论文描述的窗口。

## 方法要点

设定：on-policy distillation 的 per-token reverse KL 目标（Eq 1），采样 token 上的损失 ℓ_t = log π_θ(x_t) − log π_T(x_t)（Eq 2）。论文流程分三步。

- 识别 rock token（Eq 3–4、Eq 10–14）。对词表中每个 token v，Rock Score R(v) = mean(ℓ_v) · Freq(v)，即平均损失乘出现频率，取 Top-K（主实验 K=100）。论文还定义「pre/post 都高损失」的持久集合 O_PH（Eq 5，阈值 τ_pre、τ_post），窗口上下文 c = x_{t−w:t+w} 及其表示 f_ctx 和相似度 sim（Eq 6–8），带 γ 的 ρ(o)（Eq 10）、带 η 的 R(v)（Eq 11），上下文一致率 CCR（Eq 12），并用 R_ctx = R · CCR（Eq 13）配阈值 τ_R（Eq 14）筛选。论文没有给出 w、f_ctx 的具体形式、γ、η、τ 的取值。
- 分析（Eq 15–17）。per-token full-KL 估计（Eq 15）；梯度贡献 contrib(t) = n_t · ‖ḡ_t‖ · cos(ḡ_t, G_balanced)（Eq 16）；单 token knockout 的影响 Δ_B(v)（Eq 17）。
- 窗口感知重加权（Eq 18–19）。W_R = ∪_v W(v) 是 rock token 周围窗口的并集；L_weighted = E[Σ_t w(x_t, t) · ℓ_t]，若 x_t ∈ R 或 t ∈ W_R 则 w = λ，否则 w = 1。三种设置：Baseline（λ=1）、Rock-Freeze（λ=0）、频率匹配的随机窗口冻结（对照）。

可单独消融的组件：(1) 识别打分方式（R vs R_ctx）；(2) K 的大小；(3) 只冻 token 本身 vs 冻 token 加窗口；(4) λ 取值（论文只做了 0）；(5) rock 列表 vs 频率匹配的随机列表。

论文报告的分析结论（数字均来自 v4 正文与附录）：

- rock token 密度中位数 18%（§3；Fig 2 图注写 18.5%）。K=100（约占出现过的词表的 2.5%）覆盖约 60% 的 KL；K=50 时 Jaccard ≈ 0.81、覆盖 49%；K ≤ 100 时跨 checkpoint 的 Jaccard > 0.70，K ≥ 150 时下限约 0.57。
- 梯度：rock token 的梯度范数中位数 0.016，稀有高 KL token 为 0.54；与总梯度的 cos 为 0.040，对比高 KL token 0.025、随机 token 0.006。稀有高 KL token 的 KL 在训练中从 2.02 降到 0.85。
- knockout：「支柱」token（去掉会伤精度的）在 MATH-500 上占 3.5%，在 IFEval 上占 1.5%（附录 Table 2：MATH-500 为 7/193/0，baseline 准确率 0.750；IFEval 为 3/197/0，baseline 0.752）。附录 Table 1 中相关性 |r| < 0.075，p > 0.30。
- 加速：§4.3 写冻结约 18% 的 token 带来 1.7× wall-clock 加速，而 takeaway 框里写「30% high-cost tokens」带来 1.4×，论文内部前后不一致。

## 适用条件

- 是否 on-policy：主实验的第二阶段是 on-policy（student rollout，teacher 在 student 序列上给 full-vocab 分布，pure reverse KL）。但 rock 列表是离线从 OPD 之后的 checkpoint 统计得到的，训练时作为静态 mask 使用；第一阶段是离线 KD（teacher 回答上的 forward KL + CE）。
- 同 tokenizer：必须。rock 列表是 Qwen3 词表的 token ID（`stumbling/rock.json`，100 个，前几个是 3070、279、400、220、11、1447），换 tokenizer 就要重新统计。
- teacher 信息：论文和代码的损失是 full-vocab RKL，Σ_v p_s(v)(log p_s(v) − log p_t(v))，需要 teacher 全词表 logits（KDFlow 传 teacher hidden states，在 student 侧过 teacher lm_head）。如果改成采样 token 的 RKL（ℓ_t = log π_θ − log π_T），只要 teacher 采样 token logprob 即可，但这已经偏离论文。识别阶段（`rock_detection/rock.py`）也用 full-vocab reverse KL 累计。
- 任务奖励：训练不使用。
- 规模：只验证了 30B-A3B → 4B 一对、只在数学上（论文 Limitations 自述），并说明 mask 是二值的。

## 论文的实验设置

以下来自论文 §5 与附录 C（v4）。

| 项 | 第一阶段（离线 KD） | 第二阶段（on-policy） |
|---|---|---|
| 模型 | teacher Qwen3-30B-A3B-Instruct-2507，student Qwen3-4B-Instruct-2507，关闭 thinking | 同左 |
| 数据 | OpenThoughts3 上 20k 条 teacher 回答（temperature 0.6、top_p 0.95、max 16384） | 另取 10k prompts，每个 4 条 rollout（temperature 1.0、top_p 1.0、generate_max_len 8000、prompt 800） |
| 损失 | forward KL，kd_ratio 0.5 + CE | pure reverse KL，kd_ratio 1.0 |
| 优化 | lr 2e-5，warmup 5%，batch 128，1 epoch | lr 2e-6，warmup 5%，grad clip 1.0，全局 batch 4，1 epoch |
| 并行 | 4 × H100 80GB，FSDP2，bf16 | rollout TP=2，teacher TP=4 |

评测：lm-eval-harness zero-shot，Pass@1 取 5 次平均，数据集 AIME24、AIME25、HMMT25-Feb（共 90 题），另有 MATH-500 与 IFEval。

主结果（Fig 5）只有曲线，没有数值表：纵轴是 AIME24、AIME25、HMMT25 的平均准确率，横轴是训练 step（200 step = 8,000 prompts × 4 rollouts；仓库 `assets/selective-distillation.png` 的横轴画到约 1500）。从图上看「Ours」（Rock-Freeze）在大多数点上低于 Original OPD，明显高于频率匹配的随机冻结；目测终点约为 Original OPD 57、Ours 55、Random 42（图上读数，不是论文给的数字，误差约 ±1）。

## 与本仓实现的差异

本仓尚未实现。先列官方代码与论文的差异：

- 没有窗口。`TokenFreezeKD.training_step`（`token_freeze_kd.py:36-117`）只做 `torch.isin(student_token_ids, freeze_ids)`（L90）按 token ID 冻结，然后 `weights[freeze_mask] = freeze_weight`（L93）。仓库里搜不到窗口逻辑，Eq 18 的 W_R 没有实现。
- 疑似错位：冻结判定用的是 `student_input_ids[student_loss_mask]`（L86），而同一文件的 CE 标签是 `student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]`（L111）。这说明 L86 取到的是位置 t 的输入 token，而不是该位置要预测的下一个 token，即冻结的是「rock token 之后那一步的损失」。这是根据代码推断的，未跑数验证。
- 归一化不同：开启冻结时 `kd_loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)`（L95），不冻结时是 `token_loss.sum() / avg_token_num`（L103）。冻结组和 baseline 的 loss 尺度、等效学习率不同。
- 启动脚本 `stumbling/run_stumb_random.sh` 与论文不符：2 卡，`enable_thinking True`（论文关闭），`generate_max_len 1024`（论文 8000），max_len 2048；其余为 batch 4、lr 2e-6、warmup 0.05、n_samples 4、`kd_algorithm token_freeze_kd`、`freeze_weight 0.0`、`kd_loss_fn rkl`。默认列表是 `random.json`，跑 rock 版要设 `TOKEN_FREEZE_PATH=rock.json`。另一个 `KDFlow_localopd/run_localopd.sh`（vanilla_kd rkl、n_samples 1、generate_max_len 10000）的超参也不同，USAGE.md 承认这一点。
- 检测脚本 `rock_detection/rock.py` 依赖私有 student checkpoint `RockToken/qwen3_30b_a3b_to_4b_onpolicy_math_following5k`，teacher 用 4-bit 量化，只在 100 条 MATH-500 上 greedy 生成 256 token 统计；`rock_detection/test_stability.py:49` 的 `score_freq_weighted` 就是 n × mean。
- 评测脚本（`evaluation/`）用 vLLM TP2、3 个 seed（59、76、93）、temperature 0.6、max_gen 16384、取 3 次平均；论文写的是 5 次平均。

接入 EasyOPD/verl 的成本（假设同 tokenizer、teacher 提供 logprob）：

- 静态 token mask：低。读入 token ID 列表，在 response 上按「被预测的 token」（即 response 里的 token 本身，verl 的 `responses` 已经是预测目标，不存在 KDFlow 的错位问题）生成权重 w，乘到 OPD 的 per-token advantage 或 loss 上，并决定分母用 Σw 还是原 token 数。几十行代码，可作为 `gkd` 或 OPD 路径上的一个 per-token 权重 hook。
- full-vocab RKL：中到高。论文的损失需要 teacher 全词表分布，verl 的 ref 通路默认只给采样 token logprob；要么复用本仓已有的 full-vocab / top-k teacher 通路（先查 `gkd`），要么接受采样 token RKL 这一偏离并在报告里写明。
- 窗口与在线识别：中。论文没给 w、f_ctx、γ、η、τ，只能自己定；在线统计 R(v) 需要跨 step 累计每个 token 的频率与平均损失。真正的加速还需要把冻结 token 从 teacher 前向或 backward 中剔除，单纯乘 0 权重不省算力。
- rock 列表本身：官方 `rock.json` 是在作者私有 checkpoint 上统计的，换 student/teacher 或数据就应在本地 OPD checkpoint 上重新统计。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| 4B-Instruct ← 30B-A3B，Rock-Freeze vs OPD vs Random，AIME24/25/HMMT25 平均 | 只有 Fig 5 曲线，无数值 | 未跑 | [repro/](../repro/) |
| 加速比 | 1.7×（§4.3）或 1.4×（takeaway），自相矛盾 | 未跑 | [repro/](../repro/) |

## 待解决

- 会议未核实，等官方列表或作者在 arXiv 更新 comments 后再定。
- 加速比 1.7× 与 1.4×、冻结比例 18% 与 30% 的矛盾，需要问作者或自己测。
- 确认 KDFlow 里 L86 的错位是否真实（可在 CPU 上构造一条序列打印 freeze 命中的位置）；如果真实，官方曲线对应的是「冻结 rock token 下一步」而不是论文描述的设置。
- 冻结组与 baseline 的分母不同（Σw vs token 数），本地实现要么统一分母，要么把这一点作为消融。
- 论文主张是「省算力、精度基本不掉」，主结果里 Ours 低于 Original OPD；本地对照应同时报精度和真实 wall-clock，而不是只看精度。
- 窗口参数缺失，实现时需要自己设并写明。

## 来源

- https://arxiv.org/abs/2605.09253
- https://arxiv.org/html/2605.09253v4
- https://arxiv.org/pdf/2605.09253v4
- https://github.com/YuxuanJiang1/Rock-Token（本地克隆 commit `ed88d18`，读了 README、USAGE.md、`stumbling/kdflow/algorithms/token_freeze_kd.py`、`stumbling/run_stumb_random.sh`、`stumbling/*.json`、`KDFlow_localopd/run_localopd.sh`、`rock_detection/rock.py`、`rock_detection/test_stability.py`、`evaluation/`、`assets/selective-distillation.png`）
- OpenReview / neurips.cc：只做了搜索，未找到条目。
