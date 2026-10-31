# AOPD：非对称 on-policy distillation（`aopd`）

> 文件名：`methods/aopd.md`。一个方法一篇，常青：理解或实现变了就改，不按日期追加。
> 读之前先看 [INDEX](../INDEX.md)；时序细节放 `repro/` 和 `experiments/`，这里只放链接。

- 论文：Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level，arXiv [2605.06387](https://arxiv.org/abs/2605.06387)（v1 2026-05-07，v2 05-08，v3 05-13，v4 2026-08-30），作者 Nan Jia, Haojin Yang, Xing Ma, Jiesong Lian, Shuailiang Zhang, Weipeng Zhang, Ke Zeng, Xunliang Cai, Zequn Sun。会议：Findings of EMNLP 2026，已核实（ACL Anthology `2026.findings-emnlp.238`，"Findings of the Association for Computational Linguistics: EMNLP 2026"，第 5014–5028 页，2026 年 10 月）。
- 易混淆的论文：arXiv 2610.30105 "Asymmetric On-Policy Distillation Learns In-Context Exploration"（ASTEROID，机器人方向）与本文无关。
- 官方实现：<https://github.com/stargazer319/AOPD>，commit `ce015567f45ff5f9fd9a6e699b056c928ca08d4c`（2026-06-25），基于 VeRL v0.3.1 + vLLM 0.8.5（README），teacher 以独立 vLLM OpenAI 服务的形式提供。本地浅克隆在 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/AOPD`。
- 本仓实现：未实现（计划注册名 `aopd`）
- 状态：调研中
- 结论（一句话）：按 token 把 OPD 拆成两半，teacher 概率高于 student 的 token 照常做 OPD 的 policy gradient，其余 token 改用 teacher top-K 上的 forward KL 监督；在 Qwen3-32B→Qwen3-8B-Base（3k 步 SFT 预热）上平均 Pass@1 比 OPD 高 4.09（53.40 vs 49.31，论文 Table 1）。

## 方法要点

AOPD 沿用 OPD 的 token 级 advantage A_t = sg[log P_T(y_t|c_t) − log P_S(y_t|c_t)]（论文 Eq. 1–3）。论文把 OPD 分解为正 / 负强化后认为，OPD 的优化困难集中在 advantage 非正的区域（论文 §3.2）：在这些位置，policy gradient 只会压低 student 自己采到的 token，却不告诉它该往哪个 token 走。因此 AOPD 保留正 advantage 区域的 OPD 更新（L_Pos），而在非正区域换成 teacher 支撑集上的 forward KL（论文 Eq. 6–8）：

- S_t = TopK(P_T(·|c_t), K)（Eq. 6）。
- L^FKL_t = (1/K) Σ_{v∈S_t} P_T(v|c_t)·(log P_T(v|c_t) − log P_S(v|c_t))（Eq. 7）。
- L_AOPD = E[(1/|y|) Σ_{t∈S≤0} L^FKL_t] + L_Pos（Eq. 8）。

论文进一步用有界的概率差代替 log-ratio 作为门控（Eq. 9–10）：G_t = I(P_T(y_t) − P_S(y_t) ≤ τ)，L = E[(1/|y|) Σ_t (G_t·L^FKL_t + (1−G_t)·L^OPD_t)]。按论文的约定，τ=−1 退化为 OPD，τ=1 退化为 GKD，默认 τ=0。附录 B.1 的梯度分析表明 FKL 对 logit 的更新正比于 P_T − P_S，即「把 student 往 teacher 的整个 top-K 分布拉」，而不是只压低采样 token（论文 Eq. 12）。

官方代码的实现路径（commit `ce01556`）：

| 作用 | 位置 |
|---|---|
| teacher 打分：用 completions API（`echo=True`、`max_tokens=1`、`logprobs=top_k`）取整条序列的采样 token logprob 和每个位置的 top-K logprob / 索引；返回的字典按 logprob 降序后取前 K | `verl/utils/api_interface.py:126-194` `_call_api_single` |
| 把 teacher 结果对齐到 response 位置，产出 `teacher_log_probs`、`teacher_topk_logprobs`、`teacher_topk_indices` | `verl/workers/teacher/vllm_teacher.py:61-161` `compute_logprobs` |
| 门控与正区域 advantage：`prob_diff = P_S − P_T`，`gkd_mask = prob_diff > aopd_threshold`，在 `~gkd_mask` 上 `advantage = opd_weight·(log P_T − log P_S)`；`adv_estimator=none` 时直接作为 advantages 返回 | `verl/trainer/ppo/ray_trainer.py:190-273` `compute_advantage`（门控在 `:235-242`） |
| actor 损失：PPO clip 的 pg_loss + 在 `gkd_mask` 上的 top-K FKL（`jsd_beta=1` 分支），两者都按 `loss_agg_mode` 聚合后相加 | `verl/workers/actor/dp_actor.py:457-523`；FKL 本体 `dp_actor.py:49-140` `compute_gkd_loss` |
| teacher 服务：vLLM OpenAI server，`--max-logprobs 48`，max-model-len 32768 | `deploy_vllm.sh` |
| Megatron 版本的同一损失（含 vocab-parallel gather） | `verl/workers/actor/megatron_actor.py:56-240` |

## 适用条件

AOPD 按同 tokenizer 设计：teacher top-K 的 token id 直接 gather student 的 log-prob，两组模型对都在 Qwen3 家族内。teacher 必须是能返回 logprob 的白盒模型，需要两类信息：采样 token 的 logprob（用于 advantage 与门控），以及每个 response 位置 top-K（K=32）token 的 logprob 与索引（用于 FKL）。不需要完整词表分布。task reward 不参与训练：官方 launcher 设置 `reward_model.enable=False`、`algorithm.adv_estimator=none`。论文的 student 是 Base 模型，先在 OpenThoughts 上做 SFT 预热再蒸馏（§5.1），预热不足时 OPD 甚至低于 SFT 起点（Table 1，8B 1k 预热）。

## 论文的实验设置

流程分三段（§5.1、附录 C）：先用 LlamaFactory 在 OpenThoughts 上做 warm-up SFT（3k 或 1k 步），再在 DeepMath 上做蒸馏，最后在 ToolAlpaca 上做持续学习实验。模型对是 Qwen3-32B → Qwen3-8B-Base 与 Qwen3-8B → Qwen3-4B-Base。基线为 SeqKD、OPD、GKD（student rollout 上的 top-K forward KL）和 ExOPD，K 统一为 32。训练用 16×A100-80G，teacher 另占 8×A100。

| 项 | 论文（Table 3，附录 C） | 代码 launcher `qwen3_8b_aopd_distill.sh` |
|---|---|---|
| lr | 1e-5 | `actor.optim.lr=1e-5` |
| train batch | 512 | `data.train_batch_size=512`，`rollout.n=1` |
| mini batch | 未写 | 未设置，取 `ppo_trainer.yaml:35` 默认 256，即每步 2 次更新（推断） |
| max prompt / response | — / 12288 | 4096 / 12288 |
| rollout 温度 | 1.0 | 1.0 |
| top-K | 32 | `distill.top_k=32` |
| 门控 τ、FKL 方向 | τ=0，β=1（纯 forward KL） | `aopd_threshold=0`，`jsd_beta=1` |
| 其他 | — | `use_kl_loss=False`，`distill_signal_in_loss=False`，dynamic bsz（`ppo_max_token_len_per_gpu=20000`），rollout TP=2，`total_epochs=1`，`save_freq=5`，`loss_agg_mode` 默认 token-mean（`ppo_trainer.yaml:46`） |
| 评测采样 | T=0.6，top-p 0.95，top-k 20，最长生成 32768 | — |

评测集为 AIME24、AIME25、HMMT25 Feb，报 Pass@1 与 Pass@4。论文中没有找到每题采样次数和 seed 数，三个评测集各只有 30 题，单次差异可能在噪声内。

主要数字（三集平均 Pass@1 / Pass@4）：

| 设置 | SFT 起点 | OPD | GKD | ExOPD | AOPD | 来源 |
|---|---|---|---|---|---|---|
| 4B，3k 预热 | 41.87 / 58.56 | 47.29 / 64.15 | 46.11 | 47.50 | 49.16 / 64.07 | Table 1 |
| 8B，1k 预热 | 36.53 | 34.79 / 53.06 | 40.90 / 60.61 | — | 43.13 / 63.66 | Table 1 |
| 8B，3k 预热 | 45.90 | 49.31 / 66.64 | 49.65 | — | 53.40 / 67.43 | Table 1 |
| 4B，1k 预热 | — | 36.94 | 40.63 | — | 41.18 | Table 4 |

只写一个数的格是 Pass@1，「—」表示本文档没有摘录。摘要里的 +4.09 是 8B 3k 预热下 53.40 − 49.31，+8.34 是 8B 1k 预热下 43.13 − 34.79。注意 4B 3k 预热下 AOPD 的 Pass@4（64.07）略低于 OPD（64.15）。

其他结果：

- 持续学习（Table 2，ToolAlpaca 成功率 / 学完后数学 Pass@4 变化）：OPD 65.0 / −7.49，ExOPD 67.0 / −1.49，GKD 76.0 / −2.42，AOPD 75.0 / +0.61。
- 60 步短训对比（Table 5，Pass@1）：OPD 47.09，OPD-Norm 47.92，OPD-Clip 49.17，KDRL 51.53，EOPD 51.59，AOPD 52.01（Pass@4 68.34）。
- 消融（§5.4）：β=1 的 Pass@1 最好，β=0.9 的 Pass@4 最好（Figure 6）；K 在 8/16/32 中以 32 最好；被干预（走 FKL）的 token 比例在训练中从约 40% 降到约 30%；τ ≤ −0.2 时被干预的 token 不到 10%（附录 D.3）。

## 与本仓实现的差异

本仓尚未实现。下面先列官方实现内部「论文 vs 代码」的差异（commit `ce01556`，均为读代码所得，未实际运行），再给出在 EasyOPD 中实现的成本估计。

论文与官方代码的差异：

- 门控符号相反。代码是 `prob_diff = P_S − P_T`，`gkd_mask = prob_diff > aopd_threshold`（`ray_trainer.py:235-242`），论文是 G_t = I(P_T − P_S ≤ τ)。所以代码的 `aopd_threshold` 等于论文的 −τ：τ=−1 退化为 OPD 对应代码阈值 1。τ=0 时两者只在 P_T = P_S 的平局 token 上不同：论文用 ≤，把平局送进 FKL；代码用严格 >，把平局留在 OPD。`compute_advantage` 签名里的默认阈值是 0.1，主 launcher 显式设为 0；ToolAlpaca 的 AOPD launcher 设为 0.2（即论文 τ=−0.2），复现持续学习实验时要注意。
- 门控在两处各算一次。trainer 端用 rollout 时的 student logprob 算 advantage；actor 端用 `old_log_probs` 和 `teacher_log_probs[:, -response_length:]` 重算同一个 mask 来决定 FKL 作用在哪些 token（`dp_actor.py:504-511`），并记录 `actor/gkd_token_ratio`。复现时可用这个指标对照论文里约 40%→30% 的干预比例（§5.4）。
- FKL 的归一化与论文不一致。`compute_gkd_loss`（`dp_actor.py:86-140`）直接对 teacher top-K logprob 取 exp，并在 student 的全词表 log_softmax 上 gather，然后算 Σ_K P_T(log P_T − log P_S)。代码里既没有 Eq. 7 的 1/K，也没有在 S_t 上做重归一化，而附录 B.1 的梯度推导假设分布已经归一化。结果乘 `gkd_weight=1.0`、`gkd_mask`、`response_mask`，再按 `loss_agg_mode`（默认 token-mean）聚合，加到 policy loss 上（`dp_actor.py:519-523`）。actor 端 `jsd_beta` 的默认值是 0.5（JSD），launcher 设为 1，即纯 forward KL(T‖S)。
- 正区域走 PPO-clip。advantage 在 OPD 区域等于 `opd_weight·(log P_T − log P_S)`，在 FKL 区域为 0，经 `adv_estimator=none` 原样返回（`ray_trainer.py:269-273`），再进入 clip 0.2 的 PPO pg_loss。按 τ=0 的门控，OPD 区域的 advantage 恒为非负，所以这一项只做正强化，和 L_Pos 的设计一致。`use_kl_loss=False`，`use_kl_in_reward` 默认 False（`ppo_trainer.yaml:218`），`kl_coef=1e-4` 实际不生效。
- mini batch 未写明。AOPD launcher 没设 `ppo_mini_batch_size`，取默认 256（`ppo_trainer.yaml:35`），GKD 和 OPD 的 launcher 都显式写了 256；推断每步 2 次 optimizer 更新。
- teacher 打分用 completions API 的 echo 技巧（`api_interface.py:126-194`）：`echo=True`、`max_tokens=1`、`logprobs=top_k`、温度默认 1.0。返回的 top-K 字典按 logprob 降序截断到 K，第一个 token 的 logprob 置 0。vLLM 服务需开 `--max-logprobs 48`（≥ K=32），见 `deploy_vllm.sh`。teacher 温度和 student rollout 温度都是 1.0，所以不存在温度不一致。
- 基线脚本的命名有误导。`qwen3_8b_opd_distill.sh` 的 `opd_weight=1.25`、`gkd_weight=0`，对应的是 ExOPD（放大 reward 的 OPD），不是纯 OPD；`qwen3_8b_gkd_distill.sh` 用 `distill_mode=gkd`，在 actor 端不加 mask，即所有 token 都做 top-K FKL。复现 OPD 基线时应把 `opd_weight` 设回 1.0。
- KL 命名与本仓 `gkd` 相反。AOPD 把 KL(T‖S) 叫 forward KL，`jsd_beta=1` 即 KL(T‖S)。本仓 `easyopd/methods/gkd/core.py:57-70` 把 KL(S‖T) 叫 forward、KL(T‖S) 叫 reverse，算的是 β·KL(T‖S) + (1−β)·KL(S‖T)。所以 β=1 时两边的数学形式相同，但名字相反；0<β<1 时本仓是两个 KL 的线性插值，AOPD 是以 βP_T+(1−β)P_S 为中点的 JSD（`dp_actor.py:115-134`），两者不等价。复用时按公式对齐，不按名字。

论文本身的可疑点：

- Table 4 的 SeqKD 一行与 3k 预热块的数字完全相同，疑似复制错误，引用时只用 OPD / GKD / AOPD 三行。
- 论文没有写评测的每题采样数和 seed 数。Pass@4 至少需要每题 4 个样本，Pass@1 的采样次数未知；AIME24/25、HMMT25 各 30 题，单次 1–2 分的差异可能在噪声内。

在 EasyOPD（同 tokenizer teacher）中实现的估计：成本中等。需要的 teacher 信号有两类：采样 token 的 logprob（给 advantage 与门控）和每个位置 top-32 的 logprob / 索引（给 FKL）。OPSA 已有 top-K 通路可以复用：`verl/trainer/ppo/ray_trainer.py:1472-1488` 把 ref worker 的 `ref_topk_log_probs` / `ref_topk_indices` 作为 `opsa_teacher_topk_*` 传给 actor，`verl/workers/actor/dp_actor.py:1195-1225` 按索引 gather 并在 K 上重归一化后算 FKL/RKL（带 `clamp(min=0)` 和 `opsa_temperature`）。在此之上需要两个方法 hook：一个在 advantage 阶段算门控后的 OPD advantage（FKL 区域置 0），一个在 loss 阶段把门控后的 FKL 加到 PPO loss 上。主要的不确定点有三个：Qwen3-32B teacher 在 ref worker 里是否放得下，放不下就需要像官方一样走独立服务（可参考 `simple` 的 teacher sidecar）；归一化选哪种，是按官方代码（不除 K、不重归一化）复现数字，还是按论文 Eq. 7 与附录 B.1（除 K、在 S_t 上归一化）；OPSA 路径的 `clamp(min=0)` 和重归一化需要能关掉，才能对齐官方代码。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| Qwen3-8B-Base（OpenThoughts 3k 步 SFT 预热）← Qwen3-32B，DeepMath，三集平均 Pass@1 / Pass@4 | AOPD 53.40 / 67.43，OPD 49.31 / 66.64（Table 1） | 未跑 | — |
| Qwen3-4B-Base（3k 步预热）← Qwen3-8B，同上 | AOPD 49.16 / 64.07，OPD 47.29 / 64.15（Table 1） | 未跑 | — |
| 60 步短训，Pass@1 | AOPD 52.01，OPD 47.09（Table 5） | 未跑 | — |

60 步短训一行最便宜，适合在本仓先做「门控 + FKL 生效」的小规模验证，再决定是否跑全量。

## 待解决

- 论文的评测每题采样数、seed 数未找到；复现时至少报多次采样的均值，并说明 30 题小集上的差异是否在噪声内。
- 复现目标选官方代码的 FKL（不除 K、不在 S_t 上归一化）还是论文 Eq. 7 的写法；建议先按代码复现数字，再把论文写法作为变体对照。
- 论文 Table 3 没有 mini batch；官方 launcher 用默认 256，是否就是论文实验的设置需要问作者。
- 论文的 warm-up SFT 用 LlamaFactory 在 OpenThoughts 上做（§5.1），本仓需要另外准备这一步的 checkpoint，或者换用已有的 SFT 起点并在报告里写明偏差。
- Qwen3-32B teacher 在 EasyOPD 里用 ref worker 还是独立服务，取决于显存预算，未评估。

## 来源

- https://export.arxiv.org/abs/2605.06387
- https://arxiv.org/html/2605.06387v4
- https://aclanthology.org/2026.findings-emnlp.238/
- https://github.com/stargazer319/AOPD
