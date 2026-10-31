# IER：Information-Efficiency Ratio 稀疏 OPD（`ier`）

涉及代码：`oss_refs/IER-OPD/slime/rollout/ier_opd/ier_metrics.py`、`oss_refs/IER-OPD/slime/rollout/tip_compat.py`、`oss_refs/IER-OPD/slime/rollout/on_policy_distillation.py`、`oss_refs/IER-OPD/slime/backends/megatron_utils/loss.py`、`oss_refs/IER-OPD/scripts/train.py`、`oss_refs/IER-OPD/configs/main.json`（均在 `/mnt/NLU_disk17/chenhaoyu02/` 下）。

- 论文：1% of Tokens Can Be Enough: On Gradient Estimation in On-Policy Distillation，arXiv [2609.24432](https://arxiv.org/abs/2609.24432) v1，2026-09-21 提交。作者 Huanxin Sheng、Zhiling Ye、Haonan Wang、Jian Wang、Jinjie Gu、Jian Kang；单位 MBZUAI、Ant Group。
- 会议：arXiv comments 字段为空，README 未声明录用。结论：预印本。
- 官方实现：有。[BruceSheng1202/IER-OPD](https://github.com/BruceSheng1202/IER-OPD)，本地浅克隆 `/mnt/NLU_disk17/chenhaoyu02/oss_refs/IER-OPD`，commit `d12b0bca2f5997a253d43292ac6b88c7f4d30041`（2026-09-21）。框架是 slime（THUDM，Megatron-LM 训练 + SGLang rollout / teacher 服务），README 写明基于 slime 和 TA-OPD 代码库；`third_party/UPSTREAM.json` 记录 Python 3.12、PyTorch 2.9.1+cu129、SGLang 0.5.9（`bbe9c7e`）、Megatron-LM `3714d81`，并带 `sglang.patch`、`megatron.patch`。不是 verl。
- 本仓实现：未实现（计划注册名 `ier`）
- 状态：调研中
- 结论（一句话）：IER 用「最优标量 baseline 下单 token 梯度估计的信噪比」给 token 打分，在全 batch 内只保留 0.1%–10% 的 token 进 sampled reverse-KL 损失，可单用或与 usefulness 分数做 soft OR/AND；需要 teacher 与 student 的 Top-16 logprob，在 verl 里实现成本中等（若复用 `gkd` 的 full-vocab teacher 通路可降到低–中）。

## 方法要点

记号（论文 Sec 3.1）：固定前缀下 student 分布 p、teacher 分布 q，ρ(a) = log(p_a/q_a)，φ_a = ∇_z log p_a = e_a − p。reverse-KL 对 logits 的梯度 g = E_p[ρ(a)·φ_a]（Eq 2）。sampled-token OPD 只采一个 token，引入与动作无关的标量 baseline b 后估计量 ĝ_b(a) = (ρ(a) − b)·φ_a 仍无偏（Eq 3）。

- 度量（Sec 3.2）：作者认为欧氏 MSE 不贴合分布匹配，改用 KL 诱导的 Fisher 几何，F = diag(p) − pp^T，用 F 的伪逆范数 ‖·‖_{F⁺} 度量误差。
- Theorem 1：令 leverage L(a) = (1 − p_a)/p_a，则信号 ‖g‖²_{F⁺} = Var_p[ρ(a)]（Eq 7），噪声 E_p[‖ĝ_b − g‖²_{F⁺}] = E_p[(ρ(a) − b)²·L(a)] − Var_p[ρ(a)]（Eq 8），最优 baseline b* = E_p[ρL]/E_p[L]（Eq 9）。
- 定义（Eq 10–11）：IER = Signal / Noise（取 b = b*）。附录 B.3 说明 IER⁻¹ 是最优 baseline 下估计量的相对 MSE。论文明确说「高 IER 不代表高 usefulness」：IER 只衡量这个 token 的梯度估计是否可靠。
- 直观：L(a) 对低概率 token 很大，采到低概率 token 时梯度估计噪声大；p 越集中、ρ 方差越大，信号越强。
- 候选集近似（Sec 4，Eq 12–14）：全词表统计太贵，用 C = TopK(z_θ) ∪ TopK(z_T) ∪ {a}，在 C 上分别对 student、teacher 重新归一化得 p̂、q̂；某一侧不在自己 Top-K 里的 token 用常数补缺；ρ̂ 截断到 [−30, 30]；用 p̂ 代入 Eq 7–9 得 IÊR。附录 C.1：K = 16，16 ≤ |C| ≤ 33，缺失值填 −12，ε = 1e-8。
- 与 usefulness 融合（Eq 15）：在 rollout batch 内把 IÊR 转成归一化排名 r_j ∈ [0,1]，usefulness u_j ∈ [0,1]，IER-OR：s = 1 − (1 − u)(1 − r)；IER-AND：s = u·r。usefulness 基线有 Prefix、Entropy、TIP、TA-OPD、CA-SoftOR 五种。
- 选择（Sec 5.1、附录 C.2）：分数在整个 rollout batch 内归一化，所有 response token 在一个全局预算下竞争（如 1%），每条 response 至少保留 1 个 token；因此不同 response 的保留比例不同。
- 训练目标：选中 token 上仍是原 sampled reverse-KL 目标，student 生成完整 response，不截断 rollout（README）。

论文在 Sec 5.2 报告 IÊR 分布极度重尾，通常不到 0.1% 的 token 分数大于 1，即多数 token 的估计噪声大于信号；作者据此认为对所有 token 做全量 OPD 可能不是最优。

代码实现：

- IER 公式：`ier_metrics.py:71` `compute_position`。`_build_candidate_items`（:21）构造候选集：student / teacher Top-K 并集加采样 token，先填真实采样 token 的 logprob（:41–44），其余缺失值填 `floor_logp=-12`（:47–49）；`_renormalised_logp`（:54）在 C 上重新归一化；ρ̂ 截断在 :117–118；信号、L、b*、噪声、IER 在 :120–136：`signal_r = Var_p(c_r)`，`L_r = 1/max(p, eps) - 1`，`b_star_r = E[cL]/E[L]`，`noise_r = max(second_r - signal_r, 0)`，`ier_r = signal_r / max(noise_r, eps)`。纯 Python 逐 token 循环，在 rollout 侧 CPU 上算。
- 调用点：`tip_compat.py:68` `process_tip_compat_metrics`，在 teacher 打分后（`on_policy_distillation.py:65`）逐 token 计算；只有方法是 `ier` 或 `*_ier_or/and` 时才算 IER（:119–152）。
- 预算选择：`tip_compat.py:400` `_apply_budget_mask`。全局预算 `ceil(ratio × N_valid)`（:441），N_valid 是整个 rollout batch 的有效 token 数；再给每条 response 补足 `min_keep=1`（:444–450），所以实际选中数可能超过名义预算；融合方法先用 `_batch_ier_ranks`（:387）把 IER 转成 0…1 的序数排名，再按 OR/AND 合成（:426–433）；`random` 用固定 seed（42 + rollout id）的随机置换（:421–425）。选择结果直接写回 `sample.loss_mask`（:456）。
- usefulness 归一化：`_normalize`（:350）是 batch 内 5%–95% 分位数线性缩放并裁剪到 [0,1]（`configs/main.json` 的 `normalization_q_low/high`）；论文只写「normalized」。
- 损失：slime 的 OPD 把 reverse KL 加进 advantage，`loss.py:530` `apply_opd_kl_to_advantages`：`advantages[i] = adv - opd_kl_coef * (student_log_probs - teacher_log_probs)`（:563–564）；任务奖励在 `on_policy_distillation.py:71` 恒为 0，所以 advantage 只剩 −(log p − log q)。然后走 PPO-clip 策略梯度（`ppo_utils.py:125` `compute_policy_loss`，eps_clip 0.2 / eps_clip_high 0.28）。
- 损失分母：未开 `--calculate-per-token-loss`，用 `cp_utils.py:67–72` 的 `sum_of_sample_mean`：每条 response 先对选中 token 求均值（除以该 response 的选中数，最小 1），再对 response 求和，最后在 `loss.py:1191–1194` 除以 `global_batch_size`。即 seq-mean-token-mean：每条 response 等权，不论它选中 1 个还是 100 个 token。论文没有写分母。

## 适用条件

- 同 tokenizer：必须（候选集按 token id 取并集）。论文的 big-to-small 设置明确要求「same tokenizer」。
- teacher 信息：每个 response 位置的 teacher Top-16 logprob（ids + 值）与采样 token 的 teacher logprob。官方用 SGLang 服务在 `/generate` 上以 `max_new_tokens=0` 做 prefill 打分，`top_logprobs_num=16`（`tip_compat.py:20` `maybe_add_topk_request`，`on_policy_distillation.py:10` `reward_func`）。不需要 full-vocab。
- student 信息：student Top-16 logprob 取自 SGLang rollout 引擎生成时的返回（`sglang_rollout.py:174`、:217），即 rollout 引擎精度下的 logprob，而不是训练引擎重算的。
- 任务奖励：不使用（恒为 0）。
- `--opd-exact-cmass`（用 teacher 对 student Top-K ids 的精确 logprob 补缺）在代码里存在，但官方 launcher 未开启，因此缺失值都按 −12 填，与论文一致。

## 论文的实验设置

来自论文 Sec 5.1 与附录 C.2：

- 数学：JustRL-Nemotron-1.5B → OpenMath-Nemotron-1.5B（strong-to-weak，同尺寸同架构）；JustRL-Qwen3-4B → Qwen3-1.7B（big-to-small，同 tokenizer）。训练数据 DAPO-Math-17k，50 个 rollout 轮次。评测 AIME 2025/2026、HMMT-Feb 2025/2026，每题 32 个样本，报 Bayes@32 的均值和标准误；评测 temperature 0.7、top-p 0.9、最大 32,768 token。
- 医疗：ClinAlign-4B → Qwen3-4B，RaR-Medicine，100 个 rollout 轮次；报 HealthBench overall / Hard，用 gpt-oss-120B 当评审（meta-eval macro F1 0.6614；默认评审 GPT-4.1 为 0.709，作者因预算未用）。
- 训练：每个 rollout batch 4 个 prompt × 16 条 response，temperature 1、top-p 1；prompt / response / context 上限 2048 / 8192 / 16384；每个 rollout batch 切 8 个 minibatch（每个 8 条）；Adam，lr 1e-6，weight decay 0.1，β = (0.9, 0.98)。Qwen3 默认关闭 thinking；Sec 5.5 另做 thinking-on。
- 预算：主实验 0.1%、1%、10%；参数分析另加 5%、20%、50%、80%（Fig. 2、3）。
- 硬件：Table 3 写「Peak memory is aggregated across eight GPUs」，即 8 卡；GPU 型号论文未写。seed 数论文未写；官方配置 `seed=1234`、`rollout_seed=42`、`selector_seed=42`（`configs/main.json`）。

同预算下的 random 对照：有，但只在数学任务。Table 2 每个预算（10%、1%、0.1%）都有 Random 行；附录 C.4（Fig. 6–11）把 random 扩展到 3 次运行并加入 Liu et al. (2026d) 的 Sampled RKL max/min。医疗任务（Table 1）没有 random 行，README 也写明 `random` 只用于数学。

主要数字（Bayes@32，%，论文 Table 2，标准误 0.4–1；列为 AIME25 / AIME26 / HMMT25 / HMMT26）：

| 方法 | Nemotron-1.5B：JustRL → OpenMath | Qwen3：JustRL-4B → 1.7B |
|---|---|---|
| Student | 47.3 / 53.3 / 32.8 / 32.6 | 11.6 / 10.3 / 8.4 / 10.4 |
| Teacher | 58.6 / 62.3 / 40.6 / 35.7 | 22.5 / 19.9 / 13.7 / 20.2 |
| Full OPD（100%） | 56.3 / 59.9 / 36.9 / 34.7 | 14.4 / 12.5 / 8.7 / 13.4 |
| Random 1% | 50.9 / 55.4 / 33.1 / 32.7 | 12.8 / 12.6 / 8.3 / 11.7 |
| IER 1% | 54.8 / 58.2 / 34.8 / 34.6 | 15.8 / 12.6 / 9.2 / 13.2 |
| Sampled RKL max 1% | 56.0 / 59.8 / 38.1 / 34.9 | 12.6 / 13.9 / 7.4 / 11.7 |
| Random 0.1% | 49.6 / 55.4 / 33.5 / 32.9 | 13.6 / 12.4 / 8.7 / 10.6 |
| IER 0.1% | 53.5 / 58.9 / 34.8 / 34.1 | 15.8 / 15.2 / 9.5 / 13.0 |
| Prefix + IER-OR 0.1% | 54.9 / 59.1 / 34.0 / 34.9 | 16.0 / 15.0 / 10.3 / 13.6 |
| TIP + IER-AND 0.1% | 52.1 / 56.1 / 35.8 / 34.8 | 19.4 / 16.3 / 11.7 / 14.4 |

- 医疗（Table 1，overall / Hard）：IER 0.1% 为 45.25 / 18.37，Prefix 0.1% 为 38.30 / 8.68；Full OPD overall 45.77（Sec 5.2 正文）。论文说 0.1% 在该任务里大约每条轨迹只选 1 个 token。
- 训练效率（Table 3，8 卡，30 个稳态 step）：Full OPD 每步 2.103 ± 0.255 s、峰值显存 235.95 GiB；IER 10% 为 2.152 ± 0.270 s（+2.3%）、236.03 GiB。由此推断稀疏选择基本不省算力：teacher 打分和 student 前向仍覆盖全部 token。
- 读数注意：Qwen3 这一对里 Full OPD 只把 AIME25 从 11.6 拉到 14.4（teacher 22.5），所有方法的差距都在几个点内，而标准误 0.4–1；Nemotron 这一对 IER 单用在 1% 下全面低于 Sampled RKL max。论文自己在附录 C.4 承认「no token selector performs best universally across all six settings」。

## 与本仓实现的差异

本仓尚未实现。官方实现与论文之间的差异（论文未写、代码才有的细节）：

- 训练目标是 PPO-clip 策略梯度而非直接监督：advantage = −(log p_old − log q)，eps_clip 0.2 / 0.28，每个 rollout batch 切 8 个 minibatch 依次更新（`configs/main.json`：`num_steps_per_rollout=8`、`global_batch_size=8`），所以第 2–8 个 minibatch 已是离策略，由 ratio clip 约束。论文只说「sampled reverse-KL」。
- 损失分母是 seq-mean-token-mean（每条 response 对自己的选中 token 求均值后等权平均）；在 0.1% 预算下多数 response 只有 1 个选中 token，而每条 response 在 minibatch 损失里的权重都是 1/8，这个 token 就承担了整条 response 的权重。若本仓用 verl 默认 `token-mean`（除以选中 token 总数），每个 token 的有效权重会不同，复现时要对齐。
- 名义预算 vs 实际：全局 `ceil(ratio × N_valid)` 后再补每条 response 至少 1 个，实际选中比例会高于名义值（Table 1 标题也写「nominal budgets」）。
- 缺失值：论文写给缺失「logit」填 −12；代码在 logprob 域填 −12（`ier_metrics.py:4–5` 注释说明 serving API 返回的是 logprob），并对采样 token 优先用真实 logprob。
- student Top-K 来自 SGLang rollout 引擎，teacher Top-K 来自 SGLang teacher 服务，二者精度都是 bf16 推理，不是训练引擎的 logits。
- 训练数据只跑 50（数学）/ 100（医疗）个 rollout 轮次，总共 50 × 64 = 3200 条 response、400 次优化步（数学），训练很短。
- 依赖改过的 SGLang 与 Megatron（`third_party/patches/`），未在本仓核对补丁内容。

在本仓（同 tokenizer teacher）实现的成本：中等。

- 需要的信号：teacher 与 student 每个位置的 Top-16 logprob + 采样 token logprob。若复用 `easyopd/methods/gkd/` 已有的 full-vocab teacher log_probs 通路，Top-16 可直接 `topk` 得到（显存贵但现成）；若走 sparse 通路，则与 `res_opd` 共用「teacher Top-k + 采样 token」通道，二者可一起建。student Top-16 可在 actor 前向里从 logits 取，不必依赖 rollout 引擎。
- IER 本体可在 torch 里对 [B, T, ≤33] 向量化，约 60 行；usefulness 基线（TIP、TA-OPD 等）如要对照需另写。
- 难点是「整个 rollout batch 全局竞争的预算」：选择必须在切 mini-batch 之前、拿到全部 teacher 打分之后做（例如在 driver 侧计算 advantage 的阶段把 mask 乘进 `response_mask`），不能放在每个 micro-batch 的 loss 函数里。
- 分母和 PPO 多 minibatch 设置需要与官方对齐（见上），否则 0.1% 预算下的有效学习率差别很大。

## 复现状态

| 设置 | 论文数字 | 本地数字 | 报告 |
|---|---|---|---|
| JustRL-Qwen3-4B → Qwen3-1.7B，IER 0.1%，AIME25 / AIME26 Bayes@32（Table 2） | 15.8 / 15.2 | 未跑 | [repro/](../repro/) |
| 同上，Full OPD（Table 2） | 14.4 / 12.5 | 未跑 | [repro/](../repro/) |
| 同上，Random 0.1%（Table 2） | 13.6 / 12.4 | 未跑 | [repro/](../repro/) |
| ClinAlign-4B → Qwen3-4B，IER 0.1%，HealthBench overall（Table 1） | 45.25 | 未跑 | [repro/](../repro/) |

## 待解决

- 决定本仓分母采用 seq-mean-token-mean（对齐官方）还是 token-mean，并在实验里注明。
- 先做 CPU 级检查：用小词表精确枚举验证 Theorem 1（Eq 7–9）和候选集近似的偏差，再上 GPU。
- 判断 IER 的收益是否超出噪声：Qwen3 这对方法间差距只有 1–3 个点，需要多 seed；论文 random 在附录才有 3 次运行。
- 与 `dial_opd`、`res_opd` 共用 teacher Top-k 通道的设计，避免各写一套。

## 来源

- https://export.arxiv.org/abs/2609.24432（标题、作者、日期、comments）
- https://arxiv.org/html/2609.24432v1（正文、Theorem 1、Eq 1–15、Table 1–3、附录 B.3、C.1–C.4）
- https://github.com/BruceSheng1202/IER-OPD（浅克隆 commit `d12b0bc`，README、configs、scripts/train.py、slime 内 IER 与 OPD 代码、third_party/UPSTREAM.json）
