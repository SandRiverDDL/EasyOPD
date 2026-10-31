# 2026-10-30 Semi-OPD 与 on-policy OPD 对比（含 10-31 少轨迹消融）

- 代码：`easyopd/methods/semi_opd/`、`experiments/07_opd_repro/configs/exps/{1030_semi,1031_semifew}/`、`experiments/07_opd_repro/scripts/{gen_offline_rollouts.py,make_ablation_data.py}`
- 状态：done（10-31 补写）
- 结论（一句话）：未复现。本仓协议下 Semi-OPD 在 MATH-500 上从 step 40 起比 vanilla OPD 低 2.5–5.0 分（只有 step 40 的 −5.0 超过 3 分判据），没有看到论文里"Semi-OPD 优于 OPD"。只用 16 或 4 条完整、答对的离线回答时，Semi-OPD 和 OPD 持平。

## 要回答的问题与判据

1. `1030_semi`：Qwen3-1.7B(non-thinking) ← Qwen3-4B-Instruct-2507 这一对上，用初始 student 的固定离线回答做 OPD（Semi-OPD），是否优于在线 OPD。论文（arXiv 2610.11291 Table 1）OPD 26.2、Semi 39.8，是 AIME24/25/26 和 HMMT26 的 avg@32 平均，32K 长度。
2. `1031_semifew`：整个训练集只有 N 条离线回答（N=16/4/1）时，Semi-OPD 还剩多少效果。

- 判据：MATH-500 avg@4 差 > 3 分才算差异；AIME avg@8 差 < 5 分不下结论。单 seed。
- 适用范围：单 seed、这一对模型、本仓协议。本仓协议和论文的每步题数、长度、步数、评测集都不同，数字不和论文直接比。

## 配置

- 两个 batch 都是每步 32 题 × n=1，response 2048，100 步，lr 1e-6，fp32 主权重，seed 1，4 卡；验证 math500_dev200 每 20 步，只权重 ckpt 每 20 步。
- `semi100_semi` 和 `semi100_opd` 只差回答来源。semi 用离线回答，opd 在线采样。两边读题顺序相同（同 seed，离线 parquet 行序和 train_en 一致）。
- 离线回答：`data/opd_repro/semi_opd/train_en__Qwen3-1.7B_nothink_t1_r4096.parquet`，14116 行，T=1.0、top_p 1.0、上限 4096，平均 1777 token，6.5% 截断。训练时截到 2048，因此 32.7% 的回答被截断，没有结尾。
- `semifew_n{16,4,1}`：离线回答只取 N 条初始 student 答对、正常结束、长度 1024–2048 的（所以在 2048 处不会被截），1 ⊂ 4 ⊂ 16，复制成 3200 行（100 步 × 32）。其余配置和 semi100_semi 相同。数据由 `make_ablation_data.py --which semi` 生成。
- 方法要点和与论文的差异见 [methods/semi_opd.md](../methods/semi_opd.md)。代码版本：commit ff0a8ca 加工作区改动。

## 产物

- 运行目录：`runs/train/07_opd_repro/{semi_opd/1030_0828_semi100_semi, vanilla_opd/1030_0854_semi100_opd, semi_opd/1031_07{10,11,12}_semifew_n{16,4,1}}/`
- 评测：各 run 的 `eval/results.jsonl`；基线和评测链在 `runs/train/07_opd_repro/1030_semi_eval/`；旧 README `1030_semi100_README.md`。
- 总表：`runs/train/07_opd_repro/RUNS.md`
## 结果

离线评测的设置是 T=0.6、top_p 0.95、8192 tokens、enable_thinking=false，括号里是截断比例。student_init（Qwen3-1.7B）为 MATH-500 73.7（0%）、AIME24/25 11.2 / 11.2；teacher（4B-Instruct-2507）为 93.5 / 56.2 / 45.0。

**1030_semi：MATH-500 avg@4，固定网格**

| step | semi100_semi | semi100_opd | 差（semi − opd） |
|---|---|---|---|
| 20 | 81.0 (6%) | 79.7 (4%) | +1.3 |
| 40 | 77.5 (21%) | 82.5 (9%) | −5.0 |
| 60 | 79.8 (20%) | 82.7 (11%) | −2.9 |
| 80 | 79.8 (19%) | 82.5 (11%) | −2.7 |
| 100 | 79.8 (19%) | 82.3 (11%) | −2.5 |

AIME 在 8K 下截断 32–85%，主要在测长度。32K 只重评了两个 ckpt：semi step 20 为 AIME24/25 24.2 / 22.1（截断 22% / 10%），opd step 60 为 43.8 / 28.3（24% / 19%）。两个 ckpt 的 step 不同，只能说 opd 不比 semi 差。

**1031_semifew：每个 run 取 dev200 最高的 step（并列取后者）**

下表对五个 run 用同一条规则选 step，semifew 实际评测的就是这个 step。

| run | 离线回答 | 选中 step | MATH-500 avg@4 | AIME24 / 25 avg@8 |
|---|---|---|---|---|
| semi100_opd | 在线采样 | 80 | 82.5 (11%) | 26.7 / 18.8 |
| semifew_n16 | 16 条 | 60 | 82.8 (11%) | 25.8 / 18.8 |
| semifew_n4 | 4 条 | 80 | 82.4 (12%) | 23.8 / 20.8 |
| semifew_n1 | 1 条 | 100 | 80.3 (8%) | 21.7 / 16.7 |
| semi100_semi | 14116 条（32.7% 被截） | 60 | 79.8 (20%) | 17.9 / 19.6 |

训练时 dev200（n=1，σ 约 3 分）从 step 0 的 0.730 起，五个 run 都在 step 20 升到 0.76–0.80，之后在 0.755–0.820 之间波动，分不开。

## 观察与后续

- n16 / n4 和 OPD 的差 ≤ 0.4 分，在噪声内。n1 和 semi100_semi 比 OPD 低 2.2–2.7 分，在 3 分判据内，也分不开。所以 MATH-500 上能确认的只有 1030_semi 固定网格 step 40 的 −5.0。
- 只用 16 条离线回答的 Semi-OPD 不比用 14116 条差。主要的训练信号可能来自 teacher 对"student 风格文本"的逐 token 打分，和覆盖多少题关系不大（推断）。
- semi100_semi 的 MATH 截断率从 step 20 的 6% 升到 19–21%，semifew 只有 8–12%。两者的区别之一是 semi100 有 32.7% 的回答在 2048 处被截、没有结尾。推测在截断的回答上训练让 student 学会了不结束，未做消融验证。
- 后续：要和论文对齐，需改用论文配置（128 × 1、4096），补 AIME26 和 HMMT Feb26，并用 32K 评测。semi100_semi 的长度问题可以先用"离线回答截到 2048 时丢掉被截的"做一组消融。