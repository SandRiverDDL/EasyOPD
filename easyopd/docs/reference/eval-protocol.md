# 评测与选 ckpt 流程（07_opd_repro，2026-10-31 定）

- 代码：`experiments/07_opd_repro/scripts/evaluate.py`（离线评测）；训练时验证由 protocol 的 val 文件决定
- 依据：[analysis/2026-10-31-dapo-contamination.md](../analysis/2026-10-31-dapo-contamination.md)（污染）、[analysis/2026-10-31-ckpt-select-subset.md](../analysis/2026-10-31-ckpt-select-subset.md)（选题子集）、[analysis/2026-10-31-eval-8k-vs-32k.md](../analysis/2026-10-31-eval-8k-vs-32k.md)（截断）
- 状态：流程已定；两份验证 parquet 和选 ckpt 脚本**待实现**（见文末）。10-31 之前的 run 用的是 `math500_dev200`，验证曲线与新流程不可直接比。

## 一句话

训练时用 MATH-500 L4+L5 抽 200 题每 10 步验证一次，在有 ckpt 的步（每 20 步）上取 3 点滑动平均最高的 2 个作候选，训练结束在 MATH-500 L4+L5 全部 262 题上测 avg@4 定最终 ckpt；报告以 AIME 为主，MATH-500 为辅并注明选择偏差。

## 三个环节

| 环节 | 题目 | 评测设置 | 作用 |
|---|---|---|---|
| 训练时验证 | MATH-500 中 Level 4+5 共 262 题，固定 seed 抽 200 | 训练框架内置 val，每 10 步 | 看趋势和上升速度、发现训崩、挑候选 |
| 选 ckpt | MATH-500 中 Level 4+5 全部 262 题 | avg@4，8K | 在候选里定最终 ckpt |
| 报告 | AIME24/25（主），MATH-500 全集 avg@4（辅） | AIME 16K，MATH 8K | 对外数字 |

为什么选 MATH-500 而不是 MATH 其余 4500 题：DAPO 训练集收录了 MATH 其余部分的约 1/3（L4）到 1/2（L5），而 MATH-500 原文重复只有约 2%（10 条前缀重复，22 条含 13-gram 高覆盖）。MATH-500 是这批数据里最干净的 MATH 题源，见污染分析。

为什么用 L4+L5：L1–L3 在强 ckpt 上接近饱和（L1 平均正确率 0.90）；L4+L5 同 run 内排序与 AIME/AMC 金标的 Spearman 为 0.87，和全量 500 题的 0.89 基本相同，题数却只有一半。

## 选 ckpt 规则

1. 频率：验证每 10 步，存权重每 20 步，100 步得到 10 个验证点、5 个 ckpt。验证间隔必须整除存权重间隔，保证每个 ckpt 步都有验证分。验证密一倍是为了看分数上升速度。
2. 平滑：对验证曲线做 3 点滑动平均，只在 ckpt 步上算，即第 20k 步取第 20k−10、20k、20k+10 步的平均；最后一个 ckpt（第 100 步）没有右邻，用第 90、100 步两点平均。平滑前单点噪声较大（200 题、单次采样时 SE 约 3 分），相邻两个验证点的差多数在噪声内，看上升速度也要看平滑后的曲线。
3. 候选：在 ckpt 步中取平滑后最高的 2 个。
4. 定夺：两个候选在 262 题上逐题配对，算差值的标准误（SE）。差距 ≥ 1 SE 选高的；< 1 SE 视为打平，选**更早**的那个。
   - 依据：一个标准误规则（Breiman 等 1984 CART；Hastie 等《The Elements of Statistical Learning》§7.10），原意是在最优 1 SE 内选最简单的模型。把「最简单」对应成「训练步数最少（离初始模型最近）」是本仓的类比，原文没有。
   - 量级参考：semi100_semi / semi100_opd 相邻 ckpt 在 262 题 avg@4 上的配对 SE 为 1.1–1.8 分。
   - 副作用：峰值早的方法略占便宜；所有方法用同一规则，方法间比较仍公平。
5. 可选的无挑选数字：和 Semi-OPD 论文对齐时，另报第 60/80/100 步的平均（论文 Table 1 口径）。

## 评测设置

| 集合 | 题数 | k | max_tokens | 采样 |
|---|---|---|---|---|
| MATH-500 L4+L5（选 ckpt） | 262 | 4 | 8192 | T 0.6，top_p 0.95，`enable_thinking=false` |
| MATH-500 全集（报告） | 500 | 4 | 8192 | 同上 |
| AIME24 / AIME25 | 30 / 30 | 8 | 16384 | 同上 |

- `evaluate.py` 的 `--max_tokens` 对一次调用的所有评测集生效，所以 MATH（8K）和 AIME（16K）要分两次调用。
- AMC23 不报告：40 题里 28 题在 `train_en` 里，剩下 12 题太少，见 [pitfalls](../pitfalls.md)。
- 和 Semi-OPD 论文（T 1.0、top_p 1.0、avg@32、32K、AIME24/25/26+HMMT Feb26）不同，数字不能直接对比。AIME26、HMMT Feb26 本地还没有。
- 16K 下的截断率还没测过；32K 时 OPD/semi 在 AIME 上仍有 10–24% 截断（eval-8k-vs-32k 分析），16K 会更高，报告时要附截断率。

## 已知偏差，报告时要写明

- **MATH-500 偏高**：ckpt 是在它的 L4+L5 子集上选的，属于选择偏差（不是污染）。主结论看 AIME。
- **改写版重复没查**：13-gram 只抓原文或近原文重复，改写过的同题抓不到（arXiv 2311.04850）。MATH-500 的改写重复数未知。
- **小集合噪声**：AIME 每集 30 题、单 seed，几分以内的差异不下结论。

## 待实现

- [x] `data/opd_repro/math500_l45_dev200.parquet` 与 `math500_l45.parquet`（10-31 建：从 `math500.parquet` 取 level 4/5 共 262 题（L4 128、L5 134），`DataFrame.sample(n=200, random_state=0)` 后按原 index 排序，得 L4 96、L5 104；格式同 `math500_dev200.parquet`）
- [ ] protocol 的训练时 val 文件换成 `math500_l45_dev200`，`test_freq=10`、只权重 ckpt `save_freq=20`（目前只有批次 `1031_len` 在 base.yaml 里单独设了，common.sh 默认值未改）
- [ ] 选 ckpt 脚本：读 TB 验证曲线 → 平滑 → 取前 2 → 262 题 avg@4 → 按 1 SE 规则输出最终 ckpt
- [ ] 用 semi100 的 10 个 ckpt 验证一遍流程
