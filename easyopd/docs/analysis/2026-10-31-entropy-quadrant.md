# 2026-10-31 OPD 信号落在哪类位置：teacher 熵 × student 熵四象限

- 代码：`experiments/07_opd_repro/scripts/probes/group_quadrant.py` 的 [Q] 部分（新写；读 token 用 `position_profile.py` 的 `load`，sig2 的定义照搬 `grad_token_stability.py`）
- 产物：`runs/train/07_opd_repro/vanilla_opd/1030_0745_opd100lr1_vanilla/analysis/group_quadrant/stats.txt`
- 状态：open（一个 run，step 110 / 150 / 200）
- 结论（一句话）：OPD 的梯度几乎全在 student 熵高的位置。其中"teacher 确定、student 犹豫"只占 2.5% 的 token，却拿走 37% 的梯度能量；"两边都犹豫"占 26% 的 token、55% 的梯度能量；"student 过度自信"（teacher 犹豫、student 确定）只占 2%。三步之间基本不变。

## 问题

上一篇位置分析（[2026-10-31-position-profile.md](2026-10-31-position-profile.md)）之后，想知道 OPD 的信号集中在哪种不确定性组合上，判断按熵挑 token 的方法（EOPD、DIAL-OPD 一类）有多大空间。

## 方法

- 数据：`vanilla_opd/1030_0745_opd100lr1_vanilla` 的 token dump，step 110 / 150 / 200（只有这三步有 student 熵和 top-20）。过滤与位置分析相同：长度 ≥ 64，去掉最后的 stop token 和 EOS 类 id。
- 象限：teacher 全词表熵 H_T、student 全词表熵 H_S 各按阈值 τ 分高低，主口径 τ = 0.5 nats，另看 0.2 和 1.0。
- 指标（token 混在一起统计，因为梯度看的就是总量的分配）：
  - `tok`：token 占比，另按 correct / wrong 轨迹分开；
  - `kl_share`：student top-20 上反向 KL 的总量占比（loss 质量）；
  - `sig2_share`：student top-20 上反向 KL 精确梯度的平方和占比（logits 空间的梯度能量，不含参数空间的相互抵消）；
  - `lr`、`abs_lr`、`kl` 的均值；`top1_diff`：teacher top1 与 student top1 不同的比例。

## 证据

step 200，τ = 0.5：

| 象限 | token 占比 | kl 占比 | sig2 占比 | 平均 lr | 平均 kl | top1 不同 |
|---|---|---|---|---|---|---|
| T 低 S 低 | 65.8% | 9.7% | 5.6% | −0.020 | 0.020 | 0.3% |
| T 低 S 高（teacher 确定，student 犹豫） | 2.5% | 13.5% | 37.1% | −0.714 | 0.729 | 18.1% |
| T 高 S 低（student 过度自信） | 5.3% | 11.2% | 2.1% | −0.282 | 0.286 | 12.5% |
| T 高 S 高（都犹豫） | 26.4% | 65.6% | 55.2% | −0.339 | 0.336 | 33.5% |

- 跨步（τ = 0.5，token 占比 / sig2 占比）：

  | step | T 低 S 低 | T 低 S 高 | T 高 S 低 | T 高 S 高 |
  |---|---|---|---|---|
  | 110 | 69% / 6% | 3% / 39% | 5% / 2% | 23% / 53% |
  | 150 | 67% / 6% | 2% / 36% | 5% / 2% | 26% / 57% |
  | 200 | 66% / 6% | 3% / 37% | 5% / 2% | 26% / 55% |

- 阈值敏感性（step 200，T 低 S 高的 token / sig2）：τ = 0.2 为 2.2% / 27.5%，τ = 1.0 为 2.0% / 19.1%。τ = 1.0 时 T 低 S 低扩大到 79% 的 token、48% 的 sig2，因为不少 0.5–1 nats 的位置被划了进来。不管阈值怎么取，"T 低 S 高"的单 token 梯度都是最大的。
- correct 和 wrong 轨迹的象限占比几乎相同（step 200：T 高 S 高 25.7% 对 27.5%，其余差 < 2 个百分点）。
- "T 高 S 低"的 kl 占比（11%）远高于 sig2 占比（2%）。student 熵低时 p_S 集中在一个 token 上，(e_y − p_S) 很小，所以 loss 不小但梯度很小。

## 排除的假设

- "student 过度自信的位置是 OPD 主要在纠正的地方" → 不成立：这类位置只占 5% 的 token、2% 的梯度能量。

## 结论与根因

已验证（本 run，step 110–200，单 seed）：

- 约 92% 的梯度能量落在 student 熵 ≥ 0.5 的位置，这类位置只占约 29% 的 token。
- 单 token 信号最强的是"teacher 确定、student 犹豫"，但它的 token 占比只有 2–3%。
- 象限分布在 step 110–200 之间稳定，不随训练明显变化。

仅推测：

- 只保留 student 熵 ≥ 0.5 的位置，会去掉约 71% 的 token，只丢约 8% 的梯度能量（step 200）。这可以省算力，但不一定改变训练结果；没做训练验证。

局限：

- sig2 只算 student top-20（质量约 0.997），是 logits 空间的量，没有测参数空间的抵消。
- token 混在一起统计，长轨迹权重大。

## 影响的决定

- 按熵挑 token 的方法，空间主要在"T 高 S 高"和"T 低 S 高"这两类；只保留"T 低 S 高"会丢掉一半以上的梯度能量。
- 另见同日的组内对错分析：[2026-10-31-group-correct-wrong.md](2026-10-31-group-correct-wrong.md)。
