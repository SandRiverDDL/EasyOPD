# 踩坑记录

一条一节：现象 → 根因 → 结论。排障前先搜本文件。

## DAPO-Math-17k 的 parquet 每道题重复了约 100 遍

- 现象：`dapo-math-17k.parquet` 有 1,791,700 行，而数据集官方说法是约 17k 题。
- 根因：数据文件本身就重复了题目；按 `prompt` 去重后是 17,398 条（2026-10-29 实测）。不是下载出错。
- 结论：训练前先按 `prompt` 去重，否则 epoch 数和步数的含义会差约 100 倍。

## actor 的 `fsdp_config.model_dtype=bfloat16` 会让主权重和 Adam 状态都变成 bf16

- 现象：10-30 之前 07_opd_repro 的所有 run 都设了 `+actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16`，FSDP 分片参数（即主权重）是 bf16；AdamW 的 m、v 按参数 dtype 建，也是 bf16。旁证：bf16 run 的 `actor/grad_norm` 是 bf16 可表示的整齐值（13.8125、9.375），fp32 run 是 13.8415…、9.7728…。
- 根因：bf16 尾数 7 位，|w|≈0.02 处相邻可表示数间隔约 1.2e-4；lr=2e-6 时 Adam 单步更新约 2e-6，远小于半个间隔，会被舍入掉，且不会累积。
- 结论：actor 用 verl 默认的 fp32 主权重（bf16 计算），common.sh 已改为 `ACTOR_MODEL_DTYPE=fp32`（可用环境变量覆盖）。代价实测很小：1.7B / 4 卡 / vLLM 显存比例 0.75 下，峰值 reserved 73.7 GB（bf16 为 72.9 GB），单步耗时无明显变化（`1030_0601_smoke_fp32` vs `1030_0400_smoke_sched2`，各 2 步）。ref/teacher 只做前向，仍用 bf16。10-30 之前的 run 绝对数字可能被压低，影响程度未实测。

## DAPO 训练集包含 AMC23 的 28/40 题和 MATH 测试集的大量原题

- 现象：`train_en`（DAPO 去重后 14,116 条）与评测题逐条比对（2026-10-31）：AMC23 40 题里有 28 题在训练集中；MATH 测试集（去掉 MATH-500 后）L4 约 1/3、L5 约 1/2 是原题。AIME24/25 为 0/30，MATH-500 约 2%。
- 根因：DAPO 从 AoPS 和竞赛官网爬题并改写答案格式（DAPO 论文 §3.5），没有做去污染。改写会改动措辞，只按原文前缀比对会漏掉一部分，同一份 AMC23 用不同的归一化方法测出 16 到 26 题不等。
- 结论：
  - AMC23 不再报告（干净的只剩 12 题，太少）。`data/opd_repro/amc23_clean.parquet` 留着备查，不进报告。
  - 不要用 MATH 测试集 L4/L5（除 MATH-500 外）做验证或评测。
  - 换新评测集前先用同样的 3 种方法查一遍：归一化前缀、13-gram 覆盖、最近邻 Jaccard 人工复核。改写过的重复题，这 3 种方法都可能查不出来。
  - 细节见 [analysis/2026-10-31-dapo-contamination.md](analysis/2026-10-31-dapo-contamination.md)。
