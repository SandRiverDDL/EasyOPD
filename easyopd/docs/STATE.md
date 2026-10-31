# STATE（updated: 2026-10-31 by chenhaoyu02）

> 覆盖写，只记「现在」，≤ 80 行。历史不留在这里：代码看 git log，实验看 `repro/` `experiments/`，结论看 `analysis/`。
> 多人协作时按人分小节，只改自己的。阶段结束 / 交接前更新。

## 进行中

### 07_opd_repro 总体（chenhaoyu02）
- 10-31 晚三台节点（.80 / .79 / .12）都空闲，没有训练或评测在跑。不要碰 RL 训练的 4 个节点。
- 哪些 run 还有效、分数多少：看 `runs/train/07_opd_repro/RUNS.md`（自动生成，见 [reference/train-scheduler.md](reference/train-scheduler.md)）。现行配置是 lr 1e-6 + fp32 主权重 + `cross_eos`，满足的只有 `1030_rl` 的 3 个 run 和 `1030_semi` / `1031_semifew` 的 5 个 run；10-30 之前的 run 全部过时。
- 驱动机上的 `train_scheduler.py run`（pid 1670145，10-30 起）还是旧代码，没有每 10 分钟重建 RUNS.md 的逻辑；队列空时重启即可生效。
- 新起训练一律用训练队列，调度器日志 `runs/train/07_opd_repro/_queue/scheduler.log`。

### 评测与选 ckpt（chenhaoyu02）
- 流程已定稿，见 [reference/eval-protocol.md](reference/eval-protocol.md)；AMC23 不再报告。
- 待做：做 MATH-500 L4+L5 验证 parquet；协议里改成验证每 10 步、ckpt 每 20 步；写选 ckpt 脚本；`eval_scheduler.py` 里 `--max_tokens` 写死 8192，AIME 要改成 16K。

### 待定的实验
- vanilla OPD 1k / 2k / 4k 长度对照：方案已讨论，未起。
- 方法对照要在现行配置下重跑；第一轮 6 方法见 [experiments/2026-10-29-six-methods-80s.md](experiments/2026-10-29-six-methods-80s.md)。`1030_eosfix`（power / dial 200 步、vanilla 消融）也已过时，没写结论文档。
- Semi-OPD 按论文配置（128 × 1、4096、32K 评测）重跑，见 [experiments/2026-10-30-semi-vs-opd.md](experiments/2026-10-30-semi-vs-opd.md) 后续一节。

### 目录整理（chenhaoyu02，10-31）
- 已做：RUNS.md 生成脚本、`configs/changes.yaml`、`configs/run_notes.yaml`、三篇 experiments 文档。
- 10-31 已加 CLAUDE.md「运行与产物」里的 07 记录规则（批次文档先于开跑、changes.yaml、探针产物路径）。
- 待用户确认：探针脚本移到 `scripts/probes/`、新 run 按批次存放 `runs/.../<批次>/<tag>/`、`_baselines/`、`07_opd_repro` 是否改名。

## 杂项 / 待办
- [ ] EOS 坑（Base 学生以 `<|endoftext|>` 结束，teacher 给 −20 logp）写进 pitfalls
- [ ] 删冒烟和中断的 run（RUNS.md 里标「冒烟」「中断」的），需用户确认
- [ ] 旧实验结果表在 `experiments/*/README.md`，决定哪些迁入 `repro/` / `experiments/`
- [ ] 读代码推断、未实跑的问题：g_opd launch 不建 ref worker（`dp_actor.py:970` KeyError）；gkd 的 `+distillation.*` 无读取方；`test_all_methods.py` 的 `EXPECTED_METHODS` 缺 alm/dskd/sft/uld
- [ ] 根目录 `AGENTS.md` 待删（已被 `CLAUDE.md` 取代）

## 不要再试（本轮踩过、尚未稳定的坑；稳定后移入 pitfalls.md）
- （空）

## 最近完成（最多 10 条，超出删最旧；原文见链接）
- 10-31（chenhaoyu02）实验总表 `runs/train/07_opd_repro/RUNS.md` + 三篇 experiments 补写：[RL vs OPD](experiments/2026-10-30-rl-vs-opd.md)（MATH 上 OPD 比 DAPO 高 4–6 分，和 GRPO 分不开）、[Semi vs OPD](experiments/2026-10-30-semi-vs-opd.md)（未复现）、[六方法 80 步](experiments/2026-10-29-six-methods-80s.md)（过时）
- 10-31（chenhaoyu02）评测与选 ckpt 流程定稿（MATH 8K / AIME 16K，L4+L5 选 ckpt），见 [reference/eval-protocol.md](reference/eval-protocol.md)；依据 [DAPO 污染](analysis/2026-10-31-dapo-contamination.md)（AMC23 28/40 重复，不再报告）、[选题子集](analysis/2026-10-31-ckpt-select-subset.md)
- 10-31（chenhaoyu02）熵四象限：2.5% 的"teacher 确定、student 犹豫"token 占 37% 梯度能量，见 [analysis/2026-10-31-entropy-quadrant.md](analysis/2026-10-31-entropy-quadrant.md)；组内对错：轨迹级 teacher 信号区分不了对错，见 [analysis/2026-10-31-group-correct-wrong.md](analysis/2026-10-31-group-correct-wrong.md)
- 10-31（chenhaoyu02）vanilla OPD 位置分析：分歧前高后低，teacher 熵不随位置上升；句首分歧后期主要由 token 身份解释，见 [analysis/2026-10-31-position-profile.md](analysis/2026-10-31-position-profile.md)
- 10-31（chenhaoyu02）OPD step 200 反事实分叉实验（612 分叉）：二选一位置换 token 基本不改答对率，见 [analysis/2026-10-31-fork-probe.md](analysis/2026-10-31-fork-probe.md)
- 10-30（chenhaoyu02）ckpt 两种存法（只权重 / 完整，频率和保留数分开），见 [reference/train-scheduler.md](reference/train-scheduler.md#ckpt-两种存法)
- 10-30（chenhaoyu02）实验 yaml 分层 + 训练队列 `train_scheduler.py`，见 [reference/train-scheduler.md](reference/train-scheduler.md)
- 10-29（chenhaoyu02）11 个 OPD 方法调研页，见 [INDEX.md](INDEX.md) 方法区；同 tokenizer 挂点地图 [reference/opd-infra.md](reference/opd-infra.md)
- 10-29（chenhaoyu02）模型 / 数据 / 评测集下载到 NLU17，见 [reference/environment.md](reference/environment.md)
- 10-29（chenhaoyu02）建立 `easyopd/docs/` 骨架，重写 `CLAUDE.md`（合并掉 `AGENTS.md`）
