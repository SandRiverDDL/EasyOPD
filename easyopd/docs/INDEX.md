# EasyOPD 研究与开发文档索引

开工先读本文件和 [`STATE.md`](STATE.md)，按「什么时候读」挑文档，不要整目录扫。
新增 / 移动 / 删除文档时同步改这里，每篇一行。

## 现状

- [`STATE.md`](STATE.md) — 每次会话开工必读：进行中的事、在跑的任务、卡点、最近完成

## 常青文档（描述「现在是什么」，理解或代码变了必须跟着改）

### 开发（框架怎么接）
- [`add-new-method.md`](add-new-method.md) — 新增一个方法的 4 步和 5 类 hook；写新方法前读
- [`verl-touchpoints.md`](verl-touchpoints.md) — `verl/` 里每个 `[EasyOPD:*]` 标记的台账；改 `verl/` 必须同步改
- [`verl-upstream-mapping.md`](verl-upstream-mapping.md) — 配置字段与上游 `DistillationLossConfig` 的命名映射

### 方法（methods/，一个方法一篇，模板见 [`methods/_TEMPLATE.md`](methods/_TEMPLATE.md)）
碰某个方法前读对应一篇；`<name>` 与注册名一致。全部为 2026-10-29 调研、尚未实现。

- [`methods/eopd.md`](methods/eopd.md) — EOPD（ICML26）：teacher 高熵 token 上加 top-k FKL
- [`methods/v_opd.md`](methods/v_opd.md) — vOPD：KL baseline 降方差，需 teacher 在 student top-k 处取值
- [`methods/power_opd.md`](methods/power_opd.md) — PowerOPD（EMNLP26）：r = π_T^α − π_θ^α，只需采样 token logprob
- [`methods/revisit_opd.md`](methods/revisit_opd.md) — Revisiting OPD（COLM26）：teacher top-k 截断 RKL 强基线配方
- [`methods/aopd.md`](methods/aopd.md) — AOPD（Findings EMNLP26）：按 P_T>P_S 分流 PG / top-k FKL
- [`methods/distillm2.md`](methods/distillm2.md) — DistiLLM-2（ICML25）：teacher 回答 SKL + student 回答 SRKL
- [`methods/topd.md`](methods/topd.md) — TOP-D：reward 变换 + return + 组内 z-score + PPO-clip
- [`methods/rock_tokens.md`](methods/rock_tokens.md) — Rock Tokens：离线统计高 loss 高频 token 重加权
- [`methods/res_opd.md`](methods/res_opd.md) — ResOPD（预印本）：残差控制变量降方差，对比 vOPD
- [`methods/dial_opd.md`](methods/dial_opd.md) — DIAL-OPD（预印本）：按 |r|·|q−p| 选 token，配 random 同保留率对照
- [`methods/ier.md`](methods/ier.md) — IER（预印本）：top-16 候选上全局竞争的极少量 token 选择
- [`methods/semi_opd.md`](methods/semi_opd.md) — Semi-OPD（2610.11291）：初始 student 的固定离线 rollout 上做 OPD；10-30 已实现，复现 1.7B(NT)←4B-Instruct-2507

### 参考（reference/）
- [`reference/environment.md`](reference/environment.md) — 环境、模型、训练数据、评测集的存放位置与已知事实
- [`reference/opd-infra.md`](reference/opd-infra.md) — 同 tokenizer OPD 的挂点地图（teacher 信号从哪来、进训练的几条路、评测现状）；实现同 tokenizer 方法前读
- [`reference/train-scheduler.md`](reference/train-scheduler.md) — 07_opd_repro 的实验 yaml 分层（common.sh < 协议 < 方法 < 实验）与训练队列 `train_scheduler.py`；起训练前读
- 实验总表 `runs/train/07_opd_repro/RUNS.md` — 07_opd_repro 每个 run 的配置、状态（当前 / 过时 / 中断）、评测分数；由 `experiments/07_opd_repro/scripts/run_table.py` 自动生成，可比性规则在 `configs/changes.yaml`，手写备注在 `configs/run_notes.yaml`；查"哪些 run 还有效、分数多少"时先看它
- [`reference/eval-protocol.md`](reference/eval-protocol.md) — 07_opd_repro 的评测与选 ckpt 流程（训练时验证、1 SE 选 ckpt、各评测集长度与 k、已知偏差）；配验证 / 评测 / 写报告前读

### 踩坑
- [`pitfalls.md`](pitfalls.md) — 现象 → 根因 → 结论；排障前先搜一遍

## 时序文档（带日期，只追加，记录当时事实）

- `repro/` — 复现报告，模板见 [`repro/_TEMPLATE.md`](repro/_TEMPLATE.md)
  - （空）
- `experiments/` — 方法间对照、消融，模板见 [`experiments/_TEMPLATE.md`](experiments/_TEMPLATE.md)
  - [2026-10-29-six-methods-80s.md](experiments/2026-10-29-six-methods-80s.md)：6 个 OPD 方法 80 步对照（已过时：EOS bug、lr 2e-6、bf16）
  - [2026-10-30-rl-vs-opd.md](experiments/2026-10-30-rl-vs-opd.md)：同预算 GRPO / DAPO vs vanilla OPD，Base 1.7B 学生，200 步
  - [2026-10-30-semi-vs-opd.md](experiments/2026-10-30-semi-vs-opd.md)：Semi-OPD vs OPD（未复现论文结论），含 N=16/4/1 条离线回答的消融
  - [2026-10-31-response-length.md](experiments/2026-10-31-response-length.md)：vanilla OPD 训练 response 长度 1k/2k/4k/8k 对最终效果的影响（running）
- `analysis/` — 按问题建篇的排查 / 归因，模板见 [`analysis/_TEMPLATE.md`](analysis/_TEMPLATE.md)
  - [2026-10-31-fork-probe.md](analysis/2026-10-31-fork-probe.md)：后期 OPD 二选一位置换 token 是否改变答对率
  - [2026-10-31-position-profile.md](analysis/2026-10-31-position-profile.md)：分歧 / teacher 熵随轨迹内位置、句内位置怎么变
  - [2026-10-31-entropy-quadrant.md](analysis/2026-10-31-entropy-quadrant.md)：teacher 熵 × student 熵四象限的 token / 梯度占比
  - [2026-10-31-group-correct-wrong.md](analysis/2026-10-31-group-correct-wrong.md)：同一道题内错误与正确 rollout 的 teacher 信号对比
  - [2026-10-31-eval-8k-vs-32k.md](analysis/2026-10-31-eval-8k-vs-32k.md)：8K 评测截断是否改变 Semi-OPD / OPD 排序
  - [2026-10-31-late-kl-plateau.md](analysis/2026-10-31-late-kl-plateau.md)：后期 OPD KL 平台是否因二选一位置学不会（分组走势、固定前缀过拟合、梯度噪声尺度）
  - [2026-10-31-dapo-contamination.md](analysis/2026-10-31-dapo-contamination.md)：DAPO 训练集与 AIME / AMC23 / MATH 测试题的原文和 13-gram 重复
  - [2026-10-31-ckpt-select-subset.md](analysis/2026-10-31-ckpt-select-subset.md)：MATH-500 哪个子集最能在同一 run 内区分 ckpt

## 仓库里已有的相关文档（原地保留）

- [`../../docs/algo/contributing_a_method.md`](../../docs/algo/contributing_a_method.md) — 贡献新方法的完整清单
- [`../../experiments/`](../../experiments/) 下各实验组的 `README.md` — 旧的实验结果表（尚未迁入 `repro/` / `experiments/`）
