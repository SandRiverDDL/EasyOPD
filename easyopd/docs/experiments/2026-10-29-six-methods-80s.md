# 2026-10-29 六个 OPD 方法 80 步对照（已过时）

- 代码：`easyopd/methods/{vanilla_opd,power_opd,dial_opd,eopd,revisit_opd,topd}/`、`experiments/07_opd_repro/scripts/{common.sh,eval_scheduler.py}`
- 状态：done（10-31 补写；配置已过时，见下）
- 结论（一句话）：存疑 — 6 个方法都能跑完 80 步，但这批 run 有 EOS bug、lr 2e-6、bf16 主权重三处和现行配置不同，MATH-500 排序不能当方法结论。

## 要回答的问题与判据

同一 student / teacher / 数据 / 协议下，6 个 OPD 方法的流程是否都跑通，以及 80 步后 MATH-500 的表现。判据：6 个都跑完不崩；MATH-500 avg@8 和基线比。适用范围：单 seed、32 题 × 1、2048 长度、1.7B ← 4B、80 步；80 步的排序不外推到更长训练。

## 配置

- student `Qwen3-1.7B-Base-imend`，teacher `Qwen3-4B`，数据 train_en（DAPO 去重 14116 题）。
- 32 题 × n=1，response 2048，80 步，每 20 步存 ckpt，lr 2e-6，bf16 主权重，seed 1，每方法 4 卡。
- 和现行配置的差异（`experiments/07_opd_repro/configs/changes.yaml`）：
  - `eos-fix`：Base 学生以 `<|endoftext|>`(151643) 结束，teacher 给约 −20 logp，结尾 token 的信号被污染；power / dial 的结尾 mask 当时只含 151645。
  - `fp32-master`、`lr-1e-6`：主权重 bf16、lr 2e-6。
  - `val-math500`：训练时验证集是 aime24。
- 原始 README：`runs/train/07_opd_repro/1029_1648_b32n1r2k_80s20_README.md`。

## 产物

- 运行目录：`runs/train/07_opd_repro/<method>/1029_1648_b32n1r2k_80s20/`，hf ckpt step 20/40/60/80。
- 评测：各 run 的 `eval/results.jsonl`，由 `eval_scheduler.py` 生成。总表见 `runs/train/07_opd_repro/RUNS.md`。

## 结果

MATH-500 avg@8（500 题），T=0.6、top_p 0.95、8192 tokens，括号里是截断比例。student_init（Qwen3-1.7B-Base-imend）在同一批评测里为 36.5（27%）；10-30 `1030_rl_eval` 重测 avg@4 为 37.9。

| 方法 | step 20 | step 40 | step 60 | step 80 |
|---|---|---|---|---|
| vanilla_opd | 50.8 (12%) | 57.3 (6%) | 59.7 (5%) | 60.5 (5%) |
| eopd | 50.7 (13%) | 56.9 (6%) | 58.5 (5%) | 59.1 (5%) |
| topd | 46.9 (13%) | 55.8 (5%) | 59.8 (4%) | 30.0 (2%) |
| revisit_opd | 42.0 (23%) | 47.5 (19%) | 50.9 (16%) | 55.7 (13%) |
| dial_opd | 41.4 (24%) | 45.4 (19%) | 47.3 (17%) | 50.3 (15%) |
| power_opd | 38.8 (25%) | 41.3 (23%) | 43.0 (21%) | 45.0 (20%) |

AIME 只在 MATH 最好的 step 上评（avg@16，30 题），全部在 2.5–8.8 之间，单 seed 下分不开，不列。

## 观察与后续

- 6 个方法都跑完 80 步，流程打通，这是这批唯一可靠的结论。
- vanilla / eopd / topd（step 60 前）一组，revisit / dial / power 一组，后者截断率高 10–15 个点。dial / power 用有界变换，最受 EOS bug 影响，所以这组差距有多少来自方法本身、多少来自 bug 分不开。
- topd 在 step 80 从 59.8 掉到 30.0，截断率反而只有 2%，原因没查（推测是输出提前结束，未核实）。
- 后续：方法对照要在现行配置（lr 1e-6、fp32、cross_eos、math500 验证）下重跑，按 `reference/eval-protocol.md` 选 ckpt。10-30 的 `1030_eosfix` 只重跑了 power / dial 200 步，但仍是 lr 2e-6 + bf16，也已过时。
