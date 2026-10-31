# 2026-10-31 训练 response 长度与最终效果（vanilla OPD，1k / 2k / 4k / 8k）

- 代码：`experiments/07_opd_repro/configs/exps/1031_len/`、`experiments/07_opd_repro/common.sh`、`easyopd/methods/vanilla_opd/`
- 状态：running
- 结论（一句话）：待填

## 要回答的问题与判据

问题：vanilla OPD 训练时 rollout 的最大 response 长度（1k / 2k / 4k / 8k）怎样影响最终效果，以及代价（每步耗时、截断率）。

判据：

- 主指标：每个 run 按 [评测流程](../reference/eval-protocol.md) 选出的最终 ckpt，AIME24 / AIME25 avg@8，16K 解码。两集合并 60 题 × 8 次采样，单 seed。两两之差小于约 3 分视为无法区分（AIME 每集 30 题，见评测流程「小集合噪声」）。
- 辅指标：MATH-500 L4+L5 262 题 avg@4，8K（同时是选 ckpt 用的集合，有选择偏差）；配对差值的 SE 参考 1.1–1.8 分。
- 过程指标：训练时验证（`math500_l45_dev200`，8K 解码，每 10 步）、`response_length/mean`、`response_length/clip_ratio`（截断率）、OPD KL、`timing_s/step`。
- 读法：
  - 若 4k ≈ 8k 而 1k、2k 明显更低，说明训练长度只要覆盖大部分回答即可，可以用更短长度省时间。
  - 若 1k 不比 8k 差，说明 OPD 的信号主要在回答前段，截断不伤效果。
  - 若短长度训练后 8K 验证时的回答长度明显变短、截断率下降，而 AIME 分数下降，记为「短训练把模型推向短回答」。
- 适用范围：Qwen3-1.7B-Base-imend ← Qwen3-4B（非思考模式），DAPO-Math `train_en`，32 prompts × 8，200 步，单 seed。不外推到更大模型或思考模式。

公平性说明：四个 run 的步数、每步 prompt 数和采样数相同，训练 token 数不同（短长度看到的 token 少、每步更快）。这是「同步数」对照，不是「同 token」或「同耗时」对照。需要时用 TB 曲线按累计 token 或耗时重画做补充，不另起 run。

## 配置

- baseline：`runs/train/07_opd_repro/vanilla_opd/1030_0745_opd100lr1_vanilla`（[rl-vs-opd](2026-10-30-rl-vs-opd.md) 里的 vanilla OPD，8192 长度）
- 共同配置（`configs/exps/1031_len/base.yaml`）：protocol `opd_b32n8r8k` + method `vanilla_opd`；lr 1e-6、fp32 主权重、`terminal_stop=cross_eos`；200 步，验证每 10 步，权重 ckpt 每 20 步，完整 ckpt 第 100、200 步只留最新一个；验证集 `math500_l45_dev200`，`VAL_MAX_TOKENS=8192`；4 卡一个 run。
- 差异：只改 `MAX_RESPONSE_LEN` = 1024 / 2048 / 4096 / 8192（`r1k.yaml` … `r8k.yaml`）。验证解码长度四个 run 都是 8192。
- 相对 baseline 的其他差异：一次跑满 200 步（baseline 先跑 100 步再续到 200）、验证每 10 步（baseline 每 20）、验证集换成 L4+L5（baseline 是 `math500_dev200`）。r8k 与 baseline 除这三点外相同，两者第 200 步的差可以粗看 run 间波动。
- 成本阶梯：直接全量。8k 配置已由 baseline 跑通 200 步，1k–4k 只缩短长度、显存更小，冒烟检查不能多提供信息。
- 预计耗时：r8k 参考 baseline 约 100–200 s/步，200 步约 7–9 小时（含验证）；r1k–r4k 更快。四个 run 同时起，各占一个 GPU 半机。
- 启动命令：
  ```bash
  cd /mnt/NLU_disk17/chenhaoyu02/EasyOPD/experiments/07_opd_repro
  python3 scripts/train_scheduler.py submit configs/exps/1031_len/r{1,2,4,8}k.yaml
  ```

## 产物

- 运行目录：`runs/train/07_opd_repro/1031_len/<MMDD_HHMM>_len200_r{1,2,4,8}k/`（第一批按批次存放的 run）
- ckpt：各 run 的 `hf/global_step_{20,40,…,200}`（合并后）
- 看板：TB 8015，用正则 `1031_len` 过滤
- 离线评测：训练结束后另起。`eval_scheduler.py` 写死了 1029_1648 批次和 8192，不能直接用；AIME 16K 和 MATH 8K 要分两次调用 `evaluate.py`。

## 结果

| run | 训练长度 | 选中步 | AIME24 avg@8 (16K) | AIME25 avg@8 (16K) | MATH L4+L5 avg@4 (8K) | 截断率（训练末） | s/步 |
|---|---|---|---|---|---|---|---|
| r1k | 1024 | | | | | | |
| r2k | 2048 | | | | | | |
| r4k | 4096 | | | | | | |
| r8k | 8192 | | | | | | |

单 seed；差异在噪声内就写「无法区分」。

## 观察与后续

- 待填
