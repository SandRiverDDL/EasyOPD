# 训练调度：实验 yaml 与训练队列（07_opd_repro）

代码：`experiments/07_opd_repro/scripts/{expcfg.py,train_scheduler.py,run_table.py}`、`experiments/07_opd_repro/configs/`、`experiments/07_opd_repro/methods/*/launch.sh`

结论先行：一次实验 = 一个 yaml。`train_scheduler.py submit` 把它放进队列，常驻的 `train_scheduler.py run` 发现空闲的 4 卡半机就用 `setsid nohup` 起训练。训练和评测的衔接还没做，训练跑完只做 merge，不评测。

## 配置分层

优先级从低到高，后面的覆盖前面的：

| 层 | 位置 | 放什么 |
|---|---|---|
| 通用 base | `common.sh` 里的 `${VAR:-默认值}` | 模型、数据、batch、学习率、验证、评测等协议默认值；不搬进 yaml，免得同一个默认值有两个来源 |
| 协议 | `configs/protocols/<名>.yaml` | 一轮实验共用的改动，例如 `b32n1r2k`：4 卡、batch 32×1、response 2048、诊断开关、cross_eos |
| 方法 base | `configs/methods/<方法>.yaml` | 方法开关和方法自带参数（与原 `launch.sh` 逐项一致，已用脚本对比） |
| 实验 | `configs/exps/<轮次>/<名>.yaml` | 只写这次改的项，`extends` 上面几层 |
| 调用方 shell | 环境变量 | 手动跑 `TOTAL_STEPS=2 bash methods/x/launch.sh` 时仍能覆盖 yaml |

为什么不用 Hydra 的 defaults list：verl 入口确实是 Hydra（`verl/trainer/main_ppo.py:44`），但卡数、步数、数据路径、ckpt 策略这些属于 `common.sh` 的 shell 变量，不是 verl 配置。用 Hydra 继承会把它们混进 verl 的配置树，还得重写已跑通的启动链路。所以只用 PyYAML 做合并，合并结果再交给 `common.sh`。

## yaml 写法

```yaml
extends: [../../protocols/b32n1r2k.yaml, ../../methods/power_opd.yaml]  # 相对本文件，按顺序合并
name: 200s25eos_power      # tag = <MMDD_HHMM>_<name>，run 目录 runs/train/07_opd_repro/<批次>/<tag>/（批次 = yaml 所在的 configs/exps/<批次>/；10-31 之前的 run 在 <method>/<tag>/）
gpus: 4                    # 4 = 半台机（GPU 0-3 或 4-7），8 = 整台机
env:                       # common.sh 变量
  TOTAL_STEPS: 200
  SAVE_FREQ: 25
args:                      # Hydra 覆盖参数，原样渲染成 key=value 放进 OPD_ARGS
  +actor_rollout_ref.actor.opd.terminal_stop: null   # null = 删掉上层继承来的这个键
```

合并规则：dict 按键递归合并，标量和 list 整个替换，`null` 删除继承来的键。`args` 用 dict 而不用 list，就是为了能单独覆盖或删除某一个 Hydra 键。`power_opd` / `dial_opd` 的方法 yaml 用 `null` 删掉协议里的 `cross_eos`，因为它们按官方做法 mask 结尾 token，两种处理不叠加。

检查合并结果：`python3 scripts/expcfg.py show <yaml>`（看合并后的 yaml）或 `bash <yaml>`（看交给 `common.sh` 的 bash 变量）。

`SLOT`、`N_GPUS`、`GPU_IDS`、`RUN_TAG`、`RUN_DIR`、`EASYOPD_ROOT` 由调度器设置，写进 `env` 会在 submit 时报错。

## 使用

在驱动机（能看到 NFS、能 ssh 到节点）上，于 `experiments/07_opd_repro/` 目录执行：

```bash
python3 scripts/train_scheduler.py submit configs/exps/<轮次>/*.yaml   # 入队（立即解析校验）
python3 scripts/train_scheduler.py status                             # pending/starting/running/done/failed + 步数 + 各节点空闲情况
python3 scripts/train_scheduler.py cancel <name>                      # 只删 pending，已启动的不动
python3 scripts/train_scheduler.py run --dry_run                      # 只打印这一轮会怎么放，不启动
setsid nohup python3 scripts/train_scheduler.py run > /mnt/NLU_disk17/chenhaoyu02/runs/train/07_opd_repro/_queue/scheduler.log 2>&1 < /dev/null &
```

`run` 每 60 秒一轮，队列为空时不探测节点；同一台机器只允许一个 `run`（`/tmp/opd07_train_scheduler.lock`）。停止调度器：杀掉该 `run` 进程即可，它不持有任何训练进程。

`run` 每 10 分钟顺带重建一次实验总表 `runs/train/07_opd_repro/RUNS.md`（`scripts/run_table.py`，失败只记 WARN，不影响调度）。手动重建：`python3 scripts/run_table.py`。表里的"当前 / 过时"按 `configs/changes.yaml` 判断：改了影响可比性的默认配置（lr、主权重、结尾处理、验证集等），就在那里追加一条；规则判断不了的批次问题、结论文档链接、单个 run 的备注写 `configs/run_notes.yaml`。

## 调度规则

- **节点白名单**：只碰 `NODES` 里的 3 台（10.206.120.80 / 10.206.114.79 / 10.206.120.12），不会碰 RL 训练节点；从不杀进程，失败不自动重试。
- **空闲判定**（半台机为单位）：没有 07 的 launcher 占着（读节点上 bash 进程 environ 里的 `SLOT`/`N_GPUS`，手动 `run_ablation.sh` 起的也能识别）、没有 `evaluate.py` 占着、每张卡显存 < 2000 MiB、且 15 分钟内没有刚派到这里但进程还没出现的任务。本地盘剩余 < 300 GB 的节点不派（FSDP ckpt 写本地盘）。
- **顺序**：FIFO + first fit。4 卡任务可以越过排队等整机的 8 卡任务，8 卡任务可能一直等不到；目前只有 4 卡任务。
- **状态**：不另存状态文件。`_queue/pending/` 里是待跑；启动后挪到 `_queue/launched/`，再由 launcher.log 的 `===== <method> done =====` / `[FATAL]`、节点进程和启动时间推出 starting / running / done / failed。调度器重启是安全的。

## 每次启动做的事

1. NFS 仓库 rsync 到节点 `/root/EasyOPD/`，再用 `rsync --link-dest` 硬链接出 `/root/easyopd_code/<tag>/` 作为这次运行的 `EASYOPD_ROOT`。之后改代码、再 rsync，都不会影响已在跑的任务（rsync 是写临时文件再改名，旧 inode 留给快照）。
2. 在 run 目录写 `resolved.yaml`（合并后的最终配置 + 来源 yaml + 节点）、`code_version.txt`（commit、`git status`、`git diff`；未跟踪文件只列名不存内容）、`launch.sh`（可在节点上手动重跑，首行注释给出命令）。
3. 节点上 `env SLOT=.. N_GPUS=.. RUN_TAG=.. RUN_DIR=.. setsid nohup bash <run_dir>/launch.sh > <run_dir>/logs/launcher.log`。

## ckpt 两种存法

`common.sh` 变量，实现见 `[EasyOPD:CKPT]`（`verl/trainer/ppo/ray_trainer.py`、`verl/workers/fsdp_workers.py`）：

| 种类 | 内容 | 位置（节点本地盘） | 频率 / 保留 | 用途 |
|---|---|---|---|---|
| 完整 | model + optimizer + extra + data.pt | `fsdp/global_step_N` | `SAVE_FREQ` / `MAX_CKPT_KEEP` | 断点续训（resume 只认这类） |
| 只权重 | model 分片 | `fsdp/weights/global_step_N` | `SAVE_WEIGHTS_FREQ` / `MAX_WEIGHTS_KEEP` | merge → `hf/` → 评测 |

- 默认 `SAVE_WEIGHTS_FREQ=-1`，行为与原来一致（每 `SAVE_FREQ` 步 + 最后一步存完整 ckpt）。
- 打开只权重后：最后一步一定存权重；完整 ckpt 只在 `SAVE_FREQ` 整除时存。两者同一步都到期就两份都存，各自轮转互不依赖。
- Step 4 对两个目录按步数取并集（同一步优先用 weights），都 merge 到 `hf/global_step_N`；`MAX_*_KEEP` 只删本地盘，不删 `hf/`。
- 体量（1.7B，4 卡，bf16）：只权重约 4 GB，完整约 11 GB。
- 冒烟 `configs/exps/smoke/ckpt4.yaml`（4 步，权重每 2 步留 1 份，完整每 3 步）10-30 跑通：最终 `global_step_3`（含 optim）+ `weights/global_step_4`（只有 model），step 2 的权重按保留数被删，`latest_checkpointed_iteration.txt`=3，merge 出 `hf/global_step_{3,4}`；测试 ckpt 已删除。

## 看训练曲线（TensorBoard）

每个 run 的 event 文件写在 `<run_dir>/tensorboard/`。TensorBoard 用 `--logdir` 指向 `07_opd_repro/` 根目录，靠递归扫描自动发现新 run（手动起的、调度器起的都算）。**不要用 `--logdir_spec` 逐个列目录**：那样新 run 不会出现，10-30 就因此看不到 `1030_0318_*` 四个 run。

```bash
R=/mnt/NLU_disk17/chenhaoyu02/runs/train/07_opd_repro
cd $R && setsid nohup /usr/local/bin/tensorboard --logdir $R --port 8015 --bind_all --reload_interval 60 > $R/tensorboard_8015.log 2>&1 < /dev/null &
```

run 名是 `<批次>/<tag>/tensorboard`（10-31 前的旧 run 是 `<method>/<tag>/tensorboard`），在页面左侧用正则过滤（如批次名或 `1030_0318`）挑一轮实验。新 run 最多晚 `reload_interval`（60 秒）出现。核对已加载的 run：`curl -s --noproxy '*' http://127.0.0.1:8015/data/runs`。

## 已知限制

- 没有训练→评测衔接；`EVAL_STEPS` 走 `common.sh` 自带的串行评测，协议 `b32n1r2k` 里是 `none`。
- 失败只标记，不重试；要重跑就重新 submit（会得到新 tag）。
- `code_version.txt` 不含未跟踪文件内容；`experiments/` 尚未提交，提交后才算完整可追溯。
- 节点 `/root/easyopd_code/` 下的快照不会自动清理（硬链接，未改动的文件不占额外空间）。
