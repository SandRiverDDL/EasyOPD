# EasyOPD 的 Agent 指令

EasyOPD 是在 verl 之上做 on-policy distillation（OPD）方法的统一框架：每个方法住在 `easyopd/methods/<name>/`，通过注册表 + hook 接入，对 `verl/` 的改动只允许是最小的挂接点。

## 开发流程

### 环境搭建

本项目使用预先打好的独立环境,**不要**在仓库里新建 `.venv`,也不要自己重建环境。环境是 Python 3.11 的 venv,固定在 `/root/envs/easyopd`(路径写死,不能换位置),用 `uv` 构建,由镜像自带的 `/usr/bin/python3.11` 提供解释器,所以只能在同一作业镜像里使用。

```bash
# 新 job 里恢复环境(每个节点各做一次,先复制到本地盘再解压):
mkdir -p /root/envs
cp /mnt/NLU_disk17/chenhaoyu02/envs/easyopd-20261023.tar.zst /root/
tar -I zstd -xf /root/easyopd-20261023.tar.zst -C /root

# 使用:仓库不要用 `-e` 安装,把本地盘上的仓库放进 PYTHONPATH:
export PYTHONPATH=/root/EasyOPD:$PYTHONPATH
/root/envs/easyopd/bin/python ...
```

- 环境的依赖清单在 `/mnt/NLU_disk17/chenhaoyu02/envs/easyopd-20261023.freeze.txt`。构建脚本在 `/mnt/NLU_disk17/chenhaoyu02/easyopd_install/build_env.sh`;需要改依赖时改脚本并生成新日期的 tar,不要直接往环境里 `pip install`。
- 仓库先拷到本地盘再使用,不要让 `PYTHONPATH` 直接指向 NFS 上的仓库(小文件导入很慢)。
- `pre-commit` 等开发工具不在环境里,需要时另装,不要装进 `/root/envs/easyopd`。

### 复用优先（新方法先找现成实现）

写新方法、启动脚本、评测脚本之前，先在仓库里找同场景的现成实现，能复用就复用、能包装就包装：

- 新方法：先看 `easyopd/methods/` 里同类的（同 tokenizer 看 `gkd`；跨 tokenizer 复用 `simple` 的 teacher sidecar 和 lm_head；最小示例是 `echo_kd`），再按 `easyopd/docs/add-new-method.md` 走。
- 启动脚本：包装 `experiments/` 下已跑通的 `launch.sh`，只改本任务必需的差异项；评测复用 `experiments/_shared/scripts/`。
- 只有确认没有可复用的才从零写，并在提交说明或报告里写明「新写的、参考了什么 / 没有参考」。

### 改 verl 的底线

- 方法逻辑放 `easyopd/methods/<name>/`，不往 `verl/` 里塞。必须挂接的点才允许改 `verl/`，并用 `# [EasyOPD:NAME]` … `# [EasyOPD:NAME] End` 包住，同时登记到方法的 `verl_modified_files` 和 `easyopd/docs/verl-touchpoints.md`。
- 改完跑 `scripts/test_all_methods.py` 做注册 / 配置 / hook 契约回归。

---

## 项目文档（外部记忆）

研究与开发知识集中在 `easyopd/docs/`，**入口是 `easyopd/docs/INDEX.md`**。

- **开工**：先读 `INDEX.md` 和 `STATE.md`，按索引的「什么时候读」挑文档，不整目录扫。碰某个方法前先读 `methods/<name>.md`。
- **每个事实只存一处，其他地方只放链接**：

  | 位置 | 放什么 | 性质 |
  |---|---|---|
  | `STATE.md` | 现在：进行中、在跑的任务、卡点、最近完成 | 覆盖写，≤ 80 行 |
  | `methods/<name>.md` | 一个方法一篇：论文要点、官方实现、本仓实现位置、复现状态 | 常青，理解变了就改 |
  | `repro/` | 复现报告：论文数字 vs 本地数字、偏差、结论 | 时序，只追加 |
  | `experiments/` | 方法间对照、消融、结论 | 时序，只追加 |
  | `analysis/` | 按问题建篇的排查 / 归因 | 时序，只追加 |
  | `reference/` `pitfalls.md` | 环境、数据、指标判读；稳定下来的坑 | 常青 |

  时序文档命名 `YYYY-MM-DD-<主题>.md`，按各目录的 `_TEMPLATE.md` 写，正文首行列涉及的代码路径。
- **收工**（阶段结束 / 交接 / 上下文快满前）：更新 `STATE.md`（做完的删掉，「最近完成」留一行链接，保留 10 条）；跑完的实验写进 `repro/` 或 `experiments/`；有结论的排查写进 `analysis/`；新增 / 移动 / 删除文档时同步改 `INDEX.md`。
- 代码历史看 `git log`，不在文档里复述。

---

## 研究原则

1. **开跑前先写清再起任务**：要回答的问题、判据（看哪个指标、多少算什么结论）、结论适用的规模、选的是成本阶梯哪一级（CPU 级检查 → 单样本 / 单卡 → 缩小规模 → 全量），以及为什么不用更低一级。这些写进批次的 experiments 文档（见「运行与产物」），跑完补结果。
2. **先调研、再实现、后复现**：实现某方法前，先读论文和官方实现（有就读），把要点和与本仓的差异写进 `methods/<name>.md`；没有官方实现的明说。
3. **复现报告必须并列数字**：论文数字和本地数字放同一张表，逐项写明偏差（模型 / 数据 / 步数 / 评测集 / seed 数）。对不上就是「未复现」，不得因为趋势相似就写成功。
4. **结论带证据边界**：AIME、AMC 这类小评测集只有几十题，单次运行的差异可能在噪声内；写结论时给出样本量和 seed 数，不把推断当事实，之前错了直接改。失败的尝试也要写。
5. **对照公平**：对比的方法用同一数据、同一训练预算、同一评测脚本；只改要比的那一项。
6. **零报错不等于生效**：新逻辑要有生效证据（日志里的 loss_mode / 方法名、对应指标有变化），验证新方法真的走了新路径，而不是回退到默认路径。
7. **先定位、后固化**：排查期间的诊断日志可以临时加，用独特可 grep 的关键字并加开关（默认关），定位完成后全仓删除；生产修复必须进正式代码。根因未确认时的临时修复先不补测试，确认后再补。
8. **失败两次就换思路**：同一做法连续失败，停下说明根因再换方案；偏离用户原意或引入取舍时先确认。

## 运行与产物

- 训练、常驻服务、拷权重这类长任务用 `setsid nohup <命令> > <日志> 2>&1 < /dev/null &` 启动，记录 pid 和日志路径（直接 `nohup cmd &` 会在会话被打断时一起被杀）。
- 报「在跑」要两项证据：进程存在（`ps -eo pid,sid,args | grep "[x]xx"`，不用 `pgrep -f`）且日志 mtime 在推进。
- 运行产物放仓库外的共享盘 `runs/`（布局和 tag 规则见 `runs/INDEX.md`），**不要只留在节点本地盘**（实例销毁即丢）。每次运行的 tag 按 `<月日>_<时分>_<关键改动>` 命名。
- 模型、数据、评测集的存放位置和命名见 `easyopd/docs/reference/environment.md`；不要落到默认缓存目录。
- 跨类、影响决策的结论提炼进 `repro/` 或 `experiments/`，并链接回产物目录。
- 07_opd_repro 的记录规则（总表 `runs/train/07_opd_repro/RUNS.md` 由 `scripts/run_table.py` 生成，见 `easyopd/docs/reference/train-scheduler.md`）：
  - 起一批训练前，在 `experiments/07_opd_repro/configs/run_notes.yaml` 的 `batches` 下写这批要回答的问题，并建 `easyopd/docs/experiments/<日期>-<批次>.md`（状态 running，写问题、判据、配置），`doc` 字段指向它。不再写 `runs/` 下的批次 README。
  - 改了影响 run 可比性的默认配置（lr、主权重精度、结尾处理、验证集等），在 `experiments/07_opd_repro/configs/changes.yaml` 追加一条，旧条目不改。
  - 探针 / 分析脚本的产物放 `runs/train/07_opd_repro/analysis/<日期>-<主题>/`，结论写 `easyopd/docs/analysis/`。
