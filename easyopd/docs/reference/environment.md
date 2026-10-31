# 环境、模型与数据位置

> 常青文档。路径或事实变了就改这里；其他文档只放链接。

## 运行环境

恢复步骤见根 `CLAUDE.md`「环境搭建」。依赖清单与构建脚本的位置也在那里。

## 存放约定（NLU17，前缀 `/mnt/NLU_disk17/chenhaoyu02/`）

| 类别 | 位置 | 说明 |
|---|---|---|
| 模型 | `models/<HF仓库名>/` | 自己转换的版本在目录名后加 `-fp8` / `-bf16` / `-packed`；已被脚本引用的目录不改名 |
| 训练数据 | `data/<名字>/` | 开源数据集原样放，处理后的另建目录 |
| 评测集 | `benchmarks/<名字>/` | 只读，体量小，不和训练数据混放 |
| 运行产物 | `runs/` | 布局见 `runs/INDEX.md` |

不要落到 `~/.cache/huggingface`。

## 已就位（2026-10-29 下载，均来自 HuggingFace）

| 名称 | 路径 | 规模 | 备注 |
|---|---|---|---|
| Qwen3-4B | `models/Qwen3-4B/` | 7.6G | post-trained 版本，不是 Base |
| Qwen3-1.7B-Base | `models/Qwen3-1.7B-Base/` | 3.3G | Base |
| DeepScaleR-Preview-Dataset | `data/DeepScaleR-Preview-Dataset/deepscaler.json` | 40,315 条 | |
| DAPO-Math-17k | `data/DAPO-Math-17k/data/dapo-math-17k.parquet` | 1,791,700 行 | 按 `prompt` 去重后 17,398 条，见 [pitfalls.md](pitfalls.md) |
| MATH-500 | `benchmarks/MATH-500/test.jsonl` | 500 题 | `HuggingFaceH4/MATH-500` |
| AMC23 | `benchmarks/amc23/` | 40 题 | `math-ai/amc23`，不是 83 题的 AMC；28 题与 DAPO 训练集重复，不再用于报告，见 pitfalls |
| AIME24 | `benchmarks/aime24/` | 30 题 | `math-ai/aime24` |
| AIME25 | `benchmarks/aime25/` | 30 题 | `math-ai/aime25` |

safetensors 只确认文件存在，没做哈希校验。下载脚本和日志：`runs/dl_20261029.{sh,log}`。

## 各方法默认数据与模型

默认数据集配方在 `easyopd/data_provider.py` 的 `DATASET_RECIPES`；各方法 yaml 里的模型是占位或示例，不是推荐值。选哪个 teacher / student 以 `methods/<name>.md` 和对应 `repro/` 为准。
