# EasyOPD 的 Agent 指令

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
