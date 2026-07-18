# RTX 4090 服务器训练使用指南

本指南用于在一张 24 GB RTX 4090 上从零训练 RT-DETR-L + BTD-SE V2.5-S。默认训练 100 epoch、640 分辨率、batch 8，不加载预训练权重。

## 1. 数据保护结构

| 内容 | 保存位置 | 保护方式 |
| --- | --- | --- |
| 代码、配置、测试 | GitHub `main` | Git 版本管理 |
| VisDrone 数据集 | 服务器持久化数据盘 | 与系统盘和容器生命周期分离 |
| `last.pt` 和全部 `epochN.pt` | 持久化数据盘 | 每完成一个 epoch 保存一次 |
| 最近三个有效权重 | GitHub Release `btdse-v2.5-s-4090-live` | 上传成功并校验后才清理旧版本 |
| `results.csv`、诊断和参数 | GitHub `training-results` 分支 | 每次权重发布后自动 commit 和 push |
| GitHub Token | 持久盘独立 secrets 文件 | 权限 `600`，不进入 Git 和日志 |

远端权重使用 `btdse-last-epoch-0001.pt` 这类独立名称。这样上传第 N 轮时，第 N-1 轮仍然可下载，不存在先删除唯一备份再上传的空窗期。

## 2. 创建服务器

选择 Ubuntu、RTX 4090 24 GB，建议至少 16 核 CPU、64 GB 内存和 100 GB 可用持久化数据盘。确认平台的“关机后保留数据盘”选项已开启。

连接服务器后先检查：

```bash
nvidia-smi
df -h
```

安装基础工具。服务器已有 Python 3.10 时可跳过 Python 安装：

```bash
sudo apt-get update
sudo apt-get install -y git python3.10 python3.10-venv tmux curl
```

## 3. 克隆仓库并设置持久盘

将 `/workspace` 换成服务商提供的持久化数据盘。AutoDL 一类平台可以改成 `/root/autodl-tmp`。

```bash
export STORAGE_ROOT=/workspace/uav-btdse
export REPO_DIR=$STORAGE_ROOT/repo

mkdir -p "$STORAGE_ROOT"
git clone https://github.com/kkc236/uav-detection-baselines.git "$REPO_DIR"
cd "$REPO_DIR"
git switch main
```

执行自动配置。它会创建虚拟环境、安装 PyTorch 2.5.1+cu121 和 Ultralytics 8.4.90、检查 4090 与磁盘、建立持久化数据目录，并下载及转换 VisDrone train/val：

```bash
STORAGE_ROOT="$STORAGE_ROOT" REPO_DIR="$REPO_DIR" \
  bash scripts/setup_btdse_4090.sh
```

如果镜像只有 `python3`，使用：

```bash
PYTHON_BIN=python3 STORAGE_ROOT="$STORAGE_ROOT" REPO_DIR="$REPO_DIR" \
  bash scripts/setup_btdse_4090.sh
```

## 4. 配置 GitHub Token

在 GitHub 创建新的 fine-grained token，只选择 `kkc236/uav-detection-baselines`，仅授予 Repository permissions 中的 Contents: Read and write。不要授予 Administration 权限。

以前在聊天、截图或命令中出现过的 token 必须先撤销，不能继续使用。

用隐藏输入写入持久盘，token 不会进入 shell 历史：

```bash
TOKEN_FILE=$STORAGE_ROOT/secrets/github_token
read -rsp "GitHub token: " GITHUB_TOKEN; echo
printf '%s' "$GITHUB_TOKEN" > "$TOKEN_FILE"
unset GITHUB_TOKEN
chmod 600 "$TOKEN_FILE"
```

验证权限：

```bash
stat -c '%a %n' "$TOKEN_FILE"
```

输出必须以 `600` 开头。

## 5. 启动训练

```bash
cd "$REPO_DIR"
mkdir -p "$STORAGE_ROOT/logs"

nohup env \
  STORAGE_ROOT="$STORAGE_ROOT" \
  REPO_DIR="$REPO_DIR" \
  BATCH=8 \
  WORKERS=8 \
  bash scripts/run_btdse_4090.sh \
  > "$STORAGE_ROOT/logs/btdse_launcher.log" 2>&1 &

echo $!
```

可以断开 SSH。训练和上传都运行在服务器上，不消耗本地电脑网络；但云服务器本身必须保持开机。

脚本的行为如下：

1. 从零训练，不加载任何 `.pt` 预训练权重。
2. 每轮写入 `last.pt` 和独立的 `epochN.pt`。
3. 异常退出时校验 `last.pt`，损坏则回退到最新完整 `epochN.pt`。
4. 如发生 OOM、NaN 或 Inf，任务停止并保留最近完整检查点；论文协议固定 batch 8 和 AMP，不允许自动降级后继续。
5. 每个新 checkpoint 上传 GitHub Release，保留最近三个。
6. 指标和 SHA256 清单自动提交到 `training-results` 分支。
7. 训练完成后强制进行一次最终上传验证，不会自动删除本地数据。

## 6. 实时查看

训练输出：

```bash
tail -f "$STORAGE_ROOT/logs/btdse_4090_training.log"
```

恢复与固定参数检查：

```bash
tail -f "$STORAGE_ROOT/logs/btdse_4090_supervisor.log"
```

GitHub 上传状态：

```bash
cat "$STORAGE_ROOT/logs/btdse_github_sync.json"
tail -f "$STORAGE_ROOT/logs/btdse_github_sync.log"
```

GPU 与磁盘：

```bash
watch -n 5 nvidia-smi
watch -n 30 df -h "$STORAGE_ROOT"
```

已完成指标：

```bash
tail -n 2 "$STORAGE_ROOT/runs/btdse/scratch-rtdetr-l-btdse-100ep-4090/results.csv"
```

远端页面：

- 代码：<https://github.com/kkc236/uav-detection-baselines>
- 权重：<https://github.com/kkc236/uav-detection-baselines/releases/tag/btdse-v2.5-s-4090-live>
- 指标分支：<https://github.com/kkc236/uav-detection-baselines/tree/training-results>

## 7. 中断与自动恢复

普通训练进程崩溃不需要操作，守护脚本会自动恢复。整台服务器重启后，重新执行第 5 节的 `nohup` 命令即可；它会扫描同一持久盘并从最近有效 checkpoint 继续。

需要人工停止时：

```bash
kill -TERM "$(cat "$STORAGE_ROOT/logs/btdse_4090_supervisor.pid")"
```

当前尚未完成的 epoch 可能丢失，但上一轮的本地和 GitHub checkpoint 都保留。

## 8. 服务器丢失后的迁移

在新服务器重复第 2 至第 4 节，并保持相同的 `STORAGE_ROOT` 路径。然后从 Release 下载远端最近三个权重，选择 epoch 编号最大的文件：

```bash
mkdir -p "$STORAGE_ROOT/recovery"
cd "$STORAGE_ROOT/recovery"

gh release download btdse-v2.5-s-4090-live \
  --repo kkc236/uav-detection-baselines \
  --pattern 'btdse-last-epoch-*.pt'

latest=$(find . -name 'btdse-last-epoch-*.pt' -printf '%f\n' | sort -V | tail -n 1)
mkdir -p "$STORAGE_ROOT/runs/btdse/scratch-rtdetr-l-btdse-100ep-4090/weights"
cp "$latest" "$STORAGE_ROOT/runs/btdse/scratch-rtdetr-l-btdse-100ep-4090/weights/last.pt"
```

`gh` 未安装时，可从 Release 页面下载编号最大的文件。将它重命名为 `last.pt` 后重新启动第 5 节命令。

## 9. 保存空间说明

`save_period=1` 会留下每轮独立快照。按当前模型约 277 MB 一个 checkpoint 估算，100 轮约占 28 GB；GitHub 只保留最近三个，约占 0.9 GB，但整个训练期间会产生约 28 GB 的上传流量。脚本在持久盘低于 20 GB 可用空间时停止训练，防止写坏 checkpoint。

训练完成并确认 GitHub 最终上传状态为 `published` 后，才可以释放云服务器。持久盘中的完整 `runs` 目录建议再下载或打包保存一次。
