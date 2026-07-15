# IOQC-SA 独立实验服务器运行说明

本说明用于在单张 RTX 3090、4090 或 5090 上，从零训练 `RT-DETR-L + IOQC-SA` 100 epoch。IOQC-SA 与 BTD-SE 完全隔离，模型使用原始 `rtdetr-l.yaml`，不加载预训练权重。

代码分支：`codex/ioqc-sa`

GitHub仓库：<https://github.com/kkc236/uav-detection-baselines>

远程权重Release：<https://github.com/kkc236/uav-detection-baselines/releases/tag/ioqc-sa-rtdetr-l-live>

## 1. 自动保护内容

- 自动读取GPU名称、总显存和启动时空闲显存。
- 24GB、32GB、48GB和80GB显存分别使用不同batch档位。
- 显存峰值低且连续稳定3轮后升一级；达到总显存94%或发生OOM时降一级。
- IOQC-SA的采样统计、IoU、匹配和辅助loss固定使用FP32，主干默认使用AMP。
- 出现 `NONFINITE_LOSS` 时，本轮不会更新优化器；任务回退到最近完整checkpoint，永久切换 `AMP=False` 并降低batch。
- 每完成一个epoch保存 `last.pt` 和独立 `epochN.pt`。
- `last.pt`损坏时自动回退到最新可读取且包含optimizer/EMA的 `epochN.pt`。
- 最近3个有效权重上传到独立GitHub Release，指标和SHA256清单进入 `training-results` 分支。
- SSH断开不影响任务；服务器重启后重新执行启动命令即可继续。
- 默认训练完成后不关机。只有显式设置 `AUTO_SHUTDOWN=1`，且最终GitHub上传校验成功，才会关机。

突然断电最多丢失当前尚未完成的epoch；上一完整epoch仍保存在持久盘和GitHub。

## 2. 租用服务器

推荐Ubuntu 22.04或24.04、单卡RTX 4090/5090、至少16核CPU、64GB内存和120GB持久化磁盘。AutoDL可把持久目录设为 `/root/autodl-tmp`，其他平台替换成对应数据盘路径。

连接后检查：

```bash
nvidia-smi
df -h
```

安装基础工具：

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv tmux curl
```

## 3. 克隆正确分支

```bash
export WORK_ROOT=/root/autodl-tmp/ioqc-sa
export REPO_DIR=$WORK_ROOT/repo
export STORAGE_ROOT=$WORK_ROOT/storage

mkdir -p "$WORK_ROOT"
git clone --branch codex/ioqc-sa --single-branch \
  https://github.com/kkc236/uav-detection-baselines.git "$REPO_DIR"
cd "$REPO_DIR"
git rev-parse --abbrev-ref HEAD
```

最后一条必须输出：

```text
codex/ioqc-sa
```

## 4. 自动配置环境和数据集

```bash
cd "$REPO_DIR"
STORAGE_ROOT="$STORAGE_ROOT" REPO_DIR="$REPO_DIR" \
  bash scripts/setup_ioqc_sa_server.sh
```

安装脚本会：

- 建立持久化venv、数据集、runs、logs和secrets目录；
- 3090/4090安装项目已验证的PyTorch 2.5.1+cu121；
- 5090安装支持Blackwell的PyTorch 2.7.1+cu128，并检查wheel是否包含 `sm_120`；
- 安装Ultralytics 8.4.90；
- 下载并转换VisDrone train/val；
- 输出检测到的GPU信息和初始batch策略。

PyTorch的官方安装矩阵见：<https://pytorch.org/get-started/previous-versions/>

## 5. 配置GitHub Token

创建只授权 `kkc236/uav-detection-baselines` 的fine-grained token，只授予 `Contents: Read and write`。不要把token粘贴到聊天、脚本或命令历史。

```bash
export TOKEN_FILE=$STORAGE_ROOT/secrets/github_token
read -rsp "GitHub token: " GITHUB_TOKEN; echo
printf '%s' "$GITHUB_TOKEN" > "$TOKEN_FILE"
unset GITHUB_TOKEN
chmod 600 "$TOKEN_FILE"
stat -c '%a %n' "$TOKEN_FILE"
```

权限输出必须以 `600` 开头。

## 6. 启动100轮训练

```bash
cd "$REPO_DIR"
mkdir -p "$STORAGE_ROOT/logs"

nohup env \
  STORAGE_ROOT="$STORAGE_ROOT" \
  REPO_DIR="$REPO_DIR" \
  WORKERS=8 \
  AUTO_SHUTDOWN=0 \
  bash scripts/run_ioqc_sa_server.sh \
  > "$STORAGE_ROOT/logs/ioqc_sa_launcher.log" 2>&1 &

echo $!
```

不需要手工填写GPU型号和batch。确需限制初始batch时增加例如 `INITIAL_BATCH=4`，该数值必须属于自动生成的档位。

## 7. 查看运行状态

启动器输出：

```bash
tail -f "$STORAGE_ROOT/logs/ioqc_sa_launcher.log"
```

完整训练输出：

```bash
tail -f "$STORAGE_ROOT/logs/ioqc_sa_training.log"
```

当前epoch、batch、AMP和恢复事件：

```bash
cat "$STORAGE_ROOT/logs/ioqc_sa_status.json"
cat "$STORAGE_ROOT/runs/ioqc-sa/scratch-rtdetr-l-ioqc-sa-100ep/adaptive_state.json"
tail -f "$STORAGE_ROOT/runs/ioqc-sa/scratch-rtdetr-l-ioqc-sa-100ep/batch_history.jsonl"
```

IOQC-SA辅助loss与P3质量：

```bash
tail -f "$STORAGE_ROOT/runs/ioqc-sa/scratch-rtdetr-l-ioqc-sa-100ep/ioqc_sa_diagnostics.jsonl"
```

前10% epoch的 `active_weight` 应为0，10%-15%线性升到1，15%以后保持1。`competition_loss`、`alignment_loss`和贡献项必须始终为有限数值。

GitHub同步：

```bash
cat "$STORAGE_ROOT/logs/ioqc_sa_github_sync.json"
tail -f "$STORAGE_ROOT/logs/ioqc_sa_github_sync.log"
```

GPU和磁盘：

```bash
watch -n 5 nvidia-smi
watch -n 30 df -h "$STORAGE_ROOT"
```

## 8. 停止和重新启动

安全停止整个启动器、supervisor和训练进程组：

```bash
kill -TERM "$(cat "$STORAGE_ROOT/logs/ioqc_sa_launcher.pid")"
```

当前未完成epoch可能丢失，上一完整checkpoint保留。重新启动时再次执行第6节的 `nohup` 命令，脚本会自动校验并选择最新checkpoint。

服务器重启后若旧PID文件仍存在，supervisor会确认进程已不存在，再安全接管；不会因为残留锁永久无法启动。

## 9. 更换服务器并从GitHub继续

在新服务器完成第2至第5节，然后建立相同run名称并下载最新远程权重：

```bash
export RUN_NAME=scratch-rtdetr-l-ioqc-sa-100ep
export RUN_DIR=$STORAGE_ROOT/runs/ioqc-sa/$RUN_NAME
mkdir -p "$RUN_DIR/weights"

"$STORAGE_ROOT/venv/bin/python" scripts/restore_ioqc_sa_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --token-file "$STORAGE_ROOT/secrets/github_token"
```

工具会：

1. 从 `ioqc-sa-rtdetr-l-live` 选择epoch最大的 `ioqc-sa-last-epoch-XXXX.pt`；
2. 先下载到临时文件；
3. 校验文件大小、epoch、optimizer、EMA和SHA256；
4. 原子保存为 `weights/epochN.pt`。

随后执行第6节启动命令。恢复时新服务器的 `project`、`name`、batch和AMP设置会覆盖checkpoint中的旧路径，而optimizer、scheduler、EMA和epoch状态继续保留。

## 10. 完成判定和结果位置

训练完成且最终上传成功时，启动日志出现：

```text
IOQC-SA training and final GitHub publication are verified.
```

本地结果：

```text
$STORAGE_ROOT/runs/ioqc-sa/scratch-rtdetr-l-ioqc-sa-100ep/
```

关键文件：

- `weights/best.pt`
- `weights/last.pt`
- `weights/epochN.pt`
- `results.csv`
- `args.yaml`
- `ioqc_sa_diagnostics.jsonl`
- `batch_history.jsonl`
- `adaptive_state.json`

远程权重名称为 `ioqc-sa-last-epoch-XXXX.pt`，不会与BTD-SE权重混用。

## 11. 常见故障

### `NONFINITE_LOSS`

无需手工处理。supervisor会回退到上一完整epoch、关闭AMP并降低batch。查看：

```bash
grep -i -E 'NONFINITE_LOSS|numeric_fp32_demote' \
  "$STORAGE_ROOT/logs/ioqc_sa_training.log" \
  "$RUN_DIR/batch_history.jsonl"
```

### CUDA OOM

任务会自动降低一级batch。若已降到最低档仍OOM，检查是否有其他进程占用GPU：

```bash
nvidia-smi
```

### 5090提示不支持 `sm_120`

说明镜像或wheel不正确。删除持久化venv后重新运行setup：

```bash
rm -rf "$STORAGE_ROOT/venv"
STORAGE_ROOT="$STORAGE_ROOT" REPO_DIR="$REPO_DIR" \
  bash scripts/setup_ioqc_sa_server.sh
```

### GitHub上传失败

训练不会因此停止。检查token权限和网络，修复后重新执行启动命令；最终一次上传校验未成功时脚本不会宣告完成，也不会自动关机。
