# VSF-RMR 服务器训练说明

本说明用于单张 RTX 4090 或多张 NVIDIA GPU。实验顺序是先跑匹配 baseline，再跑独立 VSF-RMR。两组实验均从零训练 100 epoch，并使用完全相同的数据增强配置。

## 1. 持久盘目录

以 `/root/blockdata` 为例：

```bash
export WORK_ROOT=/root/blockdata/vsf-rmr
export REPO_DIR=$WORK_ROOT/repo
export STORAGE_ROOT=$WORK_ROOT/storage
mkdir -p "$WORK_ROOT"
```

代码、虚拟环境、数据集、训练结果分别保存。删除 Git checkout 不会删除权重、日志、数据集或密钥。

## 2. 下载代码

```bash
git clone --branch codex/vsf-rmr --single-branch \
  https://github.com/kkc236/uav-detection-baselines.git "$REPO_DIR"
cd "$REPO_DIR"
```

如果仓库已经存在：

```bash
cd "$REPO_DIR"
git fetch origin codex/vsf-rmr
git switch codex/vsf-rmr
git pull --ff-only origin codex/vsf-rmr
```

## 3. 安装环境与数据集

```bash
STORAGE_ROOT="$STORAGE_ROOT" REPO_DIR="$REPO_DIR" \
  bash scripts/setup_vsf_rmr_server.sh
```

脚本会自动识别 4090/5090，创建独立 venv，安装匹配的 PyTorch，配置 Ultralytics 数据目录并准备 VisDrone train/val。

## 4. 配置 GitHub Token

创建 fine-grained token，只授权 `kkc236/uav-detection-baselines`，给予 Contents read/write 权限。

```bash
TOKEN_FILE=$STORAGE_ROOT/secrets/github_token
read -rsp "GitHub token: " GITHUB_TOKEN; echo
printf '%s' "$GITHUB_TOKEN" > "$TOKEN_FILE"
unset GITHUB_TOKEN
chmod 600 "$TOKEN_FILE"
```

不要把 token 写入命令历史、代码、日志或聊天。

## 5. 先跑匹配 Baseline

```bash
cd "$REPO_DIR"
nohup env \
  REPO_DIR="$REPO_DIR" \
  STORAGE_ROOT="$STORAGE_ROOT" \
  VARIANT=baseline \
  AUTO_SHUTDOWN=0 \
  bash scripts/run_vsf_rmr_server.sh \
  > "$STORAGE_ROOT/logs/baseline_launcher.log" 2>&1 &
```

4090 默认从总 batch 6 开始，稳定三轮且峰值显存低于 82% 时升到 8；峰值达到 94% 或 OOM 时降一级。数值异常会关闭 AMP 并降一级。多卡时 batch 梯度按 GPU 数自动放大。

## 6. Baseline 完成后跑 VSF-RMR

确认 baseline 状态中的 `completed_epoch` 为 100，再启动：

```bash
cd "$REPO_DIR"
nohup env \
  REPO_DIR="$REPO_DIR" \
  STORAGE_ROOT="$STORAGE_ROOT" \
  VARIANT=vsf-rmr \
  AUTO_SHUTDOWN=0 \
  bash scripts/run_vsf_rmr_server.sh \
  > "$STORAGE_ROOT/logs/vsf-rmr_launcher.log" 2>&1 &
```

需要训练结束且 GitHub 最终上传验证成功后自动关机时，把 `AUTO_SHUTDOWN=0` 改为 `AUTO_SHUTDOWN=1`。关闭 GitHub 同步时不会自动关机。

## 7. 实时监测

Baseline：

```bash
tail -f "$STORAGE_ROOT/logs/baseline/baseline_training.log"
watch -n 5 cat "$STORAGE_ROOT/logs/baseline/baseline_status.json"
watch -n 5 nvidia-smi
```

VSF-RMR：

```bash
tail -f "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_training.log"
watch -n 5 cat "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_status.json"
tail -f "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_github_sync.log"
watch -n 5 nvidia-smi
```

状态字段含当前 batch、AMP、完成 epoch、最近事件和恢复 checkpoint。`promote` 表示升 batch，`oom_demote` 表示 OOM 降级，`numeric_fp32_demote` 表示关闭 AMP 后降级。

## 8. 安全停止与原服务器恢复

查 launcher PID 后发送 TERM：

```bash
cat "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_launcher.pid"
kill -TERM "$(cat "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_launcher.pid")"
```

重新执行第 6 节命令即可。监督器会验证 `last.pt`；若损坏，会自动回退到最新有效 `epochN.pt`。

断开 SSH 或本地网络不会终止 `nohup` 训练。数据下载、训练和 GitHub 上传使用服务器网络。

## 9. 换服务器恢复

在新服务器完成第 1 至 4 节，然后恢复对应实验。

Baseline：

```bash
RUN_DIR=$STORAGE_ROOT/runs/vsf-rmr/scratch-rtdetr-l-vsf-matched-baseline-100ep
$STORAGE_ROOT/venv/bin/python scripts/restore_vsf_rmr_checkpoint.py \
  --variant baseline \
  --run-dir "$RUN_DIR" \
  --token-file "$STORAGE_ROOT/secrets/github_token"
```

VSF-RMR：

```bash
RUN_DIR=$STORAGE_ROOT/runs/vsf-rmr/scratch-rtdetr-l-vsf-rmr-100ep
$STORAGE_ROOT/venv/bin/python scripts/restore_vsf_rmr_checkpoint.py \
  --variant vsf-rmr \
  --run-dir "$RUN_DIR" \
  --token-file "$STORAGE_ROOT/secrets/github_token"
```

恢复器只接受同实验 Release 中配对的 `.pt` 与 `.json`，并验证大小、已完成 epoch 和 SHA-256。验证成功后重新执行对应启动命令。

## 10. 数据保护位置

```text
$STORAGE_ROOT/datasets/VisDrone                 数据集
$STORAGE_ROOT/runs/vsf-rmr/.../weights          last.pt、best.pt、最近3个epoch权重
$STORAGE_ROOT/logs                              训练、同步和状态日志
$STORAGE_ROOT/state                             batch自适应状态
$STORAGE_ROOT/results-checkout                  轻量指标Git结果分支
$STORAGE_ROOT/secrets/github_token              权限600的密钥
```

GitHub Release 滚动保留最近 3 个可恢复 checkpoint 及其 SHA 清单；`training-results` 分支保存 `results.csv`、诊断 JSONL、参数和最新清单。BTD-SE、IOQC-SA、匹配 baseline 与 VSF-RMR 使用不同 Release tag 和 asset 前缀。

