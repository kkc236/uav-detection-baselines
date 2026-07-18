# RT-DETR-L Matched Baseline 新服务器运行指南

本指南用于在一台新租用的GPU服务器上，从零训练与创新点1 BTD-SE完全匹配的原始RT-DETR-L baseline。该结果同时作为创新点1和创新点2 IOQC-SA的共同对照。

代码仓库：<https://github.com/kkc236/uav-detection-baselines>

代码分支：`codex/matched-baseline`

GitHub checkpoint Release：<https://github.com/kkc236/uav-detection-baselines/releases/tag/rtdetr-l-btdse-matched-baseline-live>

## 1. 固定实验协议

baseline只使用原始 `rtdetr-l.yaml`，不包含BTD-SE、IOQC-SA或VSF-RMR。

| 参数 | 固定值 |
| --- | --- |
| 数据集 | VisDrone train/val |
| epochs | 100 |
| imgsz | 640 |
| batch | 固定8 |
| pretrained | False |
| optimizer | `auto`，Ultralytics 8.4.90实际选择MuSGD |
| lr0 | 0.01 |
| lrf | 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| warmup_epochs | 3.0 |
| AMP | 固定True |
| seed | 0 |
| deterministic | True |
| nbs | 64 |
| workers | 8 |
| mosaic | 1.0 |
| mixup | 0.0 |
| scale | 0.5 |
| translate | 0.1 |
| max_det | 300 |

训练期间不允许自动升降batch，也不允许自动关闭AMP。发生OOM或非有限损失时，任务会停止并保留最近完整checkpoint，不会静默改变实验参数。

## 2. 服务器要求

推荐配置：

- Ubuntu 22.04或24.04。
- 单张RTX 4090、5090、A100或同等级NVIDIA GPU。
- 至少16核CPU、64GB内存。
- 至少100GB持久化磁盘空间。
- 可访问GitHub和PyTorch软件源。

即使服务器有多张GPU，本实验也建议只使用一张卡，避免改变全局batch。以下命令默认使用 `DEVICE=0`。

首先检查服务器：

```bash
nvidia-smi
df -h
```

### 2.1 将Ubuntu APT切换为清华TUNA

先备份APT配置，然后替换Ubuntu官方软件源。云平台自带的NVIDIA/CUDA专用源不会被修改。

```bash
if [[ -f /etc/apt/sources.list ]]; then
  sudo cp -a /etc/apt/sources.list /etc/apt/sources.list.before-tuna
  sudo sed -i \
    -e 's|https\?://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
    -e 's|https\?://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
    -e 's|https\?://ports.ubuntu.com/ubuntu-ports|https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports|g' \
    /etc/apt/sources.list
fi

if [[ -f /etc/apt/sources.list.d/ubuntu.sources ]]; then
  sudo cp -a /etc/apt/sources.list.d/ubuntu.sources \
    /etc/apt/sources.list.d/ubuntu.sources.before-tuna
  sudo sed -i \
    -e 's|https\?://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
    -e 's|https\?://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
    -e 's|https\?://ports.ubuntu.com/ubuntu-ports|https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports|g' \
    /etc/apt/sources.list.d/ubuntu.sources
fi

sudo apt-get update
sudo apt-get install -y git python3 python3-venv curl
```

确认Ubuntu源已经指向清华：

```bash
grep -R "mirrors.tuna.tsinghua.edu.cn" \
  /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null
```

### 2.2 下载源使用边界

以下可替换内容均使用清华源：

- Ubuntu系统包：`https://mirrors.tuna.tsinghua.edu.cn/ubuntu`
- Python普通依赖：`https://pypi.tuna.tsinghua.edu.cn/simple`

以下内容保留官方地址：

- PyTorch CUDA轮子必须从 `download.pytorch.org` 获取精确的 `+cu121` 或 `+cu128` 构建；依赖包仍由清华PyPI补充。
- GitHub仓库和GitHub Release没有清华官方的任意仓库镜像。
- VisDrone数据压缩包没有清华官方镜像，下载脚本继续使用Ultralytics资源地址并支持断点续传。

不要使用来源不明的GitHub代理替换代码和checkpoint下载，否则无法保证文件完整性。

## 3. 克隆专用分支

将 `WORK_ROOT` 放在服务器的持久化数据盘。不同平台可以把 `/root/data` 换成 `/root/autodl-tmp`、`/workspace` 或实际数据盘路径。

```bash
export WORK_ROOT=/root/data/matched-baseline
export REPO_DIR=$WORK_ROOT/repo
export STORAGE_ROOT=$WORK_ROOT/storage

mkdir -p "$WORK_ROOT"
git clone --branch codex/matched-baseline --single-branch \
  https://github.com/kkc236/uav-detection-baselines.git "$REPO_DIR"

cd "$REPO_DIR"
git branch --show-current
git log -1 --oneline
```

分支必须显示：

```text
codex/matched-baseline
```

## 4. 自动配置环境和数据集

```bash
cd "$REPO_DIR"

STORAGE_ROOT="$STORAGE_ROOT" REPO_DIR="$REPO_DIR" \
  PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  bash scripts/setup_matched_baseline_server.sh
```

脚本会自动完成：

- 创建独立Python虚拟环境。
- 4090等GPU从PyTorch官方CUDA仓库安装PyTorch 2.5.1+cu121。
- 5090从PyTorch官方CUDA仓库安装PyTorch 2.7.1+cu128并检查 `sm_120`。
- 从清华PyPI安装Ultralytics 8.4.90及其他项目依赖。
- 下载并转换VisDrone train/val数据集。
- 创建runs、logs、secrets和GitHub同步目录。

完成后检查：

```bash
ls "$STORAGE_ROOT/datasets/VisDrone/images/train" | head
"$STORAGE_ROOT/venv/bin/python" -c \
  "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.get_device_name(0))"
```

## 5. 配置GitHub Token

创建只授权 `kkc236/uav-detection-baselines` 的fine-grained token，仓库权限设置为 `Contents: Read and write`。不要把token直接写进脚本、Git URL或聊天记录。

```bash
export TOKEN_FILE=$STORAGE_ROOT/secrets/github_token

read -rsp "GitHub token: " GITHUB_TOKEN; echo
printf '%s' "$GITHUB_TOKEN" > "$TOKEN_FILE"
unset GITHUB_TOKEN
chmod 600 "$TOKEN_FILE"

stat -c '%a %n' "$TOKEN_FILE"
```

权限必须显示为 `600`。

## 6. 启动100轮训练

```bash
cd "$REPO_DIR"

nohup env \
  STORAGE_ROOT="$STORAGE_ROOT" \
  REPO_DIR="$REPO_DIR" \
  DEVICE=0 \
  BATCH=8 \
  ENABLE_GITHUB_SYNC=1 \
  AUTO_SHUTDOWN=0 \
  bash scripts/run_matched_baseline_server.sh \
  > "$STORAGE_ROOT/logs/matched_baseline_launcher.log" 2>&1 &

echo $!
```

SSH断开不会停止训练。服务器流量、CPU、GPU和磁盘均由租用服务器使用，本地电脑不需要保持联网。

## 7. 首次启动必须核对

```bash
grep -E "batch=8|amp=True|optimizer:|pretrained=False" \
  "$STORAGE_ROOT/logs/matched_baseline_training.log" | head -n 20
```

必须看到：

```text
batch=8
amp=True
pretrained=False
optimizer: MuSGD(lr=0.01
```

如果出现其他batch、AMP=False、AdamW或加载预训练权重，应立即停止，该结果不能作为公平baseline。

## 8. 实时查看训练

训练日志：

```bash
tail -f "$STORAGE_ROOT/logs/matched_baseline_training.log"
```

GPU状态：

```bash
watch -n 5 nvidia-smi
```

最近指标：

```bash
tail -n 5 \
  "$STORAGE_ROOT/runs/matched-baseline/scratch-rtdetr-l-btdse-matched-baseline-100ep/results.csv"
```

GitHub上传状态：

```bash
cat "$STORAGE_ROOT/logs/matched_baseline_github_sync.json"
tail -n 20 "$STORAGE_ROOT/logs/matched_baseline_github_sync.log"
```

守护与恢复日志：

```bash
tail -f "$STORAGE_ROOT/logs/matched_baseline_supervisor.log"
```

## 9. 每轮checkpoint保护

每完成一个epoch，系统执行以下保护：

1. 在持久化磁盘写入 `last.pt` 和独立的 `epochN.pt`。
2. 校验checkpoint包含可恢复的optimizer和EMA状态。
3. 上传到独立GitHub Release `rtdetr-l-btdse-matched-baseline-live`。
4. GitHub远端滚动保留最近3个完整checkpoint及其SHA256清单。
5. 将 `results.csv`、`args.yaml` 和 `latest.json` 提交到 `training-results` 分支。
6. 新checkpoint上传并校验成功后，才清理更旧的本地epoch权重。

权重名称示例：

```text
matched-baseline-last-epoch-0001.pt
matched-baseline-last-epoch-0002.pt
matched-baseline-last-epoch-0003.pt
```

## 10. 暂停与原服务器恢复

安全暂停：

```bash
kill -TERM "$(cat "$STORAGE_ROOT/logs/matched_baseline_launcher.pid")"
```

检查GPU进程已经退出：

```bash
nvidia-smi
```

再次执行第6节的启动命令即可恢复。脚本会寻找最新可读取的 `last.pt` 或 `epochN.pt`，并进行真正断点恢复。optimizer、学习率调度、EMA和AMP scaler状态都会继续，不会从头初始化。

## 11. 更换服务器后恢复

在新服务器执行第3至5节，然后下载并校验GitHub上最新checkpoint：

```bash
export RUN_DIR=$STORAGE_ROOT/runs/matched-baseline/scratch-rtdetr-l-btdse-matched-baseline-100ep

"$STORAGE_ROOT/venv/bin/python" \
  scripts/restore_vsf_rmr_checkpoint.py \
  --variant baseline \
  --run-dir "$RUN_DIR" \
  --token-file "$STORAGE_ROOT/secrets/github_token" \
  --tag rtdetr-l-btdse-matched-baseline-live \
  --asset-prefix matched-baseline-last
```

恢复脚本会同时校验epoch、文件大小和SHA256。校验成功后，重新执行第6节启动命令。

## 12. 故障规则

普通断网、SSH关闭或外部进程中断：

- 保留本地和GitHub checkpoint。
- 最多自动使用相同配置重启3次。
- 不改变batch、AMP、优化器或学习率。

OOM、NaN、Inf或 `NONFINITE_LOSS`：

- 立即停止当前实验。
- 不自动降低batch。
- 不自动关闭AMP。
- 保留最近完整checkpoint和错误日志。
- 排查服务器是否有其他GPU进程；解决后仍须保持固定协议。

## 13. 完成验收

```bash
export RUN_DIR=$STORAGE_ROOT/runs/matched-baseline/scratch-rtdetr-l-btdse-matched-baseline-100ep

wc -l "$RUN_DIR/results.csv"
tail -n 2 "$RUN_DIR/results.csv"
grep -E "^(batch|amp|optimizer|lr0|momentum|pretrained|seed):" "$RUN_DIR/args.yaml"
cat "$STORAGE_ROOT/logs/matched_baseline_github_sync.json"
```

正式论文结果必须满足：

- `results.csv`包含表头和100轮数据。
- batch始终为8。
- AMP始终为True。
- 实际优化器为MuSGD，初始学习率0.01。
- 不加载预训练权重。
- `last.pt`记录第100轮完成状态并可读取。
- GitHub最终checkpoint及SHA256清单上传成功。

## 14. 论文使用方式

创新点1增益：

```text
BTD-SE指标 - 本matched baseline指标
```

创新点2增益：

```text
新IOQC-SA指标 - 本matched baseline指标
```

旧baseline及旧IOQC-SA结果只能作为预实验，不进入论文最终主结果表。
