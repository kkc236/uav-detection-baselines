# VSF-RMR创新点3服务器训练说明

本说明用于在单张RTX 4090上独立训练创新点3 VSF-RMR。实验只允许增加VSF-RMR模块及其辅助损失，其余训练参数必须与创新点1 BTD-SE和matched baseline完全一致。不得使用旧版AdamW、自适应batch或自动关闭AMP方案。

代码仓库：<https://github.com/kkc236/uav-detection-baselines>

代码分支：`codex/vsf-rmr`

Checkpoint Release：<https://github.com/kkc236/uav-detection-baselines/releases/tag/vsf-rmr-rtdetr-l-live>

## 1. 固定实验协议

| 参数 | 固定值 |
| --- | --- |
| 模型 | RT-DETR-L + 独立VSF-RMR |
| 数据集 | VisDrone train/val |
| epochs | 100 |
| imgsz | 640 |
| batch | 固定8，单张GPU |
| workers | 8 |
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
| mosaic | 1.0 |
| mixup | 0.0 |
| scale | 0.5 |
| translate | 0.1 |
| perspective | 0.0 |
| max_det | 300 |
| save_period | 1 |
| VSF-RMR辅助损失权重 | 0.1 |

训练期间不允许升降batch，不允许关闭AMP，也不允许修改优化器、学习率和数据增强。发生OOM、NaN或Inf时任务停止并保留最近完整checkpoint，修复资源问题后仍以相同参数恢复。

## 2. 持久化目录

新服务器使用数据盘 `/root/data`：

```bash
export WORK_ROOT=/root/data/vsf-rmr
export REPO_DIR=$WORK_ROOT/repo
export STORAGE_ROOT=$WORK_ROOT/storage
mkdir -p "$WORK_ROOT"
```

虚拟环境、数据集、日志、权重和密钥都放在 `STORAGE_ROOT`，SSH断开或删除Git checkout不会丢失训练数据。

## 3. 清华镜像和基础工具

Ubuntu系统包与普通PyPI依赖使用清华TUNA。PyTorch CUDA精确轮子保留 `download.pytorch.org` 官方源，并使用清华PyPI补充普通依赖。GitHub和VisDrone压缩包没有清华官方镜像，继续使用原地址。

```bash
cp -a /etc/apt/sources.list /etc/apt/sources.list.before-tuna
sed -i \
  -e 's|http://nova.clouds.archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
  -e 's|https\?://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
  -e 's|https\?://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
  /etc/apt/sources.list

apt-get update
apt-get install -y git curl python3 python3-venv
```

## 4. 克隆创新点3分支

```bash
git clone --branch codex/vsf-rmr --single-branch \
  https://github.com/kkc236/uav-detection-baselines.git "$REPO_DIR"

cd "$REPO_DIR"
git branch --show-current
git log -1 --oneline
```

分支必须显示 `codex/vsf-rmr`。

仓库已存在时：

```bash
cd "$REPO_DIR"
git fetch origin codex/vsf-rmr
git switch codex/vsf-rmr
git pull --ff-only origin codex/vsf-rmr
```

## 5. 安装环境和VisDrone

```bash
cd "$REPO_DIR"

STORAGE_ROOT="$STORAGE_ROOT" REPO_DIR="$REPO_DIR" \
  PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  bash scripts/setup_vsf_rmr_server.sh
```

脚本会自动完成：

- 创建独立Python虚拟环境。
- RTX 4090安装PyTorch 2.5.1+cu121。
- RTX 5090安装PyTorch 2.7.1+cu128并检查 `sm_120`。
- 从清华PyPI安装Ultralytics 8.4.90和项目依赖。
- 下载并转换VisDrone train/val。
- 创建权重、日志、状态、密钥和GitHub同步目录。

完成后检查：

```bash
"$STORAGE_ROOT/venv/bin/python" -c \
  "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.get_device_name(0))"
ls "$STORAGE_ROOT/datasets/VisDrone/images/train" | head
```

## 6. 配置GitHub Token

Token使用fine-grained类型，只授权 `kkc236/uav-detection-baselines`，仓库权限为 `Contents: Read and write`。

```bash
export TOKEN_FILE=$STORAGE_ROOT/secrets/github_token
read -rsp "GitHub token: " GITHUB_TOKEN; echo
printf '%s' "$GITHUB_TOKEN" > "$TOKEN_FILE"
unset GITHUB_TOKEN
chmod 600 "$TOKEN_FILE"
stat -c '%a %n' "$TOKEN_FILE"
```

权限必须显示600。Token不得写入代码、Git URL、日志或Git提交。

## 7. 启动创新点3训练

不需要先在本服务器重跑baseline，可直接使用论文统一的matched baseline作为对照。

```bash
cd "$REPO_DIR"

nohup env \
  REPO_DIR="$REPO_DIR" \
  STORAGE_ROOT="$STORAGE_ROOT" \
  VARIANT=vsf-rmr \
  DEVICE=0 \
  AUTO_SHUTDOWN=0 \
  bash scripts/run_vsf_rmr_server.sh \
  > "$STORAGE_ROOT/logs/vsf-rmr_launcher.log" 2>&1 &

echo $!
```

脚本内部强制 `batch=8`、单GPU、AMP=True和固定匹配协议。外部环境变量不能把优化器、学习率或batch改回旧配置。

## 8. 启动后必须核对

```bash
tail -n 120 "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_training.log"
cat "$STORAGE_ROOT/state/vsf_rmr_adaptive_state.json"
```

必须确认：

```text
batch=8
amp=True
pretrained=False
optimizer: MuSGD(lr=0.01
```

训练生成 `args.yaml` 后再次核对：

```bash
export RUN_DIR=$STORAGE_ROOT/runs/vsf-rmr/scratch-rtdetr-l-vsf-rmr-100ep
grep -E "^(batch|amp|optimizer|lr0|lrf|momentum|weight_decay|warmup_epochs|pretrained|seed|mosaic|mixup|scale|translate):" \
  "$RUN_DIR/args.yaml"
```

任一参数不一致都应停止实验，该结果不能进入论文公平对比。

## 9. 实时监测

```bash
tail -f "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_training.log"
```

```bash
watch -n 5 cat "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_status.json"
```

```bash
watch -n 5 nvidia-smi
```

```bash
tail -f "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_github_sync.log"
```

```bash
tail -n 5 "$RUN_DIR/results.csv"
```

状态中的batch必须始终为8，AMP必须始终为True。SSH或本地网络断开不会停止 `nohup` 训练。

## 10. Checkpoint和数据保护

每完成一个epoch：

1. 保存可恢复的 `last.pt` 和独立 `epochN.pt`。
2. 校验checkpoint中的optimizer、EMA和epoch状态。
3. 上传到GitHub Release `vsf-rmr-rtdetr-l-live`。
4. GitHub滚动保留最近3个checkpoint及SHA256清单。
5. 将 `results.csv`、`args.yaml`、VSF-RMR诊断和最新状态提交到 `training-results` 分支。

本地数据位置：

```text
$STORAGE_ROOT/datasets/VisDrone
$STORAGE_ROOT/runs/vsf-rmr/scratch-rtdetr-l-vsf-rmr-100ep
$STORAGE_ROOT/logs/vsf-rmr
$STORAGE_ROOT/state/vsf_rmr_adaptive_state.json
$STORAGE_ROOT/results-checkout
$STORAGE_ROOT/secrets/github_token
```

## 11. 安全停止与恢复

```bash
kill -TERM "$(cat "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_launcher.pid")"
```

重新执行第7节启动命令即可从最新完整checkpoint恢复。若 `last.pt` 损坏，恢复器会回退到最新有效 `epochN.pt`。batch、AMP、优化器和数据增强仍保持固定值。

换服务器时，完成第2至第6节后执行：

```bash
export RUN_DIR=$STORAGE_ROOT/runs/vsf-rmr/scratch-rtdetr-l-vsf-rmr-100ep

"$STORAGE_ROOT/venv/bin/python" scripts/restore_vsf_rmr_checkpoint.py \
  --variant vsf-rmr \
  --run-dir "$RUN_DIR" \
  --token-file "$STORAGE_ROOT/secrets/github_token"
```

校验成功后重新执行第7节。

## 12. 异常处理

普通SSH关闭、断网或外部进程中断：最多以完全相同参数自动重试3次。

OOM、NaN、Inf或非有限损失：立即停止，保留最近完整checkpoint和错误日志；不得降低batch、关闭AMP或改变优化器。先排查其他GPU进程、磁盘和硬件问题，再按相同协议恢复。

## 13. 完成验收

```bash
wc -l "$RUN_DIR/results.csv"
tail -n 2 "$RUN_DIR/results.csv"
cat "$STORAGE_ROOT/logs/vsf-rmr/vsf_rmr_github_sync.json"
```

正式结果必须满足：

- `results.csv`包含表头和100轮数据。
- batch始终为8，AMP始终为True。
- 实际优化器为MuSGD，初始学习率0.01。
- 不加载预训练权重。
- 数据增强与创新点1和matched baseline一致。
- 第100轮checkpoint可读取且GitHub SHA256校验成功。

最终创新点3增益统一计算为：

```text
VSF-RMR指标 - matched baseline指标
```
