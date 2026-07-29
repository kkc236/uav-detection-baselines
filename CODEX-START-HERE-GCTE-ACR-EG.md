# Codex 单文件执行包：GCTE-RTDETR / ACR-EG 新服务器续训至 100 Epoch

> 本文件既是项目交接，也是给下一位 Codex 的直接执行指令。用户把本文件和新服务器登录信息放在同一条消息中后，Codex 应读取全文并自主执行，不要只总结、评价或重新制定一个空泛方案。
>
> **单一权威入口：** 新 Codex 只需要读取这一份 Markdown；本文已经汇总项目状态、模块设计、真实证据、全部下载源与落盘路径、镜像回退、校验硬门、续训修复、正式训练、逐轮备份和最终评测。仓库中的旧 handoff 只保留历史记录，不作为执行依赖。

---

## 0. 给执行 Codex 的最高优先级指令

你正在接手一个真实的 RT-DETR-L 无人机小目标检测实验。

你的任务终点是：

```text
1. 在用户提供的全新 RTX 4090 服务器上完成环境和数据部署；
2. fresh verify 已实现的真正 YAML 集成 ACR-EG 安全续训，并通过真实 GPU smoke；
3. 从已备份的完成第 9 轮 checkpoint 继续第 10–100 轮；
4. 证明运行模型仍是 ACREGDetectionModel，而不是 stock RT-DETR；
5. 每个完成轮次都上传并校验 GitHub Release；
6. 完成 100 epoch 后运行真实 548 图端到端多视图评测；
7. 输出相对 Global baseline 的全部指标、速度和证据；
8. 未到上述终点前不要把训练 loss 或缓存诊断称为正式结果。
```

执行风格：

- 采取行动，不要只解释本文件；
- 先只读检查，再修改或启动；
- 遇到工程错误时使用 systematic debugging；
- 修改代码时先写失败测试，再写最小实现；
- 完成前运行 fresh verification；
- 服务器长任务运行时持续给用户简短真实进度；
- 不得因 SSH 暂时无输出而重复启动；
- 不得关机、重启或停止仍在运行的训练；
- 不得杀无关进程；
- 不得在聊天、日志、Git、脚本或 shell history 中输出密码和 token；
- 不得放宽科学门、改数据、改训练参数或把 stock checkpoint 冒充方法 checkpoint。
- 旧服务器 `36.103.199.151` 已失联，只是历史来源，不是迁移依赖；
- 不得再把“提供旧服务器密码”作为新服务器部署的阻塞项；
- 新服务器缺少 VisDrone 时，必须按本文的公开源/镜像下载、仓库脚本转换和签名硬门自动重建。

如果用户消息中没有新服务器 IP、用户名、端口或认证信息，这是唯一允许立即询问的阻塞项。用户提供后，不再反复询问已经能够从代码、GitHub或服务器发现的信息。

---

## 1. 权威项目身份

GitHub：

```text
repository = kkc236/uav-detection-baselines
branch = codex/gcte-rtdetr-g0
handoff document authority = clone 后的 codex/gcte-rtdetr-g0 分支 HEAD
last pre-offline handoff commit = 0a3a0312
genuine integrated source base = a22838e3e7cd1cd858d6aad9f42e5b68fab50471
safe integrated resume implementation = a6f2cdd9
checkpoint publisher implementation = 2f1e2279
real resume smoke implementation = d1403fe7ed7f67f320817fa4ed9075689244e6e8
```

`a22838e3` 是最初真正 YAML/forward/loss/optimizer/checkpoint 集成的训练基线；`a6f2cdd9`、`2f1e2279` 和 `d1403fe7` 随后补齐安全续训、逐轮发布和真实 batch 恢复 smoke。本文档提交只允许改交接内容，不得再次改变这套已验证代码。

在新服务器上从官方 GitHub 克隆。实测第三方 Git 代理可能返回滞后的 branch HEAD，因此代码仓库不走镜像；只有具备 SHA256、字节数或内容签名硬门的大文件才走镜像：

```bash
set -euo pipefail

export GCTE_REPO_ROOT=/home/ubuntu/gcte-acr-eg

git clone --single-branch \
  --branch codex/gcte-rtdetr-g0 \
  https://github.com/kkc236/uav-detection-baselines.git \
  "$GCTE_REPO_ROOT"

cd "$GCTE_REPO_ROOT"
git status --short
git rev-parse HEAD
test -f CODEX-START-HERE-GCTE-ACR-EG.md
test -f requirements-gcte-acr-eg-cu121.txt
test -f configs/rtdetr-l-acr-eg.yaml
test -f scripts/train_gcte_formal.py
test -f src/rtdetr_acr_eg.py

git diff --quiet \
  d1403fe7ed7f67f320817fa4ed9075689244e6e8 \
  HEAD \
  -- scripts src configs tests
```

最后一条必须退出 0。否则停止并报告 source identity 漂移。

---

## 2. 最新可靠训练状态（历史快照，不是迁移依赖）

旧服务器：

```text
host = 36.103.199.151
user = ubuntu
port = 22
```

不要把旧密码写入任何文件。

旧实例当前已经停机或失联。新服务器部署不得尝试通过 SSH “开机”，也不得要求用户恢复旧实例后才继续。恢复所需的代码、成熟 baseline 和完成第 9 轮 checkpoint 均已有 GitHub 权威副本；VisDrone 按第 6、9 节从公开源重建。

最后一次成功只读快照：

```text
time = 2026-07-28T18:16:00+08:00
runner PID = 31100
process = running
latest completed human epoch = 9
current human epoch = 10
latest checkpoint = epoch8.pt
```

随后在 18:33 连续两次 SSH protocol banner 不可用。因此：

```text
不能断言旧服务器训练失败；
不能断言旧服务器继续运行；
完成第 9 轮是最后可靠且已三方校验的恢复点。
```

训练期：

```text
val=False
```

所以 `results.csv` 的 precision、recall 和 mAP 全为 0 是占位值，不是性能。

第 9 轮训练记录：

```text
epoch = 9
time_seconds = 7320.23
train/giou_loss = 1.54419
train/cls_loss = 0.28404
train/l1_loss = 0.20044
```

---

## 3. 权威 checkpoint

### 3.1 成熟 Global baseline

下载：

```text
https://github.com/kkc236/uav-detection-baselines/releases/download/rtdetr-l-btdse-matched-baseline-live/matched-baseline-best-epoch-0100.pt
```

文件：

```text
matched-baseline-best-epoch-0100.pt
```

大小：

```text
66262262 bytes
```

SHA256：

```text
54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B
```

### 3.2 最新真正集成 ACR-EG checkpoint

Release：

```text
https://github.com/kkc236/uav-detection-baselines/releases/tag/gcte-acr-eg-a22838e3-epoch-009
```

下载：

```text
https://github.com/kkc236/uav-detection-baselines/releases/download/gcte-acr-eg-a22838e3-epoch-009/epoch8.pt
```

身份：

```text
completed human epoch = 9
checkpoint epoch = 8
asset = epoch8.pt
bytes = 205325084
SHA256 = 802D72326F4B8FEE55C0FF8818A5B96B7445CBEE34F5C1ED9002A6D3E6771FE6
GitHub digest = sha256:802d72326f4b8fee55c0ff8818a5b96b7445cbee34f5c1ed9002a6d3e6771fe6
```

已知 checkpoint contract：

```text
checkpoint["model"] = None
checkpoint["ema"] = ACREGDetectionModel
checkpoint["optimizer"] 存在
checkpoint["scaler"] 存在
checkpoint["epoch"] = 8
checkpoint["updates"] 存在
EMA state_dict 中存在 48 个 acr_eg.* key
```

`model=None` 是 Ultralytics 8.4.90 保存方式的正常行为；恢复权威是 `ema`。

较早恢复点：

```text
epoch 4:
https://github.com/kkc236/uav-detection-baselines/releases/tag/gcte-acr-eg-a22838e3-epoch-004

epoch 7:
https://github.com/kkc236/uav-detection-baselines/releases/tag/gcte-acr-eg-a22838e3-epoch-007
```

---

## 4. 模型与算法，不得重新解释为后处理

严格分类：

```text
A：查询级网络结构融合模块
```

整体：

```text
GCTE-RTDETR
Geometry-Canonical Tiny-Evidence RT-DETR
```

创新点 1 完整模块：

```text
GCQF
Geometry-Canonical Query Fusion
```

三阶段：

```text
Stage 1 = GeometryQueryProjector
Stage 2 = GlobalLocalQueryInteraction
Stage 3 = AnchorConditionedResidualEvidenceGate / ACR-EG
```

总体前向：

```text
原始图像
├── global 640 view
│   └── shared RT-DETR-L
│       └── final decoder queries Qg
└── four local high-resolution views
    └── the same shared RT-DETR-L
        └── final decoder queries Ql

Ql + geometry
→ GeometryQueryProjector
→ GlobalLocalQueryInteraction with Qg
→ AnchorConditionedResidualEvidenceGate
→ global_retain_logits
→ inject into final non-denoising decoder-query class logits
→ stock RT-DETR matching and detection criterion
```

它满足网络结构模块标准：

- 输入是 Decoder Query 和中间几何证据；
- 继承 `nn.Module`；
- 有可训练参数；
- 主模型 forward/loss 内调用；
- detection loss 可反向传播；
- 进入 state_dict、MuSGD optimizer、EMA 和 checkpoint；
- 不是读取最终 boxes/scores 后独立重排。

关键代码：

```text
configs/rtdetr-l-acr-eg.yaml
src/gcqf.py
src/rtdetr_acr_eg.py
src/gcte_formal_trainer.py
src/gcmv_data.py
src/acr_eg_integration.py
scripts/train_gcte_formal.py
```

YAML 硬门：

```yaml
gcte:
  enabled: true
  forward_integration: true
  query_dim: 256
  num_classes: 10
  num_heads: 8
  num_views: 4
  residual_eta: 0.2
  residual_enabled: true
  acr_eg_off: false
  gcte_off: false
```

当前正式实现的边界：

1. 正式 detection path 主要使用 `global_retain_logits`；
2. `score_residual` 和 `adjusted_local_scores` 尚未进入最终 logits/boxes；
3. local boxes 尚未直接进入最终预测集合；
4. 需要最终做逐参数真实 loss 梯度审计；
5. 需要核验 mosaic/flip/scale 下真实几何映射；
6. 当前训练期 `best.pt` 不是验证集最优；
7. 需要新的 live checkpoint evaluator，不能把旧 cache evaluator 当正式评测。

不要在本次续训前擅自扩展这些研究范围。当前优先级是保留真实集成模型身份并完成连续训练。

---

## 5. 已有实验事实

### 5.1 旧 SR-PEG 相对 Global

```text
mAP50-95 = +0.0105948
AP-tiny = +0.0134117
tiny recall = +0.0543382
AP-medium = -0.0016400
AP-large = -0.0000959
```

旧版相对内部 Fixed anchor：

```text
mAP50-95 = -0.0165785
```

失败根因：

```text
Fixed accepted local = 120326
old Full accepted local = 23283
empty max_det slots = 7440
```

### 5.2 ACR-EG 冻结 Query cache 诊断

Global：

```text
mAP50-95 = 0.19869887
AP50 = 0.36180752
AP75 = 0.18530324
AP-tiny-SBR = 0.08078923
tiny recall = 0.58207909
AP-medium-SBR = 0.25780549
AP-large-SBR = 0.15464286
```

Full-GCQF：

```text
mAP50-95 = 0.20889331
AP50 = 0.38997759
AP75 = 0.18984129
AP-tiny-SBR = 0.09400049
tiny recall = 0.64029069
AP-medium-SBR = 0.25628104
AP-large-SBR = 0.15455318
```

Full-GCQF minus Global：

```text
mAP50-95 = +0.01019444
AP-tiny-SBR = +0.01321127
tiny recall = +0.05821160
AP-medium-SBR = -0.00152445
AP-large-SBR = -0.00008967
```

这只是冻结 Query cache 诊断，不是当前 `epoch8.pt` 的 live 端到端结果。

正式主比较：

```text
Full GCQF / ACR-EG vs Global RT-DETR-L
```

Fixed-SADED 是内部开发锚点，不是外部公开 baseline。

---

## 6. 冻结数据

```text
dataset = VisDrone
train images = 6471
val images = 548
classes = 10
dataset root = /mnt/uav/datasets/VisDrone
data yaml = /mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml
```

新服务器需要的实际目录：

| 内容 | 新服务器目标路径 | 是否必需 |
|---|---|---|
| 数据根目录 | `/mnt/uav/datasets/VisDrone` | 必需 |
| 原始 train 压缩包 | `/mnt/uav/datasets/VisDrone/VisDrone2019-DET-train.zip` | 重建时必需 |
| 原始 val 压缩包 | `/mnt/uav/datasets/VisDrone/VisDrone2019-DET-val.zip` | 重建时必需 |
| train 图像 | `/mnt/uav/datasets/VisDrone/images/train` | 必需 |
| train 标签 | `/mnt/uav/datasets/VisDrone/labels/train` | 必需 |
| train ignore 标签 | `/mnt/uav/datasets/VisDrone/labels_ignore/train` | 评测语义必需 |
| val 图像 | `/mnt/uav/datasets/VisDrone/images/val` | 必需 |
| val 标签 | `/mnt/uav/datasets/VisDrone/labels/val` | 必需 |
| val ignore 标签 | `/mnt/uav/datasets/VisDrone/labels_ignore/val` | 评测语义必需 |
| 数据 YAML | `/mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml` | 必需 |
| 固定 train10 清单 | `/mnt/uav/protocols/tsgr-p2-e1/d2-train-10pct.txt` | 仅旧诊断需要；正式第 9→100 轮续训不需要 |

类：

```text
0 pedestrian
1 people
2 bicycle
3 car
4 van
5 truck
6 tricycle
7 awning-tricycle
8 bus
9 motor
```

签名：

```text
full dataset:
FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB

val 548:
A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A

train10 647:
52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0
```

数据 YAML 必须是：

```yaml
path: /mnt/uav/datasets/VisDrone
train: /mnt/uav/datasets/VisDrone/images/train
val: /mnt/uav/datasets/VisDrone/images/val
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

精确 YAML 字节内容如下，末尾必须保留一个 LF 换行：

```bash
mkdir -p /mnt/uav/protocols/tsgr-p2-e1

cat > /mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml <<'YAML'
path: /mnt/uav/datasets/VisDrone
train: /mnt/uav/datasets/VisDrone/images/train
val: /mnt/uav/datasets/VisDrone/images/val
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
YAML

echo \
  "7EB91FCEF62A687A26A8EF76E9075B9793B52BC8BB110E4235FACF3E2B958324  /mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml" \
  | sha256sum -c -
```

旧服务器不再是数据源。默认恢复路径是：

```text
Ultralytics 权威 VisDrone train/val Release 资产
→ 镜像或官方 URL 下载到固定压缩包路径
→ 当前仓库 scripts/prepare_visdrone.py 确定性转换
→ 文件数、YAML SHA256、val 内容签名全部通过
→ 才允许恢复 integrated checkpoint
```

只需要 train 和 val；不要为了正式续训额外下载 test-dev。

文件数硬门：

```bash
test "$(find /mnt/uav/datasets/VisDrone/images/train -type f | wc -l)" -eq 6471
test "$(find /mnt/uav/datasets/VisDrone/labels/train -type f | wc -l)" -eq 6471
test "$(find /mnt/uav/datasets/VisDrone/images/val -type f | wc -l)" -eq 548
test "$(find /mnt/uav/datasets/VisDrone/labels/val -type f | wc -l)" -eq 548
test "$(
  find \
    /mnt/uav/datasets/VisDrone/images/train \
    /mnt/uav/datasets/VisDrone/labels/train \
    /mnt/uav/datasets/VisDrone/images/val \
    /mnt/uav/datasets/VisDrone/labels/val \
    -type f | wc -l
)" -eq 14038
```

val 签名硬门：

```bash
cd /home/ubuntu/gcte-acr-eg

/mnt/uav/venv/bin/python - <<'PY'
from pathlib import Path
from src.sbr_artifacts import dataset_content_signature

signature = dataset_content_signature(
    Path("/mnt/uav/datasets/VisDrone"),
    "val",
)
print(signature)
assert signature.upper() == "A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A"
PY
```

---

## 7. 冻结训练参数

```text
Ultralytics = 8.4.90
GPU = RTX 4090 24 GB
Python = 3.10.12
PyTorch = 2.5.1+cu121
TorchVision = 0.20.1+cu121
CUDA runtime = 12.1

epochs total = 100
imgsz = 640
batch = 8
workers = 8
device = 0
seed = 0
deterministic = True
cache = False
pretrained = False

AMP = True
fixed AMP scale = 128

optimizer = MuSGD
lr0 = 0.01
lrf = 0.01
momentum = 0.937
weight_decay = 0.0005
warmup_epochs = 3.0
warmup_momentum = 0.8
warmup_bias_lr = 0.0
nbs = 64
cos_lr = False

queries = 300
max_det = 300
NMS = False

mosaic = 1.0
close_mosaic = 10
mixup = 0.0
scale = 0.5
translate = 0.1
degrees = 0.0
shear = 0.0
perspective = 0.0
flipud = 0.0
fliplr = 0.5
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
cutmix = 0.0
copy_paste = 0.0

save_period = 1
val during train = False
```

续训只能恢复同一个 100-epoch schedule，从完成第 9 轮进入第 10 轮。不能把它变成“再训练 100 轮”。

---

## 8. 新服务器从零环境部署

目标：

```text
Ubuntu 22.04 x86_64
RTX 4090
可用磁盘至少 80 GB，推荐 100 GB
```

先检查：

```bash
set -euo pipefail
uname -a
lsb_release -a
nvidia-smi
df -h
free -h
```

Ubuntu 22.04 可使用清华 APT 镜像，但 security 保留官方源：

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.gcte-backup

sudo sed -i \
  -e 's@http://archive.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu@g' \
  -e 's@http://cn.archive.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu@g' \
  /etc/apt/sources.list

sudo apt-get update
sudo apt-get install -y \
  git \
  gh \
  curl \
  rsync \
  ca-certificates \
  python3.10 \
  python3.10-venv \
  python3-pip \
  build-essential
```

目录：

```bash
export GCTE_ENV_ROOT=/mnt/uav/venv
export GCTE_DATA_ROOT=/mnt/uav/datasets/VisDrone
export GCTE_PROTOCOL_ROOT=/mnt/uav/protocols/tsgr-p2-e1
export GCTE_BASELINE=/home/ubuntu/matched-baseline-best-epoch-0100.pt
export GCTE_CHECKPOINT=/home/ubuntu/gcte-checkpoints/epoch8.pt
export GCTE_OUTPUT_ROOT=/home/ubuntu/gcte-acr-eg-resume-output
export GCTE_DOWNLOAD_CACHE=/mnt/uav/download-cache
export PIP_CACHE_DIR=/mnt/uav/pip-cache

sudo mkdir -p /mnt/uav/datasets /mnt/uav/protocols
sudo chown -R ubuntu:ubuntu /mnt/uav
mkdir -p /home/ubuntu/gcte-checkpoints
mkdir -p "$GCTE_OUTPUT_ROOT"
mkdir -p "$GCTE_DOWNLOAD_CACHE" "$PIP_CACHE_DIR"
```

Python：

```bash
python3.10 -m venv "$GCTE_ENV_ROOT"
"$GCTE_ENV_ROOT/bin/python" -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --upgrade pip
"$GCTE_ENV_ROOT/bin/python" -m pip config set \
  global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

cd "$GCTE_REPO_ROOT"
"$GCTE_ENV_ROOT/bin/python" -m pip install \
  --retries 10 \
  --timeout 120 \
  -r requirements-gcte-acr-eg-cu121.txt
```

`requirements-gcte-acr-eg-cu121.txt` 已固定旧服务器的 54 项运行环境。PyTorch 和 TorchVision 使用阿里云 CUDA 12.1 wheel 与 wheel SHA；其余 PyPI 包使用清华镜像。

只有当完整 requirements 安装明确失败在阿里云的 `torch`/`torchvision` wheel 下载时，才使用官方 PyTorch 回退。先从锁文件生成一个只去掉这两行 URL wheel 的副本，再安装官方 CUDA 12.1 wheel 和其余精确依赖：

```bash
grep -vE '^(torch|torchvision) @ ' \
  requirements-gcte-acr-eg-cu121.txt \
  > "$GCTE_DOWNLOAD_CACHE/requirements-gcte-acr-eg-no-torch.txt"

"$GCTE_ENV_ROOT/bin/python" -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

"$GCTE_ENV_ROOT/bin/python" -m pip install \
  --retries 10 \
  --timeout 120 \
  -r "$GCTE_DOWNLOAD_CACHE/requirements-gcte-acr-eg-no-torch.txt"
```

环境硬门：

```bash
"$GCTE_ENV_ROOT/bin/python" - <<'PY'
import platform
import torch
import torchvision
import ultralytics

print("python", platform.python_version())
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

assert platform.python_version().startswith("3.10.")
assert torch.__version__ == "2.5.1+cu121"
assert torchvision.__version__ == "0.20.1+cu121"
assert ultralytics.__version__ == "8.4.90"
assert torch.cuda.is_available()
assert "4090" in torch.cuda.get_device_name(0)
PY
```

不要为这个裸机训练安装 NVIDIA Container Toolkit。

---

## 9. 下载与校验

### 9.1 下载清单：来源与落盘路径

| 内容 | 优先下载地址 | 官方/备用地址 | 新服务器落盘路径 | 身份硬门 |
|---|---|---|---|---|
| 项目 Git 仓库 | `https://github.com/kkc236/uav-detection-baselines.git` | 不使用第三方 Git 镜像，避免 branch HEAD 滞后 | `/home/ubuntu/gcte-acr-eg` | branch + commit + code-tree diff |
| 成熟 baseline | `https://gh-proxy.com/https://github.com/kkc236/uav-detection-baselines/releases/download/rtdetr-l-btdse-matched-baseline-live/matched-baseline-best-epoch-0100.pt` | `https://github.com/kkc236/uav-detection-baselines/releases/download/rtdetr-l-btdse-matched-baseline-live/matched-baseline-best-epoch-0100.pt` | `/home/ubuntu/matched-baseline-best-epoch-0100.pt` | bytes `66262262` + SHA256 `54CE...099B` |
| integrated epoch 9 | `https://gh-proxy.com/https://github.com/kkc236/uav-detection-baselines/releases/download/gcte-acr-eg-a22838e3-epoch-009/epoch8.pt` | `https://github.com/kkc236/uav-detection-baselines/releases/download/gcte-acr-eg-a22838e3-epoch-009/epoch8.pt` | `/home/ubuntu/gcte-checkpoints/epoch8.pt` | bytes `205325084` + SHA256 `802D...1FE6` + checkpoint contract |
| VisDrone train 原包 | `https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip` | `https://ultralytics.com/assets/VisDrone2019-DET-train.zip`；GitHub 直链见下文 | `/mnt/uav/datasets/VisDrone/VisDrone2019-DET-train.zip` | bytes `1549875511` + 转换后数据硬门 |
| VisDrone val 原包 | `https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip` | `https://ultralytics.com/assets/VisDrone2019-DET-val.zip`；GitHub 直链见下文 | `/mnt/uav/datasets/VisDrone/VisDrone2019-DET-val.zip` | bytes `81638851` + val signature |
| CUDA 12.1 PyTorch wheel | 阿里云 URL 已固定在 `requirements-gcte-acr-eg-cu121.txt` | `https://download.pytorch.org/whl/cu121` | pip cache：`/mnt/uav/pip-cache`；venv：`/mnt/uav/venv` | wheel SHA + import/version hard gate |
| 其他 Python 包 | `https://pypi.tuna.tsinghua.edu.cn/simple` | `https://pypi.org/simple` | pip cache：`/mnt/uav/pip-cache`；venv：`/mnt/uav/venv` | requirements pin + import/version hard gate |
| Ubuntu 软件包 | `https://mirrors.tuna.tsinghua.edu.cn/ubuntu/` | Ubuntu archive/security 官方源 | APT 管理路径 | `apt-get update` 成功 |

镜像只负责传输，不能成为身份权威。任何从 `gh-proxy.com` 下载的模型或 checkpoint 都必须通过 SHA256；VisDrone 必须通过压缩包字节数、转换后目录计数、精确 YAML SHA256 和 val 内容签名。镜像失败时自动回退官方源，不允许放宽校验。

GitHub 直链：

```text
https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip
https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip
```

### 9.2 通用可恢复下载函数

```bash
set -euo pipefail

download_sha256() {
  local destination="$1"
  local expected_sha256="$2"
  shift 2
  local temporary="${destination}.part"
  local url

  mkdir -p "$(dirname "$destination")"
  if [ -f "$destination" ]; then
    local existing_sha256
    existing_sha256="$(sha256sum "$destination" | awk '{print toupper($1)}')"
    if [ "$existing_sha256" = "$expected_sha256" ]; then
      echo "REUSE_VERIFIED=$destination"
      return 0
    fi
    echo "REMOVE_INVALID_EXISTING=$destination actual=$existing_sha256" >&2
    rm -f "$destination"
  fi

  for url in "$@"; do
    echo "DOWNLOAD_SOURCE=$url"
    if curl -fL \
      --retry 8 \
      --retry-all-errors \
      --connect-timeout 30 \
      --continue-at - \
      --output "$temporary" \
      "$url"; then
      local actual_sha256
      actual_sha256="$(sha256sum "$temporary" | awk '{print toupper($1)}')"
      if [ "$actual_sha256" = "$expected_sha256" ]; then
        mv "$temporary" "$destination"
        return 0
      fi
      echo "SHA256_MISMATCH expected=$expected_sha256 actual=$actual_sha256" >&2
    fi
    rm -f "$temporary"
  done
  return 1
}

download_size() {
  local destination="$1"
  local expected_bytes="$2"
  shift 2
  local temporary="${destination}.part"
  local url

  mkdir -p "$(dirname "$destination")"
  if [ -f "$destination" ]; then
    local existing_bytes
    existing_bytes="$(stat -c '%s' "$destination")"
    if [ "$existing_bytes" = "$expected_bytes" ]; then
      echo "REUSE_VERIFIED=$destination"
      return 0
    fi
    echo "REMOVE_INVALID_EXISTING=$destination actual=$existing_bytes" >&2
    rm -f "$destination"
  fi

  for url in "$@"; do
    echo "DOWNLOAD_SOURCE=$url"
    if curl -fL \
      --retry 8 \
      --retry-all-errors \
      --connect-timeout 30 \
      --continue-at - \
      --output "$temporary" \
      "$url"; then
      local actual_bytes
      actual_bytes="$(stat -c '%s' "$temporary")"
      if [ "$actual_bytes" = "$expected_bytes" ]; then
        mv "$temporary" "$destination"
        return 0
      fi
      echo "SIZE_MISMATCH expected=$expected_bytes actual=$actual_bytes" >&2
    fi
    rm -f "$temporary"
  done
  return 1
}
```

### 9.3 下载 baseline 与 integrated checkpoint

```bash
download_sha256 \
  "$GCTE_BASELINE" \
  "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B" \
  "https://gh-proxy.com/https://github.com/kkc236/uav-detection-baselines/releases/download/rtdetr-l-btdse-matched-baseline-live/matched-baseline-best-epoch-0100.pt" \
  "https://github.com/kkc236/uav-detection-baselines/releases/download/rtdetr-l-btdse-matched-baseline-live/matched-baseline-best-epoch-0100.pt"

download_sha256 \
  "$GCTE_CHECKPOINT" \
  "802D72326F4B8FEE55C0FF8818A5B96B7445CBEE34F5C1ED9002A6D3E6771FE6" \
  "https://gh-proxy.com/https://github.com/kkc236/uav-detection-baselines/releases/download/gcte-acr-eg-a22838e3-epoch-009/epoch8.pt" \
  "https://github.com/kkc236/uav-detection-baselines/releases/download/gcte-acr-eg-a22838e3-epoch-009/epoch8.pt"
```

### 9.4 旧服务器离线时重建 VisDrone

先把两个权威原包下载到 `prepare_visdrone.py` 会识别的固定路径：

```bash
mkdir -p "$GCTE_DATA_ROOT"

download_size \
  "$GCTE_DATA_ROOT/VisDrone2019-DET-train.zip" \
  "1549875511" \
  "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip" \
  "https://ultralytics.com/assets/VisDrone2019-DET-train.zip" \
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip"

download_size \
  "$GCTE_DATA_ROOT/VisDrone2019-DET-val.zip" \
  "81638851" \
  "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip" \
  "https://ultralytics.com/assets/VisDrone2019-DET-val.zip" \
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
```

只使用当前仓库的固定转换器：

```bash
cd "$GCTE_REPO_ROOT"

"$GCTE_ENV_ROOT/bin/python" scripts/prepare_visdrone.py \
  --dataset-dir "$GCTE_DATA_ROOT" \
  --splits train val
```

生成第 6 节给出的精确 YAML，然后运行全部数据硬门。必须得到：

```text
train images = 6471
train labels = 6471
val images = 548
val labels = 548
images + labels total = 14038
data yaml SHA256 = 7EB91FCE...B958324
val content signature = A9A0C00D...571228A
```

如果 val signature 不匹配：

1. 禁止训练；
2. 保留两个原始 zip、转换脚本 commit、文件计数和实际 signature；
3. 先检查下载字节数、转换脚本是否为当前 commit、是否意外使用了 Ultralytics 内置转换器；
4. 不得把另一套 YOLO 标签当成同一数据集继续跑。

`d2-train-10pct.txt` 只服务于早期 10% 诊断，本次正式第 9→100 轮续训不读取它，所以旧服务器离线时不把它设为阻塞项。

### 9.5 checkpoint 身份硬门

```bash
echo \
  "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B  $GCTE_BASELINE" \
  | sha256sum -c -

echo \
  "802D72326F4B8FEE55C0FF8818A5B96B7445CBEE34F5C1ED9002A6D3E6771FE6  $GCTE_CHECKPOINT" \
  | sha256sum -c -
```

checkpoint 身份硬门：

```bash
cd "$GCTE_REPO_ROOT"

"$GCTE_ENV_ROOT/bin/python" - <<'PY'
from pathlib import Path
import torch

path = Path("/home/ubuntu/gcte-checkpoints/epoch8.pt")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
ema = checkpoint.get("ema")

assert type(ema).__name__ == "ACREGDetectionModel", type(ema)
assert checkpoint.get("optimizer") is not None
assert checkpoint.get("scaler") is not None
assert checkpoint.get("epoch") == 8
assert checkpoint.get("updates") is not None

state = ema.float().state_dict()
acr_keys = [key for key in state if key.startswith("acr_eg.")]
assert len(acr_keys) == 48, len(acr_keys)

print("ema_type", type(ema).__name__)
print("checkpoint_epoch", checkpoint["epoch"])
print("acr_eg_keys", len(acr_keys))
print("updates", checkpoint["updates"])
PY
```

---

## 10. 安全 integrated resume 已实现

当前已经没有“先开发 resume 才能迁移”的代码阻塞。官方分支依次包含：

| 提交 | 已实现内容 |
|---|---|
| `a6f2cdd9` | 校验 `ACREGDetectionModel`、完整 state keys、optimizer、scaler、epoch、updates；恢复冻结参数和新输出路径 |
| `2f1e2279` | 逐轮 checkpoint 稳定性检查、身份检查、GitHub Release 上传、远端 bytes/SHA256 回读验证、evidence 推送 |
| `d1403fe7` | 真实 VisDrone 单 batch 多视图恢复 smoke：模型身份、start epoch、optimizer、固定 AMP、forward、loss、backward、梯度和 GPU 显存 |

关键实现文件：

```text
scripts/train_gcte_formal.py
scripts/smoke_acr_eg_resume.py
scripts/sync_acr_eg_checkpoints.py
src/rtdetr_acr_eg.py
src/gcte_formal_trainer.py
src/acr_eg_smoke.py
src/acr_eg_release.py
```

当前 `--resume /home/ubuntu/gcte-checkpoints/epoch8.pt` 会在创建 trainer 前 fail-closed 校验 checkpoint，并在 `_setup_train()` 后再次核验：

```text
ema type = ACREGDetectionModel
acr_eg.* keys = 48
optimizer state non-empty
checkpoint epoch = 8
runtime start_epoch = 9
scaler scale = 128
scaler growth_interval = 2147483647
冻结 batch/workers/device/cache/val/save_period/optimizer/seed 不漂移
```

剩余服务器硬门只有：

```text
环境版本通过
数据签名通过
下载 checkpoint 身份通过
真实 GPU smoke 通过
```

这些硬门通过后直接进入第 10–100 轮，不要重复实现 resume，也不要重新从 epoch 0 开始。

---

## 11. 新服务器 fresh verification

先运行聚焦回归：

```bash
cd "$GCTE_REPO_ROOT"

"$GCTE_ENV_ROOT/bin/python" -m pytest -q \
  tests/test_gcte_formal_cli.py \
  tests/test_gcte_formal_trainer.py \
  tests/test_rtdetr_acr_eg_integration.py \
  tests/test_acr_eg_resume_smoke.py \
  tests/test_acr_eg_release.py \
  tests/test_sync_acr_eg_checkpoints.py \
  tests/test_gcmv_data.py
```

然后运行全量：

```bash
"$GCTE_ENV_ROOT/bin/python" -m pytest -q
git diff --check
```

如果全量测试包含与当前环境无关的历史实验失败，必须逐项定位并报告；聚焦测试任一失败都不得启动正式训练。

确认当前运行提交没有漂移 resume 代码：

```bash
test "$(git rev-parse HEAD)" = "$(git rev-parse codex/gcte-rtdetr-g0)"

git diff --quiet \
  d1403fe7ed7f67f320817fa4ed9075689244e6e8 \
  HEAD \
  -- scripts src configs tests
```

两条必须退出 0。

---

## 12. 真实单 batch 恢复 smoke

设置独立输出，并运行仓库已经实现的 smoke：

```bash
cd "$GCTE_REPO_ROOT"
set -o pipefail

export GCTE_NEW_FULL_COMMIT="$(git rev-parse HEAD)"
export GCTE_NEW_SHORT_COMMIT="$(git rev-parse --short=8 HEAD)"
export GCTE_SMOKE_OUTPUT="/home/ubuntu/gcte-acr-eg-resume-smoke-${GCTE_NEW_SHORT_COMMIT}"
mkdir -p "$GCTE_SMOKE_OUTPUT"

"$GCTE_ENV_ROOT/bin/python" scripts/smoke_acr_eg_resume.py \
  --data "$GCTE_PROTOCOL_ROOT/source-VisDrone-full.yaml" \
  --config "$GCTE_REPO_ROOT/configs/rtdetr-l-acr-eg.yaml" \
  --baseline-checkpoint "$GCTE_BASELINE" \
  --resume "$GCTE_CHECKPOINT" \
  --project "$GCTE_SMOKE_OUTPUT" \
  --source-commit "$GCTE_NEW_FULL_COMMIT" \
  --evidence "$GCTE_SMOKE_OUTPUT/resume-smoke-evidence.json" \
  2>&1 | tee "$GCTE_SMOKE_OUTPUT/resume-smoke.log"

"$GCTE_ENV_ROOT/bin/python" -m json.tool \
  "$GCTE_SMOKE_OUTPUT/resume-smoke-evidence.json" >/dev/null

grep -F \
  "ACR_EG_RESUME_SMOKE_PASSED" \
  "$GCTE_SMOKE_OUTPUT/resume-smoke.log"
```

smoke 脚本自身会强制检查：

```text
model type = ACREGDetectionModel
state contains 48 acr_eg.* keys
start_epoch = 9
optimizer state non-empty
scaler scale = 128
input contains local_views and source_shape
forward executes global + four local views
loss and loss_items finite
backward succeeds
ACR-EG 参数存在有限非零梯度
GPU peak VRAM > 0
```

若 smoke 失败：

1. 不启动正式续训；
2. 保留 `resume-smoke.log` 和已有 evidence；
3. 按 systematic debugging 定位根因；
4. 只有确有代码缺陷才写失败测试并新提交修复；
5. 不得绕过 smoke、改 checkpoint 或改冻结参数。

---

## 13. 正式从第 9 轮续训

设置由真实 Git commit 生成的目录：

```bash
export GCTE_NEW_FULL_COMMIT="$(git rev-parse HEAD)"
export GCTE_NEW_SHORT_COMMIT="$(git rev-parse --short=8 HEAD)"
export GCTE_RESUME_SOURCE="/home/ubuntu/gcte-acr-eg-resume-${GCTE_NEW_SHORT_COMMIT}"
export GCTE_RESUME_OUTPUT="/home/ubuntu/gcte-acr-eg-resume-output-${GCTE_NEW_SHORT_COMMIT}"
```

部署该提交到全新源码目录，并验证：

```bash
git clone --local \
  "$GCTE_REPO_ROOT" \
  "$GCTE_RESUME_SOURCE"

cd "$GCTE_RESUME_SOURCE"
git remote set-url origin https://github.com/kkc236/uav-detection-baselines.git
git checkout "$GCTE_NEW_FULL_COMMIT"
test "$(git rev-parse HEAD)" = "$GCTE_NEW_FULL_COMMIT"
mkdir -p "$GCTE_RESUME_OUTPUT"
```

运行前：

```bash
test -f "$GCTE_BASELINE"
test -f "$GCTE_CHECKPOINT"
test -d "$GCTE_DATA_ROOT"
nvidia-smi
df -h /home /mnt/uav
```

正式命令必须由修复后的同一个入口生成：

```bash
cd "$GCTE_RESUME_SOURCE"

nohup env \
  GCTE_SOURCE_COMMIT="$GCTE_NEW_FULL_COMMIT" \
  /mnt/uav/venv/bin/python scripts/train_gcte_formal.py \
    --project "$GCTE_RESUME_OUTPUT" \
    --name acr-eg-integrated-rtdetr-resume-epoch009-to-100 \
    --baseline-checkpoint /home/ubuntu/matched-baseline-best-epoch-0100.pt \
    --resume /home/ubuntu/gcte-checkpoints/epoch8.pt \
  > "$GCTE_RESUME_OUTPUT/formal.log" 2>&1 &

echo $! > "$GCTE_RESUME_OUTPUT/runner.pid"
```

启动完成硬门：

```text
runner process alive
training child process alive
GPU utilization/memory nonzero
formal.log 明确打印 ACREGDetectionModel
formal.log 同时包含 Resuming training 和 epoch 10
results.csv 继续而不是从 epoch 1 重新开始
新 checkpoint 仍有 48 acr_eg.* keys
```

只有上述都满足，才能告诉用户“已经真正续训”。

---

## 14. 每轮自动备份

完成第 `N` 轮：

```text
file = epoch{N-1}.pt
tag prefix = gcte-acr-eg-${GCTE_NEW_SHORT_COMMIT}-epoch-
```

生成精确 tag：

```bash
export GCTE_HUMAN_EPOCH=10
export GCTE_RELEASE_TAG="$(
  printf 'gcte-acr-eg-%s-epoch-%03d' \
    "$GCTE_NEW_SHORT_COMMIT" \
    "$GCTE_HUMAN_EPOCH"
)"
echo "$GCTE_RELEASE_TAG"
```

每轮：

1. 等待 checkpoint 文件大小稳定；
2. 在服务器计算 SHA256；
3. 反序列化检查；
4. 验证 `ema=ACREGDetectionModel`；
5. 验证 48 个 `acr_eg.*` keys；
6. 验证 optimizer/scaler/epoch/updates；
7. 上传 GitHub Release；
8. 查询 GitHub asset；
9. 比对 bytes 和 digest；
10. 写轻量 evidence JSON；
11. 推送 evidence JSON；
12. 只有 GitHub 校验后才允许清理较旧独立 epoch 文件；
13. 永远保留 `last.pt`、`best.pt` 和最新完整 epoch。

仓库已实现上述流程。正式训练确认启动后，在同一份精确源码 checkout 中启动一个上传器：

```bash
cd "$GCTE_RESUME_SOURCE"

export GCTE_FORMAL_RUN_DIR="$GCTE_RESUME_OUTPUT/acr-eg-integrated-rtdetr-resume-epoch009-to-100"
export GCTE_TOKEN_FILE=/home/ubuntu/.config/gcte/github-token
export GCTE_RESULTS_REPO=/mnt/uav/gcte-training-results
export GCTE_UPLOAD_LOG="$GCTE_RESUME_OUTPUT/checkpoint-uploader.log"
export GCTE_UPLOAD_PID="$GCTE_RESUME_OUTPUT/checkpoint-uploader.pid"

install -d -m 700 "$(dirname "$GCTE_TOKEN_FILE")"
umask 077
gh auth token > "$GCTE_TOKEN_FILE"
chmod 600 "$GCTE_TOKEN_FILE"

nohup "$GCTE_ENV_ROOT/bin/python" scripts/sync_acr_eg_checkpoints.py \
  --run-dir "$GCTE_FORMAL_RUN_DIR" \
  --token-file "$GCTE_TOKEN_FILE" \
  --source-commit "$GCTE_NEW_FULL_COMMIT" \
  --results-repo "$GCTE_RESULTS_REPO" \
  --run-name "gcte-acr-eg-${GCTE_NEW_SHORT_COMMIT}" \
  --start-epoch 10 \
  --end-epoch 100 \
  --interval 60 \
  --stable-seconds 30 \
  > "$GCTE_UPLOAD_LOG" 2>&1 &

echo $! > "$GCTE_UPLOAD_PID"
```

上传器的固定落盘路径：

| 内容 | 路径 |
|---|---|
| 上传器 PID | `$GCTE_RESUME_OUTPUT/checkpoint-uploader.pid` |
| 上传日志 | `$GCTE_RESUME_OUTPUT/checkpoint-uploader.log` |
| 最新上传状态 | `$GCTE_FORMAL_RUN_DIR/checkpoint-release-status.json` |
| 每轮本地 evidence | `$GCTE_FORMAL_RUN_DIR/checkpoint-release-evidence/epoch-NNN.json` |
| evidence Git checkout | `/mnt/uav/gcte-training-results` |
| evidence Git 分支 | `training-results` |
| GitHub Release tag | `gcte-acr-eg-${GCTE_NEW_SHORT_COMMIT}-epoch-NNN` |
| GitHub Release asset | `epoch{N-1}.pt` |

上传器会等待文件稳定，检查完整 checkpoint contract，发布或复用 Release，回读远端资产并核对 bytes/SHA256，最后提交轻量 evidence。运行中不得重复启动第二个上传器。

GitHub token：

- 优先使用 Codex 本机已有 `gh auth`；
- `gh auth token` 只写入上述 mode `600` 临时文件，不打印到聊天或日志；
- 若必须在服务器上传，token 文件权限必须 `600`；
- 不得输出 token；
- 上传器完成或失败后删除临时 token；
- 自定义仓库 GitHub Release 不使用不受信任的第三方代理。

磁盘少于 20 GB：

- 先备份；
- 再清理已核验废弃旧实验；
- 不删除当前源码、baseline、数据、protocol、当前 output 或未备份 checkpoint。

---

## 15. 监控

每次先只读：

```bash
cat "$GCTE_RESUME_OUTPUT/runner.pid"

ps -fp "$(cat "$GCTE_RESUME_OUTPUT/runner.pid")"

nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv

tail -n 100 "$GCTE_RESUME_OUTPUT/formal.log"

tail -n 3 \
  "$GCTE_RESUME_OUTPUT/acr-eg-integrated-rtdetr-resume-epoch009-to-100/results.csv"

df -h /home /mnt/uav
```

运行中不得重复启动。

SSH 断开时：

- 先判断端口、banner、认证还是远程进程问题；
- 连接失败本身不能证明训练失败；
- 已备份 checkpoint 是恢复权威；
- 不进行无依据重启。

---

## 16. 100 epoch 后的 live evaluator

当前 `scripts/evaluate_acr_eg_integrated.py` 主要包装旧 cache evaluator，不能作为最终 live checkpoint 评测。

必须以 TDD 实现新的 evaluator，要求：

```text
加载完成第 100 轮 integrated checkpoint
强制 model type = ACREGDetectionModel
强制 48 acr_eg.* keys
真实生成 global + four local views
真实运行 Query capture、GCQF 和 logit injection
禁止 silent stock fallback
548 val images
imgsz 640
max_det 300
NMS False
输出结构化 JSON
记录所有 SHA256 和 source commit
```

必须报告：

```text
mAP50-95
AP50
AP75
Precision
Recall
AP-tiny-SBR
tiny recall
AP-medium-SBR
AP-large-SBR
parameters
FLOPs
peak VRAM
end-to-end latency
FPS
```

正式比较：

```text
Global RT-DETR-L baseline
vs
Full GCQF / ACR-EG
```

还必须补 matched continuation control：

```text
Control:
same mature baseline
no GCQF
same continuation epochs
same data/batch/lr/seed/augmentation/validation

Method:
same mature baseline
with GCQF / ACR-EG
same continuation epochs
```

否则不能把“额外训练 100 轮”和“模块增益”完全分离。

---

## 17. 完成判据

只有全部满足才结束任务：

```text
[ ] 新服务器环境版本全部匹配
[ ] baseline SHA 匹配
[ ] epoch8 checkpoint SHA 匹配
[ ] dataset counts 和 val signature 匹配
[ ] `d1403fe7` 安全 resume / publisher / smoke 代码身份通过
[ ] focused tests 通过
[ ] broad regression 通过或每个无关失败有证据解释
[ ] 执行 branch HEAD 已推送，且 `d1403fe7 → HEAD` 的 `scripts/src/configs/tests` 无漂移
[ ] 真实 multi-view resume smoke 通过
[ ] 从第 10 轮而非第 1 轮开始
[ ] model 是 ACREGDetectionModel
[ ] 48 个 acr_eg.* keys
[ ] optimizer、scaler、epoch、updates 连续
[ ] GPU 真正运行
[ ] 第 10–100 轮每轮 Release 备份并校验
[ ] 100 epoch 完成
[ ] 最终 checkpoint 发布并校验
[ ] live 548 图 evaluator 完成
[ ] Global-relative 指标和延迟已输出
[ ] 结果、日志、protocol 和 SHA evidence 已推送
```

---

## 18. 现在允许与禁止的表述

现在允许：

- 已设计三阶段 GCQF 查询级网络模块；
- 已完成真正 YAML/forward/loss/optimizer/checkpoint 集成；
- 已实现 fail-closed integrated resume、真实 batch smoke 和逐轮 Release 发布器；
- cache 诊断相对 Global 有正增益；
- 真正集成模型已完成并备份第 9 轮；
- checkpoint 是 `ACREGDetectionModel`。

现在禁止：

- 第 9 轮 live mAP 已提升；
- 100 epoch 已成功；
- 所有尺度都提升；
- `best.pt` 是验证集最优；
- local boxes 已进入最终输出；
- latency 已确定；
- 当前方法已经达到 CCF-C 录用水平；
- 已完成多 seed、第二数据集或 SOTA 对比。

---

## 19. 镜像和来源

| 类型 | 使用地址 | 说明 |
|---|---|---|
| PyPI 镜像 | `https://pypi.tuna.tsinghua.edu.cn/simple` | 默认 |
| PyPI 镜像帮助 | `https://mirrors.tuna.tsinghua.edu.cn/help/pypi/` | 清华官方说明 |
| Ubuntu 镜像 | `https://mirrors.tuna.tsinghua.edu.cn/ubuntu/` | 默认 |
| Ubuntu 镜像帮助 | `https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/` | 清华官方说明 |
| PyTorch CUDA 12.1 镜像 | `https://mirrors.aliyun.com/pytorch-wheels/cu121/` | requirements 中固定 wheel URL 和 SHA |
| PyTorch 官方回退 | `https://download.pytorch.org/whl/cu121` | 镜像失败时使用 |
| VisDrone 文档 | `https://docs.ultralytics.com/datasets/detect/visdrone/` | 数据规模与自动转换的权威说明 |
| VisDrone 权威资产 | `https://ultralytics.com/assets/VisDrone2019-DET-{train,val}.zip` | 最终重定向到 Ultralytics GitHub Release |
| GitHub 读取加速 | `https://gh-proxy.com/<完整 GitHub URL>` | 仅下载；第三方传输层 |
| 本项目 GitHub | `https://github.com/kkc236/uav-detection-baselines` | 代码、baseline、checkpoint 权威源 |

安全边界：

- 第三方 GitHub 代理只用于具备严格身份硬门的公开大文件读取，不用于 Git clone；
- Git push、Release 上传和 token 绝不经过第三方代理；
- baseline/checkpoint 以 SHA256 为准；
- VisDrone 以官方资产字节数、仓库转换器、目录计数、YAML SHA256 和 val 内容签名共同为准；
- 任何镜像不可用时只回退官方源，不临时寻找未知网盘版本。

---

## 20. 执行 Codex 的第一条实际回复

读完本文件后，不要复述全文。第一条回复只应包含：

```text
1. 已识别的新服务器；
2. 将采用的最后可靠恢复点：完成第 9 轮 epoch8.pt；
3. 安全 integrated resume、真实 smoke 和逐轮发布器已经实现；
4. 立即开始的动作：只读服务器预检、镜像优先部署、fresh tests 和 GPU smoke；
5. 明确不会运行 stock RT-DETR、不会重复启动、不会关机。
```

然后立即调用工具执行。
