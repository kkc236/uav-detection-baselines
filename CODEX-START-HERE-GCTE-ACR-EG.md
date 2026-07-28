# Codex 单文件执行包：GCTE-RTDETR / ACR-EG 新服务器续训至 100 Epoch

> 本文件既是项目交接，也是给下一位 Codex 的直接执行指令。用户把本文件和新服务器登录信息放在同一条消息中后，Codex 应读取全文并自主执行，不要只总结、评价或重新制定一个空泛方案。

---

## 0. 给执行 Codex 的最高优先级指令

你正在接手一个真实的 RT-DETR-L 无人机小目标检测实验。

你的任务终点是：

```text
1. 在用户提供的全新 RTX 4090 服务器上完成环境和数据部署；
2. 以严格 TDD 方式补齐真正 YAML 集成 ACR-EG checkpoint 的安全续训；
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

如果用户消息中没有新服务器 IP、用户名、端口或认证信息，这是唯一允许立即询问的阻塞项。用户提供后，不再反复询问已经能够从代码、GitHub或服务器发现的信息。

---

## 1. 权威项目身份

GitHub：

```text
repository = kkc236/uav-detection-baselines
branch = codex/gcte-rtdetr-g0
single-file handoff commit = 9153a44e96411ba4f31f36d8a15f23d77232c75a
genuine integrated source base = a22838e3e7cd1cd858d6aad9f42e5b68fab50471
```

从 `a22838e3` 到 `9153a44e`，`scripts/`、`src/`、`configs/` 和 `tests/` 的代码树没有变化；后续提交只增加或修订交接、证据和依赖锁。

在新服务器上克隆：

```bash
set -euo pipefail

export GCTE_REPO_ROOT=/home/ubuntu/gcte-acr-eg

git clone --branch codex/gcte-rtdetr-g0 \
  https://github.com/kkc236/uav-detection-baselines.git \
  "$GCTE_REPO_ROOT"

cd "$GCTE_REPO_ROOT"
git checkout 9153a44e96411ba4f31f36d8a15f23d77232c75a
git status --short
git rev-parse HEAD

git diff --quiet \
  a22838e3e7cd1cd858d6aad9f42e5b68fab50471 \
  9153a44e96411ba4f31f36d8a15f23d77232c75a \
  -- scripts src configs tests
```

最后一条必须退出 0。否则停止并报告 source identity 漂移。

---

## 2. 最新可靠训练状态

旧服务器：

```text
host = 36.103.199.151
user = ubuntu
port = 22
```

不要把旧密码写入任何文件。

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

旧服务器仍可访问时，使用 `rsync --partial --append-verify` 从 `36.103.199.151` 迁移。密码只交互输入。旧服务器不可访问时，要求用户提供同一权威数据拷贝，不得自动下载另一版本替代。

文件数硬门：

```bash
test "$(find /mnt/uav/datasets/VisDrone/images/train -type f | wc -l)" -eq 6471
test "$(find /mnt/uav/datasets/VisDrone/labels/train -type f | wc -l)" -eq 6471
test "$(find /mnt/uav/datasets/VisDrone/images/val -type f | wc -l)" -eq 548
test "$(find /mnt/uav/datasets/VisDrone/labels/val -type f | wc -l)" -eq 548
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

sudo mkdir -p /mnt/uav/datasets /mnt/uav/protocols
sudo chown -R ubuntu:ubuntu /mnt/uav
mkdir -p /home/ubuntu/gcte-checkpoints
mkdir -p "$GCTE_OUTPUT_ROOT"
```

Python：

```bash
python3.10 -m venv "$GCTE_ENV_ROOT"
"$GCTE_ENV_ROOT/bin/python" -m pip install --upgrade pip
"$GCTE_ENV_ROOT/bin/python" -m pip config set \
  global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

cd "$GCTE_REPO_ROOT"
"$GCTE_ENV_ROOT/bin/python" -m pip install \
  --retries 10 \
  --timeout 120 \
  -r requirements-gcte-acr-eg-cu121.txt
```

`requirements-gcte-acr-eg-cu121.txt` 已固定旧服务器的 54 项运行环境。PyTorch 和 TorchVision 使用阿里云 CUDA 12.1 wheel 与 wheel SHA；其余 PyPI 包使用清华镜像。

官方 PyTorch 回退：

```bash
"$GCTE_ENV_ROOT/bin/python" -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
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

```bash
curl -fL \
  --retry 10 \
  --retry-all-errors \
  --connect-timeout 30 \
  -o "$GCTE_BASELINE" \
  https://github.com/kkc236/uav-detection-baselines/releases/download/rtdetr-l-btdse-matched-baseline-live/matched-baseline-best-epoch-0100.pt

echo \
  "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B  $GCTE_BASELINE" \
  | sha256sum -c -

curl -fL \
  --retry 10 \
  --retry-all-errors \
  --connect-timeout 30 \
  -o "$GCTE_CHECKPOINT" \
  https://github.com/kkc236/uav-detection-baselines/releases/download/gcte-acr-eg-a22838e3-epoch-009/epoch8.pt

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

## 10. 当前唯一代码阻塞：真正 integrated resume

当前 `scripts/train_gcte_formal.py` 在 `main()` 中主动拒绝：

```python
if args.resume:
    raise ValueError("GCTE_ACR_EG_RESUME_REQUIRES_INTEGRATED_CHECKPOINT")
```

当前 `src/rtdetr_acr_eg.py` 中：

```python
def get_model(self, cfg=None, weights=None, verbose: bool = True):
    if weights is not None:
        raise ValueError("ACR-EG formal run must load only the sealed mature baseline")
```

因此不能直接运行现有 `--resume`。

不要：

- 用 stock RT-DETR trainer；
- 只加载 EMA state_dict 后从 epoch 0 重新计数；
- 丢弃 optimizer 或 scheduler；
- 把 `epoch8.pt` 当 mature baseline；
- 在旧提交上删除异常后未经测试直接跑；
- 覆盖旧 output。

---

## 11. TDD 实现安全 resume

### Task 1：为 integrated checkpoint contract 写失败测试

修改：

```text
tests/test_gcte_formal_cli.py
tests/test_rtdetr_acr_eg_integration.py
```

新增以下精确测试名：

```text
test_integrated_resume_requires_custom_ema_optimizer_scaler_epoch_and_updates
test_integrated_resume_rejects_stock_rtdetr_checkpoint
test_integrated_resume_requires_all_acr_eg_state_keys
test_resume_reapplies_new_project_name_and_frozen_runtime_overrides
test_resume_model_is_acr_eg_and_loads_integrated_weights
```

断言分别为：

1. 缺少 `ema`、`optimizer`、`scaler`、`epoch` 或 `updates` 中任意字段都 fail closed；
2. `ema=torch.nn.Linear(1, 1)` 被 `ACR_EG_RESUME_MODEL_IDENTITY_MISMATCH` 拒绝；
3. 从真实 `ACREGDetectionModel` 删除任意一个 `acr_eg.*` key 后被拒绝；
4. resume 后 `project/name` 指向新输出，且 `batch=8`、`workers=8`、`device=0`、`cache=False`、`save_period=1`、`val=False`；
5. `ACREGFormalTrainer.get_model(weights=source)` 返回 `ACREGDetectionModel`，并且返回模型与 source 的完整 state keys 和每个 tensor 值一致。

先运行：

```bash
"$GCTE_ENV_ROOT/bin/python" -m pytest -q \
  tests/test_gcte_formal_cli.py \
  tests/test_rtdetr_acr_eg_integration.py
```

预期：新增测试失败，原因必须对应当前缺失的 resume 功能，而不是 import、路径或测试本身错误。

### Task 2：增加 checkpoint validator

优先在：

```text
src/rtdetr_acr_eg.py
```

增加单一职责函数，接口建议：

```python
def validate_acr_eg_resume_checkpoint(
    checkpoint: dict,
    *,
    expected_model_state_keys: set[str] | None = None,
) -> nn.Module:
    if not isinstance(checkpoint, dict):
        raise ValueError("ACR_EG_RESUME_CHECKPOINT_NOT_MAPPING")

    ema = checkpoint.get("ema")
    if not isinstance(ema, ACREGDetectionModel):
        raise ValueError("ACR_EG_RESUME_MODEL_IDENTITY_MISMATCH")

    for key in ("optimizer", "scaler", "epoch", "updates"):
        if key not in checkpoint or checkpoint[key] is None:
            raise ValueError(f"ACR_EG_RESUME_MISSING_{key.upper()}")

    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < 99:
        raise ValueError("ACR_EG_RESUME_EPOCH_INVALID")

    state_keys = set(ema.state_dict())
    acr_keys = {key for key in state_keys if key.startswith("acr_eg.")}
    if len(acr_keys) != 48:
        raise ValueError("ACR_EG_RESUME_STATE_IDENTITY_MISMATCH")

    if expected_model_state_keys is not None and state_keys != expected_model_state_keys:
        raise ValueError("ACR_EG_RESUME_STATE_KEYS_MISMATCH")

    return ema
```

它必须：

```python
if not isinstance(checkpoint, dict):
    raise ValueError("ACR_EG_RESUME_CHECKPOINT_NOT_MAPPING")

ema = checkpoint.get("ema")
if not isinstance(ema, ACREGDetectionModel):
    raise ValueError("ACR_EG_RESUME_MODEL_IDENTITY_MISMATCH")

for key in ("optimizer", "scaler", "epoch", "updates"):
    if checkpoint.get(key) is None:
        raise ValueError(f"ACR_EG_RESUME_MISSING_{key.upper()}")

epoch = checkpoint["epoch"]
if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < 99:
    raise ValueError("ACR_EG_RESUME_EPOCH_INVALID")

state_keys = set(ema.state_dict())
acr_keys = {key for key in state_keys if key.startswith("acr_eg.")}
if len(acr_keys) != 48:
    raise ValueError("ACR_EG_RESUME_STATE_IDENTITY_MISMATCH")

if expected_model_state_keys is not None and state_keys != expected_model_state_keys:
    raise ValueError("ACR_EG_RESUME_STATE_KEYS_MISMATCH")

return ema
```

不要仅硬编码 48 后忽略完整模型 key 集；构建目标模型后必须比较完整 keys。

### Task 3：让 custom trainer 安全加载 resume weights

修改：

```text
src/rtdetr_acr_eg.py
```

目标行为：

```python
def get_model(self, cfg=None, weights=None, verbose: bool = True):
    model = ACREGDetectionModel(
        self.model_yaml,
        nc=self.data["nc"],
        ch=self.data["channels"],
        verbose=verbose and RANK == -1,
    )

    if weights is None:
        load_mature_baseline(model, self.baseline_checkpoint)
        return model

    if not isinstance(weights, ACREGDetectionModel):
        raise ValueError("ACR_EG_RESUME_MODEL_IDENTITY_MISMATCH")

    source = weights.float().state_dict()
    destination = model.state_dict()
    if set(source) != set(destination):
        raise ValueError("ACR_EG_RESUME_STATE_KEYS_MISMATCH")

    model.load_state_dict(source, strict=True)
    return model
```

必须使用 `self.model_yaml`，不能信任未知 checkpoint 中的 stock YAML。

### Task 4：允许主入口 resume，但先验证 payload

修改：

```text
scripts/train_gcte_formal.py
```

把主动拒绝替换为：

```python
if args.resume:
    resume_path = Path(args.resume).resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    validate_acr_eg_resume_checkpoint(payload)
```

补充 import：

```python
import torch
from src.rtdetr_acr_eg import (
    ACREGFormalTrainer,
    validate_acr_eg_resume_checkpoint,
)
```

避免在文件顶层过早 import Ultralytics/CUDA 组件时，可把 import 保持在 `main()` 中，但 validator 必须在创建 trainer 前调用。

### Task 5：恢复时重新应用运行位置和冻结参数

Ultralytics `BaseTrainer.check_resume()` 会从 checkpoint 的 `train_args` 重建参数，只允许少量覆盖；默认不会安全迁移新的 `project/name`。

在 `ACREGFormalTrainer` 覆盖 `check_resume()`：

```python
def check_resume(self, overrides):
    requested = {
        key: overrides.get(key)
        for key in (
            "project",
            "name",
            "imgsz",
            "batch",
            "workers",
            "device",
            "close_mosaic",
            "save_period",
            "cache",
            "val",
            "plots",
        )
    }
    super().check_resume(overrides)
    if not self.resume:
        return
    for key, value in requested.items():
        if value is not None:
            setattr(self.args, key, value)
```

然后加显式冻结断言：

```python
if self.args.imgsz != 640:
    raise ValueError("GCTE_FORMAL_INPUT_PROTOCOL_DRIFT")
if self.args.batch != 8 or self.args.workers != 8:
    raise ValueError("GCTE_FORMAL_INPUT_PROTOCOL_DRIFT")
if self.args.device != "0":
    raise ValueError("GCTE_FORMAL_DEVICE_OR_SEED_DRIFT")
if self.args.cache is not False or self.args.val is not False:
    raise ValueError("GCTE_FORMAL_RUNTIME_PROTOCOL_DRIFT")
if self.args.save_period != 1:
    raise ValueError("GCTE_FORMAL_CHECKPOINT_PROTOCOL_DRIFT")
```

必须确认 `project/name` 在 `get_save_dir()` 之前生效，并且新目录不存在。

### Task 6：固定 AMP scaler 的恢复连续性

当前 `GCTEFormalTrainer._setup_train()` 会在 `super()._setup_train()` 后重新创建固定 scale 128 scaler。因为原运行固定 scale 且禁止增长，安全恢复至少必须证明：

```text
checkpoint scaler scale = 128
resume 后 scaler scale = 128
growth interval 仍是固定协议值
optimizer state 已恢复
start_epoch = 9
```

新增测试，禁止只检查布尔 AMP：

```python
assert float(checkpoint_scaler_scale) == 128.0
assert float(trainer.scaler.get_scale()) == 128.0
assert trainer.start_epoch == 9
```

如果 Ultralytics 在 `super()._setup_train()` 内恢复 scaler 后又被当前 override 覆盖，应在代码注释和 protocol JSON 中记录：

```text
fixed-scale scaler has no dynamic growth state;
reconstruction at the same exact scale is the frozen protocol.
```

不能丢 optimizer state 或 epoch。

### Task 7：测试转绿和广泛回归

运行：

```bash
"$GCTE_ENV_ROOT/bin/python" -m pytest -q \
  tests/test_gcte_formal_cli.py \
  tests/test_rtdetr_acr_eg_integration.py \
  tests/test_gcte_formal_trainer.py \
  tests/test_gcmv_data.py

"$GCTE_ENV_ROOT/bin/python" -m pytest -q
```

如全量测试包含与当前环境无关的历史实验失败，必须逐项定位并报告，不能只宣称“主要测试通过”。

运行：

```bash
git diff --check
```

提交：

```bash
git add \
  scripts/train_gcte_formal.py \
  src/rtdetr_acr_eg.py \
  src/gcte_formal_trainer.py \
  tests/test_gcte_formal_cli.py \
  tests/test_rtdetr_acr_eg_integration.py

git commit -m "feat: safely resume integrated ACR-EG training"
git push origin codex/gcte-rtdetr-g0
```

记录新的完整 40 位 commit。后续服务器源码目录、输出目录、Release tag 必须使用该新 commit 短哈希，不再使用 `a22838e3` 作为实际运行 commit。

---

## 12. 真实单 batch 恢复 smoke

在正式启动前，使用第 9 轮 checkpoint 和真实 VisDrone batch 做一次恢复 smoke。

硬门：

```text
model type = ACREGDetectionModel
state contains 48 acr_eg.* keys
start_epoch = 9
optimizer state non-empty
scaler scale = 128
input contains local_views and source_shape
forward executes global + four local views
loss finite
backward succeeds
at least one acr_eg parameter on the actual logit path has nonzero finite grad
GPU memory is used
```

smoke 输出必须进入独立目录。完成修复提交后先生成真实标识：

```bash
export GCTE_NEW_FULL_COMMIT="$(git rev-parse HEAD)"
export GCTE_NEW_SHORT_COMMIT="$(git rev-parse --short=8 HEAD)"
export GCTE_SMOKE_OUTPUT="/home/ubuntu/gcte-acr-eg-resume-smoke-${GCTE_NEW_SHORT_COMMIT}"
mkdir -p "$GCTE_SMOKE_OUTPUT"
```

不要覆盖正式输出。

若 smoke 失败：

1. 不启动 100 epoch；
2. 保留日志；
3. 写失败测试；
4. 新提交修复；
5. 重新 smoke。

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
git clone --branch codex/gcte-rtdetr-g0 \
  https://github.com/kkc236/uav-detection-baselines.git \
  "$GCTE_RESUME_SOURCE"

cd "$GCTE_RESUME_SOURCE"
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
```

启动一次后记录 PID：

```bash
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

GitHub token：

- 优先使用 Codex 本机已有 `gh auth`；
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
[ ] resume TDD 红绿循环完成
[ ] focused tests 通过
[ ] broad regression 通过或每个无关失败有证据解释
[ ] 新 resume commit 已推送
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

```text
PyPI TUNA:
https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

Ubuntu TUNA:
https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/

Aliyun PyTorch CUDA 12.1 wheels:
https://mirrors.aliyun.com/pytorch-wheels/cu121/

PyTorch official previous versions:
https://docs.pytorch.org/get-started/previous-versions/

GitHub Release mirror limitations:
https://mirrors.tuna.tsinghua.edu.cn/help/github-release/
```

自定义 GitHub repo 和 checkpoint 使用官方 GitHub 地址。

---

## 20. 执行 Codex 的第一条实际回复

读完本文件后，不要复述全文。第一条回复只应包含：

```text
1. 已识别的新服务器；
2. 将采用的最后可靠恢复点：完成第 9 轮 epoch8.pt；
3. 当前唯一代码阻塞：integrated resume；
4. 立即开始的动作：只读服务器预检、环境部署和 TDD；
5. 明确不会运行 stock RT-DETR、不会重复启动、不会关机。
```

然后立即调用工具执行。
