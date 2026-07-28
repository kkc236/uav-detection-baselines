# GCTE-RTDETR / ACR-EG 全新服务器迁移、恢复与后续评测手册

更新时间：2026-07-28 18:20（Asia/Shanghai）

仓库：`kkc236/uav-detection-baselines`

分支：`codex/gcte-rtdetr-g0`

正式集成源码提交：

```text
a22838e3e7cd1cd858d6aad9f42e5b68fab50471
```

本文档解决四个问题：

1. 现在真正完成了什么；
2. 当前服务器训练到哪里、哪些检查点已经安全备份；
3. GCTE / GCQF / ACR-EG 到底是什么网络结构；
4. 如何在一台全新的 Ubuntu 服务器上从零配置环境，并安全地继续实验。

最重要的边界：

> `a22838e3` 已经完成真正的 YAML、主模型 forward、loss、optimizer 和 checkpoint 集成，但该提交的 `--resume` 主入口仍主动拒绝恢复集成 checkpoint。因此，最新权重已经安全备份，却不能在没有新增测试和修复提交的情况下直接一键续训。不要用 stock RT-DETR 的恢复入口冒充 ACR-EG 续训。

---

## 1. 当前状态快照

### 1.1 旧服务器

```text
服务器：36.103.199.151
用户：ubuntu
端口：22
```

密码不写入 Git、Markdown、shell history 或脚本。连接时只交互输入。

正式源码目录：

```text
/home/ubuntu/gcte-acr-eg-formal-a22838e3
```

正式输出目录：

```text
/home/ubuntu/gcte-acr-eg-formal-output-a22838e3
```

训练 run：

```text
/home/ubuntu/gcte-acr-eg-formal-output-a22838e3/acr-eg-integrated-rtdetr-100
```

2026-07-28 18:16 的只读核验：

| 项目 | 状态 |
|---|---|
| runner PID | `31100` |
| 进程 | 仍在运行 |
| 当前训练 | 第 `10/100` 轮 |
| 当前 batch | 核验时约 `198/809`，之后继续推进 |
| 最新完整轮次 | 第 `9` 轮 |
| 最新独立文件 | `epoch8.pt` |
| GPU | RTX 4090，核验时仍有显存和利用率 |
| `/home` 剩余空间 | 约 `12 GB` |
| 完成标记 | 尚无 |
| 失败标记 | 尚无 |

Ultralytics 的文件名使用零基 epoch：

```text
完成第 9 轮
→ checkpoint 字段 epoch = 8
→ 文件名 epoch8.pt
```

训练没有被停止、重启或重复启动。

2026-07-28 18:33 再次只读检查时，服务器连续两次在 SSH protocol banner 阶段断开，因此无法取得比 18:16 更新的权威状态。不能据此断言训练失败，也不能断言服务器仍在线。本文以已经完成服务器、本地和 GitHub 三方 SHA256 闭环的第 9 轮为最后可靠恢复点。

### 1.2 第 9 轮训练日志

`results.csv` 的最新完整记录：

```csv
epoch,time,train/giou_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/giou_loss,val/cls_loss,val/l1_loss,lr/pg0,lr/pg1,lr/pg2,lr/pg3,lr/pg4,lr/pg5,lr/pg6,lr/pg7
9,7320.23,1.54419,0.28404,0.20044,0,0,0,0,0,0,0,0.027624,0.009208,0.027624,0.009208,0.027624,0.009208,0.027624,0.009208
```

这些 `mAP=0` 不是检测结果。正式训练设置了：

```text
val=False
```

所以训练期不运行验证，`results.csv` 的 precision、recall 和 mAP 列只是占位零值。只有之后的独立 548 图冻结评测才是正式结果。

### 1.3 磁盘风险

旧服务器 `/home` 总量只有约 39 GB，核验时剩余约 12 GB。日志已经出现过训练 cache 因空间不足无法保存的警告，但 `cache=False`，训练仍然继续。

新服务器建议：

```text
系统和环境：至少 30 GB
数据集：约 2 GB
源码与日志：至少 5 GB
100 个约 205 MB epoch checkpoint：约 21 GB
last.pt + best.pt + 临时文件：至少 1 GB
安全余量：至少 30 GB
推荐可用磁盘：100 GB 或更多
```

---

## 2. 已完成的 GitHub 备份

### 2.1 成熟 baseline

Release：

```text
https://github.com/kkc236/uav-detection-baselines/releases/tag/rtdetr-l-btdse-matched-baseline-live
```

资产：

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

### 2.2 真正集成 ACR-EG 的 epoch 检查点

| 完成轮次 | Ultralytics 文件 | GitHub tag | 大小 | SHA256 |
|---:|---|---|---:|---|
| 3 | `epoch2.pt` | `gcte-acr-eg-a22838e3-epoch-003` | 205324828 | `7AB2CAC3...`，完整值见 Release |
| 4 | `epoch3.pt` | `gcte-acr-eg-a22838e3-epoch-004` | 205324956 | `344951097A89A5F2B49FFFB0DE09E5D4DF2857DC2A5F1874A44CA733127B58AA` |
| 7 | `epoch6.pt` | `gcte-acr-eg-a22838e3-epoch-007` | 205324892 | `ACEED252061AF2E6D72FA65461D138E94DDED6B69474E1B0706BEBAD4C4929CB` |
| 9 | `epoch8.pt` | `gcte-acr-eg-a22838e3-epoch-009` | 205325084 | `802D72326F4B8FEE55C0FF8818A5B96B7445CBEE34F5C1ED9002A6D3E6771FE6` |

直接 Release：

```text
https://github.com/kkc236/uav-detection-baselines/releases/tag/gcte-acr-eg-a22838e3-epoch-003
https://github.com/kkc236/uav-detection-baselines/releases/tag/gcte-acr-eg-a22838e3-epoch-004
https://github.com/kkc236/uav-detection-baselines/releases/tag/gcte-acr-eg-a22838e3-epoch-007
https://github.com/kkc236/uav-detection-baselines/releases/tag/gcte-acr-eg-a22838e3-epoch-009
```

最新第 9 轮资产直链：

```text
https://github.com/kkc236/uav-detection-baselines/releases/download/gcte-acr-eg-a22838e3-epoch-009/epoch8.pt
```

GitHub 已返回：

```text
draft=false
prerelease=false
state=uploaded
size=205325084
digest=sha256:802d72326f4b8fee55c0ff8818a5b96b7445cbee34f5c1ed9002a6d3e6771fe6
```

该 digest 与服务器、本地下载文件完全一致。

### 2.3 检查点内部身份

已验证的正式集成 checkpoint 具有以下结构：

```text
checkpoint["model"] = None
checkpoint["ema"] = ACREGDetectionModel
checkpoint["optimizer"] 存在
checkpoint["scaler"] 存在
checkpoint["epoch"] 存在
checkpoint["updates"] 存在
EMA state_dict 中存在 48 个 acr_eg.* key
```

`model=None` 是 Ultralytics 8.4.90 当前保存逻辑的正常行为；恢复和最终权重以 `ema` 为权威。不能因为顶层 `model` 为 `None` 就误判检查点损坏。

---

## 3. 当前方法到底是什么

### 3.1 正式名称和分类

整体模型：

```text
GCTE-RTDETR
Geometry-Canonical Tiny-Evidence RT-DETR
```

创新点 1 的完整网络模块：

```text
GCQF
Geometry-Canonical Query Fusion
```

第三阶段：

```text
ACR-EG
Anchor-Conditioned Residual Evidence Gate
```

严格分类：

```text
A：查询级网络结构融合模块
```

理由：

- 输入包含 RT-DETR Decoder 最终 Query；
- `ACREGDetectionModel` 继承 `RTDETRDetectionModel`；
- GCQF 和 ACR-EG 都是 `nn.Module`；
- 有可训练参数；
- 在主模型 `predict()` / `loss()` 中统一调用；
- 在检测损失计算之前改变最终 Query 分类 logits；
- 梯度从 RT-DETR criterion 回传；
- 模块参数进入 `state_dict()`、MuSGD optimizer、EMA 和正式 checkpoint。

它不是只读取 boxes、scores 后重排的 B 类后处理，也不是无参数规则 C。

### 3.2 总体数据流

```text
原始图像
├── 全局 640 视图
│   └── 共享 RT-DETR-L
│       └── 最终 Decoder Query：Qg
└── 四个局部高分辨率视图
    └── 同一个共享 RT-DETR-L
        └── 最终 Decoder Query：Ql

Ql + 视图几何
→ Stage 1：GeometryQueryProjector
→ Stage 2：GlobalLocalQueryInteraction
→ Stage 3：AnchorConditionedResidualEvidenceGate
→ global_retain_logits
→ 注入最后一层非 denoising Query 分类 logits
→ 原始 RT-DETR 匹配与检测损失
```

当前计算量近似：

```text
1 次 global RT-DETR forward
+ 4 次 local RT-DETR forward
+ 1 次 GCQF forward
```

正式延迟必须在训练完成后实测，不能仅用理论倍数代替。

### 3.3 Stage 1：GeometryQueryProjector

作用：

> 把不同局部视图的 Query 和其坐标、尺度、视图身份编码到统一的全局规范空间。

输入：

```text
local decoder query
local box geometry
view id
local-to-global mapping
```

核心思想：

```text
canonical_local_query
= local_query
+ learnable_geometry_embedding
+ view_embedding
```

它解决的不是普通多尺度特征融合，而是：

> 同一目标在不同 crop 坐标系中的 Query 不能直接比较和交互。

代码主体：

```text
src/gcqf.py
GeometryQueryProjector
```

### 3.4 Stage 2：GlobalLocalQueryInteraction

作用：

> 让局部 tiny Query 读取全局场景上下文，避免局部裁剪把背景、道路关系和大目标上下文全部切掉。

结构：

```text
query = canonical local queries
key/value = global decoder queries
operation = multi-head cross-attention
```

输出：

```text
context-aware local query evidence
```

代码主体：

```text
src/gcqf.py
GlobalLocalQueryInteraction
```

### 3.5 Stage 3：ACR-EG

作用：

> 在已有 tiny anchor 的基础上，学习残差准入、全局保护和证据可靠性，而不是用一个独立硬阈值把大量局部证据全部删掉。

输入包含：

```text
canonical local query
global-context local query
global query
anchor condition / anchor mask
几何与尺度信息
```

它学习：

```text
tiny utility
non-tiny risk
global retain
anchor-conditioned residual evidence
```

当前正式 forward 真正使用的主要输出：

```text
global_retain_logits
```

该输出被注入最后一层非 denoising Query 分类 logits，然后再进入 stock RT-DETR criterion。

代码主体：

```text
src/gcqf.py
AnchorConditionedResidualEvidenceGate
```

### 3.6 正式模型集成

模型类：

```text
src/rtdetr_acr_eg.py
ACREGDetectionModel
```

关键行为：

1. 临时 forward hook 捕获 RT-DETR Decoder 最后一层未 detach Query；
2. global forward 保持梯度；
3. 四个 local forward 使用共享 detector；
4. local query 进入 GCQF；
5. ACR-EG 输出 Query retain logit；
6. 只对非 denoising Query 注入；
7. 使用原 RT-DETR GIoU、分类和 L1 损失；
8. detection loss 反向更新 ACR-EG。

训练器：

```text
src/rtdetr_acr_eg.py
ACREGFormalTrainer
```

冻结协议训练器：

```text
src/gcte_formal_trainer.py
GCTEFormalTrainer
```

YAML：

```text
configs/rtdetr-l-acr-eg.yaml
```

YAML 关键段：

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

### 3.7 当前正式实现仍需如实承认的限制

以下不是推测，而是当前代码审计结果：

1. 当前正式输出只注入 `global_retain_logits`；
2. `score_residual` 和 `adjusted_local_scores` 虽然被计算，但尚未进入最终 logits、boxes 或正式 detection loss；
3. local boxes 尚未作为端到端最终检测集合输出；
4. 部分 residual head 虽进入 optimizer/state_dict/checkpoint，但需要真实 formal loss 的逐参数梯度审计；
5. `gcte_off` 和 `acr_eg_off` 当前都回退 stock global，尚未形成完整三状态消融；
6. 当前 `_live_geometry()` 主要按 `source_shape` 重建固定几何，尚未完整消费 mosaic、flip、random perspective 的真实变换矩阵；
7. 训练期关闭验证，现有 `best.pt` 不能解释为“验证集最优”；
8. 当前独立 evaluator 仍不足以证明 live checkpoint 的真正 1+4 视图端到端指标；
9. 从成熟 baseline 再训练 100 epoch 的结果，必须配 matched continuation control，不能只与原 100-epoch baseline 直接比较并把全部增益归因于模块。

---

## 4. 到目前为止验证了什么

### 4.1 旧 SR-PEG 诊断

旧版 SR-PEG 相对 Global RT-DETR-L：

| 指标 | 增量 |
|---|---:|
| mAP50-95 | `+0.0105948` |
| AP-tiny | `+0.0134117` |
| tiny recall | `+0.0543382` |
| AP-medium | `-0.0016400` |
| AP-large | `-0.0000959` |

它相对内部 Fixed anchor：

```text
mAP50-95 = -0.0165785
```

根因：

```text
Fixed accepted local = 120326
旧 Full accepted local = 23283
空出 max_det 槽 = 7440
```

结论：

> 多视图 tiny 证据本身有效，失败点主要在第三阶段硬门证据利用不足，而不是总体方向无效。

### 4.2 ACR-EG 冻结 Query cache 诊断

这是模块诊断，不是当前 live detector checkpoint 的最终指标。

Global：

| 指标 | 数值 |
|---|---:|
| mAP50-95 | `0.19869887` |
| AP50 | `0.36180752` |
| AP75 | `0.18530324` |
| AP-tiny-SBR | `0.08078923` |
| tiny recall | `0.58207909` |
| AP-medium-SBR | `0.25780549` |
| AP-large-SBR | `0.15464286` |

Full-GCQF / ACR-EG：

| 指标 | 数值 |
|---|---:|
| mAP50-95 | `0.20889331` |
| AP50 | `0.38997759` |
| AP75 | `0.18984129` |
| AP-tiny-SBR | `0.09400049` |
| tiny recall | `0.64029069` |
| AP-medium-SBR | `0.25628104` |
| AP-large-SBR | `0.15455318` |

相对 Global：

| 指标 | 增量 |
|---|---:|
| mAP50-95 | `+0.01019444` |
| AP-tiny-SBR | `+0.01321127` |
| tiny recall | `+0.05821160` |
| AP-medium-SBR | `-0.00152445` |
| AP-large-SBR | `-0.00008967` |

冻结诊断中通过：

```text
map_gain_vs_global
tiny_gain_vs_global
tiny_recall_gain_vs_global
medium_budget_vs_global
large_budget_vs_global
anchor_reference_exact
protected_global_exact
max_det_respected
residual_is_active
residual_not_saturated
medium_recovery_vs_fixed
```

冻结诊断中未通过：

```text
map_nonnegative_vs_fixed
```

但 Fixed-SADED 是内部开发锚点，不是外部公开基线，也不再作为正式主比较的唯一硬门。正式主比较是：

```text
完整方法 vs Global RT-DETR-L
```

### 4.3 真正 YAML 集成已验证

当前 `a22838e3` 正式 run 已证明：

- YAML 中真实声明 GCTE；
- 主模型类是 `ACREGDetectionModel`；
- 输入真实包含 global + 4 local views；
- Decoder Query 被真实捕获；
- ACR-EG 在主 forward/loss 内运行；
- 检测 loss 可向 ACR-EG 反向传播；
- 模块参数进入 MuSGD optimizer；
- 48 个 `acr_eg.*` key 进入 EMA/checkpoint；
- 真实 RTX 4090 多轮训练可运行；
- 每轮 checkpoint 可反序列化；
- checkpoint 含 optimizer、scaler、epoch、updates；
- 已安全完成并备份到第 9 轮。

尚未证明：

- 当前 live checkpoint 的 mAP；
- 当前 live checkpoint 的 AP-tiny / medium / large；
- 当前 live checkpoint 已达到 cache 诊断增益；
- 100 epoch 最终效果；
- 真实延迟和 FLOPs；
- matched continuation control；
- 三个内部阶段各自的正式 live 消融。

---

## 5. 冻结数据与训练协议

### 5.1 数据

```text
数据集：VisDrone
train images：6471
val images：548
类别数：10
```

类名：

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

数据集路径：

```text
/mnt/uav/datasets/VisDrone
```

数据 YAML：

```text
/mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml
```

YAML：

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

冻结签名：

```text
完整数据集：
FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB

548 图 val：
A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A

固定 647 图 train10：
52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0
```

### 5.2 训练参数

| 参数 | 冻结值 |
|---|---|
| Ultralytics | `8.4.90` |
| model | YAML 集成 RT-DETR-L |
| stock 初始化 | 成熟 baseline checkpoint |
| `pretrained` | `False`，禁止 Ultralytics 自动下载预训练权重 |
| epochs | `100` |
| imgsz | `640` |
| batch | `8` |
| workers | `8` |
| device | `0` |
| seed | `0` |
| deterministic | `True` |
| cache | `False` |
| AMP | `True` |
| 固定 AMP scale | `128` |
| optimizer | `MuSGD` |
| lr0 | `0.01` |
| lrf | `0.01` |
| momentum | `0.937` |
| weight_decay | `0.0005` |
| warmup_epochs | `3.0` |
| warmup_momentum | `0.8` |
| warmup_bias_lr | `0.0` |
| nbs | `64` |
| cos_lr | `False` |
| query 数 | `300` |
| max_det | `300` |
| NMS | `False` |
| mosaic | `1.0` |
| close_mosaic | `10` |
| mixup | `0.0` |
| scale | `0.5` |
| translate | `0.1` |
| degrees | `0.0` |
| shear | `0.0` |
| perspective | `0.0` |
| flipud | `0.0` |
| fliplr | `0.5` |
| hsv_h | `0.015` |
| hsv_s | `0.7` |
| hsv_v | `0.4` |
| cutmix | `0.0` |
| copy_paste | `0.0` |
| save_period | `1` |
| val during train | `False` |

---

## 6. 全新服务器硬件和系统要求

推荐严格对齐：

```text
Ubuntu：22.04 x86_64
Python：3.10.12
GPU：NVIDIA GeForce RTX 4090 24 GB
驱动：550.142 或能够稳定支持 CUDA 12.1 PyTorch wheel 的兼容驱动
PyTorch：2.5.1+cu121
TorchVision：0.20.1+cu121
Ultralytics：8.4.90
可用磁盘：推荐 100 GB 以上
```

旧服务器实际环境：

```text
Ubuntu 22.04.5 LTS
Python 3.10.12
torch 2.5.1+cu121
torchvision 0.20.1+cu121
CUDA runtime 12.1
cuDNN 9.1
ultralytics 8.4.90
numpy 2.2.6
opencv-python 5.0.0.93
PyYAML 6.0.3
```

不要为这个裸机 Python 训练额外安装 NVIDIA Container Toolkit。

---

## 7. 新服务器从零安装

以下命令在已经登录的新服务器 shell 中执行。

### 7.1 系统检查

```bash
set -euo pipefail

uname -a
lsb_release -a
nvidia-smi
df -h
free -h
```

必须确认：

```text
Ubuntu 22.04
x86_64
RTX 4090
GPU 可见
可用磁盘至少 80 GB，推荐 100 GB
```

### 7.2 APT 镜像

先备份：

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.gcte-backup
```

Ubuntu 22.04 可将 main、updates 和 backports 使用清华 TUNA；security 保留官方源，避免安全更新同步延迟：

```bash
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

如果系统并非 Ubuntu 22.04，不要继续套用这份 sources 配置。

### 7.3 固定工作目录

```bash
export GCTE_REPO_ROOT=/home/ubuntu/gcte-acr-eg
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

不要把 `$HOME`、`~` 或根目录作为清理脚本目标。

### 7.4 拉取源码

这个仓库是自定义项目，没有可信的通用 GitHub Release 镜像。源码和自定义 checkpoint 使用 GitHub 官方地址，配合重试，不使用来源不明的 GitHub 代理。

```bash
git clone --branch codex/gcte-rtdetr-g0 \
  https://github.com/kkc236/uav-detection-baselines.git \
  "$GCTE_REPO_ROOT"

cd "$GCTE_REPO_ROOT"
git fetch --all --tags
git checkout a22838e3e7cd1cd858d6aad9f42e5b68fab50471
git status --short
git rev-parse HEAD
```

期望：

```text
a22838e3e7cd1cd858d6aad9f42e5b68fab50471
```

如果要使用后续“安全 resume 修复提交”，必须明确记录那个新提交，不能继续把它标成 `a22838e3`。

### 7.5 创建 Python 环境

```bash
python3.10 -m venv "$GCTE_ENV_ROOT"
"$GCTE_ENV_ROOT/bin/python" -m pip install --upgrade pip
```

设置 PyPI 清华镜像：

```bash
"$GCTE_ENV_ROOT/bin/python" -m pip config set \
  global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

本仓库提供旧服务器精确运行环境 lock：

```text
requirements-gcte-acr-eg-cu121.txt
```

因为训练源码必须固定在 `a22838e3`，而该 lock 是之后新增的交接文件，先从交接分支下载 lock：

```bash
curl -fL \
  --retry 10 \
  --retry-all-errors \
  --connect-timeout 30 \
  -o /home/ubuntu/requirements-gcte-acr-eg-cu121.txt \
  https://raw.githubusercontent.com/kkc236/uav-detection-baselines/codex/gcte-rtdetr-g0/requirements-gcte-acr-eg-cu121.txt
```

安装：

```bash
cd "$GCTE_REPO_ROOT"
"$GCTE_ENV_ROOT/bin/python" -m pip install \
  --retries 10 \
  --timeout 120 \
  -r /home/ubuntu/requirements-gcte-acr-eg-cu121.txt
```

该 lock 对 PyTorch 和 TorchVision 使用阿里云 CUDA 12.1 wheel 直链和 wheel SHA256；其他 PyPI 包使用清华镜像。

如果阿里云 wheel 暂时不可用，官方 PyTorch 回退命令为：

```bash
"$GCTE_ENV_ROOT/bin/python" -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

之后再安装其余依赖，且不得改变 torch、torchvision 和 ultralytics 版本。

### 7.6 环境核验

```bash
"$GCTE_ENV_ROOT/bin/python" - <<'PY'
import platform
import torch
import torchvision
import ultralytics
import cv2
import numpy
import yaml

print("python", platform.python_version())
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("opencv", cv2.__version__)
print("numpy", numpy.__version__)
print("pyyaml", yaml.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_runtime", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
```

硬门：

```text
Python 3.10.x
torch 2.5.1+cu121
torchvision 0.20.1+cu121
ultralytics 8.4.90
cuda_available True
GPU NVIDIA GeForce RTX 4090
```

---

## 8. 下载并验证基线和最新检查点

### 8.1 成熟 baseline

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
```

### 8.2 最新完整 ACR-EG 第 9 轮 checkpoint

```bash
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

### 8.3 检查 checkpoint 身份

必须在仓库根目录运行，以便反序列化自定义类：

```bash
cd "$GCTE_REPO_ROOT"

"$GCTE_ENV_ROOT/bin/python" - <<'PY'
from pathlib import Path
import torch

checkpoint_path = Path("/home/ubuntu/gcte-checkpoints/epoch8.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
ema = checkpoint.get("ema")

assert type(ema).__name__ == "ACREGDetectionModel", type(ema)
assert checkpoint.get("optimizer") is not None
assert checkpoint.get("scaler") is not None
assert checkpoint.get("epoch") == 8
assert checkpoint.get("updates") is not None

state = ema.float().state_dict()
acr_keys = sorted(key for key in state if key.startswith("acr_eg."))
assert len(acr_keys) == 48, len(acr_keys)

print("checkpoint_epoch", checkpoint["epoch"])
print("ema_type", type(ema).__name__)
print("acr_eg_keys", len(acr_keys))
print("optimizer", type(checkpoint["optimizer"]).__name__)
print("scaler_present", checkpoint["scaler"] is not None)
print("updates", checkpoint["updates"])
PY
```

任何断言失败都禁止恢复训练。

---

## 9. 迁移数据集

### 9.1 旧服务器仍在线时

在新服务器运行：

```bash
mkdir -p /mnt/uav/datasets/VisDrone

rsync -aH \
  --partial \
  --append-verify \
  --info=progress2 \
  ubuntu@36.103.199.151:/mnt/uav/datasets/VisDrone/ \
  /mnt/uav/datasets/VisDrone/

mkdir -p /mnt/uav/protocols/tsgr-p2-e1

rsync -aH \
  --partial \
  --append-verify \
  ubuntu@36.103.199.151:/mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml \
  /mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml

rsync -aH \
  --partial \
  --append-verify \
  ubuntu@36.103.199.151:/mnt/uav/protocols/tsgr-p2-e1/d2-train-10pct.txt \
  /mnt/uav/protocols/tsgr-p2-e1/d2-train-10pct.txt
```

SSH 密码只在交互提示中输入，不写进命令。

### 9.2 旧服务器已关闭时

使用同一份权威 VisDrone 数据拷贝，放到完全相同路径。不能用另一个转换版本、另一个 label 版本或自动重新划分的数据代替。

### 9.3 文件数核验

```bash
find /mnt/uav/datasets/VisDrone/images/train -type f | wc -l
find /mnt/uav/datasets/VisDrone/labels/train -type f | wc -l
find /mnt/uav/datasets/VisDrone/images/val -type f | wc -l
find /mnt/uav/datasets/VisDrone/labels/val -type f | wc -l
```

期望：

```text
train images = 6471
train labels = 6471
val images = 548
val labels = 548
```

验证集内容签名：

```bash
cd "$GCTE_REPO_ROOT"

"$GCTE_ENV_ROOT/bin/python" - <<'PY'
from pathlib import Path
from src.sbr_artifacts import dataset_content_signature

root = Path("/mnt/uav/datasets/VisDrone")
signature = dataset_content_signature(root, "val")
print(signature)
assert signature.upper() == "A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A"
PY
```

完整数据集父级 seal 还必须保持：

```text
FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB
```

---

## 10. 启动前代码和模型硬门

### 10.1 Git 身份

```bash
cd "$GCTE_REPO_ROOT"
git status --short
git rev-parse HEAD
git diff --check
```

从头启动时必须是：

```text
a22838e3e7cd1cd858d6aad9f42e5b68fab50471
```

### 10.2 运行聚焦测试

```bash
cd "$GCTE_REPO_ROOT"

"$GCTE_ENV_ROOT/bin/python" -m pytest -q \
  tests/test_gcte_formal_cli.py \
  tests/test_rtdetr_acr_eg.py \
  tests/test_gcmv_data.py
```

如果测试文件名在后续提交中变化，以那个提交的明确测试清单为准，但不能跳过：

```text
YAML 配置测试
ACREGDetectionModel 构建测试
Query forward 测试
loss/backward 测试
checkpoint identity 测试
resume identity 测试
```

### 10.3 Dry run

```bash
cd "$GCTE_REPO_ROOT"
mkdir -p "$GCTE_OUTPUT_ROOT"

GCTE_SOURCE_COMMIT="$(git rev-parse HEAD)" \
"$GCTE_ENV_ROOT/bin/python" scripts/train_gcte_formal.py \
  --project "$GCTE_OUTPUT_ROOT" \
  --name acr-eg-dry-run \
  --baseline-checkpoint "$GCTE_BASELINE" \
  --dry-run
```

检查 protocol JSON：

```bash
python3 -m json.tool \
  "$GCTE_OUTPUT_ROOT/acr-eg-dry-run.protocol.json"
```

必须包含：

```text
gcte_enabled=true
gcte_forward_integration=true
gcte_acr_eg_off=false
gcte_off=false
batch=8
workers=8
imgsz=640
seed=0
optimizer=MuSGD
amp_scale=128
baseline SHA 正确
source commit 正确
```

---

## 11. 两种启动模式

### 11.1 模式 A：在全新服务器上从训练 epoch 0 重新开始

这个模式在 `a22838e3` 上已经可执行并已通过真实训练验证。

它的含义：

```text
从成熟 baseline 初始化 stock RT-DETR 参数
+ 新建并训练 YAML 注册的 GCQF / ACR-EG 参数
→ 新的 100-epoch integrated run
```

命令：

```bash
cd "$GCTE_REPO_ROOT"

export GCTE_FRESH_OUTPUT=/home/ubuntu/gcte-acr-eg-formal-output-a22838e3
mkdir -p "$GCTE_FRESH_OUTPUT"

nohup env \
  GCTE_SOURCE_COMMIT=a22838e3e7cd1cd858d6aad9f42e5b68fab50471 \
  "$GCTE_ENV_ROOT/bin/python" scripts/train_gcte_formal.py \
    --project "$GCTE_FRESH_OUTPUT" \
    --name acr-eg-integrated-rtdetr-100 \
    --baseline-checkpoint "$GCTE_BASELINE" \
  > "$GCTE_FRESH_OUTPUT/formal.log" 2>&1 &

echo $! > "$GCTE_FRESH_OUTPUT/runner.pid"
```

只运行一次。启动后先只读核验，不要因为终端暂时没有输出而重复启动。

### 11.2 模式 B：从 GitHub 第 9 轮继续到 100 轮

这是节省时间的目标模式，但当前 `a22838e3` 尚不能安全执行。

当前脚本明确包含：

```python
if args.resume:
    raise ValueError("GCTE_ACR_EG_RESUME_REQUIRES_INTEGRATED_CHECKPOINT")
```

当前 trainer 还包含：

```python
if weights is not None:
    raise ValueError("ACR-EG formal run must load only the sealed mature baseline")
```

所以以下做法全部禁止：

```text
直接在 a22838e3 上传 --resume
用 stock RT-DETR trainer 恢复 epoch8.pt
只加载 state_dict 后当成连续训练
只恢复 EMA 而丢 optimizer/scaler/epoch
把第 9 轮 checkpoint 当 mature baseline
把旧 098da04c stock checkpoint 当 ACR-EG
```

这些做法会失败，或破坏训练连续性，或静默退化为 stock 模型。

---

## 12. 安全集成 resume 必须怎样实现

必须在新提交中测试先行完成，不能直接在服务器旧源码目录修改。

### 12.1 测试先行

先增加失败测试，要求：

1. 只接受 `ACREGDetectionModel` checkpoint；
2. checkpoint 中恰有 48 个 `acr_eg.*` key；
3. 必须有 optimizer；
4. 必须有 scaler；
5. `epoch` 必须为非负且小于 99；
6. `updates` 必须存在；
7. 模型 YAML 必须是 GCTE 开启状态；
8. baseline SHA、dataset 和冻结参数不得漂移；
9. 恢复后 `start_epoch = checkpoint["epoch"] + 1`；
10. 恢复后模型类仍是 `ACREGDetectionModel`；
11. ACR-EG 参数仍在 optimizer；
12. 一个真实 multi-view batch 可 forward/backward；
13. 至少一个实际进入检测路径的 `acr_eg.*` 参数梯度非零；
14. 新输出目录不能覆盖旧输出。

### 12.2 实现原则

Ultralytics 8.4.90 的标准恢复顺序是：

```text
check_resume()
→ self.args.model 指向 .pt
→ setup_model() 反序列化 checkpoint["ema"]
→ get_model(cfg, weights)
→ 创建 optimizer / scaler
→ resume_training(ckpt)
→ 恢复 optimizer、scaler、EMA updates、epoch
```

ACR-EG 修复应：

1. 保留 `ACREGFormalTrainer`；
2. `get_model(weights=...)` 只接受验证通过的 `ACREGDetectionModel`；
3. 用 `configs/rtdetr-l-acr-eg.yaml` 重建模型；
4. 严格加载 stock + `acr_eg.*` 全部 state；
5. 让 Ultralytics `resume_training(ckpt)` 恢复 optimizer、scaler、updates 和 epoch；
6. 重新强制 batch、workers、imgsz、device、save_period、cache、val 等运行参数；
7. 固定 AMP scale 128 的恢复要在载入 scaler 后再次验证；
8. 恢复日志必须打印原 checkpoint epoch、下一 epoch、模型类和 ACR key 数。

### 12.3 通过门

只有满足以下条件，才允许在新服务器真正启动续训：

```text
聚焦测试通过
广泛回归通过
checkpoint SHA 通过
ema type = ACREGDetectionModel
acr_eg keys = 48
optimizer/scaler/updates/epoch 存在
一个真实 multi-view batch forward/backward 通过
ACR-EG 有非零梯度
GPU 真正占用
日志从第 10 轮开始
新输出目录存在且旧证据未覆盖
```

在上述新提交尚未完成前，本文档不提供一个伪装成可用的 `--resume` 命令。

---

## 13. 训练监控

### 13.1 进程

```bash
export GCTE_ACTIVE_OUTPUT=/home/ubuntu/gcte-acr-eg-formal-output-a22838e3

cat "$GCTE_ACTIVE_OUTPUT/runner.pid"
ps -fp "$(cat "$GCTE_ACTIVE_OUTPUT/runner.pid")"
```

### 13.2 GPU

```bash
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv
```

### 13.3 训练日志

```bash
tail -n 100 "$GCTE_ACTIVE_OUTPUT/formal.log"
```

### 13.4 完成轮次

```bash
wc -l \
  "$GCTE_ACTIVE_OUTPUT/acr-eg-integrated-rtdetr-100/results.csv"

tail -n 3 \
  "$GCTE_ACTIVE_OUTPUT/acr-eg-integrated-rtdetr-100/results.csv"

find \
  "$GCTE_ACTIVE_OUTPUT/acr-eg-integrated-rtdetr-100/weights" \
  -maxdepth 1 \
  -type f \
  -name 'epoch*.pt' \
  -printf '%f %s\n' \
  | sort -V \
  | tail
```

### 13.5 磁盘

```bash
df -h /home /mnt/uav
du -sh "$GCTE_ACTIVE_OUTPUT"
```

磁盘低于 20 GB 时：

1. 先上传最新完整 epoch；
2. 核验 GitHub asset size 和 digest；
3. 保留 `last.pt`、`best.pt` 和最新完整 epoch；
4. 只删除已经确认废弃、已经备份的旧实验；
5. 不删除当前源码、baseline、数据、protocol、当前 run 或未备份 checkpoint。

---

## 14. 每轮 GitHub 备份规范

第 `N` 个完成轮次对应：

```text
epoch file = epoch{N-1}.pt
release tag = gcte-acr-eg-a22838e3-epoch-NNN
```

例如：

```text
完成第 9 轮
file = epoch8.pt
tag = gcte-acr-eg-a22838e3-epoch-009
```

每轮必须记录：

```text
source commit
完成轮次
checkpoint 内部 epoch
asset bytes
server SHA256
下载后 SHA256
GitHub digest
Release URL
训练协议身份
```

GitHub 上传成功不以命令退出为唯一依据，必须再查询 Release 资产并比较：

```text
size
digest
draft=false
state=uploaded
```

禁止把训练 checkpoint 称为最终精度结果。

---

## 15. 100 epoch 完成后的正式评测

### 15.1 先实现 live-checkpoint evaluator

评估器必须：

1. 从 checkpoint 加载 `ACREGDetectionModel`；
2. 拒绝 stock RT-DETR 模型；
3. 检查 48 个 `acr_eg.*` key；
4. 对 548 张 val 图真实生成 global + 4 local views；
5. 强制执行 GCQF 和 ACR-EG；
6. 检查没有静默回退 stock global；
7. 输出预测和指标 JSON；
8. 输出 checkpoint、YAML、dataset、源码 SHA；
9. 测量端到端 latency、GPU memory、参数量和 FLOPs。

### 15.2 正式指标

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
参数量
FLOPs
峰值显存
端到端 latency
FPS
```

### 15.3 主比较

论文主表：

```text
RT-DETR-L Global baseline
vs
RT-DETR-L + 完整 GCQF / ACR-EG
```

同时必须补 matched continuation control：

```text
Control：
同一个 mature baseline
→ 不加 GCQF
→ 继续相同训练轮数

Method：
同一个 mature baseline
→ 加 GCQF / ACR-EG
→ 继续相同训练轮数
```

两组保持：

```text
数据
batch
学习率
seed
训练步数
增强
验证集
```

唯一差别是网络模块。

### 15.4 消融

至少需要：

| 实验 | Stage 1 GQP | Stage 2 GLQI | Stage 3 ACR-EG |
|---|---:|---:|---:|
| Baseline | 否 | 否 | 否 |
| + GQP | 是 | 否 | 否 |
| + GQP + GLQI | 是 | 是 | 否 |
| Full GCQF | 是 | 是 | 是 |

还需：

```text
Global only
Multi-view without learnable gate
ACR-EG residual off
anchor condition off
不同 local view 数
不同 residual_eta
```

不能让 `acr_eg_off` 和 `gcte_off` 都返回同一个 stock 状态后声称完成了三状态消融。

---

## 16. 现在允许和禁止的论文结论

### 16.1 现在可以说

- SADED-SM 证明局部高分辨率多视图能恢复 tiny 证据；
- 基于该现象设计了三阶段 GCQF 网络模块；
- GCQF 输入 global/local RT-DETR Decoder Query；
- GQP 学习几何规范化局部 Query；
- GLQI 用 cross-attention 注入全局上下文；
- ACR-EG 学习 anchor-conditioned residual evidence；
- 模块进入 YAML、主模型 forward、loss、optimizer、EMA 和 checkpoint；
- 冻结 Query cache 诊断相对 Global 提升 mAP、AP-tiny 和 tiny recall；
- 真正集成训练已完成至少 9 个 epoch，并有可验证检查点。

### 16.2 现在不能说

- 当前 live integrated 模型已经超过 baseline；
- 当前第 9 轮或最终 100 轮 mAP 已知；
- 所有尺度指标都提升；
- local boxes 已端到端进入最终检测集合；
- score residual 已被正式 detection loss 有效训练；
- `best.pt` 是验证集最优；
- 当前 run 与 baseline 总训练预算相同；
- 延迟只增加 2–3 倍；
- 已完成三 seed、第二数据集、test-dev 或 SOTA 对比。

---

## 17. 故障处理原则

发生异常时：

1. 先只读检查进程、GPU、磁盘、日志和 checkpoint；
2. 不重复启动；
3. 不删除当前输出；
4. 先下载最新完整 epoch；
5. 计算 SHA256；
6. 确认 checkpoint 可反序列化；
7. 写失败测试；
8. 在本地新提交修复；
9. 部署到新的源码目录；
10. 使用新的输出目录；
11. 只从经过身份验证的集成 checkpoint 恢复。

禁止：

```text
服务器代码就地热改
覆盖旧输出
放宽科学门伪造通过
把 stock checkpoint 标成 ACR-EG
只加载模型权重却称为 true resume
关闭服务器
杀无关进程
把密码或 GitHub token 写进仓库
```

---

## 18. 权威文件索引

| 内容 | 路径 |
|---|---|
| 正式训练入口 | `scripts/train_gcte_formal.py` |
| YAML 集成模型 | `configs/rtdetr-l-acr-eg.yaml` |
| 正式模型和训练器 | `src/rtdetr_acr_eg.py` |
| GCQF 三阶段主体 | `src/gcqf.py` |
| 数据集多视图生成 | `src/gcmv_data.py` |
| 冻结 MuSGD / AMP | `src/gcte_formal_trainer.py` |
| ACR-EG 配置加载 | `src/acr_eg_integration.py` |
| 精确 Python 环境 | `requirements-gcte-acr-eg-cu121.txt` |
| 完整方法审计 | `docs/handoffs/2026-07-28-gcte-acr-eg-complete-method-status.md` |
| 本迁移手册 | `docs/handoffs/2026-07-28-gcte-acr-eg-clean-server-migration-and-resume.md` |
| ACR-EG cache 评估 | `docs/evidence/gcte-acr-eg-round1-evaluation.json` |
| 最新备份证据 | `docs/evidence/gcte-acr-eg-integrated-checkpoint-backups.json` |

---

## 19. 镜像与官方来源

PyPI 清华镜像：

```text
https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
```

Ubuntu 清华镜像：

```text
https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/
```

阿里云 PyTorch CUDA 12.1 wheel：

```text
https://mirrors.aliyun.com/pytorch-wheels/cu121/
```

PyTorch 官方历史版本：

```text
https://docs.pytorch.org/get-started/previous-versions/
```

清华 GitHub Release 镜像说明：

```text
https://mirrors.tuna.tsinghua.edu.cn/help/github-release/
```

该 GitHub Release 镜像只覆盖其收录项目，不保证包含本自定义仓库。因此本项目源码和 checkpoint 使用 GitHub 官方地址，不使用不受信任的第三方代理。

---

## 20. 最短决策

如果新服务器必须立刻开跑且不能等待代码修复：

```text
使用模式 A
→ 从 mature baseline 重新开始完整 100 epoch
```

如果目标是保留旧服务器已经完成的 9 个 epoch：

```text
先完成安全集成 resume 的 TDD 修复
→ 新提交、新源码目录、新输出目录
→ 从 GitHub epoch8.pt 开始第 10 轮
```

无论选择哪一个模式，都不能：

```text
退化为 stock RT-DETR
跳过 dataset/checkpoint SHA
改变冻结训练参数
覆盖已有证据
把训练期占位 mAP 当结果
```
