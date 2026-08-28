# VisDrone LRS System 跨服务器运行手册

## Material Passport

| 字段 | 内容 |
| --- | --- |
| Origin Skill | `academic-research-suite / experiment-agent` |
| Mode | `plan` |
| Date | `2026-08-28` |
| Status | `ANALYZED` |
| Scope | VisDrone 上的 LRS-FDR、BPDD、FIA 三个正式训练臂，以及 UAVDT 的后续两臂迁移计划 |

本手册只记录可由当前源码和已冻结协议复现的运行边界。权重、数据集和初始状态文件均为服务器上的外部输入，不随源码仓库提交。

## 1. 源码、臂映射与入口

源码仓库为 `https://github.com/kkc236/uav-detection-baselines.git`，本手册对应分支为 `codex/lrs-system-visdrone-rebuild`。唯一统一入口为 [`scripts/train_visdrone_lrs_system.py`](../scripts/train_visdrone_lrs_system.py)。

VisDrone 三个臂的映射固定如下；字母只是论文表中的别名：

| 臂 | 论文身份 | 配置 | LRS-FDR | BPDD | FIA |
| --- | --- | --- | :---: | :---: | :---: |
| `g` | LRS-FDR + BPDD | [`configs/rtdetr-l-lrs-fdr-bpdd.yaml`](../configs/rtdetr-l-lrs-fdr-bpdd.yaml) | 是 | 是 | 否 |
| `h` | LRS-FDR + FIA | [`configs/rtdetr-l-lrs-fdr-fia.yaml`](../configs/rtdetr-l-lrs-fdr-fia.yaml) | 是 | 否 | 是 |
| `i` | LRS-FDR + BPDD + FIA | [`configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml`](../configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml) | 是 | 是 | 是 |

入口不接受通过命令行改变 epochs、seed、batch、optimizer、学习率、增强、FIA 私有 seed 或 resume 状态；这些值由源码中的 Formal100 冻结设置提供。

## 2. 精确环境与安装

目标环境是 Python 3.10、PyTorch `2.5.1+cu121`、Torchvision `0.20.1+cu121`、CUDA 12.1 运行时和 Ultralytics `8.4.90`。建议使用 Linux 主机上的 CUDA 12.1 wheel 索引安装 PyTorch；其余依赖由 [`requirements.txt`](../requirements.txt) 安装。

```bash
git clone https://github.com/kkc236/uav-detection-baselines.git
cd uav-detection-baselines
git switch codex/lrs-system-visdrone-rebuild
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

可用以下只读检查确认解释器和关键包：

```bash
python -c "import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__)"
```

## 3. 外部输入与目录约定

训练前必须准备两个外部输入：

1. `VISDRONE_ROOT`：已经转换为 YOLO 目录结构的 VisDrone 根目录，至少包含 `images/train`、`images/val`、`labels/train` 和 `labels/val`。当前 VisDrone authority 要求 train 6471 张、val 548 张、10 类，并由入口调用 [`scripts/train_rtdetr_fdr.py`](../scripts/train_rtdetr_fdr.py) 的数据准备和签名校验。
2. `INITIAL_STATE`：结构上有效的 FDR 初始状态 artifact。它必须通过入口的结构验证；不要求与历史服务器使用的原始 hash 相同。历史 raw hash 不是本次跨服务器启动的输入条件。

本手册使用且仅使用以下环境变量：

```bash
export VISDRONE_ROOT=/data/uav/datasets/VisDrone
export INITIAL_STATE=/data/uav/weights/fdr-initial-state.pt
export RUNS_ROOT=/data/uav/runs/lrs-system-visdrone
mkdir -p "$RUNS_ROOT"
```

请把三个值替换为当前服务器上的真实路径；路径必须在执行 dry-run 前可读。不要把数据或初始状态复制到 Git 工作树中。

## 4. Frozen Formal100 设置

下表是三个臂共同的训练协议。除模型配置和运行名称外，`g/h/i` 的解析设置必须相同。

| 项目 | 冻结值 |
| --- | --- |
| 数据 | VisDrone train/val；6471 / 548 张；10 类 |
| 初始化 | scratch，`pretrained=False` |
| epochs | 100 |
| seed / deterministic | 0 / `True` |
| imgsz / batch / workers | 640 / 8 / 8 |
| device | `0` |
| optimizer | `MuSGD` |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight decay | 0.937 / 0.0005 |
| warmup epochs / momentum / bias lr | 3.0 / 0.8 / 0.0 |
| nominal batch `nbs` / cosine LR | 64 / `False` |
| AMP / cache | `True` / `False` |
| queries / max_det / NMS | 300 / 300 / `False` |
| mosaic / close_mosaic | 1.0 / 10 |
| mixup / cutmix / copy_paste | 0.0 / 0.0 / 0.0 |
| scale / translate | 0.5 / 0.1 |
| degrees / shear / perspective | 0.0 / 0.0 / 0.0 |
| flipud / fliplr | 0.0 / 0.5 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 |

LRS 图固定为 `preliminary_box=false`、`distribution_feedback=false`、FGL `fgl_weight=0.15`、`supervise_pre_boxes=false`、`supervise_dn_fdr=false`、`edge_adaptive_fgl=false`、`reliability_shrinkage_alpha=0.25`。BPDD（仅 `g` 和 `i`）固定为 weight 0.5、temperature 0.5、margin 0.02、matched layer `final`、`include_dn=false`。FIA（仅 `h` 和 `i`）固定为 P3-only 图层，并保持零初始化 residual scale 与独立私有初始化。

## 5. Dry-run 与训练命令

每个命令都显式绑定臂、数据根、初始状态和输出根。先执行 dry-run；它会完成数据、初始状态、源码和配置 authority 校验，写出 authority JSON，并在构造 Trainer 前返回。

### `g`: LRS-FDR + BPDD

```bash
python scripts/train_visdrone_lrs_system.py \
  --arm g \
  --dataset-root "$VISDRONE_ROOT" \
  --initial-state "$INITIAL_STATE" \
  --output-root "$RUNS_ROOT" \
  --dry-run

python scripts/train_visdrone_lrs_system.py \
  --arm g \
  --dataset-root "$VISDRONE_ROOT" \
  --initial-state "$INITIAL_STATE" \
  --output-root "$RUNS_ROOT"
```

### `h`: LRS-FDR + FIA

```bash
python scripts/train_visdrone_lrs_system.py \
  --arm h \
  --dataset-root "$VISDRONE_ROOT" \
  --initial-state "$INITIAL_STATE" \
  --output-root "$RUNS_ROOT" \
  --dry-run

python scripts/train_visdrone_lrs_system.py \
  --arm h \
  --dataset-root "$VISDRONE_ROOT" \
  --initial-state "$INITIAL_STATE" \
  --output-root "$RUNS_ROOT"
```

### `i`: LRS-FDR + BPDD + FIA

```bash
python scripts/train_visdrone_lrs_system.py \
  --arm i \
  --dataset-root "$VISDRONE_ROOT" \
  --initial-state "$INITIAL_STATE" \
  --output-root "$RUNS_ROOT" \
  --dry-run

python scripts/train_visdrone_lrs_system.py \
  --arm i \
  --dataset-root "$VISDRONE_ROOT" \
  --initial-state "$INITIAL_STATE" \
  --output-root "$RUNS_ROOT"
```

如需自定义运行名，只能使用一个安全的路径组件，例如增加 `--name formal-seed0-lrs_fdr_bpdd-v1-retry`；不可使用路径分隔符。一次 authority JSON 已存在且内容不同，入口会拒绝覆盖。

## 6. Authority、训练结果与复评边界

默认运行名和结果路径如下：

| 臂 | authority JSON | 训练结果目录 |
| --- | --- | --- |
| `g` | `$RUNS_ROOT/authority/formal-seed0-lrs_fdr_bpdd-v1.json` | `$RUNS_ROOT/formal-seed0-lrs_fdr_bpdd-v1/` |
| `h` | `$RUNS_ROOT/authority/formal-seed0-lrs_fdr_fia-v1.json` | `$RUNS_ROOT/formal-seed0-lrs_fdr_fia-v1/` |
| `i` | `$RUNS_ROOT/authority/formal-seed0-lrs_fdr_bpdd_fia-v1.json` | `$RUNS_ROOT/formal-seed0-lrs_fdr_bpdd_fia-v1/` |

每条训练结果目录中应保留 Ultralytics 的 `results.csv`、`args.yaml` 和 `weights/best.pt`、`weights/last.pt`（实际保存状态以训练过程为准）。入口生成的正式数据 YAML 位于 `$RUNS_ROOT/authority/data/formal-data.yaml`。authority JSON 记录臂、method、源码 commit/tree hash、配置文件 hash、初始状态 hash、数据签名和完整解析设置；它是跨服务器审计的首要结果文件。

正式比较应使用两个 checkpoint 的同一独立 evaluator、同一验证预处理、`imgsz=640`、`batch=8`、`conf=0.001`、IoU `0.50:0.05:0.95`、`max_det=300`、NMS 关闭和相同类别映射。至少归档 Precision、Recall、AP50、AP75、mAP50-95，以及可用时的逐类指标；不得把不同 evaluator 或不同 best 选择规则的数值混在一起。

## 7. UAVDT 第二数据集计划

第二数据集正式名称是 **UAVDT**。用户有时将其称为 **UAT**，但为避免歧义，论文、配置和目录名称一律使用 `UAVDT`，不使用 `UAT` 作为数据集标识。

### 7.1 只保留两臂矩阵

UAVDT 只做以下两臂：

| 臂 | 身份 | 状态 |
| --- | --- | --- |
| Baseline | 已完成的既有参考 | 可复用，但必须先绑定其完整 authority |
| Full | `LRS-FDR+BPDD+FIA` | 待启动；仅在 Baseline 绑定和 UAVDT authority 就绪后训练 |

Baseline 只是已完成结果的可复用参考，不重新编造或补写缺失实验值。启动 Full 之前，必须从 Baseline 的 args、训练 log 和 authority 中抽取并绑定：checkpoint 路径与 hash、source commit、数据 YAML、split、类别映射与签名、seed、imgsz、batch 或 effective batch、epochs、optimizer、schedule、augmentation、preprocessing、evaluator、confidence、max_det、best 选择规则及全部已报告指标。缺失字段必须从这些既有材料中提取；不得猜测或填入默认值。

Full 必须逐项匹配已绑定 Baseline 的 dataset split、class mapping、seed、image size、batch/effective batch、epochs、optimizer/schedule、augmentation、preprocessing、evaluator、confidence、max_det 和 best-val 选择规则。唯一允许的实验差异是 method graph；数据、训练、预处理和评估协议不得因换数据集而隐式漂移。Only method graph may differ。

### 7.2 UAVDT 入口前置条件

不要直接把 [`scripts/train_visdrone_lrs_system.py`](../scripts/train_visdrone_lrs_system.py) 用于 UAVDT；该入口调用 VisDrone dataset authority，会拒绝非 VisDrone 签名。开始 Full 之前必须先实现并审计一个 dataset-specific UAVDT launcher/data authority（dataset-specific UAVDT launcher/data authority is required before starting Full）。该 launcher 应复用已签入的 Full 图配置 [`configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml`](../configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml)，并由已验证的 UAVDT data YAML 提供并覆盖 `nc`；不得把 VisDrone 的类别数或目录校验硬编码带入 UAVDT。

### 7.3 标注准备与数据 authority 清单

清单保持 source-agnostic，但每一项都必须绑定 Baseline 的精确映射：

- 使用官方 UAVDT images 与 annotations，并保存来源和版本记录。
- 按 Baseline 使用的确定性规则生成完全相同语义的 train/val split；不自行发明 split 比例或张数。
- 转为 YOLO normalized `cx cy w h`，类别 id 为从 0 开始且连续的整数；类别顺序和名称逐项复制 Baseline authority。
- 对 ignored regions 的丢弃或处理方式与 Baseline 完全一致，并在 authority 中说明规则。
- 校验缺失标签、空标签、坐标越界、非法宽高、图片—标签配对和重复/孤立文件。
- 记录每类 histogram、图片数、标签数、目标数、目录结构和最终数据签名；所有数量必须来自实际扫描或 Baseline authority，不能臆造。
- 在生成 Full launcher 的 dry-run 中同时校验数据 YAML、split、class mapping、签名和 `nc`，并把这些值写入不可变 UAVDT authority。

### 7.4 接受标准与 claim boundary

Full 必须在匹配协议下完成 seed0 训练。随后用同一 evaluator 独立评估 Baseline 与 Full 的两个 best checkpoint，报告 Precision、Recall、AP50、AP75、mAP50-95，并在 evaluator 提供时报告逐类指标。若 Full 的 mAP50-95 高于 Baseline，这是对跨数据集支持的正向证据；若不高，不否定 VisDrone 结论，但不能据此声称方法具有跨数据集泛化优势。

这两个臂只能检验整体 transfer；不能隔离单个模块贡献，也不能据此主张统计显著性。UAVDT 结论必须保持为匹配协议下的两臂、单 seed 证据，除非后续另行完成预先规定的多 seed 设计。

执行优先级固定为：

1. 绑定 Baseline；
2. 准备并验证 UAVDT 数据；
3. 创建 UAVDT-specific launcher 并完成 dry-run；
4. 只训练 Full；
5. 成对独立评估；
6. 发布 evidence。

在第 3 步完成前，不得开始 UAVDT Full 训练。
