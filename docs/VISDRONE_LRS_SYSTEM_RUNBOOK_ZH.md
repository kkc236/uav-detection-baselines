# VisDrone 当前最终方案跨服务器运行手册

## 1. 适用范围

本文指导当前最终源码在 VisDrone 上运行三个训练臂：

| 臂 | 论文身份 | 启动器 method | 配置 |
| --- | --- | --- | --- |
| g | LRS-FDR + AC-BPDD | lrs_fdr_ac_bpdd | configs/rtdetr-l-lrs-fdr-bpdd.yaml |
| h | LRS-FDR + FIA | lrs_fdr_fia | configs/rtdetr-l-lrs-fdr-fia.yaml |
| i | Full：LRS-FDR + AC-BPDD + FIA | lrs_fdr_ac_bpdd_fia | configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml |

统一入口为 scripts/train_visdrone_lrs_system.py，参数使用 --arm {g,h,i}。
字母只是启动别名，论文和结果统计以上表的 method 身份为准。

当前代码分支为 codex/fdr-feasible-geometry-uavdt。旧 tag
visdrone-lrs-system-v1 指向修改前的 445120b7...，不包含 AC-BPDD 和新的
FDR 合法几何投影，禁止用它启动当前最终方案。

UAVDT 的 Baseline/Full 实验见
[UAVDT 运行手册](UAVDT_EXPERIMENT_HANDOFF_ZH.md)。两个数据集的启动器不能互换。

## 2. 当前模块语义

三个臂共同使用当前 LRS-FDR 和 FDR 合法几何投影。Integral 输出进入
distance2bbox 前，左右距离与上下距离分别做成对投影，使最终宽高至少为
1e-3。当原始距离可行时，该投影是严格恒等映射；它不是将每个负坐标独立置零。

AC-BPDD 只在 g 和 i 中启用：

~~~yaml
bpdd_loss:
  enabled: true
  weight: 0.15
  temperature: 0.5
  margin: 0.02
  eps: 1.0e-6
  assignment_mode: consistent
  include_dn: false
~~~

它只在浅层和未来层保持相同 (batch, query, ground-truth) 匹配时蒸馏边界
分布，并用 better-only reliability gate 抑制无收益教师。它只作用于训练损失，
不增加推理参数和后处理。

FIA 只在 h 和 i 中启用，位于 P3 路径，保持零初始化 residual scale 和
独立私有初始化。LRS-FDR 的公共设置为：

~~~yaml
fdr_loss:
  fgl_weight: 0.15
  supervise_pre_boxes: false
  supervise_dn_fdr: false
  edge_adaptive_fgl: false
  reliability_shrinkage_alpha: 0.25
~~~

当前严格内部消融是 h 对 i：二者都含 LRS-FDR、合法几何投影和 FIA，主要差异
是 i 增加 AC-BPDD。历史旧 BPDD 或旧 Full 结果只能作历史参考，不能替代同源码重训。

## 3. 新服务器获取源码

~~~bash
git clone https://github.com/kkc236/uav-detection-baselines.git
cd uav-detection-baselines
git fetch origin
git switch --detach origin/codex/fdr-feasible-geometry-uavdt
git status --short
git rev-parse HEAD
~~~

git status --short 必须为空。保存 git rev-parse HEAD 的输出；启动器也会把
source commit/tree 写入 authority JSON。跨服务器比较时优先要求源码身份一致。

## 4. 环境安装

仓库固定 ultralytics==8.4.90。推荐沿用 PyTorch 2.5.1+cu121、
Torchvision 0.20.1+cu121 和 CUDA 12.1 运行时：

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
~~~

~~~bash
nvidia-smi
python - <<'PY'
import torch, torchvision, ultralytics
assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY
~~~

使用不同 Python、CUDA 或 GPU 时必须记录实际版本；同表比较优先使用相同环境。

## 5. 必要外部输入

1. VISDRONE_ROOT：YOLO 结构的 VisDrone 根目录，至少包含 images/train、
   images/val、labels/train、labels/val。当前 authority 期望 train 6471 张、
   val 548 张、10 类。
2. INITIAL_STATE：能通过仓库 weights-only 验证器读取的 FDR initial-state。
   三个臂必须使用同一文件，禁止某一臂续训旧 checkpoint。
3. RUNS_ROOT：新的输出根目录，不要复用已有同名 authority 的目录。

~~~bash
export VISDRONE_ROOT=/data/uav/datasets/VisDrone
export INITIAL_STATE=/data/uav/protocols/fdr/initial-state.pt
export RUNS_ROOT=/data/uav/runs/visdrone-ac-bpdd
mkdir -p "$RUNS_ROOT"
~~~

历史 formal initial-state 的登记 SHA-256 为：

~~~text
51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D
~~~

该文件缺失时，可以用结构有效的新 initial-state 启动新的同源三臂实验，但必须
让 g/h/i 共用新文件；这种实验不得冒充与旧 hash 严格同初始化。

~~~bash
test -d "$VISDRONE_ROOT/images/train"
test -d "$VISDRONE_ROOT/images/val"
test -d "$VISDRONE_ROOT/labels/train"
test -d "$VISDRONE_ROOT/labels/val"
test -f "$INITIAL_STATE"
sha256sum "$INITIAL_STATE"
~~~

## 6. 冻结 Formal100 协议

启动器不暴露 epochs、seed、batch、优化器、学习率、增强、BPDD 权重、FIA seed
或 resume 参数。三个臂除模型配置和运行名外共享以下设置：

| 项目 | 固定值 |
| --- | --- |
| epochs / seed / deterministic | 100 / 0 / True |
| scratch policy | pretrained=False，禁止 resume |
| imgsz / batch / workers / device | 640 / 8 / 8 / 0 |
| optimizer | MuSGD |
| lr0 / lrf / cosine | 0.01 / 0.01 / False |
| momentum / weight decay | 0.937 / 0.0005 |
| warmup epochs / momentum / bias lr | 3.0 / 0.8 / 0.0 |
| nbs / AMP / cache | 64 / True / False |
| max_det / NMS | 300 / False |
| mosaic / close_mosaic | 1.0 / 10 |
| scale / translate | 0.5 / 0.1 |
| fliplr / flipud | 0.5 / 0.0 |
| mixup / cutmix / copy_paste | 0.0 / 0.0 / 0.0 |
| save / save_period / val / plots | True / -1 / True / True |

## 7. 必须先做 dry-run

~~~bash
for arm in g h i; do
  python scripts/train_visdrone_lrs_system.py \
    --arm "$arm" \
    --dataset-root "$VISDRONE_ROOT" \
    --initial-state "$INITIAL_STATE" \
    --output-root "$RUNS_ROOT" \
    --dry-run
done
~~~

dry-run 会验证数据、初始状态结构、配置和源码身份，并在
$RUNS_ROOT/authority/ 写入 JSON；它不会构造 Trainer 或开始训练。逐项确认：

- g.method == lrs_fdr_ac_bpdd；
- h.method == lrs_fdr_fia；
- i.method == lrs_fdr_ac_bpdd_fia；
- 三份记录的 source、dataset、initial-state 和除 model/name 外的设置一致；
- 配置 SHA-256 与当前 checkout 的文件一致；
- resume 不存在，epochs == 100，seed == 0。

## 8. 正式训练

建议串行执行。若只验证最终系统，可以只跑 i；但 h 对 i 的当前 AC-BPDD
内部消融必须两臂都重新训练。

### G：LRS-FDR + AC-BPDD

~~~bash
python scripts/train_visdrone_lrs_system.py \
  --arm g \
  --dataset-root "$VISDRONE_ROOT" \
  --initial-state "$INITIAL_STATE" \
  --output-root "$RUNS_ROOT"
~~~

### H：LRS-FDR + FIA

~~~bash
python scripts/train_visdrone_lrs_system.py \
  --arm h \
  --dataset-root "$VISDRONE_ROOT" \
  --initial-state "$INITIAL_STATE" \
  --output-root "$RUNS_ROOT"
~~~

### I：Full（LRS-FDR + AC-BPDD + FIA）

~~~bash
python scripts/train_visdrone_lrs_system.py \
  --arm i \
  --dataset-root "$VISDRONE_ROOT" \
  --initial-state "$INITIAL_STATE" \
  --output-root "$RUNS_ROOT"
~~~

默认结果路径：

| 臂 | 结果目录 | authority JSON |
| --- | --- | --- |
| g | formal-seed0-lrs_fdr_ac_bpdd-v1/ | authority/formal-seed0-lrs_fdr_ac_bpdd-v1.json |
| h | formal-seed0-lrs_fdr_fia-v1/ | authority/formal-seed0-lrs_fdr_fia-v1.json |
| i | formal-seed0-lrs_fdr_ac_bpdd_fia-v1/ | authority/formal-seed0-lrs_fdr_ac_bpdd_fia-v1.json |

重跑时使用新的 RUNS_ROOT，或用 --name 提供新的单一安全目录名；不要覆盖已有结果。

## 9. 结果统计

每个结果目录至少应保存 results.csv、args.yaml、weights/best.pt 和
weights/last.pt。论文表使用各臂截至 100 epoch 的 best-val mAP50-95 选轮，
并同时报告该行的 P、R、AP50：

~~~bash
python - \
  "$RUNS_ROOT/formal-seed0-lrs_fdr_ac_bpdd-v1/results.csv" \
  "$RUNS_ROOT/formal-seed0-lrs_fdr_fia-v1/results.csv" \
  "$RUNS_ROOT/formal-seed0-lrs_fdr_ac_bpdd_fia-v1/results.csv" <<'PY'
import csv
import sys
from pathlib import Path

columns = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
for filename in sys.argv[1:]:
    path = Path(filename)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"empty results.csv: {path}")
    missing = [key for key in columns if key not in rows[0]]
    if missing:
        raise SystemExit(f"missing columns in {path}: {missing}")
    best = max(rows, key=lambda row: float(row[columns[-1]]))
    print(path.parent.name)
    print({"epoch": best.get("epoch"), **{key: best[key] for key in columns}})
PY
~~~

比较时使用“各自截至同一总训练预算的 best”，不要把某臂单轮指标与另一臂
截至当时的 best 混用。当前仓库尚未提供统一 AP75/逐类独立复评入口，因此补齐
统一 evaluator 前，只直接比较同一 Trainer schema 的 P、R、AP50、mAP50-95。

## 10. 验收和论文边界

- 三臂源码 commit、数据签名、initial-state hash 和冻结设置一致；
- 三臂都 fresh 开始，没有续训旧 checkpoint；
- 全程没有负 giou_loss、非有限 loss/gradient 或 AMP 反复降 scale；
- g/i 的 AC-BPDD loss 和 active stable-match 不应始终为零；当前 VisDrone
  启动器尚未像 UAVDT 入口一样持久化 full-runtime.jsonl，因此 results.csv
  本身不能证明 AC-BPDD 实际激活，论文若使用该机制统计需另行补录运行诊断；
- i 相对 h 的 best mAP50-95 为正，才支持 AC-BPDD 在 Full 中的内部增益；
- i 相对同协议 baseline 的整体提升，才支持完整方法的主结果。

单 seed 正结果可以作为一般会议实验依据，但应表述为单次匹配协议结果，不声称
统计显著性。VisDrone 使用当前三臂、UAVDT 只使用 Baseline 和 Full 时，第二数据集
支持的是整体方法可迁移性，不是每个子模块的独立泛化证明。
