# VisDrone LRS v2 现有初始状态四臂服务器交接

> 这份交接包把实验设计改为使用当前已有的 `935D...` 初始状态，供另一台服务器直接执行。它不沿用当前交接包中不存在的 `CA8B...` 初始状态。

## 1. 实验身份

| 项目 | 固定值 |
|---|---|
| 实验 ID | `LRS-V2-FOUR-ARM-EXISTING-STATE-20260907-1` |
| 代码分支 | `codex/visdrone-v2-existing-state-handoff-20260907` |
| 方法 revision | `v2-fp32-extent-logspace-ac-bpdd` |
| 数据 | VisDrone，train 6471 / val 548 / test-dev 1610 |
| 数据签名 | `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB` |
| 初始状态文件 | `initial-state-seed0.pt` |
| 初始状态字节数 | `261864454` |
| 初始状态 SHA-256 | `935D68428BFC5848DE51F82844658ABBC7EC8C99CBBF6BBF823166D38CB65D5B` |
| 训练预算 | 100 epochs / seed 0 / batch 8 / imgsz 640 |
| 设备 | 单张 CUDA GPU，推荐 RTX 4090 |
| 环境 | Python 3.10，PyTorch 2.5.1+cu121，torchvision 0.20.1，Ultralytics 8.4.90 |

初始状态不提交到 GitHub：文件约 250 MiB，需通过安全文件传输单独放到服务器；SHA 不一致必须停止，不得继续训练。

## 2. 四臂矩阵

| 臂 | 运行身份 | AC-BPDD | FIA | 用途 |
|---|---|---:|---:|---|
| f | `lrs_fdr_feasible` | 关 | 关 | v2 几何基线 |
| g | `lrs_fdr_ac_bpdd` | 开 | 关 | BPDD 主效应 |
| h | `lrs_fdr_fia` | 关 | 开 | FIA 主效应 |
| i | `lrs_fdr_ac_bpdd_fia` | 开 | 开 | Full 联合臂 |

四臂必须使用同一 commit、同一数据签名、同一初始状态、同一 seed 和同一训练设置。不得用 `g-formal100-best.pt` 初始化其他臂；其他臂必须从 `initial-state-seed0.pt` fresh start。

## 3. 新服务器准备

```bash
git clone --branch codex/visdrone-v2-existing-state-handoff-20260907 \
  https://github.com/kkc236/uav-detection-baselines.git
cd uav-detection-baselines
git status --porcelain
git rev-parse HEAD
python -V
python -c 'import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__)'
```

若环境尚未安装，可先使用国内 PyPI 镜像安装通用依赖；CUDA 版 PyTorch 按服务器 CUDA 兼容性安装：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics==8.4.90
```

然后确认 `torch==2.5.1+cu121`、`torchvision==0.20.1` 和 CUDA 可用；不要为了绕过 OOM 自动改变 batch 或降级 AMP。

## 4. 放置并核验输入

```bash
export VISDRONE_ROOT=/data/datasets/VisDrone
export INITIAL_STATE=/data/weights/initial-state-seed0.pt
export RUNS_ROOT=/data/runs/visdrone-lrs-v2-existing-state
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

test -f "$INITIAL_STATE"
test "$(stat -c%s "$INITIAL_STATE")" = "261864454"
test "$(sha256sum "$INITIAL_STATE" | awk '{print toupper($1)}')" = \
  "935D68428BFC5848DE51F82844658ABBC7EC8C99CBBF6BBF823166D38CB65D5B"
test -d "$VISDRONE_ROOT"
mkdir -p "$RUNS_ROOT"
```

## 5. 先 dry-run，后正式训练

逐臂 dry-run 必须全部成功，并检查输出的 `authority/*.json` 中的 source、config、initial_state 和 dataset SHA：

```bash
for arm in f g h i; do
  python scripts/train_visdrone_lrs_system.py \
    --arm "$arm" \
    --dataset-root "$VISDRONE_ROOT" \
    --initial-state "$INITIAL_STATE" \
    --output-root "$RUNS_ROOT" \
    --dry-run
done
```

确认四臂 authority 均指向同一个初始状态 SHA 后，按 f → g → h → i 串行执行正式 100 轮：

```bash
set -euo pipefail
for arm in f g h i; do
  python scripts/train_visdrone_lrs_system.py \
    --arm "$arm" \
    --dataset-root "$VISDRONE_ROOT" \
    --initial-state "$INITIAL_STATE" \
    --output-root "$RUNS_ROOT"
done
```

每臂输出目录默认为：

```text
$RUNS_ROOT/formal-seed0-lrs_fdr_feasible-v2
$RUNS_ROOT/formal-seed0-lrs_fdr_ac_bpdd-v2
$RUNS_ROOT/formal-seed0-lrs_fdr_fia-v2
$RUNS_ROOT/formal-seed0-lrs_fdr_ac_bpdd_fia-v2
```

每臂完成条件：`results.csv` 100 行、`full-runtime.jsonl` 100 行、`weights/best.pt` 和 `weights/last.pt` 存在，且 `nonfinite_geometry_observations=0`。当前 launcher 不支持 resume；中断后应保留现场并由负责人决定是否重新 fresh start。

## 6. 当前进展（非新四臂正式主表）

此前已完成一个旧 v2 G 运行：最佳 epoch 92，mAP50-95 `0.29628`；历史 LRS-FDR best 为 `0.29703`，差 `-0.075 pp`。该结果使用同一个现有初始状态，但不是本交接包下的四臂配对正式结果；F 仅完成 44 轮，H/I 尚未运行。它只作为进展参考，不能替代新服务器上的 f/g/h/i fresh runs。

## 7. 结果解释边界

- 主表预先固定使用各臂 best、epoch 100 和末段 96–100 均值。
- 只报告描述性差值：`g-f`、`h-f`、`i-g`、`i-h`、`i-f`，不宣称单 seed 统计显著。
- test-dev 只用于最终审计，不用于选模或调参。
- GitHub 只提交代码、配置、命令、哈希和结果摘要；不提交密码、服务器密钥、完整数据集或 `.pt` 二进制。
