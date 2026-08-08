# GLGM 配对实验包

本目录用于验证 `RT-DETR-X + GLGM`，不会修改原始代码仓库。正式对照仅有一处结构差异：

```text
control: RT-DETR-X -> AIFI -> Identity -> FPN/PAN -> Decoder
GLGM:    RT-DETR-X -> AIFI -> GLGM     -> FPN/PAN -> Decoder
```

## 文件

- `configs/rtdetr-x-glgm-control.yaml`：带 Identity 占位层的等价对照组。
- `configs/rtdetr-x-glgm-only.yaml`：只增加 GLGM，不启用 IRA/FrequencyCM。
- `scripts/audit_visdrone.py`：检查完整 VisDrone 数据集并生成内容清单。
- `scripts/glgm_experiment.py`：配对初始化、训练、重评估、比较和延迟测试。
- `scripts/setup_glgm_server.sh`：创建固定 Python 环境。
- `scripts/run_glgm_pair.sh`：顺序完成一组配对实验。
- `GLGM实验方案与服务器执行指南.md`：完整方法、命令、门检和消融方案。

## 最短执行入口

```bash
export REPO_DIR=/root/data/ultralytics-main
export DATA_YAML=/root/dataset/dataset_visdrone/data.yaml
export WORK_ROOT=/root/data/glgm/formal100-seed0
export PYTHON=/root/data/glgm/env/venv/bin/python
export BASE_WEIGHTS=/root/data/weights/rtdetr-x.pt
export EPOCHS=100 BATCH=4 DEVICE=0 SEED=0 FRACTION=1.0 SAVE_PERIOD=5
export STRICT_PAIR=1 PARALLEL=0 CONTROL_DEVICE=0 GLGM_DEVICE=0 EVAL_DEVICE=0

bash scripts/run_glgm_pair.sh
```

运行前先按完整指南完成环境安装、数据审计和 2 epoch 冒烟测试。每个阶段必须使用新的 `WORK_ROOT`。严格配对模式不允许单臂恢复；任一 arm 中断后，必须在新的 `WORK_ROOT` 中让两组从配对初态重跑。

正式实验默认 `STRICT_PAIR=1`，强制 Control 和 GLGM 在同一块 GPU 上顺序训练，并且只有该模式能生成 `COMPLETED`。双卡并行仅可用于冒烟或探索：显式设置 `STRICT_PAIR=0 PARALLEL=1 CONTROL_DEVICE=0 GLGM_DEVICE=1`，结果只生成 `EXPLORATORY_COMPLETED`，不得作为正式严格配对结论。
