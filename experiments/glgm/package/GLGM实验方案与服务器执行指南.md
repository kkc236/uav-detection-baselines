# GLGM 创新点分析、配对实验方案与服务器执行指南

## 1. 结论先行

本实验要回答的唯一主问题是：在相同 RT-DETR-X、相同数据、相同公共初始权重和相同训练协议下，位于 AIFI 后的 GLGM 是否能提高 VisDrone 小目标检测能力，并且收益足以抵偿其参数量和延迟开销。

不能使用仓库当前同时启用 `GLGM + IRA + FrequencyCM` 的配置与原始模型比较，然后把全部差值归因于 GLGM。正式主对照必须是：

```text
Control: RT-DETR-X + Identity
Method:  RT-DETR-X + GLGM only
```

实验包已实现上述严格配对。Identity 占位使 GLGM 后面的所有层在两组中具有相同层号和 `state_dict` 键名。预检实测结果如下：

| 项目 | Control | GLGM | 差值 |
|---|---:|---:|---:|
| 参数量 | 67,324,002 | 73,963,081 | +6,639,079 |
| 参数增幅 | - | - | +9.8614% |
| 公共张量 | 1,241 | 1,241 | 字节级一致 |
| GLGM 私有张量 | 0 | 27 | 全部位于 `model.16.*` |

GLGM 在 640 输入下接收 P5 的约 `20 x 20 x 384` 特征。对 GLGM 单模块的 THOP 实测为约 5.314 GFLOPs（按一次乘加计 2 FLOPs）。它不是“几乎无开销”的轻量模块，论文必须同时报告精度、参数、延迟、显存，不能只报告 mAP。

## 2. 实际代码中的 GLGM

实际前向过程为：

```text
x
├─ residual -------------------------------------------┐
└─ 3x3 Conv                                            │
   ├─ 3x3 Conv, dilation=1                             │
   └─ 3x3 Conv, dilation=3                             │
        -> Concat -> 3x3 Conv(768->384) -> BN          │
        -> channel attention(avg+max, ECA-style)       │
        -> spatial attention(mean+max, 7x7, CBAM-style)│
        -> + residual -> ReLU --------------------------┘
```

设计意图是保留邻域细节，同时补充较大范围的道路/车流上下文，再由通道和空间门控抑制背景纹理。但论文表述需要比说明书更谨慎：

1. 说明书称 concat 后使用 `1 x 1` 融合，当前代码实际为 `3 x 3`。实验和复杂度必须以代码为准。
2. 当前模块的“全局”信息主要来自前置 AIFI；GLGM 自身是 dilation=1/3 的局部与较大邻域卷积，不是新的全局注意力。
3. 两个分支先直接拼接，再使用共享注意力，并没有显式的二选一分支权重。因此“选择性”目前体现为融合后的通道/空间重标定，而不是显式尺度路由。
4. 虽然存在残差，末端 ReLU 且残差支路没有零初始化缩放，模块初始状态并非严格恒等映射。
5. 参数增加约 9.86%。若精度增益很小或速度显著下降，不能把它判为有效的高效创新。

这些不是阻止实验的问题，而是需要通过消融和诊断回答的问题。当前实现应命名为 `GLGM-v1`，不要在主实验中偷偷换成轻量或零初始化版本。

## 3. 研究假设和指标

### 3.1 预注册假设

- 主假设 H1：GLGM 提高验证集 `mAP50-95`。
- 次假设 H2：GLGM 提高 `Recall`、`AP50` 和 `AP75`，且 AP75 不退化，说明不是只增加宽松匹配框。
- 次假设 H3：拥挤类别（pedestrian、people、car、bicycle）至少多数获得一致收益，而不是均值由单一类别偶然拉高。
- 效率假设 H4：精度收益足以解释约 9.86% 参数增长，FP16 batch=1 延迟增幅可接受。

### 3.2 必报结果

- Precision、Recall、F1、AP50、AP75、mAP50-95。
- 10 个类别的 P/R/F1/AP50/AP75/mAP50-95。
- 参数量、FP16 batch=1 平均/P50/P95 延迟、FPS、峰值显存。
- 每轮训练/验证损失与 mAP 曲线，最佳轮次和最后轮次。
- 训练是否出现 NaN/Inf、OOM、中断、恢复、数据或源码哈希变化。

`best.pt` 和 `last.pt` 都要统一重评估。固定训练预算下以 `last.pt` 为主要结论，`best.pt` 为补充；不能只挑对 GLGM 最有利的不同轮次。

当前脚本尚未宣称提供 AP-tiny/AP-small。若论文必须报告尺寸分层 AP，应在固定 COCO/VisDrone 评估器中预先定义面积区间后，对两组同一次预测统一计算；不能把训练器内部普通 mAP 政名为 AP-small。

## 4. 实验变量控制

| 变量 | 规定 |
|---|---|
| 数据 | VisDrone train 6471 张、val 548 张；两组共享内容哈希 |
| 类别 | 10 类，类别顺序完全相同 |
| 输入 | 640 x 640 |
| 模型 | RT-DETR-X；仅 layer 16 为 Identity/GLGM |
| 初始化 | 推荐同一官方 RT-DETR-X 权重受控迁移；分类头不兼容部分用相同随机初始化 |
| 随机性 | 公共 seed=`s`；GLGM 私有 seed=`10000+s` |
| 优化器 | Ultralytics `optimizer=auto`，两组相同；实际解析结果从 `args.yaml` 留档 |
| batch | 默认 4；若 OOM，两组均改为 2 并从初始状态重跑 |
| 训练 | AMP、deterministic、相同增强、相同 epoch、无早停 |
| 验证 | 同一 val、imgsz、batch、max_det=300 |
| 顺序 | 多种子时交替先后顺序，降低温度/系统负载次序偏差 |

### 4.1 初始化模式

推荐主实验使用官方 `rtdetr-x.pt`：脚本把 stock 层号映射到带 Identity 占位的目标层号，目标层 17 及以后对应 stock 层号减 1。仅迁移名称映射且形状一致的公共张量，COCO 80 类与 VisDrone 10 类不兼容的分类参数保留相同随机初始化。预训练公共参数覆盖率低于 95% 时预检直接失败。

未设置 `BASE_WEIGHTS` 时，脚本会生成严格配对的 scratch 初始化，并在 manifest 标记 `mode=scratch`。scratch 结果不能与 pretrained 结果混合；若从零训练，建议正式预算提高到 300 epochs。

## 5. 实验包结构

```text
glgm-experiment-package/
├── configs/
│   ├── rtdetr-x-glgm-control.yaml
│   └── rtdetr-x-glgm-only.yaml
├── scripts/
│   ├── audit_visdrone.py
│   ├── glgm_experiment.py
│   ├── run_glgm_pair.sh
│   └── setup_glgm_server.sh
├── README.md
└── GLGM实验方案与服务器执行指南.md
```

原仓库的 `train.py` 当前训练的是 `rtdetr-r18.yaml`，不是 RT-DETR-X，也不是 GLGM，不能用于本实验。

## 6. 服务器部署

以下假定：

```bash
REPO_DIR=/root/data/ultralytics-main
PACKAGE_DIR=/root/data/glgm-experiment-package
DATA_YAML=/root/dataset/dataset_visdrone/data.yaml
ENV_ROOT=/root/data/glgm/env
```

先将整个实验包上传到服务器。不要覆盖原仓库中的模型 YAML，也不要修改主仓库后再继续同一实验。

### 6.1 建立环境

```bash
cd "$PACKAGE_DIR"
export REPO_DIR=/root/data/ultralytics-main
export WORK_ROOT=/root/data/glgm/env
bash scripts/setup_glgm_server.sh
```

脚本固定安装 PyTorch 2.5.1 + CUDA 12.1 wheel，以 editable 方式安装当前 Ultralytics 源码，并保存 `pip-freeze.txt`、`nvidia-smi-q.txt` 和关键源码 SHA-256。要求服务器驱动可运行 CUDA 12.1 wheel；若平台镜像已固定其他可用 PyTorch，不要在两组中途更换环境。

### 6.2 准备推荐的基础权重

```bash
mkdir -p /root/data/weights
cd /root/data/weights
/root/data/glgm/env/venv/bin/python -c "from ultralytics import RTDETR; m=RTDETR('rtdetr-x.pt'); print(m.ckpt_path)"
sha256sum /root/data/weights/rtdetr-x.pt
```

自动下载失败时手动上传官方 `rtdetr-x.pt`。必须保留 SHA-256；不要使用之前训练过的 FDR、SCADS、Screen30 或其他改造模型作为基础权重。

### 6.3 单独审计数据

```bash
cd "$PACKAGE_DIR"
/root/data/glgm/env/venv/bin/python scripts/audit_visdrone.py \
  --data /root/dataset/dataset_visdrone/data.yaml \
  --output /root/data/glgm/data-audit.json \
  --expected-train 6471 \
  --expected-val 548 \
  --hash-content
```

审计会检查：图像数量、每张图的标签文件、五列 YOLO 格式、类别 0-9、坐标有限且归一化、每类目标数量，并对图像和标签内容生成清单哈希。失败时先修数据，不要启动训练。

包装脚本还会在每个 arm 训练前及统一评估前重新计算内容哈希并与初始审计比较，防止长时间顺序训练期间数据发生变化。

## 7. 分阶段执行

每个阶段必须使用新的 `WORK_ROOT`。包装脚本依次执行数据审计、配对初始化、Control 训练、GLGM 训练、best/last 重评估、FP16 延迟测试、JSON 对比和 SHA-256 汇总。

### 7.1 F0：预检

完整包装脚本会自动执行，也可先手动运行：

```bash
export PYTHONPATH=/root/data/ultralytics-main
/root/data/glgm/env/venv/bin/python "$PACKAGE_DIR/scripts/glgm_experiment.py" \
  --repo-dir /root/data/ultralytics-main \
  preflight \
  --artifact-dir /root/data/glgm/preflight \
  --public-seed 0 \
  --private-seed 10000 \
  --base-weights /root/data/weights/rtdetr-x.pt \
  --device 0 \
  --full-forward
```

通过标准：

- Control/GLGM 公共参数哈希完全相同。
- GLGM 私有张量只在 `model.16.*`。
- 两个初始检查点保存后重新加载，公共/私有哈希仍一致。
- GLGM 单模块 forward/backward 有有限梯度。
- 640 全模型前向输出有限，无 CUDA OOM。
- 预训练公共参数覆盖率至少 95%。

日志中的 `no model scale passed. Assuming scale='x'` 是当前 Ultralytics 对 RT-DETR `scales.x` 的已知解析提示；预检会用参数量和结构哈希验证实际采用 x 规模。不要为了消除提示修改正式 YAML。

### 7.2 F1：2 epoch 冒烟测试

冒烟测试只证明流程正确，不用于论文精度结论：

```bash
cd "$PACKAGE_DIR"
REPO_DIR=/root/data/ultralytics-main \
DATA_YAML=/root/dataset/dataset_visdrone/data.yaml \
WORK_ROOT=/root/data/glgm/smoke2-seed0 \
PYTHON=/root/data/glgm/env/venv/bin/python \
BASE_WEIGHTS=/root/data/weights/rtdetr-x.pt \
EPOCHS=2 FRACTION=0.05 BATCH=4 WORKERS=4 DEVICE=0 SEED=0 SAVE_PERIOD=1 \
bash scripts/run_glgm_pair.sh
```

通过标准：两组训练和重评估都完成，loss/metrics 全部有限，GPU 持续工作，`comparison-last.json` 可解析。小样本精度高低不作为保留或淘汰 GLGM 的依据。

冒烟仍必须保持 `imgsz=640`。RT-DETR 默认使用 300 个 query，过度缩小到 64 会使多层特征候选总数不足并触发 `topk k out of range`，这属于无效测试条件。

### 7.3 S30：全数据 30 epoch 筛选

必须从配对初始状态重新开始，不能继承 smoke 权重：

```bash
cd "$PACKAGE_DIR"
REPO_DIR=/root/data/ultralytics-main \
DATA_YAML=/root/dataset/dataset_visdrone/data.yaml \
WORK_ROOT=/root/data/glgm/screen30-seed0 \
PYTHON=/root/data/glgm/env/venv/bin/python \
BASE_WEIGHTS=/root/data/weights/rtdetr-x.pt \
EPOCHS=30 FRACTION=1.0 BATCH=4 WORKERS=4 DEVICE=0 SEED=0 SAVE_PERIOD=5 \
bash scripts/run_glgm_pair.sh
```

筛选门槛在查看结果前固定：

- 任一组出现 NaN/Inf、数据变化或恢复不一致：实验无效，先修故障。
- GLGM 的 last mAP50-95 明显下降超过 0.3 个百分点：暂停 Formal100，优先分析结构。
- mAP50-95 接近持平但 Recall/AP75/关键类别多数改善：可进入 Formal100，但标为不确定。
- mAP50-95 提升至少 0.3 个百分点，且 AP75 不下降、延迟增幅可接受：进入 Formal100。

0.3 个百分点是工程筛选阈值，不是统计显著性结论。最终结论依赖多种子和成对统计。

### 7.4 M100：预训练主实验

```bash
cd "$PACKAGE_DIR"
REPO_DIR=/root/data/ultralytics-main \
DATA_YAML=/root/dataset/dataset_visdrone/data.yaml \
WORK_ROOT=/root/data/glgm/formal100-seed0 \
PYTHON=/root/data/glgm/env/venv/bin/python \
BASE_WEIGHTS=/root/data/weights/rtdetr-x.pt \
EPOCHS=100 FRACTION=1.0 BATCH=4 WORKERS=4 DEVICE=0 SEED=0 SAVE_PERIOD=5 \
STRICT_PAIR=1 PARALLEL=0 CONTROL_DEVICE=0 GLGM_DEVICE=0 EVAL_DEVICE=0 \
ARM_ORDER='control glgm' \
bash scripts/run_glgm_pair.sh
```

若 seed0 有正向结果，再运行 seed1、seed2。每个种子都从同一官方基础权重重新生成公共初态，私有随机种子分别为 10001、10002：

```bash
# seed1
WORK_ROOT=/root/data/glgm/formal100-seed1 SEED=1 ARM_ORDER='glgm control' ... bash scripts/run_glgm_pair.sh

# seed2
WORK_ROOT=/root/data/glgm/formal100-seed2 SEED=2 ARM_ORDER='control glgm' ... bash scripts/run_glgm_pair.sh
```

上面省略号代表与 seed0 完全相同的 `REPO_DIR/DATA_YAML/PYTHON/BASE_WEIGHTS/EPOCHS/FRACTION/BATCH/WORKERS/DEVICE/SAVE_PERIOD`，执行时必须完整写出，不能把 `...` 原样输入 shell。

### 7.5 Scratch 补充实验

若需要证明模块不依赖 COCO 预训练，另建独立目录并去掉 `BASE_WEIGHTS`。建议 300 epochs：

```bash
WORK_ROOT=/root/data/glgm/scratch300-seed0 \
EPOCHS=300 FRACTION=1.0 BATCH=4 SEED=0 SAVE_PERIOD=10 \
REPO_DIR=/root/data/ultralytics-main \
DATA_YAML=/root/dataset/dataset_visdrone/data.yaml \
PYTHON=/root/data/glgm/env/venv/bin/python \
bash scripts/run_glgm_pair.sh
```

不要把 scratch300 与 pretrained100 放在同一指标表中做直接差值。

## 8. 中断恢复与监控

监控命令：

```bash
watch -n 30 nvidia-smi
tail -f /root/data/glgm/formal100-seed0/logs/glgm-seed0-e100.log
df -h /root/data
```

正常训练时应看到 GPU 利用率周期性接近满载、显存稳定、`results.csv` 每轮新增一行、`last.pt` 持续更新。默认只保留 best、last 和每 5 轮周期检查点，逐轮指标仍完整保留。

严格配对模式不允许单臂恢复。Ultralytics 的普通 resume 不能证明随机数、数据加载顺序、AMP 状态和未提交梯度与不中断训练完全一致，因此任一 arm 中断、OOM 或异常退出后，必须在新的 `WORK_ROOT` 中让 Control 和 GLGM 都从配对初态重跑。旧目录保留作失败审计，不得重新作为包装脚本输入。

正式实验必须使用默认的 `STRICT_PAIR=1 PARALLEL=0`，并令 `CONTROL_DEVICE` 与 `GLGM_DEVICE` 指向同一块单 GPU。双卡并行会把模型差异与物理 GPU、温度、功耗和并发 I/O 差异混在一起，只允许用于 2 epoch 冒烟或探索：

```bash
STRICT_PAIR=0 PARALLEL=1 CONTROL_DEVICE=0 GLGM_DEVICE=1 EVAL_DEVICE=0 \
bash scripts/run_glgm_pair.sh
```

探索模式只生成 `EXPLORATORY_COMPLETED`；只有同 GPU 顺序执行、训练行数和有限值检查、角色绑定的 best/last 重评估及最终哈希复核全部通过后，才生成正式 `COMPLETED`。

## 9. 结果目录和判定

核心文件：

```text
artifacts/paired_preflight_manifest.json
artifacts/visdrone-audit.json
artifacts/control-last-metrics.json
artifacts/glgm-last-metrics.json
artifacts/control-best-metrics.json
artifacts/glgm-best-metrics.json
artifacts/comparison-last.json
artifacts/comparison-best.json
artifacts/control-best-benchmark.json
artifacts/glgm-best-benchmark.json
artifacts/SHA256SUMS.txt
runs/*/results.csv
runs/*/args.yaml
logs/*.log
```

最终报告至少包含：

| 组别 | P | R | F1 | AP50 | AP75 | mAP50-95 | Params | FP16 ms | FPS | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | | | | | | | 67.324M | | | |
| GLGM | | | | | | | 73.963M | | | |
| Delta | | | | | | | +9.861% | | | |

成功不能只定义为“某个指标为正”。建议满足：

1. 三个种子的 mAP50-95 平均差值为正，至少两个种子同方向；
2. 成对 bootstrap 的 95% CI 不跨 0，或给出 CI 跨 0 的诚实不确定结论；
3. AP75、Recall 和主要小目标类别没有系统性退化；
4. 没有 NaN、协议漂移或选择性汇报；
5. 精度增益能够解释 9.86% 参数和实测延迟增幅。

若只在 seed0 或 best.pt 上出现微小正差，而 last.pt、多种子或关键类别不支持，应结论为“证据不足”，不是“验证成功”。

## 10. 后续消融实验

主实验确认 GLGM-v1 有效后再做消融，避免先扩大搜索空间。每项仍使用相同配对初始化和训练协议。

### 10.1 组件消融

| 编号 | 分支卷积 | 通道注意力 | 空间注意力 | 目的 |
|---|---|---|---|---|
| A0 | 无 | 无 | 无 | Identity 对照 |
| A1 | d1+d3 | 无 | 无 | 测多感受野本身 |
| A2 | d1+d3 | 有 | 无 | 测通道门控增量 |
| A3 | d1+d3 | 无 | 有 | 测空间门控增量 |
| A4 | d1+d3 | 有 | 有 | 完整 GLGM-v1 |

这些变体需要新增独立类或配置，不能在正式主实验中动态修改同一个类。先 30 epoch 筛选，再对有价值变体做完整预算。

### 10.2 融合卷积消融

比较当前 `3 x 3 Conv(768->384)` 与说明书声称的 `1 x 1 Conv(768->384)`。该消融直接回答大部分参数和 FLOPs 是否必要。若 1x1 精度接近而明显更快，它更符合“高效”主张，但应命名为后续版本，不能改写 GLGM-v1 的既有结果。

### 10.3 插入位置消融

优先只比较：AIFI 后 P5、FPN 的 P4、最高分辨率 P3。由于卷积成本按空间面积增长，P4/P3 的计算量约为当前 P5 的 4/16 倍，应先做显存和 2 epoch 预检。若 P3 OOM 或延迟不可接受，应如实停止，不为凑表强行训练。

## 11. 模块专项诊断

精度异常时，按以下顺序定位：

1. 查看 `model.16.*` 梯度范数是否长期接近 0 或爆炸。
2. 记录通道权重和空间权重的均值、标准差、接近 0/1 的比例，判断 sigmoid 是否饱和。
3. 比较 dilation=1 与 dilation=3 分支激活范数和余弦相似度；若长期高度相似，双分支可能冗余。
4. 分类别查看误检/漏检，特别是 pedestrian/people 与复杂背景区域。
5. 对相同图像可视化 Control/GLGM 的真阳性、漏检和新增误检，而不是只展示成功案例。

只有主实验出现可信增益后，注意力热图才是解释材料；热图本身不能证明检测性能提升。

## 12. 已知风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 模块约增加 9.86% 参数 | 可能不符合高效目标 | 强制报告延迟；做 1x1 融合消融 |
| AIFI 已提供全局信息 | GLGM 上下文分支可能重复 | 分支消融和激活相似度诊断 |
| 末端 ReLU 非恒等初始化 | 初期扰动主干特征 | 先观察稳定性；后续单独测试零初始化缩放版 |
| 单个种子噪声 | 误判小幅提升 | seed 0/1/2 和成对置信区间 |
| scratch 100 欠收敛 | 把训练不足误判为模块失败 | pretrained100 主实验或 scratch300 |
| val 同时用于训练监控 | best.pt 存在选择偏差 | last 为主、best 为辅；条件允许增加独立 test |
| 顺序/温度偏差 | 延迟比较不公平 | 多种子交替训练顺序，统一独立 benchmark |

## 13. 最终执行顺序

```text
代码/数据审计
  -> pretrained 配对初始化预检
  -> 2 epoch 小比例冒烟
  -> 全数据 Screen30
  -> 依据预注册门槛决定是否继续
  -> Formal100 seed0
  -> 正向时补 seed1/seed2
  -> 统一 best/last 重评估与 FP16 benchmark
  -> 成对统计、逐类误差和失败审计
  -> 主结论成立后再做组件/融合/位置消融
```

这套流程首先验证当前 GLGM-v1 是否真实有效；若失败，数据、初始化、训练预算和变量隔离均有审计证据，能够进一步判断是实现问题、优化问题，还是模块假设本身不成立。
