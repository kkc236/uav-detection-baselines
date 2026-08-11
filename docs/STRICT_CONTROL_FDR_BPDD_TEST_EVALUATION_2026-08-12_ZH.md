# Control、FDR 与 FDR+BPDD 的同协议 Test 评估

## 1. 结论

在同一张 NVIDIA GeForce RTX 4090、同一份 1610 图 VisDrone test、同一评估器和同一推理参数下，已完成以下三臂 epoch100 EMA 评估：

1. 纯 Ultralytics RT-DETR-L Control；
2. RT-DETR-L + FDR；
3. RT-DETR-L + FDR + BPDD 训练监督。

BPDD 相对 FDR 的 mAP50-95 提升 **+0.366 pp**，AP50 提升 **+0.535 pp**，AP75 提升 **+0.374 pp**，F1 提升 **+0.471 pp**。这证明 BPDD 的 100 轮模型在独立 test 上仍有增量，而不是只在 val 上涨点。

但 BPDD 不是全面无代价提升：Precision 上升 **+1.927 pp**，Recall 下降 **-0.554 pp**；10 个类别中 7 类 mAP 上升、3 类轻微或明显下降。论文应将其描述为提高边界质量和综合 AP 的训练期可靠蒸馏，而不能写成“所有指标、尺度和类别全面提升”。

## 2. 实验身份与证据边界

### 2.1 严格一致部分

- 数据集、类别映射和 test 图片完全相同；
- `imgsz=640`、`batch=8`、`workers=8`；
- `conf=0.001`、`max_det=300`、`NMS=False`；
- `device=0`、`half=False`、`cache=False`；
- 三臂都使用 epoch100 checkpoint 的 EMA；
- 每个模型只进行一次 test 预测；
- FDR 与 BPDD 使用相同 FDR 协议 SHA256：`2545F68302639EEB217EC8B53FDB229681B2DAFC91463C61F6B5CEC9B5486302`；
- FDR 与 BPDD 使用相同公共初始状态 SHA256：`51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D`。

### 2.2 仍需保留的论文限制

FDR checkpoint 来自源码 authority `d97e1eb7`；BPDD checkpoint 来自加入训练期 BPDD 的源码 authority `848f00cb`。两者的初始状态和 FDR 基础协议一致，但此前 fresh BPDD Formal100 配对中的 FDR 臂按用户要求在 epoch24 停止。

因此，本结果是**同 test、同初始状态、同 FDR 基础协议的精确 epoch100 比较**，但不是同一次 fresh paired Formal100 的最终统计证据。论文最终版仍建议重跑 fresh FDR100/BPDD100 配对；在此之前应标记为“Formal100 exact same-test preliminary comparison”。

## 3. Test 数据权威

| 项目 | 数值 |
|---|---:|
| 图片数 | 1610 |
| 标签文件数 | 1610 |
| 目标实例数 | 75,102 |
| 损坏图片 | 0 |
| 测试归档 SHA256 | `45543DB16745616BB203BAD23532623D30099CA8FF38E502A9B996F5F1A58CFB` |

## 4. 三臂核心结果

| 指标 | 纯 Control | FDR | FDR+BPDD | FDR-Control | BPDD-FDR |
|---|---:|---:|---:|---:|---:|
| Precision | 0.417978 | 0.503401 | **0.522672** | +8.542 pp | **+1.927 pp** |
| Recall | 0.373013 | **0.431677** | 0.426137 | +5.866 pp | **-0.554 pp** |
| F1 | 0.394218 | 0.464788 | **0.469494** | +7.057 pp | **+0.471 pp** |
| AP50 | 0.322208 | 0.398340 | **0.403688** | +7.613 pp | **+0.535 pp** |
| AP75 | 0.175503 | 0.228903 | **0.232647** | +5.340 pp | **+0.374 pp** |
| mAP50-95 | 0.177416 | 0.228001 | **0.231659** | +5.058 pp | **+0.366 pp** |

FDR+BPDD 相对纯 Control 的最终增益为：

- Precision：`+10.469 pp`；
- Recall：`+5.312 pp`；
- F1：`+7.528 pp`；
- AP50：`+8.148 pp`；
- AP75：`+5.714 pp`；
- mAP50-95：`+5.424 pp`。

## 5. 分尺度诊断

以下尺度指标是项目自定义诊断：在 640×640 输入空间按预测框和 GT 框各自的边长分桶。它不是标准 COCO area-range AP，预测框尺寸跨桶会影响结果，必须与总体 mAP 分开解释。

| 尺度 | Control | FDR | FDR+BPDD | BPDD-FDR |
|---|---:|---:|---:|---:|
| Tiny | 0.065916 | **0.105919** | 0.105632 | -0.029 pp |
| Small | 0.186776 | 0.247489 | **0.252218** | **+0.473 pp** |
| Medium | 0.281605 | **0.331795** | 0.328986 | -0.281 pp |
| Large | 0.428914 | **0.445307** | 0.404552 | -4.075 pp |

Small 正向，与无人机小目标定位动机一致；Tiny 基本持平。Large 的诊断值明显下降，需要在 fresh pair 中复核，并补充标准 COCO 按 GT area range 的尺度指标。当前不得写“BPDD四尺度全面提升”。

## 6. 逐类别 mAP50-95

| 类别 | Control | FDR | FDR+BPDD | BPDD-FDR |
|---|---:|---:|---:|---:|
| pedestrian | 0.102217 | **0.153998** | 0.153962 | -0.004 pp |
| people | 0.063075 | 0.100983 | **0.104671** | +0.369 pp |
| bicycle | 0.040175 | **0.075494** | 0.074946 | -0.055 pp |
| car | 0.455676 | 0.506470 | **0.510520** | +0.405 pp |
| van | 0.243826 | 0.290208 | **0.299351** | +0.914 pp |
| truck | 0.214840 | 0.283554 | **0.296726** | +1.317 pp |
| tricycle | 0.093960 | **0.145698** | 0.137445 | -0.825 pp |
| awning-tricycle | 0.087723 | 0.121258 | **0.127799** | +0.654 pp |
| bus | 0.358868 | 0.428150 | **0.433344** | +0.519 pp |
| motor | 0.113804 | 0.174197 | **0.177827** | +0.363 pp |

BPDD 相对 FDR 为 7/10 类正向。主要正收益来自 truck、van 和 awning-tricycle；tricycle 是最明显的退化类别，应进入错误案例分析。

## 7. 模块属性与推理开销

BPDD 是**参数为零、仅训练期启用的 Decoder 边界概率分布蒸馏模块**，不是新增推理检测头，也不是后处理算法。

训练阶段，BPDD从当前 Decoder 层之后的未来层构造 GT 一致的 Softmin 混合教师，并使用 detached better-only 权重，仅在教师优于当前层时施加分布蒸馏。它复用最终层 stock Hungarian 匹配，不改变分类、Query 选择和匹配集合。

推理阶段完全移除 BPDD，checkpoint 由普通 FDR 图加载。本次运行已强制验证：

- 模型类型为 `FDRRTDETRDetectionModel`；
- 模型摘要与 FDR 相同：32,334,278 参数、103.7 GFLOPs；
- 没有 BPDD 推理分支；
- 本轮 validator 统计均为约 4.6 ms/图。

因此 BPDD 相对 FDR 的理论推理参数量和 GFLOPs增量均为 0；论文中的严格延迟结论仍应使用同机多次 FP16 benchmark。

## 8. 与既有 val 结果的关系

既有官方 548 图 val 独立评估中，BPDD 相对既有严格 FDR 的 mAP 增益约为 `+0.260 pp`、AP75 增益约为 `+0.557 pp`。本次 1610 图 test 中对应增益为 `+0.366 pp` 和 `+0.374 pp`。

两个划分均显示 mAP/AP75 正向，说明 BPDD 的增益不是只存在于单一 val 指标；但两者均属于跨训练 authority 的比较，不能替代 fresh paired Formal100。

## 9. Checkpoint 权威

| 模型 | epoch100 SHA256 |
|---|---|
| Control | `9C242711F44B7E68B360AF904AB7C44F64505C7136B7E7F90481092AE3308AF7` |
| FDR | `C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2` |
| FDR+BPDD | `E8342C208CE9F5AA8A5F1B341A168170C7D4551E10730E08F05B9794E57CCE4B` |

BPDD publication ledger 为连续 `100/100`，同步状态为 `verified`。

## 10. 证据文件

- `evidence/strict-control-test-d97e1eb7/strict-control-test-eval.json`
- `evidence/strict-control-test-d97e1eb7/strict-fdr-test-eval.json`
- `evidence/strict-control-test-d97e1eb7/strict-fdr-bpdd-test-eval.json`
- `evidence/strict-control-test-d97e1eb7/fdr-bpdd-eval.log`
- `evidence/strict-control-test-d97e1eb7/evaluate_strict_fdr_bpdd_test.py`
- `evidence/strict-control-test-d97e1eb7/test-authority.json`

新增 BPDD test JSON SHA256：`3E0695A81218145AA0936BA34A2162A66C0F54E857239FE0BC8961EF71FB55A6`。

新增 BPDD test 日志 SHA256：`D61B929EAA72F0192BF7FFFB1AC0D1C867B24BBAF8751BE040D7A4B0979AB20F`。
