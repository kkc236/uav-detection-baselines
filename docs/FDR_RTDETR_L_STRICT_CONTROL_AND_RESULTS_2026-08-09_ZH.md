# FDR-RTDETR-L：方法、严格 Control 重跑与完整结果报告

> 更新日期：2026-08-09（Asia/Shanghai）
> 仓库：[`kkc236/uav-detection-baselines`](https://github.com/kkc236/uav-detection-baselines)
> 结论口径：只把可定位到 Git commit、机器可读结果、SHA-256 或 GitHub Release 的内容作为事实；筛选结果、训练端点和统一独立复评严格分开。

## 1. 当前结论

FDR-RTDETR-L 与同一 formal authority 的 stock Control 均已完成 VisDrone 全数据、seed0、100 epoch。统一独立 evaluator 对两臂进行相同预处理和指标计算后，FDR 的 mAP50-95 从 `0.21911` 提高到 `0.28966`，绝对提升 `+0.07055`，即 **`+7.055 pp`**。

主要结论如下：

- Precision、Recall、F1、AP50、AP75、mAP50-95 全部提高；
- Tiny、Small、Medium、Large 四个尺度的 mAP、AP50、AP75 全部提高；
- VisDrone 10 个类别的 mAP、AP50、AP75 全部提高；
- FDR 参数量增加 `329,988`，相对 stock 为 `+1.00524%`；
- 固定 10% 子集的 seed0 paired Screen30 已提前通过冻结 Gate2；
- FDR formal100 与 strict Control formal100 都有公开 epoch100 checkpoint 和 SHA-256；
- 当前严格结论仍限于 seed0，不能写成多 seed 均值或统计显著性结果；
- 统一复评 JSON 仍标为 `preliminary_same_evaluator`，且复评使用的 `last.pt` 与滚动 Release epoch100 文件哈希不同，最终投稿前仍需做 tensor-level identity 封口。

## 2. 研究动机与方法定位

Ultralytics RT-DETR-L 的六层 Decoder 使用四维连续 MLP 直接预测 `(cx, cy, w, h)`。在 VisDrone 大量小目标、密集遮挡和模糊边界场景中，每个坐标只由一个连续点值表达，难以显式建模相邻边界位置之间的竞争关系。

FDR-RTDETR-L 将定位路径改造成“粗框参考 + 四边细粒度分布细化”：

```text
HGNetv2 Backbone / Hybrid Encoder / Query Selection（不变）
                         │
                  6-layer Decoder（不变）
                         │
          ┌──────────────┴──────────────┐
          │                             │
   Classification Heads           FDR Box Path
       （完全保留）                     │
                                  preliminary box
                                        │
                         6 × 4 × 33-bin logits
                                        │
                         cumulative distribution residual
                                        │
                      non-uniform Integral / distance decode
                                        │
                                  refined boxes
```

该实现迁移 D-FINE 的 FDR/FGL 机制，不声称原创 D-FINE 的基础公式，也不是完整 D-FINE 复现。当前工程明确排除了 DDF、GO-LSD、LQE、teacher/student、boundary、trajectory、LPR、OAR、FrequencyCM 等其他路径。

## 3. 相对 stock RT-DETR-L 的网络改动

### 3.1 保持不变的部分

- HGNetv2 Backbone；
- P3/P4/P5 Hybrid Encoder；
- 300 个 Query；
- Query selection；
- 六层 Decoder attention 与 FFN；
- 六层分类头；
- Hungarian matcher 及其 cost；
- VFL、L1、GIoU、encoder auxiliary、decoder auxiliary 和 denoising 主路径；
- Query-class Top-300、`max_det=300`、`NMS=False`；
- 推理输出 schema。

### 3.2 核心结构改动

1. 将六个 `256→256→256→4` 连续框回归 MLP 替换为六个 `256→256→256→132` 分布回归 MLP；
2. `reg_max=32`，每条边有 33 个分箱，四边共 132 个 logits；
3. 新增一个四维 preliminary-box MLP，先建立粗定位参考；
4. 六层对四边分布 residual 进行累计，而不是每层独立重新预测；
5. 使用固定非均匀 Integral，参数为 `reg_scale=4.0`、`up=0.5`；
6. 将四边距离相对 preliminary box 解码回连续边界框。

### 3.3 训练监督改动

Stock loss 完整保留。在此基础上：

- 增加 FGL，权重 `fgl_weight=0.15`；
- 增加 preliminary box 的 L1/GIoU 辅助监督；
- FGL 与 preliminary supervision 复用 stock matcher 已产生的匹配索引；
- 不调用第二个 matcher；
- preliminary reference 与 FGL target 中需要隔离的证据均按冻结实现执行 `detach`。

因此当前获得正式结果的是完整组合：

```text
FDR distribution representation
+ cumulative refinement
+ non-uniform Integral
+ FGL
+ preliminary-box supervision
```

在单变量消融完成前，不能把 `+7.055 pp` 单独归因给其中某一个组件。

## 4. 统一实验协议

### 4.1 环境与数据

| 项目 | Control 与 FDR 的统一设置 |
|---|---|
| 基础模型 | Ultralytics RT-DETR-L |
| Ultralytics | 8.4.90 |
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| Driver | 550.142 |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| CUDA | 12.1 |
| 数据集 | 同一份 VisDrone train/val |
| train / val | 6471 / 548 张 |
| 类别数 | 10 |
| 目标数（统一 val evaluator） | 38,759 |
| 数据集 SHA-256 | `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB` |
| 固定 10% 子集 | 647 张 |
| 10% 子集 SHA-256 | `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0` |

### 4.2 训练配置

| 项目 | 固定值 |
|---|---|
| 初始化 | `pretrained=False`，从零训练 |
| formal epochs | 100 |
| imgsz / batch / workers | 640 / 8 / 8 |
| device | 0，单卡 |
| AMP / fixed AMP scale | True / 128 |
| seed | 0 |
| deterministic / cache | True / False |
| optimizer | MuSGD |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight decay | 0.937 / 0.0005 |
| warmup epochs / momentum / bias lr | 3.0 / 0.8 / 0.0 |
| nbs / cos_lr | 64 / False |
| Query / max_det / NMS | 300 / 300 / False |
| mosaic / close_mosaic | 1.0 / 10 |
| mixup / cutmix / copy_paste | 0.0 / 0.0 / 0.0 |
| scale / translate | 0.5 / 0.1 |
| degrees / shear / perspective | 0.0 / 0.0 / 0.0 |
| flipud / fliplr | 0.0 / 0.5 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 |

两臂还要求同一公共参数初始化、相同样本顺序与增强随机序列、相同验证预处理、类别映射、checkpoint/resume 规则和指标代码。FDR 私有参数使用隔离随机初始化，不改变公共参数的初始化字节。

## 5. 工程与筛选证据

### 5.1 F0–F4

正式训练前完成：

- F0：固定 D-FINE commit 的公式/golden parity；
- F1：neutral behavior 与 stock loss 隔离；
- F2：张量形状、空目标和边界条件；
- F3：真实 VisDrone batch8 CUDA forward/backward、MuSGD、AMP128、checkpoint；
- F4：分布表示覆盖率与饱和诊断。

F0–F4 全部通过后才允许执行 paired Screen30。

### 5.2 固定 10% 子集 paired Screen30

机器证据：[`gate2.json`](../research/fdr/evidence/d97e1eb7/fdr-gate-d97e1eb7/gate2.json)

| Gate2 指标 | Control | FDR | FDR - Control | 结果 |
|---|---:|---:|---:|---|
| final mAP50-95 | 0.00026 | 0.01827 | **+0.01801** | 通过 |
| final AP75 | 0.00003041 | 0.01545827 | **+0.01542786** | 通过 |
| tail-3 mean mAP50-95 | 0.00090333 | 0.01654 | **+0.01563667** | 通过 |
| 连续30轮、finite、manifest、row/paired authority | — | — | — | 全部通过 |

该结果只用于候选筛选。由于 10% 子集 control 地板很低，不能把 `+1.801 pp` 当作 full-data 100 epoch 的正式效应量。

## 6. FDR formal100 完成情况

| 审计项 | FDR formal100 |
|---|---|
| stage / variant / seed | formal / fdr / 0 |
| completed epoch | 100/100 |
| source authority | `d97e1eb7f98414752a1c1f38287697db3f2a0679` |
| D-FINE authority | `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6` |
| formal initial-state SHA-256 | `51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D` |
| protocol SHA-256 | `2545F68302639EEB217EC8B53FDB229681B2DAFC91463C61F6B5CEC9B5486302` |
| run id | `fdr-formal-seed0-b54fbb2dfe73-2545f6830263` |
| epoch100 training-log mAP50-95 | 0.28971 |
| epoch98 / 99 / 100 mAP50-95 | 0.29007 / 0.28996 / 0.28971 |
| epoch100 checkpoint SHA-256 | `C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2` |
| epoch100 checkpoint bytes | 200,024,985 |
| Release | [`fdr-formal-d97e1eb7-live`](https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-d97e1eb7-live) |

FDR 的 100 轮轻量训练证据已经发布到 `training-results` 分支。当前公开 Release 保留 epoch98、99、100 的 checkpoint/manifest，不应写成 1–100 每轮重 checkpoint 均在 Release。

## 7. 严格 stock Control 重跑情况

此前历史 baseline 使用过不同 optimizer/authority，只能作为早期参考。为闭合正式比较，随后按 FDR formal authority fresh 启动 stock Control：

```text
variant: control
stage: formal
seed: 0
epochs: 100
pretrained: false
initial state: 与 formal authority 对齐
dataset / optimizer / batch / augmentation / evaluator: 与 FDR 对齐
resume from FDR or Screen: false
```

当前状态：**Control 已完成 100/100 epoch，不需要再次重跑 seed0 Control。**

| Control checkpoint | SHA-256 | bytes | GitHub 状态 |
|---|---|---:|---|
| epoch98 | `D064FBA559D3D2BC9B7D8003E9FB2998D2C9B5DEF5855860BF4F89820E281A21` | 197,664,716 | 已上传 |
| epoch99 | `ACEDB0AF2D02D5D0D55AED149DD9C72EF09B6D903277AB39718500C4391E0F33` | 197,664,908 | 已上传 |
| epoch100 | `9C242711F44B7E68B360AF904AB7C44F64505C7136B7E7F90481092AE3308AF7` | 197,665,100 | 已上传 |

Release：[`fdr-formal-control-d97e1eb7-live`](https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-control-d97e1eb7-live)

公开 Release 当前保存 Control epoch98–100 的重 checkpoint 和 JSON，以及统一独立复评 JSON。现有公开证据不支持声称 Control 的 epoch1–97 重 checkpoint 全部上传；论文复现需要的是协议、训练日志、最终 checkpoint 与统一复评闭环，而不是伪造缺失的历史重资产。

## 8. 统一独立复评协议

两臂使用同一 evaluator：

| 项目 | 值 |
|---|---|
| val images / targets | 548 / 38,759 |
| imgsz / batch / workers | 640 / 8 / 8 |
| conf | 0.001 |
| IoU thresholds | 0.50:0.05:0.95 |
| max_det / NMS | 300 / False |
| predictions per arm | 164,400 |
| 类别映射 | VisDrone 10 类统一映射 |
| 结果文件 | `fdr-vs-control-strict-final-paper-metrics.json` |
| 结果文件 SHA-256 | `8FFD439C4C48044C0D1937019CE58DDB857CE8FCED64C0082CBC28EDD44333E8` |

F1 是统一 evaluator 当前置信度口径下由 Precision 和 Recall 计算的值，不是遍历完整 PR 曲线得到的最大 F1。

## 9. 正式总体指标

| 指标 | 严格 Control | FDR | 绝对差值 | 百分点变化 |
|---|---:|---:|---:|---:|
| Precision | 0.46761 | 0.56911 | +0.10150 | **+10.150 pp** |
| Recall | 0.41731 | 0.49278 | +0.07546 | **+7.546 pp** |
| F1 | 0.43657 | 0.52484 | +0.08827 | **+8.827 pp** |
| AP50 | 0.38663 | 0.48468 | +0.09805 | **+9.805 pp** |
| AP75 | 0.21302 | 0.29253 | +0.07951 | **+7.951 pp** |
| mAP50-95 | 0.21911 | 0.28966 | +0.07055 | **+7.055 pp** |

这个结果同时改善宽松 IoU 下的检出能力和严格 IoU 下的定位质量，不是单纯的 Precision/Recall 交换。

## 10. 分尺度结果

尺寸定义：Tiny `<256 px²`；Small `[256,1024) px²`；Medium `[1024,9216) px²`；Large `≥9216 px²`。

| 尺度 | GT | Control mAP | FDR mAP | ΔmAP | Control AP50 | FDR AP50 | ΔAP50 | Control AP75 | FDR AP75 | ΔAP75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 20,861 | 0.08684 | 0.14480 | **+5.795 pp** | 0.20242 | 0.30322 | +10.080 pp | 0.06019 | 0.11485 | +5.466 pp |
| Small | 12,420 | 0.21784 | 0.28998 | **+7.214 pp** | 0.37701 | 0.46100 | +8.399 pp | 0.22544 | 0.31944 | +9.401 pp |
| Medium | 5,348 | 0.32499 | 0.39630 | **+7.130 pp** | 0.46969 | 0.54957 | +7.987 pp | 0.35851 | 0.44447 | +8.596 pp |
| Large | 130 | 0.31822 | 0.38608 | **+6.786 pp** | 0.39276 | 0.45976 | +6.700 pp | 0.35713 | 0.38931 | +3.218 pp |

四个尺度全部正向。Large 只有 130 个 GT，方差会高于其他尺度，不能据此单独做过强机制结论。

## 11. 十类别 mAP / AP50 / AP75

下表中的变化均为 FDR 减去严格 Control，单位为百分点。

| 类别 | GT | Control mAP | FDR mAP | ΔmAP pp | Control AP50 | FDR AP50 | ΔAP50 pp | Control AP75 | FDR AP75 | ΔAP75 pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pedestrian | 8,844 | 0.17638 | 0.27277 | **+9.638** | 0.41722 | 0.56665 | +14.943 | 0.11627 | 0.22404 | +10.777 |
| people | 5,125 | 0.13004 | 0.20888 | **+7.884** | 0.33741 | 0.49829 | +16.088 | 0.06693 | 0.13556 | +6.863 |
| bicycle | 1,287 | 0.05132 | 0.11044 | **+5.912** | 0.12410 | 0.23732 | +11.322 | 0.02806 | 0.08441 | +5.635 |
| car | 14,064 | 0.54785 | 0.60930 | **+6.145** | 0.80905 | 0.85692 | +4.787 | 0.60952 | 0.68546 | +7.593 |
| van | 1,975 | 0.31458 | 0.37972 | **+6.513** | 0.46538 | 0.53207 | +6.669 | 0.35274 | 0.43397 | +8.123 |
| truck | 750 | 0.21277 | 0.26403 | **+5.125** | 0.34079 | 0.39085 | +5.006 | 0.22671 | 0.28742 | +6.071 |
| tricycle | 1,045 | 0.12815 | 0.20574 | **+7.759** | 0.24221 | 0.36624 | +12.404 | 0.12539 | 0.19978 | +7.439 |
| awning-tricycle | 532 | 0.08782 | 0.11370 | **+2.588** | 0.14871 | 0.18469 | +3.597 | 0.09283 | 0.12190 | +2.907 |
| bus | 251 | 0.34460 | 0.43801 | **+9.341** | 0.51086 | 0.60786 | +9.700 | 0.38279 | 0.50753 | +12.474 |
| motor | 4,886 | 0.19754 | 0.29401 | **+9.647** | 0.47060 | 0.60595 | +13.535 | 0.12896 | 0.24518 | +11.622 |

10 类 mAP、AP50 和 AP75 均严格正向。`awning-tricycle` 的增益最小，但没有出现类别退化。

## 12. 参数、GFLOPs 与 checkpoint 开销

| 指标 | Stock Control | FDR | 增量 | 相对变化 |
|---|---:|---:|---:|---:|
| 参数量 | 32,826,626 | 33,156,614 | +329,988 | **+1.00524%** |
| GFLOPs | 108.0318976 | 108.2291200 | +0.1972224 | **+0.18256%** |
| epoch100 checkpoint | 197,665,100 bytes | 200,024,985 bytes | +2,359,885 bytes | +1.194% |

不能写“参数增加小于1%”，正确值是约 `+1.005%`。最终统一 batch1 latency/FPS/P90/显存表尚未形成与本主表同等级的冻结证据，因此本报告不提前填写推测值。

## 13. YAML 可插拔和消融准备

正式 FDR 已从隐式 head 替换收口为 YAML 显式 `FDRRTDETRDecoder`。主要配置：

- `configs/rtdetr-l-fdr.yaml`：完整 FDR；
- `configs/rtdetr-l-fdr-no-fgl.yaml`：关闭 FGL；
- `configs/rtdetr-l-fdr-no-prebox-loss.yaml`：关闭 preliminary-box 辅助损失；
- `configs/rtdetr-l-fdr-no-cumulative.yaml`：关闭累计细化；
- `configs/rtdetr-l-fdr-no-prebox.yaml`：关闭 preliminary-box 结构。

五份 YAML 均完成 formal checkpoint 严格加载验证：950 个张量，`0 missing / 0 unexpected`，并完成 legacy checkpoint/resume-step 验证。配置可加载不等于对应消融已经训练；论文消融必须从相同公共初始状态独立训练。

## 14. GitHub 发布状态

### 已上传

- FDR Screen30 两臂结果、Gate2 和机器证据；
- FDR formal100 的 100 轮轻量训练记录；
- FDR formal epoch98–100 checkpoint 和 manifest；
- strict Control formal epoch98–100 checkpoint 和 manifest；
- strict Control/FDR 统一复评 JSON 及 SHA-256；
- FDR YAML、四个单变量消融 YAML；
- checkpoint strict-load、legacy reconstruction 与 resume-step 证据；
- 本报告。

### 不能夸大的部分

- 当前 Release 不是两臂 epoch1–100 全部重 checkpoint 的永久存档；
- seed1/seed2 尚未形成严格配对正式结果；
- 四个消融 YAML 尚无完整训练结论；
- 最终 latency/FPS 审计尚未冻结；
- 统一复评 actual `last.pt` 与滚动 epoch100 asset 的 tensor identity 尚未封口。

## 15. Artifact identity 审计边界

统一复评记录的实际 checkpoint：

| Arm | 统一复评 `last.pt` SHA-256 | Release epoch100 SHA-256 |
|---|---|---|
| Control | `7CCDAE649426505F157CB78AEBAA1981CDABB28B483338743983D2A264B50E4F` | `9C242711F44B7E68B360AF904AB7C44F64505C7136B7E7F90481092AE3308AF7` |
| FDR | `2C1ADE3FD9DC59B8FE5B816B8B95183037E47839186BB8F68C93774B0B60451A` | `C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2` |

文件哈希不同可能来自保存时机、EMA/model字段或序列化差异，但没有 tensor-by-tensor equality 报告前不能假定权重完全相等。正确结论是：

- 两臂训练和统一 same-evaluator 结果均真实存在；
- `+7.055 pp` 是当前最强的 seed0 严格同评估器结果；
- 最终投稿包仍应上传复评实际使用的 exact checkpoint，或发布 model/EMA tensor equality JSON；
- 在完成该项前保留结果文件的 `preliminary_same_evaluator` 状态说明。

## 16. 当前可写与不可写的论文结论

### 可以写

> 在统一 Ultralytics RT-DETR-L、VisDrone、seed0 和从零训练协议下，FDR-RTDETR-L 将 mAP50-95 从 21.911% 提升至 28.966%，提高 7.055 个百分点；AP50 和 AP75 分别提高 9.805 和 7.951 个百分点。四个尺度以及十个类别的 mAP、AP50、AP75 均获得正增益。

同时必须说明：FDR/FGL 基础机制来自 D-FINE，本文贡献是面向 Ultralytics RT-DETR-L/VisDrone 的隔离迁移、结构化适配、YAML 可声明集成和统一协议验证。

### 暂时不能写

- 多 seed 均值、方差或统计显著性；
- “FDR 基础公式由本文首次提出”；
- “参数增加低于1%”；
- “延迟增加低于3%”；
- “每个 FDR 子组件单独有效”；
- “1–100轮所有重 checkpoint 均已公开”；
- “复评 checkpoint identity 已完全封口”。

## 17. 后续工作

1. 上传统一复评实际使用的两个 exact `last.pt`，或输出 tensor-level equality 报告；
2. 冻结 batch1 latency、P90、FPS、峰值显存和同硬件测量脚本；
3. 按相同 initial-state 运行四个单变量消融；
4. 若论文需要统计结论，再补 seed1/seed2 的成对 Control/FDR，而不是只补方法臂；
5. 新增小模块必须以 FDR 为直接 Control，使用独立 YAML 层和 fresh paired protocol，不能拿历史跨-authority baseline 拼表。

## 18. 复现与证据入口

- [FDR 方法实现与当前验证](../research/fdr/FDR_RTDETR_METHOD_AND_CURRENT_VALIDATION_ZH.md)
- [FDR YAML 声明式模块说明](FDR_YAML_DECLARATIVE_MODULE.md)
- [FDR 当前交接](CURRENT_FDR_SADED_HANDOFF_2026-08-05_ZH.md)
- [FDR Screen30 Gate2](../research/fdr/evidence/d97e1eb7/fdr-gate-d97e1eb7/gate2.json)
- [FDR formal100 训练结果](https://github.com/kkc236/uav-detection-baselines/tree/training-results/results/fdr-formal-d97e1eb7-seed0-fdr)
- [FDR formal checkpoint Release](https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-d97e1eb7-live)
- [strict Control 与统一复评 Release](https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-control-d97e1eb7-live)
- [FDR YAML 与兼容性 Release](https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-yaml-declarative-v1)

## 19. 最终审计意见

FDR 已不再是“有希望的早期方案”，而是当前项目中证据最完整、效应量最大的正式模块：seed0 全数据 100 epoch、同 authority stock Control、统一独立复评、四尺度和十类别结果均已形成。严格 Control 也已经完成，不应继续把“重跑 Control”列为未完成训练任务。

当前主要缺口是证据封口和论文完整性，而不是重新证明 seed0 是否上涨：需要完成 exact-checkpoint tensor identity、效率表、单变量消融和必要的多 seed 统计。任何后续小模块都应直接以该 FDR 作为强 Control，并保持网络路径、实验 authority 与证据发布彼此隔离。
