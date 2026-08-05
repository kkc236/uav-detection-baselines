# RT-DETR-L / VisDrone 当前成果与下一步执行交接

> 交接日期：2026-08-05（Asia/Shanghai）<br>
> 交接性质：面向下一位研究与工程负责人，可直接据此恢复代码、下载权重、复核证据并继续实验<br>
> 主仓库：<https://github.com/kkc236/uav-detection-baselines><br>
> 当前主线：FDR-only 定位回归模块<br>
> 已取得正向结果的独立推理方案：SADED-SM<br>
> 安全说明：本文不包含服务器密码、GitHub Token 或其他凭据

## 0. 一页结论

当前最成熟的**网络结构创新点**是 FDR-RTDETR-L：在保持 Backbone、Hybrid Encoder、Query、Decoder 注意力/FFN、分类头、匹配和后处理不变的前提下，将 Decoder 的四维连续框回归路径替换为 preliminary box、六层 132 维四边分布回归、跨层累计分布细化及 FGL 监督。它已经完成声明式 YAML、四份单变量消融配置、正式 epoch-100 权重兼容、真实 resume-step 和 30-epoch 配对筛选。

当前最强的**可运行推理正收益方案**是 SADED-SM：同一个 RT-DETR-L checkpoint 对一张图执行一次全图和四次局部 tile 推理，再以固定尺度规则保护非 tiny 全图预测，只向 tiny 路径引入局部结果。其 seed0 VisDrone development-val 结果为 mAP50-95 `+2.5849 pp`、AP-tiny `+3.9194 pp`、tiny recall `+10.1778 pp`、AP75 `+2.0237 pp`，但每图需要 5 次 detector forward，不能写成零开销方法。

当前最重要的未完成项不是继续堆结构，而是：

1. 用原 formal authority 重跑严格 matched stock control 100 epoch；
2. 对 FDR/control 使用同一独立 evaluator 给出论文完整指标；
3. 训练四个 FDR 单变量消融，而不是只对同一完整权重切换 YAML；
4. 冻结参数、GFLOPs、端到端 latency/FPS、显存与逐类别指标；
5. 决定 SADED-SM 是作为独立推理创新点，还是仅作 tiny-object 增强实验。

截至本交接，不能声称“FDR 全数据 100 epoch 已严格超过 matched baseline”，因为 FDR 的正式权重已经存在，但与其完全同 authority 的 formal control 对照尚未形成可审计最终报告。

## 1. GitHub、分支、提交与 Release 地图

### 1.1 FDR 当前工程主线

| 内容 | 地址或版本 |
|---|---|
| 仓库 | <https://github.com/kkc236/uav-detection-baselines> |
| 当前分支 | <https://github.com/kkc236/uav-detection-baselines/tree/codex/fdr-yaml-module> |
| YAML 模块验证代码提交 | [`dc9a0529f2e69a2f3a1477e84cbae0b15de29a17`](https://github.com/kkc236/uav-detection-baselines/commit/dc9a0529f2e69a2f3a1477e84cbae0b15de29a17) |
| 正式模型/训练 authority | [`d97e1eb7f98414752a1c1f38287697db3f2a0679`](https://github.com/kkc236/uav-detection-baselines/commit/d97e1eb7f98414752a1c1f38287697db3f2a0679) |
| D-FINE 机制 authority | `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6` |
| FDR YAML Release | <https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-yaml-declarative-v1> |
| FDR formal checkpoint Release | <https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-d97e1eb7-live> |
| FDR 30-epoch screen Release | <https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-screen-d97e1eb7-live> |
| 完整方法说明 | [`docs/FDR_YAML_DECLARATIVE_MODULE.md`](FDR_YAML_DECLARATIVE_MODULE.md) |

必须区分两个提交的职责：

- `d97e1eb7...` 是正式 FDR 训练的模型、协议和运行语义 authority；严格 matched control 应以该提交和原始 protocol/initial-state 为准。
- `dc9a0529...` 是训练完成后的 YAML 声明式模块与兼容性收口；它可以严格加载正式 FDR checkpoint，并提供四份消融 YAML，但不是当时 formal100 训练进程的源码提交。

### 1.2 SADED-SM 独立历史

SADED-SM 位于另一条 Git 历史，**当前 `codex/fdr-yaml-module` 工作树不包含其源码**。不要因为仓库相同就假定 `src/saded.py` 已经在当前分支。

| 内容 | 地址或版本 |
|---|---|
| 正式结果提交 | [`e3f651fa289888679a3e228d4082f5cff09342a1`](https://github.com/kkc236/uav-detection-baselines/commit/e3f651fa289888679a3e228d4082f5cff09342a1) |
| 后继源码提交 | [`7f1f1e11f0c0c6d373e6172a7511ee645b4421cd`](https://github.com/kkc236/uav-detection-baselines/commit/7f1f1e11f0c0c6d373e6172a7511ee645b4421cd) |
| 结果交接 | <https://github.com/kkc236/uav-detection-baselines/blob/e3f651fa289888679a3e228d4082f5cff09342a1/docs/final-saded-single-model-result-handoff.md> |
| 机器证据 | <https://github.com/kkc236/uav-detection-baselines/blob/e3f651fa289888679a3e228d4082f5cff09342a1/docs/evidence/saded_single_model_final/formal_adjudication.json> |
| 核心源码 | <https://github.com/kkc236/uav-detection-baselines/blob/7f1f1e11f0c0c6d373e6172a7511ee645b4421cd/src/saded.py> |
| 路由 CLI | <https://github.com/kkc236/uav-detection-baselines/blob/7f1f1e11f0c0c6d373e6172a7511ee645b4421cd/scripts/route_saded.py> |
| 评估 CLI | <https://github.com/kkc236/uav-detection-baselines/blob/7f1f1e11f0c0c6d373e6172a7511ee645b4421cd/scripts/evaluate_saded.py> |

推荐使用独立 worktree 检出 SADED-SM，暂时不要把两个历史直接 merge：

```powershell
git clone https://github.com/kkc236/uav-detection-baselines.git
Set-Location uav-detection-baselines
git worktree add ..\saded-sm-source 7f1f1e11f0c0c6d373e6172a7511ee645b4421cd
git worktree add ..\fdr-formal-authority d97e1eb7f98414752a1c1f38287697db3f2a0679
git switch codex/fdr-yaml-module
```

## 2. FDR-RTDETR-L：方法、代码和消融边界

### 2.1 方法定位

论文中的准确定位应为：

> 将 commit-pinned D-FINE FDR/FGL 机制结构化适配到 Ultralytics RT-DETR-L 的 Decoder 定位路径，并在 VisDrone 小目标检测任务上完成隔离集成、YAML 声明、checkpoint 兼容和统一协议验证。

不能声称原创 D-FINE 的 FDR、FGL、Integral 或非均匀 weighting 公式。本项目的贡献是面向 Ultralytics RT-DETR-L 的结构迁移、模块化实现、公平初始化、损失隔离、可消融性与实验验证。

### 2.2 网络改动

保持不变：

- HGNetv2 Backbone；
- P3/P4/P5 与 Hybrid Encoder；
- 300 个 Query 与 Query selection；
- 六层 Decoder 的 deformable attention、FFN、LayerNorm；
- 分类分支、分类 logits；
- Hungarian matcher 及其 cost；
- Top-300、`max_det=300`、`NMS=False`；
- 推理输出合同 `[batch, 300, 6]`。

只改 Decoder box path：

```text
Decoder hidden state
        |
        +--> preliminary 4-D box
        |
        +--> 6 x 132-D distribution heads
                   |
             cumulative logits
                   |
         non-uniform Integral decode
                   |
              refined boxes
```

关键固定参数：

| 参数 | 值 | 含义 |
|---|---:|---|
| `reg_max` | 32 | 每条边 33 个 bin，四边共 132 维 |
| `reg_scale` | 4.0 | 非均匀 Integral 的尺度 |
| `up` | 0.5 | 非均匀密度参数 |
| `cumulative` | `true` | 六层分布 residual 跨层累计 |
| `preliminary_box` | `true` | 使用粗框建立分布解码参考 |
| `fgl_weight` | 0.15 | FGL 损失权重 |
| `supervise_pre_boxes` | `true` | 开启 pre-box L1/GIoU 辅助监督 |
| `private_seed` | 10000 | 私有 FDR 参数初始化 authority |

损失保持 stock VFL/L1/GIoU 和原匹配索引，额外加入 FGL 与 preliminary-box L1/GIoU。FGL 不进行第二次 matcher，不修改 Query 分配。

### 2.3 文件入口

| 功能 | 文件 |
|---|---|
| 完整模型 YAML | `configs/rtdetr-l-fdr.yaml` |
| 无 FGL 消融 | `configs/rtdetr-l-fdr-no-fgl.yaml` |
| 无 pre-box loss 消融 | `configs/rtdetr-l-fdr-no-prebox-loss.yaml` |
| 无累计分布消融 | `configs/rtdetr-l-fdr-no-cumulative.yaml` |
| 无 preliminary-box reference 消融 | `configs/rtdetr-l-fdr-no-prebox.yaml` |
| Decoder/FDR head | `src/fdr_head.py` |
| Integral 与 box 数学 | `src/fdr_math.py` |
| FGL 与 loss 隔离 | `src/fdr_loss.py` |
| Model/Trainer 集成 | `src/rtdetr_fdr.py` |
| 冻结实验协议 | `src/fdr_protocol.py` |
| 训练入口 | `scripts/train_rtdetr_fdr.py` |
| 权重/YAML 验证 | `scripts/verify_fdr_yaml_checkpoint.py` |
| 旧 checkpoint resume-step 验证 | `scripts/verify_fdr_legacy_resume_step.py` |

Stock baseline 继续使用 Ultralytics 原生 `rtdetr-l.yaml`。不要尝试用同一个 132 维 FDR head 的布尔开关伪装 stock 4 维 box head；两者的参数合同不同。

### 2.4 已完成工程验证

当前冻结结果：

- 专项测试：`167 passed, 3 skipped`；
- 五份 YAML 均可构建 `FDRRTDETRDecoder`；
- 正式 epoch-100 EMA checkpoint 对五份 YAML 均 strict load；
- 每份配置均为 950 tensors、missing/unexpected `0/0`；
- 每份配置有限输出 `[1, 300, 6]`；
- legacy checkpoint 自动将内嵌旧 head 名规范化为 `FDRRTDETRDecoder`；
- `950/950` 权重恢复；
- MuSGD 8 个 param group、581 个 optimizer state 恢复；
- AMP scale 128 保持；
- EMA updates `10556 -> 10557`；
- 已完成一次有限的 `128x128` forward/backward/MuSGD/EMA resume step。

机器证据：

- `research/fdr/evidence/d97e1eb7/yaml-module-final/checkpoint-compatibility-all-configs.json`
- `research/fdr/evidence/d97e1eb7/yaml-module-final/legacy-resume-step.json`
- `research/fdr/evidence/d97e1eb7/yaml-module-final/README.md`

### 2.5 已完成科学筛选

固定 10% 子集、seed0、30-epoch cutoff 的 paired screen 已通过：

| 指标 | Control epoch30 | FDR epoch30 | FDR - Control |
|---|---:|---:|---:|
| Precision | 0.00662 | 0.07229 | +0.06567 |
| Recall | 0.02501 | 0.13717 | +0.11216 |
| mAP50-95 | 0.00026 | 0.01827 | **+0.01801** |
| AP75 | 0.0000304090 | 0.0154582695 | **+0.0154278605** |

尾三轮差值：

- mean mAP50-95：`+0.0156366667`；
- mean AP75：`+0.0139583318`。

Gate2 的三个预注册条件均严格正向，工程检查也全部通过。证据文件：

`research/fdr/evidence/d97e1eb7/fdr-gate-d97e1eb7/gate2.json`

但 control 在该子集上的绝对性能异常低，所以 `+1.801 pp` 只能叫“30-epoch 子集筛选增益”，不能当作正式 100-epoch 论文提升。

### 2.6 正式 FDR 100-epoch checkpoint

| 项目 | 值 |
|---|---|
| 文件 | `fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt` |
| SHA256 | `c2f638744508adfe7b6c4a1ef3e08c503273f628062e4650ad59ffff4c6588c2` |
| 大小 | 200,024,985 bytes |
| Release | <https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-d97e1eb7-live> |

checkpoint 内部 epoch-100 端点元数据的初步参考值：

| 指标 | 值 |
|---|---:|
| Precision | 0.56778 |
| Recall | 0.49350 |
| mAP50 | 0.48480 |
| mAP50-95 | 0.28971 |
| best fitness | 0.29078 |

尾三轮 mAP50-95 为 `0.29007 / 0.28996 / 0.28971`。这些值是 checkpoint/run 端点参考，不是独立 evaluator 的最终论文表，也不是相对严格 matched control 的增益。

### 2.7 当前开销数据

| 指标 | Stock | FDR | 增量 | 相对增幅 |
|---|---:|---:|---:|---:|
| Parameters | 32,826,626 | 33,156,614 | +329,988 | **+1.00524%** |
| GFLOPs | 108.0318976 | 108.2291200 | +0.1972224 | **+0.18256%** |

参数增幅略高于 1%，因此不能写“参数量增幅 `<1%`”。最终端到端 latency/FPS 尚未冻结，也不能提前写“延迟增幅 `<3%`”。

## 3. SADED-SM：方法、结果与边界

### 3.1 方法概述

SADED-SM 是 **Single-Model Scale-Aware Multi-View Routing**。它不是训练另一个 detector，而是复用同一个 100-epoch RT-DETR-L checkpoint：

1. 原图 640 推理一次；
2. 四个固定局部 tile 分别推理，共四次；
3. 将 tile box 映射回原图坐标；
4. 网络帧有效尺寸大于 16 px 的 full-view prediction 全部保护；
5. 同类 full/local 候选仅在 IoU 严格大于 0.5 时匹配；
6. tiny 匹配项按 `alpha(s)=sigmoid(ln(9)/8*(16-s))` 调整分数，并采用 local box；
7. unmatched local 仅在尺寸不大于 16、来源完整且不是 protected 同类框碎片时进入；
8. 以固定 score/source/query/index 顺序截取 Top-300；
9. evaluator 只接收一个统一 prediction JSON。

固定推理条件为 `conf=0.001`、每视图 `max_det=300`、`NMS=False`。该方案不利用 GT 做路由。

### 3.2 seed0 development-val 结果

| 指标 | Arm A | SADED-SM | 差值 |
|---|---:|---:|---:|
| AP-tiny-SBR | 0.0710571443 | 0.1102511667 | **+0.0391940224** |
| mAP50-95 | 0.1806213966 | 0.2064703073 | **+0.0258489108** |
| Tiny recall | 0.5537479711 | 0.6555260440 | **+0.1017780729** |
| AP75 | 0.1666655849 | 0.1869023302 | **+0.0202367453** |
| AP-large-SBR | 0.1458467938 | 0.1439375721 | **-0.0019092217** |

决策为 `SADED_SINGLE_SEED_GO`，五个冻结 Gate 均通过。服务器测试记录为 `829 passed`，路由 prediction JSON SHA256 为：

`4c8e4998f0cbdbbc5963fecbf05ac4dc26d56db6b95d71a076fd129a66aa740e`

使用的 detector checkpoint SHA256：

`54ce60289dd34c6750b8ba5f7516eefcf3afef6c174c6e4f3b1ef810c883099b`

### 3.3 论文表述边界

可以写：单模型、多视图、训练自由的尺度感知路由在 seed0 development-val 上取得上述正收益。

不能写：

- 三 seed 均值；
- 独立 test-dev 确认；
- 零开销；
- 五模型 ensemble；
- 与 FDR 增益天然可相加。

它是一个 checkpoint 的五视图推理，但每张图确实执行 5 次 detector forward。把 SADED-SM 应用于 FDR checkpoint 是新的组合实验，必须重新生成缓存、独立评估并测开销，不能直接把两组 delta 相加。

## 4. 已冻结的失败路线与仍有价值的上限证据

这些路线不应在没有新信息的情况下原样重跑：

| 路线 | 冻结结论 | 接手注意事项 |
|---|---|---|
| LPR v1 | seed0、seed1 均未超过 control | 定位损失变化没有转化为 AP；seed2 已按用户指令取消 |
| LPR-G v2 | `scientific_failed` | 有效 gate 均值约 `4.92e-5`，乘法 gate 坍缩，不继续长训 |
| IBER-BE B3 | matched IoU `+0.008041`，但 Gate-1 失败 | `edge_over_b0=false`、`edge_over_b1=false`，tiny/small direction 未过线；probe IoU 不是 detector mAP |
| P2 boundary oracle | `scientific_failed` | context+P2 低于 context-only，未证明 P2 的独立方向信息 |
| sparse OAR Top-K D0 | `scientific_failed` | 没有 K 恢复 90% oracle 增益，不扩网格、不降门槛 |
| learnable C0/C1/Q quality probe | `scientific_failed` | perfect oracle 很强，但学习器不能稳定复现其排序能力 |

Boundary 方法与证据说明见：

[`docs/IBER_BE_BOUNDARY_EVIDENCE_METHOD_ZH.md`](IBER_BE_BOUNDARY_EVIDENCE_METHOD_ZH.md)

最强的不可部署上限是 class-conditional quality-reordering oracle：

- mAP `0.2416484499 -> 0.3973619056`，提升 `+0.1557134557`；
- AP75 `0.2391637546 -> 0.3883675964`。

它使用 `quality[q,c] = max IoU(query_box, same-class GT)` 调整分数，推理时需要真实标注，因此只能证明“排序质量存在巨大理论空间”，不能部署，也不能作为方法结果。设计记录：

`docs/superpowers/specs/2026-08-04-objective-aligned-reranker-design.md`

## 5. 冻结环境、数据与统一协议

### 5.1 环境和数据

| 项目 | 冻结值 |
|---|---|
| 基础模型 | Ultralytics RT-DETR-L |
| Ultralytics | 8.4.90 |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| Driver | 550.142 |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| CUDA | 12.1 |
| 数据集 | 同一份 VisDrone train/val |
| Train / Val | 6471 / 548 |
| 类别数 | 10 |
| 数据集 SHA256 | `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB` |
| 固定 10% 子集 | 647 张 |
| 子集 SHA256 | `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0` |

### 5.2 训练与后处理

| 项目 | 冻结值 |
|---|---|
| 初始化 | `pretrained=False`，从零训练 |
| Formal | 全数据、seed0、fresh 100 epoch |
| Screen | 固定 647 张、共同 50-epoch schedule、第 30 轮 cutoff |
| `imgsz / batch / workers` | `640 / 8 / 8` |
| `device` | 0，单卡 |
| AMP | True，固定 scale 128 |
| `deterministic / cache` | `True / False` |
| optimizer | MuSGD |
| `lr0 / lrf` | `0.01 / 0.01` |
| momentum | 0.937 |
| weight decay | 0.0005 |
| warmup | epochs 3.0，momentum 0.8，bias LR 0.0 |
| `nbs / cos_lr` | `64 / False` |
| Query / max_det | `300 / 300` |
| NMS | False |
| 增强 | mosaic 1.0，close_mosaic 10，mixup 0，scale 0.5，translate 0.1 |
| 几何增强 | degrees/shear/perspective/flipud 均 0，fliplr 0.5 |
| HSV | 0.015 / 0.7 / 0.4 |
| 其他 | cutmix 0，copy_paste 0 |

FDR 的当前 authority 是 `src/fdr_protocol.py`，其 protocol SHA256 为：

`2545F68302639EEB217EC8B53FDB229681B2DAFC91463C61F6B5CEC9B5486302`

公共 initial-state SHA256：

`51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D`

注意：仓库中的 `docs/EXPERIMENT_CONTROL_PROTOCOL.md` 是较早的其他创新线协议，包含 `optimizer=auto`、`warmup_bias_lr=0.1` 等历史值，**不能覆盖当前 FDR authority**。

## 6. 下载包和权重

### 6.1 GitHub 下载

完整 FDR YAML：

<https://github.com/kkc236/uav-detection-baselines/releases/download/fdr-yaml-declarative-v1/rtdetr-l-fdr.yaml>

完整 YAML 与四份消融：

<https://github.com/kkc236/uav-detection-baselines/releases/download/fdr-yaml-declarative-v1/FDR-YAMLs-v1.zip>

源码 overlay：

<https://github.com/kkc236/uav-detection-baselines/releases/download/fdr-yaml-declarative-v1/FDR-YAML-Declarative-v1-overlay.zip>

离线 Git bundle：

<https://github.com/kkc236/uav-detection-baselines/releases/download/fdr-yaml-declarative-v1/FDR-YAML-Declarative-v1.bundle>

SHA256 清单：

<https://github.com/kkc236/uav-detection-baselines/releases/download/fdr-yaml-declarative-v1/FDR-YAML-Declarative-v1-SHA256SUMS.txt>

正式 epoch-100 权重：

<https://github.com/kkc236/uav-detection-baselines/releases/download/fdr-formal-d97e1eb7-live/fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt>

### 6.2 当前 Windows 本地副本

```text
C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\downloads\rtdetr-l-fdr.yaml
C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\downloads\FDR-YAMLs-v1.zip
C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\downloads\FDR-YAML-Declarative-v1-overlay.zip
C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\downloads\FDR-YAML-Declarative-v1.bundle
C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\downloads\FDR-YAML-Declarative-v1-SHA256SUMS.txt
```

下载 checkpoint 后先核对：

```powershell
Get-FileHash .\fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt -Algorithm SHA256
```

Linux：

```bash
sha256sum fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt
```

## 7. GitHub 上传现状

截至 2026-08-05 的远端实际资产审计：

- `fdr-yaml-declarative-v1`：12 个公开资产，包含完整/消融 YAML、overlay、bundle、说明和两份机器证据；
- `fdr-formal-d97e1eb7-live`：公开 epoch 98、99、100 的 `.pt` 与 `.json`，共 6 个资产；
- `fdr-screen-d97e1eb7-live`：公开若干中间/最终 checkpoint，其中包括 FDR epoch30 和 control epoch30；
- 30-epoch 两臂完整轻量结果、Gate2 和 YAML 兼容证据已经进入 Git 仓库。

因此当前**不能说 formal 1–100 每轮 checkpoint 都已上传 GitHub Release**。训练时的逐 epoch 本地证据/发布队列与 GitHub 当前公开资产必须分开表述。若服务器或备份中仍保留早期 checkpoint/manifest，应按原文件名和 SHA256 补传；若已不存在，只能诚实标记缺失，不能重建或伪造旧轮次。

## 8. 接手后任务清单

### P0：恢复 strict formal authority 并跑 matched stock control

目标：获得可与 FDR formal100 直接写入论文主表的 stock control。

必须先恢复以下原始 authority 文件，而不是重新随机生成：

```text
/data/uav/protocols/fdr-d97e1eb7/protocol.json
/data/uav/protocols/fdr-d97e1eb7/initial-state.pt
```

验收哈希：

- protocol：`2545F68302639EEB217EC8B53FDB229681B2DAFC91463C61F6B5CEC9B5486302`；
- initial-state：`51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D`；
- source commit：`d97e1eb7f98414752a1c1f38287697db3f2a0679`。

启动前必须执行 dry-run。Linux 示例：

```bash
python scripts/train_rtdetr_fdr.py \
  --variant control \
  --stage formal \
  --protocol-manifest /data/uav/protocols/fdr-d97e1eb7/protocol.json \
  --initial-state /data/uav/protocols/fdr-d97e1eb7/initial-state.pt \
  --dataset-root /data/uav/VisDrone \
  --output-root /data/uav/runs/fdr-formal-control-d97e1eb7 \
  --dry-run
```

dry-run 全过后去掉 `--dry-run`。验收标准：

- 从 epoch1 fresh 训练，不继承 FDR 或 screen checkpoint；
- 连续 100 epoch；
- source/protocol/initial-state/data/seed 均与 authority 一致；
- 每轮轻量 JSON/CSV 和 checkpoint manifest 可恢复；
- 不改 batch、workers、optimizer、AMP、增强或 checkpoint 规则；
- 最终独立 evaluator 与 FDR 使用同一版本、预处理和类别映射。

如果原始 protocol 或 initial-state 缺失，严格比较被阻塞。不要用新 initial-state 冒充原 authority；新生成状态只能启动一对新的 control/FDR replication。

### P1：生成论文完整比较表

对严格 control 和正式 FDR epoch100 统一评估，至少输出：

- Precision、Recall；
- mAP50、mAP50-95、AP75；
- AP-tiny、AP-small、AP-medium、AP-large，并明确尺寸定义；
- 10 类逐类别 AP50、AP75、mAP50-95；
- 每类 GT 数、prediction 数，防止类别映射错误；
- best 与 final checkpoint 分开报告；
- tail-3 仅作训练稳定性辅助，不能代替 final 独立评估；
- PR 曲线、F1/confidence 曲线、混淆矩阵；
- 参数量、GFLOPs、端到端 latency、FPS、peak VRAM、checkpoint 大小。

验收标准：同一 548 图 val、同一预处理、同一类别映射、同一 evaluator 代码提交，报告保存原始 prediction/metric JSON 和 SHA256。只报告真实有符号 delta，不事后调整门槛。

### P2：训练四个 FDR 单变量消融

四个消融分别回答：

1. `fgl_weight: 0.15 -> 0.0`；
2. `supervise_pre_boxes: true -> false`；
3. `cumulative: true -> false`；
4. `preliminary_box: true -> false`。

当前验证只证明五份 YAML 能加载同一 checkpoint，不证明消融有效。论文消融必须让每一臂从相同公共 initial-state 独立训练。先按固定 10%/seed0/30-epoch cutoff 做等协议筛选，再按预注册规则决定是否进入 full100；不得用“完整模型权重 + 推理时关开关”替代训练消融。

### P3：决定并验证 SADED-SM 的论文角色

优先顺序：

1. 在独立 worktree 用原 checkpoint SHA 复现既有 SADED-SM evidence；
2. 测量 1-view Arm A 与 5-view SADED-SM 的 latency/FPS/VRAM；
3. 若要与 FDR 组合，先冻结规则，不用 FDR val 重新调阈值；
4. 比较 `FDR full-view` 与 `FDR + SADED-SM`，不能与旧 Arm A 跨 checkpoint 拼表；
5. 组合结果正向后再决定是否作为论文第二创新点。

### P4：补齐发布和灾备

- 核对训练服务器/本地备份是否仍有 formal epoch1–97 checkpoint 与 manifest；
- 补传真实存在且 SHA 可核验的资产；
- 将完整轻量 evidence 提交到专用 results 分支；
- 每次训练使用不可变 run 目录和 append-only publication queue；
- GitHub 443 暂时不可达时训练不能停止，先保存在本地队列，恢复后补传；
- 不把数据集、密码、Token、私钥写入 Git 或日志。

当前用户指令是 seed0-only；不要自行启动 seed2。若以后需要统计显著性，应先由用户重新授权并为 control/final method 统一增加 seed，而不是只补某一臂。

## 9. 接手后的前 30 分钟

按顺序执行：

1. clone 仓库并创建 FDR/SADED/formal-authority 三个 worktree；
2. `git status --short` 确认各工作树干净；
3. 下载 epoch100 checkpoint 和 SHA256 清单；
4. 验证 checkpoint SHA256；
5. 在 `codex/fdr-yaml-module` 执行五 YAML strict-load 验证；
6. 查找原 protocol.json 与 initial-state.pt，并核对两个 authority SHA；
7. 只有 authority 完整时才启动 strict control dry-run；
8. 建立新的不可变 run、日志目录、publication queue 和 GitHub Release tag；
9. 记录 PID/GPU/首轮 metrics/checkpoint/resume smoke；
10. 训练继续运行时，并行准备统一 evaluator 和论文指标导出脚本。

五 YAML checkpoint 验证命令：

```powershell
python scripts/verify_fdr_yaml_checkpoint.py `
  --cfgs configs/rtdetr-l-fdr.yaml `
         configs/rtdetr-l-fdr-no-fgl.yaml `
         configs/rtdetr-l-fdr-no-prebox-loss.yaml `
         configs/rtdetr-l-fdr-no-cumulative.yaml `
         configs/rtdetr-l-fdr-no-prebox.yaml `
  --checkpoint .\fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt `
  --output .\checkpoint-compatibility-all-configs.json `
  --nc 10 `
  --imgsz 128
```

成功报告必须满足：

```text
all_configs_verified = true
config_count = 5
strict_load = true
missing_keys = 0
unexpected_keys = 0
finite_output = true
head_type = FDRRTDETRDecoder
```

## 10. 论文可用结论与禁止越界

### 当前可以写

- FDR-only 已完成 Ultralytics RT-DETR-L 的定位路径结构适配和 YAML 模块化；
- F0–F4、30-epoch paired screen、正式 checkpoint 五配置兼容和真实 resume-step 已完成；
- FDR 30-epoch screen 的 mAP/AP75/tail3 mAP 均严格正向；
- SADED-SM 在 seed0 development-val 的五个预注册 Gate 均通过；
- quality-reordering oracle 证明分数—定位质量错位具有很高理论上限；
- Boundary/LPR/OAR 的失败证据解释了为何当前主线转向 FDR-only。

### 当前不能写

- FDR formal100 已严格超过 matched control；
- FDR 对 AP-tiny、AP-small 和所有 10 类均提升；
- FDR 参数量增幅 `<1%`；
- FDR latency 增幅 `<3%`；
- 四个 FDR 消融已经完成训练；
- SADED-SM 是零开销或五模型 ensemble；
- SADED-SM 已完成三 seed 或 test-dev 验证；
- perfect quality oracle 可部署；
- formal 1–100 每轮 checkpoint 已全部公开到 GitHub。

## 11. 安全与交接责任

此前服务器与 GitHub 凭据曾通过会话传递。下一位负责人开始工作前应轮换已暴露的密码/Token，并只通过环境变量或凭据管理器使用；任何报告、命令历史、Git commit、Release manifest 都不得包含明文凭据。

接手人应保留所有失败实验和既定阈值。工程失败可以修代码并续跑；科学失败不能通过改阈值、挑最好 epoch、换 evaluator 或删除负结果变成“通过”。

本交接的最终执行顺序为：

```text
恢复 formal authority
  -> strict stock control 100 epoch
  -> control/FDR 统一独立评估
  -> 完整论文指标与开销
  -> FDR 四项训练消融
  -> SADED-SM 开销与 FDR 组合验证
  -> 冻结论文表格、权重、哈希和可复现包
```
