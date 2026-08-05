# FDR-RTDETR-L：YAML 声明式细粒度分布定位模块

> 文档性质：论文方法章节、工程实现说明与复现实验指南
>
> 基础模型：Ultralytics RT-DETR-L 8.4.90
>
> 上游机制 authority：D-FINE commit `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6`
>
> 正式模型/训练 authority：`d97e1eb7f98414752a1c1f38287697db3f2a0679`
>
> 证据边界：30-epoch 配对筛选结果已冻结；epoch-100 checkpoint 已对完整配置和四个消融 YAML 完成严格加载与有限推理；旧格式内嵌 YAML 的 resume 重建及 optimizer/scaler/EMA 元数据保留已经验证。本文不把工程兼容等同于最终精度结论。

## 摘要

针对 RT-DETR-L 四维连续框回归对密集小目标边界表达粒度有限的问题，本项目将 D-FINE 的 Fine-grained Distribution Refinement（FDR）与 Fine-Grained Localization（FGL）机制迁移到 Ultralytics RT-DETR-L 的定位路径中。方法保持 Backbone、Hybrid Encoder、Query 选择、六层 Transformer Decoder 的注意力与 FFN、分类分支、匈牙利匹配、Top-300 后处理以及 `NMS=False` 不变，只重构 Decoder 的边界框表示与定位监督：先生成 preliminary box，再由六个 132 维分布回归头预测四条边的 33-bin 分布，逐层累计分布残差，并通过固定非均匀 Integral 解码为连续边界框。训练时保留 stock VFL/L1/GIoU，复用原匹配索引增加 FGL 与 preliminary-box 辅助定位损失。

为使结构可见、可复现且便于消融，当前实现将完整定位功能单元声明为 YAML 最后一层 `FDRRTDETRDecoder`，不再依赖“先构建 stock 模型、再在模型外部替换回归头”的隐式入口。完整配置和四份单变量消融配置均保留同一 RT-DETR-L 图结构、同一参数命名合同和同一训练协议。

固定 10% VisDrone 子集、seed0、30 epoch 的配对 Gate2 已通过：最终 mAP、尾三轮平均 mAP、最终 AP75 相对 control 分别为 `+0.01801`、`+0.0156366667` 和 `+0.0154278605`。此外，正式 epoch-100 EMA checkpoint 已由五套 YAML 模型逐一严格加载，950 个状态张量均实现 0 missing / 0 unexpected，并完成有限输出推理。真实旧格式 checkpoint 也已通过声明式模型重建，保留 epoch、updates、optimizer、scaler 和 EMA 载荷。该证据证明新 YAML 入口可承接既有正式权重与恢复载荷，但不单独证明全数据最终精度提升或所有消融均有效。

## 1. 研究动机与方法定位

Ultralytics RT-DETR-L 在每个 Decoder 层通过一个四维 MLP 直接回归边界框：

\[
\hat B_l=\sigma\left(H_l^{box}(h_l)+\sigma^{-1}(R_l)\right),
\]

其中，\(h_l\) 为第 \(l\) 层 hidden state，\(R_l\) 为当前 reference box。该点估计形式紧凑，但不会显式表示某条边在相邻位置上的竞争关系。VisDrone 中大量目标像素尺寸小、遮挡多、边界模糊，定位误差容易直接影响 AP75 和小目标 AP。

FDR 将四维点回归改为四条边的细粒度分布回归：

- 每条边 33 个分箱，四边共 `4 × 33 = 132` 个 logits；
- 六层 Decoder 在同一个 preliminary-box 参考系内累计分布残差；
- 固定的非均匀 Integral 将概率分布映射为连续四边距离；
- FGL 使用匹配目标的相邻分箱插值与 detached IoU 权重监督分布。

本项目的准确定位是“FDR/FGL 面向 Ultralytics RT-DETR-L 与 VisDrone 的结构化迁移、隔离集成和统一协议验证”，而不是声称原创 D-FINE 的 FDR 公式，也不是完整复现 D-FINE 检测器。

## 2. 网络结构修改

### 2.1 总体修改边界

```text
P3 / P4 / P5 Encoder Features
              │
              ▼
  Stock RT-DETR Decoder Attention + FFN
              │
       ┌──────┴─────────┐
       │                │
Stock Class Heads   FDR Box Path
   （不变）             │
                   Preliminary Box
                        │
                  6 × 132-d Heads
                        │
             Cumulative Distribution Logits
                        │
             Non-uniform Integral / Decode
                        │
                  Refined Boxes
```

完整 FDR 配置与 stock RT-DETR-L 的第 0--27 层完全一致，只替换最后一个 Decoder head 的类型与定位参数。以下部分保持 stock Ultralytics 8.4.90 行为：

- HGNetv2 Backbone；
- P3/P4/P5 多尺度特征路径与 Hybrid Encoder；
- uncertainty-minimal Query selection；
- Query 数 300；
- 六层 Decoder 的 deformable attention、FFN、LayerNorm；
- 六层分类头与分类 logits；
- denoising Query 构造；
- 匈牙利匹配及其 cost；
- Query-by-class Top-300 后处理；
- `max_det=300` 与 `NMS=False`；
- 推理输出合同 `[batch, 300, 6]`。

### 2.2 YAML 声明的完整功能单元

完整配置 `configs/rtdetr-l-fdr.yaml` 的最后一层为：

```yaml
- [[21, 24, 27], 1, FDRRTDETRDecoder,
   [nc, [256, 256, 256],
    {hidden_dim: 256, num_queries: 300, num_decoder_layers: 6,
     reg_max: 32, reg_scale: 4.0, up: 0.5, cumulative: true,
     preliminary_box: true, private_seed: 10000}]]
```

`FDRRTDETRDecoder` 是 YAML 层级上的完整定位功能单元。它继承 stock `RTDETRDecoder`，先按原结构建立特征投影、Encoder 输出头、Query、Decoder layers 和分类头，再仅对 box path 安装 FDR 子结构。这样既让结构在 YAML 中可见，又不人为拆散必须在 Decoder 逐层循环内协同工作的操作。

### 2.3 Preliminary box

第 0 层 hidden state 产生传统四维粗框：

\[
B_{pre}=\sigma\left(H_{pre}(h_0)+\sigma^{-1}(R_0)\right).
\]

`H_pre` 的权重路径固定为：

```text
model.<head>.decoder.pre_bbox_head.*
```

其 MLP 结构与 stock 第 0 层 box head 对齐，得到的 `B_pre.detach()` 作为完整配置中六层分布解码的共享几何参考。该分支实现“粗定位—细粒度边界细化”的职责分离。

### 2.4 六层分布回归头

`reg_max=32` 时，每条边包含 33 个 bin。每层分布头为：

```text
256 → 256 → 256 → 132
```

稳定权重路径为：

```text
model.<head>.dec_bbox_head.0.*
...
model.<head>.dec_bbox_head.5.*
```

六个输出层的 weight 与 bias 零初始化，使初始 residual logits 不偏向任一分箱。私有初始化种子为 `10000 + experiment_seed`；公共模型初始化仍由配对 initial-state authority 控制。

### 2.5 跨层累计细化

六层分布残差定义为：

\[
\Delta Z_0=H_0^{dist}(h_0),
\]

\[
\Delta Z_l=H_l^{dist}\left(h_l+\operatorname{sg}(h_{l-1})\right),\quad l>0,
\]

\[
Z_l=Z_{l-1}+\Delta Z_l.
\]

`cumulative=true` 时，后层在前层累计分布上继续修正；`cumulative=false` 时，每层只解码本层的 `ΔZ_l`。hidden-state detach、reference detach、logits 累计与 distance-to-box 转换均依赖 Decoder 层循环，不被伪拆为 YAML 外部串行层。

### 2.6 非均匀 Integral 与边界框解码

将每条边 logits reshape 为 33-bin 概率后，固定 Integral 计算：

\[
d_l^e=\sum_{k=0}^{32}\operatorname{softmax}(Z_l^e)_k W(k),
\]

其中，`W(k)` 由固定 `reg_scale=4.0` 和 `up=0.5` 生成，靠近零偏移的区域具有更高分辨率。Integral 投影向量是 buffer，不包含可训练参数。四边距离相对 reference 经 `distance2bbox` 恢复为 \((c_x,c_y,w,h)\)。

正式实现只迁移 commit-pinned D-FINE 中的 weighting、Integral、`distance2bbox`、`bbox2distance` 和 FGL primitive；未迁移 DDF、GO-LSD、LQE 或 teacher/student 机制。

## 3. 损失函数与匹配隔离

模型 YAML 同时声明训练期定位监督：

```yaml
fdr_loss:
  fgl_weight: 0.15
  supervise_pre_boxes: true
```

Stock VFL、L1、GIoU、Encoder/Decoder auxiliary losses 与 DN losses 完整保留。FDR 增加：

1. 主层和辅助层 FGL；
2. preliminary box 的 L1/GIoU 辅助监督；
3. 存在 DN Query 时对应的 FDR/FGL 监督。

FGL 不调用第二个 matcher。它复用每个 stock prediction group 已经产生的一对一匹配索引，以 detached preliminary/reference box 将 GT 四边距离编码到相邻非均匀分箱，并用 detached matched IoU 加权相邻分箱交叉熵。完整方法的优化目标可概括为：

\[
L=L_{stock}+0.15L_{FGL}^{main}+0.15\sum_lL_{FGL,l}^{aux}
+L_{pre}^{L1}+L_{pre}^{GIoU}+L_{DN,FDR}.
\]

因此，已经通过 30-epoch Gate2 的实验臂是“分布表示 + 跨层累计 + FGL + preliminary-box 参考与辅助监督”的完整组合，不能在缺少消融结果时把全部收益单独归因给其中一项。

## 4. 完整功能单元与消融边界

### 4.1 为什么不把每个算子都拆成 YAML 层

一个可独立声明的功能单元应具有明确输入、完整计算逻辑和稳定输出合同。`FDRRTDETRDecoder` 满足这一条件；而以下操作无法脱离六层 Decoder 循环独立工作：

- distribution logits 的跨层累计；
- previous hidden state 的 detach 与相加；
- refined reference 的逐层更新；
- distribution-to-distance-to-box 转换；
- normal/DN Query 的同步证据记录。

强行把这些细节拆成外部层会改变 Ultralytics 的 Decoder 调用合同、state-dict 路径或 checkpoint 兼容性。因此，它们作为模块内部策略，由 YAML 参数控制，而不是为增加“模块数量”而形式化拆分。

### 4.2 五份 YAML：一份完整配置与四份单变量消融

| 配置 | 相对完整配置唯一变化 | 研究问题 |
|---|---|---|
| `rtdetr-l-fdr.yaml` | 无 | 完整 FDR 方法 |
| `rtdetr-l-fdr-no-fgl.yaml` | `fgl_weight: 0.15 → 0.0` | 分离 FGL 监督贡献 |
| `rtdetr-l-fdr-no-prebox-loss.yaml` | `supervise_pre_boxes: true → false` | 分离粗框辅助监督贡献 |
| `rtdetr-l-fdr-no-cumulative.yaml` | `cumulative: true → false` | 分离跨层分布累计贡献 |
| `rtdetr-l-fdr-no-prebox.yaml` | `preliminary_box: true → false` | 分离 preliminary-box 几何参考功能 |

四个消融 YAML 均是可独立构建的完整模型文件，不依赖运行时命令修改字段。静态配置测试已证明每个消融相对完整 YAML 只改变表中一个叶字段，并且第 0--27 层与 Ultralytics 8.4.90 stock 图一致。

为兼容正式 checkpoint，`no-prebox` 不删除 `pre_bbox_head` 参数，只关闭它作为分布 reference 的作用；有效 criterion 同时关闭 pre-box 辅助损失。FGL target encoding 在 `preliminary_box=false` 时使用原始 `references[0]`，DN 分支使用 `dn_references[0]`，不再使用未路由的 pre-box；该语义已由故意构造不同 reference/pre-box 的测试覆盖。

Stock baseline 不属于第五个 FDR 消融。它继续使用 Ultralytics 原生 `rtdetr-l.yaml` 和四维连续回归头。

## 5. YAML 与正式权重兼容设计

### 5.1 从隐式注入迁移到声明式构建

历史正式实现的计算语义是：先创建 stock RT-DETR-L，再在 Python 中替换 Decoder box path。新实现改为：

1. 仓库注册自有 `FDRRTDETRDecoder`；
2. `FDRRTDETRDetectionModel` 直接从 `configs/rtdetr-l-fdr.yaml` 构建；
3. 自定义 head 在自身构造函数内部完成与历史实现相同的 FDR box-path 安装；
4. 不编辑 site-packages 中的 Ultralytics 源码；
5. 保持 `pre_bbox_head`、六个 `dec_bbox_head` 和 `integral` 的原 state-dict 路径、形状和 dtype。

这是一项“声明与构建入口重构”，不是重新设计或重新训练正式 FDR 算法。

### 5.2 已证明的兼容性

当前已有以下直接证据：

- 历史注入模型与完整声明式模型的 state-dict 键、形状及初始值精确一致；
- 加载同一权重后，完整声明式模型与历史注入模型的 eval 输出精确一致；
- 五份 YAML 均可构造 state-compatible 模型；
- `fgl_weight` 与 `supervise_pre_boxes` 从 YAML 进入 criterion；
- 完整正式 epoch-100 EMA checkpoint 已在真实文件上对五套 YAML 逐一严格加载并完成有限推理；
- FDR/FDR 与 FDR/Control 并发构建由同一进程级锁隔离，解析器别名不会泄漏；
- 私有分布头使用独立 `torch.Generator`，与历史 seed 10000 初始化逐位一致且不污染公共 RNG；
- 旧格式 checkpoint 的 stock-named 内嵌 YAML 可被识别并规范化为声明式 FDR YAML，普通 stock checkpoint 不会被误识别。

已留存的核心回归快照为：

```text
167 passed, 3 skipped
```

该结果由最终集成工作树独立重跑获得，覆盖 authority、数学、head、loss、protocol、preflight、model、训练 CLI、五套 YAML 与 checkpoint 验证器。

### 5.3 正式 epoch-100 checkpoint 严格加载报告

实际 checkpoint：

```text
fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt
SHA256: c2f638744508adfe7b6c4a1ef3e08c503273f628062e4650ad59ffff4c6588c2
Size: 200024985 bytes
```

公开发布页：<https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-d97e1eb7-live>。下载后必须先核对上述字节数与 SHA256，再执行反序列化和严格加载；Release 页面地址是公开证据入口，不需要也不应在文档或命令中嵌入访问令牌。

仓库内冻结报告 `research/fdr/evidence/d97e1eb7/yaml-module-final/checkpoint-compatibility-all-configs.json`：

| 字段 | 结果 |
|---|---:|
| checkpoint source | `ema`（artifact 中 `model=null`） |
| verified YAML count | 5 |
| state tensor count | 950 |
| strict load | `true` |
| missing keys | 0 |
| unexpected keys | 0 |
| YAML head type | `FDRRTDETRDecoder` |
| deterministic smoke output | `[1,300,6]` |
| finite output | `true` |

五个配置均得到相同的严格加载结论和 `[1,300,6]` 有限输出。另一次真实旧格式 checkpoint resume-step 审计确认：内嵌 `RTDETRDecoder` YAML 被规范化为 `FDRRTDETRDecoder`，模型转移 `950/950`，恢复 8 个 MuSGD 参数组、581 个 optimizer state、AMP scale 128 与 EMA updates 10556；随后真实执行一次 `128x128` 前向、反向、MuSGD step 和 EMA update，loss/梯度有限，EMA updates 增至 10557。机器报告位于 `research/fdr/evidence/d97e1eb7/yaml-module-final/legacy-resume-step.json`。

### 5.4 兼容性结论边界

已经可以主张：五套 YAML 的网络状态合同与正式权重兼容；完整配置与历史注入式实现的 state-dict 和 eval 输出一致；旧格式 checkpoint 可完成声明式模型重建；并发解析与私有 RNG 已隔离。仍不能把这些工程测试写成“全部消融有效”或“最终精度提升”，因为消融训练、严格 matched baseline 精度比较及多 seed 统计属于独立的科学验证问题。

## 6. 与 baseline 对齐的统一实验协议

### 6.1 环境与数据

| 项目 | 冻结值 |
|---|---|
| Model | Ultralytics RT-DETR-L |
| Ultralytics | 8.4.90 |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| Driver | 550.142 |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| CUDA | 12.1 |
| Dataset | 同一份 VisDrone train/val |
| Train / Val | 6471 / 548 |
| Classes | 10 |
| Dataset SHA256 | `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB` |
| 固定 10% 子集 | 647 张 |
| 子集 SHA256 | `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0` |

YAML 文件保留 stock 模板的 `nc: 80`，但训练构建时由 VisDrone data authority 显式覆盖为 `nc=10`。该行为与 Ultralytics 原生模型构建一致。

### 6.2 训练与推理参数

| 项目 | 冻结值 |
|---|---|
| 初始化 | `pretrained=False`，从零训练 |
| Screen | 固定 647 张子集、seed0、共同 50-epoch schedule、第 30 轮截止 |
| Formal | 全部 6471 张、seed0、fresh 100 epoch |
| `imgsz` / `batch` / `workers` | 640 / 8 / 8 |
| `device` | 0，单卡 |
| AMP | True，固定 scale 128 |
| `deterministic` / `cache` | True / False |
| Optimizer | MuSGD |
| `lr0` / `lrf` | 0.01 / 0.01 |
| `momentum` | 0.937 |
| `weight_decay` | 0.0005 |
| Warmup | epochs 3.0，momentum 0.8，bias LR 0.0 |
| `nbs` / `cos_lr` | 64 / False |
| Queries / `max_det` | 300 / 300 |
| NMS | False |

### 6.3 数据增强

```text
mosaic=1.0, close_mosaic=10, mixup=0.0
scale=0.5, translate=0.1
degrees=0.0, shear=0.0, perspective=0.0
flipud=0.0, fliplr=0.5
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
cutmix=0.0, copy_paste=0.0
```

Control 与 FDR 还必须共享公共参数初始状态、样本顺序、增强随机序列、验证预处理、类别映射、checkpoint 规则和指标代码。方法臂允许的差异仅限 FDR box path、FDR 私有初始化及对应监督。

## 7. 已完成检验与实验证据

### 7.1 F0--F4 工程门检

| Gate | 目标 | 已冻结结果 |
|---|---|---|
| F0 | 固定 D-FINE commit 的 weighting、Integral、box transform、FGL golden parity | 通过 |
| F1 | neutral encode/decode、累计 residual、`FGL=0` stock loss isolation、分类/匹配/后处理隔离 | 通过 |
| F2 | normal/DN/aux、空 GT、边界值、finite forward/backward、AMP128 | 通过 |
| F3 | RTX 4090、真实 VisDrone batch8、forward/backward、MuSGD、validation、checkpoint | 通过 |
| F4 | 表示重建误差、分箱饱和、tiny/small 分层 | 通过 |

F4 在 35,246 个 matched targets 上得到：未饱和 reconstruction L1 `6.2766e-09`、max `5.9605e-08`、总 edge saturation `6.44399%`。Tiny saturation 为 `9.77196%`，高于 small 的 `2.03930%`，这是必须在论文中保留的表示范围局限。

### 7.2 固定 10% 子集、seed0、30-epoch Gate2

Gate2 要求三项同时严格为正：

1. epoch30 mAP 差值；
2. epoch28--30 平均 mAP 差值；
3. epoch30 AP75 差值。

最终轮结果：

| 指标 | Control epoch30 | FDR epoch30 | FDR - Control |
|---|---:|---:|---:|
| Precision | 0.00662 | 0.07229 | +0.06567 |
| Recall | 0.02501 | 0.13717 | +0.11216 |
| mAP50-95 | 0.00026 | 0.01827 | **+0.01801** |
| AP75 | 0.0000304090 | 0.0154582695 | **+0.0154278605** |

尾三轮结果：

| 指标 | Control | FDR | FDR - Control |
|---|---:|---:|---:|
| epoch28--30 mean mAP | 0.0009033333 | 0.01654 | **+0.0156366667** |
| epoch28--30 mean AP75 | 0.0002738164 | 0.0142321482 | +0.0139583318 |

机器判定为：

```text
engineering.complete = true
gate.passed = true
formal_eligible = true
```

这组结果只用于筛选：control 在 647 张子集上的绝对值很低，从零训练方差大，因此 `+1.801 pp` 不能直接写成正式 100-epoch 提升，也不能代表多 seed 稳定性。

### 7.3 开销的现有参考数据

| 指标 | Stock | FDR | 增量 | 相对增幅 |
|---|---:|---:|---:|---:|
| Parameters | 32,826,626 | 33,156,614 | +329,988 | **+1.00524%** |
| GFLOPs | 108.0318976 | 108.2291200 | +0.1972224 | **+0.18256%** |

参数增幅略高于 1%，因此论文不能写“参数增幅 `<1%`”。端到端 latency/FPS 尚未在最终统一审计中冻结，不能提前写成 `<3%`。

## 8. 使用与复现

以下命令均应在仓库根目录运行。路径示例是占位符，不包含任何凭据。

### 8.1 静态查看完整模型与四份消融

```powershell
Get-Content configs/rtdetr-l-fdr.yaml
Get-Content configs/rtdetr-l-fdr-no-fgl.yaml
Get-Content configs/rtdetr-l-fdr-no-prebox-loss.yaml
Get-Content configs/rtdetr-l-fdr-no-cumulative.yaml
Get-Content configs/rtdetr-l-fdr-no-prebox.yaml
```

### 8.2 FDR 30-epoch screen

训练入口内部使用共同 50-epoch schedule，并在第 30 轮以 callback 截止：

```powershell
python scripts/train_rtdetr_fdr.py `
  --variant fdr `
  --stage screen `
  --protocol-manifest C:\path\to\protocol-manifest.json `
  --initial-state C:\path\to\paired-initial-state.pt `
  --dataset-root C:\path\to\VisDrone `
  --output-root C:\path\to\runs `
  --publication-queue C:\path\to\publication-queue.jsonl
```

### 8.3 Stock control 30-epoch screen

```powershell
python scripts/train_rtdetr_fdr.py `
  --variant control `
  --stage screen `
  --protocol-manifest C:\path\to\protocol-manifest.json `
  --initial-state C:\path\to\paired-initial-state.pt `
  --dataset-root C:\path\to\VisDrone `
  --output-root C:\path\to\runs `
  --publication-queue C:\path\to\publication-queue.jsonl
```

`variant=fdr` 默认使用 `configs/rtdetr-l-fdr.yaml`；`variant=control` 使用 Ultralytics 原生 `rtdetr-l.yaml`。

### 8.4 正式全数据 100 epoch

```powershell
python scripts/train_rtdetr_fdr.py `
  --variant fdr `
  --stage formal `
  --protocol-manifest C:\path\to\protocol-manifest.json `
  --initial-state C:\path\to\formal-initial-state.pt `
  --dataset-root C:\path\to\VisDrone `
  --output-root C:\path\to\formal-runs `
  --publication-queue C:\path\to\formal-publication-queue.jsonl
```

Formal 必须从正式 seed0 initial state fresh 启动，禁止 resume 30-epoch screen checkpoint。

### 8.5 启动前 dry-run

```powershell
python scripts/train_rtdetr_fdr.py `
  --variant fdr `
  --stage formal `
  --protocol-manifest C:\path\to\protocol-manifest.json `
  --initial-state C:\path\to\formal-initial-state.pt `
  --dataset-root C:\path\to\VisDrone `
  --output-root C:\path\to\formal-runs `
  --dry-run
```

Dry-run 会校验 source authority、数据集哈希、initial-state 和冻结配置，但不会创建 Trainer。

### 8.6 验证 epoch-100 权重与五套 YAML

```powershell
python scripts/verify_fdr_yaml_checkpoint.py `
  --cfgs configs/rtdetr-l-fdr.yaml `
         configs/rtdetr-l-fdr-no-fgl.yaml `
         configs/rtdetr-l-fdr-no-prebox-loss.yaml `
         configs/rtdetr-l-fdr-no-cumulative.yaml `
         configs/rtdetr-l-fdr-no-prebox.yaml `
  --checkpoint artifacts/formal-checkpoint/fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt `
  --output artifacts/fdr-yaml-checkpoint-compatibility-all-configs.json `
  --nc 10 `
  --imgsz 128
```

成功报告必须同时满足：`all_configs_verified=true`、`config_count=5`，且每个配置均为 `strict_load=true`、`missing_keys=0`、`unexpected_keys=0`、`finite_output=true`、`head_type=FDRRTDETRDecoder`。

### 8.7 同一不可变 run 内续跑

训练 CLI 的接口为：

```powershell
python scripts/train_rtdetr_fdr.py `
  --variant fdr `
  --stage formal `
  --protocol-manifest C:\path\to\protocol-manifest.json `
  --initial-state C:\path\to\formal-initial-state.pt `
  --dataset-root C:\path\to\VisDrone `
  --output-root C:\path\to\formal-runs `
  --resume C:\path\to\same-run\weights\epoch56.pt
```

Resume 文件必须位于原 run 的 `weights` 目录，且相邻 `fdr-run.json` 中的 run identity、protocol、source、stage 和 variant 必须一致。历史 checkpoint 内嵌模型 YAML 即使仍命名为 stock `RTDETRDecoder`，只要其 state signature 确认为 FDR，Trainer 会将其规范化为声明式 FDR YAML；普通 stock checkpoint 不会走该兼容分支。真实 epoch-100 checkpoint 已验证模型重建、950/950 权重转移以及 epoch/updates/optimizer/scaler/EMA 载荷保留。

## 9. 论文中的原创性边界

### 9.1 可以主张

- 将 commit-pinned FDR/FGL 机制适配到 Ultralytics RT-DETR-L 的 box path；
- 在不改分类、Query、Encoder、匹配与后处理的条件下完成结构隔离；
- 建立公共参数字节一致、私有初始化隔离、原匹配索引复用的公平协议；
- 将隐式 Python 注入重构为 YAML 声明式、checkpoint-compatible 的独立定位模块；
- 提供完整配置与四个单变量 YAML 消融入口；
- 在 VisDrone 上完成 F0--F4、配对 screen 与正式权重兼容验证。

### 9.2 不可以主张

- 原创 D-FINE 的 FDR、FGL、Integral 或非均匀 weighting 公式；
- 完整复现 D-FINE 的检测器或训练配方；
- 已经证明每个消融都带来正收益；
- 仅凭 30-epoch 子集结果证明全数据 100-epoch 最终提升；
- 已完成多 seed 稳定性或统计显著性验证；
- epoch-100 权重严格加载等同于完整 Trainer resume；
- 参数增幅 `<1%` 或延迟增幅 `<3%`；
- 在没有单独消融时把完整方法收益只归因于 FDR、FGL、累计策略或 pre-box 中任一项。

论文贡献建议写为：

> 本文在保持 Ultralytics RT-DETR-L 主干、编码器、Query、分类分支、匹配机制与后处理不变的条件下，将细粒度分布定位机制封装为 YAML 声明式 `FDRRTDETRDecoder`。该模块通过 preliminary box、六层累计四边分布回归和 FGL 监督重构 Decoder 定位路径，同时保持既有正式权重的严格加载兼容，并以单变量配置支持各功能单元的可复现实验归因。

## 10. 当前结论

当前 YAML 声明式 FDR 模块已经满足四个核心工程条件：结构在模型配置中显式可见、五套配置可严格承接正式 epoch-100 权重、旧 checkpoint 可恢复为声明式模型、固定 10% 子集配对筛选获得 mAP 与 AP75 同向正收益。它可以作为论文三个创新点中的“定位回归机制改进”进行描述。

但最终论文结论仍必须区分三件事：

1. `+0.01801` 是 30-epoch 子集 Gate2 的筛选增益；
2. epoch-100 checkpoint 的当前报告是架构/权重兼容证据；
3. 最终全数据精度、严格 matched baseline 差值、APtiny/APsmall/逐类别指标、各 YAML 消融和延迟审计必须由后续独立评估分别给出。

最严谨的表述是：“完整 FDR 方法已通过工程门检、配对筛选、五配置权重兼容与旧 checkpoint 恢复审计；最终效果归因仍需各消融训练，全量收益仍以严格 matched baseline 独立对照为准。”

## 11. 实现与证据索引

| 内容 | 文件 |
|---|---|
| 完整 YAML | `configs/rtdetr-l-fdr.yaml` |
| 四份消融 YAML | `configs/rtdetr-l-fdr-no-*.yaml` |
| YAML head 与 Decoder box path | `src/fdr_head.py` |
| weighting / Integral / box transforms | `src/fdr_math.py` |
| FGL 与 stock loss 隔离 | `src/fdr_loss.py` |
| Model / Trainer 集成 | `src/rtdetr_fdr.py` |
| 冻结实验协议 | `src/fdr_protocol.py` |
| 训练与逐 epoch 证据入口 | `scripts/train_rtdetr_fdr.py` |
| checkpoint 严格验证器 | `scripts/verify_fdr_yaml_checkpoint.py` |
| legacy resume-step 验证器 | `scripts/verify_fdr_legacy_resume_step.py` |
| 30-epoch Gate2 | `research/fdr/evidence/d97e1eb7/fdr-gate-d97e1eb7/gate2.json` |
| checkpoint 五配置兼容报告 | `research/fdr/evidence/d97e1eb7/yaml-module-final/checkpoint-compatibility-all-configs.json` |
| 真实 legacy resume-step 报告 | `research/fdr/evidence/d97e1eb7/yaml-module-final/legacy-resume-step.json` |

## 参考文献

1. Lv W, Zhao Y, Chang Q, et al. RT-DETR: DETRs Beat YOLOs on Real-time Object Detection. 2023.
2. Peng Y, et al. D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement. ICLR 2025. 本项目迁移 authority 固定为官方仓库 commit `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6`。
