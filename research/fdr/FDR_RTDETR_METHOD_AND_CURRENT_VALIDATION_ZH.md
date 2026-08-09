# FDR-RTDETR-L：面向 VisDrone 定位回归的细粒度分布细化方法与正式验证

> 文档性质：论文“方法设计—实现细节—实验设置—正式结果”章节草稿
> 证据更新时间：2026-08-09（Asia/Shanghai）
> 当前结论边界：固定 10% 子集 seed0 paired Screen30 已通过 Gate2；全数据 seed0 的 FDR 与同 formal authority stock Control 均已完成 100 epoch。统一独立复评显示 mAP50-95 `0.21911 → 0.28966`（`+7.055 pp`）。该结果仍限于 seed0，且 exact-checkpoint tensor identity、最终 latency 和多 seed 统计尚未封口。

## 摘要

针对 RT-DETR 在密集小目标场景中直接回归四维边界框时定位表达粒度有限的问题，本文在 Ultralytics RT-DETR-L 基线上迁移 D-FINE 的 Fine-grained Distribution Refinement（FDR）与 Fine-Grained Localization（FGL）机制，构建 FDR-RTDETR-L。该方法不改变主干网络、混合编码器、Query 选择、Transformer 解码层、分类分支、匈牙利匹配、Top-300 后处理与 NMS 设置，仅将六层解码器中的四维连续框回归头替换为每条边 33 个离散位置、共 132 维输出的分布回归头，并增加一个由原第 0 层回归头复制得到的 preliminary box 分支。各解码层在同一 preliminary box 参考系内累计分布残差，经非均匀 Integral 解码为四边距离，再恢复为边界框；训练阶段复用原始匹配索引，增加 IoU 加权的相邻分箱 FGL 损失与 preliminary box 的 L1/GIoU 监督。因此，当前实际实验臂是“FDR 表示 + FGL + preliminary-box 辅助定位”的完整组合，而不是仅替换输出维度的单一变量。

为排除训练协议漂移，Control 与 FDR 使用相同公共参数初始化、样本顺序、数据增强随机序列、优化器、AMP、验证流程和指标代码。F0--F4 工程与表示门检全部通过。在固定 647 张训练子集、548 张验证集和 seed0 的配对筛选中，FDR 相对 Control 的最终 mAP、尾三轮平均 mAP 和最终 AP75 分别提高 `0.01801`、`0.0156367` 和 `0.0154279`，通过代码冻结的 Gate2。随后 FDR 与 stock Control 均在 6471 张完整训练集上从相同 formal authority fresh 完成 100 epoch。统一 same-evaluator 对 548 张验证图和 38,759 个目标进行独立复评后，FDR 的 Precision、Recall、F1、AP50、AP75 和 mAP50-95 分别较 Control 提高 `10.150/7.546/8.827/9.805/7.951/7.055 pp`；Tiny、Small、Medium、Large 以及十个类别的 mAP、AP50 和 AP75 全部正向。正式结果与完整证据边界见[严格 Control 与完整结果报告](../../docs/FDR_RTDETR_L_STRICT_CONTROL_AND_RESULTS_2026-08-09_ZH.md)。

**关键词：** RT-DETR；D-FINE；细粒度分布回归；边界框定位；VisDrone；小目标检测

---

## 1. 研究问题与方法定位

### 1.1 基线模型的定位表达

Ultralytics RT-DETR-L 使用 300 个 Query，通过六层 Transformer decoder 迭代产生类别分数和边界框。原始边界框分支在每个解码层输出四个连续量，可写为：

\[
\hat B_l=\sigma\left(H_l^{box}(h_l)+\sigma^{-1}(R_l)\right),
\]

其中，\(h_l\) 为第 \(l\) 层 decoder hidden state，\(R_l\) 为当前 reference box，\(H_l^{box}\) 为四维回归 MLP。该表示计算高效，但每个坐标只由单一连续值表达；对于 VisDrone 中大量像素尺度较小、边界模糊或相互遮挡的目标，连续点估计可能缺少对“边界位置不确定性”和“相邻位置竞争关系”的显式建模。

### 1.2 本文方法的边界

本文方法不是完整复现 D-FINE，也不迁移 D-FINE 的 backbone、Hybrid Encoder、decoder gateway、DDF、GO-LSD、LQE、teacher/student 或训练配方。本文只迁移 commit-pinned D-FINE 中与 FDR/FGL 直接相关的公式和结构，并将其嵌入统一的 Ultralytics RT-DETR-L 实验框架。

因此，论文中允许的表述是：

> 在冻结 Ultralytics RT-DETR-L 分类、Query、编码器、匹配和训练协议的条件下，迁移并验证 FDR/FGL 定位回归机制。

不允许写成：

> 本文提出了 D-FINE 的 FDR，或本文完整复现了 D-FINE。

本文可主张的研究贡献主要是：

1. 将官方 FDR/FGL 机制和 preliminary-box 辅助定位迁移到 Ultralytics RT-DETR-L 的独立定位路径，并保持检测器其他关键契约不变；
2. 建立公共参数字节级一致、私有参数随机数隔离、无额外匹配调用的公平配对实现；
3. 建立从公式 golden parity、损失隔离、真实 CUDA 单步到表示覆盖率的多级验证链；
4. 在 VisDrone 固定子集上获得正向配对筛选证据，并启动全数据正式训练。

---

## 2. FDR-RTDETR-L 网络结构

### 2.1 总体结构

方法只修改 RT-DETR head 内的 decoder box path：

```text
Backbone / Hybrid Encoder / Query Selection                （保持 stock）
                         │
                 6-layer Decoder                           （层本身保持 stock）
                         │
          ┌──────────────┴──────────────┐
          │                             │
   Classification Heads           FDR Box Path
       （保持 stock）                    │
                                  preliminary box
                                        │
                          6 × 132-d distribution heads
                                        │
                          cumulative distribution residual
                                        │
                         non-uniform Integral + box decode
                                        │
                                  refined boxes
```

实际集成位置为 [`src/rtdetr_fdr.py`](../../src/rtdetr_fdr.py)：模型首先按 Ultralytics 8.4.90 构建完整 RT-DETR-L，然后保留原 decoder layers 和 score heads，仅替换 `head.decoder` 的 box 更新实现以及 `head.dec_bbox_head`。RT-DETR-L 的 YAML 仍只声明原 `RTDETRDecoder(P3,P4,P5)`，FDR 没有增加新的 backbone、neck 或 YAML feature path。

### 2.2 Preliminary box 分支

在第 0 个 decoder layer 上，使用原始 `dec_bbox_head[0]` 的深拷贝构造四维 preliminary box head。其结构仍是三层 `256→256→256→4` MLP：

\[
B_{pre}=\sigma\left(H_{pre}(h_0)+\sigma^{-1}(R_0)\right).
\]

实现见 [`src/fdr_head.py`](../../src/fdr_head.py) 中的 `FDRDeformableTransformerDecoder`。`H_pre` 的结构和初始公共权重来自 stock 第 0 层 box head，输出仍为 \((c_x,c_y,w,h)\)。得到 \(B_{pre}\) 后，代码使用 `B_pre.detach()` 作为六层分布回归共享的几何参考。这一设计有三个作用：

1. 为四条边的离散距离提供稳定参考系；
2. 将“粗框生成”与“细粒度边界细化”分离；
3. 避免后续 FGL 目标编码通过 reference 路径反向改变 preliminary reference。

Preliminary box 不是无监督中间量。训练时，它复用第一个 decoder prediction group 的 stock 匹配索引，额外计算 stock 权重体系下的 L1 和 GIoU 损失，生成 `loss_bbox_pre` 与 `loss_giou_pre`。由于 preliminary box 在 FDR 几何解码和 FGL 标签编码前均被 detach，最终 decoded-box loss 与 FGL 不会通过参考框路径回传到 pre-head；pre-head 主要由这两个新增的辅助定位损失训练。

### 2.3 六层四边分布回归头

设 `reg_max=32`，则每条边包含 \(32+1=33\) 个分箱，四条边合计输出：

\[
4\times 33=132
\]

个 logits。六个 decoder layer 分别使用一个三层 MLP：

```text
256 → 256 → 256 → 132
```

代码由 [`src/fdr_head.py`](../../src/fdr_head.py) 的 `build_distribution_heads` 构造。六个分布头的私有随机种子固定为 `10000 + experiment_seed`；最后一层 linear 的 weight 和 bias 均初始化为 0，从而使训练开始时各分箱 logits 中性，不给某一边界位置施加人为偏置。

### 2.4 跨层累计分布残差

第 \(l\) 层分布头不独立预测最终结果，而是预测相对于前一层累计 logits 的残差：

\[
\Delta Z_0=H_0^{dist}(h_0),
\]

\[
\Delta Z_l=H_l^{dist}\left(h_l+\operatorname{sg}(h_{l-1})\right),\quad l>0,
\]

\[
Z_l=Z_{l-1}+\Delta Z_l.
\]

其中，\(Z_l\in\mathbb{R}^{B\times Q\times 132}\)，`sg` 表示 stop-gradient。代码中的 `output_detach` 保存上一层 hidden state 的 detached 版本，而 `cumulative_corners` 保存可继续反向传播的累计分布 logits。该设计使后层学习“在现有边界分布上继续修正”，而不是六层分别从头回归。

训练时，解码后的 refined box 在送入下一 decoder layer 作为 reference 前被 detach；推理时则直接使用 refined box。该行为保持 RT-DETR 迭代 reference 更新的梯度边界。

### 2.5 非均匀 Integral 与边界框恢复

每条边的 33 个 logits 经 softmax 得到概率：

\[
p_l^e(k)=\frac{\exp Z_l^e(k)}{\sum_{j=0}^{32}\exp Z_l^e(j)},
\]

其中 \(e\in\{left,top,right,bottom\}\)。随后使用 D-FINE 的非均匀投影向量 \(W(k)\) 计算距离期望：

\[
d_l^e=\sum_{k=0}^{32}p_l^e(k)W(k).
\]

投影向量不是线性等间隔，而是由固定 `reg_scale=4.0` 与 `up=0.5` 生成的非均匀刻度，使靠近零偏移的位置具有更细粒度。投影向量注册为 buffer，不是可训练参数。实现位于 [`src/fdr_math.py`](../../src/fdr_math.py) 的 `weighting_function` 与 `Integral`，其公式机械对齐 D-FINE commit `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6`。

四边距离再相对 \(B_{pre}\) 经 `distance2bbox` 解码：

\[
B_l=\mathcal{D}\left(B_{pre},d_l;\,reg\_scale=4\right).
\]

最终只将第六层 box 和 stock classification logits 交给 Ultralytics 原后处理。预测接口、类别分数、Query-by-class Top-300、`max_det=300` 和 `NMS=False` 均不变；FGL logits、reference 和 preliminary box 不进入部署输出。当前 `distance2bbox` 直接产生 `cxcywh`，不再经过 stock box path 的末端 sigmoid，也不额外 clamp，因此理论上可产生超出 `[0,1]` 的中间 refined box；F2/F3 已验证正常、空目标、边界目标和真实 CUDA batch 下输出有限，但最终论文仍应报告边界样本统计。

### 2.6 关键张量与训练/推理路径

设 batch 为 \(B\)，正常 Query 数为 \(N=300\)，denoising Query 数为 \(D\)，则训练时总 Query 数 \(Q=N+D\)。主要张量为：

| 张量 | 形状 |
|---|---|
| 单层 decoder hidden state | `[B,Q,256]` |
| preliminary box | `[B,Q,4]` |
| 单层 distribution logits | `[B,Q,132]` |
| 六层累计 corner logits | `[6,B,Q,132]` |
| 六层 decoded boxes | `[6,B,Q,4]` |
| 六层 classification logits | `[6,B,Q,10]` |
| DN 切分后的正常部分 | corners `[6,B,300,132]`，pre-box `[B,300,4]` |
| 推理后处理结果 | `[B,300,6]` |

训练时保留 encoder prediction、六层 decoder prediction 和 DN prediction，并向 criterion 传入 FDR evidence；推理时仅保留 `eval_idx=5` 的最终分布解码框和 stock 分类分数。因此，FGL 和 preliminary-box 辅助监督只增加训练期计算，不增加部署输出字段。

---

## 3. 损失函数与匹配隔离

### 3.1 Stock 损失完全保留

基础 criterion 仍为 Ultralytics 8.4.90 `RTDETRDetectionLoss`，保留：

- Varifocal classification loss；
- L1 box loss，权重 5.0；
- GIoU loss，权重 2.0；
- encoder prediction；
- 六层 decoder 主/辅助损失；
- denoising prediction 与对应辅助损失；
- 原 Hungarian matcher 的 cost、调用顺序和一对一分配。

实现类 [`src/fdr_loss.py`](../../src/fdr_loss.py) 中的 `FDRDetectionLoss` 先调用 `super().forward(...)` 完成完整 stock loss，再消费已经记录的 assignment。FGL 不执行第二次 matcher，也不构造跨层匹配并集。

### 3.2 Fine-Grained Localization 损失

对每个 stock-matched prediction/GT pair，首先以 detached preliminary box 为参考，通过 `bbox2distance` 将 GT 的四边连续距离编码到两个相邻非均匀分箱。设目标位于分箱 \(k\) 与 \(k+1\) 之间，对应线性插值权重为 \(w_l,w_r\)，则单边 FGL 为：

\[
L_{FGL}^{e}=-q\left[w_l\log p^e(k)+w_r\log p^e(k+1)\right],
\]

其中 \(q=\operatorname{sg}(IoU(\hat B,B^{gt}))\) 为 detached matched IoU。四边损失按正样本数归一化。主层、五个辅助层以及存在时的 denoising layers 均使用各自的 stock assignment。

FGL 权重固定为：

\[
\lambda_{FGL}=0.15.
\]

### 3.3 实际总损失

为准确反映当前代码，实际优化目标应写为：

\[
L_{total}=L_{stock}+0.15L_{FGL}^{main}+0.15\sum_{l=0}^{4}L_{FGL,l}^{aux}
+L_{pre}^{L1}+L_{pre}^{GIoU}+L_{DN,FDR},
\]

其中，\(L_{stock}\) 已包含 stock encoder、decoder、auxiliary 和 denoising 的 VFL/L1/GIoU；\(L_{DN,FDR}\) 表示存在 denoising metadata 时的 FGL 与 preliminary-box DN 项。`L_pre` 复用 stock 的 L1/GIoU 权重，不新增 matcher。

这比简单写成“stock loss + 0.15 FGL”更完整，因为当前生产实现确实启用了 `supervise_pre_boxes=True`。据此，当前 30-epoch 正增益只能归因于完整方法臂，不能在缺少消融的情况下单独归因为分布表示、FGL 或 pre-box 辅助监督中的任一项。

---

## 4. 与 baseline 的修改边界

| 组件 | Stock RT-DETR-L | FDR-RTDETR-L | 是否保持一致 |
|---|---|---|---|
| Backbone | Ultralytics RT-DETR-L | 原样复用 | 是 |
| Hybrid Encoder / feature projection | Stock | 原样复用 | 是 |
| Query selection | Ultralytics 8.4.90 stock encoder Top-K selection | 原样复用 | 是 |
| Query 数 | 300 | 300 | 是 |
| Decoder attention / FFN layers | 6 层 stock layers | 复用同一组 layers | 是 |
| Classification heads | 每层 stock score head | 原样复用 | 是 |
| Box head | 6 个 `256→256→256→4` 连续回归 MLP | 1 个 `256→256→256→4` preliminary head + 6 个 `256→256→256→132` 分布 MLP | **否，核心改动** |
| Box reference 更新 | 连续框迭代 | FDR 解码框迭代，训练 reference detach | 表示改变，接口一致 |
| Hungarian matcher | Stock | Stock，FGL 复用索引 | 是 |
| VFL/L1/GIoU | Stock | 完整保留 | 是 |
| 新增监督 | 无 | FGL + preliminary L1/GIoU | **否，核心改动** |
| 梯度裁剪 | 全体参数单组，max norm=10 | 公共参数与 FDR 私有 box-path 参数分别按 max norm=10 裁剪 | **分组不同，阈值相同** |
| Denoising builder | Stock | 原样复用，并对 DN box 增加隔离 FDR 监督 | 构造一致 |
| Top-K / max_det | Query-class Top-300 / 300 | 原样复用 | 是 |
| NMS | False | False | 是 |
| 推理输出 schema | boxes/scores/classes | 不增加输出字段 | 是 |
| DDF / GO-LSD / LQE / teacher | 无 | 明确排除 | 是 |
| Boundary / trajectory / LPR / OAR | 无 | 明确不混入 | 是 |

---

## 5. 公平性与统一实验协议

### 5.1 环境和数据

| 项目 | Control 与 FDR 统一值 |
|---|---|
| 基础模型 | Ultralytics RT-DETR-L |
| Ultralytics | 8.4.90 |
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| 驱动 | 550.142 |
| Python | 3.10.12 |
| PyTorch / Torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| CUDA | 12.1 |
| 数据集 | 同一份 VisDrone train/val |
| 训练 / 验证图片 | 6471 / 548 |
| 类别数 | 10 |
| 数据集 SHA256 | `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB` |
| 固定 10% 子集 | 647 张 |
| 子集 SHA256 | `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0` |

### 5.2 训练与增强参数

| 参数 | 统一值 |
|---|---:|
| 初始化 | `pretrained=False`，从零训练 |
| seed | 0 |
| imgsz / batch / workers | 640 / 8 / 8 |
| device | 0，单卡 |
| AMP | True，固定 scale 128 |
| deterministic / cache | True / False |
| optimizer | MuSGD |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight decay | 0.937 / 0.0005 |
| warmup epochs / momentum / bias lr | 3.0 / 0.8 / 0.0 |
| nbs / cos_lr | 64 / False |
| mosaic / close_mosaic | 1.0 / 10 |
| mixup / cutmix / copy_paste | 0 / 0 / 0 |
| scale / translate | 0.5 / 0.1 |
| degrees / shear / perspective | 0 / 0 / 0 |
| flipud / fliplr | 0 / 0.5 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 |
| query / max_det / NMS | 300 / 300 / False |
| gradient clipping | max norm 10；control 单组，FDR 公共/私有参数分组 |

除网络私有层和相应辅助损失外，两臂还要求以下证据一致：

- 同一公共参数状态，公共 tensor 字节级 SHA256 相同；
- 同一样本顺序与数据增强随机序列；
- 相同类别映射、验证预处理与指标代码；
- 相同 checkpoint/resume 规则；
- 相同训练子集和验证集；
- 相同公共 optimizer 参数覆盖；
- FDR 私有参数在 `torch.random.fork_rng` 中以独立 seed 初始化，不推进公共 RNG。

F3 实测中，control 与 FDR 的公共参数 SHA256 均为：

```text
208E36E5943899C8E89A833283875575EDD8034E696FECED6B1A5F6170F98533
```

数据顺序 SHA256 均为固定子集哈希 `52660F...D93A0`。因此，30-epoch screen 的差异被限制为 FDR box path、其私有监督及与私有参数对应的分组裁剪；基础检测器、公共参数、数据与训练超参数保持一致。

### 5.3 Screen 与 formal 的关系

Screen 使用固定 647 张子集、共同 50-epoch schedule，并在配对评估前以代码冻结的第 30 epoch 截止比较。它不是把一个事后挑选的“50轮实验前30轮”冒充最终结果，而是两臂从启动前就共同冻结 `schedule=50, cutoff=30`。由于 `close_mosaic=10` 是相对 50-epoch schedule 计算的，mosaic 关闭发生在约 epoch40，晚于 screen cutoff；因此该 screen 只检验早期学习能力。Formal 则重新从相同 seed0 正式初始状态启动完整 6471 张训练集、100 epoch；它不 resume screen checkpoint。

### 5.4 “baseline 对齐”的两级定义

本项目需要区分两种 baseline 证据：

1. **30-epoch 配对 control：严格字段对齐。** 它与 FDR 由同一 source、protocol、initial-state、子集、样本顺序、增强随机序列和 evaluator 产生，适合判断当前结构臂能否进入正式训练。
2. **历史 100-epoch matched baseline authority：指标已冻结，但协议文档仍需最终复核。** 用户当前重新冻结的正式协议是本节列出的 MuSGD、`warmup_bias_lr=0.0`、固定 AMP scale 128 等字段；仓库中更早的 baseline 文档仍存在 `optimizer=auto`、不同 warmup bias learning rate 或动态 AMP 等旧字段，不能单独用这些旧文档证明与当前 FDR formal 字段完全等价。

因此，本报告可以确认“当前 paired screen 的 control/FDR 严格对齐”，但在 100 epoch 结束前不会无条件宣称“任意历史 baseline run 与当前 FDR formal 完全字段同源”。最终主表将先完成 checkpoint、协议 authority 和 evaluator 的再次核对，再计算正式差值。

---

## 6. 工程验证与表示验证

### 6.1 当前专项测试

在当前代码分支上执行 FDR authority、math、head、loss、protocol、preflight、runtime integration、trainer CLI 和 Gate evaluator 专项测试，结果为：

```text
127 passed, 3 skipped in 35.41s
```

跳过项为环境依赖测试，不是失败项。除此之外，Gate evaluator 在服务器不可变 source 上独立通过 7 项测试；GitHub 同步修复相关测试通过 24 项。

### 6.2 F0--F4 预训练门检

| Gate | 验证内容 | 实际结果 |
|---|---|---|
| F0 | D-FINE 固定 commit、weighting、Integral、box transform、FGL golden parity | 全部通过；上游 commit 为 `7fe2f888...64ae6` |
| F1 | neutral encode/decode、累计残差、`FGL weight=0` stock loss exact、分类/matcher/Top300/NMS 隔离 | 7 项全部通过 |
| F2 | normal/DN/aux、空 GT、混合空 GT、边界 clipping、finite forward/backward、AMP128 | 通过；shape 为 boxes `[6,8,300,4]`、scores `[6,8,300,10]`、logits `[6,8,300,132]`，AMP skipped step=0 |
| F3 | RTX4090 真实 VisDrone batch8 forward/backward/MuSGD/validation/checkpoint | 全部通过；梯度有限、optimizer step=1、unexpected trainable parameters=0 |
| F4 | GT offset 表示 oracle、重建误差、分箱饱和率、尺度分层 | 通过；35,246 matched targets，0 invalid/nonfinite |

上述结果的原始报告已冻结于 [`live-snapshot-epoch0010/fdr-preflight`](./evidence/d97e1eb7/live-snapshot-epoch0010/fdr-preflight)，总判定见 [`decision.json`](./evidence/d97e1eb7/live-snapshot-epoch0010/fdr-preflight/d97e1eb7-seed0-attempt001/decision.json)。

F4 的详细数据如下：

| 指标 | 数值 |
|---|---:|
| 匹配目标数 | 35,246 |
| 全体 reconstruction L1 | 0.0009513123 |
| 全体 reconstruction max | 0.7069315 |
| 未饱和样本数 | 30,726 |
| 未饱和 reconstruction L1 | `6.2766e-09` |
| 未饱和 reconstruction max | `5.9605e-08` |
| 总 edge saturation rate | 6.44399% |
| left / top / right / bottom saturation | 7.42496% / 5.41054% / 7.26891% / 5.67157% |
| tiny saturation | 9.77196% |
| small saturation | 2.03930% |
| other saturation | 0.87483% |

未饱和区域几乎达到 float32 数值精度，说明公式迁移和 decode parity 正确；全体 max error 主要来自表示范围饱和，而不是公式实现错误。Tiny 目标饱和率更高，意味着当前 `reg_max=32, reg_scale=4` 在极端小框上仍存在表示覆盖限制。该项应作为论文局限保留，不能隐藏。

---

## 7. 固定 10% 子集的 30-epoch 配对筛选

### 7.1 Gate2 判定规则

Gate2 的阈值在配对评估前写入 evaluator 并冻结，评估后未事后放宽。三个条件必须同时满足：

1. 第 30 epoch 的 mAP(FDR) - mAP(control) > 0；
2. 第 28--30 epoch 平均 mAP 的差值 > 0；
3. 第 30 epoch 的 AP75 差值 > 0。

评估器还要求两臂各有连续 30 条证据、JSONL 与 `results.csv` 一致、source/protocol/initial-state/data/seed/stage authority 严格配对。所有工程检查通过后才允许计算科学 Gate。

### 7.2 最终轮结果

| 指标 | Control epoch30 | FDR epoch30 | FDR - Control |
|---|---:|---:|---:|
| Precision | 0.00662 | 0.07229 | +0.06567 |
| Recall | 0.02501 | 0.13717 | +0.11216 |
| mAP50 | 0.00125 | 0.04000 | +0.03875（非 Gate 条件） |
| mAP50-95 | 0.00026 | 0.01827 | **+0.01801** |
| AP75 | 0.0000304090 | 0.0154582695 | **+0.0154278605** |

### 7.3 尾三轮与最佳轮

| 项目 | Control | FDR | 差值 |
|---|---:|---:|---:|
| epoch28--30 mean mAP | 0.0009033333 | 0.01654 | **+0.0156366667** |
| 独立最佳 mAP | 0.00143（epoch29） | 0.01827（epoch30） | +0.01684 |
| 尾三轮 mean AP75 | 0.0002738164 | 0.0142321482 | +0.0139583318 |

三个代码冻结条件全部为正，因此：

```text
engineering_complete = true
Gate2 passed = true
formal_eligible = true
```

### 7.4 对筛选结果的正确解释

该结果证明的是：在同一固定子集、同一训练随机序列和相同协议下，当前完整方法臂在这一轮 seed0 screen 中比 stock control 更容易形成可检测的定位结果，且收益同时出现在 mAP 与 AP75，而非仅表现为回归 loss 下降。

但 control 的绝对 mAP 很低并且尾部波动明显，说明从零训练 RT-DETR-L 在 647 张子集上的方差较大。Gate evaluator 的 `engineering.complete=true` 只表示两臂各有连续 30 条、有限且相互一致的机器证据以及配对 authority，不代表已完成多 seed、远端下载重哈希或最终统计显著性审计。因此，`+1.801 pp` 不能直接写成最终 100-epoch baseline 提升，也不能据此推断多 seed 稳定性；它也不能被单独归因给 FDR、FGL 或 pre-box 监督。它只具有筛选意义，这也是 formal 必须在全数据上 fresh start 的原因。

---

## 8. 全数据 100-epoch 正式实验与严格 Control

### 8.1 FDR formal100 完成状态

Gate2 通过后，FDR 正式实验使用：

```text
stage: formal
variant: FDR
train images: 6471
val images: 548
seed: 0
epochs: 100
initialization: shared formal seed0 initial state
resume from screen: false
```

FDR 已完成 `100/100` epoch。epoch98--100 的训练日志 mAP50-95 为 `0.29007/0.28996/0.28971`；epoch100 checkpoint 为 `fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt`，SHA-256 为 `C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2`，大小为 200,024,985 bytes。100 轮轻量结果位于远端 `training-results` 分支，epoch98--100 重 checkpoint 位于 GitHub Release [`fdr-formal-d97e1eb7-live`](https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-d97e1eb7-live)。

### 8.2 严格 stock Control 重跑

此前历史 matched baseline 与 FDR formal 在 optimizer、warmup bias lr 和 initial-state authority 上存在差异，因此只适合作为早期参考。随后使用 FDR formal authority fresh 启动 stock Control，保持数据、seed、公共初始化、MuSGD、batch、AMP、增强、验证和 checkpoint 规则一致；Control 不继承 FDR 或 Screen checkpoint。

严格 Control 已完成 `100/100` epoch。公开 Release 保存 epoch98--100 checkpoint 和 manifest，其中 epoch100 checkpoint SHA-256 为 `9C242711F44B7E68B360AF904AB7C44F64505C7136B7E7F90481092AE3308AF7`，大小为 197,665,100 bytes。证据位于 [`fdr-formal-control-d97e1eb7-live`](https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-control-d97e1eb7-live)。当前不需要再次重跑 seed0 Control；若补多 seed，必须成对补 Control/FDR。

### 8.3 统一独立复评正式结果

两臂在 548 张 val、38,759 个目标、`imgsz=640`、`batch=8`、`conf=0.001`、`NMS=false`、`max_det=300` 下由同一 evaluator 复评：

| 指标 | 严格 Control | FDR | FDR - Control |
|---|---:|---:|---:|
| Precision | 0.46761 | 0.56911 | **+10.150 pp** |
| Recall | 0.41731 | 0.49278 | **+7.546 pp** |
| F1 | 0.43657 | 0.52484 | **+8.827 pp** |
| AP50 | 0.38663 | 0.48468 | **+9.805 pp** |
| AP75 | 0.21302 | 0.29253 | **+7.951 pp** |
| mAP50-95 | 0.21911 | 0.28966 | **+7.055 pp** |

Tiny、Small、Medium、Large mAP 分别提高 `+5.795/+7.214/+7.130/+6.786 pp`；十个类别的 mAP、AP50 和 AP75 均严格正向。完整分尺度、分类别表、Control checkpoint 状态和 SHA-256 见[严格 Control 与完整结果报告](../../docs/FDR_RTDETR_L_STRICT_CONTROL_AND_RESULTS_2026-08-09_ZH.md)。

统一复评 JSON 的 SHA-256 为 `8FFD439C4C48044C0D1937019CE58DDB857CE8FCED64C0082CBC28EDD44333E8`。该 JSON 仍标记 `preliminary_same_evaluator`；复评实际使用的 `last.pt` 与 rolling epoch100 asset 文件哈希不同。最终投稿前应上传 exact `last.pt` 或发布 model/EMA tensor equality 报告，但这一剩余工作是 artifact identity 封口，不是重新训练 strict Control。

---

## 9. 参数量、GFLOPs 与推理开销

在相同 `rtdetr-l.yaml, nc=10, imgsz=640` 下完成静态审计：

| 指标 | Stock RT-DETR-L | FDR-RTDETR-L | 增量 | 相对增幅 |
|---|---:|---:|---:|---:|
| Parameters | 32,826,626 | 33,156,614 | +329,988 | **+1.00524%** |
| GFLOPs | 108.0318976 | 108.2291200 | +0.1972224 | **+0.18256%** |

参数增量主要来自六个 `256→256→256→132` 分布头以及额外 preliminary head；同时原六个四维 decoder box heads 被替换，因此表中为净增量。

需要注意：最初提出的“参数量增幅 <1%”严格阈值被超出约 0.00524 个百分点。用户已允许放宽该限制，因此不影响结论，但论文中不能写成“参数增幅低于1%”。GFLOPs 增幅仍明显低于 1%。端到端 latency/FPS 尚未形成与正式主表同等级的冻结证据，不能提前声称 `<3%`；应以同一 RTX 4090、相同 batch、warmup、同步和统计口径补测。

30-epoch screen 的累计训练/验证耗时和 CUDA 峰值显存只能作为工程参考，不能代替推理延迟：

| 工程指标 | Control | FDR | 差值 |
|---|---:|---:|---:|
| 累计 screen wall time | 1425.63 s | 1627.05 s | +201.42 s（+14.13%） |
| CUDA peak memory | 10,715.81 MiB | 10,793.08 MiB | +77.27 MiB |

训练开销增幅高于静态 GFLOPs 增幅，原因可能包括六层 132 维 logits、FGL 目标编码、额外辅助损失和证据记录；该解释尚需 profiler 验证。

---

## 10. 有效性威胁与必要消融

当前实验设计已经控制公共初始化、数据顺序和主要超参数，但仍存在以下必须公开的有效性边界：

1. **组件归因尚未分离。** 当前方法同时改变分布表示、加入 FGL、加入 preliminary-box L1/GIoU，并对 FDR 私有参数独立裁剪；正式 `+7.055 pp` 不能只归因给其中一项。
2. **筛选 control 地板过低。** epoch30 control mAP 仅为 0.00026，导致 screen 适合做候选淘汰，不适合估计最终效应量。
3. **只有 seed0。** 按当前决策不运行 seed2，也未完成 seed1，因此不能报告跨 seed 均值、方差或显著性。
4. **子集并非分层抽样。** 固定 647 张子集由确定性 SHA 规则生成，不保证类别、tiny/small 比例与完整训练集严格同分布。
5. **表示范围在 tiny 上更紧张。** F4 的 tiny edge saturation 为 9.77%，显著高于 small 和 other；这可能限制极小目标的最终收益。
6. **Artifact identity 尚未最终封口。** 统一复评实际使用的 `last.pt` 与 rolling epoch100 Release 文件哈希不同；在 exact checkpoint 或 tensor equality JSON 上传前，应保留这一审计说明。

正式 100-epoch 结果已经为正，论文仍应补充以下消融，才能回答贡献来源：

| 消融臂 | 目的 |
|---|---|
| Stock RT-DETR-L | 正式对照 |
| Stock + preliminary-box supervision | 分离辅助粗框监督贡献 |
| FDR representation，`fgl_weight=0`，无 pre-box supervision | 分离分布表示贡献 |
| FDR + FGL，无 pre-box supervision | 测量 FGL 增量 |
| FDR + FGL + preliminary-box supervision | 当前完整方法 |

这些消融不改变当前 100-epoch 主线，但决定论文能否把收益正确归因到具体机制。

---

## 11. 当前能够成立与不能成立的论文结论

### 11.1 已有证据支持的结论

1. FDR/FGL 数学实现与固定 D-FINE commit 的 golden reference 一致；
2. 模型改动集中在 decoder box representation、FGL/pre-box 监督与对应私有梯度组，不改变分类、Query、encoder、matcher 和 postprocess；
3. FGL 与 preliminary supervision 复用 stock assignment，没有第二次匹配；
4. 30-epoch paired control/FDR 的公共初始状态和数据顺序哈希一致；
5. F0--F4、真实 RTX4090 单步和专项回归测试均通过；
6. 固定 10% 子集 seed0 的 30-epoch 配对 Gate2 通过；
7. 全数据 seed0 FDR 与同 formal authority stock Control 均已完成 100 epoch；
8. 统一独立复评中，FDR 的 mAP50-95 从 `0.21911` 提高到 `0.28966`，提升 `+7.055 pp`；
9. Precision、Recall、F1、AP50、AP75、四尺度 mAP/AP50/AP75 和十类别 mAP/AP50/AP75 均严格正向；
10. FDR 和 strict Control epoch100 checkpoint、统一复评 JSON 与 SHA-256 已上传 GitHub Release。

### 11.2 目前不能写入论文结论的内容

1. 不能把历史跨-authority baseline 的 `+4.801 pp` 当成正式主结果；正式主结果是 strict Control 下的 `+7.055 pp`；
2. 不能把 30-epoch 子集增益 `+0.01801` 写成 full-data 100-epoch 效应量；
3. 不能宣称多 seed 稳定或统计显著，因为当前严格主结果只有 seed0；
4. 不能宣称参数增幅 `<1%`；实际为 `+1.00524%`；
5. 不能宣称端到端延迟增幅 `<3%`，因为最终 latency audit 尚未冻结；
6. 不能把 D-FINE 的 FDR/FGL 基础公式归为本文首创；本文贡献是受控迁移、隔离集成、YAML 声明和 VisDrone 验证；
7. 不能把 F4 的 matched-IoU/representation 指标当成 detector mAP；
8. 不能在缺少消融时把当前收益单独归因给 FDR、FGL 或 preliminary-box supervision；
9. 不能声称 formal 1–100 每轮重 checkpoint 都已公开；
10. 不能声称统一复评 checkpoint identity 已完全封口。

---

## 12. 可复现实现索引

| 内容 | 证据位置 |
|---|---|
| 方法设计与禁止项 | [`docs/superpowers/specs/2026-08-04-ultralytics-fdr-only-design.md`](../../docs/superpowers/specs/2026-08-04-ultralytics-fdr-only-design.md) |
| 固定 D-FINE authority | [`third_party/dfine_7fe2f888/AUTHORITY.json`](../../third_party/dfine_7fe2f888/AUTHORITY.json) |
| 非均匀 weighting / Integral / box transform / FGL primitive | [`src/fdr_math.py`](../../src/fdr_math.py) |
| preliminary head、六层分布头与累计解码 | [`src/fdr_head.py`](../../src/fdr_head.py) |
| Stock loss 隔离、assignment 复用、FGL/pre loss | [`src/fdr_loss.py`](../../src/fdr_loss.py) |
| Ultralytics model/trainer 集成 | [`src/rtdetr_fdr.py`](../../src/rtdetr_fdr.py) |
| 冻结协议、公共/私有参数 authority | [`src/fdr_protocol.py`](../../src/fdr_protocol.py) |
| 固定训练与逐 epoch 证据 | [`scripts/train_rtdetr_fdr.py`](../../scripts/train_rtdetr_fdr.py) |
| Gate2 evaluator | [`scripts/evaluate_fdr_gate.py`](../../scripts/evaluate_fdr_gate.py) |
| 30-epoch 完整配对证据 | [`research/fdr/evidence/d97e1eb7`](./evidence/d97e1eb7) |
| Gate2 machine-readable report | [`gate2.json`](./evidence/d97e1eb7/fdr-gate-d97e1eb7/gate2.json) |
| F0--F4 与 formal epoch10 冻结快照 | [`live-snapshot-epoch0010`](./evidence/d97e1eb7/live-snapshot-epoch0010) |
| 快照来源与 SHA256 | [`live-snapshot-epoch0010/README.md`](./evidence/d97e1eb7/live-snapshot-epoch0010/README.md) |
| Formal epoch12 运行状态更新 | [`runtime-update-epoch0012`](./evidence/d97e1eb7/runtime-update-epoch0012) |
| FDR formal100 训练结果 | [`training-results/results/fdr-formal-d97e1eb7-seed0-fdr`](https://github.com/kkc236/uav-detection-baselines/tree/training-results/results/fdr-formal-d97e1eb7-seed0-fdr) |
| FDR epoch100 checkpoint | [`fdr-formal-d97e1eb7-live`](https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-d97e1eb7-live) |
| strict Control 与统一复评 | [`fdr-formal-control-d97e1eb7-live`](https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-control-d97e1eb7-live) |
| 完整正式结果报告 | [`FDR_RTDETR_L_STRICT_CONTROL_AND_RESULTS_2026-08-09_ZH.md`](../../docs/FDR_RTDETR_L_STRICT_CONTROL_AND_RESULTS_2026-08-09_ZH.md) |

关键版本：

```text
D-FINE upstream authority: 7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6
formal model/training authority: d97e1eb7f98414752a1c1f38287697db3f2a0679
Gate2 evaluator authority: 1cc64045560b69a07b9a8699019bc02fe298c488
screen evidence commit: d6b0b3e1
```

---

## 13. 正式结论

当前结果表明，FDR 主线完整方法是近期候选中第一个同时完成“官方机制对齐、工程门检通过、严格配对 Screen30 正向、FDR/stock Control 两臂 full-data 100 epoch 完成、统一独立复评正向”的定位回归方案。其主要优势不是增加新的特征融合或 Query 机制，而是在保持 RT-DETR 检测骨架不变的情况下，将四维点回归替换为可逐层累计的四边分布表示，并以 FGL 和 preliminary-box 辅助监督训练该表示，使 Decoder 能以更细粒度学习目标边界。

最有价值的证据已经从早期筛选升级为正式结果：统一 same-evaluator 下，FDR 相对 strict Control 的 mAP50-95 提升 `+7.055 pp`，AP50/AP75、四个尺度和十个类别全部正向。因此可以表述为“FDR 在当前 seed0 严格协议中取得正式性能提升”。仍需保留的边界是：当前只有 seed0；基础 FDR/FGL 机制来自 D-FINE；单变量消融、最终 latency 和 exact-checkpoint tensor identity 尚未闭环。后续工作的重点不是再次重跑 seed0 Control，而是完成这些论文证据封口，并要求新增模块直接以 FDR 为强 Control。

## 参考文献

1. Lv W, Zhao Y, Chang Q, et al. RT-DETR: DETRs Beat YOLOs on Real-time Object Detection. 2023.
2. Peng Y, et al. D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement. ICLR 2025. 本项目迁移 authority 固定为官方仓库 commit `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6`。
