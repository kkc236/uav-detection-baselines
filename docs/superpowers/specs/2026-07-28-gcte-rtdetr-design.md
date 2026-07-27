# GCTE-RTDETR 成功率优先设计

日期：2026-07-28  
状态：冻结候选，等待实现  
目标：把已验证有效的全局—局部多视图检测证据重构为可训练网络模块，并用最短实验路径判断创新点 1 是否值得进入正式训练。

## 1. 决策

停止当前 GCMV-EI 的 P3 特征注入路线，不再围绕 `gamma`、PEG 可靠性乘积或辅助损失权重继续调参。

新路线命名为：

**GCTE-RTDETR：Geometry-Canonical Tiny-Expert RT-DETR**

中文：

**几何规范化局部微小目标专家 RT-DETR**

论文和模型配置中只暴露一个顶层网络模块：

**GCTENetworkModule**

该模块内部包含三个参与前向传播和训练更新的可消融阶段：

1. DSTE：Detection-Supervised Tiny Expert；
2. GCQL：Geometry-Canonical Query Lifter；
3. APFG：Anchored Protected Fusion Gate。

三个内部阶段共同组成一个完整的创新点 1，不作为三个平行创新点拆写。GCTENetworkModule 不改变全局 RT-DETR-L 的检测能力，局部路径只负责增加可靠的 tiny 检测证据。

顶层代码接口冻结为：

```python
class GCTENetworkModule(nn.Module):
    def __init__(
        self,
        query_dim: int = 256,
        num_classes: int = 10,
        num_views: int = 4,
        max_local_queries_per_view: int = 64,
        adapter_ratio: float = 0.5,
        residual_cap: float = 0.2,
    ) -> None:
        super().__init__()
        self.dste = DetectionSupervisedTinyExpert(...)
        self.gcql = GeometryCanonicalQueryLifter(...)
        self.apfg = AnchoredProtectedFusionGate(...)

    def forward(
        self,
        global_queries,
        global_logits,
        global_boxes,
        local_queries,
        local_logits,
        local_boxes,
        crop_geometry,
        *,
        targets=None,
        learned_residual_enabled: bool = True,
    ):
        ...
```

顶层输出冻结为：

- `unified_predictions`：送入统一评估器的最终预测；
- `local_predictions`：DSTE 的局部检测结果；
- `canonical_queries`：GCQL 的全局坐标查询；
- `gate_outputs`：APFG 的锚点、残差和最终准入；
- `losses`：DSTE、GCQL 和 APFG 的分项训练损失；
- `diagnostics`：保护不变量、候选数量、延迟和门控统计。

配置文件只声明一个整体：

```yaml
gcte:
  module: GCTENetworkModule
  query_dim: 256
  num_classes: 10
  num_views: 4
  max_local_queries_per_view: 64
  adapter_ratio: 0.5
  residual_cap: 0.2
  internal_stages:
    dste: true
    gcql: true
    apfg: true
```

普通 Ultralytics YAML 不能独立表达 query、crop geometry 和候选集合接口，因此由 RT-DETR wrapper 在 decoder 输出边界显式调用这个顶层模块；YAML 负责冻结结构配置和消融开关。

## 2. 设计依据

### 2.1 已成立的正证据

同一个 100-epoch RT-DETR-L checkpoint 进行一次全图和四次局部视图推理后，SADED-SM 相比全局 Arm-A 得到：

| 指标 | Arm-A | SADED-SM | Delta |
|---|---:|---:|---:|
| mAP50-95 | 0.180621 | 0.206470 | +0.025849 |
| AP-tiny-SBR | 0.071057 | 0.110251 | +0.039194 |
| tiny recall | 0.553748 | 0.655526 | +0.101778 |
| AP75 | 0.166666 | 0.186902 | +0.020237 |
| AP-large-SBR | 0.145847 | 0.143938 | -0.001909 |

这证明局部高分辨率视图能够产生检测级的有效 tiny 证据。

### 2.2 GCMV-EI 已暴露的失败机制

成熟 baseline 诊断中：

- Method-On − Method-Off 的 AP-tiny 为 `-0.000002`；
- Method-On − Method-Off 的 mAP50-95 为 `+0.000014`；
- `gamma=0.020734`，接近初始值 `0.02`；
- 有效 gate 均值为 `0.008182`；
- 平均残差尺度约为 `gamma × gate ≈ 0.00017`；
- Method-Off − Control 的 AP-large 为 `-0.020910`，说明辅助训练路径扰动了全局检测器。

因此新设计必须满足：

1. 局部证据接受真实检测损失，不再以 tiny heatmap 作为主要代理；
2. 全局检测器在 G0 阶段完全冻结；
3. 全局非 tiny 预测具有结构性保护，学习模块无权删除；
4. 学习路径的初始行为锚定到已经成功的 SADED 规则，而不是从“关闭局部证据”开始；
5. 不在 P3 上混合全局和局部特征。

## 3. 总体架构

```text
                               Input Image
                                    |
                 ---------------------------------------
                 |                                     |
                 v                                     v
          Global View 640                    Four Local Views 1088
                 |                                     |
                 v                                     v
       Frozen RT-DETR-L Global Path          Shared Frozen RT-DETR Extractor
                 |                                     |
        Global decoder queries                         v
        Global boxes/scores                    Local decoder evidence
                 |                                     |
                 |                                     v
                 |                          DSTE: tiny expert adapter
                 |                                     |
                 |                          Local tiny queries/boxes
                 |                                     |
                 |                                     v
                 |                          GCQL: geometry lifting
                 |                                     |
                 |                          Canonical global queries
                 |                                     |
                 ------------------- APFG --------------
                                    |
                                    v
                         Protected unified prediction set
```

全局分支是不可破坏的基线。局部路径只能：

- 增加被证明可靠的 tiny 候选；
- 校准局部候选分数；
- 替换全局路径中低质量的 tiny 候选；
- 不能删除受保护的 global non-tiny 候选。

从模型结构图角度，创新点 1 只画成一个大模块：

```text
Global decoder state ───────────────┐
                                    │
Local decoder states ───────┐       │
Crop geometry ──────────────┼───────┼──> GCTENetworkModule
                            │       │       ├── DSTE
                            │       │       ├── GCQL
                            │       │       └── APFG
                            │       │
                            └───────┘       └──> Unified predictions
```

模块必须能够通过一个顶层 `enabled` 开关整体关闭，并通过三个内部开关完成消融。整体关闭时，输出必须逐预测恢复 global baseline。

## 4. M1：DSTE

全称：

**Detection-Supervised Tiny Expert**

中文：

**检测监督局部微小目标专家**

### 4.1 输入

每个局部视图产生的 decoder query 表示、分类 logits、边框和定位质量：

\[
Q_v,\quad P_v^{cls},\quad B_v,\quad U_v.
\]

局部检测器来自同一个成熟 RT-DETR-L，不训练第二个完整模型。

### 4.2 可训练结构

在局部 query 上增加轻量残差专家：

```text
Local query 256
    -> LayerNorm
    -> Linear 256→128
    -> SiLU
    -> Linear 128→256
    -> bounded residual
```

输出：

\[
\widehat Q_v
=
Q_v+\alpha_a\tanh(A(Q_v)),
\qquad
0\le\alpha_a\le0.2.
\]

分类、边框和质量头使用全局 detector 对应头初始化，并允许局部专家副本更新。

### 4.3 直接监督

局部 crop GT 经过截断可见性过滤后，直接使用：

- RT-DETR 分类损失；
- L1 box loss；
- GIoU loss；
- 查询定位质量损失。

不以 tiny heatmap 代替检测监督。

### 4.4 tiny 范围

所有尺度判断按当前网络输入坐标计算：

\[
s=\sqrt{wh}.
\]

主 tiny 阈值冻结为：

\[
s\le16.
\]

局部专家可以学习更大尺寸目标以获得上下文，但只有映射回全局后满足 tiny 准入条件的候选能进入 APFG。

## 5. M2：GCQL

全称：

**Geometry-Canonical Query Lifter**

中文：

**几何规范化查询提升模块**

### 5.1 确定性坐标映射

局部预测框先通过已知 crop 变换映射到原图，再映射到全局检测坐标：

\[
B_v^g=T_v(B_v^l).
\]

映射必须通过现有 SADED 坐标契约和合成脉冲测试。

### 5.2 可学习几何编码

为每个局部查询构造几何描述：

\[
z_v=
[x_v,y_v,w_v,h_v,
s_v,
d_v^{edge},
r_v^{visible},
e_v^{view}].
\]

其中：

- `s_v`：局部到全局缩放倍率；
- `d_v^{edge}`：查询框到 crop 边界的归一化距离；
- `r_v^{visible}`：预测框在有效 crop 区域中的可见比例；
- `e_v^{view}`：TL/TR/BL/BR 视图嵌入。

几何编码网络：

```text
geometry vector
    -> Linear
    -> SiLU
    -> Linear 256
```

规范化查询：

\[
Q_v^g
=
\widehat Q_v+
\beta_q\tanh(\operatorname{MLP}(z_v)),
\qquad
0\le\beta_q\le0.2.
\]

GCQL 是可学习网络模块；确定性坐标换算只是其几何锚点。

## 6. M3：APFG

全称：

**Anchored Protected Fusion Gate**

中文：

**锚定式保护融合门**

### 6.1 固定锚点

锚点 `m0` 复用已经通过正式判定的 SADED-SM 逻辑：

- global non-tiny 预测受保护；
- local 只进入 tiny 路径；
- 同类全局—局部候选使用固定几何关系建立对应；
- crop 边界碎片不得覆盖受保护 global 预测；
- 最终候选保持固定稳定排序。

APFG 初始状态必须严格复现该锚点。

### 6.2 可训练输入

每个 local canonical query 使用：

\[
x=
[Q_v^g,
P_v^{cls},
U_v,
B_v^g,
d_v^{edge},
r_v^{visible},
c_v^{global}].
\]

`c_v^{global}` 是该 local query 与同类 global query 的受限对应表示。

### 6.3 门控网络

```text
query/geometric evidence
    -> LayerNorm
    -> Linear  → 128
    -> SiLU
    -> Linear  → 32
    -> SiLU
    -> Linear  → quality residual + admission residual
```

学习输出只作为固定锚点的有界残差：

\[
\Delta s=\eta_s\tanh(h_s(x)),
\qquad |\eta_s|\le0.2,
\]

\[
r=\sigma(h_r(x)).
\]

最终 local 分数：

\[
s_l^*
=
m_0\cdot s_l\exp(\Delta s)\cdot r.
\]

### 6.4 结构保护

APFG 必须满足以下不可学习约束：

1. global non-tiny 查询完整保留；
2. local 查询不能替换受保护 global non-tiny 查询；
3. local 只与 global tiny 或未占用容量竞争；
4. 关闭学习残差时，输出逐项恢复固定 SADED 锚点；
5. APFG 不向冻结 global detector 反向传播梯度。

### 6.5 监督

local query 映射到全局后直接与全局 GT 匹配：

- 匹配未覆盖 tiny GT：正样本；
- 与已有正确 global tiny 重复且无质量改善：重复负样本；
- 与受保护 non-tiny 重叠：保护负样本；
- crop 截断碎片或背景框：可靠性负样本。

损失：

\[
\mathcal L_{APFG}
=
\lambda_a\mathcal L_{admit}
+
\lambda_q\mathcal L_{quality}
+
\lambda_p\mathcal L_{protect}.
\]

## 7. 训练策略

### 7.1 冻结范围

G0 阶段冻结：

- global backbone；
- Hybrid Encoder；
- global query selection；
- global decoder；
- global detection heads；
- 所有 BatchNorm running buffers。

仅训练：

- DSTE adapter 和局部头副本；
- GCQL geometry MLP；
- APFG gate。

这样 Method-Off 必须与 Control 保持 detector 权重一致，不允许再次出现旧 GCMV 的训练漂移。

### 7.2 初始化

- 全局分支加载正式 100-epoch baseline；
- 局部分支复用相同 baseline 权重；
- DSTE 分类和回归头从 baseline 对应头初始化；
- DSTE residual、GCQL residual 和 APFG residual 零初始化；
- APFG admission 初始输出匹配固定 SADED 锚点。

### 7.3 分阶段训练

#### G0-A：锚点复现

不训练，验证新流水线逐预测复现固定 SADED：

- 输出数量一致；
- 坐标一致；
- 分类一致；
- 分数一致；
- 最终排序一致；
- SBR 指标一致。

#### G0-B：APFG 快速筛查

冻结 detector 和 DSTE，只训练 GCQL/APFG。

优先使用缓存的全局/局部查询和预测，避免每个 epoch 重跑 detector。

#### G0-C：DSTE 联合筛查

只在 G0-B 通过后启用 DSTE，继续冻结全局 detector。

#### G1：正式短训

使用完整 train、val 和固定协议进行配对 10-epoch 训练。

只有 G1 通过预注册门槛，才能进入多 seed 或 100-epoch 实验。

## 8. 效率设计

“效率第一”首先指研发效率，其次指推理效率。

### 8.1 研发效率

执行顺序固定为：

1. 缓存 mature baseline 的 global/local 查询、框、分数和几何信息；
2. 离线训练 GCQL/APFG；
3. 在 val 上执行三状态评估；
4. 通过后才接入 DSTE；
5. 不先写完整训练系统，不先跑 fresh100。

### 8.2 推理效率

- 四个 local view 允许合并成一个 view batch；
- global 与 local 使用同一 checkpoint；
- DSTE/GCQL/APFG 参数量目标小于 2M；
- 不增加 global decoder query 数；
- local 候选每视图最多保留 64；
- APFG 前 local 总候选最多 256；
- 最终 `max_det=300`；
- 目标端到端延迟不超过 baseline 的 3 倍；
- 若显存不足，允许 local view micro-batch，但必须报告真实延迟。

## 9. G0 对照

至少包含：

| 实验 | Global | Local evidence | GCQL | APFG |
|---|:---:|:---:|:---:|:---:|
| Control | ✓ |  |  |  |
| Raw local union | ✓ | ✓ | fixed mapping |  |
| Fixed SADED anchor | ✓ | ✓ | fixed mapping | fixed |
| Learned gate | ✓ | ✓ | ✓ | ✓ |
| Full GCTE | ✓ | DSTE | ✓ | ✓ |
| Full GCTE, learning off | ✓ | ✓ | fixed anchor | fixed |

三状态必须继续报告：

1. Control；
2. Method-On；
3. Method-Off / learned residual off。

## 10. G0 预注册门槛

进入正式 G1 的必要条件：

- `Δ AP-tiny-SBR >= +0.010`，Method-On 对 Control；
- `Δ mAP50-95 >= +0.005`；
- `Δ tiny recall >= +0.020`；
- `Δ AP-medium-SBR >= -0.002`；
- `Δ AP-large-SBR >= -0.005`；
- Learned Gate 相比 Raw Local Union 有正向 mAP 或 AP-tiny 贡献；
- Method-Off 精确恢复固定 SADED 锚点；
- global protected predictions 的坐标、类别和分数零漂移；
- 端到端延迟不超过 Control 的 3 倍；
- 所有输出、checkpoint、配置和指标具有 SHA-256 闭环。

任一核心门槛失败：

- 不调更多超参数；
- 不启动 fresh100；
- 停止当前多视图网络化方向；
- 重新选择创新点 1。

## 11. 风险与处理

### 11.1 学习门不优于固定锚点

处理：只允许一次目标或输入审计；仍无正贡献则判定该网络化方向失败。

### 11.2 DSTE 不优于原始 local detector

处理：检查 query/head 初始化和 crop GT；不得通过增加 decoder 层数无限扩张。

### 11.3 延迟超过 3 倍

处理顺序：

1. 四视图批处理；
2. local query 截断到 64/view；
3. 缓存或复用 local feature projection；
4. 仍超限则判 G0 失败。

### 11.4 被认为是 SAHI 后处理

论文必须证明：

- DSTE 接受检测损失；
- GCQL 有可训练几何查询编码；
- APFG 在网络前向中产生分数和准入残差；
- 三个模块都有参数、梯度、消融和结构图位置；
- 固定 SADED 只是初始化锚点和对照，不是最终唯一方法。

## 12. 完成定义

创新点 1 只有在以下条件全部满足时才冻结：

1. G0 和 G1 均通过预注册门槛；
2. 三个模块都有梯度与参数证据；
3. 完整模型的 tiny、mAP 和保护指标通过；
4. 至少两个 seed 正向，正式论文优先三个 seed；
5. 第二数据集或第二 detector 至少完成一个；
6. 完整复杂度、延迟和显存报告已生成；
7. 失败案例与适用边界如实记录。
