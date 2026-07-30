# SQDA-SGC 最终紧凑设计规格

日期：2026-07-30
状态：内部逻辑审查通过，等待用户最终复核
目标模型：Ultralytics RT-DETR-L
目标数据集：VisDrone2019-DET
约束：seed 0，单卡 RTX 4090/5090，15 天内完成主实验

## 1. 最终决策

最终网络模块冻结为：

**SQDA-SGC：Shadow-Query Semantic–Geometry–Context Adapter**

中文名称：

**语义—几何—上下文影子查询适配器**

SQDA-SGC 位于 RT-DETR encoder Top-300 query selection 与 stock decoder 之间。每个原生 object query 临时生成六个角色 token：

- 一个中心语义 shadow：`C`；
- 四个边界几何 shadows：`L/R/T/B`；
- 一个只读上下文 probe：`O`。

`C/L/R/T/B` 读取 raw stride-4 C2 的目标内部和四侧边界；`O` 只读取局部平滑 C2 的外环。上下文不直接写入 query，也不从目标特征中相减，只对语义证据的可信度做有下限的调制。语义和几何证据经过 16 组门控后进入同一个融合投影，并通过一个小尺度残差写回原 query。

最终仍输出原生 300 queries。Shadow/probe tokens 使用后立即销毁，不参加 Hungarian matching，不独立预测类别或边框。

## 2. 三种方案比较

| 方案 | 主要结构 | 优点 | 主要风险 | 决策 |
|---|---|---|---|---|
| Full SGC | 中心、边界、外环；显式高低频；背景相减；双残差 | 容量最大 | 背景相减伤害密集同类目标；高频增强与噪声抑制对冲；分支过多 | 否决 |
| Dual-path SGC-v2 | 上下文可信度；语义/几何双门和双 LayerScale | 任务解释清晰 | 同一 context scalar 同时压制几何；两个残差写入同一 query，可能抵消 | 不作为主版本 |
| Compact SQDA-SGC | 五个写回角色、一个只读 probe；共享投影；单融合残差 | 信息流闭合、参数少、冲突最小 | 容量略低于双路版 | **最终采用** |

## 3. 证据链与设计推论

### 3.1 稀疏读取 raw C2

QueryDet 证明高分辨率特征对小目标有效，同时说明应避免在整张高分辨率特征上执行冗余密集计算。ACCV 2024 的小目标结构研究也显示 stride-4 特征有价值，但冗余深层分支可以删除。

设计推论：

- 使用 raw C2；
- 先采样、后投影；
- 不增加全局 P2 检测头；
- 不建立新的 C2/P3/P4/P5 多尺度 shadow pyramid；
- 每个原 query 固定读取 20 个点。

### 3.2 中心与边界角色分工

Decoupled DETR 的实验证据表明，显著内部区域更适合分类，边界区域更适合定位。

设计推论：

- `C` 读取目标内部，提供语义/存在性证据；
- `L/R/T/B` 读取四侧边界，提供尺寸和定位证据；
- 不复制双 decoder；
- 不修改 stock decoder 的分类/定位注意力。

### 3.3 外环不是背景标签

Affinity-Aware Relation Network 表明航拍目标经常密集成群，参考目标附近可能存在相似真实目标。

设计推论：

- 禁止 `target - context`；
- 禁止把外环监督为背景；
- 禁止跨 query 邻居传播；
- 外环只能是只读可信度 probe；
- context 调制不得完全关闭语义更新，更不得压制几何更新。

### 3.4 语义空间需要显式对齐

SAM-DETR 表明 query 与图像特征的语义空间错位会增加匹配难度。

设计推论：

- point attention 使用 shadow token 与采样特征的投影相似度；
- context reliability 使用显式 cosine similarity；
- 不依赖普通拼接 MLP 独自判断一致性。

### 3.5 Top-300 是能力边界

RT-DETR 和 Salience DETR 均表明 two-stage query selection 的质量和尺度偏差会限制 decoder 上限。

设计推论：

- SQDA-SGC 不修改 query selection；
- 训练前必须测量 encoder Top-300 proposal recall；
- 若漏检 GT 周围不存在可用 proposal，SQDA-SGC 不承诺恢复该目标；
- 不通过动态 query 数量、DDQ、SQR 或 query recollection 扩大本创新点范围。

## 4. 最终信息流

```text
raw C2 ----------------------------------------------|
  |                                                  |
  |--> target sampler --> shared point projector     |
  |                                                  |
  |--> AvgPool 3x3 --> context sampler --------------|
                                                     |
P3/P4/P5 -> Hybrid Encoder -> Top-300 q,b            |
                                  |                  |
                                  v                  v
                         geometry-role generator
                                  |
               ----------------------------------------
               |            |                         |
          center C      edges L/R/T/B       read-only context O
           4 points        8 points                 8 points
               |            |                         |
          semantic d       geometry d              context d
               |            |                         |
               |       grouped evidence gates        |
               |            |                         |
               ----- context-safe single fusion <-----
                                  |
                       one bounded LayerScale residual
                                  |
                         enhanced Top-300 queries
                                  |
                         stock RT-DETR decoder
                                  |
                    stock cls + L1 + GIoU losses
```

## 5. 顶层接口

```python
class SQDASGCAdapter(nn.Module):
    def __init__(
        self,
        detail_channels: int,
        hidden_dim: int = 256,
        gate_groups: int = 16,
        residual_cap: float = 0.05,
        residual_init: float = 1e-3,
        enabled: bool = True,
    ) -> None:
        ...

    def forward(
        self,
        object_queries: Tensor,      # [B, 300, 256]
        reference_boxes: Tensor,     # [B, 300, 4], normalized cxcywh
        raw_c2: Tensor,              # [B, C2, H/4, W/4]
        *,
        identity_override: bool = False,
    ) -> tuple[Tensor, dict]:
        ...
```

输出：

- `enhanced_queries`：`[B,300,256]`；
- `diagnostics`：采样有效率、point attention、edge attention、context reliability、group gates、语义/几何分量余弦、残差范数和 LayerScale。

`identity_override` 只用于 G0 测试，不得暴露给正式 train/val/predict 结果 CLI。

## 6. 冻结 reference geometry

SQDA-SGC 使用：

\[
\bar b_i=\operatorname{stopgrad}(b_i).
\]

所有 shadow/probe 坐标均由 \(\bar b_i\) 生成。该约束隔离两条梯度目标：

- RT-DETR 回归负责把 reference box 推向 GT；
- SQDA-SGC 负责读取给定 reference box 周围的 C2 证据。

SQDA-SGC 不允许为了获得更强纹理而反向移动 encoder proposal。

## 7. 共享几何角色生成器

先建立 query-conditioned geometry code：

\[
p_i^{geo}
=
\operatorname{MLP}_q(\operatorname{LN}(q_i))
\odot
\operatorname{MLP}_b(\operatorname{PE}(\bar b_i)).
\]

`\(\operatorname{PE}\)` 使用 4D box 的标准 sine-cosine encoding，每个 \(cx,cy,w,h\) 分配 64 维，总计 256 维；`MLP_b` 和 `MLP_q` 均输出 256 维。

六个角色使用共享主投影和轻量 role-FiLM：

\[
s_i^k
=
\operatorname{LN}
\left(
\left[1+0.1\tanh(\gamma_k)\right]\odot W_sq_i
+
\beta_k
+
p_i^{geo}
\right),
\]

\[
k\in\{C,L,R,T,B,O\}.
\]

`role-FiLM` 取代六套低秩适配器：

- 角色仍有内容依赖的不同表示；
- 参数更少；
- 不复制 MI-DETR 式并行 decoder/head；
- 避免小数据集上六个独立适配器过拟合。

\(\gamma_k\) 初始化为零；\(\beta_k\) 使用小方差正常初始化。因此初始 role scale 为 1，不会在第一步放大 query。

## 8. 固定角色采样模板

定义具有 C2 cell 下限的半尺度：

\[
u_i^x=\max(\bar w_i/2,\;1/W_2),
\qquad
u_i^y=\max(\bar h_i/2,\;1/H_2).
\]

固定 20 个基础点：

### 8.1 中心 C：4 点

\[
(\pm0.5u_i^x,\;\pm0.5u_i^y).
\]

### 8.2 四边界：每边 2 点，共 8 点

\[
L=(-u_i^x,\;\pm0.5u_i^y),
\quad
R=(u_i^x,\;\pm0.5u_i^y),
\]

\[
T=(\pm0.5u_i^x,\;-u_i^y),
\quad
B=(\pm0.5u_i^x,\;u_i^y).
\]

### 8.3 外环 O：8 点

\[
(\pm1.5u_i^x,0),
\quad
(0,\pm1.5u_i^y),
\]

\[
(\pm1.25u_i^x,\;\pm1.25u_i^y).
\]

每个点只允许一个有界修正：

\[
p_i^{k,m}
=
\bar c_i
+
\Delta_i^{k,m}
+
0.1(u_i^x,u_i^y)
\odot
\tanh(W_p^{k,m}s_i^k).
\]

所有 point-offset 输出层必须使用零初始化：

```text
W_p.weight = 0
W_p.bias = 0
```

因此正式训练从固定中心、边界和外环模板开始；offset 参数在第一次反向传播后仍可获得梯度，但不会在初始化时把小目标采样点随机推入背景。第一版保持 `0.1u` 上限，不把扩大 offset 搜索范围作为 G1/G2 调参项。

第一版不预测全局自适应半径。固定模板已经覆盖中心、边界和外环；再增加 radius network 会与 point offset 功能重复。

越界点使用 validity mask。单个角色全部越界时，其 descriptor 置零；所有写回角色均无效时，最终残差严格为零，禁止对全负无穷 logits 执行 softmax。

## 9. 先采样、后共享投影

目标和边界点直接读取 raw C2：

\[
\bar z_i^{k,m}
=
\operatorname{GridSample}(C_2,p_i^{k,m}),
\quad
k\in\{C,L,R,T,B\}.
\]

上下文只读取轻微平滑视图：

\[
C_2^{ctx}
=
\operatorname{AvgPool}_{3\times3}(C_2),
\]

Average pooling 固定为 kernel 3、stride 1、padding 1、`count_include_pad=False`，保持与 raw C2 相同的空间尺寸。

\[
\bar z_i^{O,m}
=
\operatorname{GridSample}(C_2^{ctx},p_i^{O,m}).
\]

所有点共享同一个 projector：

\[
z_i^{k,m}
=
W_{v2}
\operatorname{SiLU}
\left(
\operatorname{LN}(W_{v1}\bar z_i^{k,m})
\right),
\]

\[
W_{v1}:C_2\rightarrow128,
\qquad
W_{v2}:128\rightarrow256.
\]

禁止构造：

- \(C_2-\operatorname{AvgPool}(C_2)\) 显式高频分支；
- FFT、DCT 或小波分支；
- target/context 两套完整 projector；
- 全图 256-channel C2 projection。

所有采样点按 batch、query、role 和 point 维度打包为一次向量化 `grid_sample`；使用 `align_corners=False` 和 zero padding。Validity mask 仍由未裁剪的归一化坐标独立计算。

## 10. 角色内 point attention

每个角色 token 自身作为 attention query：

\[
\ell_i^{k,m}
=
\frac{
(W_qs_i^k)^\top(W_kz_i^{k,m})
}{
\sqrt{256}
},
\]

\[
a_i^{k,m}
=
\operatorname{softmax}_m(\ell_i^{k,m}),
\]

\[
d_i^k
=
\sum_m a_i^{k,m}z_i^{k,m}.
\]

禁止由父 query 统一生成六个角色的 attention weights，否则角色只剩坐标差异。

## 11. 三类证据

中心语义：

\[
d_i^{sem}=d_i^C.
\]

边界几何使用父 query 在四边之间聚合：

\[
\beta_i^k
=
\operatorname{softmax}_{k\in\{L,R,T,B\}}
\left(
\frac{
(W_{eq}q_i)^\top(W_{ed}d_i^k)
}{
\sqrt{256}
}
\right),
\]

\[
d_i^{geo}
=
\sum_{k\in\{L,R,T,B\}}\beta_i^kd_i^k.
\]

外环上下文：

\[
d_i^{ctx}=d_i^O.
\]

SQDA-SGC 逐 query 独立计算，不接收其他 \(q_j\) 或 \(b_j\)。

## 12. 只读上下文可信度

语义对齐：

\[
r_i^{sem}
=
\cos(W_rq_i,\;W_rd_i^{sem}),
\]

\[
r_i^{ctx}
=
\cos(W_rq_i,\;W_rd_i^{ctx}).
\]

\[
r_i^{geo}
=
\cos(W_rq_i,\;W_rd_i^{geo}).
\]

Cosine similarity 统一使用 \(10^{-6}\) 范数下限；descriptor 为零时相似度定义为零，禁止产生 NaN。

上下文只调制语义证据。为避免固定上下文分支在语义与上下文同等相似时仍把新增语义证据压到 `0.75`，引入一个全局可学习、严格有界的 context strength：

\[
\lambda_{ctx}
=
0.25\,\sigma(a_{ctx}),
\qquad
\lambda_{ctx,\mathrm{init}}=0.05,
\qquad
a_{ctx,\mathrm{init}}=\operatorname{logit}(0.2)\approx-1.3863.
\]

语义可信度定义为：

\[
c_i^{sem}
=
1-\lambda_{ctx}\,
\sigma
\left(
2(r_i^{ctx}-r_i^{sem})
\right).
\]

因此：

\[
0.75<c_i^{sem}<1.
\]

初始化时约有 \(0.95<c_i^{sem}<1\)，且 \(r_i^{sem}=r_i^{ctx}\) 时 \(c_i^{sem}=0.975\)，近似中性而不是固定衰减 25%。

该式满足：

- context 不直接写入 query；
- context 不从 target descriptor 中相减；
- 外环存在同类邻居时最多衰减 25% 的新增语义证据；
- context 不压制 geometry branch；
- context descriptor 无效时固定 \(c_i^{sem}=1\)。

## 13. 16 组证据门

构造：

\[
h_i=
[
\operatorname{LN}(q_i),
\operatorname{LN}(d_i^{sem}),
\operatorname{LN}(d_i^{geo}),
\operatorname{LN}(q_i)\odot\operatorname{LN}(d_i^{sem}),
\operatorname{LN}(q_i)\odot\operatorname{LN}(d_i^{geo}),
r_i^{sem},
r_i^{geo},
\log\bar w_i,
\log\bar h_i
].
\]

\(\bar w_i,\bar h_i\) 在取对数前截断到不小于 \(10^{-6}\)。

一个共享 gate MLP 输出 32 个值：

\[
[g_i^{sem},g_i^{geo}]
=
\sigma(\operatorname{MLP}_{gate}(h_i)),
\]

\[
g_i^{sem},g_i^{geo}\in\mathbb{R}^{16}.
\]

Gate MLP 冻结为 `Linear(input,128) -> SiLU -> Linear(128,32)`；输出层 bias 初始化为零。

每个 gate 值重复 16 次扩展到 256 维。该设计位于单标量 gate 与完全自由 256-channel gate 之间：

- 能区分不同特征组；
- 比两个 256-channel gates 参数更少；
- 不需要人工 tiny/large 开关；
- 尺度只作为输入，不硬关闭 medium/large queries。

## 14. 单融合、单写回

语义和几何先进入一个融合投影：

\[
f_i
=
W_f
\left[
c_i^{sem}\,
\widehat g_i^{sem}\odot d_i^{sem}
;
\widehat g_i^{geo}\odot d_i^{geo}
\right],
\]

\[
W_f:\mathbb{R}^{512}\rightarrow\mathbb{R}^{256}.
\]

`W_f.bias` 固定初始化为零。如果 `C/L/R/T/B` 全部无效，则在 fusion 前应用 writeback-valid mask，强制 \(f_i=0\)。

只保留一个残差：

\[
q_i'
=
q_i+\alpha f_i.
\]

LayerScale：

\[
\alpha
=
0.05\,\sigma(a),
\qquad
\alpha_{\mathrm{init}}=10^{-3}.
\]

`W_f.weight` 使用 \(\mathcal N(0,0.01^2)\) 初始化，不使用零权重初始化。这样所有角色从第一次反向传播就能获得梯度。

单写回是本轮内部审查的关键收缩：

- 不存在语义 residual 与几何 residual 显式相减；
- context 不能写入残差；
- 只有一个幅度上限；
- 不使用 `tanh` 压缩融合证据；
- 原 query 始终逐元素保留。

G0 使用 `identity_override=True` 强制 \(\alpha=0\)，不改变正式训练参数初始化。

## 15. RT-DETR 集成边界

调用位置冻结在 `_get_decoder_input()` 之后、`self.decoder(...)` 之前。

训练含 denoising queries 时：

```text
unchanged denoising queries
        +
SQDA-SGC(native 300 object queries only)
        ->
stock decoder
```

硬约束：

- C2 不进入 RT-DETR `_get_encoder_input()`；
- SQDA-SGC 不修改 encoder Top-300 selection；
- SQDA-SGC 不修改 detached reference boxes；
- SQDA-SGC 不处理 denoising queries；
- SQDA-SGC 不修改 attention mask；
- SQDA-SGC 不改变 query 数量或顺序；
- SQDA-SGC 不修改 decoder、heads、matcher、loss 或 postprocess；
- train/val/predict 使用同一个 module forward；
- 参数进入 optimizer、state dict 和 checkpoint。

## 16. 与其他候选模块的隔离

主论文第一阶段只允许：

```text
stock RT-DETR-L + SQDA-SGC
```

不同时叠加 IQC-BC、VSF-RMR、动态 query、SQR、DDQ、EASE、DEIM 或新的 loss。不同位置不等于没有梯度干涉；任何叠加都必须在 SQDA-SGC 单模块通过 G2 后作为独立后续实验。

SQDA-SGC 本身：

- 不跨 query；
- 不做 query competition；
- 不改变 feature pyramid；
- 不增加辅助检测监督；
- 不改变匹配和正负样本。

## 17. 训练前能力上限诊断

在实现 SQDA-SGC 前，对 baseline encoder Top-300 boxes 做 class-agnostic proposal recall：

\[
R_{\mathrm{Top300}}(\operatorname{IoU}\ge0.3),
\qquad
R_{\mathrm{Top300}}(\operatorname{IoU}\ge0.5).
\]

分别报告 tiny、small、medium、large，并额外统计：

> baseline 最终漏检的 GT 中，有多少仍能在 Top-300 找到 IoU≥0.3 的 proposal。

该诊断不修改网络，也不作为论文创新。它用于判断：

- 有 proposal、缺语义或定位：SQDA-SGC 有改善空间；
- 没有 proposal：SQDA-SGC 的 Recall 上限有限。

在获得该诊断前，不得把“四项全部不下降概率超过 80%”写成事实。

## 18. 训练与准入

### G0：严格恒等

- `enabled=False`：query、logits、boxes 与 baseline 一致；
- `enabled=True, identity_override=True`：完整读取路径执行，但最终残差为零；
- native logits/boxes 最大绝对误差小于 \(10^{-6}\)；
- native Precision、Recall、mAP50、mAP50-95 一致；
- denoising queries、reference boxes、encoder outputs 一致。

G0 失败只能修复集成错误，禁止调参或训练。

### G1：冻结短训

冻结整个 baseline，只训练 SQDA-SGC：

- seed 0；
- stock cls/L1/GIoU losses；
- 3 epochs；
- \(\alpha_{\mathrm{init}}=10^{-3}\)；
- \(0<\alpha<0.05\)；
- optimizer 固定为 AdamW；
- module learning rate 固定为 \(10^{-4}\)；
- `betas=(0.9,0.999)`；
- projector、attention、gate 和 fusion 矩阵使用 `weight_decay=1e-4`；
- bias、LayerNorm、\(\alpha\)、\(\lambda_{ctx}\) 和其他标量参数使用 `weight_decay=0`；
- module-only gradient norm clip 固定为 `0.1`；
- 不新增辅助 loss；
- 不改变数据、增强、输入尺寸、阈值或 validator。

### G2：正式训练

G1 通过后从同一成熟 baseline 重新开始正式训练，继续冻结 baseline，只训练 SQDA-SGC 10 epochs；不得把 G1 的 3 epochs 叠加成 13-epoch schedule。只有满足四项硬约束的 checkpoint 才进入候选集：

\[
\Delta P\ge0,\quad
\Delta R\ge0,\quad
\Delta mAP50\ge0,\quad
\Delta mAP50\text{-}95\ge0.
\]

同时要求预注册的 AP-small 提升不少于 0.2 个百分点。若 AP-small 不可由 native evaluator稳定复现，则在第一次训练前固定替代指标，训练后不得更换。

任一主指标下降即失败；不得用平均值、多 seed、阈值变化或只报告有利指标掩盖。

## 19. 预注册消融

| 实验 | Center | Edges | Context probe | Geometry code | Group gate | Single residual |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline |  |  |  |  |  |  |
| Center-only | ✓ |  |  | ✓ | ✓ | ✓ |
| Center + Edges | ✓ | ✓ |  | ✓ | ✓ | ✓ |
| Full without context modulation | ✓ | ✓ | 只采样不调制 | ✓ | ✓ | ✓ |
| Full SQDA-SGC | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Full, geometry code off | ✓ | ✓ | ✓ |  | ✓ | ✓ |

不得把 adaptive radius、dual residual、frequency branch 或 cross-query propagation 作为临时救火消融。

## 20. 功能性测试

### 单元测试

- 输入输出 shape、dtype、device；
- 300×20 sampling tensor；
- center/edge/ring 固定模板；
- point-offset 输出层权重和 bias 零初始化；
- tiny box 的 C2 cell 尺度下限；
- detached reference boxes 无梯度；
- 角色 token 使用自己的 point attention；
- 越界 mask 和 all-invalid fallback 无 NaN；
- cosine 的零向量输入无 NaN；
- context 无效时 reliability 为 1；
- \(\lambda_{ctx}\) 的范围、初始化和 context modulation 上下界；
- context descriptor 不进入 fusion tensor；
- 16-group gate 正确扩展到 256；
- `W_f.bias=0` 且所有写回角色无效时 residual 严格为零；
- LayerScale 范围和初始化；
- enabled/off 与 identity override 恒等性。

### 集成测试

- 模块注册在模型树；
- 参数进入 optimizer/state dict/checkpoint；
- 一次反向传播后所有写回角色、context reliability、gate 和 fusion projection 均有有限非零梯度；
- baseline 参数和 buffers 在 G1 前后逐项不变；
- denoising query 顺序和值不变；
- native train/val/predict 运行；
- zero-GT batch 和 mixed precision 无 NaN/Inf；
- checkpoint strict reload。

### 诊断测试

- center/edge/context attention heatmap；
- context reliability 分布及其按拥挤度分桶；
- semantic/geometry fusion 分量余弦；
- group gate 饱和率；
- residual-to-query norm ratio；
- 参数量、FLOPs、显存和端到端延迟。

## 21. 内部冲突审查矩阵

| 可能冲突 | 风险 | 最终处理 | 结论 |
|---|---|---|---|
| raw C2 细节 vs 背景纹理 | Precision 下降 | context 只读可靠度 + grouped gate | 已隔离 |
| context vs 密集同类邻居 | Recall 下降 | 不相减、不写回、可学习强度、语义调制下限 0.75、初始化近似 1 | 已缓解 |
| context vs geometry | 抑制 AP75 | context 不调制 geometry | 已消除结构性冲突 |
| semantic vs geometry 双 residual | 两路抵消或放大 | 单融合、单 LayerScale | 已消除显式对冲 |
| sampling vs box regression | proposal 为采样迁移 | reference box stop-gradient | 已隔离 |
| 角色专门化 vs 参数臃肿 | 过拟合和延迟 | 共享投影 + role-FiLM | 已简化 |
| scale normalization vs tiny 点坍缩 | 采样点重合 | 一个 C2 cell 下限 | 已处理 |
| LayerScale 安全 vs 初始无梯度 | 短训无效 | 正常初始化 + \(10^{-3}\) gain | 已处理 |
| SQDA vs query competition | 归因混乱 | 单模块主实验、无跨 query | 已隔离 |
| SQDA vs query selection 上限 | 无 proposal 无法恢复 | Top-300 recall 预诊断 | 已声明边界 |

结论：

- 最终信息流中不存在一个分支直接增强、另一个分支再显式减去同一证据的结构；
- context、semantic、geometry 的职责不重叠；
- 只有一个最终残差写回点；
- 仍无法在训练前保证 learned branches 不产生统计性干扰，因此保留 G0/G1/G2 和诊断门槛。

## 22. 明确删除项

最终版本删除：

- 背景/context 特征直接相减；
- context 直接写回；
- context 同时压制语义和几何；
- 显式 high-frequency/detail 分解；
- FFT/DCT/小波；
- 六套低秩 role adapters；
- 两个独立 LayerScale residuals；
- 完全 256-channel 自由 gates；
- adaptive sampling radius；
- multi-scale shadows；
- query 邻域传播；
- SQR/query recollection；
- dynamic query count；
- auxiliary loss；
- 与其他候选模块同时训练。

## 23. 复杂度边界

- 最终 query 数：300；
- 临时角色：6/query；
- 采样点：20/query，6000/image；
- context 平滑：固定 3×3 average pooling；
- gate groups：16；
- 新增参数目标：小于 1M；
- 端到端延迟目标：不超过 baseline 的 1.20 倍；
- 若超出任一目标，先减少隐藏维度，不删除中心/边界/context 的逻辑边界。

### 23.1 权威实验底座

交接材料只提供实验底座，不继承其中的 ACR-EG/GCQF 网络、checkpoint 或续训任务。SQDA-SGC 的权威底座冻结为：

```text
repository = kkc236/uav-detection-baselines
implementation branch = codex/sqda-sgc
branch base = codex/matched-baseline@b08bc2ac
baseline model = stock RT-DETR-L
baseline checkpoint = matched-baseline-best-epoch-0100.pt
baseline checkpoint bytes = 66262262
baseline checkpoint SHA256 = 54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B
dataset = VisDrone2019-DET train/val
train images = 6471
val images = 548
classes = 10
seed = 0
imgsz = 640
batch = 8
queries = 300
max_det = 300
NMS = False
```

服务器目标环境：

```text
Ubuntu = 22.04 x86_64
GPU = RTX 4090 24GB
Python = 3.10.x
Ultralytics = 8.4.90
PyTorch = 2.5.1+cu121
TorchVision = 0.20.1+cu121
CUDA runtime = 12.1
dataset root = /root/data/uav/datasets/VisDrone
data yaml = /root/data/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml
```

服务器实际根目录与旧交接中的 `/mnt/uav` 不同，因此本次统一落盘到 93GB 数据盘 `/root/data/uav`，禁止占满系统盘。部署时必须重新验证数据文件数、YAML SHA256、val 内容签名和 baseline checkpoint SHA256。

ACR-EG 的 `epoch8.pt`、多视图输入、GCQF、logit injection、MuSGD 续训和 100-epoch resume 均不属于 SQDA-SGC，禁止混入实现、训练或论文结果。

## 24. 论文可主张与不可主张

允许的核心主张：

> 每个已选原生 query 被临时分解为中心语义、四边几何和只读上下文角色；这些角色在尺度归一化的 raw-C2 邻域中读取互补证据，利用不伤害密集邻居的上下文可信度与分组融合形成单一小尺度残差，并在不改变原生预测集合的情况下补充小目标细节。

不得把以下单独写成原创：

- deformable/local sampling；
- 高分辨率特征；
- 中心/边界解耦；
- average pooling；
- cosine similarity；
- grouped gate；
- LayerScale；
- geometry positional encoding。

## 25. 完成定义

只有同时满足以下条件，SQDA-SGC 才可作为论文主网络模块：

1. Top-300 proposal recall 诊断完成；
2. G0 native 输出严格回退 baseline；
3. 模块参数、梯度、optimizer 和 checkpoint 闭环完成；
4. train/val/predict 使用同一 forward；
5. 最终仍为原生 300-query 预测集；
6. Precision、Recall、mAP50、mAP50-95 四项均不下降；
7. AP-small 或训练前预注册替代指标提升不少于 0.2 个百分点；
8. 完成预注册消融；
9. 参数量、延迟、显存和失败案例如实报告；
10. 所有结果具有配置、日志、checkpoint SHA-256 和 git commit 证据。

未满足任一条，不得宣称模块验证成功。

## 26. 主要参考边界

- Yang et al., *QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection*, CVPR 2022.
- Zhang et al., *Accelerating DETR Convergence via Semantic-Aligned Matching*, CVPR 2022.
- Zhang et al., *Decoupled DETR: Spatially Disentangling Localization and Classification for Improved End-to-End Object Detection*, ICCV 2023.
- Hou et al., *Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement*, CVPR 2024.
- Sun et al., *SET: Spectral Enhancement for Tiny Object Detection*, CVPR 2025.
- Hu et al., *A Universal Structure of YOLO Series Small Object Detection Models*, ACCV 2024.
- Fang et al., *Affinity-Aware Relation Network for Oriented Object Detection in Aerial Images*, ACCV 2022.
- Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection*, CVPR 2024 / arXiv:2304.08069.

HEDS-DETR、UAV-DETR 和 EFSI-DETR 只作为辅助工程证据；在其正式录用状态未独立核实前，不作为 A/B/C 会创新性依据。
