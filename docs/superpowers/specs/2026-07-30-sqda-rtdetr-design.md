# SQDA-P-RTDETR 历史设计规格

日期：2026-07-30
状态：已被 `2026-07-30-sqda-sgc-final-design.md` 取代，仅保留设计演化记录
目标模型：Ultralytics RT-DETR-L
目标数据集：VisDrone2019-DET
研发约束：单卡 RTX 4090/5090，seed 0，15 天内完成主实验

## 1. 决策

创新点冻结为：

**SQDA-P：Performance-Oriented Shadow-Query Detail Adapter**

中文名称：

**效果优先影子查询细节适配器**

SQDA-P 是插在 RT-DETR encoder query selection 与 stock decoder 之间的可训练网络模块。模块保留原始 300 个 object queries，为每个 object query 在模块内部生成五个短寿命 shadow queries：一个中心 shadow 和四个象限 shadow。Shadow queries 作为真实的 attention queries，从 backbone 的 raw stride-4 C2 高分辨率特征中读取局部细节，经 query-conditioned 聚合和通道级一致性门控后，以有界小增益残差回注原 object query。

Shadow queries 不进入最终预测集、不参加 Hungarian matching、不独立输出检测框。SQDA-P 输出仍是与 baseline 同形状的 300 个 object queries；原 decoder、检测头、损失函数和推理后处理保持不变。

本设计明确不是：

- 输入切图或多视图推理；
- NMS、Soft-NMS、WBF 或预测框合并；
- 置信度重排或阈值搜索；
- 动态改变最终 query 数量；
- 只在推理阶段启用的算法；
- 损失函数创新或普通超参数调整；
- 把 raw C2 全图直接加到原 neck 的普通 P2 特征融合。

## 2. 论文级问题定义

RT-DETR 从 P3/P4/P5 编码特征中选择高质量 object queries，但无人机图像中的微小目标可能在 stride-8 及更低分辨率特征中丢失局部结构。直接把 raw C2 作为 P2 融入整个 neck 会把大量道路纹理、建筑边缘和背景高频噪声传播到所有位置，存在 Precision 下降风险。

本文统一把 backbone 的 stride-4 输出称为 `raw C2`。SQDA-P 不构造常规 neck P2，也不在 raw C2 上增加检测头。

SQDA-P 采用另一种连接规则：

> 由已经选出的 object query 主动生成局部 shadow queries，仅在其参考框邻域读取 C2 细节；高分辨率证据只有与原 query 语义一致时才允许回注。

该规则将“是否读取高分辨率细节”和“读取哪里的细节”绑定到现有 object query，而不是在全图建立新的检测分支。

## 3. 在 50 篇论文样本中的位置

冻结前审计的 50 篇 A/B 会论文中：

- 29 篇涉及 Decoder 或 Query；
- 12 篇涉及 Backbone 或 Neck；
- 7 篇涉及 Head；
- 11 篇涉及 Training。

位置统计允许一篇论文被计入多个类别。审计文件为：

`docs/AB_VENUE_50_PAPER_NOVELTY_AUDIT_2026-07-30.md`

SQDA-P 选择 query-decoder 交界，原因是该位置既符合 DETR 研究的主流创新落点，又能避免再次提出普通 FPN/P2 融合模块。

与最接近工作的差异冻结如下：

- QueryDet：通过低分辨率位置稀疏启动高分辨率检测计算；SQDA-P 不启动新的高分辨率检测头，而是把局部细节回注原 DETR query。
- DQ-DETR：根据密度预测改变 query 数量和位置；SQDA-P 不预测密度，最终 query 数量固定为 300。
- EASE-DETR：建模 leading/trailing query 竞争；SQDA-P 不判断 leading/trailing，不抑制任何原 query。
- DDQ：生成密集 queries 并筛选 distinct queries；SQDA-P 的 shadow queries 只作为内部特征读取器，不进入匹配和预测集。
- RT-DETR：从 encoder 特征中选择低不确定性 query；SQDA-P 在选择之后为这些 queries 补充 query-conditioned 高分辨率证据。

论文不得声称“全球首个 shadow query”。允许的主张是：

> 本文提出一种面向无人机微小目标的临时查询读取—语义一致性过滤—原查询残差回注机制，并将其实现为 RT-DETR 的端到端可训练适配器。

## 4. 总体架构

```text
Input image
    |
    v
Backbone
    |---------------- C2, stride 4 --------------------|
    |                                                   |
    v                                                   v
C3/C4/C5 -> Hybrid Encoder -> Query Selection      raw C2
                                  |                     |
                         300 object queries             |
                         300 reference boxes            |
                                  |                     |
                                  ------- SQDA-P --------
                                           |
                                  enhanced 300 queries
                                           |
                                  Stock RT-DETR Decoder
                                           |
                           stock cls / bbox / GIoU losses
```

Ultralytics 当前 `RTDETRDecoder.forward()` 在调用 `_get_decoder_input()` 后得到：

- `embed`：decoder query embeddings；
- `refer_bbox`：decoder reference boxes；
- `enc_bboxes` 和 `enc_scores`：encoder top-k 输出。

SQDA-P 的调用位置冻结在 `_get_decoder_input()` 之后、`self.decoder(...)` 之前。

训练阶段如存在 denoising queries，SQDA-P 仍只处理最后 300 个 object queries；denoising queries 保持逐元素不变，并在 SQDA-P 输出后按原顺序拼回。

## 5. 顶层模块接口

```python
class ShadowQueryDetailAdapter(nn.Module):
    def __init__(
        self,
        detail_channels: int,
        hidden_dim: int = 256,
        num_shadows: int = 5,
        points_per_shadow: int = 4,
        residual_cap: float = 0.25,
        residual_init: float = 1e-3,
        enabled: bool = True,
    ) -> None:
        ...

    def forward(
        self,
        object_queries: Tensor,      # [B, 300, 256]
        reference_boxes: Tensor,     # [B, 300, 4], normalized cxcywh
        c2_feature: Tensor,          # [B, C2, H/4, W/4]
        *,
        identity_override: bool = False,  # test-only; never exposed by result CLI
    ) -> tuple[Tensor, dict]:
        ...
```

输出：

- `enhanced_queries`：`[B, 300, 256]`，供 stock decoder 使用；
- `diagnostics`：shadow offsets、point attention、shadow attention、sampling validity、channel-gate mean/std、residual scale、residual norm 和 scale-bin 统计，仅用于训练诊断，不影响预测。

接口不接收预测框列表、NMS 输出或推理阈值。

## 6. 内部组件

### 6.1 SampledC2Projector

职责：先在 backbone raw C2 上执行稀疏、可微采样，再把采样到的低维特征投影到 decoder hidden dimension。禁止先把整张 C2 投影成 256 通道。

对每个采样点：

\[
\bar z_i^{k,m}
=
\operatorname{GridSample}(C_2,p_i^{k,m}),
\]

\[
z_i^{k,m}
=
W_{v2}\operatorname{SiLU}
\left(
\operatorname{LN}(W_{v1}\bar z_i^{k,m})
\right),
\]

其中：

\[
W_{v1}: C_2\rightarrow128,
\qquad
W_{v2}:128\rightarrow256.
\]

该顺序把投影计算从整张 stride-4 特征图缩小到每张图的 \(300\times5\times4=6000\) 个采样点。LayerNorm 不维护 running buffers，冻结 baseline 时不会产生 BatchNorm 漂移。

### 6.2 ShadowQueryGenerator

每个原 query 生成五个 shadow queries：

\[
s_i^k
=
\operatorname{LN}
\left(
W_sq_i+e_k+W_b\phi(b_i)
\right),
\qquad
k=1,\ldots,5.
\]

\(\phi(b_i)\) 编码 normalized \(cx,cy,w,h,\log w,\log h\)，其中 \(w,h\) 在取对数前截断到不小于 \(10^{-6}\)。五个固定角色为：

\[
\Delta_k\in
\{
(0,0),
(-\rho,-\rho),
(\rho,-\rho),
(-\rho,\rho),
(\rho,\rho)
\},
\qquad
\rho=0.2.
\]

第一个 shadow 负责中心主体，后四个分别负责左上、右上、左下、右下局部。Shadow queries 是依赖当前 object query 和 reference box 动态生成的内部张量，不注册为全局 learnable query bank。

### 6.3 QueryConditionedDetailSampler

每个 shadow query 读取四个采样点。点模板为：

\[
\epsilon_m
\in
\{
(-0.5,-0.5),
(0.5,-0.5),
(-0.5,0.5),
(0.5,0.5)
\}.
\]

可学习偏移为：

\[
\delta_i^{k,m}
=
0.1\tanh
\left(
W_p^m s_i^k
\right).
\]

基础局部尺度保留一个 C2 cell 的下限：

\[
r_i^x=\max(0.2w_i,\;1/W_2),
\qquad
r_i^y=\max(0.2h_i,\;1/H_2).
\]

最终采样坐标：

\[
p_i^{k,m}
=
c_i
+
(w_i,h_i)\odot\Delta_k
+
(r_i^x,r_i^y)\odot
\left(
0.5\epsilon_m+\delta_i^{k,m}
\right).
\]

采样坐标保持连续可微。越界点不先裁剪到边界，而是生成 validity mask；无效点的注意力 logit 置为负无穷，避免多个越界点折叠到同一边缘像素。如果同一 shadow 的四个点全部无效，则该 shadow evidence 直接置零并标记为无效，禁止对四个负无穷执行 softmax。

每个 shadow 自身作为 attention query：

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
\sum_{m=1}^{4}
a_i^{k,m}z_i^{k,m}.
\]

禁止退化为由父 query \(q_i\) 直接计算五个 shadow 的点注意力；否则不同 shadow 只剩几何采样差异。

### 6.4 QueryConditionedShadowAggregator

五个 shadow 不做直接拼接。父 query 对五个 shadow evidence 计算竞争权重：

\[
\beta_i^k
=
\operatorname{softmax}_k
\left(
\frac{
(W_qq_i)^\top(W_dd_i^k)
}{
\sqrt{256}
}
\right),
\]

\[
d_i
=
\sum_{k=1}^{5}
\beta_i^kd_i^k.
\]

中心和象限 shadow 的选择完全由当前 query-evidence 一致性决定，不使用类别阈值或人工 tiny/large 分支。Shadow softmax 会屏蔽上一阶段标记为无效的 shadow；若五个 shadow 全部无效，则 \(d_i=0\)，最终残差也必须为零。该聚合只在同一父 query 的五个 shadow 内计算，不在 300 个 object queries 之间增加全局注意力。

### 6.5 ChannelwiseConsistencyGate

Gate 接收原 query、聚合 detail vector、逐通道交互和连续尺度编码：

\[
z_i=
[
\operatorname{LN}(q_i),
\operatorname{LN}(d_i),
\operatorname{LN}(q_i)\odot\operatorname{LN}(d_i),
\operatorname{MLP}_{scale}(\log w_i,\log h_i)
].
\]

\[
\mathbf g_i
=
\sigma
\left(
\operatorname{MLP}_{gate}(z_i)
\right),
\qquad
\mathbf g_i\in\mathbb{R}^{256}.
\]

尺度只作为 gate 输入，不作为硬乘法，也不强制 medium/large queries 的 gate 接近零。通道级 gate 允许纹理、边缘、形状和背景抑制通道独立选择。Gate 不根据最终分类置信度筛选 query。

### 6.6 IdentitySafeResidual

最终输出：

\[
q_i'
=
q_i
+
\alpha\,
\mathbf g_i\odot
\tanh(W_od_i).
\]

残差幅度冻结为正值、有界的可学习标量：

\[
\alpha
=
\alpha_{\max}\sigma(a),
\qquad
\alpha_{\max}=0.25.
\]

正式训练初始化：

\[
\alpha_{\mathrm{train-init}}=10^{-3},
\qquad
W_o\sim\mathcal N(0,0.01^2),
\qquad
b_o=0.
\]

实现中：

\[
a_{\mathrm{init}}
=
\operatorname{logit}
\left(
\frac{10^{-3}}{0.25}
\right)
\approx-5.517.
\]

因此 shadow generator、采样器、point attention、shadow aggregation 和 channel gate 从第一次反向传播开始都能获得非零梯度。G0 恒等性通过测试态 `identity_override=True` 强制 \(\alpha=0\) 完成；该开关不用于正式训练、验证或论文结果。

`enabled=False` 时直接返回输入 object queries，不执行 detail branch，并要求输出 tensor 逐元素等于输入。

效果优先版暂不引入双分支增强/抑制残差。该扩展会同时增加两个 channel gates 和符号耦合，扩大 15 天主实验的搜索空间；只有单残差版本通过 G2 后才允许作为论文外的后续研究。

## 7. RT-DETR 集成边界

集成使用专用 decoder wrapper，而不是修改预测后处理：

```python
class SQDAPRTDETRDecoder(RTDETRDecoder):
    def forward(self, x, batch=None):
        c2_feature = x[0]
        encoder_features = x[1:]

        feats, shapes = self._get_encoder_input(encoder_features)
        ...
        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        dn_count = 0 if dn_embed is None else dn_embed.shape[1]
        dn_queries = embed[:, :dn_count]
        object_queries = embed[:, dn_count:]
        object_boxes = refer_bbox[:, dn_count:].sigmoid()

        object_queries, diagnostics = self.sqda_p(
            object_queries,
            object_boxes,
            c2_feature,
        )
        embed = torch.cat([dn_queries, object_queries], dim=1)

        return self.decoder(...)
```

约束：

- `encoder_features` 的内容和顺序与 stock RT-DETR-L 完全相同；
- C2 不进入 `_get_encoder_input()`，因此不把 baseline 的三层 encoder 改成四层；
- SQDA-P 不修改 `refer_bbox`；
- SQDA-P 不修改 denoising queries；
- SQDA-P 不修改 `attn_mask`；
- SQDA-P 不修改 encoder scores、decoder heads、postprocess 或 validator；
- 训练和推理调用相同的 SQDA-P forward；
- SQDA-P 参数必须进入 optimizer、state dict 和 checkpoint。

## 8. 优化协议

### 8.1 初始化

- 加载与 baseline 完全相同的 mature RT-DETR-L checkpoint；
- baseline 所有参数逐项复制；
- SQDA-P 除输出投影外的新层使用常规初始化；
- `IdentitySafeResidual.W_o` 使用标准差 0.01 的零均值正态初始化，bias 为零；
- learnable residual scale 初始化为 \(10^{-3}\)，并通过 sigmoid 参数化限制在 \((0,0.25)\)；
- 主实验不加载此前任何候选模块权重。

### 8.2 Phase G0：恒等性审计

不训练，执行以下检查：

1. `enabled=False` 时 object query 输出与输入逐元素相等；
2. `enabled=True`、`identity_override=True` 时 enhanced query 与输入逐元素相等；
3. native validator 的 Precision、Recall、mAP50、mAP50-95 与 baseline 一致；
4. decoder 原生预测 tensor 的 shape、顺序和数值一致；
5. denoising queries、reference boxes 和 encoder outputs 一致；
6. SQDA-P 参数出现在 `state_dict()`；
7. 正式训练设置 `identity_override=False` 且 \(\alpha=10^{-3}\) 时，一次 optimizer step 后各子模块获得有限且非零的梯度；
8. `identity_override` 不得由正式 train/val/predict CLI 暴露，避免把它变成结果选择开关。

若 native validator 任一主指标不能复现 baseline，G0 失败，禁止进入训练。

### 8.3 Phase G1：模块独立短训

冻结：

- backbone；
- hybrid encoder；
- query selection；
- stock decoder；
- stock classification and box heads；
- baseline normalization parameters and buffers。

仅训练 SQDA-P。

协议：

- seed 0；
- 3 epochs；
- stock RT-DETR classification、L1 box 和 GIoU losses；
- 不新增辅助损失；
- 不使用新的推理阈值；
- 使用与 baseline 相同的数据、输入尺寸、增强、batch size 和 validator；
- 每个 epoch 保存 checkpoint 和完整指标。

G1 是主成功筛查。主设计不依赖解冻 baseline 参数才能成立。

### 8.4 Phase G2：正式训练

只有 G1 通过成功门槛才进入 G2。

正式训练继续冻结 baseline，只训练 SQDA-P 10 epochs。选择 checkpoint 时先执行四项非下降硬约束，再在合格 checkpoint 中选择 mAP50-95 最高者。

“解冻 decoder 最后两层”只允许作为附加消融，不作为主方法结果；它不得替代冻结 baseline 的主实验。

## 9. 成功门槛

所有比较均使用同一 seed 0、同一 baseline checkpoint、同一 native Ultralytics validator 和同一输入设置。

四项硬约束：

\[
\Delta Precision\ge0,
\]

\[
\Delta Recall\ge0,
\]

\[
\Delta mAP50\ge0,
\]

\[
\Delta mAP50\text{-}95\ge0.
\]

二级指标在第一次训练前冻结为：

- native AP75；
- COCO area definition 下的 AP-small；
- 10 个 VisDrone 类别各自的 AP50-95。

同时至少满足以下一项：

- Precision、Recall、mAP50 或 mAP50-95 提升不少于 0.2 个百分点；
- 上述预注册二级指标中的至少一项提升不少于 0.2 个百分点。

若只有二级指标提升，论文必须准确写明具体指标，例如“保持总体性能并改善 AP-small”或“改善 pedestrian 类 AP”，不得写成总体 mAP 显著提升。

单个 checkpoint 只要有一项主指标下降，即不合格，不允许用平均值掩盖。

“82%–86%成功率”只作为带 G0/G1 止损流程的研发判断，不作为论文实验结论或统计保证。

## 10. 消融设计

冻结五组：

| 实验 | Sparse raw-C2 | Shadow queries | Channel gate | Small-gain residual |
|---|:---:|:---:|:---:|:---:|
| Baseline |  |  |  |  |
| Direct sampled-C2 residual | ✓ |  |  | ✓ |
| Single center shadow | ✓ | 1 | ✓ | ✓ |
| Five shadows without gate | ✓ | 5 |  | ✓ |
| Full SQDA-P | ✓ | 5 | ✓ | ✓ |

附加机制审计：

- SQDA-P enabled 但通过 identity override 将 residual 强制为零；
- shadow sampling 可视化；
- gate 按 tiny/small/medium/large 尺度分桶统计；
- query residual norm 分布；
- 参数量、FLOPs、显存和端到端延迟。

消融只用于解释模块贡献，不允许把失败的 ablation 重新包装成独立创新点。

## 11. 测试要求

### 11.1 单元测试

- 输入输出 shape 和 dtype；
- `num_shadows=5`、`points_per_shadow=4` 和每图 6000 点的 tensor 维度；
- 归一化坐标与 C2 grid 坐标转换；
- 图像边缘 sampling validity mask；
- 单个 shadow 全部越界和五个 shadow 全部越界时无 NaN、残差为零；
- tiny box 最小一个 C2 cell 的采样半径；
- gate 输出范围；
- residual cap；
- enabled/off 恒等性；
- identity override 恒等性；
- denoising/object query split 和重组顺序。

### 11.2 集成测试

- SQDA-P module 注册在模型树中；
- 参数进入 optimizer；
- 参数进入 checkpoint 并能严格 reload；
- stock RT-DETR loss 可以对 SQDA-P 反向传播；
- 一次反向传播后所有预期子模块有有限且非零的梯度；
- baseline 参数在 G1 前后逐项不变；
- native train/val/predict 模式均能运行；
- batch 中无 GT 时训练不崩溃；
- mixed precision 下无 NaN/Inf。

### 11.3 指标测试

- G0 native validator 精确复现；
- G1/G2 输出四项主指标和所有尺度指标；
- 保存原始 validator JSON、训练配置、checkpoint SHA-256 和 git commit；
- 禁止用自定义预测融合评测替代 native validator。

## 12. 风险与停止条件

### 12.1 G0 不能复现 baseline

说明集成改变了原 decoder 输入、query 顺序或数值。只允许修复集成错误，不允许开始调学习率或训练。

### 12.2 Precision 下降

优先检查 channelwise consistency gate 是否失效、raw-C2 detail 是否被背景纹理主导。只允许一次结构审计；不得通过改变推理置信度阈值掩盖。

### 12.3 Recall 下降

检查 residual 是否破坏 reference-query 对齐以及 gate 是否对 tiny query 过度关闭。不得删除原 query 或改变 query 数量补救。

### 12.4 mAP50-95 下降而 mAP50 上升

说明 detail residual 改善了发现能力但损害定位。检查 sampling 半径和 residual norm；不得通过更换 IoU loss 把问题改写成第二创新点。

### 12.5 G1 无有效提升

若四项主指标不下降但所有指标提升均小于 0.2 个百分点，判定证据不足。完成一次 full SQDA-P 与 single-center shadow 的对照后仍无提升，则停止 SQDA-P，不进入 10-epoch 正式训练。

### 12.6 任一主指标下降

若没有 checkpoint 同时满足四项非下降，SQDA-P 判定失败。不得用多 seed 平均、修改验证阈值或只报告有利指标绕过。

## 13. 15 天交付节奏

| 天数 | 工作 |
|---|---|
| Day 1 | 接口实现、单元测试、G0 tensor 恒等性 |
| Day 2 | native validator G0、梯度和 checkpoint 闭环 |
| Day 3–5 | G1 三 epoch 模块独立短训 |
| Day 5 | 四项非下降硬闸门 |
| Day 6–10 | 通过后执行 G2 十 epoch 正式训练 |
| Day 11–12 | 五组消融、复杂度和可视化 |
| Day 13 | 可选 UAVDT 外部验证；主实验未通过则不做 |
| Day 14 | 模型图、表格和失败案例 |
| Day 15 | 方法、实验与局限性正文 |

## 14. 完成定义

SQDA-P 只有同时满足以下条件才可作为论文网络模块：

1. G0 native validator 与 baseline 一致；
2. SQDA-P 参数进入 forward、optimizer、state dict 和 checkpoint；
3. 模块从 stock detection loss 获得梯度；
4. 训练和推理使用同一 forward；
5. 最终仍为原生 300-query 预测集；
6. Precision、Recall、mAP50、mAP50-95 四项均不下降；
7. 至少一个主指标或预注册小目标指标提升不少于 0.2 个百分点；
8. 完成至少四个有效消融和复杂度报告；
9. 论文如实报告失败案例、延迟和适用边界；
10. 所有结果可由配置、checkpoint、日志、哈希和 git commit 复核。

未满足任一条时，不得宣称 SQDA-P 已验证成功。

## 15. 主要参考边界

- Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection*, CVPR 2024 / arXiv:2304.08069.
- Yang et al., *QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection*, CVPR 2022.
- Huang et al., *DQ-DETR: DETR with Dynamic Query for Tiny Object Detection*, ECCV 2024.
- Gao et al., *EASE-DETR: Easing the Competition among Object Queries*, CVPR 2024.
- Zhang et al., *Dense Distinct Query for End-to-End Object Detection*, CVPR 2023.

正式论文必须引用这些近邻工作，并依据最终实现继续补充同义词与引用链检索；本设计冻结不等于宣称完成全球唯一性或专利检索。
