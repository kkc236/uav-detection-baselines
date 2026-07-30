# SQDA-ABDR 冻结设计规格

日期：2026-07-31
状态：用户确认设计；尚未实现或训练
基础：Ultralytics RT-DETR-L + matched VisDrone baseline
目标：修复 SQDA-SGC G2 的“Recall/mAP 微升而 Precision 微降”现象，不改变训练协议、预测集合或推理后处理。

## 1. 可复核的触发证据

SQDA-SGC G2 在完整 VisDrone 验证集、`imgsz=640`、`batch=8`、`max_det=300`、`NMS=False`、AMP 下完成 10 epoch 精确复验。保留 checkpoint 中，`completed_epoch9` 相对同一成熟 baseline 的变化为：

| 指标 | 差值 |
|---|---:|
| Precision | -0.00053845 |
| Recall | +0.00014790 |
| mAP50 | +0.00005305 |
| mAP50-95 | +0.00003102 |
| COCO AP-small | +0.00002474 |
| COCO AP-medium | -0.00006061 |
| COCO AP-large | -0.00061589 |

所以 G2 **不通过**“四项主指标均不下降”的预注册门。冻结张量审计同时证明 941 个 stock 张量全部逐项相同；问题不是 baseline 污染，而是新增残差的效果方向尚不稳定。

类别 AP 的正负变化混合，不能证明 Precision 损失由某个类别单独造成。下述“语义方向扰动”只是一项与证据相符、可被消融验证的机制假设，不作为既成因果结论。

## 2. 文献复核与取舍

1. [RT-DETR](https://arxiv.org/abs/2304.08069) 将分类和定位共同视为 query 质量的一部分。因此不能为补 Precision 而篡改其 Top-300 selection、query 数或预测排序。
2. [SAM-DETR](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Accelerating_DETR_Convergence_via_Semantic-Aligned_Matching_CVPR_2022_paper.html) 表明 query 与图像特征的语义对齐是合理的结构动机；本设计只借鉴“对齐应显式建模”的原则，不复制其 salient-point matching。
3. [Decoupled DETR](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Decoupled_DETR_Spatially_Disentangling_Localization_and_Classification_for_Improved_End-to-End_ICCV_2023_paper.html) 说明分类和定位适合不同空间证据。本设计不增加双 decoder、双 attention 或 alignment loss，而是在 decoder 前对同一 query 的残差方向施加轻量归纳偏置。
4. [QR-DETR](https://openaccess.thecvf.com/content/ACCV2024/html/Senthivel_QR-DETR__Query_Routing_for_Detection_Transformer_ACCV_2024_paper.html) 和 [QueryDet](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf) 分别改变 decoder query 路由和候选位置计算，均越过本实验“固定原生 300 query”的边界，明确排除。

此前的 G1S 已实测否决“第三个 no-write softmax 分支 + 全残差正交化”：no-write 退化，且 Precision 更差。ABDR 绝不重新引入该结构。

## 3. 模块定义

模块全称：**Agreement-Bounded Directional Residual (ABDR)**，中文：**一致性有界方向残差**。

ABDR 位于 SQDA-SGC 当前的 `semantic` / `geometry` descriptor、16 组两路 softmax gate 之后，并在 stock RT-DETR decoder 之前写回一次。它保留：

- 原 C2 角色采样、外环 context 的只读语义调制；
- 原生 300 object queries 与其顺序；
- detached reference boxes；
- 现有两路 16 组语义/几何竞争门；
- 一个 RMS-bounded LayerScale residual。

它不保留当前把两路证据直接拼接后由一个 `fusion` 投影任意混合的写回方式。

设原 query 为 `q`，context 调制后的语义描述子为 `s`，几何描述子为 `g`，两路 group gate 展开后为 `G_s,G_g`。令

\[
u=\operatorname{normalize}(\operatorname{stopgrad}(q)),
\]

\[
z_s=W_s(G_s\odot s),\qquad z_g=W_g(G_g\odot g),
\]

\[
r_s=\langle z_s,u\rangle u,\qquad
r_g=z_g-\langle z_g,u\rangle u.
\]

其中 `r_s` 只沿冻结 query 的方向变化，作为语义强度校正；`r_g` 与该方向严格正交，作为边界/定位细节校正。两者内积理论上为零，避免两个残差分量在同一子空间相互抵消。

### 3.1 有界一致性调制

从 query、两种描述子、其逐元素积与绝对差、三种相似度以及 `log(w),log(h)` 构造输入：

\[
h=[\mathrm{LN}(q);\mathrm{LN}(s);\mathrm{LN}(g);
\mathrm{LN}(s)\odot\mathrm{LN}(g);
|\mathrm{LN}(s)-\mathrm{LN}(g)|;
\rho_s;\rho_g;\rho_c;\log w;\log h].
\]

一个 `Linear(5D+5,64) → SiLU → Linear(64,1)` 产生调制值：

\[
a=a_{\min}+(1-a_{\min})\sigma(\operatorname{MLP}(h)),
\quad a_{\min}=0.80,\quad a_{\mathrm{init}}=0.98.
\]

最终只有一次写回：

\[
f=\operatorname{RMSBound}(a(r_s+r_g)),\qquad q'=q+\alpha f.
\]

`α` 沿用 SQDA-SGC 的既有单一有界 LayerScale；若五个写回角色均无效，`f` 严格为零。

`a` 永远大于 0.80，因此它不是 no-write 分支、不会筛掉 query、不会 early exit，也不会让新增模块把所有残差塌缩为零。它只在两种证据相互不支持时温和衰减同一个残差。

## 4. 初始化、梯度与复杂度

- `W_s`、`W_g` 均为 `Linear(D,D)`，权重初始化 `Normal(0,0.01)`，bias 为零；
- 一致性 MLP 的末层权重为 `Normal(0,0.01)`，bias 设为令 `a_init=0.98` 的反 sigmoid，确保所有角色从首步获得梯度；
- `u` 对 stock query stop-gradient，避免把“方向参考”变成可借由 frozen detector 参数逃逸的训练通道；新增模块参数仍完全可训练；
- 不增加第二个 LayerScale、第二个 residual、辅助 head 或辅助 loss；
- 两个 `D×D` projector 取代原 `2D→D` fusion projector，主投影参数量近似不变；仅新增约 82K 的一致性 MLP 参数，远低于 1M 约束；
- 不产生跨 query attention、动态 query 路由、动态采样半径或额外高分辨率特征图。

## 5. 内部冲突审查

| 潜在冲突 | ABDR 约束 | 预期结果 |
|---|---|---|
| 语义残差旋转 query，影响 Precision | 语义仅可沿 `u` 写入 | 降低类别方向扰动，而非承诺校准成功 |
| 几何证据改变类别强度 | 几何仅可写入 `u` 的正交补 | 降低几何对语义幅度的直接干扰 |
| 两路残差抵消 | `r_s ⟂ r_g` 且只做一次相加 | 无同子空间显式对消 |
| G1S no-write 塌缩 | `a∈(0.80,1)`，不产生第三分支 | 结构上排除完全放弃写入 |
| context 压低边界证据 | context 仍只调制 `s`，不调制 `g` | 维持原职责隔离 |
| 变成推理算法 | train/val/predict 调用同一 adapter forward | 属于参与反向传播的网络层 |

该设计不是“保证 Precision 上升”的机制；它只是把当前未经约束的语义–几何混合，改为有可检验职责边界的单残差结构。

## 6. 实现前测试与门控

### 6.1 单元与集成测试

1. 关闭模块或 `identity_override=True` 时，输出逐元素等于输入；
2. `r_s` 与 `u` 平行，`r_g` 与 `u` 的点积在 FP32 容差内为零；
3. `r_s` 与 `r_g` 的点积在 FP32 容差内为零；
4. `a` 严格在 `(0.80,1)`，无效 context 不产生 NaN；
5. 所有写回角色无效时残差严格为零；
6. 一步反传后，语义 projector、几何 projector、一致性 MLP、原 group gate、context reliability 和 LayerScale 均有有限非零梯度；
7. 941 个 stock 参数和 buffer 在一步训练前后逐项不变；
8. checkpoint 可严格 reload，DN query 不被 ABDR 修改。

### 6.2 筛选协议

训练数据、增强、输入尺寸、batch、seed、AMP、max detections、NMS、baseline checkpoint 及评估器均沿用 G2。不得因结果改变阈值、损失、query 数、采样点数或优化器。

1. G0：严格 identity、Top-300 与 stock 输出一致；
2. G1：从成熟 baseline 独立训练 3 epoch；
3. G2：仅当 G1 无数值异常、冻结审计通过后，从同一 baseline 独立训练 10 epoch；
4. 每个保留 checkpoint 都对照精确 baseline 的 Precision、Recall、mAP50、mAP50-95；任一主指标下降即不通过；
5. 对最优候选补算 COCO AP/AP50/AP75/AP-small/AP-medium/AP-large、逐类 AP、冻结张量审计和资源开销；
6. 只有 G2 严格通过才允许从带 optimizer state 的 checkpoint 条件续跑至 100 epoch。

## 7. 论文表述边界

可主张：ABDR 在冻结 RT-DETR 的原生 object query 上，以语义方向校正、几何正交校正和非零一致性调制形成一个可训练的单次残差模块。

不可主张：语义方向等同于分类 head、正交方向等同于定位 head、必然改善所有尺度或保证任意 seed 成功。论文结果必须以通过后的精确评估为准。
