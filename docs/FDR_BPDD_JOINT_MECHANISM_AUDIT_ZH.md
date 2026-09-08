# FDR 与 BPDD 联合机理审查及候选调整

日期：2026-09-08。原代码基准：94da3f65；本次本地候选尚未部署、未进行数据集训练。

## 1. 当前判断

应把 FDR 的表示/累计梯度结构与 BPDD 的监督目标一起审查。已复现两个交互风险：KL 的局部更新可能背离定位目标；同 assignment 的深层蒸馏会通过累计 logits 直接更新不同 assignment 的浅层残差。当前 +0.008 pp 的单次 best 差值不能确认它们是真实退化原因。

另外，上次将 F 相对历史的 Precision/Recall 差值直接归因于几何修正过强；跨版本差值只能作为观察线索，P/R 也不能单独说明分类、定位或分数排序谁发生了变化。

## 2. 完整处理流程

```text
图像 → backbone/neck → encoder proposals，得到初始参考 r0
                           ↓
第 l 层 Transformer：h_l = Decoder_l(h_(l-1), stopgrad(b_(l-1)))
                           ↓
回归输入：h_l + stopgrad(h_(l-1))（第1层历史项为0）
                           ↓
四边残差 logits Δz_l → 累计 z_l = Σ_(k≤l) Δz_k
                           ↓
p_l = softmax(z_l)，μ_l = Σ_i p_l,i W_i，W∈[-4,4]
                           ↓
使用固定 r0 解码 → FP32 中心/宽高 → extent floor → b_l
                           ↓
匹配 + L1/GIoU/VFL；匹配框的 FGL 权重由 LRS 调整
                           ↓
BPDD：source/future 同 query、同 GT → 逐边 NLL softmin 混合老师
      → NLL better-only reliability → 可选完整框 IoU 门控 → 蒸馏损失
```

注意力参考随层变化，但所有分布的解码参考仍是 r0。这不是坐标错误：它让各层分布使用共同坐标；若改成动态解码参考，必须同时重映射累计分布和跨层老师，不能直接把代码中的 r0 换成上一层框。

相关源码：`src/fdr_head.py` 的 decoder forward；`src/fdr_math.py` 的 weighting_function / decode_feasible_fdr_boxes / bbox2distance；`src/fdr_loss.py` 的 layerwise_reliability_shrinkage / _fgl_group；`src/bpdd_loss.py` 的 assignment_consistent_bpdd_loss。模型整合见 `src/rtdetr_fdr.py`。

## 3. FDR 的机制评估

### 3.1 累计残差：保留有效结构，检查其监督路径

`z_l = z_(l-1) + Δz_l` 不对历史 z 停止梯度。直接加法路径上，浅层残差接收所有后续被监督累计输出的梯度和。它本来可促进多层联合学习；不是单凭存在就能判为 bug。

但 BPDD 的 assignment 过滤只检查 source/future，不检查 source 之前的层。三层同一个 query 匹配 A/B/B 时，第 2 层可向第 3 层蒸馏 B，第 1 层残差仍直接收到这一梯度。

生产损失 + 真实累计辅助函数反例：candidate_source_matches=2，stable_source_matches=1；第 1、2、3 层残差梯度范数分别为 **0.01669685、0.01669685、0**。这证明存在未被 assignment 条件覆盖的直接回传路径，不证明 A/B 监督在每个真实样本上必然冲突。

累计 logits 的 softmax 等价于各残差 exp 因子的归一化乘积，但熵不一定随层单调下降。需要真实记录 logit 范围、熵、边界质量、期望 Jacobian 范数以及残差更新量，才能判断饱和是否成为瓶颈。现在不据此重设层数或温度。

### 3.2 固定参考 + 有限支持：增加层数不能无限扩展可达范围

支持点为 [-4,4]，reg_scale=4。横向原始解码：

`x_left = cx0 - (2+μ_left) w0/4`

`x_right = cx0 + (2+μ_right) w0/4`

因此 x_left∈[cx0−1.5w0,cx0+0.5w0]，x_right∈[cx0−0.5w0,cx0+1.5w0]；合法未触发 floor 的最大宽为 **3w0**。纵向同理。额外 decoder 层可以改变分布，但不能突破固定支持；数值 floor 的极小框不属于放大可达范围的机制。

例：r0 宽 0.2，GT 同中心宽 1.0，需要左右原始距离均为 8，超出支持 +4。FGL 会把不可表示的目标编码到边界，BPDD 的边界 NLL 改善不等价于能够达到 GT。

LRS 的 `representable_fgl_targets` 只是保守屏蔽边界附近目标的“权重收缩”，**没有禁用这些目标的基础 FGL**。此前笼统叫“可表示性防火墙”容易误解。新 M 候选独立检查未截断原始距离，避免把超范围 GT 当作可靠均值方向监督。

若真实匹配样本的支持越界很少，不应更改 FDR 表示。若明显集中在 tiny/尺度错配上，再单独比较更合适的参考、支持尺度或分布重映射；它们应有无 BPDD 配对，不与当前候选一并改变。

### 3.3 几何修正：前向有效，但 floor 区域采用代理梯度

当前 FP32 解码与中心/宽高直接计算解决数值问题；extent floor 保证正尺寸。不能因一次历史对比差值删除这些保护。

当 floor 激活，真实前向宽度对微小距离变化可保持不变，但当前实现保留 raw width 的反向导数。实测 r0 宽0.2、四边距离−3时：宽度约 **0.00005**；宽度对左边距离的 autograd 导数 **0.05**，中心差分导数 **0**。这是代码明确设计的 straight-through 行为，不是普通 hard clamp 的导数。

所以独立 logits 的定位方向分析不能直接推广到 floor 区域 IoU。需要按 matched/unmatched、normal/DN、层、目标尺寸记录 floor，并分别评估。当前候选不修改生产几何公式。

### 3.4 LRS / FGL 与 BPDD 的监督预算并不对等

LRS 在同图同层 eligible 匹配框内：`w_i=(1−α_l)IoU_i+α_l mean(IoU)`，α_l=0.25(1−l/5)，最后层不收缩。实测首层 [0,0.8] → [0.1,0.7]；全0 → 全0。它重新分配质量权重，不额外创造质量，也不能挽救全组零 IoU。

FGL：四边损失按每层匹配框数归一化，再对六层求和。BPDD：按所有 source 层候选匹配框数×4整体平均，且再乘 reliability。即便二者配置同为0.15，实际预算也不同。

假设六层每层恰有M个匹配，某条 source 边的 BPDD 系数是 `0.15*r/(20M)`，同层 FGL 系数是 `0.15*w/M`。这里只比较标量系数，不代表实际梯度范数比；KL/交叉熵/Huber 的梯度和累计网络路径不同，不能据此机械乘20或上调权重。

## 4. BPDD 的目标与更新方向风险

NLL 更好而解码 IoU 更差的问题已有反例和旧 IoU 候选。新增反例更强：老师 NLL 和 IoU 都更好，KL 局部更新仍伤害学生定位。

实际 33-bin 中只给索引16、17、32较大质量，其余各1e−8后归一化：

| 项目 | 学生 | 老师 |
|---|---:|---:|
| 三个主 bin 概率 | 0.0099686 / 0.9107678 / 0.0792636 | 0.1856762 / 0.8113069 / 0.0030168 |
| 四边相同期望 | 0.3862635 | 0.0737182 |
| 解码 IoU | 0.7024626 | 0.9301661 |

旧 IoU 候选 active_edge_ratio=1，loss=0.04153652；对独立学生 logits 作 `z←z−∇L` 后，IoU 降至 **0.7023848**。这是固定合成 logits 的单位步，不是训练学习率或真实网络更新。

因为 `∇_z KL(q||p)=p−q`，而 `∂μ/∂z_i=p_i(W_i−μ)`，完整老师期望更好并不能约束两者内积的方向。完整终点质量门控忽略了 softmax Jacobian、边权重和共享参数。

## 5. 本次实现的两个独立选项

### R：只对当前层残差直接回传

新增 `residual_gradient_only`，默认false。BPDD 使用 `stopgrad(z_l)+(r_l−stopgrad(r_l))`，其中 r_l=z_l−z_(l−1)。前向 logits 保持相同，累计链上直接历史梯度取消；FGL/L1/GIoU仍使用原始累计结构。

三层反例中，R 的 loss 与原版完全相同，第1层残差直接梯度变为0，第2层梯度保持，第3层仍为0。此处用独立残差检验直接路径；共享 Transformer 特征梯度、其他损失以及下一步累计输出的变化都仍存在，不能称“完全隔离所有浅层参数”。

### M：把 BPDD 目标改为 GT 同向的分布期望

新增 `distribution_objective: mean_huber`，默认仍为kl。保留同 assignment 未来老师混合和 NLL 门控；进一步要求 GT 原始边距离可表示，老师期望在学生与 GT 之间且误差严格减小，再检查有效四边完整框 IoU。

蒸馏损失为加权 `SmoothL1(μ_student, stopgrad(μ_teacher), beta=1)`，保持原候选分母。NLL 是额外置信筛选，FGL继续监督33-bin分布形状。两个条件可能重复或降低覆盖，需真实数据验证。

独立单边 logits 的一阶更新方向可证明朝老师期望移动，且当前筛选保证它朝GT移动。随机512组×4边检查了所有保留边的GT平方边误差方向导数为负；固定KL反例上也朝真值改善。该性质不保证完整框IoU、共享网络、Adam、有限步或最终mAP改善。

这属于“分布期望蒸馏”，不是全分布KL的等价替代，也有退化成额外位置回归的风险。后续必须加入相同预算的直接GT期望辅助损失控制；否则即使涨点，也不能说明增益来自跨层教师知识。

### 为什么没有选直接重做 FDR

核心 FDR 已有接近0.297的可用基座；本次证据首先指向监督与累计路径之间的不匹配。动态参考/支持扩展、去累计、EMA外部教师和直接改成Wasserstein都会引入额外变量，且无法自动解决当前全部问题。

定位自蒸馏本身有既有研究，例如 [D-FINE 原论文](https://arxiv.org/abs/2410.13842)；用Wasserstein替代KL也已有 [NeurIPS 2024研究](https://papers.neurips.cc/paper_files/paper/2024/file/78526d7ad4a2532bd91416e948b9644c-Paper-Conference.pdf)。这里以项目内机制修正为目标，未做穷尽新颖性检索，不宣称全新首创，也不据实现来源裁定作者独立提出过程。

## 6. 验证与下一步

- 最终一次联合回归 **149 passed in 27.19s**，包含FDR、FGL/LRS、BPDD、模型集成与launcher，以及CUDA FP32/FP16/BF16三项。新增机制文件共19项通过。
- 新选项加入前：机制测试2项复现通过、12项因选项尚不存在失败；实现后全部通过。
- 相同seed构造原G与新候选模型，所有state_dict键和张量逐项相同；criterion保留LRS alpha=0.25。新增选项不增加模型参数。
- 已验证默认模式前向loss和logit梯度一致；支持外GT、越过GT老师、空匹配、禁用、错误选项均有测试。
- CUDA只是损失原语/梯度检查，不是完整网络AMP训练，更不是VisDrone预检或涨点证明。

最终验证命令（工作目录为当前代码仓）：

```text
C:\uav_env\Scripts\python.exe -m pytest tests/test_bpdd_joint_mechanism.py tests/test_bpdd_iou_gate.py tests/test_bpdd_loss.py tests/test_bpdd_fdr_criterion.py tests/test_fdr_head.py tests/test_fdr_loss.py tests/test_fdr_feasible_geometry.py tests/test_lrs_system_launcher.py tests/test_bpdd_fdr_integration.py -q
```

`git diff --check` 通过（Git提示Windows行尾转换，不是检查失败）。

候选配置：`configs/rtdetr-l-lrs-fdr-bpdd-joint-candidate.yaml`。通用LRS trainer的get_model会使用固定ARM_CONFIGS，所以直接把此YAML交给旧G入口可能被覆盖；本次没有把它注册成正式新臂。当前交付是可构造、可测试的本地候选，不是新的100轮部署包。

下一步先在可用早/中/后期checkpoint上固定真实训练批次：

1. 统计每层原始支持越界、floor、期望Jacobian、匹配切换，并按尺寸/normal-DN区分。
2. 统计原BPDD、IoU候选、R/M的覆盖率，以及“老师终点改善但小步方向变差”的频率。分布直方图的分母必须包含未通过gate的source匹配。
3. 固定同一参数组，测BPDD/FGL/L1/GIoU梯度范数和cosine；额外测真实Adam状态的一步更新。使用checkpoint拷贝，记录并恢复模型buffers/optimizer/RNG，以免诊断改变正式状态。
4. 单因素分解R与M，然后比较组合；同0.15不代表同预算。预算选择在训练集固定批次上完成，验证集用于预先约定的性能比较。
5. 只有真实覆盖与梯度证据支持才安排正式新臂。FDR表示是否要改，由支持越界和饱和数据决定。全部真实频率、逐类/AP75/tiny变化、跨seed效果目前未测。
