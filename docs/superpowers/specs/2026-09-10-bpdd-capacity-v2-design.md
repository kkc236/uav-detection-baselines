# BPDD Capacity v2：独立局部精修教师与质量门控蒸馏

## Material Passport

- Origin Skill: brainstorming + academic-research-suite / experiment-agent / plan
- Origin Date: 2026-09-10
- Version Label: bpdd_capacity_v2_design
- Verification Status: APPROVED-CONCEPT / UNIMPLEMENTED / UNMEASURED
- Parent evidence: workspace `diagnostics/2026-09-10-bpdd-capacity-redesign-proposal.md` and `diagnostics/2026-09-10-bpdd-capacity-design-audit.md`
- User decision: 采用“独立局部精修头 + 质量门控 BPDD”，保留 LRS-GFDR，修复 v1 工程问题后重新训练。

## 1. 目标与边界

本设计用于验证两种可分离增益：局部视觉精修本身是否改善 LRS-GFDR，以及质量门控蒸馏是否在同一增容架构上提供额外提升。主目标仍是 VisDrone Formal100、seed 0 下相对配对 LRS-GFDR 提升 best-val mAP50-95；`+0.5 pp` 是研发目标，不是预测、承诺或统计显著性界线。

以下内容保持不变：六层 FDR 累积分布、每边 33 bins、同一初始参考、非均匀支持、LRS `alpha=0.25`、可行几何解码、300 个普通 query、输入 640、batch 8、MuSGD、固定 AMP、数据与增强协议。v2 不加入 FIA，不改 backbone 宽度，不从 v1 的 OOM checkpoint 续训。

v1 保留为“简化增容候选 / OOM 中止”证据；不能重命名成 v2，也不能用其第 1 轮指标评价 v2。

## 2. 模型结构

### 2.1 图结构与特征来源

检测 decoder 继续读取节点 `[21, 24, 27]` 的 P3/P4/P5，保持原路径与公共初始化不变。FDR head 的 YAML 输入扩展为 `[1, 21, 24, 27]`：

- 节点 1：stride-4 P2，仅供局部精修；
- 节点 21：stride-8 P3，同时供原 decoder 和局部精修；
- 节点 24：stride-16 P4，同时供原 decoder 和局部精修；
- 节点 27：stride-32 P5，仅供原 decoder。

head 必须显式拆分 local features 与 decoder features，不采用 forward hook 或跨批次缓存。模型构造测试必须核对实际 stride/channel，不能只相信 P2/P3/P4 命名。

### 2.2 独立 Local Boundary Expert

局部精修以原第六层 normal-query 隐状态、分布、分类 logits 和解码框为输入。采样框坐标停止梯度；P2/P3/P4 特征值保留梯度。

- 保留现有每尺度 40 个边界邻域点和 1 个中心点，共 41 点；三个尺度共 123 tokens/query。
- 先在原通道特征图上执行 `grid_sample`，再对采样 token 使用 Linear 投影到 256 维；禁止先把完整 P2 映射成 256 通道。
- 使用尺度/坐标位置编码以及四边熵、四边期望作为 query 条件。
- 使用两个 256 维、8 heads、FFN=1024、dropout=0 的局部交叉注意力块。
- 输出 132 维分布残差和 `nc` 维分类残差；两个输出层权重和偏置为零，使初始精修输出与原第六层严格一致。
- 所有私有参数由显式 CPU `torch.Generator(seed=30000)` 初始化；构造过程中禁止调用会改变全局 CPU/CUDA RNG 的 `torch.manual_seed`。CPU、已初始化 CUDA 和尚未初始化 CUDA 三种场景都要测试。

训练时保存两套 normal-query 输出：原六层输出保持不变，精修输出单独缓存。推理只使用精修后的分布、框和分类。原第六层不能被覆盖或从训练 stack 中删除。

### 2.3 DN 隔离与显存约束

精修只处理最后 300 个 normal queries。DN 前缀沿原 FDR 路径训练，不进入局部采样、精修直接损失或蒸馏。

normal query 按 32 个一组处理。每组的两个注意力块使用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`；dropout 为零，因此可关闭 checkpoint RNG 保存，但必须用输出与梯度等价测试确认。checkpoint 是以计算换显存的措施，不能声称不增加运行时间。

模型不得跨迭代保存带计算图的 loss：`last_fdr_losses`、教师统计和运行记录一律 detach。v1 的 query chunk 仅限制瞬时张量、不能限制全部反向激活，本设计不能再把“减小 chunk”当作单独的 OOM 修复。

## 3. 监督与匹配

### 3.1 原路径监督

原 encoder 与六层 decoder 保持现有 stock detection loss 和 FGL。原第六层的 Hungarian assignment 是唯一的精修身份权威。

### 3.2 精修直接监督

精修头复用原第六层 assignment，不重新运行 matcher。对相同匹配计算分类、L1、GIoU 和 FGL；未匹配 query 的分类监督沿用同一固定 assignment 的 stock 规则。精修直接监督总系数固定为 `1.0`，FGL 权重与基座相同为 `0.15`。B、C 两个增容臂的该项完全一致。

精修头必须靠 GT 直接监督学习，不能只模仿原 FDR；否则它没有成为更强教师的独立信息来源。

## 4. Quality-Gated BPDD

### 4.1 定位教师与共同支持

C 臂以停止梯度的精修分布为教师，原六层分布均可作为学生。对每个源层，只保留该层 assignment 与原第六层 assignment 中 `(batch, query, GT)` 完全一致的匹配。

用未截断的真实 target distance 检查每条边是否位于当前非均匀支持 `[W(0), W(32)]`；超支持边关闭分布蒸馏，但保留原检测框损失。不能用已经夹到边界 bin 的标签反推“可表示”。

对每条可表示边计算学生、教师的插值目标 NLL：

\[
a_e=\max(\mathrm{NLL}_{S,e}-\mathrm{NLL}_{T,e}-m_{loc},0),\qquad
r_e=\frac{a_e}{a_e+\tau_{loc}}.
\]

第一版冻结 `m_loc=0.02`、`tau_loc=0.1`。教师始终 detach，定位项为共同支持上的 teacher-to-student KL。

### 4.2 实际解码几何门控

对每个匹配 query，将 `r_e>0` 的边换成教师分布、其余边保留学生分布，形成“实际将要蒸馏的混合框”。使用生产版可行几何解码分别计算学生框与混合框 IoU。只有 `IoU_hybrid > IoU_student` 时保留该 query 的定位 KD；不能只检查完整教师框，也不能以分布 NLL 代替框质量。

该门控排除已复现的“NLL 更低但 IoU 明显更差”反例，但不保证一次联合优化后 IoU 必然提高。运行证据必须继续记录定位 KD 与主定位损失在 decoder/FDR/精修参数上的梯度范数及余弦。

### 4.3 分类蒸馏

仅对上述身份一致的 matched normal queries 计算。用一热 GT 对所有 `nc` 类计算学生与精修教师的 BCE；仅当教师 BCE 比学生至少低 `m_cls=0.02` 时启用 Bernoulli KL。分类可靠性同样使用绝对有界形式，`tau_cls=0.1`。

分类与定位的门控、损失和统计独立；分类置信度不能替代定位质量，分类 KD 不覆盖 unmatched negatives 或 DN queries。

### 4.4 时序与权重

- epoch 1–10：只训练原路径和精修直接监督，定位/分类 KD 系数为 0；
- epoch 11–20：KD 系数线性增加；
- epoch 21–100：定位 KD 峰值 `0.15`，分类 KD 峰值 `0.10`。

epoch 使用完成轮次的一基计数；恢复训练时由 checkpoint epoch 恢复同一调度位置。以上是预注册筛选默认值，不宣称最优，正式配对过程中不根据验证集涨跌临时更改。

每个源层先按其 eligible 元素归一化，再对有 eligible 元素的源层取均值，避免匹配数量多的层独占损失预算。没有 eligible 样本时返回连图的精确零。

## 5. 优化器、记录和恢复

梯度裁剪分为三个互斥且完备的组，每组 `max_norm=10`：

1. common：backbone、encoder、原 decoder 与分类路径；
2. fdr_private：六个分布头和 pre-box 私有参数；
3. expert_private：局部投影、位置/分布编码、注意力块和精修输出头。

每个可训练参数必须恰好出现在一个组。优化器类型、学习率、权重衰减与 Formal100 协议不变。

每轮至少记录：精修相对原第六层的 IoU/NLL/BCE 改善（筛选前和筛选后）、支持越界率、身份一致率、定位/分类 active ratio 与可靠性、各项 loss、三组总梯度、分项梯度范数/余弦、几何统计、CUDA allocated/reserved/max allocated、AMP 是否跳步。字段名和 method revision 必须改为 v2，不能继续写旧 AC-BPDD revision。

checkpoint 必须包含 expert 与调度状态；resume 需验证代码/config/初始协议身份，并在合成 batch 上验证保存前后输出严格一致。

## 6. 实验矩阵与归因

| 臂 | 独立局部精修 | Quality-Gated BPDD | 作用 |
|---|---:|---:|---|
| A | 否 | 否 | 已有 LRS-GFDR 配对基座 |
| B | 是 | 否 | 测新增容量、P2 局部信息和直接监督 |
| C | 是 | 是 | 测蒸馏在相同增容模型上的额外贡献 |

关键差值：`B-A` 是精修增容收益，`C-B` 才是 BPDD 收益，`C-A` 是完整系统收益。已有原 AC-BPDD 可作历史参考，但不能替代 B/C 严格配对。若只有 B 提升，论文不得把 B 的提升归给 BPDD。

A/B/C 共享参数必须来自同一 initial-state 内容；B/C 的 expert tensors 必须字节一致。新增模块不得改变共享参数、数据顺序、augmentation 或公共 RNG 状态。B 先运行；B 正常完成后再运行 C，两者使用独立 create-only 输出目录，不覆盖 v1。

## 7. 分级验证门槛

### Gate 0：CPU 单元与集成

- 41 点坐标、越界 mask、raw target support、混合框门控和分类门控有确定性测试；
- 覆盖已复现的 NLL/IoU 反例，要求定位 KD 精确为零；
- 原六层输出在 expert 开启前后字节一致，精修初始输出与原第六层一致；
- DN 不进入 expert，normal 恰为 300；
- B/C 共享与 expert 初始化一致，CPU/CUDA RNG 均不变化；
- direct loss 固定使用原末层 assignment，matcher 调用次数不增加；
- 参数分组互斥完备，checkpoint roundtrip 与恢复调度通过。

### Gate 1：CUDA 与真实密集批次

在正式 4090、batch 8、640、AMP 下至少覆盖空 GT、普通批次和数据集中高目标数批次；完成 forward/backward/optimizer step，无非有限数或 AMP skip。连续运行足够批次覆盖 optimizer 状态和 allocator 稳态，不能只做一个合成 batch。峰值显存必须低于设备总量并保留至少 1 GiB 余量；否则不启动 Formal100。

### Gate 2：训练启动证据

B 臂先启动，确认首个完整 epoch、验证结果、runtime JSONL、optimizer evidence、权重文件与 GPU 进程。B 完成后按相同源代码和协议启动 C。异常不自动重试或篡改 batch；保留日志并报告。

### Gate 3：结果解释

主指标为统一验证器的 best mAP50-95，同时报告 P/R、mAP50、最后十轮均值、参数量、实测显存和推理延迟。单 seed 的小差值只能报告为观察值；若候选达到研发目标，再安排多个配对 seed 验证稳定性。

## 8. 失败处理与论文边界

- B 不优于 A：先判精修教师能力不足，不能靠放大 KD 权重挽救；检查 P2 噪声、采样有效率、直接监督和固定支持越界。
- B 优于 A、C 不优于 B：保留精修贡献，判当前 BPDD 未提供额外价值。
- C 优于 B：才能把 `C-B` 归因于 Quality-Gated BPDD，并进一步做分类 KD、几何门控等内部消融。
- 再次 OOM：保留证据，先审计实际 query 数、带图缓存和保存激活；不把降低 batch 后的结果冒充原 Formal100 配对。

工作名固定为 `bpdd-capacity-v2`。在获得完整配对结果和相关工作审查前，不将其正式改名为论文创新，也不宣称成功率或新颖性已得到验证。
