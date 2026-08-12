# FDR + BPDD + RA-GLGM 单臂 Formal100 实验预注册

> 文档性质：训练启动前冻结的实验协议与判定规则  
> 实验类型：历史对照复用下的单臂组合探索实验  
> 新训练臂：RT-DETR-L + FDR + BPDD + RA-GLGM v1.1  
> 代码底座：RA-GLGM v1.1 Full100 authority `69c188b`  
> BPDD 算法 authority：`848f00cb`  
> 训练数据：VisDrone train 6471 张  
> 评估数据：VisDrone 官方 val 548 张  
> 随机种子：`seed=0`  
> 训练预算：100 epoch  
> 重要限制：本实验只新训练组合臂，因此默认属于跨 authority 探索性比较，不自动等价于四臂 fresh 严格配对实验。

## 1. 预注册目的

本实验检验：在已经包含 FDR 的 RT-DETR-L 上，同时加入训练期 BPDD 和推理期 RA-GLGM v1.1，能否获得超出两个单模块历史结果的增量，而不是仅重复 FDR、BPDD 或 RA-GLGM 已有收益。

四个分析对象固定为：

| 代号 | 模型 | 来源 | 本次是否训练 |
|---|---|---|---|
| A | RT-DETR-L + FDR | GitHub 历史 Formal100 | 否，仅统一复评 |
| B | RT-DETR-L + FDR + BPDD | GitHub Release 历史 Formal100 | 否，仅统一复评 |
| C | RT-DETR-L + FDR + RA-GLGM v1.1 | 服务器历史 Full100 | 否，仅统一复评 |
| D | RT-DETR-L + FDR + BPDD + RA-GLGM v1.1 | 本实验 fresh Formal100 | **是** |

主要研究假设不是 `D > A`，而是组合臂是否超过两个更强的单模块臂：

\[
\mathrm{mAP}_{D} > \max(\mathrm{mAP}_{B},\mathrm{mAP}_{C}).
\]

若 D 只高于 A、但不高于 B 或 C，则只能说明组合保持了部分单模块收益，不能证明 BPDD 与 RA-GLGM 存在互补或协同。

## 2. 已核验的历史证据

### 2.1 FDR 记录完整性

GitHub `training-results` authority 已保存 FDR Formal100 的完整 100 轮记录，包括 `results.csv`、逐轮 CSV、逐轮 JSONL、运行身份、优化器证据和最新状态。该事实证明 A 臂的训练轨迹可审计，但仍需在本实验结束时取得其 epoch100 权重并使用本次锁定评估器重算最终指标。

### 2.2 BPDD 证据及其边界

BPDD Release 保存了 BPDD 单臂 epoch1--100 的逐轮 checkpoint 与相应 JSON 资产。已核验结果为：

| 证据 | mAP 增量 | AP75 增量 | 证据边界 |
|---|---:|---:|---|
| 严格配对 Screen30 | +0.189 pp | 已记录但不作为本表主门槛 | 同一 Screen authority 内可作严格筛选证据 |
| Formal100 历史比较 | +0.260 pp | +0.557 pp | **跨 authority**，不能当作 fresh paired 因果结论 |

BPDD 的 fresh paired FDR 对照臂只训练至 epoch24，未完成 100 epoch。因此，虽然 BPDD 方法臂有完整 100 轮权重和记录，其 Formal100 差值不是完整的 fresh paired Formal100 证据。本次不得把 `+0.260 pp` 预先视为必须复现的真实性效应，也不得把它与 RA 的 test-dev 差值直接相加预测 D。

### 2.3 RA-GLGM v1.1 证据及其边界

RA-GLGM v1.1 Full100 在服务器保留了 FDR 和 FDR+RA 两臂各 100 轮的 `results.csv` 与逐轮 JSONL。已核验的 test-dev 结果如下：

| 指标 | FDR -> FDR+RA 增量 |
|---|---:|
| mAP50--95 | +0.310 pp |
| AP75 | +0.389 pp |
| AP-tiny | +0.061 pp |
| AP-small | +0.679 pp |

这些数字证明 RA v1.1 在该 test-dev authority 下存在小幅正向信号，但不能代替本次官方 val 的统一复评。特别是，test-dev 与官方 val 的样本、标注可用性和评估用途不同，禁止把两者混入同一主结果表。

### 2.4 当前能够和不能够推断的内容

当前证据支持“BPDD 与 RA 均可能在 FDR 上提供小幅增量，值得测试组合臂”。当前证据不支持以下推断：

- 两个增量可以线性相加；
- D 必然优于 B 和 C；
- BPDD Formal100 已完成严格配对验证；
- test-dev 的 RA 增量可以直接用作官方 val 门槛；
- 单 seed 的小幅提升已经达到统计显著。

## 3. 方法组合与唯一新增变量

### 3.1 冻结的 FDR 路径

FDR 的分布回归表示、FGL、preliminary-box 监督、Query 数量、Hungarian 匹配、decoder 层数、分类分支和后处理全部保持不变。不得为了改善组合结果修改 FDR 权重、匹配代价、`reg_max`、`reg_scale`、FGL 权重或 preliminary-box 配置。

### 3.2 BPDD 的作用边界

BPDD 锁定为 `848f00cb` 算法：它只在训练时读取 FDR decoder 各层累计角点分布，使用已经存在的正常 Query 最终层匹配，构造来自未来层的分布教师并对早期层施加渐进式蒸馏。BPDD：

- 不新增模型参数；
- 不进行第二次 Hungarian 匹配；
- 不包含 denoising Query；
- 不进入推理图；
- 不修改 FDR、FGL、分类或原始 box loss；
- 固定 `weight=0.5`、`temperature=0.5`、`margin=0.02`、`eps=1e-6`。

### 3.3 RA-GLGM v1.1 的作用边界

RA-GLGM v1.1 沿用 `69c188b` 底座的已冻结实现，仅在 FDR decoder 的 P3 输入前执行局部/全局专家、空间支持与尺度条件化残差。P4、P5、FDR decoder、Query 和匹配保持不变。恒等残差初始化必须保留，使组合模型在训练开始前的公共路径输出严格等于 FDR，新增 RA 私有分支不允许通过加载历史 RA 权重获得先验优势。

### 3.4 本实验的处理变量

与 C 相比，D 的唯一主要变量是启用锁定 BPDD 训练损失；与 B 相比，唯一主要变量是在 P3 上启用锁定 RA-GLGM v1.1 推理模块及其冻结辅助监督。不得同时加入 RA v1.2、PFCR、FrequencyCM、SCADS、重排序、额外 Query、额外数据或改变增强策略。

## 4. Authority 与公平性要求

### 4.1 源码身份

- RA/FDR 公共底座必须从 `69c188b` 派生；
- BPDD 数学和损失行为必须与 `848f00cb` 锁定算法一致；
- 组合提交、配置、训练入口、测试和依赖锁文件必须在启动前计算 Git commit 与 SHA-256；
- 工作树必须可解释：任何未提交文件都要进入 source manifest，禁止训练后无记录地修改源码；
- 训练期间若必须修复实现，旧 run 立即关闭，新代码、新预注册修订和新 authority fresh start，禁止从旧 checkpoint 续训。

### 4.2 初始化

D 必须从 fresh 共同 scratch 初始化开始，不得加载 A、B、C 的训练权重，也不得从 Screen、Smoke 或中断 checkpoint 晋级。这里的“共同”指公共 FDR 参数使用与历史正式协议相同、字节可核验的 seed0 初始状态生成规则；RA 私有参数使用冻结的独立固定种子规则。启动前应保存：

- 公共参数键集合与 SHA-256；
- RA 私有参数键集合与 SHA-256；
- 全模型 initial-state SHA-256；
- 与 A/B/C 可比的公共键逐张量 equality 报告。

若无法证明 D 的公共初始化、数据协议和训练协议与某个历史臂一致，该历史臂只能标记为跨 authority 参考，不得称为 matched control。

### 4.3 数据 authority

- 训练集：VisDrone train，6471 张；
- 验证集：VisDrone 官方 val，548 张；
- 类别数：10；
- 输入尺寸：640 x 640；
- 启动前冻结图像列表、标签列表、逐文件内容哈希、类别映射、ignore 区域处理和数据 YAML 哈希；
- 不使用 647 张 Screen 子集代替正式训练；
- 不将 test-dev 结果与官方 val 指标混算；
- 不增加伪标签、外部图像、预训练检测权重或数据清洗规则。

## 5. 冻结训练规格

| 项目 | 固定值 |
|---|---|
| 训练臂 | D：FDR + BPDD + RA-GLGM v1.1 |
| seed | 0 |
| epochs | 100 |
| imgsz | 640 |
| batch | 8 |
| nbs | 64 |
| 梯度累积 | 8 个 micro-batch 对应有效 batch 64 |
| workers | 8 |
| AMP | 开启 |
| optimizer | MuSGD，沿用 FDR/RA 冻结协议 |
| cache | false |
| deterministic | true |
| 初始化 | fresh scratch，共同 seed0 公共初始状态 |
| DDP | 禁止 |
| 自动降 batch | 禁止 |
| resume | 仅允许同一 authority 的异常恢复；不得用于阶段晋级 |

学习率、weight decay、warmup、增强、验证频率、Query、匹配和 FDR/RA/BPDD loss 权重均继承锁定协议，不得根据中间指标调整。若 batch8 发生 OOM，不允许在同一 run 静默切换 batch；应关闭该 authority，查明是否为异常占用，再决定是否预注册新的 batch 规格。

## 6. 执行流程

### 阶段 0：只读证据盘点

1. 固定 A/B/C 的 epoch100 checkpoint、训练记录和哈希；
2. 检查 A 的完整 100 轮 CSV/JSONL；
3. 检查 B 的 epoch1--100 Release 资产和每轮 JSON，同时明确 fresh FDR 只到 epoch24；
4. 检查 C 两臂各 100 轮 CSV/JSONL；
5. 建立历史证据 manifest，列出来源、commit、协议、初始化、数据、seed、checkpoint SHA 和缺失字段。

若 A、B 或 C 的 epoch100 权重无法取得或无法安全加载，不阻止 D 的工程探索训练，但相应比较不得进入主要科学结论。

### 阶段 1：静态与语义预检

必须在启动训练前通过：

- FDR 原测试与 RA v1.1 原测试；
- BPDD `848f00cb` golden/回归测试；
- 组合入口的模型构建、严格 initial-state 加载和 checkpoint round-trip；
- RA 开启、BPDD 关闭时与锁定 C 数学行为一致；
- RA 关闭、BPDD 开启时与锁定 B 数学行为一致；
- BPDD 和 RA 都关闭时与锁定 A 的 FDR loss/forward 一致；
- BPDD 只出现在训练 loss，`eval()` 和导出推理图中不存在 BPDD 路径；
- RA 零初始化时公共输出与 FDR 在容差内一致；
- 正常 Query、DN Query、空 GT、混合空 GT、AMP 和有限梯度测试；
- 参数键归属审计：BPDD 新参数数必须为 0，RA 私有参数集合必须与 v1.1 一致。

任一项失败即禁止启动 Formal100。

### 阶段 2：组合臂 Smoke2

在正式数据管线和同一 RTX 4090 上 fresh 运行 2 epoch，只验证工程可运行性，不产生科学结论。必须确认：

- forward、backward、MuSGD step、验证和 checkpoint round-trip 成功；
- loss、指标、梯度、AMP scale 均有限，AMP skipped step 为 0；
- FDR 公共梯度、RA 私有梯度和 BPDD 对早期分布层的梯度均有限且非零；
- BPDD `active_edge_ratio`、`mean_reliability`、`matched_queries` 等诊断有记录；
- RA support、router、scale 诊断有记录且不是常数/NaN；
- 峰值显存低于安全阈值，磁盘预测足以完成 Formal100；
- Smoke 权重不作为 Formal 初始化。

Smoke2 通过后删除或归档其重型 checkpoint，只保留轻量证据；Formal100 必须再次 fresh start。

### 阶段 3：单臂 Formal100

D 从冻结 initial state 开始训练 100 epoch。训练期间不得查看结果后改变损失权重、冻结层、增强、终止轮、best 选择规则或评估阈值。

每轮必须：

- 更新一个可恢复的 `last.pt`；
- 校验 `last.pt` 可加载并记录 SHA-256、大小和 epoch；
- 写入一条不可重复的轻量 JSONL/CSV 记录；
- 记录训练/验证指标、各 loss、学习率、梯度范数、AMP scale/skips、峰值显存、epoch wall time；
- 记录 BPDD 与 RA 私有诊断；
- 写入本地审计队列和哈希链。

每 5 轮（5、10、...、100）额外保留一个独立、不可被 `last.pt` 覆盖的 checkpoint，并保存对应 manifest。为降低磁盘占用，不保存其他独立中间权重。未经用户对具体 checkpoint 子集的明确批准，任何 `.pt` 文件都不得上传 GitHub、Release 或其他远端。

### 阶段 4：统一锁定评估

训练前冻结一个独立 evaluator authority。D 完成后，用完全相同的 evaluator、数据哈希、letterbox、ignore 过滤、`conf`、`max_det`、类别映射和数值精度，对 A/B/C/D 的 epoch100 权重重新评估。统一评估至少输出：

- Precision、Recall、F1；
- AP50、AP75、mAP50--95；
- AP-tiny、AP-small；
- 10 类逐类 AP；
- 推理参数量、GFLOPs、峰值显存、延迟/FPS（同硬件、warmup、同步和 batch）；
- 每张图的预测 JSON、评估摘要和 SHA-256。

同一结果表不得混用历史 README 数字、原生 Ultralytics 指标、test-dev 指标和新 evaluator 指标。历史数字只放在“先导证据”表；主结果表只能来自统一锁定复评。

## 7. 主要终点与尾五轮规则

### 7.1 主终点

主终点是官方 val 上、锁定 evaluator 得到的 epoch100 mAP50--95。主要成功条件为：

\[
\mathrm{mAP}_{D,100} > \max(\mathrm{mAP}_{B,100},\mathrm{mAP}_{C,100}).
\]

差值必须以绝对百分点和 `[0,1]` 原始尺度同时报告，避免把 `0.003` 错写为 `0.003 pp`。

### 7.2 尾五轮

尾五轮固定为 epoch96--100，使用每轮训练时同一验证实现自动生成并由 JSONL/CSV 双重记录的指标，主要查看 mAP50--95 均值与方向。由于独立 checkpoint 固定每 5 轮保存，epoch96--99 的尾段指标属于哈希绑定的在线验证证据，而非可事后重跑的四个独立 checkpoint；报告中必须明确这一层级。

尾五轮稳定条件为：

- D 的 epoch96--100 mAP 均为有限值；
- 尾五轮均值高于可比较历史臂中更高者的尾五轮均值；
- 至少 3/5 个对应 epoch 的 D mAP 高于 B 和 C 的同轮较高值；
- epoch100 不是依赖单轮尖峰得出的唯一正向结果。

若历史臂尾五轮的验证实现或数据 hash 不一致，则该条件只作稳定性描述，不作跨臂正式门槛；不得为了补齐结果而把 best checkpoint 替代尾五轮。

### 7.3 次要终点

次要指标为 AP75、AP-tiny、AP-small、AP50、Recall、Precision/F1 和逐类 AP。Best checkpoint 只作补充，不替代 epoch100 和尾五轮主结论。

## 8. 预注册判定规则

### 8.1 跨 authority 端点探索成功

由于本轮只 fresh 训练 D，A/B/C 均来自历史运行，因此以下条件最多把 D 记为“统一复评下的跨 authority 端点探索成功”，不能直接声称统计显著、因果增量或模块协同：

1. 工程审计全部通过，100 个 epoch 连续且无证据缺口；
2. 统一复评的 `mAP(D) > max(mAP(B), mAP(C))`；
3. D 相对更强单模块臂的 AP75、AP-tiny 和 AP-small 均严格大于 0；
4. 尾五轮满足第 7.2 节的稳定条件；若历史尾五轮不可比，则只能按第 8.2 节降级为端点部分成功；
5. D 至少 7/10 类 AP 高于 A，且至少 6/10 类 AP 高于 B、C 中对应类别的较高值；
6. Recall 不低于更强单模块臂超过 0.2 pp，AP50 不下降；
7. BPDD active-edge/reliability 诊断和 RA support/router/scale 诊断均未坍缩；
8. 推理期开销只来自 RA，BPDD 不进入推理图，参数量与 RA v1.1 相同。

所有“严格提高”均指未四舍五入原始值 `>`，表格显示时再转换为百分点。

本轮 `D > max(B,C)` 只是是否值得继续 fresh 多臂复验的筛选门，不是超加性协同的充分证据。后续若要声称协同，至少需要在可比的 fresh 四臂上检验交互对比 `(D-C) > (B-A)`，等价于 `D+A > B+C`，并做多 seed 复验。

### 8.2 部分成功

出现以下任一情况记为“部分成功/无协同证据”：

- D > A，但 `D <= max(B,C)`；
- 总 mAP 超过 B/C，但 tiny、small 或 AP75 任一退化；
- epoch100 正向但尾五轮不稳定；
- 只有 best checkpoint 正向；
- 统一复评可完成，但 authority 无法证明严格可比。
- D 的 epoch100 端点正向，但历史 B/C 尾五轮无法在同一验证 authority 下比较。

部分成功可以用于后续机制诊断，不能在论文中表述为“组合模块优于两个组成模块”。

### 8.3 失败

出现以下任一科学结果记为失败：

- `mAP(D) <= max(mAP(B), mAP(C))`；
- D 相对 A 也无提升；
- 组合只提高 AP75、但明显牺牲 AP50/Recall/tiny/small；
- BPDD 或 RA 诊断显示分支未学习、恒等残差未打开或门控坍缩；
- 增益只存在于事后选择的 best epoch。

工程失败与科学失败必须分开报告。OOM、NaN、checkpoint 损坏或证据断链属于工程失败，不能被解释为方法精度失败。

## 9. 停止、恢复与异常规则

训练只因以下预注册原因停止：

- loss、指标或梯度出现 NaN/Inf；
- 任意一次 AMP skipped step，或任意一个真实 optimizer step 边界未更新；
- checkpoint 不可加载、epoch/optimizer state 不一致或哈希冲突；
- 训练源码、数据、initial-state 或 protocol hash 漂移；
- GPU 身份变化，或非授权进程持续占用显存/算力并改变冻结 batch8 协议；
- 可用磁盘低于 12 GiB 时告警，低于 8 GiB 时强制停止；
- 用户明确要求停止。

允许从同一 authority 的最后一个已验证 `last.pt` 做故障恢复，但必须保留中断原因、恢复时间、前后 checkpoint SHA、optimizer/scaler state 和首个恢复 batch 的连续性证据。改变 batch、workers、数据、seed、代码、损失权重或模型结构均不属于恢复，必须新建 authority 并 fresh start。

不设置基于中间精度的早停。即使中期低于 B/C，也完成 100 epoch，除非出现上述工程停止条件；这样可避免重复过去“按早期曲线决定结论”的风险。

## 10. 对抗性审计清单

### 10.1 启动前

- 服务器上无残留训练进程和占用 GPU 的未知进程；
- GPU UUID、显存、驱动、CUDA、PyTorch、Ultralytics 和依赖锁定；
- 数据为 6471/548，逐文件内容哈希、空标签、越界框、类别范围和 ignore sidecar 全量审计；
- 源码 commit/source manifest 与运行时 import 路径一致，排除导入旧副本；
- A/B/C/D 模型身份、参数键和初始状态差异符合预期；
- BPDD 参数增量严格为 0；D 相比 B 的参数增量只来自 RA；
- 磁盘预算覆盖 20 个五轮 checkpoint、一个滚动 last、日志、预测和安全余量。

### 10.2 每轮

- 恰有一条 epoch 记录和一条本地队列记录；
- `results.csv`、JSONL、checkpoint epoch 和监督器状态一致；
- 所有 loss/metric/gradient 有限；
- optimizer step、AMP scale 和 skipped step 合法；
- BPDD 统计和 RA 统计存在且有限；
- `last.pt` 可严格加载，文件 SHA 与 manifest 一致；
- GPU 利用率、显存、温度、磁盘和 wall time 无异常漂移。

### 10.3 每五轮与结束后

- 五轮 checkpoint 与 `last.pt` 对应 epoch 的模型张量完全一致；
- 独立 checkpoint 可严格加载并通过最小前向；
- 100 轮连续，无重复、缺失、回退或跨 run 混入；
- 锁定 evaluator 从原始预测重新计算指标，不信任手工汇总；
- A/B/C/D 的 evaluator、数据和配置 SHA 完全一致；
- 检查训练集/val/test-dev 混用、类别索引偏移、ignore 过滤、letterbox 和 maxDet 差异；
- 生成最终 checksum manifest，并验证所有非权重发布物；
- 不发布任何 `.pt`。

## 11. 主要风险及控制

| 风险 | 影响 | 预注册控制 |
|---|---|---|
| BPDD 与 RA 增益重叠 | D 可能不超过更强单模块 | 以 `D > max(B,C)` 而非 `D>A` 为主判据 |
| BPDD 改变早期层梯度，RA 改变 P3 特征 | 两者可能相互干扰 | 记录两类私有诊断和分层梯度，不事后调 loss 权重 |
| BPDD Formal100 非 fresh 配对 | 历史差值有 authority 偏差 | 对 epoch100 权重统一复评并明确探索性边界 |
| RA 历史数字来自 test-dev | 与官方 val 不可直接比较 | 主表只使用锁定 evaluator 的同数据复评 |
| 单 seed 小效应 | 可能是随机波动 | 不声称统计显著；成功后再预注册多 seed 复验 |
| best checkpoint 选择偏差 | 放大偶然峰值 | epoch100 与尾五轮为主，best 仅补充 |
| 只训练 D | 无法完全消除训练环境漂移 | 锁定公共初始化/协议/评估；不可证时降级为跨 authority |
| 中间权重占用磁盘 | 可能中断训练 | 仅滚动 last + 每五轮独立权重，保留全部轻量记录 |
| 服务器异常恢复 | 可能改变样本顺序或优化器 | 只允许严格 true-resume，并记录恢复连续性证据 |

## 12. 结果报告模板

统一复评后必须生成以下主表，空值不得用历史异构指标填补：

| 模型 | P | R | F1 | AP50 | AP75 | mAP50--95 | AP-tiny | AP-small | 来源/authority |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A FDR |  |  |  |  |  |  |  |  | 历史权重统一复评 |
| B FDR+BPDD |  |  |  |  |  |  |  |  | 历史权重统一复评 |
| C FDR+RA |  |  |  |  |  |  |  |  | 历史权重统一复评 |
| D FDR+BPDD+RA |  |  |  |  |  |  |  |  | 本次 fresh 单臂 |
| D - A |  |  |  |  |  |  |  |  | pp |
| D - B |  |  |  |  |  |  |  |  | pp |
| D - C |  |  |  |  |  |  |  |  | pp |

还必须单列：

- epoch96--100 的逐轮轨迹与尾五轮均值；
- 10 类逐类 AP 和类别覆盖数；
- BPDD active edge、reliability、teacher improvement；
- RA support、router、scale 分布与残差幅度；
- 参数量、GFLOPs、同机延迟/FPS、峰值显存和每 epoch wall time；
- 所有 source/data/protocol/init/checkpoint/evaluator SHA-256；
- 中断与恢复事件（若有）。

## 13. 证据发布与保留策略

允许发布到授权 GitHub 分支的内容仅包括：

- 冻结源码、配置、测试和预注册；
- 数据/环境/初始化/协议 manifest；
- 每轮轻量 CSV/JSONL、审计日志和 Gate 报告；
- 统一评估 JSON/CSV、逐类表、曲线图、最终报告与 checksums；
- 不含原始数据、不含凭据且不含权重的复现说明。

禁止发布：

- 任意 `.pt` checkpoint，包括 best、last、epoch100 和每五轮快照；
- 密码、PAT、SSH key、带凭据 URL 或服务器私有信息；
- 未审计的临时预测、缓存和混合 authority 产物。

服务器本地至少保留 D 的 epoch100、best、last、每五轮 checkpoint、完整轻量记录和哈希；清理中间文件前先生成删除 manifest，并确认不存在唯一证据被删除。

## 14. 最终证据等级与后续决策

即使本实验完整成功，其默认结论仍应写为：

> 在锁定协议和统一评估下，fresh 训练的 FDR+BPDD+RA-GLGM seed0 组合臂超过了可取得的历史单模块 epoch100 权重；由于只新训练组合臂，且 BPDD 的历史 Formal100 缺少完整 fresh paired FDR 对照，该结果属于有控制的跨 authority 探索证据。

只有在随后 fresh 训练至少一个严格 matched 的更强单模块对照，并完成多 seed 配对复验后，才能把组合增量升级为确认性证据。若 D 未超过 `max(B,C)`，应停止继续堆叠模块，优先分析 BPDD 对早期 decoder 分布梯度与 RA 门控/残差响应的相互作用，不得通过改用 best、改评估集或事后放宽类别门槛挽救结论。
