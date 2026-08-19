# AP-FDR Internal Ablation Design

## 目标

用最少新增代码和最少正式训练，补足 AP-FDR 作为论文第一创新点所需的内部证据。实验必须回答两个高权重问题：preliminary reference 是否有效，以及 DN 侧的 assignment-consistent FDR supervision 是否有效。

## 方案比较

### 方案 A：最小双消融（采用）

- 保留现有 AP-FDR Full 结果，不重复训练。
- 复用 `rtdetr-l-fdr-no-prebox.yaml`，正式训练一次 `w/o preliminary reference`。
- 新增 `supervise_dn_fdr` 损失开关和 `rtdetr-l-fdr-no-dn.yaml`，正式训练一次 `w/o DN FDR supervision`。
- 其余 `w/o FGL`、`w/o pre-box loss`、`w/o cumulative` 只在主双消融不充分时先做 30-epoch 筛选。

优点：直接支撑论文中最具个人差异性的两个设计点，新增状态最少，算力成本最低。缺点：不能一次性给出五行完整机制拆解。

### 方案 B：五项全部正式训练

对已有五个 AP-FDR 功能单元都训练 100 epochs。证据最完整，但需要五倍左右新增算力，且若多个消融结果接近会稀释正文重点。

### 方案 C：继续增加新结构

增加独立 matcher、额外 attention 或新的分布公式。潜在涨点不确定，显著增加与现有 FDR/BPDD/FIA 的冲突风险，也会扩大审稿攻击面，因此不采用。

## 行为设计

### `supervise_dn_fdr`

`FDRDetectionLoss` 新增布尔参数 `supervise_dn_fdr`，默认 `true`。关闭时：

- 保留 Ultralytics 原生 DN classification/L1/GIoU；
- 保留 normal-query FGL；
- 保留 normal-query preliminary-box L1/GIoU；
- 只移除 `loss_fgl_dn`、`loss_fgl_aux_dn`、`loss_bbox_pre_dn`、`loss_giou_pre_dn`；
- 不改变网络、参数量、推理图、matcher 次数或 normal-query assignment。

这使消融问题严格限定为“额外的 DN 侧 FDR 监督是否贡献增益”，不会误删 stock RT-DETR 的 DN 训练。

### 配置

- 所有现有 FDR 家族配置显式声明 `supervise_dn_fdr: true`，避免默认值成为隐藏协议。
- 新建 `configs/rtdetr-l-fdr-no-dn.yaml`，与 Full 配置只差 `supervise_dn_fdr: false`。
- `FDRRTDETRDetectionModel` 和 `FDRBPDDDetectionModel` 都把该选项传入 criterion，保证组合模块保持兼容。

### 训练入口

新增 `scripts/train_ap_fdr_ablation.py`，只允许以下正式消融：

- `no_preliminary_reference`
- `no_dn_fdr`

入口复用现有 FDR seed0、640、batch 8、100 epochs 和优化器协议；运行前校验 VisDrone 数据签名和 FDR initial state，记录源码、配置、初始权重、数据与完整 settings 的哈希/清单。输出不覆盖已有运行。

## 证据与论文使用

主消融表按同一 val 协议报告 P、R、AP50、AP75、mAP50-95，优先解释 AP75。两条新运行完成后再用各自 best checkpoint 统一跑 test，并与 AP-FDR Full 的同协议 best 结果比较。

正式论文只把 Full 与两条严格消融作为 AP-FDR 核心证据。短程筛选结果不得混入正式主表；只有补足到 100 epochs 并统一 best-eval 后才能进入论文数值表。

## 测试与失败边界

- 单元测试验证关闭开关时 normal FDR losses 仍存在、DN FDR losses 精确消失、stock DN losses 仍存在。
- YAML 测试验证新配置只改一个叶子字段，且所有配置保持 stock graph。
- 集成测试验证 FDR 与 BPDD criterion 都读取该开关。
- 训练启动前 fail closed：源码/数据/initial state/config 任一指纹不可验证就不启动。
- 实验异常只保留证据，不自动把中断或 screen 数值写入论文材料。

## 非目标

- 不增加新 matcher、attention、推理分支或可训练参数。
- 不重跑已经可信的 AP-FDR Full。
- 不在本轮优先补第二数据集或多种子。
- 不把 D-FINE 的分布表示、Integral、相邻 bin soft label 或 FGL 原语宣称为首次提出。
