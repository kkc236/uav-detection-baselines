# ICASSP 2027 三模块论文声明—证据映射

> 更新日期：2026-08-11
>
> 用途：约束中文论文模板中的数字、原创性和结论边界。
>
> 状态标签：`【实测·严格】`、`【实测·初步】`、`【预估】`、`【待测】`。

## 1. 使用规则

1. `【实测·严格】`仅用于具备统一协议、可追溯 checkpoint、独立评估和冻结证据的结论。
2. `【实测·初步】`表示确有实验结果，但存在跨 authority、未完成严格配对或单 seed 等限制。
3. `【预估】`仅用于提前组织论文成功情景，不得在投稿稿中伪装为实验结果。
4. `【待测】`表示方法、表格或论证位置已经确定，但尚无可引用数字。
5. 最终英文稿不得保留未经替换的 `{{TOKEN}}`，也不得删除状态限定词后继续使用初步或预估数字。

## 2. 核心声明映射

| Claim ID | 组件 | 规划声明 | 证据状态 | 数值或来源 | 允许措辞 | 禁止措辞 |
|---|---|---|---|---|---|---|
| C-FDR-1 | FDR | 连续四维框回归被重构为 preliminary box 引导的六层累计四边分布回归 | 【实测·严格】 | [FDR YAML 文档](FDR_YAML_DECLARATIVE_MODULE.md) | 面向 Ultralytics RT-DETR-L 的结构化适配 | 首次提出 FDR/FGL/Integral |
| C-FDR-2 | FDR | Control→FDR 的 mAP50-95 从 0.21911 提高到 0.28966 | 【实测·严格】 | `+0.07055`，即 `+7.055 pp` | 同一 formal authority、seed0、100 epoch 下严格提升 | 多 seed 普适提升或统计显著 |
| C-FDR-3 | FDR | AP50 与 AP75 同时提高 | 【实测·严格】 | AP50 `+9.805 pp`；AP75 `+7.951 pp` | 同时改善检出与严格定位 | 所有子机制分别贡献该增益 |
| C-FDR-4 | FDR | 四尺度和十类别均正向 | 【实测·严格】 | 仅限当前 seed0 统一评估 | 当前 seed0 的分尺度和分类结果全部正向 | 已在多 seed 全面稳定上涨 |
| C-FDR-5 | FDR | 计算开销较小 | 【实测·严格/待测】 | Params `+1.00524%`；GFLOPs `+0.18256%`；延迟待封口 | 参数和理论计算量小幅增加 | 参数增加低于1%或延迟低于3% |
| C-BPDD-1 | BPDD | 目标级 future-only 混合教师与 better-only gate 提供训练期分布蒸馏 | 【实测·初步】 | [BPDD Authority](../research/bpdd/BPDD_AUTHORITY.md) | 风险受控的具体 Decoder 分布蒸馏重构 | 首次提出自蒸馏、跨层定位蒸馏或 better-teacher gate |
| C-BPDD-2 | BPDD | 严格配对 Screen30 通过 | 【实测·严格】 | final mAP `+0.189 pp`；AP75 `+0.185 pp`；tail3 mAP `+0.056 pp` | 候选筛选门全部正向 | Screen30 等同 Formal100 效应量 |
| C-BPDD-3 | BPDD | Formal100 相对既有 FDR100 进一步提高 | 【实测·初步】 | mAP `+0.260 pp`；AP75 `+0.557 pp` | 单臂 Formal100 独立评估的初步跨 authority 正信号 | fresh paired Formal100 已证实 |
| C-BPDD-4 | BPDD | 不增加推理分支和参数 | 【实测·严格/待测】 | checkpoint 可由普通 FDR 推理图加载；严格同机延迟待封口 | BPDD 为参数零、training-only 模块 | 已完成零延迟实测证明 |
| C-BPDD-5 | BPDD | 对 Small 和严格 IoU 定位更有利 | 【实测·初步】 | Small mAP `+0.778 pp`；四尺度 AP75 均正向 | 初步结果显示 Small 和 AP75 受益更明显 | 四尺度 mAP 全面上涨 |
| C-RA-1 | RA-GLGM | P3 尺度路由局部—全局残差增强改善尺度表征 | 【预估】 | 成功情景结构假设 | 论文规划中的第三模块动机与方法 | 已被现有 v1.1 正式验证 |
| C-RA-2 | RA-GLGM | 在 FDR+BPDD 上增加约 0.5 pp mAP | 【预估】 | `{{RA_DELTA_MAP}} ≈ +0.5 pp` | 成功情景目标、待严格100轮替换 | 实测提升0.5个百分点 |
| C-RA-3 | RA-GLGM | 增强 tiny/small 且不伤害其他尺度 | 【待测】 | `{{RA_AP_TINY}}`、`{{RA_AP_SMALL}}`、`{{RA_AP_MEDIUM}}`、`{{RA_AP_LARGE}}` | 最终必须由统一 evaluator 验证 | 当前已全面改善各尺度 |
| C-FULL-1 | Full | 三组件从尺度表征、边界表示和优化轨迹形成互补链 | 【待测】 | 五行模块消融 | 统一设计假设 | 已证明三者协同 |
| C-FULL-2 | Full | 完整模型优于最强双模块配置 | 【待测】 | `{{FULL_DELTA_OVER_BEST_PAIR}}` | 最终 Gate：严格高于最强双模块 | 只因完整模型高于 Control 就证明协同 |

## 3. 可直接引用的冻结结果

### 3.1 Control 与 FDR

| 指标 | 严格 Control | FDR | 绝对提升 |
|---|---:|---:|---:|
| Precision | 0.46761 | 0.56911 | +10.150 pp |
| Recall | 0.41731 | 0.49278 | +7.546 pp |
| AP50 | 0.38663 | 0.48468 | +9.805 pp |
| AP75 | 0.21302 | 0.29253 | +7.951 pp |
| mAP50-95 | 0.21911 | 0.28966 | +7.055 pp |

来源：[FDR 严格 Control 与结果](FDR_RTDETR_L_STRICT_CONTROL_AND_RESULTS_2026-08-09_ZH.md#9-统一独立评估总体指标)。

### 3.2 FDR 分尺度

| 尺度 | GT | Control mAP | FDR mAP | ΔmAP |
|---|---:|---:|---:|---:|
| Tiny | 20,861 | 0.08684 | 0.14480 | +5.795 pp |
| Small | 12,420 | 0.21784 | 0.28998 | +7.214 pp |
| Medium | 5,348 | 0.32499 | 0.39630 | +7.130 pp |
| Large | 130 | 0.31822 | 0.38608 | +6.786 pp |

Large 只有130个GT，不得据此单独作强机制归因。

### 3.3 BPDD筛选与初步Formal100

| 阶段 | 指标 | 增益 | 状态 |
|---|---|---:|---|
| paired Screen30 | final mAP50-95 | +0.189 pp | 【实测·严格】 |
| paired Screen30 | final AP75 | +0.185 pp | 【实测·严格】 |
| paired Screen30 | tail3 mAP50-95 | +0.056 pp | 【实测·严格】 |
| Formal100 | Precision | +0.152 pp | 【实测·初步】 |
| Formal100 | Recall | +0.169 pp | 【实测·初步】 |
| Formal100 | AP50 | +0.172 pp | 【实测·初步】 |
| Formal100 | AP75 | +0.557 pp | 【实测·初步】 |
| Formal100 | mAP50-95 | +0.260 pp | 【实测·初步】 |

Formal100的fresh FDR臂在epoch24按用户要求停止，因此这些值不能标成严格paired Formal100。

## 4. RA-GLGM成功情景约束

模板把RA-GLGM作为第三项成立贡献来组织，但必须满足以下替换条件才能进入最终投稿稿：

1. 唯一冻结版本完成同协议100 epoch；
2. 相对 `FDR+BPDD` 的 `{{RA_DELTA_MAP}}` 达到规划目标约 `+0.5 pp`，或至少达到预注册的投稿门槛；
3. APtiny、APsmall、AP75不退化，路由没有坍缩；
4. 至少优于等参数卷积、uniform router与单专家控制；
5. 参数、GFLOPs、median/P90延迟和显存完成同机审计；
6. Full Model严格高于 `FDR+BPDD` 与 `FDR+RA-GLGM` 中的较强者。

现有RA-GLGM v1.1 Screen10报告是科学门禁失败记录，不能充当上述成功证据。论文模板中的RA文字属于未来成功版本的写作预案。

## 5. 正式主表替换令牌

### 5.1 BPDD严格配对

`{{BPDD_STRICT_PRECISION}}`、`{{BPDD_STRICT_RECALL}}`、`{{BPDD_STRICT_AP50}}`、`{{BPDD_STRICT_AP75}}`、`{{BPDD_STRICT_MAP}}`。

### 5.2 RA-GLGM

`{{RA_PRECISION}}`、`{{RA_RECALL}}`、`{{RA_AP50}}`、`{{RA_AP75}}`、`{{RA_MAP}}`、`{{RA_DELTA_MAP}}`。

### 5.3 完整模型

`{{FULL_PRECISION}}`、`{{FULL_RECALL}}`、`{{FULL_AP50}}`、`{{FULL_AP75}}`、`{{FULL_MAP}}`、`{{FULL_DELTA_MAP}}`、`{{FULL_DELTA_OVER_BEST_PAIR}}`。

### 5.4 效率

`{{RA_PARAMS}}`、`{{RA_GFLOPS}}`、`{{RA_LAT_MED_MS}}`、`{{RA_LAT_P90_MS}}`、`{{RA_FPS}}`、`{{RA_VRAM_MB}}`、`{{FULL_PARAMS}}`、`{{FULL_GFLOPS}}`、`{{FULL_LAT_MED_MS}}`、`{{FULL_LAT_P90_MS}}`、`{{FULL_FPS}}`、`{{FULL_VRAM_MB}}`。

## 6. 投稿前必须关闭的证据缺口

1. fresh same-source、same-initial-state FDR/BPDD Formal100严格配对；
2. official GO-LSD与相同better-only gate的固定最终层教师对照；
3. BPDD fixed-final、no-gate、hard-gate等消融；
4. RA-GLGM唯一版本、完整消融和同机效率；
5. 完整五行模块消融与组合协同验证；
6. 至少一个额外seed或paired bootstrap置信区间，最好补第二数据集；
7. 统一F1口径；现阶段主表不列F1；
8. 正式协议以MuSGD与`warmup_bias_lr=0.0`的机器权威为准，旧`optimizer=auto`控制文档不得混入。

## 7. 论文中推荐的边界声明

> 本文不将D-FINE中的FDR、FGL和Integral视为原创，而将其作为RT-DETR-L细粒度定位基础；本文的新增方法集中于风险受控的目标级渐进式Decoder蒸馏，以及经严格验证后才能进入最终结论的尺度路由局部—全局增强。

> 当前BPDD Formal100结果为相对既有严格FDR authority的初步比较；最终投稿主表将由fresh同源严格配对结果替换。

> RA-GLGM及Full Model的数值在本模板中属于成功情景预估，不是当前实验结论。
