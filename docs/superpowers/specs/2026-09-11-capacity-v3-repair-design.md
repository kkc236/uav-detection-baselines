# Capacity-v3 Repair 备选实现设计

## 状态与目标

- 状态：设计已获用户确认，待文档审阅后进入实施计划。
- 目标：在不改变当前服务器正在运行的 C 版本、不改变 LRS/GFDR 几何算法和初始化协议的前提下，修复此前审计已经复现的显存、监督方向、层归一化、残差回传和证据记录问题。
- 版本名：`capacity-v3-repair`。
- 论文口径：这是候选实现与对照版本，不预先宣称新的论文创新或涨点。

## 问题边界

本版本只处理以下已复现问题：

1. 精修分支的 query 范围和反向激活需要受控，避免 DN query 或全部 query 放大显存。
2. 仅用 FDR 目标 bin NLL 判断教师质量会放行解码框更差的教师；分类 BCE 与 VFL 的 IoU 软目标也可能方向冲突。
3. 定位和分类项对没有有效样本的层取平均时会被稀释。
4. FDR 的累计残差会使跨层 assignment 的蒸馏梯度回到不对应的历史残差。
5. v2 统计生产者与 runtime recorder 字段不一致，无法判断新版 BPDD 是否真正激活。
6. 精修输出、`last_corner_logits` 和几何统计存在口径混合。

不包含：重新设计 backbone、改变输入尺寸、引入 FIA、重新定义 GFDR 几何公式、调整数据划分、改变训练轮数、修改当前服务器进程或复用测试集挑选配置。

## 方案概览

### 组件一：normal-only 精修与显存边界

`FDRRTDETRDecoder` 在训练和评估时只把 normal query 的切片交给 `LocalBoundaryExpert`。DN query 不进入局部采样、注意力、精修输出或精修直接监督。调用方必须显式传入 `normal_query_count`，并校验其在 decoder query 范围内；normal query 数为零时跳过精修并返回空统计。

局部特征采样继续使用现有 P2/P3/P4、边界点和中心点定义。每个 query 的局部注意力按 `chunk_size` 分块，并在训练图中使用非重入 checkpoint；实现需要确保 checkpoint 只包住局部注意力计算，不包住有副作用的统计或随机状态。新增开关 `local_expert_checkpoint` 默认开启，关闭时只用于诊断。

该组件的资源约束通过测试而不是参数估算确定：记录 normal query 数、DN query 数、chunk 数、峰值显存和前向/反向是否出现非有限值。若所有 query 都无有效采样点，仍保留 query token 路径，输出必须有限。

### 组件二：质量一致的 BPDD 教师门控

定位门控按同一 query/GT assignment 计算：

- FDR 支持可表示性；不可表示边不进入定位蒸馏；
- 学生与教师的解码框 IoU 和逐边几何误差；
- 教师必须在规定 margin 后优于学生，所有门控量 detach；
- 通过后再计算分布 KL，教师分布本身不回传梯度。

分类门控不再只比较 one-hot BCE。它使用对应匹配的 IoU 作为正类软目标，评估教师和学生对同一质量目标的误差；负类抑制仍保留，且只有身份一致的 normal query 可进入分类蒸馏。教师框质量不优于学生时，分类蒸馏关闭，避免过度自信但定位更差的教师与 VFL 冲突。

定位和分类的 margin、tau、权重、warmup 仍由 YAML 固定，不从验证集循环搜索。所有门控覆盖率和放行前后的教师优势都写入统计，以便识别“全关闭”和“错误放行”两种失败模式。

### 组件三：有效层独立归一化

对每个 decoder source layer 分别收集有效定位项和有效分类项。定位只除以该层的有效边数，分类只除以该层的有效匹配数；最终均值只在对应非空层之间计算。若全部层为空，返回与学生计算图相连的零值，不产生 NaN，也不改变主检测损失。

统计至少包含：候选匹配数、身份一致匹配数、可表示边数、有效边数、有效分类匹配数、定位有效层数和分类有效层数。

### 组件四：residual-only 蒸馏路径

在 `assignment_mode=consistent` 时提供 `residual_gradient_only`。开启后，蒸馏前向值保持累计分布不变，但梯度只经过当前层增量：

```text
z_l_for_kd = stopgrad(z_l) + (r_l - stopgrad(r_l))
r_l = z_l - z_(l-1)
```

首层使用自身增量。共享 Transformer 特征仍可能受到间接影响；该开关只切断累计残差的直接跨层路径，不能宣称完全隔离。关闭该开关时恢复原始累计路径，作为同配置单因素对照。

### 组件五：运行证据与输出口径

`RuntimeEvidenceRecorder` 改为读取 capacity-v3 的真实字段，并在每个 epoch 记录：

- capacity BPDD 定位/分类 loss、有效层数、激活比例、教师优势；
- normal-only 精修的几何总数、不可行计数、最小解码尺寸；
- 主参数、FDR 私有参数、expert 参数的梯度范数及有限性；
- 方法修订号、配置摘要、normal/DN query 数和峰值显存（可用时）。

精修后的 corners、classes 和 boxes 作为评估输出；对应的 `last_corner_logits`、FDR 几何统计和 expert 几何统计必须来自同一输出路径。普通 FGL 仍使用定义好的 FDR 分布，但记录中明确标注其来源层和是否为精修输出。

断点恢复在本候选中不默认打开。新增恢复检查只验证配置、初始化身份、optimizer/EMA/scheduler 状态和 epoch 对齐；恢复信息缺失时明确失败，不能静默当作 fresh run。

## 接口与文件职责

- `src/bpdd_capacity.py`：局部特征采样、normal-only expert、checkpoint 边界、资源统计。
- `src/bpdd_capacity_loss.py`：质量门控、质量感知分类目标、有效层归一化及 detached 统计。
- `src/bpdd_loss.py`：保留并复用 residual-only 的 assignment 一致性接口；不改变旧模式默认行为。
- `src/fdr_head.py`：normal query 切片、精修输出和 expert 几何统计汇总。
- `src/rtdetr_bpdd_capacity.py`：v3 配置解析、criterion 接线、方法修订号和梯度分组。
- `src/lrs_runtime_evidence.py`：capacity-v3 字段聚合、梯度有限性和几何/显存记录。
- `configs/rtdetr-l-lrs-gfdr-capacity-v3-repair.yaml`：备选版本完整配置，默认打开修复项。
- `tests/test_bpdd_capacity_v3.py`：normal/DN、checkpoint 和资源边界测试。
- `tests/test_bpdd_capacity_loss_v3.py`：质量门控、分类目标、有效层归一化测试。
- `tests/test_bpdd_residual_v3.py`：residual-only 前向等价与梯度路径测试。
- `tests/test_lrs_runtime_evidence_v3.py`：字段接线、几何输出和梯度有限性测试。
- `tests/test_rtdetr_bpdd_capacity_v3.py`：模型/criterion/config 集成测试。

## 数据与错误处理

- 所有 geometry、teacher quality、统计值在写入前转为 detached FP32；非有限值计数并触发明确的诊断失败，不以零掩盖。
- `normal_query_count`、assignment 索引、reference/GT 尺寸和 logits 形状不符合接口时立即抛出 `ValueError`。
- 空 assignment、空 GT、全无效采样点和不可表示支持是合法输入，必须返回有限零损失或回退 query 路径。
- teacher detach、推理无 GT 依赖和精修输出 shape 由单元测试与集成测试共同验证。
- 任何 OOM、NaN、AMP skip 异常只记录实际观测，不把短程通过写成正式训练成功。

## 实验与归因协议

备选训练使用与当前 C 相同的数据、初始状态、输入尺寸、optimizer、scheduler、epoch 数和验证器。只新增一个完整修复候选，并保留四个诊断开关用于短程筛查：

| 臂 | 目的 |
|---|---|
| Repair-full | 所有 v3 修复开启，作为备选候选 |
| Repair-no-residual | 只关闭 residual-only，检查累计路径影响 |
| Repair-no-quality | 只关闭新质量门控，检查门控贡献 |
| Repair-no-checkpoint | 只关闭 checkpoint，仅用于资源诊断，不作为精度结论 |

正式 100 轮只在实现通过本地回归和固定批次诊断后启动。主指标为同一验证协议的 best mAP50-95；同时保存 mAP50、P、R、AP75、尺度分层、逐类 AP、最后十轮均值、参数量、GFLOPs、推理延迟、峰值显存和训练成本。连续 epoch 不视作多次独立重复。

## 测试策略

实现采用测试优先：每个修复先写一个能复现旧问题的失败测试，确认失败原因后写最小修复，再运行相关和全量回归。测试分层如下：

1. 纯函数：边界采样、质量门控、分类质量目标、有效层归一化和 residual-only 梯度路径。
2. head/criterion：含 DN、空 GT、全无效采样、AMP/FP32 几何和精修输出口径。
3. recorder：字段非空、有限性、方法修订号和 expert 几何统计。
4. 集成：模型构造、criterion 接线、训练反向、参数分组完整性和配置开关。
5. 资源烟雾：小 batch CPU/CUDA（可用时）记录 query 范围、checkpoint 开关和峰值显存；不把 CPU 结果当成正式 GPU 结论。

## 验收标准

- 现有相关回归测试保持全绿；新增测试覆盖每个已复现问题并先红后绿。
- 训练前向、反向和评估在有/无 DN、空 GT、边界框和 AMP 条件下均有限。
- recorder 中 capacity、expert geometry、梯度有限性字段不再全部为 null，且字段值与模型实际状态一致。
- 关闭所有修复开关时，前向数值和旧模式行为保持兼容；开启修复只改变明确列出的路径。
- 本地固定批次诊断能报告教师质量改善率、错误放行率、有效层数和分项梯度方向。
- 不以短程 loss、单 checkpoint IoU 或单 seed 早期 mAP 宣布候选成功；正式结果必须按同协议完整记录。

## 风险与降级路径

- 若 normal-only 后仍 OOM，先减少局部 token 或改用更节省激活的融合实现；不静默降低 batch 并继续比较。
- 若质量门控覆盖率接近零，保留诊断结果，先做门控阈值的固定训练集预检；不从验证集调到有利数值。
- 若 residual-only 让主定位损失明显变差，关闭该开关作为兼容候选；不能把它描述成必然改进。
- 若修复版只改善稳定性而不提升 mAP，仍保留其工程价值，论文主张回退为可靠性/可审计性改进。

## 审计后修订约束

本节覆盖审计报告提出的冲突，并优先于前文的同名概括：

1. **已实现项按回归保护处理。** v2 已有的 normal-only query 切片、非重入局部注意力 checkpoint、解码 IoU 门控和定位/分类有效层均值不再重复宣称为 v3 新修复；v3 必须先保留它们的行为测试，再修改未完成部分。
2. **多匹配掩码契约固定。** `decoded_teacher_iou_gate` 的 `active_edges` 必须是 `[matches, 4]`；分类调用端使用完整四边布尔掩码。`matches=0/1/2/4/5`、禁用、warmup 和零分类权重均走真实函数测试。
3. **分类目标写成可执行公式。** 对每个匹配，先用学生解码框得到 detached `q_iou`；分类目标为 `y_class = one_hot(gt) * q_iou`。对每个类别分别计算学生/教师 BCE，只有教师误差在该类别上改善超过 margin 才蒸馏；分类 KL 也逐类别加权。正类、负类激活数和 signed advantage 分开记录。这样“质量感知”不再由全类别平均值代替。
4. **定位 KL 的结论降级。** decoded IoU、逐边误差和支持检查只证明教师终点质量更好，不能证明一次 KL 更新方向安全。v3 将 residual-only 作为可测路径隔离；若要改变定位目标，必须另设期望回归/有界投影候选，并用同预算 GT-only 辅助项对照，不能把 KL 门控写成方向保证。
5. **residual-only 必须接到 capacity。** `CapacityBPDDOptions`、YAML parser、criterion 和 `quality_gated_capacity_distillation` 全链路传递 `residual_gradient_only`。该公式只作用于定位分布 logits，分类 logits 不套用累计残差表达式。
6. **非有限值保持可见。** recorder 对 missing、non-finite 和真实 zero 使用独立计数/状态；`NaN` 不得经过 `or 0.0` 写成数值零。分项梯度夹角是固定批次离线诊断，不由三组总梯度范数替代。
7. **训练和评估缓存分契约。** 训练保留 base 六层 `base_*` 和 expert 独立输出；评估的 `reported_boxes/classes/corners` 三者必须同源、同 query 数。base geometry 与 expert geometry 分开记录，不能用一个 `last_corner_logits` 同时代表两者。
8. **归因矩阵补同容量无 KD。** 除 Repair-full、no-residual 和 no-quality 外，保留 expert-only 配置作为同容量直接监督对照。`Repair-full - expert-only` 才用于估计修复版蒸馏增量；no-checkpoint 只作资源诊断。
9. **基准身份写清。** v3 兼容基准为本工作树提交 `c4d31e1a` 的 capacity-v2 行为，历史服务器运行提交仅用于材料对照。关闭 v3 新开关时要求前向数值等价；已有 v2 修补不在关闭开关后撤回。
