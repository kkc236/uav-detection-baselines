# LPR-RTDETR 定位先验残差头设计

日期：2026-07-31
状态：冻结，进入实现
目标数据集：VisDrone2019-DET
基础模型：Ultralytics RT-DETR-L 8.4.90

## 1. 研究目标

在不修改 RT-DETR 的 Query selection、Encoder、decoder layer、cross-attention、分类头和内部 reference-box 迭代轨迹的前提下，只增强 decoder 输出框的定位回归能力。首轮方法必须满足：

1. LPR 关闭时与 stock RT-DETR 输出逐位等价；
2. 新模块零门控初始化，训练第一步仍能获得非零门控梯度；
3. 10 epoch 从零训练筛选不低于历史同协议 baseline；
4. 通过筛选后从同一 run 的 `last.pt` 续训到总计 100 epoch；
5. 最终同时报告 mAP50-95、mAP50、AP75、定位损失、参数量、GFLOPs 和端到端延迟。

参数量和 GFLOPs 增幅 `<1%`、端到端延迟增幅 `<3%` 是优选目标，不是绝对硬门槛。若稳定定位收益足够，可以接受更高开销，但必须真实报告。

## 2. 方案选择

考虑三种实现：

1. **输出隔离式 LPR（采用）**：每层 decoder 先完成 stock box 计算，LPR 只修正送往监督或最终 postprocess 的输出框；内部 reference box 继续使用 stock 结果。风险最低，仍保留六层深监督。
2. **内部迭代式 LPR（拒绝）**：把 LPR 结果反馈为下一 decoder layer 的 reference box。它会改变后续 deformable attention 采样和 Query 轨迹，与“不碰 Decoder/Query”的约束冲突。
3. **D-FINE/FDR 迁移（暂缓）**：直接把四坐标回归替换为离散分布。成熟度和潜力更高，但工程面、损失和训练协议变化更大，不适合作为第一条低风险 LPR 筛选线。

采用方案 1。D-FINE 只作为 fixed-FDR 后续参考，不进入本轮代码或消融贡献。

## 3. 模块结构

每个 decoder layer 配置一个轻量 `LocalizationPriorRefiner`，输入为：

- 当前层 decoder hidden state `h ∈ R^256`；
- stock 输出框 `b=(cx, cy, w, h)`，仅作为停止梯度的几何先验。

几何向量为：

```text
g(b) = [2cx-1, 2cy-1, log(w), log(h), log(w*h), log(w/h)]
```

其中宽高先截断到 `1e-6`，对数特征截断到 `[-12, 12]`。网络为：

```text
query path:    LayerNorm(256) -> Linear(256, 64) -> SiLU
geometry path: Linear(6, 16) -> SiLU
fusion:        concat(64, 16) -> Linear(80, 4) -> tanh
```

候选框在 logit 空间生成：

```text
b_candidate = sigmoid(inverse_sigmoid(b_stock) + max_logit_delta * tanh(r))
```

最终输出使用门控残差插值：

```text
gate = 0.5 * tanh(alpha)
b_lpr = clamp(b_stock + gate * (b_candidate - b_stock), 1e-6, 1-1e-6)
```

`alpha` 是每层一个标量参数，初始化为 0；残差分支使用固定局部随机种子初始化，同时通过 `torch.random.fork_rng` 保持全局 RNG 状态不变。这样：

- 初始 `gate=0`，`b_lpr` 与 `b_stock` 逐位相同；
- 候选分支初始非零，因此 `alpha` 在第一步可得到梯度；
- 新模块初始化不会改变 stock 模型参数或数据顺序的随机状态；
- 最大插值幅度受限，避免前几轮破坏定位。

六个 refiner 不共享权重。预计新增参数约 11 万，远低于 RT-DETR-L 的 1%。

## 4. Decoder 数据流

自定义 `LPRDeformableTransformerDecoder` 复用原 decoder 的 `layers`、`eval_idx` 和所有 stock heads。每层按 Ultralytics 8.4.90 原实现计算：

1. 执行 stock decoder layer；
2. 执行 stock `dec_bbox_head[i]`；
3. 计算 stock `refined_bbox`；
4. 按 stock 训练逻辑得到本层监督框 `stock_output_bbox`；
5. 仅对 `stock_output_bbox` 应用 LPR，追加到 `dec_bboxes`；
6. `last_refined_bbox` 与下一层 `refer_bbox` 仍使用未经过 LPR 的 stock `refined_bbox`。

分类分数、denoising query 切分、encoder box、Hungarian matcher、所有原始损失和 postprocess 均不改变。训练和推理统一使用 LPR 输出框，内部 decoder 轨迹保持 stock。

## 5. 集成边界

新增文件：

- `src/lpr_head.py`：纯 PyTorch refiner 与 LPR decoder；
- `src/rtdetr_lpr.py`：自定义 detection model、trainer、checkpoint 兼容；
- `scripts/train_rtdetr_lpr.py`：冻结训练协议、诊断和 10→100 resume；
- `scripts/evaluate_lpr_gate.py`：读取历史 baseline 与当前 run，输出机器可读 gate report；
- `scripts/benchmark_lpr.py`：参数量、GFLOPs 和延迟对照；
- 对应单元、集成和 CLI 测试。

不修改 site-packages，不改模型 YAML，不修改现有 BTD-SE、VSF-RMR、IOQC-SA 或 NWD 代码。

stock checkpoint 通过 Ultralytics 的交集加载机制映射到相同 state-dict 路径；新增 `lpr_refiners.*` 参数保持初始化。LPR checkpoint 必须可完整 resume。

## 6. 训练协议

10 epoch 和 100 epoch 使用现有 matched baseline 的冻结协议：

```text
model=rtdetr-l.yaml
data=VisDrone.yaml
imgsz=640
batch=8
workers=8
optimizer=auto
lr0=0.01
lrf=0.01
momentum=0.937
weight_decay=0.0005
warmup_epochs=3.0
amp=True
deterministic=True
seed=0
pretrained=False
nms=False
max_det=300
mosaic=1.0
mixup=0.0
scale=0.5
translate=0.1
perspective=0.0
```

10 epoch 从零开始，不从 100 epoch baseline warm-start。原因是最终论文对比必须与历史 scratch baseline 保持相同训练起点。若通过，使用 10 epoch run 的 `last.pt` resume，将总 epochs 改为 100；不重新初始化或另起随机种子。

每轮额外记录：六层 gate、候选偏移均值/最大值、LPR 参数梯度范数、CUDA 峰值显存，以及 validator 的 AP75。

## 7. 筛选与优化门槛

历史 matched baseline epoch 10：

```text
mAP50-95 = 0.04098
mAP50    = 0.08404
val GIoU = 1.27020
val L1   = 0.19467
```

由于历史 CSV 未保存 epoch-10 AP75，10 epoch gate 使用可审计的历史指标：

1. `mAP50-95 >= 0.04098`；
2. `mAP50 >= 0.08404`，或在 mAP50-95 上至少领先 `0.002` 时允许 mAP50 最多下降 `0.002`；
3. val L1 不高于 `0.19467`，或 val GIoU 至少改善 2%；
4. 无 NaN/Inf，gate 非零，LPR 有非零梯度；
5. 训练、验证和 resume 都成功。

满足 1、3、4、5，且满足 2 的任一分支，判定通过。AP75 仍从新 run 开始记录，并在最终 100 epoch 与重新验证的成熟 baseline AP75 比较。

若首个 10 epoch 未通过，按证据只改一个变量后从相同 seed 重跑：

1. 若 gate 过大或定位损失振荡：把最大 gate 从 `0.5` 降为 `0.25`；
2. 若中间层干扰明显：只启用最后两层 LPR，其余层 gate 固定为 0；
3. 若 gate 几乎不学习：保留零输出等价，给 `alpha` 单独设置 10 倍 optimizer 学习率。

每次优化都必须保留独立 run、配置、日志和 gate report，不覆盖失败证据。

## 8. 100 epoch 判定

100 epoch 完成后，与成熟 baseline `mAP50-95=0.24170`、`mAP50=0.41451` 以及重新验证得到的 AP75 比较：

- 最低通过：mAP50-95 不低于 0.24170，AP75 为正增益；
- 有效论文信号：mAP50-95 至少 `+0.3 pp`，或 AP75 至少 `+0.5 pp` 且总 AP 不下降；
- 同时报告最后三轮平均值，避免只取偶然峰值；
- 若最终未通过，如实停止 LPR，不将 D-FINE/FDR 临时混入同一实验救火。

## 9. 测试与失败安全

实现必须先写失败测试，再写生产代码。覆盖：

- 几何先验数值与极小框稳定性；
- 零门控逐位等价；
- 零门控时 `alpha` 首步梯度非零；
- 全局 RNG 不漂移；
- decoder 内部 reference 轨迹不受 LPR 影响；
- train/eval 输出形状和 denoising 路径；
- stock checkpoint 加载与 LPR checkpoint resume；
- frozen CLI 拒绝协议漂移；
- gate report 判定边界；
- 全项目回归测试。

训练在 `tmux` 中运行，每 epoch 保存 checkpoint。异常退出只从经过哈希/可加载检查的 `last.pt` 恢复，不删除失败 run。

## 10. 批准记录

用户已明确选择“先 LPR”，接受推理开销门槛可适当放宽，并在 2026-07-31 指示部署后执行最新方案、先跑 10 epoch、失败则优化重跑、通过后继续至 100 epoch。本规格把该指令视为对上述低风险输出隔离式 LPR 的执行批准。
