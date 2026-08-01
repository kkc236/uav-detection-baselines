# I-TBER v1.1 冻结式轨迹—边界证据细化设计

日期：2026-08-01

状态：方案 A 已获用户批准，等待书面规格复核

基础模型：Ultralytics RT-DETR-L 8.4.90

目标数据集：VisDrone2019-DET

设计版本：`i-tber-v1.1`

## 1. 决策与研究命题

LPR-G v2 的 30-epoch cutoff 筛选已经判定为 `scientific_failed`。其同 checkpoint 消融中，refined mAP50-95 未超过 stock，且有效 gate 均值约为 `4.92e-5`，表明无显式监督的乘法 gate 发生坍缩。该失败 run、checkpoint、逐 epoch 发布账本和比较报告永久保留，不覆盖、不改写，也不再启动 LPR-G 长训。

I-TBER v1.1 改为冻结成熟 RT-DETR-L，只训练推理时真实存在的私有细化模块。它验证的命题是：

> 在检测器、Query、匹配、分类分数和公共参数完全冻结时，最后三层框轨迹与预测框四边附近的稀疏视觉证据，能否为逐边定位修正提供独立且可测的增益？

该设计不把独立训练进程中的公共参数波动解释成方法收益。核心因果证据始终是同一 I-TBER checkpoint 的 `refined` 与 `stock` 输出差值。

## 2. 冻结基线权威

I-TBER 只使用以下成熟 seed0 baseline：

```text
path: /data/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt
bytes: 66262262
sha256: 54ce60289dd34c6750b8ba5f7516eefcf3afef6c174c6e4f3b1ef810c883099b
Ultralytics: 8.4.90
training epoch: 100
pretrained: False
imgsz: 640
batch: 8
seed: 0
```

该 checkpoint 的历史第 100 轮 mAP50-95 为 `0.24170`；在当前 Ultralytics 原生验证协议下复评约为 `0.241803`。实施前必须在当前 4090 环境重新生成一次 stock 权威评估，并把 checkpoint、数据集、类别映射、验证配置和输出指标 SHA256 一并锁定。

当前 LPR-G 10% cutoff control 只有接近零的 mAP，不作为 I-TBER 冻结起点。

## 3. 冻结边界

RT-DETR 的以下部分永久 `requires_grad=False`，并始终保持 `eval()`：

- backbone；
- hybrid encoder；
- query selection；
- 六层 decoder；
- stock 分类头与回归头；
- denoising 路径；
- BatchNorm 统计量和其他缓冲区。

检测器前向在 `torch.no_grad()` 下执行。所有送入 I-TBER 的张量都显式 `detach`。训练后必须验证每一个 detector 参数的梯度均为 `None`，且权重和缓冲区 SHA256 与冻结起点逐值相同。

I-TBER 不修改 Hungarian matcher，不进行第二次匹配，不改变分类分数，不增加 Query，不引入 P2，不使用 NMS，并保持 `max_det=300`。

## 4. 证据提取

冻结检测器为每张图像和每个 normal Query 提供：

```text
h_last:       [B, 300, 256]
box_L2:       [B, 300, 4]
box_L1:       [B, 300, 4]
box_L:        [B, 300, 4]
score_logits: [B, 300, 10]
F3:           [B, C3, H3, W3]
```

`box_L2/box_L1/box_L` 表示最后三层 normal-query stock box，统一转换为归一化 `xyxy`。`F3` 是 RT-DETR head 接收到的现有最高分辨率特征图，不创建 P2 路径。

实现采用只记录证据、不改变 stock 输出的 decoder 包装器。工程 Canary 必须证明包装前后 stock boxes、scores 和指标逐值相等。denoising Query 不写入缓存、不进入私有 loss。

## 5. I-TBER 私有模块

### 5.1 Query、几何与质量

Query 路径：

```text
LayerNorm(256) -> Linear(256, 64) -> SiLU
```

几何与质量输入为：

```text
[2cx-1, 2cy-1, log(w), log(h), log(w*h), log(w/h), q, entropy]
```

其中 `q=max_c(sigmoid(score_logits[c]))`。该向量经过 `Linear(8,16) -> SiLU`。质量只作为特征，不再使用 `(1-q)` 作为硬 gate 上限。

### 5.2 有符号轨迹

对四条边分别计算：

```text
s  = [w, h, w, h]
v1 = (edge_L1 - edge_L2) / (s + eps)
v2 = (edge_L  - edge_L1) / (s + eps)
T  = [v1, v2, abs(v1)+abs(v2), v2-v1, v1*v2, abs(v2)/(abs(v1)+eps)]
```

轨迹保留方向、幅度、加速度、反向震荡和稳定趋势。未通过 Probe 前只称为 `edge evolution state`，不得提前宣称为不确定性估计。

### 5.3 四边稀疏视觉证据

每条边沿边界在 `{0.25, 0.50, 0.75}` 三个位置采样；每个位置沿法线读取 `outside/edge/inside`，每个 Query 共 `4*3*3=36` 个点。

归一化法线距离为：

```text
d = clip(0.08 * min(w,h), 1/640, 4/640)
```

F3 先经私有 `1x1 Conv` 投影至 32 通道，再使用 `grid_sample` 双线性采样。`align_corners` 必须在规格、缓存 manifest 和测试中固定为 `False`。边界证据为：

```text
z_e = [f_edge, f_inside-f_outside, abs(f_inside-f_outside)]
Linear(96,32) -> SiLU
```

采样坐标必须使用 letterbox 后的网络输入坐标系；极小框、贴边框和越界点采用合法截断，不得产生 NaN/Inf。

### 5.4 逐边融合与输出

每条边的固定输入槽为：

```text
Query:             64
Boundary evidence: 32
Geometry/quality:  16
Trajectory:         6
Edge embedding:     8
Total:             126
```

融合网络为：

```text
Linear(126,64) -> SiLU -> Linear(64,64) -> SiLU
```

四条边共享融合网络、gate head 和 residual head，通过 8 维 edge embedding 区分方向。输出为逐边标量：

```text
g_e = sigmoid(gate_raw_e)
r_e = tanh(residual_raw_e)
```

最大修正比例固定为 `rho=0.05`：

```text
edge_refined = edge_stock + rho * s_e * g_e * r_e
```

更新后截断到 `[1e-6,1-1e-6]`，保证 `x_right>x_left`、`y_bottom>y_top`，再转回 `cxcywh`。分类 scores 完全沿用 stock 输出。

gate 与 residual 输出层的 weight/bias 全部零初始化，因此初始 refined box 与 stock box 逐值相等。所有私有参数在 `torch.random.fork_rng` 中初始化，不推进全局 RNG。

## 6. v1.1 监督分解

v1.0 同时令 gate target 和 residual target 编码修正幅度，会使最终乘积近似按误差平方衰减。v1.1 明确分工：gate 只学习幅度，residual 只学习方向。

对 stock 匹配的每条边定义：

```text
u_e = clip((edge_gt-edge_stock)/(rho*s_e+eps), -1, 1)
t_e = abs(u_e)
d_e = 0                         if t_e == 0
      u_e / t_e                 otherwise
```

其中 `t_e` 是 gate 软目标，`d_e` 是 residual 方向目标。最终理想乘积 `t_e*d_e=u_e`，不再发生双重幅度衰减。

损失包括：

1. `L_box`：复用 stock match，对 refined box 计算原尺度 L1 与 GIoU；
2. `L_dir`：对 matched edges 计算按 `t_e` 加权并归一化的 SmoothL1(`r_e`,`d_e`)；零误差边不提供不稳定方向监督；
3. `L_gate_pos`：matched edges 上 BCEWithLogits(`gate_raw_e`,`t_e`)；
4. `L_gate_neg`：unmatched normal queries 上 BCEWithLogits(`gate_raw_e`,0)；
5. `L_noop`：unmatched Query 的 `q*abs(g_e*r_e)` 均值，保护高分未匹配预测。

正负 Query 分开归一化，避免 300 个 Query 中负样本数量压倒 matched positive：

```text
L_gate    = 0.5 * L_gate_pos + 0.5 * L_gate_neg
L_private = L_box + 1.0*L_dir + 0.25*L_gate + 0.05*L_noop
```

所有 target、stock match 和 detector evidence 均 detach。私有 loss 只能更新 I-TBER 参数。

## 7. P0-P3 同容量 Probe

四个 Probe 使用同一模块、相同参数量、相同初始化和相同训练协议；缺失模态使用同形状零张量，不删除层：

| Probe | Query/geometry | Trajectory | Boundary evidence |
|---|---:|---:|---:|
| P0 | yes | zero | zero |
| P1 | yes | yes | zero |
| P2 | yes | zero | yes |
| P3 | yes | yes | yes |

Gate 1 为避免四个 Probe 重复运行冻结 detector，在 train647 与 val548 上生成一次无随机增强、固定 letterbox 的分片 evidence cache。F3 以 FP16 保存，boxes、scores、匹配索引和标签使用能保持指标一致的精度。每个 shard 记录图像 ID、shape、letterbox 元数据、baseline SHA256、数据集 SHA256、类别映射 SHA256、源代码 commit、字节数和自身 SHA256。该缓存只用于信息量 Probe，不作为 Gate 2 或正式 AP 训练输入。

缓存 Canary 必须证明：

- 缓存 stock 输出与直接 detector 输出逐值一致；
- 缓存 stock AP 与直接验证 AP 一致；
- shard 缺失、顺序错误或哈希错误会被拒绝；
- train/val 不交叉；
- 同一 manifest 可重复加载并产生逐值一致结果。

P0-P3 均从相同私有初始状态 fresh 训练 12 epoch。P3 只有同时满足以下条件才通过 Gate 1：

1. 相对 P0，val edge MAE 至少下降 5%；
2. 相对 P2，val edge MAE 至少再下降 1.5%；
3. 同 checkpoint refined matched IoU 相对 stock 至少提高 `0.005`；
4. tiny/small correction-direction accuracy 相对 P0 至少提高 3 个百分点；
5. P3 在四个 Probe 中取得最佳 val edge MAE 与 matched IoU；
6. gate、residual、loss 和梯度均有限，且 gate/residual 未坍缩或饱和。

任一核心条件失败即停止完整 I-TBER 检测筛选，不得通过修改阈值、追加 P2、新 Query、attention 或多尺度融合补救。

## 8. 优化与训练协议

I-TBER 是冻结 detector 的独立后训练阶段，因此采用方案 A 的私有优化器，不冒充端到端 baseline 优化器：

```text
optimizer: AdamW
lr: 1e-3
weight_decay: 1e-4
betas: (0.9, 0.999)
private gradient clip: 10
AMP: True
fixed AMP scale: 128
seed: 0
imgsz: 640
batch: 8
workers: 8
cache: Gate 1 evidence cache only; no Ultralytics image cache
```

P0-P3、快速筛选和正式私有模块训练使用完全相同的私有优化规则。该 AdamW 例外只适用于冻结 I-TBER 私有参数；不会改变已完成的 RT-DETR baseline，也不与历史端到端创新实验混表为同一优化过程。

Gate 1 明确报告为固定 evidence cache 上的信息量 Probe，不声称复现端到端训练增强。Gate 2 与全数据正式训练不读取 Gate 1 cache，而是按统一 VisDrone 参数逐 batch 应用增强，并通过冻结 detector 的 `no_grad` 前向即时产生证据；这样保留每 epoch 的预注册样本顺序和增强随机序列，同时不持久化体积不可控的多 epoch F3 缓存。

## 9. Gate 0：工程 Canary

进入长任务前必须全部满足：

1. 零初始化时 stock/refined boxes 和指标逐值相等；
2. decoder 记录包装器不改变 stock boxes/scores；
3. detector 参数梯度全部为 `None`，私有参数存在有限非零梯度；
4. detector 权重与缓冲区在训练前后 SHA256 相同；
5. 只复用最后层 normal-query stock match，denoising Query 不进入私有 loss；
6. matched/unmatched gate 分开归一化；
7. 极小框、贴边框、越界采样和空匹配 batch 不产生 NaN/Inf；
8. `align_corners=False` 与 letterbox 坐标变换有逐值测试；
9. checkpoint 可切换 `stock/refined`，并完整恢复 optimizer、AMP scaler、epoch 和私有参数；
10. 参数、GFLOPs 和 CUDA 延迟基准脚本可执行并如实记录。

Gate 0 失败只修工程问题，不进入 Probe，也不解释科学效果。

## 10. Gate 2：固定10%快速筛选

Gate 1 通过后，P3 从同一私有初始状态 fresh 启动，不复用 Probe 的已训练权重：

```text
train: fixed 647 images
val: full 548 images
seed: 0
epochs: 12
detector: frozen mature 100-epoch baseline
```

每 epoch 都运行同 checkpoint `stock/refined` 验证并发布 GitHub 证据。第12轮必须同时满足：

1. `delta mAP50-95 >= +0.0020`；
2. `delta AP75 >= +0.0030`；
3. `delta AP50 >= -0.0005`；
4. AP-tiny 或 AP-small 至少一项为正；
5. matched positives 中 IoU 改善数量大于恶化数量；
6. unmatched 有效修正 RMS 不超过 matched positive 的 25%；
7. gate 与 residual 非零、未坍缩、未饱和；
8. 三次独立评估输出逐值一致。

不得用单个最佳 epoch 替代第12轮冻结门槛。Gate 2 失败即停止，不运行全数据私有训练。

## 11. 全数据正式实验

Gate 2 通过后冻结模块结构、损失权重、优化器和阈值。正式实验从同一确定性私有初始状态 fresh 开始，不 resume 10% checkpoint：

```text
train: full 6471 images
val: full 548 images
seed: 0 only
detector: frozen mature 100-epoch baseline
I-TBER private epochs: 30
```

基础 detector 已完成100轮；这里的30轮仅训练 I-TBER 私有插件。正式继续条件为：

- 最终 mAP50-95 至少 `+0.0030`；
- 最终 AP75 至少 `+0.0050`；
- AP-tiny/AP-small 至少一项稳定提高；
- 最后5轮 refined 平均值高于同 checkpoint stock；
- matched IoU 改善数大于恶化数；
- detector SHA256 全程不变。

本轮只使用 seed0，不启动 seed1/seed2。参数量与 GFLOPs 增幅目标仍为 `<1%`，端到端延迟目标为 `<3%`；按照用户授权，超出延迟目标必须如实报告和分析，但不单独覆盖科学指标判定。

## 12. 逐 epoch GitHub 保护与恢复

使用新的 run ID、结果目录、release asset prefix 和设计版本，禁止复用或覆盖 LPR/LPR-G 证据。每个完成 epoch 必须发布：

- train/private losses 与 gate/residual 分布；
- stock/refined AP、AP50、AP75、AP-tiny、AP-small；
- matched edge MAE、matched IoU、改善/恶化计数；
- matched/unmatched correction RMS；
- detector SHA256、baseline checkpoint SHA256、cache manifest SHA256；
- 私有 checkpoint、optimizer、scheduler、AMP scaler、epoch；
- 参数/GFLOPs/延迟报告或其固定基准引用；
- source commit、环境、数据和类别映射哈希；
- GitHub result commit、release asset ID、bytes、SHA256 与远端验证状态。

每个 epoch 只有在远端 checkpoint 与轻量结果均验证成功后才记为 `verified=true`。本地和 GitHub 滚动保留最近3个私有 checkpoint，完整轻量历史永久保留。恢复只接受同 baseline SHA、同 cache manifest、同 probe、同 stage、同 seed 和同设计版本的最高已验证 checkpoint。

GitHub token 只从服务器权限为600的 token 文件读取，不写入命令行、日志、manifest、checkpoint 或 Git 历史。

## 13. 评估与消融

必须保留以下同 checkpoint 消融：

- stock；
- plain residual MLP；
- P0 Query+geometry；
- P1 P0+trajectory；
- P2 P0+boundary evidence；
- P3 full I-TBER；
- shuffled trajectory；
- shuffled boundary sampling coordinates；
- no inside-outside difference；
- per-query gate 代替 per-edge gate；
- no explicit gate supervision；
- `rho` 为 `0.025/0.05/0.10`；
- F3 与下一层低分辨率特征对照。

结构选择只由预注册 Gate 1 与 Gate 2 决定。正式30轮开始后不得再根据 val 修改结构或超参数。

## 14. 失败止损

- Gate 0 失败：修复工程隔离、坐标、缓存、恢复或评估问题后重跑 Canary；不跑 GPU 长任务。
- Gate 1 失败：判定轨迹或边界证据没有达到增量信息门槛，停止 I-TBER，不实现检测 AP 长训。
- Gate 1 通过但 Gate 2 失败：只检查匹配索引、正负 gate 归一化、残差饱和、采样坐标和 unmatched no-op；不新增模块。
- boundary evidence 有效但 trajectory 无效：停止以 I-TBER 命名，另立不含 trajectory 的设计版本。
- 同 checkpoint refined/stock AP 接近零：判定插件无真实推理贡献，不用公共训练波动解释。
- 任何 NaN/Inf、detector SHA 漂移、缓存哈希不一致或未验证 epoch：停止并保留证据，不降低 batch、不关闭 AMP、不跳过发布账本。

## 15. 实施边界

不得修改 site-packages，不得删除或覆盖历史 run，不得把训练-only loss 写成论文核心贡献。正式贡献必须是推理时实际执行的：

```text
冻结 RT-DETR-L
+ 最后三层有符号框轨迹
+ F3 四边 inside/edge/outside 稀疏采样
+ Query/geometry/quality 条件化
+ 四边显式监督 gate
+ 四边有界方向 residual
```

本规格获书面复核后，下一步才编写逐文件、逐测试、逐提交的实施计划；在实施计划获准前不修改生产训练代码。
