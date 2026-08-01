# LPR-G v2 隔离式质量门控定位细化设计

日期：2026-08-01  
状态：设计已口头批准，等待书面规格复核  
基础模型：Ultralytics RT-DETR-L 8.4.90  
目标数据集：VisDrone2019-DET

## 1. 决策与研究边界

LPR v1 在 seed0、seed1 的严格配对筛选中均未超过 control；同一 LPR checkpoint 的 gate-on/gate-off 独立验证差值又接近零。因此，当前证据不支持“推理时 residual 过度修正好框”这一单一解释。更可信的根因是：LPR v1 把 refined box 送入原始六层主损失，改变了共享 decoder/回归头的训练轨迹，而最终 refinement 本身几乎没有产生可测推理收益。

LPR-G v2 只验证一个更窄、更可证伪的命题：

> 在 stock RT-DETR 的训练目标、匹配、公共参数梯度和内部 reference-box 轨迹完全保留时，最后一层的独立、逐 Query、质量门控定位分支能否带来真实 AP 收益。

本轮不加入 P2/P3、边缘编码、分布回归、Query 改造、分类重打分或 denoising refinement。只运行 seed0，不再使用 seed2，也不把已经完成的 seed2 结果用于结构选择或结论。

## 2. 方案比较与选择

考虑过三种实现：

1. **隔离并行 refinement（采用）**：stock 输出和 stock loss 原样保留；最后一层另算 refined box，并用原 stock 匹配索引计算私有 L1/GIoU。它能直接检验 refinement 的独立贡献。
2. **refined box 替代 stock box 进入主损失（拒绝）**：这是 LPR v1 已暴露风险的路径，会继续改变公共参数的优化轨迹。
3. **纯推理后处理、无独立监督（拒绝）**：隔离最强，但零初始化 residual 没有可靠学习信号，容易再次得到近零推理差值。

采用方案 1。筛选使用固定 10% 子集、seed0、control/LPR-G 各 50 epoch；只有通过预先冻结的门禁，才进入全数据集 100 epoch。

## 3. LPR-G v2 结构

### 3.1 输入与作用位置

只在最后一个 decoder layer 的 normal queries 上产生有效 refinement loss。输入为：

- 最后一层 decoder hidden state `h`；
- 最后一层 stock box `b_stock=(cx, cy, w, h)`；
- 最后一层 stock classification logits `s_stock`。

三者进入私有分支前全部停止梯度。refinement 不回灌下一层 reference box，因为不存在后续 decoder layer；也不改变 encoder、decoder layers、Query selection、分类头、stock bbox heads 或 denoising 路径。

### 3.2 定位质量先验

RT-DETR 当前 criterion 使用 Varifocal Loss。直接复用最后一层 stock 分类分数作为已有质量代理：

```text
q = max_c(sigmoid(detach(s_stock[c])))
```

`q` 是每个 Query 一个标量，范围 `[0,1]`。本版不训练新的 IoU/quality target，也不让 refinement loss 改变 VFL 分数。

### 3.3 几何先验和逐 Query gate

几何向量沿用已验证的稳定编码：

```text
geometry(b) = [2cx-1, 2cy-1, log(w), log(h), log(w*h), log(w/h)]
```

宽高先截断到 `1e-6`，对数特征截断到 `[-12,12]`。私有特征网络为：

```text
query path:    LayerNorm(256) -> Linear(256, 64) -> SiLU
geometry path: Linear(6, 16) -> SiLU
features:      concat(query_feature, geometry_feature)
gate head:     Linear(80, 1)
residual head: Linear(80, 4)
```

逐 Query gate 为：

```text
g_learned = sigmoid(gate_head(features))
g = (1 - q) * g_learned
```

高质量 stock Query 自动得到更小的最大修正权限；低质量 Query 是否修正仍由私有 gate head 决定。分类分数本身不被修改。

### 3.4 有界 residual 与零初始化

残差在 box-logit 空间施加：

```text
delta = max_logit_delta * tanh(residual_head(features))
b_refined = sigmoid(logit(clamp(detach(b_stock))) + g * delta)
```

`max_logit_delta` 初始冻结为 `0.5`。`residual_head` 和 `gate_head` 的 weight/bias 都全零初始化，因此初始 `g_learned=0.5`，且 `b_refined` 与 `b_stock` 逐值相等。第一步由非零的 box loss 梯度先驱动 residual head；residual 离开零点后，gate head 才获得决定修正幅度的梯度。这避免 gate 的随机初值在早期制造 Query 偏置。

所有私有初始化都在 `torch.random.fork_rng` 内完成，不推进公共模型或数据增强使用的全局 RNG。

## 4. 训练数据流与损失隔离

### 4.1 stock 路径保持原样

训练前向仍向 Ultralytics stock criterion 传入原始：

- encoder boxes/scores；
- 六层 decoder stock boxes/scores；
- denoising boxes/scores 和 `dn_meta`。

stock 的 VFL、L1、GIoU、auxiliary losses、denoising losses、归一化和权重均不改名、不改值、不改计算顺序。refined box 不替代任何 stock box，不进入 stock 主损失或辅助损失。

### 4.2 精确复用 stock 最后一层匹配

criterion 在计算 normal-query 最后一层 stock loss 时照常执行 Hungarian matcher。自定义 criterion 只记录这一次已经产生的 `match_indices`，再把同一索引提供给 refinement loss；它不执行第二次 matcher，也不把这组索引强行用于 stock auxiliary layers。

这条限制很重要：Ultralytics 的 auxiliary layers 默认可以各自匹配。把最后一层索引传给整个 stock criterion 会改变 baseline，因此禁止这样实现。

### 4.3 私有 refinement loss

只对最后一层 normal-query refined boxes 计算：

```text
L_refine = L1_refine + GIoU_refine
```

两项复用 stock criterion 的 bbox loss 实现、增益和 GT 归一化，初始额外系数为 `1.0`。不计算 refinement 分类损失，不细化 denoising queries。日志单独命名为 `loss_bbox_refine` 和 `loss_giou_refine`，不覆盖 stock 的 `loss_bbox`、`loss_giou`。

由于 `h`、`b_stock`、`s_stock` 均 detach，`L_refine` 只能更新 LPR-G 私有参数。AMP unscale 后，stock 参数按 baseline 原规则单独做 `max_norm=10` 梯度裁剪；LPR-G 私有参数另做 `max_norm=10` 裁剪，禁止把两类参数放入同一个全局 norm，否则私有梯度会间接改变 stock 更新幅度。

## 5. 推理与独立评估

训练时保存 stock 和 refined 两套最后层输出；默认验证/部署使用 `b_refined` 与原 stock 分类分数。模型必须提供不改 checkpoint 的输出切换：

- `refined`：LPR-G box + stock score；
- `stock`：同一 checkpoint 的 stock box + stock score。

独立评估 CLI 接受本地 checkpoint 或从 GitHub 恢复的 checkpoint，能够在不训练的情况下分别验证 `stock`/`refined`，生成 mAP50-95、mAP50、AP75、precision、recall、L1/GIoU 和 gate/residual 分布报告。该同 checkpoint 消融是正式门禁，避免再次把公共参数变化误判成 refinement 收益。

## 6. 冻结实验协议

### 6.1 公共环境与参数

control 和 LPR-G 必须统一为：

```text
Ultralytics RT-DETR-L 8.4.90
NVIDIA GeForce RTX 4090 24GB；driver 550.142
Python 3.10.12
PyTorch 2.5.1+cu121；Torchvision 0.20.1+cu121；CUDA 12.1
pretrained=False；imgsz=640；batch=8；workers=8；device=0
AMP=True；固定 scale=128；deterministic=True；cache=False；seed=0
optimizer=MuSGD；lr0=0.01；lrf=0.01；momentum=0.937
weight_decay=0.0005；warmup_epochs=3.0；warmup_momentum=0.8
warmup_bias_lr=0.0；nbs=64；cos_lr=False
queries=300；max_det=300；NMS=False
mosaic=1.0；close_mosaic=10；mixup=0.0；scale=0.5；translate=0.1
degrees=0.0；shear=0.0；perspective=0.0；flipud=0.0；fliplr=0.5
hsv_h=0.015；hsv_s=0.7；hsv_v=0.4；cutmix=0.0；copy_paste=0.0
```

数据集语义 SHA256 必须为 `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`。筛选子集固定 647 张，SHA256 必须为 `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`。

### 6.2 50 epoch 筛选

- 数据：固定 10%/647 张训练子集，同一完整 val 548 张；
- seed：仅 seed0；
- 两臂：fresh `control` 50 epoch、fresh `lprg-v2` 50 epoch；
- 顺序：control 后 LPR-G；
- 初始化：两臂加载同一冻结公共参数；LPR-G 只额外加载确定性的私有零输出状态；
- 每臂 scheduler 都从 epoch 0 开始配置为总计 50 epoch；
- 不从 LPR v1、历史 10 epoch 或历史 100 epoch checkpoint warm-start。

同一公共初始化、样本顺序、数据增强随机序列、验证预处理、指标代码、类别映射、checkpoint/resume 规则必须由机器可读 manifest 证明。

Ultralytics 8.4.90 对 `deterministic=True` 的实际实现是
`torch.use_deterministic_algorithms(True, warn_only=True)`。PyTorch 2.5.1+cu121 的
RT-DETR CUDA 路径包含没有确定性反向实现的 `grid_sampler_2d_backward_cuda`，因此两个
独立训练进程即使拥有相同初始化和 RNG 序列，更新后的公共参数/optimizer SHA256 也不能
作为 bitwise 相等门禁。隔离门禁改为：公共初始化 fingerprint 严格相等；初始 stock
outputs 和 stock loss 项逐值相等；目标 4090 上一步公共梯度全局相对 L2 漂移不超过
`0.005`。逐 epoch 公共 fingerprint 仍完整记录，用于发现断点、缺失和异常轨迹，但不以
跨 arm SHA256 相等作为工程有效性的必要条件。

### 6.3 全数据集 100 epoch

50 epoch 筛选通过后，在 6471 张完整训练集上从相同 seed0 公共初始状态重新开始 100 epoch；禁止把 10% 子集 checkpoint resume 到全量数据。默认运行 fresh control/LPR-G 配对 100 epoch。只有现有 full-data seed0 baseline 的环境、协议、初始化和数据顺序 manifest 全部逐项一致时，才允许复用；任一证据缺失就重跑 control。

不启动 seed1/seed2。100 epoch 两臂继续执行相同逐 epoch GitHub 保护和独立 `stock/refined` 评估。

## 7. 每 epoch GitHub 保护、续跑与审计

### 7.1 每轮必须上传的关键数据

每个已完成 epoch 都生成并上传：

- 完整追加式 `results.csv`；
- control/LPR-G 的 AP75 与定位 loss 诊断 JSONL；
- LPR-G 的 gate、quality、residual、私有梯度、stock/private clip norm 统计；
- `args.yaml`、冻结协议 manifest、源码 commit、环境和数据哈希；
- checkpoint manifest：completed epoch、文件字节数、SHA256、GitHub Release asset ID；
- 发布账本：本 epoch 的结果分支 commit SHA、release URL 和远端验证状态。

轻量数据提交到专用 `training-results` 分支，目录按实验和 arm 隔离。每个 epoch 是独立 Git commit，因此即使工作文件持续追加，Git 历史仍保留每轮状态。

### 7.2 可续跑 checkpoint

设置 `save_period=1`。每轮 `on_model_save` 后对 checkpoint 做原子 staging 和 Ultralytics 可恢复性检查，再上传到专用 GitHub Release；远端 asset 与 manifest 的字节数、epoch 和 SHA256 验证通过后，才把该 epoch 标记为 `published`。

远端和服务器都滚动保留最近 3 个已验证 checkpoint；旧 checkpoint asset 可以轮换删除，但其指标、manifest 和发布 commit 永久保留在 Git 历史。恢复时选择最高 completed epoch 的 checkpoint/manifest 配对，下载到 `.tmp`，校验大小、SHA256、epoch、optimizer、EMA/model、协议和 arm 后原子改名，再用 `resume=True` 恢复 optimizer、scheduler、AMP scaler 和 epoch。

同步失败最多自动重试 10 次，每次间隔 30 秒。仍失败则训练以可恢复错误退出，保留本地 checkpoint；监督器先补齐该 epoch 的 GitHub 发布，再从它 resume。禁止跳过未发布 epoch 后继续声明实验完成。

control 和 LPR-G 使用不同 asset prefix，禁止跨 arm、跨 10%/full stage 或跨设计版本恢复。

## 8. 50 epoch 预注册判定

### 8.1 工程有效性门禁

以下任一失败都判为“实验无效”，先修工程问题，不做科学结论：

1. 环境、数据、子集、源码、公共初始化或协议哈希不一致；
2. 一步 canary 中 method 的 stock outputs 或 stock loss 字典与 control 不逐值相等；
3. 公共初始化 fingerprint 不相等、目标 4090 canary 的公共梯度相对 L2 漂移超过
   `0.005`，或任一 epoch 缺失合法的公共参数/optimizer fingerprint；
4. AMP scale 不是 128、发生 skipped step、出现 NaN/Inf；
5. 任一完成 epoch 缺少 GitHub 指标 commit 或已验证 checkpoint 发布记录；
6. resume canary 不能从 GitHub checkpoint 完整恢复并完成一次独立评估。

### 8.2 科学通过门禁

工程有效后，LPR-G v2 必须同时满足：

1. epoch 50 的 mAP50-95 严格高于 paired control；
2. epoch 41-50 的平均 mAP50-95 严格高于 paired control；
3. epoch 50 或 epoch 41-50 平均 AP75 至少一项严格高于 paired control，且另一项不低于 control；
4. epoch 50 和尾 10 轮平均 mAP50 都不得比 control 低超过 `0.001`；
5. 同一 LPR-G checkpoint 的 `refined` 独立评估必须同时高于 `stock` 的 mAP50-95 和 AP75；
6. refinement L1/GIoU、gate、residual、私有梯度均有限；`p95(gate)>1e-3`，residual RMS 非零；
7. 参数量、GFLOPs 和端到端延迟增幅完成实测并如实报告。`<1%/<1%/<3%` 仍是优选目标，但按用户授权不是硬淘汰线。

本轮只有 seed0，因此“通过”只表示结构可行并获准进入全量 100 epoch，不表示跨 seed 统计显著性或论文结论已经成立。

## 9. 未通过后的改良规则

50 epoch 未通过时先输出固定诊断包：逐 epoch paired 曲线、尾 10 轮统计、同 checkpoint stock/refined 消融、按 quality 分桶的 gate/residual/IoU 变化、匹配 Query 的 L1/GIoU 变化、公共参数隔离审计和推理开销。

随后遵守以下规则：

1. 工程门禁失败只修复隔离、发布、恢复或评估错误，设计版本不变，并从相同初始状态重跑受影响的完整 arm；
2. 科学门禁失败时，每个新版本只修改一个有诊断证据支持的结构变量；
3. 每次修改先写规格增补、测试和新 commit，使用新 run ID，绝不覆盖失败 run；
4. 公共协议未改变时复用同一 50 epoch control；任何会改变 baseline 轨迹的参数、数据或损失变化都必须重跑 control；
5. 不根据单个最好 epoch 选模型，不把 seed2 或历史非配对结果补进门禁。

优先诊断顺序为：先确认隔离和匹配索引，再检查 residual 是否学习，再检查 gate 是否随 quality 分化，最后才考虑缩小 `max_logit_delta` 或增加显式 gate 监督。没有证据时不叠加模块。

## 10. 测试与交付边界

实现必须遵循测试驱动开发，至少覆盖：

- residual 零初始化和初始 stock/refined 逐值等价；
- VFL quality、hidden、geometry 三路 detach；
- gate 形状为 `[batch, query, 1]` 且逐 Query 可变化；
- 极小框 logit/geometry 数值稳定；
- 只细化最后层 normal queries，denoising 不进入私有 loss；
- stock criterion 的输出键和值与 Ultralytics 8.4.90 完全一致；
- 只记录主 normal layer 的原匹配索引，auxiliary matcher 行为不变；
- refinement loss 只给私有参数梯度；stock 和 private 独立裁剪；
- common-state 公共初始化逐值审计、一步梯度相对漂移门禁和逐 epoch fingerprint 完整性审计；
- stock/refined 独立评估切换；
- checkpoint 本地/Release 恢复、损坏拒绝、跨 arm/stage 拒绝；
- 每 epoch 发布队列不漏号、失败重试、远端校验后确认、最近 3 份轮换；
- 50 epoch 比较器的边界条件和全项目回归测试。

不修改 site-packages，不覆盖 LPR v1 证据，不删除失败 checkpoint/run。生产实现、部署和训练只有在本设计书面复核后进入详细实施计划。
