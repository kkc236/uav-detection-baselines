# IBER-BE v1.0：双分辨率隔离式边界证据细化器

日期：2026-08-02

基础模型：Ultralytics RT-DETR-L 8.4.90

目标数据集：VisDrone2019-DET

实验范围：seed0、固定 10% 子集 30 epoch 筛选

## 1. 决策与结论边界

I-TBER v1.1 已在工程有效的 P0–P3 Gate-1 中判定为
`scientific_failed`。四臂均正常学习，但完整 P3 没有优于 P0/P2：

- P0 matched IoU 增量约 `+0.004760`；
- P2 matched IoU 增量约 `+0.004993`，是四臂最佳；
- P3 matched IoU 增量约 `+0.004853`；
- P3 edge MAE 没有比 P0 下降；
- P3 tiny/small correction-direction accuracy 分别比 P0 低约
  `0.55/0.62` 个百分点。

权重审计证明 F3 boundary 与 trajectory 路径都收到显著更新，因此失败不能解释为
分支未连接或没有梯度。I-TBER v1.1 的结果、checkpoint、Gate 报告和日志永久保留，
不得覆盖、改阈值或绕过 Gate-1。

IBER-BE v1.0 采用用户确认的成功率优先路线：

> 永久删除 trajectory，只验证双分辨率稀疏 boundary evidence 是否能在冻结
> RT-DETR 的条件下带来可测定位收益。

IBER-BE 不再使用 I-TBER 名称，不把 v1.1 的失败结果改写为成功前导。

## 2. 为什么不能只重复 F3 boundary

v1.1 的 P2 已经验证：只加入现有 stride-8 F3 boundary evidence，matched IoU 有极小
改善，但 edge MAE 和 tiny/small correction direction 没有形成增量。继续只改 fusion、
loss 权重或训练轮数，无法增加输入证据的空间分辨率。

IBER-BE 保留 F3 作为语义边界上下文，同时从检测器已经接收的 640×640 RGB 输入中只在
300 个 stock box 的四条边附近稀疏采样。RGB 证据不经过全图卷积、不建立 P2、不写回
backbone/encoder/decoder，仅供私有边界细化器使用。

## 3. 冻结公共权威

以下内容沿用 I-TBER v1.1 已通过的当前服务器权威：

- baseline checkpoint SHA256：
  `54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B`；
- dataset SHA256：
  `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`；
- 固定 10% 子集：647 张，SHA256：
  `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`；
- val：548 张；类别数：10；
- GPU：NVIDIA GeForce RTX 4090；
- Python `3.10.12`、PyTorch `2.5.1+cu121`、Torchvision
  `0.20.1+cu121`、CUDA `12.1`、Ultralytics `8.4.90`；
- 当前执行驱动 `570.133.07`，继续绑定已批准 runtime amendment；
- imgsz `640`、batch `8`、workers `8`、device `0`、AMP 固定 scale
  `128`、seed0、deterministic、cache=False；
- query 数 `300`、max_det `300`、NMS=False；
- 训练增强保持已冻结 VisDrone 参数；
- detector 永久 `eval()`、`requires_grad=False`，detector SHA256 全程不变。

本实验使用成熟 baseline checkpoint 的同-checkpoint stock/refined 对照，不重新训练
公共 RT-DETR 参数，因此不会把私有插件收益与公共训练波动混合。

## 4. 输入与隔离边界

IBER-BE 接收：

```text
h_last:       [B, 300, 256]
box_stock:    [B, 300, 4]
score_logits: [B, 300, 10]
F3:           [B, C3, H3, W3]
image_rgb:    [B, 3, 640, 640]
```

所有输入进入私有模块前均 `detach`。`image_rgb` 是同一次 detector forward 使用的、
已 letterbox/增强并归一化到 `[0,1]` 的 RGB tensor，不重新读取图片，也不改变增强序列。

明确删除：

- `box_L2`、`box_L1`；
- trajectory state；
- trajectory loss、shuffle trajectory 消融及相关缓存字段。

IBER-BE 不修改 Backbone、Hybrid Encoder、Query Selection、Decoder、stock 分类/回归头、
Hungarian matcher、denoising、分类分数、NMS 或 max_det。

## 5. 双分辨率边界证据

### 5.1 公共边界几何

stock box 转为归一化 `xyxy`。每条边沿切向固定取：

```text
0.25, 0.50, 0.75
```

所有 grid 坐标使用 `align_corners=False`，越界位置使用 border padding。极小框、贴边框
和轻微越界 stock box 必须有限且可复现。

### 5.2 F3 语义边界证据

保留 v1.1 的 F3 路径作为历史可比项：

1. 私有 `1×1 Conv` 将 F3 投影到 32 通道；
2. 法向距离
   `d_f3 = clip(0.08 * min(w,h), 1/640, 4/640)`；
3. 每个切向位置读取 outside/edge/inside；
4. 每条边形成
   `[edge, inside-outside, abs(inside-outside)]`；
5. `Linear(96,32) -> SiLU` 得到 F3 boundary embedding。

### 5.3 RGB 高分辨率边界证据

RGB 路径直接在输入 tensor 上稀疏读取，不创建高分辨率特征图。每个切向位置使用两个
固定、尺度受控的法向半径：

```text
d_near = clip(0.08 * min(w,h), 1/640, 4/640)
d_far  = clip(0.20 * min(w,h), 2/640, 8/640)
```

其中 `w,h` 为归一化 stock box 尺寸。每条边对三个切向位置取均值后形成：

```text
edge_rgb                         3
near_inside - near_outside      3
abs(near_inside-near_outside)   3
far_inside - far_outside        3
abs(far_inside-far_outside)     3
总计                            15
```

编码器固定为：

```text
Linear(15,16) -> LayerNorm(16) -> SiLU
```

RGB 路径不得加入 Sobel、全图卷积、可学习采样偏移、attention、多尺度图像金字塔或 P2。
首版只验证显式双半径 inside/outside 对比。

## 6. 私有细化头

### 6.1 基础条件路径

```text
query:    LayerNorm(256) -> Linear(256,64) -> SiLU
geometry: Linear(8,16) -> SiLU
edge id:  Embedding(4,8)
base:     Linear(88,64) -> SiLU -> Linear(64,64) -> SiLU
```

geometry 继续使用中心、`log(w)`、`log(h)`、`log(area)`、`log(w/h)`、detached
quality 和 entropy。

### 6.2 独立 boundary 路径

```text
boundary input: [F3 embedding 32, RGB embedding 16]
boundary:       Linear(48,32) -> SiLU
conditioned:    [boundary 32, query projection 32, edge embedding 8]
boundary trunk: Linear(72,64) -> SiLU -> Linear(64,64) -> SiLU
```

四个 Probe 的禁用证据在进入 boundary encoder 前置零；网络结构、参数数量和初始化完全相同。
Query projection 在四臂中始终保留，因此 B0 是同容量的“无边界证据”对照，不是删层后的
小模型。

### 6.3 输出与零初始化

基础路径和 boundary 路径分别输出逐边 gate/residual logits：

```text
gate_raw = gate_base(base) + gate_boundary(boundary_trunk)
res_raw  = res_base(base)  + res_boundary(boundary_trunk)
g        = sigmoid(gate_raw)
r        = tanh(res_raw)
e'       = e_stock + rho * [w,h,w,h] * g * r
rho      = 0.05
```

四个最终 `Linear(...,1)` 的 weight/bias 全部初始化为零。因此初始 `r=0`，
`refined=stock` 逐值相等；不能通过初始化 gate 偏置制造默认修正。

同一 checkpoint 必须支持：

- `stock`：stock box + stock score；
- `refined`：IBER-BE box + 同一 stock score；
- `boundary_off`：关闭 boundary 输出但保留基础私有路径。

## 7. 匹配与私有损失

只复用最后一层 normal-query stock Hungarian 索引，不进行第二次匹配。损失沿用已验证的
显式监督：

```text
L_private = L_box + 1.0 L_direction + 0.25 L_gate + 0.05 L_noop
L_box     = L1(refined,gt) + GIoU(refined,gt)
```

- matched residual 监督有界 correction direction；
- matched gate 使用 correction magnitude 软标签；
- matched/unmatched gate 独立归一化；
- unmatched 使用 detached stock quality 加权 no-op；
- 不使用 HQ 困难样本加权、Top-K、focal reweight 或新的 AP surrogate；
- 不修改 stock loss，也不向 detector 传播梯度。

## 8. Gate-1：同容量信息量 Probe

先复用已验证、不可变的无增强 evidence-cache 流程，但新 cache 必须增加对应 RGB tensor，
并绑定新的设计版本、源码 commit、runtime amendment 和每个 shard SHA256。
RGB 在 cache 中以 letterbox 后的 contiguous `uint8 CHW` 保存，加载后确定性转换为
`float32/255`；禁止用有损图像格式二次编码。Gate-2 直接使用当前 batch 的 `[0,1]`
RGB tensor，不读取该 cache。

四臂从相同私有初始状态 fresh 训练 12 epoch：

| Probe | F3 boundary | RGB boundary |
|---|---:|---:|
| B0 | zero | zero |
| B1 | yes | zero |
| B2 | zero | yes |
| B3 | yes | yes |

优化器继续使用冻结 detector 私有插件例外：AdamW、lr `1e-3`、weight decay `1e-4`、
betas `(0.9,0.999)`、clip `10`、AMP scale `128`。

B3 必须同时满足原数值门槛：

1. val edge MAE 相对 B0 至少下降 5%；
2. val edge MAE 相对 B1 至少再下降 1.5%，证明高分辨率 RGB 有独立增量；
3. refined matched IoU 相对 stock 至少提高 `0.005`；
4. tiny 和 small correction-direction accuracy 相对 B0 均至少提高 3 个百分点；
5. B3 在四臂中同时取得最佳 edge MAE 与 matched IoU；
6. gate/residual/loss/梯度有限、非零且未饱和；
7. 四臂参数量、初始化指纹、epoch 数和 cache authority 完全一致。

任何核心条件失败即写入 `scientific_failed` 并停止，不进入 30 epoch，不降低门槛。

## 9. Gate-2：固定 10% 子集 30 epoch

Gate-1 通过后，B3 从同一私有 seed 的初始状态 fresh 启动，不续训 Probe checkpoint：

```text
train: fixed 647 images
val:   full 548 images
seed:  0
epochs: 30
batch/workers/imgsz: 8/8/640
detector: frozen mature baseline checkpoint
private optimizer: fixed AdamW rule
```

训练每个 batch 使用同一冻结 detector 即时生成 F3、RGB 和 stock 证据；Gate-1 cache 不进入
Gate-2。训练增强保持公共 VisDrone 参数，`mosaic=1.0`、`close_mosaic=10` 不得关闭。

每 epoch 使用同一 checkpoint 分别评估 stock/refined，并先发布 GitHub 证据再允许下一轮。
第 30 轮必须同时满足：

1. `delta mAP50-95 >= +0.0020`；
2. `delta AP75 >= +0.0030`；
3. `delta AP50 >= -0.0005`；
4. AP-tiny 或 AP-small 至少一项为正；
5. matched IoU 改善数量大于恶化数量；
6. unmatched correction RMS 不超过 matched correction RMS 的 25%；
7. gate/residual 非零、未坍缩、未饱和；
8. 三次独立评估逐值一致；
9. 最后 5 epoch refined mAP 均值高于同 checkpoint stock 均值。

不得用最佳 epoch 替代 epoch30 冻结判定。该 30 epoch 只决定是否值得进入全数据正式实验，
不会自动启动 100 epoch 或多 seed。

## 10. GitHub、恢复与不可变证据

IBER-BE 使用新的源码分支、结果分支、run ID、release tag 和 asset prefix。不得复用或覆盖
LPR、LPR-G、I-TBER 的结果目录。

Gate-2 每个已完成 epoch 必须发布：

- checkpoint 与 SHA256 manifest；
- train/private losses；
- stock/refined AP、AP50、AP75、AP-tiny、AP-small；
- edge MAE、matched IoU、改善/恶化计数；
- F3/RGB boundary embedding RMS；
- gate/residual 与 matched/unmatched correction RMS；
- detector、baseline、dataset、cache、源码和 runtime amendment 哈希；
- optimizer、AMP scaler、RNG、completed epoch；
- 结果分支 commit 和远端验证回执。

只有远端 checkpoint/manifest 和结果 commit 均验证成功，epoch 才算完成。恢复只从最高连续、
已验证 epoch 继续；未发布完成的 epoch 必须重跑。

## 11. 工程预算

- 参数量增幅目标 `<1%`；
- GFLOPs 增幅目标 `<1%`；
- 同机 FP16 端到端延迟增幅目标 `<3%`；
- 超出开销目标必须如实报告，但科学 Gate 仍由定位与 AP 指标决定；
- RGB 稀疏采样不得退化为全图高分辨率卷积或多次 detector forward。

## 12. 测试与工程 Gate-0

实现前先写失败测试，至少覆盖：

1. trajectory 字段、参数和计算完全不存在；
2. RGB/F3 采样位置、方向、半径和 `align_corners=False` 逐值正确；
3. B0–B3 参数量和初始化指纹一致；
4. 四个输出头零初始化时 stock/refined 逐值相等，包括轻微越界 stock box；
5. stock detector 参数梯度始终为 `None`，SHA256 不变；
6. RGB/F3 两条 boundary 路径在启用时有有限非零梯度；
7. matched/unmatched loss 归一化、空匹配和极小框有限；
8. cache shard、runtime amendment、checkpoint 和 resume authority 拒绝漂移；
9. 30 epoch、固定 647 子集、统一增强和每 epoch GitHub 发布不可由 CLI 改写；
10. stock/refined 三次独立评估和参数/GFLOPs/延迟脚本可执行。

Gate-0 失败只修工程问题；Gate-1/Gate-2 科学失败不得改阈值或绕过监督器。

## 13. 概率边界与止损

基于 v1.1 的真实边界证据结果，本设计不能承诺 60% 成功率。当前主观区间为：

| 目标 | 概率区间 |
|---|---:|
| Gate-1 通过 | 30%–40% |
| Gate-2 refined 超过 stock | 25%–35% |
| Gate-2 达到 `+0.20 pp` | 15%–25% |
| 后续全数据达到 `+0.30 pp` | 10%–20% |

该区间高于继续使用 trajectory、HQ loss reweight 或重复 stride-8 F3-only fusion，原因是
它引入了此前缺失的真实高分辨率稀疏证据，同时保持 detector 隔离。

失败处理：

- Gate-1 失败：终止 IBER-BE，不实现 30 epoch AP 筛选；
- B2 明显优于 B1、B3 不优于 B2：正式候选简化为 RGB-only，但必须另立设计版本；
- B1/B2/B3 均不优于 B0：判定 boundary-only 插件路线失败；
- Gate-2 失败：只做坐标、匹配、gate/no-op 和采样诊断，不添加 P2、trajectory、Query、
  attention、分类重打分或多尺度融合；
- Gate-2 通过：冻结结构和超参数，先提交完整比较报告，再由用户决定全数据正式预算。
