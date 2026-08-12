# PR-IRA 与 BPDD 梯度防火墙设计规范

日期：2026-08-13

状态：已批准设计，等待实施计划
目标组合：`RT-DETR-L + FDR + BPDD + PR-IRA`

## 1. 设计目标

在保留 FDR 和 BPDD 既有机制、训练协议与证据边界的前提下，将原 IRA 改造成可与 BPDD 共存的第三个独立创新模块。PR-IRA 必须满足：

1. 作为独立 YAML 网络层插入 P3，能够单独启用或移除；
2. 初始化时与 `FDR+BPDD` 图严格等价；
3. 只向 P3 增加受限局部细节，不允许整体压低或替换 stock P3；
4. BPDD 继续训练 FDR 与共享检测参数，但不得更新 PR-IRA 私有参数；
5. 相对 `FDR+BPDD` 取得可重复的独立正收益，才进入最终三模块模型。

PR-IRA 的中文名称为“受保护的局部残差表征适配器”，英文全称为 **Protected Residual Image Representation Adapter**。

## 2. 已观测失败与根因

原 `FDR+BPDD+IRA` 已完成全量 VisDrone seed0 100 epoch，并使用 exact epoch100 EMA 独立复评：

| 指标 | FDR+BPDD | FDR+BPDD+IRA | 组合变化 |
|---|---:|---:|---:|
| Precision | 0.57063 | 0.55576 | -1.487 pp |
| Recall | 0.49446 | 0.49834 | +0.388 pp |
| AP50 | 0.48641 | 0.48124 | -0.517 pp |
| AP75 | 0.29810 | 0.29510 | -0.299 pp |
| mAP50-95 | 0.29226 | 0.28995 | -0.231 pp |

原 IRA 公式为：

\[
y=x+\alpha\left(R(x)-x\right).
\]

其最终 `residual_scale` 达到 `0.43547`。`R(x)` 的末端是空间与通道 Sigmoid 注意力，因此 `R(x)-x` 在大量背景位置包含明显负项。该结构实际上能够以较大比例压低整张 P3，而不仅是补充局部细节。实验中 Recall 上升而 Precision、AP50 和 mAP 下降，与“候选增多但背景干扰及低质量框增加”一致。

BPDD 又从 IRA 改写后的 FDR Decoder 分布构造跨层教师。BPDD 梯度能够反向更新 IRA 私有参数，使 IRA 同时服务于主检测目标和跨层蒸馏目标。这会扩大两个模块的优化耦合。根因不是网络无法运行，而是：

- P3 修改方式缺少 stock 特征保护；
- 修改幅度过大且全局共享；
- 背景和目标使用同一增强强度；
- BPDD 与 IRA 私有参数之间没有梯度边界。

## 3. 方案选择

本设计采用已批准的第 3 种路线：

> **受限局部增量 PR-IRA + BPDD→PR-IRA 梯度防火墙。**

不采用以下低成功率路线：

- 只调小原 IRA 的全局 `residual_scale`：仍会在背景位置执行 `R(x)-x` 式整体修改；
- 只冻结 IRA 若干 epoch：只能推迟冲突，不能改变最终的错误作用形式；
- 让 PR-IRA 完全不接受任何训练梯度：无法学习任务相关的局部证据；
- 关闭 BPDD：会破坏已经取得正式正收益的创新点二。

## 4. 网络位置与模块边界

PR-IRA 保持原 IRA 的 P3-only 位置，P4/P5 继续绕过该模块：

```text
P3 RepC3 ──┬───────────────────────────────┐
           └─ PR-IRA local residual ───────┤
                                            ↓
                                      Enhanced P3
P4 RepC3 ───────────────────────────────────┤
P5 RepC3 ───────────────────────────────────┤
                                            ↓
                                     FDR Decoder
                                            ↓
                               stock + FGL + BPDD losses
```

职责严格隔离：

- FDR 仍是唯一的边界分布回归路径；
- BPDD 仍是参数为零、仅训练期启用的分布蒸馏损失；
- PR-IRA 只改变送入 Decoder 的 P3 表征；
- Query 数、分类头、Hungarian 匹配、P4/P5、NMS 和后处理均不改变。

YAML 中新增一个完整功能单元：

```yaml
- [21, 1, PRIRA, [256, 0.20]]
```

FDR Decoder 输入保持为 PR-IRA-P3、stock P4 和 stock P5。删除该行并恢复 Decoder 的 P3 索引即可得到 `FDR+BPDD` 消融，不需要编辑 FDR 或 BPDD 内部代码。

## 5. PR-IRA 数学定义

### 5.1 局部残差支路

令 stock P3 为 \(x\)。两级深度可分离局部块只负责产生候选细节：

\[
d_{raw}=R_{local}(x)-x.
\]

与原 IRA 不同，注意力模块不再直接输出 `gated_feature` 并参与 `refined-x`。它只预测门控值。

为避免局部支路通过数值幅度绕过外层保护，对候选残差进行逐样本、逐通道 RMS 标定：

\[
d=\frac{d_{raw}}
{\operatorname{RMS}_{HW}(d_{raw})+\epsilon}
\cdot\operatorname{stopgrad}\left(\operatorname{RMS}_{HW}(x)\right).
\]

该操作保留残差方向，但使其平均通道尺度与 stock P3 可比。`epsilon` 固定为 `1e-6`。

### 5.2 逐位置、逐通道门控

门控从局部残差的绝对响应中产生：

\[
G_c=\sigma\left(MLP(GAP(|d_{raw}|))\right),
\]

\[
G_s=\sigma\left(Conv_{7\times7}
[Mean_c(|d_{raw}|),Max_c(|d_{raw}|)]\right),
\]

\[
G=G_c\odot G_s,\qquad G\in[0,1].
\]

这使平坦背景可以被压低，而局部纹理、边缘及小目标响应可以获得更高门值。第一版不增加目标框掩码监督或 gate sparsity loss，避免把多个变量混入兼容性验证。

通道门和空间门的末层 bias 初始化为 0，初始门值分别为 0.5；两者乘积的初始有效门值为 0.25。全部 PR-IRA 私有参数使用独立 seed `20000 + experiment_seed`，构造过程不得消耗公共 CPU/CUDA RNG。

### 5.3 受保护增量输出

最终输出为：

\[
y=x+\rho(t)\cdot\alpha_{max}\tanh(a)\cdot G\odot d,
\]

其中：

- `a` 是唯一全局可学习幅度原参数，初始化为 0；
- `alpha_max=0.20`，任何时刻绝对幅度不超过 0.20；
- `rho(t)` 是训练进度调度，范围为 `[0,1]`；
- stock P3 的恒等路径始终完整保留；
- PR-IRA 只能增加有界残差，不能用注意力直接压低 stock P3。

初始化时 `tanh(a)=0`，因此逐元素严格满足 `y == x`。

## 6. 启用调度与私有学习率

调度使用训练总轮数的相对比例，以保持 Screen30 与 Formal100 的阶段一致性：

\[
\rho(t)=
\begin{cases}
0,&p<0.10,\\
\frac{p-0.10}{0.20},&0.10\le p<0.30,\\
1,&p\ge0.30,
\end{cases}
\qquad p=\frac{epoch+1}{epochs}.
\]

对应关系：

| 阶段 | 恒等期 | 线性放开期 | 完全可用期 |
|---|---:|---:|---:|
| Screen30 | epoch 1–3 | epoch 4–9 | epoch 10–30 |
| Formal100 | epoch 1–10 | epoch 11–30 | epoch 31–100 |

恒等期内，PR-IRA 所有私有参数的 `.grad` 在优化步前设为 `None`，从而同时阻止梯度、动量和 weight decay 造成隐式漂移。

PR-IRA 私有参数使用公共参数学习率的 `0.1×`，仍采用同一个 MuSGD 优化器、momentum 0.937 和 weight decay 0.0005。FDR、BPDD 与公共参数的优化协议一字不改。

## 7. BPDD→PR-IRA 梯度防火墙

### 7.1 目标

总训练目标保持：

\[
L=L_{stock}+L_{FGL}+L_{pre}+L_{BPDD}.
\]

对非 PR-IRA 参数 \(\theta_o\)：

\[
\nabla_{\theta_o}L
=\nabla_{\theta_o}(L_{main}+L_{BPDD}).
\]

对 PR-IRA 私有参数 \(\theta_i\)：

\[
\nabla_{\theta_i}L
=\nabla_{\theta_i}L_{main}.
\]

因此 BPDD 仍可训练 FDR Decoder 和公共检测网络，但不能改变 PR-IRA 的局部门控与残差支路。

### 7.2 单前向、单优化步实现

不复制整个 Ultralytics 训练循环，也不进行第二次 Detector 前向。每个 micro-batch 执行：

1. 正常前向，得到 `main_loss` 与 `loss_bpdd`；
2. 在总损失 backward 前，调用 `torch.autograd.grad(loss_bpdd, ira_private_params, retain_graph=True, allow_unused=True)`，得到 BPDD 对 PR-IRA 私有参数的未缩放梯度贡献；
3. 将这些贡献以 FP32 累加到当前 optimizer accumulation window 的 firewall buffer；
4. 按原训练器执行 `scaler.scale(main_loss + loss_bpdd).backward()`；
5. optimizer step 时先执行 `scaler.unscale_(optimizer)`；
6. 在任何梯度裁剪之前，从 PR-IRA `.grad` 中减去累计 firewall buffer；
7. 分组裁剪 common、FDR-private、PR-IRA-private 梯度，执行一次 MuSGD step；
8. 清空普通梯度和 firewall buffer，再更新 EMA。

固定协议为单卡，因此不存在 DDP world-size 缩放差异。`nbs=64` 带来的梯度累积必须同时累积 firewall buffer，不能只保存最后一个 micro-batch。

### 7.3 故障保护

- firewall 必须在 AMP unscale 后、clip 前扣除；
- buffer 数量、形状、dtype 和参数身份必须逐项匹配；
- OOM 重建、异常 batch、resume 和 optimizer zero-grad 时必须同步清空 buffer；
- epoch checkpoint 只能在 optimizer step 完成且 buffer 为空时保存；
- 任一私有参数出现非有限 firewall 梯度时立即失败；
- 固定 AMP scale 必须保持 128，不允许静默 skipped step；
- 禁止通过 `loss_bpdd.detach()` 实现隔离，因为这会同时取消 BPDD 对 FDR 的有效梯度。

## 8. 代码与配置单元

实施后应形成以下独立单元：

| 单元 | 职责 |
|---|---|
| `src/pr_ira.py` | PR-IRA 局部残差、RMS 标定、空间/通道门与有界输出 |
| `src/rtdetr_fdr_bpdd_pr_ira.py` | 组合模型、私有初始化、状态映射、调度和梯度防火墙 |
| `configs/rtdetr-l-fdr-bpdd-pr-ira.yaml` | YAML 可插拔完整模型 |
| `tests/test_pr_ira.py` | 模块数学、形状、幅度和恒等性测试 |
| `tests/test_bpdd_pr_ira_firewall.py` | 梯度等价、累积、AMP、裁剪顺序与清理测试 |
| `tests/test_bpdd_pr_ira_integration.py` | YAML、状态、真实模型、checkpoint 与推理契约测试 |

不得修改成熟 FDR 和 BPDD 的数学实现。若需要从 BPDD criterion 暴露 `main_loss` 与 `loss_bpdd`，只能增加兼容接口；原 `FDR+BPDD` 调用与数值结果必须保持不变。

## 9. 工程验收

### P0：模块数学

- BCHW 输入输出形状完全一致；
- `a=0` 时逐元素 bit-exact 恒等；
- `G` 全部有限且位于 `[0,1]`；
- `rho` 全部有限且位于 `[0,1]`；
- 有效残差 RMS 相对 stock P3 不超过 `alpha_max + 1e-5`；
- 空张量、错误通道、错误 dtype/device 失败关闭。

### P1：初始化隔离

- `FDR+BPDD` 和组合模型所有公共/FDR参数 bit-exact 相同；
- 新增状态只能位于 PR-IRA YAML 层；
- 构造 PR-IRA 不改变 CPU/CUDA 公共 RNG；
- 初始推理输出与 `FDR+BPDD` 数值一致。

### P2：梯度防火墙

用同一真实 batch 构造三种梯度：

1. `main-only`；
2. `main+BPDD` 无防火墙；
3. `main+BPDD` 有防火墙。

必须证明：

- PR-IRA 梯度：`firewall == main-only`，容差 `rtol=1e-5, atol=1e-7`；
- 非 PR-IRA 梯度：`firewall == main+BPDD`；
- 至少一个 FDR 分布头的 BPDD 梯度非零，证明 BPDD 没被整体 detach；
- 8 个 micro-batch 累积后仍满足同一等价关系；
- firewall 在 unscale 后、clip 前执行；
- optimizer step 后 buffer 为空。

### P3：真实 CUDA 训练步

- VisDrone train batch8；
- MuSGD、AMP scale128、单 optimizer step；
- stock/FGL/pre-box/BPDD 损失均有限；
- common/FDR/PR-IRA 三组梯度均有限；
- 无 skipped step；
- save/reload 后输出、调度状态和 EMA 一致。

### P4：开销

- 参数增量允许不超过 FDR 的 10%；
- GFLOPs 增量报告真实值，不预设小于 1%；
- FP16 延迟使用相同 RTX 4090、batch1、warmup50、runs200；
- BPDD 仍为推理零开销，PR-IRA 是唯一新增推理图。

## 10. 科学验证流程

### 10.1 严格 Screen30

在固定 647 张 10% 子集上 fresh 配对：

```text
Control: FDR + BPDD
Method : FDR + BPDD + PR-IRA
```

两臂使用相同 shared/FDR 初始参数、样本顺序、增强随机序列、seed0 和统一训练协议。PR-IRA 只有私有参数不同。两臂均从零训练 30 epoch，不继承既有 checkpoint。

预注册通过条件：

- final mAP50-95 严格正向；
- tail3 mean mAP50-95 严格正向；
- final AP75 严格正向；
- tail3 mean AP75 严格正向；
- final AP50 不得低于对照 `0.0005` 以上；
- final Precision 不得低于对照 `0.0020` 以上；
- tiny 或 small mAP 至少一项严格正向；
- 所有梯度有限，PR-IRA 幅度未长期饱和在 `±0.20`。

任何条件失败均冻结为科学失败，不根据 val 结果修改门槛。

### 10.2 创新点三归因筛选

只有兼容性 Screen30 通过后，才补跑同一 source、initial state 和随机序列下的：

```text
Control: FDR
Method : FDR + PR-IRA
```

该配对用于证明 PR-IRA 并非只在 BPDD 存在时才有效。通过条件复用 10.1 的 mAP/AP50/AP75/Precision/尺度条件。若完整组合正向但 `FDR+PR-IRA` 不正向，PR-IRA 不能单独声明为第三个独立贡献，只能写成 BPDD 的兼容附属结构。

### 10.3 Formal100

Screen30 通过后，从相同 formal initial state fresh 启动全量 VisDrone seed0 100 epoch，不继承 screen 权重。完成：

- 逐 epoch create-only checkpoint、指标、梯度、防火墙、门控和 SHA256 证据；
- 每 epoch GitHub 上传并支持精确续跑；
- exact epoch100 EMA 独立 val 复评；
- P/R/F1/AP50/AP75/mAP、四尺度、10 类 AP/AP50/AP75；
- 参数、GFLOPs、FP16 median/P95、FPS、显存；
- 与 `FDR+BPDD` 权威的初步比较。

论文最终消融阶段再 fresh 重跑相同 authority 的 `FDR+BPDD` Formal100。test 集只在所有设计与 checkpoint 选择冻结后评估一次。

## 11. 消融与论文贡献

最终至少保留以下消融：

| 方法 | 用途 |
|---|---|
| RT-DETR-L | stock baseline |
| +FDR | 创新点一 |
| +FDR+BPDD | 创新点二 |
| +FDR+PR-IRA | 创新点三独立贡献 |
| +FDR+BPDD+原IRA | 失败机制对照 |
| +FDR+BPDD+PR-IRA，无 firewall | 证明梯度隔离贡献 |
| +FDR+BPDD+PR-IRA | 完整模型 |

三个创新点的功能边界为：

1. **FDR**：解决连续框回归对小目标边界表达粒度不足；
2. **BPDD**：解决 Decoder 未来层边界知识利用不足及不可靠教师负迁移；
3. **PR-IRA**：解决高分辨率局部细节增强容易破坏成熟 P3 并与分布蒸馏耦合的问题。

PR-IRA 的原创声明必须限制在“受保护的局部增量结构、RMS 幅度约束、相对进度调度与 BPDD 私有梯度防火墙的具体组合”。不得声称首次提出残差注意力、通道/空间门控、ReZero 或梯度阻断的一般概念。

## 12. 成功与停止规则

成功定义：

- PR-IRA 单独相对 FDR 正向；
- 完整模型相对 `FDR+BPDD` 正向；
- 正收益同时出现在 mAP 与 AP75，并且 AP50/Precision 没有明显交换式退化；
- 结构、训练和推理证据可由 YAML、checkpoint 和独立评估复现。

停止规则：

- 若梯度防火墙无法通过 P2 等价测试，不启动训练；
- 若 Screen30 失败，不直接跑 Formal100；
- 若完整模型仅增加 Recall 但再次降低 mAP/AP50/Precision，则判定 PR-IRA 仍与 BPDD 不兼容；
- 不通过降低门槛、挑选单个有利 epoch 或打开 test 集救结果。

## 13. 预期风险

- `0.1×` 私有学习率可能导致 PR-IRA 学习不足；该值在首轮冻结，不进行 val 后调参；
- RMS 标定可能放大小幅噪声，因此分母含 `1e-6` 且外层仍受 gate 与 `alpha_max` 双重限制；
- `autograd.grad` 会增加训练时间和显存，但不增加推理开销；
- 梯度防火墙只隔离 PR-IRA 私有参数，BPDD 仍可通过共享主干更新 stock P3，这是保留 BPDD 原训练作用的有意选择；
- 单 seed Screen30 只能用于筛选，不能证明统计显著性。
