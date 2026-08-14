# PR-IRA：面向 FDR+BPDD 的受保护局部残差适配算法

日期：2026-08-14

状态：候选算法已冻结，尚未实现与训练，不得写成已取得正收益

目标模型：`Ultralytics RT-DETR-L + FDR + BPDD + PR-IRA`

## 1. 结论与成功率

PR-IRA 是当前“挽救 IRA 并使其与 BPDD 共存”路线中成功率最高的设计，但不是高确定性方案。它针对原 IRA 的两个已验证失败原因做了结构性修复：

1. 原 IRA 会用较大的全局残差比例压低整张 P3，破坏成熟特征；
2. BPDD 的辅助目标会直接更新 IRA 私有参数，造成表征增强与分布蒸馏之间的优化耦合。

基于现有 100 epoch 证据、历史失败模式和当前代码路径，对成功率作如下工程判断。区间是证据驱动的主观概率，不是统计置信区间。

| 目标 | 当前估计 | 含义 |
|---|---:|---|
| P0–P4 工程验收全部通过 | **85%–92%** | 模块、初始化、AMP、累积梯度和防火墙能按数学定义运行 |
| Screen30 相对 FDR+BPDD 的 mAP 严格正向 | **45%–55%** | 只要求 mAP 最终值正向，不代表完整门检通过 |
| Screen30 通过全部预注册门检 | **35%–45%** | 同时要求 mAP/AP75、尾三轮、AP50、Precision 和尺度指标合格 |
| PR-IRA 单独相对 FDR 通过独立贡献筛选 | **35%–45%** | 能作为第三个独立创新点，而非 BPDD 附件 |
| 两个 Screen30 均通过 | **22%–32%** | 完整模型正向且第三模块归因成立；两项结果并非独立事件 |
| 已通过两个 Screen30 后，Formal100 仍超过 FDR+BPDD | **55%–65%（条件概率）** | 早期门检通过后仍存在后期收益收缩风险 |
| 从当前阶段直接算，最终成为论文可用第三创新点 | **15%–25%** | 包含工程、两次 Screen30、Formal100 与独立评估全链路 |
| Formal100 相对 FDR+BPDD 提升至少 `+0.3 pp` mAP | **10%–18%** | 属于较有说服力的小模块收益 |
| Formal100 相对 FDR+BPDD 提升至少 `+0.5 pp` mAP | **4%–9%** | 不应作为第一版的合理预期 |

最可能结果是：PR-IRA 消除原 IRA 的大部分负作用，使完整模型回到 FDR+BPDD 附近，并取得 `0～+0.2 pp` 的轻量增益；最需要防范的结果是“Recall 继续增加，但 Precision/AP50 再次下降”。

因此建议执行，但必须先做 P0–P4 和严格 Screen30，不应未经门检直接投入 Formal100。

## 2. 已有证据与问题定义

### 2.1 成熟基础

| 方法 | Precision | Recall | AP50 | AP75 | mAP50–95 | 证据边界 |
|---|---:|---:|---:|---:|---:|---|
| FDR | 0.56911 | 0.49278 | 0.48468 | 0.29253 | 0.28966 | 历史正式权威 |
| FDR+BPDD | 0.57063 | 0.49446 | 0.48641 | 0.29810 | 0.29226 | Formal100 跨权威初步比较 |
| FDR+BPDD+原 IRA | 0.55576 | 0.49834 | 0.48124 | 0.29510 | 0.28995 | exact epoch100 EMA 独立评估 |

原 IRA 相对 FDR+BPDD：

- Precision：`-1.487 pp`；
- Recall：`+0.388 pp`；
- AP50：`-0.517 pp`；
- AP75：`-0.299 pp`；
- mAP50–95：`-0.231 pp`。

这不是工程故障。100/100 epoch 均完成且梯度有限，说明原 IRA 确实改变了学习结果，但改变方向不利。最终 `residual_scale=0.43547`，说明全局修改强度过大。

### 2.2 目标问题

无人机图像中的 tiny/small 目标需要 P3 局部纹理和边缘，但 P3 同时包含大量道路、建筑和树冠背景。直接增强整张高分辨率特征会扩大背景响应。BPDD 又利用 Decoder 多层边界分布进行训练期蒸馏；如果其梯度直接进入特征增强模块，局部表征会同时服从主检测目标和跨层教师目标，容易破坏已由 FDR+BPDD 形成的稳定解。

PR-IRA 的目标不是“让特征更强”，而是：

> 在完整保留 stock P3 的前提下，只注入有界、可定位的局部增量，并阻断 BPDD 对该增量生成器私有参数的直接优化。

## 3. 模块位置与职责

```text
stock P3 ────────────────┬─────────────────────────────┐
                         │                             │
                         └─ PR-IRA 局部增量分支 ──────┤
                                                       ↓
                                                  enhanced P3
stock P4 ──────────────────────────────────────────────┤
stock P5 ──────────────────────────────────────────────┤
                                                       ↓
                                                 FDR Decoder
                                                       ↓
                                 stock + FGL + pre-box + BPDD
```

- FDR：负责四边细粒度分布回归；
- BPDD：负责训练期 Decoder 渐进分布蒸馏，不增加推理参数；
- PR-IRA：只在 P3 产生受保护的局部表征增量；
- Query、分类头、Hungarian 匹配、P4/P5、NMS 和后处理不改变。

YAML 层定义：

```yaml
- [21, 1, PRIRA, [256, 0.20]]
```

删除该层并恢复 Decoder 的 P3 索引即可回到 `FDR+BPDD`，保证消融可拔插。

## 4. PR-IRA 数学定义

### 4.1 局部候选残差

给定 stock P3 特征 \(x\in\mathbb{R}^{B\times C\times H\times W}\)，两级深度可分离局部块产生候选特征：

\[
d_{raw}=R_{local}(x)-x.
\]

局部分支只负责提出增量方向，不允许直接替换主干特征。

### 4.2 残差尺度标定

对每个样本、每个通道在空间维计算 RMS：

\[
d=\frac{d_{raw}}
{\operatorname{RMS}_{HW}(d_{raw})+\epsilon}
\operatorname{stopgrad}\left(\operatorname{RMS}_{HW}(x)\right),
\qquad \epsilon=10^{-6}.
\]

该步骤保留候选残差方向，同时防止局部分支通过无限放大数值绕过外层幅度上限。

### 4.3 通道—空间局部门控

门控只观察候选残差的绝对响应：

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

通道门与空间门的末层 weight 和 bias 均初始化为 0，保证初始门值分别严格为 0.5、联合门值严格为 0.25。该零化在私有 seed 初始化完成后执行，避免被通用初始化函数覆盖。

### 4.4 受保护输出

\[
y=x+\rho(t)\alpha_{max}\tanh(a)G\odot d,
\qquad \alpha_{max}=0.20.
\]

- \(a\) 为可学习标量，初始化为 0；
- \(\rho(t)\) 为训练进度调度；
- stock P3 的恒等路径始终完整保留；
- 任意时刻的外层幅度不超过 0.20；
- 初始化时 \(y=x\) 逐元素严格成立。

## 5. 相对进度调度

令 \(e\) 为从 1 开始的当前 epoch，\(E\) 为总 epoch 数，并定义两个整数里程碑：

\[
e_0=\lceil0.10E\rceil,\qquad e_1=\lfloor0.30E\rfloor+1.
\]

\[
\rho(t)=
\begin{cases}
0,&e\le e_0,\\
\frac{e-e_0}{e_1-e_0},&e_0<e<e_1,\\
1,&e\ge e_1.
\end{cases}
\]

显式整数边界用于消除连续比例在第 9/10 轮和第 30/31 轮处的歧义。

| 实验 | 恒等期 | 线性开放期 | 完全开放期 |
|---|---:|---:|---:|
| Screen30 | epoch 1–3 | epoch 4–9 | epoch 10–30 |
| Formal100 | epoch 1–10 | epoch 11–30 | epoch 31–100 |

恒等期内 PR-IRA 私有参数的 `.grad` 设为 `None`，同时阻止梯度、动量和 weight decay 造成隐式漂移。开放后使用公共学习率的 `0.1×`，优化器仍为同一个 MuSGD。

## 6. BPDD 私有梯度防火墙

定义：

\[
L_{main}=L_{stock}+L_{FGL}+L_{pre},
\qquad
L=L_{main}+L_{BPDD}.
\]

对共享与 FDR 参数 \(\theta_o\)：

\[
\nabla_{\theta_o}L=\nabla_{\theta_o}(L_{main}+L_{BPDD}).
\]

对 PR-IRA 私有参数 \(\theta_i\)：

\[
\nabla_{\theta_i}L=\nabla_{\theta_i}L_{main}.
\]

防火墙只隔离 BPDD 对 PR-IRA 私有参数的直接梯度。BPDD 仍会更新共享 Backbone/Neck 和 FDR Decoder，因此不能声称“完全切断 BPDD 对 P3 的所有间接影响”。

### 算法 1：单前向、单优化步的私有梯度防火墙

```text
输入：micro-batch 序列，模型参数 θo/θi，固定 AMP scale=128
状态：FP32 firewall_buffer，与 θi 一一对应，初始为零

for each micro-batch do
    单次前向，得到 Lmain 与 LBPDD
    gb ← autograd.grad(LBPDD, θi, retain_graph=True, allow_unused=True)
    firewall_buffer ← firewall_buffer + FP32(gb)
    scaler.scale(Lmain + LBPDD).backward()

    if 到达 optimizer accumulation 边界 then
        scaler.unscale_(optimizer)
        grad(θi) ← grad(θi) - firewall_buffer
        使用原版全模型 clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(MuSGD)
        scaler.update()
        optimizer.zero_grad()
        firewall_buffer ← 0
        更新 EMA
    end if
end for
```

关键顺序必须是：`BPDD 私有梯度采集 → 总损失 backward → AMP unscale → 防火墙扣除 → 原版全模型 clip → step`。不得对 `loss_bpdd` 整体 detach，也不得将原版全模型裁剪改成分组裁剪。

## 7. 参数与初始化协议

| 项目 | 固定值 |
|---|---|
| 插入位置 | P3-only |
| 通道数 | 256 |
| 局部块 | 2 个深度可分离残差块 |
| `alpha_max` | 0.20 |
| `epsilon` | `1e-6` |
| PR-IRA 私有 seed | `20000 + experiment_seed` |
| PR-IRA 学习率 | 公共参数学习率的 `0.1×` |
| 优化器 | 同一 MuSGD |
| AMP | True，固定 scale 128 |
| 梯度裁剪 | 原版全模型 max norm 10.0 |

构造 PR-IRA 时必须使用私有 RNG 域，不得消耗公共 CPU/CUDA RNG。FDR、BPDD 和所有共享参数必须与对照臂 bit-exact 初始化一致。

## 8. 工程门检

### P0：模块数学

- BCHW 输入输出同形；
- `a=0` 时输出 bit-exact 等于输入；
- 门值、调度和全部中间量有限；
- 门值位于 `[0,1]`；
- 有效残差 RMS 不超过 `alpha_max + 1e-5`；
- 非法通道、dtype、device 和空张量 fail closed。

### P1：初始化隔离

- 与 FDR+BPDD 共享的所有状态 bit-exact 相同；
- 只有 PR-IRA 层出现新状态；
- 模块构造不改变公共 RNG；
- 初始推理输出 bit-exact 相同。

### P2：防火墙等价

在同一真实 batch 上证明：

- PR-IRA 梯度：`firewall == main-only`；
- 非 PR-IRA 梯度：`firewall == main+BPDD`；
- FDR 分布头保留非零 BPDD 梯度；
- 8 个 micro-batch 累积后关系仍成立；
- 防火墙发生在 unscale 后、原版全模型 clip 前；
- step 后 buffer 为空。

### P3：真实 CUDA 单步

VisDrone、batch8、MuSGD、AMP128 下执行真实 forward/backward/step/save/reload，所有损失、梯度、门控和 checkpoint 状态有限且可恢复。

### P4：开销

报告参数、GFLOPs、训练显存/速度和 RTX 4090 FP16 推理延迟。BPDD 仍为推理零开销；PR-IRA 是唯一新增推理图。

## 9. 科学验证与停止规则

### 9.1 兼容性 Screen30

```text
Control: FDR + BPDD
Method : FDR + BPDD + PR-IRA
```

固定 10% 子集、seed0、相同共享/FDR 初始状态、样本顺序、增强随机序列和全部训练参数。两臂 fresh 训练 30 epoch。

通过要求：

- final 与 tail3 mAP 均严格正向；
- final 与 tail3 AP75 均严格正向；
- final AP50 降幅不超过 `0.0005`；
- final Precision 降幅不超过 `0.0020`；
- tiny 或 small mAP 至少一项严格正向；
- 梯度有限，幅度不长期饱和在 `±0.20`。

### 9.2 独立贡献 Screen30

兼容性通过后再运行：

```text
Control: FDR
Method : FDR + PR-IRA
```

只有该配对也通过，PR-IRA 才能作为独立第三创新点；否则只能写成 BPDD 兼容附属结构。

### 9.3 Formal100

两个 Screen30 均通过后，从统一 formal initial state fresh 启动全量 seed0 100 epoch，不继承 screen 权重。最终以 exact epoch100 EMA 独立评估，并在论文消融阶段 fresh 重跑同 authority 的 FDR+BPDD 对照。

以下任一情况发生即冻结为科学失败：

- 只增加 Recall，但再次降低 mAP/AP50/Precision；
- 仅某个有利 epoch 正向，final/tail3 不通过；
- 只有完整组合正向，PR-IRA 独立配对不正向；
- 需要打开 test 集或修改预注册阈值才能通过。

## 10. 对抗性风险

1. **追回门槛不低。** 原 IRA 相对 BPDD 已落后 `0.2305 pp` mAP，新算法需先消除负作用再产生净收益。
2. **保护也可能导致欠学习。** `0.1×` 私有学习率、恒等期和 `0.20` 幅度上限共同降低破坏风险，也会压缩上限。
3. **防火墙不是全图隔离。** BPDD 仍会通过共享参数间接改变 P3。
4. **RMS 标定可能放大小噪声。** 外层 gate 与幅度上限只能限制大小，不能保证方向一定有用。
5. **梯度采集增加训练开销。** `autograd.grad` 会提高训练显存和时间，但不增加推理开销。
6. **单 seed 只能做工程与方向筛选。** 最终 seed0 结果可以支持初步论文结论，但不能宣称统计显著。

## 11. 论文贡献边界

可以声明：

> 本文提出受保护的局部残差表征适配器 PR-IRA，通过恒等主路、RMS 残差尺度标定、有界通道—空间门控、相对进度开放和 BPDD 私有梯度防火墙，在保持 FDR 定位路径与 BPDD 训练作用的同时，降低高分辨率局部增强对成熟 P3 表征的破坏。

不得声明：

- 首次提出残差学习、通道/空间注意力、ReZero 或梯度阻断；
- BPDD 与 PR-IRA 已经取得协同增益；
- 防火墙完全切断了 BPDD 对所有 P3 相关参数的影响；
- 在 Screen30/Formal100 实测前写入任何预估增益作为实验结果。

## 12. 当前判定

PR-IRA 值得进入实现和 Screen30，但属于“中等风险、受证据约束的第三创新候选”，不是已经成熟的第三创新点。其成功标准不是恢复原 IRA 的名字，而是同时证明：

1. 相对 FDR+BPDD 有严格正收益；
2. 相对 FDR 有独立正收益；
3. AP50/Precision 不再以 Recall 增长为代价；
4. 所有改变可由独立 YAML 层、防火墙测试和 exact checkpoint 复现。

完整设计规范：`docs/superpowers/specs/2026-08-13-pr-ira-bpdd-gradient-firewall-design.md`。

实施计划：`docs/superpowers/plans/2026-08-14-pr-ira-bpdd-gradient-firewall.md`。
