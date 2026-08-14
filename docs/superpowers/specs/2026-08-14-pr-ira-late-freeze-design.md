# PR-IRA 预注册后期冻结设计

## 1. 目标

在不改变 PR-IRA 前向结构、FDR、BPDD、数据协议、优化器公共参数和推理图的前提下，为 Formal100 增加一个固定的后期私有参数冻结阶段，降低浅层特征适配器在训练后期持续漂移并破坏 Precision、AP50 和 mAP 的风险。

## 2. 证据与根因

旧 IRA 在 FDR+BPDD 上使 Recall 增加约 `0.388 pp`，但 Precision、AP50、AP75 和 mAP 分别下降约 `1.487 pp`、`0.517 pp`、`0.299 pp` 和 `0.231 pp`。FrequencyCM、SCADS、GLGM 和旧 IRA 还反复出现前期正向、后期持平或转负的共同轨迹。

当前 PR-IRA 已经通过受保护的 P3 局部残差、RMS 幅度上限、恒等初始化、渐进开启、私有小学习率和 BPDD 私有梯度防火墙解决了直接抑制 stock P3、残差过强和辅助损失冲突。剩余最直接的风险是：公共 FDR/BPDD 网络接近收敛后，PR-IRA 私有卷积和门控仍受检测梯度、动量和权重衰减驱动，导致已经形成的局部增量继续漂移。

## 3. 选定方案

只增加预注册的 Formal100 后期冻结，不增加前景标签、尺度标签、辅助损失、动态验证集选择或新推理分支。

### Screen30

- epoch 1–3：恒等阶段，PR-IRA 私有参数不更新；
- epoch 4–9：线性开启，PR-IRA 私有参数更新；
- epoch 10–30：完全开启，PR-IRA 私有参数更新；
- Screen30 不设置后期冻结，保留完整可学习性筛选能力。

### Formal100

- epoch 1–10：恒等阶段，PR-IRA 私有参数不更新；
- epoch 11–30：线性开启，PR-IRA 私有参数更新；
- epoch 31–60：完全开启，PR-IRA 私有参数更新；
- epoch 61–100：保持完全开启的前向输出，但冻结全部 PR-IRA 私有参数。

冻结轮次固定写入协议并参与协议 SHA256。它不能根据验证集曲线、最佳 checkpoint 或最终结果动态改变。

## 4. 训练行为

冻结采用“optimizer step 前将 PR-IRA 私有梯度设为 `None`”的现有身份阶段机制扩展实现：

1. 仍然完成 AMP unscale；
2. 仍然执行 BPDD→PR-IRA 私有梯度防火墙减法并验证有限性；
3. 当当前 epoch 不在私有更新窗口内时，将所有 PR-IRA 私有参数的 `.grad` 设为 `None`；
4. 公共参数梯度保持不变；
5. 执行同一个全局 `clip_grad_norm_(..., 10.0)`、MuSGD step 和 EMA 更新。

设置 `.grad=None` 而不是零张量，确保 SGD 不对这些参数执行 momentum 和 weight decay 更新。PR-IRA 参数仍保留在 optimizer 和 checkpoint 中，因此 resume 不需要重建 optimizer，也不改变推理结构。

## 5. 不变项

- PR-IRA 前向公式和 YAML 图不变；
- `alpha_max=0.20` 不变；
- 私有学习率倍率 `0.1` 不变；
- FDR 和 BPDD 数学定义、损失权重及梯度路径不变；
- stock P3、P4、P5、Decoder、Query、分类、匹配和数据增强不变；
- Screen30 和 Formal100 必须 fresh 启动，不能继承旧 IRA 或旧 PR-IRA checkpoint；
- 推理参数量、GFLOPs 和延迟不因冻结机制增加。

## 6. 协议表达

`PR_IRA_PROTOCOL["pr_ira"]["schedule"]` 增加：

```python
"screen30": {
    "epochs": 30,
    "identity": [1, 3],
    "linear_open": [4, 9],
    "fully_open": [10, 30],
    "private_update": [4, 30],
    "private_frozen": [],
},
"formal100": {
    "epochs": 100,
    "identity": [1, 10],
    "linear_open": [11, 30],
    "fully_open": [31, 100],
    "private_update": [11, 60],
    "private_frozen": [61, 100],
},
```

协议模块提供纯函数 `pr_ira_private_update_enabled(epoch, epochs)`，只接受冻结的 30 或 100 epoch 协议，并拒绝越界、布尔值和未知总轮数。

## 7. 测试与验收

必须测试：

1. Screen30 的 epoch 3 关闭、4 和 30 开启；
2. Formal100 的 epoch 10 关闭、11 和 60 开启、61 和 100 关闭；
3. 未知总轮数及非法 epoch fail closed；
4. 身份阶段和冻结阶段均只把私有梯度设为 `None`；
5. 公共梯度在两个阶段都保留；
6. 更新窗口内私有梯度保持不变；
7. 含 momentum 和 weight decay 的真实 SGD step 在冻结阶段不能改变任何私有参数或其 optimizer state；
8. 公共参数在同一步仍正常更新；
9. BPDD 防火墙、AMP、梯度累积、保存和 resume 回归测试全部通过；
10. 全量 PR-IRA/FDR/BPDD 回归测试无失败。

## 8. 科学门检

代码测试只证明工程正确。科学结论仍由冻结 Screen30 门检决定：相对 FDR+BPDD，final 与 tail3 mAP、final 与 tail3 AP75 必须正向，AP50、Precision 和 tiny/small 条件不得违反既定阈值。Formal100 只能在 Screen30 通过后 fresh 启动。

该修正的论文表述限于“预注册的后期稳定化训练策略”，不能依据最终结果反向宣称 epoch 60 是最优点，也不能把冻结策略单独包装成新的第四创新点。
