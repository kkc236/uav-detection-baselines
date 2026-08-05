# FDR YAML 声明式模块与正式权重兼容设计

日期：2026-08-05  
基线：Ultralytics RT-DETR-L 8.4.90  
兼容目标：现有 FDR-only 全数据 seed0 100 epoch 正式权重

## 1. 目标

将当前“先按原生 `rtdetr-l.yaml` 构建，再由 Python 替换 Decoder 框回归路径”的 FDR 实现改为 YAML 显式声明的网络模块。完整模型必须能从 YAML 直接构建，同时严格兼容现有正式权重、保持已验证的计算图和数值行为。

本次只改变模型的声明和构建入口，不改变算法、参数、损失权重、训练协议或正式实验结论。

## 2. 设计原则

1. 权重兼容优先于 YAML 形式上的拆分。
2. 只有具有完整输入、独立计算逻辑和明确输出契约的功能单元才作为独立模块。
3. 无法脱离六层 Decoder 循环而独立工作的操作不强拆为外部串行层。
4. 原生 Decoder 注意力层、FFN、分类头、Encoder、Query、匹配和后处理保持不变。
5. 每个消融配置只改变一个功能开关，完整配置必须复现现有 FDR。

## 3. 模块边界

### 3.1 YAML 显式网络模块：`FDRRTDETRDecoder`

`FDRRTDETRDecoder` 是完整的定位解码功能单元，输入为 P3/P4/P5 三尺度 Encoder 特征和训练目标，输出合同与原生 `RTDETRDecoder` 完全一致。

该模块复用原生 RT-DETR Decoder 的：

- 多尺度特征投影；
- Encoder 输出头；
- Query 初始化和去噪 Query；
- 六层 Deformable Decoder 注意力及 FFN；
- 六层分类头；
- Top-300 和后处理。

该模块只替换框定位路径。

### 3.2 内部独立子模块

以下单元保留独立模块和稳定权重命名：

1. `decoder.pre_bbox_head`
   - 输入：第一层 Decoder hidden state 与初始 reference box。
   - 逻辑：四维连续粗框回归。
   - 输出：preliminary box。
   - 权重路径保持现有正式实现不变。

2. `dec_bbox_head.0 ... dec_bbox_head.5`
   - 输入：对应层 hidden state 与上一层 detach hidden state。
   - 逻辑：每层输出 `4 × (reg_max + 1)` 个四边分布 logits；正式配置为 132 维。
   - 输出：六层分布残差。
   - 权重路径和形状保持现有正式实现不变。

3. `decoder.integral`
   - 输入：累计后的四边分布 logits。
   - 逻辑：softmax 与固定非均匀投影。
   - 输出：四边连续距离。
   - 只有固定 buffer，无可训练参数。

以下操作不能脱离 Decoder 逐层循环，不单独暴露成 Ultralytics 外部网络层：

- 分布 logits 的六层累计；
- reference box detach 更新；
- hidden state detach 累加；
- distance-to-box 变换。

它们作为 `FDRRTDETRDecoder` 的内部计算策略，用 YAML 开关控制消融。

### 3.3 训练期独立组件

训练损失不是推理网络层，单独配置：

- `FGL`：细粒度分布定位监督，正式权重 `0.15`；
- `PreBoxLoss`：preliminary box 的 L1/GIoU 辅助监督；
- 原生 VFL/L1/GIoU 及原匹配结果完整保留。

## 4. YAML 契约

完整模型文件为 `configs/rtdetr-l-fdr.yaml`。最后一层必须显式写为：

```yaml
- [[21, 24, 27], 1, FDRRTDETRDecoder, [nc, [256, 256, 256], {hidden_dim: 256, num_queries: 300, num_decoder_layers: 6, reg_max: 32, reg_scale: 4.0, up: 0.5, cumulative: true, preliminary_box: true, private_seed: 10000}]]
```

训练损失配置放在同一模型 YAML 的 `fdr_loss` 字段：

```yaml
fdr_loss:
  fgl_weight: 0.15
  supervise_pre_boxes: true
```

模型类读取该 YAML 构造损失，不再依赖写死的 FDR 超参数。

## 5. 权重兼容要求

完整 YAML 模型与当前 Python 注入式正式模型必须满足：

1. `state_dict` 键集合完全相等；
2. 每个键的张量形状完全相等；
3. 正式 100 epoch checkpoint 可严格加载，不允许静默缺键或多键；
4. 加载同一权重后，相同输入的原始输出和后处理输出在既定浮点容差内一致；
5. checkpoint 中仍保留原 `model.*` 层号，不移动 Backbone/Encoder/Head 索引；
6. 不修改已安装的 Ultralytics 源码。

## 6. 消融配置

提供以下单变量 YAML：

- `rtdetr-l-fdr.yaml`：完整 FDR；
- `rtdetr-l-fdr-no-fgl.yaml`：只将 `fgl_weight` 设为 0；
- `rtdetr-l-fdr-no-prebox-loss.yaml`：只关闭 pre-box 辅助监督；
- `rtdetr-l-fdr-no-cumulative.yaml`：只关闭跨层分布累计；
- `rtdetr-l-fdr-no-prebox.yaml`：只关闭 preliminary box 作为分布 reference，模块参数保留以兼容 checkpoint。

原生连续回归 baseline 继续使用 Ultralytics `rtdetr-l.yaml`，不伪装成 FDR 消融，也不要求与 132 维分布头权重互载。

## 7. 构建和加载流程

1. 注册仓库自有 `FDRRTDETRDecoder`，不编辑 site-packages。
2. `FDRRTDETRDetectionModel` 直接用 `configs/rtdetr-l-fdr.yaml` 构建完整网络。
3. 删除构建完成后的 head/decoder 替换逻辑。
4. Trainer 的默认 FDR cfg 改为完整 FDR YAML；Control 仍使用原生 `rtdetr-l.yaml`。
5. resume 和正式 checkpoint 通过兼容加载器进行严格键/形状审计。

## 8. 验证

按 TDD 执行：

1. YAML 中确实存在 `FDRRTDETRDecoder`，且能被解析器直接实例化；
2. 新旧构建方式的 `state_dict` 键和形状完全相等；
3. 正式 checkpoint 严格加载成功；
4. 同权重、同输入的训练输出和推理输出一致；
5. FGL 和 pre-box loss 从 YAML 生效；
6. 五份 YAML 各自只改变预期单变量；
7. stock RT-DETR control 构建不受影响；
8. 运行现有 FDR authority、math、head、loss、model 和 protocol 测试集。

## 9. 非目标

- 不重训 100 epoch FDR；
- 不改变正式 FDR 算法；
- 不引入 boundary、trajectory、OAR 或 LPR；
- 不为了模块数量而拆开强耦合的 Decoder 循环；
- 不修改既有实验阈值和结果。
