# I-TBER v1.1 运行时驱动修订

日期：2026-08-01

状态：方案 A 已获用户批准，等待书面规格复核

适用设计：`itber-v1.1`

适用服务器：NVIDIA GeForce RTX 4090，实际驱动 `570.133.07`

## 1. 决策

不把服务器驱动从 `570.133.07` 降级到 baseline 历史环境的 `550.142`。本次 I-TBER 执行采用当前驱动，并把差异作为显式、不可变的运行环境修订记录；不得把当前环境描述成与 baseline 历史环境逐字段一致。

选择该方案的原因是：I-TBER 的主效应来自同一服务器、同一私有 checkpoint、同一冻结 detector 下的 `stock` 与 `refined` 配对比较。两路推理共享相同驱动和 CUDA 内核，驱动差异不会成为方法分支之间的混杂因素。强制降级需要使用 NVIDIA `.run` 安装器并重启，工程风险高于其对配对 AP 比较的预期收益。

## 2. 两套环境身份

baseline 历史环境永久保留为：

```text
GPU: NVIDIA GeForce RTX 4090, 24GB
driver: 550.142
Python: 3.10.12
PyTorch: 2.5.1+cu121
Torchvision: 0.20.1+cu121
CUDA: 12.1
Ultralytics: 8.4.90
```

本次执行环境锁定为：

```text
GPU name: NVIDIA GeForce RTX 4090
reported memory: 49140 MiB
driver: 570.133.07
Python: 3.10.12
PyTorch: 2.5.1+cu121
Torchvision: 0.20.1+cu121
CUDA: 12.1
Ultralytics: 8.4.90
```

除驱动和云主机上报显存外，其余环境字段必须与 baseline 历史环境一致。显存差异不得用于提高 batch、workers、imgsz、query 数或任何训练容量。

## 3. 不变的科学权威

以下权威不因本修订改变：

- baseline checkpoint SHA256：`54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B`；
- 数据集 SHA256：`FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`；
- 固定 647 张子集 SHA256：`52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`；
- 类别映射、Ultralytics 源文件和代码 commit；
- seed0、imgsz640、batch8、workers8、AMP128、deterministic、cache=False；
- 全部数据增强、query=300、max_det=300、NMS=False；
- I-TBER 私有优化器、Gate 顺序、阈值、screen12 和 formal30；
- 冻结 detector、同 checkpoint stock/refined、逐 epoch 评估与 GitHub 保护规则。

baseline 的原始 MuSGD、100 epoch 和从零训练事实继续记录在 `BASELINE_TRAINING_CONTRACT` 中。I-TBER 的 AdamW 仍只更新隔离私有头，不得反向更新 baseline。

## 4. Gate 0 行为

Gate 0 必须同时保存：

1. `baseline_reference_environment`：含历史驱动 `550.142`；
2. `execution_environment`：含实际驱动 `570.133.07` 和显存上报；
3. `runtime_amendment`：固定标识、批准日期、允许的唯一驱动差异和配对比较理由；
4. baseline、数据、子集、类别和源代码的实际 SHA；
5. stock 包装一致性、冻结性、梯度和数值 Canary。

当执行驱动精确为 `570.133.07` 且其余权威通过时，Gate 0 可以返回 `passed_with_runtime_amendment`，流水线将其视为允许继续的通过状态。以下情况仍为 `engineering_invalid`：

- 驱动不是 `570.133.07`；
- GPU 名称、Python、PyTorch、Torchvision、CUDA 或 Ultralytics 漂移；
- baseline/data/subset/category/source 任一权威漂移；
- 修订字段缺失或发生变化；
- 任一 Canary 失败。

不得将普通 `passed` 与 `passed_with_runtime_amendment` 混为同一状态。

## 5. baseline 复评与比较口径

进入私有训练前，必须在当前执行环境重新评估冻结 stock baseline，并保存完整验证配置、环境、指标、日志和 SHA。历史约 `0.24170/0.241803` 只作为参考，不作为本次 I-TBER 的直接对照值。

所有 screen/formal 结论只使用同一私有 checkpoint 的：

```text
delta = refined(current environment) - stock(current environment)
```

不得使用历史 550.142 环境的 baseline 指标与当前 refined 指标相减。

## 6. 效率结果口径

参数量和 GFLOPs 与驱动无关，继续使用预注册阈值 `<1%`。端到端延迟可能受驱动影响，因此延迟报告必须标注 `570.133.07`、实际 GPU 名称、显存上报、输入尺寸、FP16、warmup 和迭代数；其结论只适用于当前执行环境。stock/control/refined 仍按轮换顺序在同一进程中测量，阈值保持 `<3%`，不因驱动修订放宽。

## 7. 实现与测试边界

实现只允许：

- 将 baseline 历史环境与本次执行环境拆分为两个不可变常量；
- 为 Gate 0 增加结构化 runtime amendment；
- 允许流水线识别 `passed_with_runtime_amendment`；
- 在评估、benchmark、checkpoint 和 GitHub 轻量结果中写入两套环境身份及修订哈希；
- 更新部署验证脚本和操作文档。

实现不得改变模型、loss、数据读取、增强、优化器、学习率、随机数、epoch、阈值、checkpoint/resume 或发布行为。

测试必须先证明旧代码会因 `570.133.07` 拒绝，再证明：只有精确批准的驱动差异可以通过；任意第二个环境漂移仍被拒绝；修订状态和哈希进入全部要求的产物；普通未修订环境仍保持原有严格验证语义。

## 8. 启动条件

完成代码修订后必须重新执行：

- 本地全部 I-TBER 测试；
- 本地完整测试套件；
- 服务器全部 I-TBER 测试；
- 服务器 authority 审计；
- Gate 0 Canary；
- 当前环境 stock baseline 复评。

只有 Gate 0 为 `passed_with_runtime_amendment` 且 stock 复评完整，才允许生成 evidence cache 和启动 P0-P3。任何失败继续停止流水线，不得手工跳过。
