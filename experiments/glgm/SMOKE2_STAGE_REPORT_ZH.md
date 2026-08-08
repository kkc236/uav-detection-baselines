# GLGM Smoke2 阶段报告

## 1. 结论

`smoke2-seed0-v2` 已完成并通过工程门检，可以进入全数据 Screen30。该实验使用 5% 训练数据、2 epochs，并行运行在两块 RTX 4090 上，因此仅验证训练、评估、延迟测试和审计链路，不用于判断 GLGM 的最终精度收益。

完成标记为 `EXPLORATORY_COMPLETED`，比较报告明确记录 `strict_pair=false`；正式结论必须来自同一物理 GPU 顺序运行的严格实验。

## 2. 固定输入

- 数据集：VisDrone2019-DET，train 6471 张、val 548 张。
- 有效标注：train 343204 个、val 38759 个。
- 数据规则：跳过官方忽略标注，并过滤、记录 1 个零高度训练框。
- 输入尺寸：640 x 640。
- 基础权重：RT-DETR-X，SHA-256 `DCFF6D3AEABA176924EB9E664E5B0880597E0828C109F3F8AAE7EE5A07546D5C`。
- 公共初始化状态 SHA-256：`325E80C7FA9826028169F1D99071C09DA1C900FBABB029CBE43B675C151F6BE3`。
- Control：RT-DETR-X + 参数对齐的 GLGM control block。
- 实验组：RT-DETR-X + GLGM。

## 3. 完整性结果

- Control 与 GLGM 均完成 2/2 epochs，`results.csv` 均为 2 行且全部为有限值。
- best/last 检查点均按角色、绝对路径和 SHA-256 绑定。
- 17 个发布产物通过 `SHA256SUMS.txt` 复核。
- 数据清单在初始审计、两臂训练前和统一评估前保持一致。
- 独立日志扫描未发现 Traceback、ERROR、Exception、NaN、Inf、OOM、fatal、killed 或 failed。
- 已知 CUDA 非确定性警告已保留；Smoke2 的探索性标记没有将其包装成严格配对证据。

## 4. last.pt 统一评估

| 指标 | Control | GLGM | GLGM - Control |
|---|---:|---:|---:|
| Precision | 0.853026 | 0.834164 | -0.018862 |
| Recall | 0.070627 | 0.064731 | -0.005896 |
| F1 | 0.051422 | 0.033411 | -0.018011 |
| AP50 | 0.041278 | 0.025630 | -0.015648 |
| AP75 | 0.025372 | 0.016077 | -0.009296 |
| mAP50-95 | 0.024441 | 0.015223 | -0.009217 |
| 推理延迟（ms/image） | 17.8961 | 18.5434 | +0.6473 |

GLGM 参数量增加 6,639,079，增幅 9.861%。以上精度差值来自极短、低数据比例的流程测试，不能作为模块有效或失败的结论。

## 5. 下一阶段

`screen30-seed0-v1` 使用完整训练集、30 epochs，并设置：

```text
STRICT_PAIR=1
PARALLEL=0
CONTROL_DEVICE=0
GLGM_DEVICE=0
EVAL_DEVICE=0
```

Control 与 GLGM 必须从共同配对初态开始，在同一块 GPU 上顺序训练。任一 arm 中断后不得单臂恢复，必须保留失败目录并在新目录重跑整对实验。

