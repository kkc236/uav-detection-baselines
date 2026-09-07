# VisDrone LRS v2 进展记录（现有初始状态）

## 当前已完成

此前完成了一个旧 v2 G 运行：`LRS-FDR + v2 geometry + AC-BPDD`，seed 0，100 epochs。

| 指标 | G best（epoch 92） | G final（epoch 100） | 历史 LRS-FDR best |
|---|---:|---:|---:|
| mAP50-95 | 0.29628 | 0.29510 | 0.29703 |
| mAP50 | 0.49368 | 0.49412 | 0.49188 |
| Precision | 0.58538 | 0.58127 | 0.57860 |
| Recall | 0.50028 | 0.50395 | 0.50141 |

G best 相对历史 best 的 mAP50-95 差值为 `-0.075 pp`。运行时 `nonfinite_geometry_observations=0`，但该结果不是新服务器四臂 fresh paired 主表。

## 未完成项

- F：旧 v2 运行仅完成 44 轮，不能作为正式 100 轮基线。
- G：已有结果只作进展参考；按本交接包的统一 authority 仍应重新 fresh run。
- H：未运行。
- I：未运行。

## 交接边界

当前 G 的 checkpoint 不用于初始化其他臂。新服务器的 f/g/h/i 都必须使用 `INITIAL_STATE_MANIFEST.json` 中的同一初始状态 SHA，并从 epoch 1 独立开始。
