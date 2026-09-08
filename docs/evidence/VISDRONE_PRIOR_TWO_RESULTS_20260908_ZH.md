# VisDrone 前两项几何校准成果归档

归档日期：2026-09-08。以下两项均为服务器已完成的独立运行；当前正在进行的 LRS-GFDR 新训练不包含在本摘要中。

## 统一协议

- 数据集：VisDrone，train 6471 / val 548
- 初始状态：`initial-state-seed0.pt`
- 初始状态 SHA-256：`935D68428BFC5848DE51F82844658ABBC7EC8C99CBBF6BBF823166D38CB65D5B`
- seed：0；训练预算：100 epoch；batch：8；imgsz：640；单卡 RTX 4090
- AMP 开启，确定性设置开启；未使用 BPDD/FIA

## 结果

| 运行 | 最佳 epoch | best mAP50 | best mAP50-95 | final mAP50 | final mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Clean-FDR + 几何校准 | 91 | 0.49130 | 0.29361 | 0.48921 | 0.29193 |
| LRS-FDR + 几何校准 | 92 | 0.49382 | 0.29620 | 待从原始运行记录补录 | 待从原始运行记录补录 |

## 服务器原始目录

- Clean-FDR + 几何校准：`/root/sj-tmp/runs/visdrone-v2-clean-fdr-geometry-20260908/formal-seed0-clean-fdr-geometry-v1`
- LRS-FDR + 几何校准：`/root/sj-tmp/runs/visdrone-v2-existing-state-20260907/formal-seed0-lrs_fdr_feasible-v2`

## 解释边界

几何校准的首要作用是保证 FDR 解码框的有限性和正宽高，避免第二数据集上的非法框传播或验证崩溃；mAP 差值仅作为描述性结果。原始 `results.csv`、`args.yaml` 和运行时记录仍保存在服务器目录，未上传 `.pt` 权重或数据集。
