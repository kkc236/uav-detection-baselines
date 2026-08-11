# 纯 RT-DETR-L Control 与 FDR 的严格 Test 配对评估

## 1. 结论

已在同一张 NVIDIA GeForce RTX 4090、同一份 VisDrone test、同一评估器和完全一致的推理参数下，完成纯 Ultralytics RT-DETR-L Control 与 FDR 的 epoch100 EMA 严格配对评估。

FDR 相对纯 Control 的 mAP50-95 提升 **+5.058 pp**，AP50 提升 **+7.613 pp**，AP75 提升 **+5.340 pp**。Precision、Recall、F1、四个尺度分组和 10 个类别的 mAP 全部正向，没有类别退化。

本结果补齐了此前缺失的纯 baseline test 对照，可作为 FDR 消融实验和论文 test 结果的首选严格证据。

## 2. 严格对齐关系

两臂训练均使用同一份冻结协议 `2545F68302639EEB217EC8B53FDB229681B2DAFC91463C61F6B5CEC9B5486302`，公共初始状态 SHA256 为 `51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D`。

| 项目 | 纯 Control | FDR |
|---|---:|---:|
| 基础模型 | Ultralytics RT-DETR-L | Ultralytics RT-DETR-L + FDR |
| Ultralytics | 8.4.90 | 8.4.90 |
| 初始化 | `pretrained=False` | `pretrained=False` |
| epoch | 100 | 100 |
| seed | 0 | 0 |
| 训练图片 / 验证图片 | 6471 / 548 | 6471 / 548 |
| imgsz / batch / workers | 640 / 8 / 8 | 640 / 8 / 8 |
| 优化器 | MuSGD | MuSGD |
| lr0 / lrf | 0.01 / 0.01 | 0.01 / 0.01 |
| momentum / weight decay | 0.937 / 0.0005 | 0.937 / 0.0005 |
| AMP / 固定 scale | True / 128 | True / 128 |
| deterministic / cache | True / False | True / False |
| query / max_det / NMS | 300 / 300 / False | 300 / 300 / False |
| 数据增强、样本顺序和随机序列 | 相同 | 相同 |
| 唯一结构差异 | 原生连续框回归 | FDR 分布式框回归与配套训练监督 |

Test 评估参数也严格一致：`split=test`、`imgsz=640`、`batch=8`、`workers=8`、`conf=0.001`、`max_det=300`、`NMS=False`、`half=False`、`cache=False`，且每个模型只进行一次预测。

## 3. Test 数据权威

| 项目 | 数值 |
|---|---:|
| 图片数 | 1610 |
| 标签文件数 | 1610 |
| 目标实例数 | 75,102 |
| 损坏图片 | 0 |
| 测试归档 SHA256 | `45543DB16745616BB203BAD23532623D30099CA8FF38E502A9B996F5F1A58CFB` |

## 4. 核心指标

| 指标 | 纯 Control | FDR | 绝对提升 |
|---|---:|---:|---:|
| Precision | 0.417978 | 0.503401 | **+8.542 pp** |
| Recall | 0.373013 | 0.431677 | **+5.866 pp** |
| F1 | 0.394218 | 0.464788 | **+7.057 pp** |
| AP50 | 0.322208 | 0.398340 | **+7.613 pp** |
| AP75 | 0.175503 | 0.228903 | **+5.340 pp** |
| mAP50-95 | 0.177416 | 0.228001 | **+5.058 pp** |

该结果说明 FDR 并非依靠降低召回率换取精度：Precision 和 Recall 同时显著上升；AP50 与 AP75 同时上升，也说明收益同时覆盖目标发现和更严格的边界定位。

## 5. 分尺度 mAP50-95

尺度按 640×640 网络输入空间中的目标边长划分：tiny `<16`、small `[16,32)`、medium `[32,96)`、large `>=96`。

| 尺度 | 纯 Control | FDR | 绝对提升 |
|---|---:|---:|---:|
| Tiny | 0.065916 | 0.105919 | **+4.000 pp** |
| Small | 0.186776 | 0.247489 | **+6.071 pp** |
| Medium | 0.281605 | 0.331795 | **+5.019 pp** |
| Large | 0.428914 | 0.445307 | **+1.639 pp** |

FDR 对 small 的提升最大，同时 tiny、medium、large 全部正向。该现象支持“细粒度边界分布对小尺度目标的连续坐标量化和边界不确定性更敏感”的设计动机。

## 6. 逐类别 mAP50-95

| 类别 | 纯 Control | FDR | 绝对提升 |
|---|---:|---:|---:|
| pedestrian | 0.102217 | 0.153998 | **+5.178 pp** |
| people | 0.063075 | 0.100983 | **+3.791 pp** |
| bicycle | 0.040175 | 0.075494 | **+3.532 pp** |
| car | 0.455676 | 0.506470 | **+5.079 pp** |
| van | 0.243826 | 0.290208 | **+4.638 pp** |
| truck | 0.214840 | 0.283554 | **+6.871 pp** |
| tricycle | 0.093960 | 0.145698 | **+5.174 pp** |
| awning-tricycle | 0.087723 | 0.121258 | **+3.353 pp** |
| bus | 0.358868 | 0.428150 | **+6.928 pp** |
| motor | 0.113804 | 0.174197 | **+6.039 pp** |

10/10 类别全部提升，类别一致性强于仅报告全局 mAP 的结果。

## 7. 复杂度

评估器加载后的模型摘要为：

| 指标 | 纯 Control | FDR | 变化 |
|---|---:|---:|---:|
| 参数量 | 32,004,290 | 32,334,278 | +329,988（约 +1.03%） |
| GFLOPs | 103.5 | 103.7 | +0.2（约 +0.19%） |
| 本次 FP32 test 推理 | 4.4 ms/图 | 4.6 ms/图 | +0.2 ms/图 |

这里的速度是本轮 validator 控制台统计，不替代论文中应使用的独立 FP16 warmup/runs 延迟基准。

## 8. Checkpoint 与远端权威

| 资产 | SHA256 | GitHub Release |
|---|---|---|
| Control epoch100 | `9C242711F44B7E68B360AF904AB7C44F64505C7136B7E7F90481092AE3308AF7` | `fdr-formal-control-d97e1eb7-live` |
| FDR epoch100 | `C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2` | `fdr-formal-d97e1eb7-live` |

- Control Release: https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-control-d97e1eb7-live
- FDR Release: https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-d97e1eb7-live

## 9. 与既有截图数值的边界

此前截图中的 FDR test `mAP=0.232` 不是本轮“同评估器、同精确 epoch100 EMA、同数据权威”重新计算得到的数值。本轮严格 FDR 为 `0.228001`。截图可能来自 best checkpoint、不同运行目录或不同验证入口；在其 checkpoint SHA256 和完整参数未追溯前，只能作为历史参考，不应与本轮精确 Control 直接组成主论文配对表。

即使采用更保守的本轮严格值，FDR 对纯 Control 的 mAP 提升仍为 **+5.058 pp**，结论不受影响。

## 10. 元数据更正说明

首版 Control JSON 复用了 BPDD/FDR 通用 checkpoint 加载器。该加载器在传入自定义纯 Stock 模型工厂时仍硬编码写入 `checkpoint.strict_fdr_inference_graph=true`。运行脚本在评估前已强制断言 `type(loaded.model) is RTDETRDetectionModel`，并以 `strict=True` 加载纯 Control EMA，因此指标确实来自纯原生 RT-DETR-L。

原 JSON 保留不覆盖；`strict-control-test-eval-amendment.json` 对该单一元数据字段建立可审计更正。数值、预测和 checkpoint 均未修改，也不需要重跑。

## 11. 证据文件

- `evidence/strict-control-test-d97e1eb7/strict-control-test-eval.json`
- `evidence/strict-control-test-d97e1eb7/strict-control-test-eval-amendment.json`
- `evidence/strict-control-test-d97e1eb7/strict-fdr-test-eval.json`
- `evidence/strict-control-test-d97e1eb7/test-authority.json`
- `evidence/strict-control-test-d97e1eb7/eval.log`
- `evidence/strict-control-test-d97e1eb7/fdr-eval.log`
- `evidence/strict-control-test-d97e1eb7/evaluate_strict_control_test.py`
- `evidence/strict-control-test-d97e1eb7/evaluate_strict_fdr_test.py`

核心报告 SHA256：

- Control JSON：`6FE5F2AACBF75CF88C70F57EADBB157A957229E6A253BAE236579500C331006B`
- FDR JSON：`630C29806466262ED7CDE12D66C891C78430D298800420041DCAAC7B55E2A661`
- Control 日志：`A487EC82A5B314E8E623B025007C32772E2EF15AC5A87A955A45292320A15AE5`
- FDR 日志：`1830CFC8431D37DA6910A43F4CB18CC026435007F90FE8E104AC82915A43DA55`
