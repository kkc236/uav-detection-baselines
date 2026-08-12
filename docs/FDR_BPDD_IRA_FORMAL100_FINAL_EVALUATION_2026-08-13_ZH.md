# FDR + BPDD + IRA Formal100 独立终评

## 结论

FDR + BPDD + IRA 已完成全量 VisDrone、seed0、100 epoch 训练和 exact epoch100 EMA 独立验证。该组合相对历史 FDR 的 mAP 和 AP75 仍为轻微正向，但低于 FDR + BPDD；因此本次实验不支持 BPDD 与 IRA 在 FDR 上形成稳定叠加协同，当前最成熟的小模块组合仍是 FDR + BPDD。

## 协议与证据状态

- 基础模型：Ultralytics RT-DETR-L 8.4.90
- 数据：VisDrone train 6471、val 548，10 类
- 训练：从统一 initial state fresh 启动，`pretrained=False`，seed0，100 epoch
- 统一参数：`imgsz=640`、`batch=8`、`workers=8`、MuSGD、AMP scale 128、300 queries、`NMS=False`、`max_det=300`
- 源码 authority：`756e025efc28088402cdbaf806ac06d4b93ee5eb`
- 终评工具修正：`d72009062bf6431b3cf475825f8f981c1db2605e`
- epoch 证据：100/100，全部梯度有限，publication ledger 100/100 verified
- exact checkpoint：epoch100 EMA，checkpoint SHA-256 `3EA6CD138C20AD38A35EF652407B6D77C423379A06E669028F771517843E069F`
- EMA state SHA-256：`CC45CDCD7F4B5788E2E713DF6B5FF3D3F16E34B7730BF59CC77FDABE9390876F`

## 独立 val 指标

| 指标 | FDR 历史权威 | FDR + BPDD 历史权威 | FDR + BPDD + IRA | 相对 FDR | 相对 FDR + BPDD |
|---|---:|---:|---:|---:|---:|
| Precision | 0.56911 | 0.57063 | 0.55576 | -1.335 pp | -1.487 pp |
| Recall | 0.49278 | 0.49446 | 0.49834 | +0.557 pp | +0.388 pp |
| F1 | 0.52484 | 0.52983 | 0.52549 | +0.065 pp | -0.434 pp |
| AP50 | 0.48468 | 0.48641 | 0.48124 | -0.344 pp | -0.517 pp |
| AP75 | 0.29253 | 0.29810 | 0.29510 | +0.258 pp | -0.299 pp |
| mAP50-95 | 0.28966 | 0.29226 | 0.28995 | +0.029 pp | -0.231 pp |

这些差值属于跨历史 authority 的初步参考，不是 fresh paired Formal100。FDR authority 因 exact-final-EMA 身份未完全封口被评估器降级；FDR + BPDD authority 的数据路径元数据不完全一致，也被降级。它们可用于方向判断，但最终论文消融仍应重跑严格配对。

FDR + IRA 历史实验只有 test 截图，没有同协议 val 权威 JSON，也缺少 AP75、尺度和类别完整信息。本报告将该行明确标记为 `unavailable`，没有补写或推测缺失数据。

## 分尺度 mAP

| 尺度 | FDR + BPDD + IRA |
|---|---:|
| Tiny | 0.14632 |
| Small | 0.29354 |
| Medium | 0.38544 |
| Large | 0.40602 |

## 逐类别指标

| 类别 | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| pedestrian | 0.56084 | 0.22686 | 0.27310 |
| people | 0.49452 | 0.13411 | 0.20902 |
| bicycle | 0.23596 | 0.08319 | 0.10971 |
| car | 0.85759 | 0.68844 | 0.61255 |
| van | 0.53214 | 0.43356 | 0.38260 |
| truck | 0.37758 | 0.28764 | 0.25553 |
| tricycle | 0.35745 | 0.21706 | 0.20651 |
| awning-tricycle | 0.18526 | 0.12699 | 0.11273 |
| bus | 0.60403 | 0.50921 | 0.44188 |
| motor | 0.60702 | 0.24397 | 0.29589 |

## 效率

| 指标 | FDR + BPDD + IRA |
|---|---:|
| 参数量 | 33,458,090 |
| GFLOPs | 111.6481 |
| FP16 中位延迟 | 27.488 ms |
| FP16 P95 延迟 | 27.943 ms |
| FP16 FPS | 36.379 |
| 峰值显存 | 248.93 MiB |

该结果是当前组合的单臂 benchmark，不应与来自不同 runtime 的历史速度数字直接作论文级差值。

## 科学判断

IRA 增加了召回，但同时降低了 Precision 与 AP50。其 AP75 相对 FDR 有轻微收益，说明残差路径更偏向严格定位；但它干扰了 BPDD 已形成的最佳训练解，最终 mAP 未能超过 FDR + BPDD。后续不建议围绕该三模块组合继续调参，应冻结为“兼容但不协同”的消融证据，正文主线优先使用 FDR + BPDD，第三模块另选功能更正交且有严格证据的候选。

## 文件

- 独立终评：`evidence/bpdd-ira-formal-756e025e/fdr-bpdd-ira-independent-eval.json`
- 终评日志：`evidence/bpdd-ira-formal-756e025e/final-eval.log`
- FDR 历史权威封装：`evidence/bpdd-ira-formal-756e025e/fdr-historical-authority.json`
- 独立终评 SHA-256：`5AE8CA9AC71FA49D7C754E69CDFF87934D44B02FE7992754A89AE2151701FC5F`

