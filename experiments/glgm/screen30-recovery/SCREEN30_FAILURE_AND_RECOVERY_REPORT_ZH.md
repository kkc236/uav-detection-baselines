# GLGM Screen30 故障与恢复报告

## 结论

`screen30-seed0-v1` 不具备正式实验效力。Control 在 epoch 5 的训练损失与检测指标仍为有限值，但三项训练期验证损失同时写入 `NaN`；即使 epoch 6 恢复有限值，严格协议要求每一行均有效，因此整组配对必须作废，不能续跑或复用。

## 直接证据

- 失败目录：`/home/ubuntu/glgm/work/screen30-seed0-v1`
- 运行时间：`2026-08-08T13:18:17Z` 至 `2026-08-08T14:11:43Z`
- supervisor 退出码：`1`
- epoch 5：Precision `0.47268`、Recall `0.39309`、mAP50 `0.36512`、mAP50-95 `0.21075`，但 `val/giou_loss`、`val/cls_loss`、`val/l1_loss` 均为 `NaN`
- epoch 6：所有字段恢复有限值，说明这不是持续性的训练损失发散，但不能消除 epoch 5 已污染记录的事实

## 原因与修复

Ultralytics 训练期验证原本在 `trainer.amp=True` 时同时对 RT-DETR 推理与 `model.loss()` 启用 autocast。该路径可产生瞬时非有限验证损失，而原生恢复逻辑只检查训练损失和 fitness，不检查每个验证损失字段。

采取两层共同修复，Control 与 GLGM 完全一致：

1. 训练仍保持 AMP；仅训练期验证强制 FP32，避免改变两组训练优化协议。
2. 在 `on_fit_epoch_end` 检查 metrics、累计训练损失、末批损失和 fitness；发现任何 NaN/Inf 立即抛出异常。
3. 将 `ultralytics/engine/validator.py` 纳入预检源代码哈希，防止未绑定补丁的代码参与实验。

修复后本地与服务器单元测试均为 `5 passed`，完整 CUDA 前向/反向预检通过。修复版预检仍使用同一公共初始化指纹 `325E80C7FA9826028169F1D99071C09DA1C900FBABB029CBE43B675C151F6BE3`。

## 启动恢复记录

`screen30-seed0-v2` 在 epoch 1 前因启动工作目录未命中已核验的 `yolo26n.pt` AMP 自检缓存而退出，没有产生训练结果。该目录同样不复用。随后独立 AMP 自检以缓存 SHA-256 `9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF` 通过。

对抗性审计随后指出仅在 `on_fit_epoch_end` 检查会晚于检查点保存。`screen30-seed0-v3` 因此在 epoch 1 完成前主动停止，没有生成 `results.csv`。最终版本将全训练状态有限值检查移至 `_handle_nan_recovery()` 对应的保存前位置，同时保留保存后回调复核，并禁止原生 NaN 恢复。强守卫版 `smoke2-strictnan-v3` 随后完成 Control/GLGM 各 2 epochs、独立评估、benchmark 和全部产物校验，两轮对抗性审计结论均为 `GO`。

`screen30-seed0-v4` 启动后与独立审计的 Smoke2 收尾阶段形成并发 GPU 负载，不满足正式实验的资源隔离要求，因此在 epoch 1 完成前停止且不生成 `results.csv`。正式权威改为资源空闲后启动的 v5。

## 当前正式权威

只有 `/home/ubuntu/glgm/work/screen30-seed0-v5` 可成为本轮 Screen30 的正式权威。它采用全量 6471/548 数据、30 epochs、batch 4、Control 后 GLGM、同一物理 GPU 0 顺序训练，并从同一次预检生成的配对初始化开始。v1 至 v4 仅作为失败审计证据保留。
