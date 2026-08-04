# FDR d97e1eb7 预检与正式训练 epoch 10 证据快照

本目录冻结了 FDR-RTDETR-L 当前论文说明所引用的两类机器可读证据：

1. `fdr-preflight/d97e1eb7-seed0-attempt001/`：F0--F4 预检报告与总判定；
2. `fdr-formal-d97e1eb7/formal-seed0-fdr-fdr-v1/`：全数据 seed0、100 epoch 正式训练在完成 epoch 10 时的运行清单、逐轮指标和优化器证据。

## 来源与完整性

- 服务器运行根目录：`/data/uav/runs`
- 源压缩包：`fdr-preflight-formal-epoch0010-v1.tar.gz`
- 压缩包 SHA256：`D850615D587455CE8D7F8474A3004D7F91C0CDDA92E031CBA5FDEFBB5E59C450`
- 快照时间：2026-08-04 19:00（Asia/Shanghai）
- 模型/训练 authority：`d97e1eb7f98414752a1c1f38287697db3f2a0679`

压缩包在下载后重新计算 SHA256，结果与服务器生成值一致。为便于 Git 审查，本目录保存解压后的轻量 JSON/JSONL/CSV 证据，不提交模型 checkpoint。

## 结论边界

该快照只能证明：F0--F4 已通过，并且正式训练在 epoch 10 时仍正常运行。正式训练目标为 100 epoch，因此这里的 epoch 1--10 指标不是最终论文结果，不能与 100-epoch baseline 直接相减。
