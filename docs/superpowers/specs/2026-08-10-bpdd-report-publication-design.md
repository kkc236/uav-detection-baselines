# BPDD 关键证据报告与 GitHub 发布设计

## 目标

在不提交大型 checkpoint 和全量逐轮日志的前提下，向 GitHub 发布一份可供论文写作、复核和交接使用的 BPDD 方法报告及其关键轻量证据。

## 交付物

1. `docs/BPDD_FDR_METHOD_FORMAL100_REPORT_2026-08-10_ZH.md`
   - 方法动机、结构、公式和代码映射；
   - 与 FDR、GO-LSD 的关系及原创声明边界；
   - 固定训练协议与对齐项；
   - Screen30、Formal100、官方 548 图独立评估；
   - 总体、尺度、类别、开销和投稿风险分析；
   - GitHub Release 与源码证据索引。
2. `evidence/bpdd-formal-848f00cb/`
   - 独立评估 JSON；
   - Screen30 Gate2 关键结果；
   - Formal100 epoch100 与尾部趋势摘要；
   - 逐 epoch 发布完成状态；
   - 文件 SHA256 清单。

## 明确排除

- 不提交约 200 MB 的 checkpoint；checkpoint 继续由 GitHub Release 承载。
- 不复制 100 个 epoch 的完整训练日志；只保留能复核结论的末轮、尾三轮和发布状态。
- 不把既有 FDR100 参考写成当前 BPDD 运行的 fresh paired control。
- 不把 FDR 基础公式、一般自蒸馏、后层监督前层等既有思想声明为原创。

## 证据边界

- Screen30 是同 authority 严格配对证据。
- BPDD Formal100 为真实 full-data、seed0、100 epoch、独立官方 val 结果。
- Formal100 的增量暂时相对既有严格 FDR100 authority 计算，只能称为初步对照；最终论文严格对照留待完整消融重跑。
- 未完成的多 seed 与严格 paired Formal100 不得写成已经完成。

## 验证

- 所有数值必须来自机器可读 JSON/CSV/日志，不手工臆造。
- Markdown 中的汇总数值与证据 JSON 交叉核对。
- 对轻量证据生成 SHA256 清单。
- 提交前执行 Markdown 占位符扫描、`git diff --check`、相关测试和远端分支 OID 核验。
