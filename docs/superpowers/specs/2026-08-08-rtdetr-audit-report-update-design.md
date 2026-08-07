# RT-DETR 研究进展与证据审计报告更新设计

## 目标

完整更新工作区根目录的 `RTDETR_导师审计版研究进展与证据报告_2026-08-08.md`，使其与截至 2026-08-08 已同步的 GitHub 远程证据及当前研究决策一致。

## 证据 authority

- FDR formal100：`origin/training-results@c899a33c`。
- FDR YAML 与交接：`origin/codex/fdr-yaml-module@505dd06c`。
- SCADS 实现：`origin/codex/scads-fdr@c93855fe`。
- SCADS 最终报告：`origin/codex/scads-publisher-fix@51b8e38d`。
- SCADS Gate JSON SHA-256：`7F86BD000CC12B8069941709BB8E04C8EF3F6E4E3A22F5B700DF52F92004002E`。

截至本次同步，没有晚于 2026-08-07 的 SCADS 远程提交，也没有 SCADS Formal100 已启动或完成的上传证据。

## 更新原则

1. 保留预注册事实：SCADS 原 Gate 为 `8/9`，`gate.passed=false`，`formal100_eligible=false`。
2. 不把旧 Gate 失败改写成通过，也不声称 Formal100 已经启动。
3. 区分两类结论：
   - 机制确认：tiny edge saturation 相对下降必须达到 50%，本轮未达到；
   - 检测效果：统一独立评估和训练尾窗指标均显示正收益。
4. 纳入 oracle 诊断：SCADS oracle tiny edge saturation 为 `0.12101536`，相对 fixed-base 的理论下降仅 `48.619%`，说明冻结的 50% 门槛甚至略高于当前 support/oracle 可达值。该事实只能解释门槛和设计的错配，不能追溯性修改旧 Gate。
5. 若研究目标改为“检测指标稳定正收益”，允许另立并冻结新的探索性 Formal100 协议；该长训不得宣称由旧 Gate 授权。
6. 所有历史跨-authority FDR 对比、SADED 限定和失败路线边界继续保留。

## 报告结构修改

### 结论摘要

- 将“SCADS 严格失败、不得继续”改为“旧机制 Gate 失败，但 detector screen 稳定正向；可在新冻结的经验增益标准下进入探索性 Formal100”。
- 明确远程尚无 Formal100 新上传。

### SCADS 章节

- 补充 F1、相对增益、final/tail-3 训练窗口。
- 解释独立评估 `+0.00240166` 与训练末轮 `+0.00256` 来自不同统计口径，不构成数据冲突。
- 补充 oracle、route accuracy、balanced accuracy、wide overflow 和各尺度路由统计。
- 将“为什么仍判定失败”改成“旧 Gate 为什么失败，以及这一失败能说明什么”。
- 增加“新的探索性推进条件”：fresh paired full-data FDR/SCADS、同 authority、事前冻结判据、不得用旧 FDR 结果冒充严格 control。

### 论文主张与任务顺序

- 可主张 SCADS 在 Screen30 中获得一致的 detector 正收益。
- 不可主张已验证 50% 饱和缓解、已通过旧 Gate、Formal100 已完成。
- 将后续 P4 从“禁止继续”改为“若目标仅为稳定正收益，先冻结新探索性协议，再启动 paired Formal100；若仍以机制确证为目标，则先改 support/router 并重新 Screen30”。

### 最终审计意见

- 保留 FDR 为成熟主模型候选。
- 将 SCADS 定位为“检测效果已过经验筛选、机制门未过、具备探索性长训价值”的小模块候选。

## 验证

- 检查 Markdown 标题层级和表格。
- 搜索并消除“SCADS 不得启动 Formal100”等与新决策冲突的绝对措辞。
- 搜索所有数值并与 `gate-report.json`、CSV 和远程提交一致。
- 确认没有写入“Formal100 已启动/完成”的无证据陈述。
- 保留原文件 UTF-8 编码。
