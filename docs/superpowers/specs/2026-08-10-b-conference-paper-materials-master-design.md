# B 会论文主材料总索引设计

## 目标

新增一份中文权威主材料 Markdown，统一梳理当前 GitHub 仓库中真正与 B 会论文写作有关的最新方法、实验结果、代码入口、证据资产、原创边界和待补实验。历史文档继续保留，但主材料负责指出哪些结论已经被更新、哪些只能作为研究过程记录。

## 正文主线

1. 研究问题：VisDrone 密集小目标场景中的 RT-DETR 定位精度瓶颈。
2. 严格公共协议：Ultralytics 8.4.90、RT-DETR-L、VisDrone、scratch、seed0、100 epoch 及完整训练参数。
3. 严格 Stock Control：统一独立评估及其 GitHub 证据。
4. FDR：结构改动、FGL、YAML 可插拔实现、相对严格 Control 的完整总体/尺度/类别结果和开销。
5. BPDD：未来层 softmin 混合教师、better-only gate、训练期隔离、Screen30 严格配对、Formal100 独立结果及相对既有 FDR100 的初步对照。
6. 最终投稿结构：当前可成立贡献、不可宣称内容、第三创新点缺口、必须补齐的严格消融和多 seed。
7. 论文可直接复用材料：摘要事实、方法段落、贡献点、表格、图示、实验章节目录和 GitHub 链接。

## 附录

按时间与科学结论归档 LPR/LPR-G、IBER/Boundary、quality reranking/OAR/PFCR、FrequencyCM、SCADS、GLGM 等尝试。每项只记录：动机、最关键结果、失败原因、对当前方法的启示和对应证据入口。失败方案不得在正文中伪装成有效贡献。

## 文件策略

- 新增：`docs/CCF_B_PAPER_MATERIALS_MASTER_2026-08-10_ZH.md`。
- 新增：`evidence/paper-master-2026-08-10/` 下的关键轻量 JSON/CSV/SHA256。
- 不提交 checkpoint；使用现有 GitHub Release 链接。
- 不删除、移动或批量改写现有历史文件，避免破坏引用和证据链。
- 在主材料中标注旧文档的 authority 状态：current、superseded 或 historical-only。

## 结论边界

- FDR 基础公式来自 D-FINE，本文只能声明面向 Ultralytics RT-DETR-L/VisDrone 的结构化适配、隔离集成和统一协议验证。
- BPDD 不声明发明一般自蒸馏、分布蒸馏或后层监督前层；仅声明已审计的窄组合。
- BPDD Screen30 是严格同 authority 配对；BPDD Formal100 与既有严格 FDR100 的比较仍为初步跨 authority 对照。
- 不得把未完成的 fresh paired Formal100、多 seed、最终 FP16 latency 或第三创新点写成已完成。

## 验证与发布

- 所有正文数字必须映射到机器可读证据或现有冻结报告。
- 对总体、尺度、类别和差值执行自动交叉核对。
- 扫描矛盾、占位符、旧结论和过度原创声明。
- 生成关键文件 SHA256，提交并推送 `codex/bpdd-fdr`，再核对远端 OID 和 GitHub 原始文件可访问性。
