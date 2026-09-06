# 无服务器部署方案：对抗性复核与修订（2026-09-06）

审计对象为 `521a5325` 中的部署设计以及前序优化建议。采用
academic-research-suite 的反方证据检查思路，随后按作者已授权的部署任务实施。
这是一轮内部算法/工程检查，不是会议适配评价；criteria_binding_unavailable。

## 最强反方意见

修复方案不能仅靠“FP32、log-space、严格对照”等术语证明成立。一个固定的
归一化框宽高下限会改变小目标的建模空间；将 log_softmax 加在教师候选上，
却继续先指数化再混合，仍会丢失极小目标概率；四臂虽然补了 F，但若因 G
单独不涨点就取消 I，会主动放弃研究交互效应。更关键的是 dry-run 的数据与
权重检查不能证明真实模型能反传，CPU 测试也不能替代服务器上的 CUDA、正式
数据和长训证据。因此部署必须把每个保证缩到可执行验证的边界，并保留明确的
未验证项。原设计的优点是拒绝复用旧结果和新增性能模块；这些约束予以保留。

## 发现与裁决

| 等级 | 证据位置与问题 | 裁决/最小修正 | 置信度 |
| --- | --- | --- | --- |
| Major | equation: 设计 §4.2，归一化宽高下限没有精确数值且可能被理解为 1e-3 | 保留历史 distance-space 1e-3，只追加 `8*eps*max(1,abs(center))` 机器精度边界；直接计算 CXCYWH，避免相减丢失小框 | 5，数值反例与公式 |
| Critical | equation: AC-BPDD 教师 `log(clamp(p,1e-6))` 与学生 log_softmax 不同口径 | NLL20 学生/NLL40 教师错误门控 0.308224；候选、权重归一化、教师混合、评分均使用 log-space，空教师分支保持有限 | 5，失败测试复现 |
| Major | text: 前序建议“若 g-f 不明显为正，停止 BPDD” | 撤回硬淘汰规则；g-f 不能推出 i-h，保留组合交互实验 | 5，二因素对照定义 |
| Minor | text: 前序建议“小于约 0.2–0.3 pp”补 seed | 该数值不是显著性阈值；报告单 seed 差值，额外 seed 用于估计训练变异，不承诺特定显著性 | 5，证据边界 |
| Major | absence: 设计 §7 — expected 真实模型反向与推理证据; checked dry-run/测试清单 | 增加 f/g/h/i 完整图、criterion、backward、一步参数更新和 eval 的 FP32/CPU BF16 测试 | 5，执行路径区分 |
| Major | absence: VisDrone main — expected 逐 epoch 几何/BPDD记录; checked 三臂 launcher 和 UAVDT recorder | 提取共享 recorder 并接入四臂；缺失观测写 null，不冒充零激活或梯度失败 | 5，代码调用路径 |
| Major | code: 新 FP32 reference 直接输入整个模型 `.half()` 后的 pos_mlp 导致 Float/Half 冲突 | attention/MLP 使用网络精度的 reference 副本；解码保持 FP32；Integral 支持值跟随计算输入精度 | 5，两个实际模型失败复现 |
| Major | runtime: 本机纯 CPU FP16 grid_sample 对有限 value/grid 返回 NaN | 已定位至 stock cross-attention 的采样调用；同一输入 FP32 正常。当前入口明确拒绝 CPU FP16，建议 FP32/CPU BF16；不修改 FIA 结构，不宣称 CUDA 存在同一问题 | 5，本机实测，外推范围受限 |

Critical 仅针对“当前门控严格 better-only”的工程声明，不用于否定 BPDD 的
独立构造或论文整体创新。没有进行独立跨模型复审，结果不是外部专家背书。

## 已固定的方法边界

- 固定初始 reference `r0` 用于所有层的分布解码；attention reference 仍逐层更新。
- FP32 几何路径位于显式禁用 autocast 的区域，保留原始非均匀 Integral 支持值。
- 整个模型 `.half()` 会量化模型 buffer，随后 FP32 解码不会恢复被量化的位精度；
  不能据此声称与原 FP32 输出逐位等同。当前 F/I CPU FP16 测试验证明确拒绝，
  不是验证半精度完整推理成功。
- 保留 `distance2bbox` 官方原语及其精确比对测试；新增 wrapper 是本项目数值约束。
- 有限分布、有限归一化非负 reference extents 是输入合同。零 reference 在前向
  可得到数值正框，但不声称其 raw extent 梯度非零，也不把 NaN/Inf 静默替换为正常框。
- extent 使用 surrogate gradient；自动微分与硬截断的有限差分不相等是明确设计，
  不是“数学上精确的投影梯度”。持续非法 raw extent 仍需要运行日志判断。
- AC-BPDD 的 better 指插值目标边 NLL 更低。它不保证解码 IoU、更高 AP 或更强泛化。
- legacy final-assignment BPDD 保持历史公式；v2 修复是当前 AC-BPDD 的新实验身份。
- 所有性能数字继续保留原始源码身份。f/g/h/i 的 v2 Formal100 均未完成。

## 未覆盖的执行条件

当前本机 Python 为 CPU PyTorch。真实 CUDA FP16、服务器 baseline 环境、正式
数据 dry-run、真实 initial-state 实物、640/batch8 训练和导出后 FP16 engine 均需
在对应环境单独验证。CPU BF16 完整图成功不能外推这些结果。128 分辨率随机输入
用于执行检查，无 AP、收敛或真实小目标数据集结论。

CPU FP16 排错时，失败采样输入形状 `[8,32,16,16]`，grid 形状
`[8,300,4,2]`；二者均有限，绝对值最大值分别约 `0.0109863/0.987305`。
采样首先输出 NaN，同一 value/grid 转 FP32 后采样有限；关闭 MHA fastpath 或
MKLDNN 未解决。该观察只绑定本机版本，不宣称完成了 PyTorch 上游根因修复。

## 交付约定

最终相关源码回归：176 passed、1 skipped；跳过项依赖 CUDA。八个完整图检查
覆盖四臂 FP32/CPU BF16；两项 CPU FP16 检查验证明确拒绝。此数目不代表仓库
所有历史路线的全部测试，更不代表数据集精度测试。

源码由代码仓 v2 分支维护，材料仓收到 exact-commit ZIP、SHA-256、JUnit 原始
结果和 JSON 报告。历史日志、权重摘要及既有结果数值保持原样。
