# LRS-GFDR 方法说明

## 模块定位

LRS-GFDR（Layerwise Reliability Shrinkage with Geometrically Feasible FDR）将 LRS 的层级可靠性收缩与 FDR 的几何可行域解码封装为一个联合迁移模块。

它不是把两个结果简单相加，而是固定一条联合处理链：

1. FDR 输出四条边的分布式偏移；
2. 几何可行域解码在 FP32 中计算边界，约束宽高为有限且严格为正；
3. 可行框作为后续 decoder reference 更新，避免非法框传播到第二数据集；
4. LRS 以 `alpha=0.25` 对 FGL 监督进行可靠性收缩。

## 明确排除

本模块不包含 BPDD、FIA、DN supervision 或额外推理分支，主干和推理参数规模保持与 LRS-FDR 一致。几何校准的论文角色是跨数据集可行性与数值稳定性保障，不把“未校准而崩溃”的实验当作有效精度对照。

## 复现实验

配置文件为 `configs/rtdetr-l-lrs-gfdr.yaml`。训练使用与 Formal100 LRS-FDR 相同的数据、初始状态、随机种子、优化器和 100 epoch 预算，输出到独立的 `visdrone-v2-lrs-gfdr-20260908` 目录。
