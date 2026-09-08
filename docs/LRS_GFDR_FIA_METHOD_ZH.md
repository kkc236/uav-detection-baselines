# LRS-GFDR-FIA 方法说明

LRS-GFDR-FIA 是 LRS-GFDR 的一次单变量扩展：保留 LRS 的可靠性收缩和 FDR 的几何可行域解码，只在 P3 特征上加入现有的 FIA 分支。FIA 采用独立私有初始化，残差系数从零开始；P4/P5 保持 stock bypass，BPDD、DN supervision 和额外推理分支均不加入。

该实验用于回答“在几何可行性已经固定后，FIA 是否带来额外收益”，而不是重新定义 LRS-GFDR。它与 LRS-GFDR 使用同一数据签名、初始状态、seed、优化器和 100 epoch 预算，输出目录独立为 `visdrone-v2-lrs-gfdr-fia-20260909`。
