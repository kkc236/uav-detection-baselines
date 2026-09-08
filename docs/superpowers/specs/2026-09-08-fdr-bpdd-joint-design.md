# FDR / BPDD 联合机理调整设计

日期：2026-09-08。范围：本地候选代码、反例与测试；无服务器训练结果。

## 证据与选择

当前 FDR 使用固定初始参考、六层累计 logits、非均匀 33-bin Integral、FP32 可行域解码，以及 LRS 加权 FGL。BPDD 对未来同 query/GT 层按逐边 NLL 混合，使用 detached reliability 加权 KL。

本次已用生产函数复现两种风险：

1. 老师 IoU 0.930166 > 学生 0.702463，原 IoU 候选仍把独立 logits 的一次负梯度步推到 IoU 0.702385。目标终点改善不保证 KL 局部方向改善。
2. 三层累计结构中，匹配为 A/B/B。只有第 2 层有稳定老师，但第 1、2 层残差梯度范数均为 0.01669685。source/future assignment 一致不意味着被直接更新的全部历史残差 assignment 一致。

这些证明机制可能发生，不证明其真实数据频率或是本次 mAP 持平的原因。

比较三个方向：

- 仅加 IoU 门控：已有实现，无法消除上述两项问题。
- 保持 FDR 前向，独立调整 BPDD 的梯度回传与蒸馏目标：本次选定的低侵入候选。
- 更换 FDR 动态参考、支持点或累计结构：会连带改变分布语义和教师坐标，暂作为需要真实支持饱和证据的后备路线。

## 候选 R：当前残差直接回传

设累计 logits z_l=sum_{k<=l} delta_k。BPDD 使用前向值相同的代理：

`r_l = z_l - z_(l-1)`；`z_bpdd_l = stopgrad(z_l) + r_l - stopgrad(r_l)`。

实现括号为 `stopgrad(z_l) + (r_l - stopgrad(r_l))`，确保有限输入前向值完全相同。第 1 层无需处理。

只改变 BPDD 分支，不改变 FGL/L1/GIoU、生产解码或推理。阻断的是累计加法链的直接梯度；Transformer 特征仍跨层连接，不能声称历史层参数完全无梯度。

配置：`residual_gradient_only: true`，默认 false，仅 consistent 模式可用。

## 候选 M：GT 同向的期望蒸馏

保留未来层候选、assignment 和 NLL reliability。使用实际支持点 W 计算学生/老师期望 mu_s / mu_t。GT 的原始未截断四边距离为 d*。

逐边保留条件：d* 在 [-4,4] 内；老师与学生相对 GT 同侧或老师恰在 GT；老师绝对边误差严格更小；原 NLL reliability > 0。随后对被保留边组成的完整框执行已有 decoded IoU 门控。

损失为 `weight * sum(reliability * SmoothL1(mu_s, stopgrad(mu_t), beta=1)) / candidate_edges`，四边距离单位仍为 FDR 支持坐标。默认 `distribution_objective: kl`；候选为 `mean_huber`，必须 consistent 且开启 decoded IoU gate。

独立单边 logits 上，小步期望变化与 mu_t-mu_s 同号，因为 d(mu)/d(z_i)=p_i(W_i-mu)，负梯度产生 `-rho'(mu_s-mu_t) * sum_i [p_i(W_i-mu_s)]^2`。这只给出单边独立 logits 局部性质，不保证共享参数、累计路径、Adam、有限步或完整 IoU 改善。

M 不再复制完整老师分布，应称为“分布期望蒸馏候选”，不能继续声称全分布暗知识蒸馏。它与已有 L1/GIoU 有冗余风险，必须与直接 GT 期望辅助损失比较。

## 不变部分与实验

保持 FDR 固定参考、FP32 解码、extent floor、LRS alpha=0.25、FGL、初始化和推理参数不变。两个选项默认关闭，旧配置数值与梯度必须兼容。候选 YAML 单独存放，不改已有训练臂。

最小因果分解：当前 G；G+R；已有 IoU 候选；IoU+M；IoU+M+R。先固定真实训练批次比较匹配切换、原始支持越界、floor 分组、老师错误放行、有效覆盖、分层梯度范数与主损失夹角，再决定正式训练预算。不能仅以本地测试通过推断涨点。

KL 与 Huber 的 0.15 不代表相同梯度预算；保留数值用于兼容和原型，正式比较必须记录或预先校准同一参数组梯度范数，不能按验证 best 反复挑系数。
