# BPDD 解码框质量门控候选：实现与对抗性审计

日期：2026-09-07。状态：本地候选实现通过相关测试，未部署、未训练；不能据此声称涨点或估计成功概率。

## 实现范围

保留 assignment consistency、原逐边 NLL 门控、未来层混合、weight=0.15、temperature=0.5、margin=0.02 和原损失分母。仅增加 detached 框级 IoU 条件。默认关闭，候选配置单独存放于 `configs/rtdetr-l-lrs-fdr-bpdd-iou-candidate.yaml`。

流程：先计算原 reliability；将 reliability=0 的边保留为学生分布，其他边替换为混合老师分布；使用实际非均匀支持点求期望，经生产 v2 解码器解码，再比较相对 GT 的完整框 IoU。严格改善超过 iou_margin（候选为0）才保留原逐边 reliability。

该规则比较的是实际被蒸馏的混合目标：若只检查完整老师框，未被激活的边可能制造虚假改善。门控、老师和 reference 均停止梯度；质量计算禁用 autocast，使用 FP32。IoU 使用相对中心与宽高计算，减少微小框角点相减的精度损失。

## 已检验的攻击面

- NLL 更低但定位更差：旧损失为正，新损失为零，学生梯度为零。
- 定位与分箱均改善：保留非零学生梯度；两层合成输入中老师 logits/reference 无梯度。
- 部分边激活：与独立 Integral + FP64 角点 IoU 计算对照，覆盖 64 个随机样本与 1e-6 至 1e-1 reference 尺寸。
- 所有边不激活：IoU 差为零，门控关闭。
- NaN/Inf：质量函数报错；不将异常自动修复为合格老师。
- 非法 margin、字符串布尔、非 consistent 模式：拒绝。
- 独立候选 YAML：真实模型构造及选项解析通过；现有配置默认保持关闭。

验证命令：

```text
python -m pytest tests/test_bpdd_iou_gate.py tests/test_bpdd_loss.py tests/test_bpdd_fdr_criterion.py tests/test_lrs_system_launcher.py -q
```

结果：52 passed。git diff --check 通过。

## 审计后仍存在的局限

1. 目标框更好不保证梯度更新改善定位：KL 约束全分布，实际参数还受 FGL/L1/GIoU 和共享网络影响。两层 logits 的停止梯度测试不代表所有未来层参数都不会间接受影响。
2. 部分边具有不同 reliability；当前检查完整替换的终点，不保证实际小步更新的方向。需要真实批次 BPDD/主损失梯度夹角和一步更新诊断。
3. 门控可能减少有效监督，尤其早期无相交框的 IoU 同为0。首版不同时加入 GIoU 回退、权重放大或 warmup；先测保留率。
4. margin=0 未调参；极小正 IoU 差可能是数值波动。应记录差值分布，不能将接近零的差理解为可靠改善。
5. 几何修正可能掩盖原始不可行框；需分组记录修正触发样本上的门控效果。
6. 当前没有真实 VisDrone 批次收益/覆盖率报告，也没有新增 epoch 级质量统计持久化；该候选不是完成 GPU 门禁的服务器交付版。
7. 通用 Formal100 launcher 仍使用原 ARM_CONFIGS，不会自动选候选 YAML；正式实验需单独候选身份与配置路由，不能编辑已有 G 运行配置冒充同一实验。

## 下一步实验决策

在可用早/中/后期 checkpoint 上，用固定真实批次统计原 NLL 激活边数、错误放行比例、新门控保留率、IoU 差及 tiny/small 分组，并测共享参数梯度范数与 cosine。以此判断是否值得正式训练；不预设涨点阈值或声称成功率最高。

若新门控主要过滤劣质老师且保留足够监督，再建立独立 G-IoU 候选运行，与原 G 使用相同初始化和训练预算比较。已有 F/G 训练不修改。真实数据若显示错误放行很少，应回到蒸馏梯度与分布目标诊断，而非继续收紧门控。
