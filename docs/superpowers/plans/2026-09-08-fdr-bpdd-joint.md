# FDR / BPDD 联合候选执行记录

目标：在当前权威工作树内完成可选 R / M，实现和审查在当前任务串行进行。

1. 新增 tests/test_bpdd_joint_mechanism.py，固定生产 KL 反例与累计梯度反例；先运行新选项测试确认失败。
2. src/bpdd_loss.py 添加默认关闭的 residual_gradient_only 和默认 kl 的 distribution_objective；仅修改 consistent 路径。M 使用未截断 GT 距离作支持与方向筛选，采用现有完整框门控。
3. src/rtdetr_fdr_bpdd.py 注册 YAML 字段；新增单独候选 YAML，不改 F/G。
4. 验证旧模式默认/显式关闭数值和梯度一致、残差路径前向相同且直接历史梯度消除、KL 反例下期望损失方向改善、老师/ref 停止梯度、不可表示目标及反向/越界老师被拒绝、禁用和空匹配为零、无参数增量。
5. 运行 BPDD、FDR 与 launcher 的现有相关测试，记录实际输出与能力边界到 docs/FDR_BPDD_JOINT_MECHANISM_AUDIT_ZH.md。

候选配置仍需对应 launcher 路由和真实数据预检才可启动正式训练，本次不把通用 G 身份复用为新方法。
