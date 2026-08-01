# LPR-G v2：30-Epoch Cutoff 筛选设计

## 1. 目标与结论边界

把 seed0 固定 10% 子集筛选从两臂各完成 50 epoch，改为两臂各在 epoch 30
截止并比较，以缩短结构可行性筛选时间。

该实验必须命名为：

> 50-epoch schedule 下的 30-epoch cutoff 筛选

它不是“总训练轮数配置为 30 epoch”的实验，也不能与采用 30-epoch scheduler 的结果混写。

## 2. 保持不变的训练轨迹

- Ultralytics Trainer 的 `epochs` 仍配置为 50；
- 学习率 scheduler 仍按 50 epoch 计算；
- `close_mosaic=10` 仍按 50-epoch schedule 执行，因此 epoch 1–30 不提前关闭 mosaic；
- control 和 LPR-G 都从同一 seed0 公共初始状态开始；
- 数据、batch、workers、AMP、优化器、增强、验证和 checkpoint 规则保持不变；
- 每个完成 epoch 仍必须先写入指标、审计和 optimizer evidence，再上传并验证 GitHub
  checkpoint 和结果提交。

## 3. 截止行为

- 正式筛选 arm 的有效 epoch 必须恰好为 1–30；
- epoch 30 的 checkpoint 与轻量结果完成 GitHub 远端验证后，Trainer 才允许正常停止；
- 不允许生成 epoch 31 的完整训练/验证结果；
- preflight 仍为两臂各 1 epoch；
- 当前 control run 可以保留并使用，因为它本来就是相同 50-epoch schedule；只取其完整、
  已发布的 epoch 1–30；
- LPR-G 必须 fresh 启动，并采用与 control 完全相同的 50-epoch schedule，在 epoch 30
  完成发布后截止。

当前旧监督器仍以 50 为完成条件。迁移时只保留已经完整发布的 epoch；任何正在执行但尚未
形成完整结果和已验证发布账本的 epoch 都不计入证据。已有 run 不删除、不覆盖。

## 4. 30 轮比较门禁

比较器必须只接受连续且无重复的 epoch 1–30，并拒绝 epoch 31 及以后记录。

原 50 轮门禁作如下等价替换：

- “epoch 50”替换为“epoch 30”；
- “epoch 41–50 均值”替换为“epoch 21–30 均值”；
- 两臂各要求 30 条 results、diagnostics 和 common-state audit；optimizer evidence 必须覆盖
  截止 epoch 30 的全部优化器尝试且编号连续；
- 两臂各要求 30 条连续、远端验证成功的 publication ledger，共 60 条；
- 同 checkpoint stock/refined 消融使用 LPR-G epoch 30 checkpoint；
- AP、AP75、mAP50、安全退化线、私有分支活性和开销条件保持原定义。

筛选通过只表示该结构获准进入全数据 100 epoch，不构成跨 seed 统计结论。

## 5. 通过后的正式实验

30-epoch cutoff 筛选通过后，control 和 LPR-G 在全量 6471 张训练集上分别从相同 seed0
公共初始状态 fresh 启动 100 epoch。禁止把 cutoff checkpoint resume 到正式实验。

正式 100 epoch 的 scheduler、close_mosaic、逐 epoch GitHub 发布、评估和比较规则均不变。

## 6. 工程与测试要求

- 训练设置测试必须同时证明 `epochs=50` 和 cutoff `30`，防止误改成 30-epoch scheduler；
- callback 顺序必须保证 diagnostics → common audit → publication → cutoff stop；
- cutoff stop 只有在 epoch 30 发布账本 `verified=true` 后生效；
- supervisor 对 screen arm 的完成条件为 30，对 formal arm 仍为 100；
- 比较器测试覆盖：恰好 1–30 通过、缺轮/重复/epoch31 拒绝、60 条发布记录要求；
- 恢复测试覆盖：最高已验证 epoch 小于 30 时继续，等于 30 时直接认定 arm 完成；
- 现有全项目测试必须全部通过后才能部署新提交。
