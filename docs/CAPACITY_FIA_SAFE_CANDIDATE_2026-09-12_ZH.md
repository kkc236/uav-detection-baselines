# LRS-GFDR + Capacity-BPDD v3-Safe + FIA 候选说明

## 结论先行

本候选用于替代已经完成的三模块联合旧版进行下一次严格配对试验，但当前仅完成代码与本地回归，**尚无 Formal100 精度结果，不能宣称涨点**。历史配置、历史结果和历史启动器均保留不变。

候选方法修订号：`v3-safe-loc-kd-decay80-100-no-cls-kd`。

## 修改依据

对联合旧版、Capacity-BPDD 单独臂和 FIA 单独臂的运行记录与固定样本反向探针复核后，得到以下边界：

1. expert 直接监督与主检测目标在 FIA/P3 参数上的梯度大体同向，没有证据支持“FIA 与整个 expert 天生冲突”。
2. expert 相对 base 的匹配框 IoU 增量很小且符号不稳定，说明旧版 BPDD 教师优势不足，不能靠继续扩大蒸馏权重稳定获得收益。
3. 分类蒸馏梯度虽然量级很小，但存在真实反例：教师类别置信度更高时，可能推动正类概率远离 VFL 所需的 IoU 软目标。旧版门控只能证明教师框终点较好，不能证明分类更新方向安全。
4. 联合旧版在后十轮均值明显落后于 FIA 单独臂；这支持给 BPDD 辅助项增加后期退出机制，让 FIA 和主检测目标完成收敛，但不能据此保证新候选一定提升。
5. 几何可行域修正未被移除。它是 LRS-GFDR 在跨数据集迁移时的合法框保障，不是本次冲突修复的变量。

## 候选结构

模型图与旧三模块版本完全一致：

- LRS-GFDR：保留 `alpha=0.25` 和可行域几何解码；
- FIA：仍只作用于 P3，P4 保持旁路，残差系数仍从 0 初始化并学习；
- Local Boundary Expert：结构、容量、初始化、直接监督均保持不变；
- Capacity-BPDD 定位蒸馏：权重仍为 `0.15`，第 10–20 轮线性热身；
- Capacity-BPDD 分类蒸馏：权重固定为 `0.0`；
- Capacity-BPDD 后期调度：第 80 轮保持 1.0，第 80–100 轮线性衰减，第 100 轮为 0；只衰减 BPDD 两个辅助项，不衰减 expert 直接监督、FIA、FGL 或主检测损失。

因此，这不是扩大参数量的新版网络，而是针对联合优化时目标方向和退出时机的训练控制候选。

## 同时修复的技术口径

- 评估态报告的 `boxes`、`classes`、`last_corner_logits` 现在均来自 expert，同源且 query 数一致；训练态仍保存 base 六层分布供 FGL/BPDD 使用。
- runtime recorder 对 capacity 字段先做完整性与有限性检查。真实 0 会进入均值，缺失值和 NaN/Inf 分别计数，坏观测不再伪装成 0。
- runtime 记录新增 `capacity_schedule_scale_mean`、`fia_gradient_norm`、capacity 缺失/非有限/拒绝观测数以及梯度缺失/非有限计数。
- Safe 模型会校验冻结的 BPDD 参数；若误传旧 YAML，会明确失败，避免旧配置被错误标记为 v3-Safe。

## 文件入口

- 配置：`configs/rtdetr-l-lrs-gfdr-capacity-v3-safe-bpdd-fia.yaml`
- 模型与 trainer：`src/rtdetr_bpdd_capacity_fia.py`
- 正式启动器：`scripts/train_lrs_gfdr_capacity_v3_safe_fia.py`

## 服务器运行命令

在已安装项目环境且工作树 clean 的服务器中执行：

```bash
python scripts/train_lrs_gfdr_capacity_v3_safe_fia.py \
  --dataset-root /path/to/VisDrone \
  --initial-state /path/to/initial-state.pt \
  --output-root /path/to/formal100-runs \
  --dry-run

python scripts/train_lrs_gfdr_capacity_v3_safe_fia.py \
  --dataset-root /path/to/VisDrone \
  --initial-state /path/to/initial-state.pt \
  --output-root /path/to/formal100-runs
```

先运行 `--dry-run` 核对 authority JSON 中的源码、配置、数据集和初始状态哈希，再启动正式训练。该启动器固定 100 轮、seed 0、640 输入及既有 Formal100 优化器/批量协议，不提供从旧三模块 checkpoint 接跑的入口。

## 结果判定

必须与同初始状态、同数据、同训练协议的 LRS-GFDR、LRS-GFDR+FIA、LRS-GFDR+Capacity-BPDD 及旧三模块版本比较：

- 主判据：best mAP50-95；
- 同时报告：best 轮次、mAP50、P、R、最后十轮均值及波动；
- 机理证据：BPDD schedule、有效边比例、定位优势、expert/FIA/FDR/common 梯度范数与非有限计数；
- 若只改善末期稳定性而 best 不升，应如实描述为优化稳定性候选，不能包装成精度提升。

