# UAVDT revised Full 跨服务器训练交接

## 方法身份

- 方法：`LRS-FDR+AC-BPDD+FIA`
- 固定身份：`lrs_fdr_ac_bpdd_fia`
- 配置：`configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml`
- Trainer：`src.rtdetr_lrs_system.TRAINER_TYPES["i"]`
- 训练要求：必须从 FDR 初始状态重新训练，禁止续训旧 Full checkpoint。

当前 AC-BPDD 只蒸馏在浅层与未来层保持相同
`(batch, query, ground-truth)` 匹配的分布，权重固定为 `0.15`。FDR 解码在
Integral 后执行成对合法几何投影，禁止负宽高，但不修改 stock GIoU、AMP、
优化器或学习率计划。

## 必要输入

1. UAVDT 数据 YAML，必须包含非空 `train`、`val`、连续 `names`；`nc` 若存在必须与 `names` 一致。
2. 已完成 UAVDT baseline 的 `args.yaml`。除运行身份和路径字段外，Full 将继承其中全部训练参数。
3. FDR 初始状态 `.pt`，由仓库安全的 weights-only 加载器验证。
4. 一个新的输出目录。

## 先执行 dry-run

```bash
python scripts/train_uavdt_full.py \
  --data-yaml /absolute/path/uavdt.yaml \
  --baseline-args /absolute/path/baseline/args.yaml \
  --initial-state /absolute/path/initial-state.pt \
  --output-root /absolute/path/runs \
  --dry-run
```

检查打印和 `runs/authority/*.json` 中的：

- `method == lrs_fdr_ac_bpdd_fia`；
- `arm == i`；
- config、baseline args、data YAML、initial-state 和 source 哈希；
- `epochs`、`seed`、`batch`、`imgsz`、优化器、学习率及增强参数与 baseline `args.yaml` 一致；
- `resume` 不存在且 `exist_ok == false`。

## 正式训练

确认 dry-run 后，使用完全相同的命令，仅删除 `--dry-run`：

```bash
python scripts/train_uavdt_full.py \
  --data-yaml /absolute/path/uavdt.yaml \
  --baseline-args /absolute/path/baseline/args.yaml \
  --initial-state /absolute/path/initial-state.pt \
  --output-root /absolute/path/runs
```

## 训练验收

- 不出现负 `giou_loss`、非有限梯度或 AMP scale 从 128 降到 64；
- 解码后的宽高始终为正；
- 记录原始横向/纵向非法 extent 数量及最小值，防止投影掩盖持续崩塌；
- AC-BPDD 的 stable-match ratio、active-edge ratio 和 loss 必须非零，才可声称 BPDD 实际参与；
- 代码稳定只属于工程验收。三模块内部消融成功仍要求同协议 Full best mAP50-95 高于 `LRS-FDR+FIA`。
