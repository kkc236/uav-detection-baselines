# FDR YAML 声明式模块最终兼容证据

本目录保存可进入 Git 的轻量机器证据。正式 checkpoint 因大小为
`200024985` bytes，不进入 Git；其公开下载、YAML overlay、Git bundle 和
哈希清单位于 GitHub Release：

<https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-yaml-declarative-v1>

正式 checkpoint authority：

- 文件：`fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt`
- SHA256：`c2f638744508adfe7b6c4a1ef3e08c503273f628062e4650ad59ffff4c6588c2`
- 原发布页：<https://github.com/kkc236/uav-detection-baselines/releases/tag/fdr-formal-d97e1eb7-live>

最终验证：

- 专项回归：`161 passed, 3 skipped`；
- 完整 FDR + 四个消融 YAML：全部严格加载；
- 每个配置：950 个 tensor、missing/unexpected `0/0`、有限输出 `[1,300,6]`；
- 旧格式 checkpoint：950/950 模型重建，恢复 8 个 MuSGD 参数组、581 个
  optimizer state、AMP scale 128、EMA updates 10556；
- 恢复后实际执行一次 `128x128` 前向、反向、MuSGD step 与 EMA update：
  loss 有限、梯度有限、EMA updates `10556 -> 10557`。

机器可读证据：

- `checkpoint-compatibility-all-configs.json`
- `legacy-resume-step.json`

可重复执行的验证入口：

- `scripts/verify_fdr_yaml_checkpoint.py`
- `scripts/verify_fdr_legacy_resume_step.py`
