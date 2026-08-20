# AP-FDR 自动发布设计

## 目标

在 `no_preliminary_reference` 与 `no_dn_fdr` 两项 seed0 Formal100 训练均成功结束后，将可复核的训练制品自动发布到私密材料仓库 `kkc236/icassp2027-fdr-bpdd-fia-material` 的独立 GitHub Release。发布过程不得干扰训练，也不得自动改写论文正式结果表。

## 发布边界

- 触发条件是 `/home/ubuntu/ap-fdr-ablation/all.completed` 存在。
- 两个运行目录都必须有连续 100 行 `results.csv`、`args.yaml`、`weights/best.pt` 与 `weights/last.pt`。
- 两个训练日志、两个 dry-run JSON、两个 authority JSON 必须存在。
- 任一门禁失败时，记录错误并退出；不得创建或更新 Release。
- 发布目标固定为私密材料仓库的 tag `ap-fdr-internal-ablation-seed0-20260820`。
- 不提交大权重到 Git 历史；模型作为 Release assets 上传。
- 不自动写入 `MAIN_TABLE_ZH.md`、`RESULTS_ZH.md` 或机器结果 JSON。

## 制品结构

每个变体生成一个压缩包，包含：

- `best.pt` 与 `last.pt`
- `results.csv`
- `args.yaml`
- 完整训练日志
- 对应 dry-run JSON 与 authority JSON
- `artifact-manifest.json`，记录源提交、运行名、epoch 数、文件字节数和 SHA-256

此外上传顶层 `publication-manifest.json`，绑定两个变体、Release tag、源提交 `ebb349aeb2cf092d4880751e165e22614c3c9d8c` 与所有资产摘要。

## 架构

单独启动低开销 watcher，不修改正在运行的训练监督器。watcher 轮询完成标记；触发后调用纯 Python 发布脚本完成门禁、归档、摘要计算、GitHub Release 创建/更新、资产上传和远端尺寸核验。上传使用幂等文件名：同名同尺寸资产跳过，同名不同尺寸资产先删除再重传。网络错误采用有限重试，最终状态写入服务器日志和 JSON。

## 凭据与故障处理

- GitHub token 保存于权限 `0600` 的服务器文件，文件路径不写入仓库，token 不进入命令行或日志。
- 上传成功并验证后删除 token 文件；失败时保留 token 以便自动重试，服务器目录仍仅 `ubuntu` 用户可读。
- watcher 使用 `nohup` 脱离 SSH，PID、标准输出和最终状态均落盘。
- 训练失败、epoch 不足、缺文件、哈希失败或 GitHub 验证失败都不得生成成功标记。
- 服务器保持开机，不执行关机命令。

## 验证

- 单元测试覆盖：100 epoch 门禁、缺失文件拒绝、确定性 manifest、同尺寸幂等上传、不同尺寸替换、API 失败重试边界。
- 部署前用假运行目录验证失败门禁，再以 `--check-only` 对当前真实目录验证“尚未完成”的安全状态。
- 部署后确认 watcher 存活、训练主进程未变、token 文件权限为 `0600`，并确认 GitHub API 对目标私密仓库可鉴权。
