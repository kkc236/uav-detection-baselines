# I-TBER v1.1 裸机部署与恢复指南

本指南只适用于已冻结的 I-TBER v1.1 方案。服务器端统一使用 `/data/uav`，不得把运行目录、数据、checkpoint 或 token 放在用户主目录。正式实验所需最小可用空间为至少 80 GiB。

## 1. 授权边界与 SSH host key

任何会改变远端状态的命令，都只能在用户明确提供并授权服务器 endpoint 之后执行。第一次连接先只读取得 SSH host key 指纹，通过独立渠道确认后固定保存；后续连接若指纹改变，立即停止，不能使用自动接受新 key 的方式绕过。

```bash
ssh-keyscan -t ed25519 SERVER_IP 2>/dev/null | ssh-keygen -lf -
```

确认后的指纹应写入本次交接记录。密码、GitHub token 和 SSH 私钥不得进入仓库、命令行参数、日志或 manifest。

## 2. 目录布局

```text
/data/uav/
├── source/uav-detection-baselines
├── datasets/VisDrone
├── weights/matched_baseline
├── venvs/itber-v1.1
├── staging/itber-v1.1-wheelhouse
├── cache/itber-v1.1
├── runs/itber-v1.1
├── logs
├── config
├── deploy/markers
└── HANDOFFS/secrets
```

## 3. 源码和依赖传输

源码优先从已验证的 Git commit 克隆；GitHub 链路不稳定时，在可信本地生成 `git bundle` 后传输。不得传输未提交的工作树快照冒充版本化源码。

```bash
git clone https://github.com/kkc236/uav-detection-baselines.git /data/uav/source/uav-detection-baselines
git -C /data/uav/source/uav-detection-baselines checkout COMMIT_SHA
```

或者：

```bash
rsync -avP repository.bundle ubuntu@SERVER_IP:/data/uav/staging/
git clone /data/uav/staging/repository.bundle /data/uav/source/uav-detection-baselines
git -C /data/uav/source/uav-detection-baselines checkout COMMIT_SHA
```

wheelhouse 可在兼容 Linux 主机上提前生成并用 `rsync` 上传。上传后必须校验 manifest 中的 bytes 与 SHA256：

```bash
/data/uav/venvs/itber-v1.1/bin/python \
  /data/uav/source/uav-detection-baselines/deploy/itber/verify_bundle.py \
  --root /data/uav/staging/itber-transfer \
  --manifest /data/uav/staging/itber-transfer-manifest.json
```

## 4. baseline 和 VisDrone 权威

正式起点只能是：

```text
/data/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt
SHA256: 54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B
```

数据目录只能包含同一份 VisDrone train/val：训练 6471 张、验证 548 张、10 类；数据集 SHA256 为 `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`，固定 647 张子集 SHA256 为 `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`。

建议从权威存储节点用断点续传方式复制：

```bash
rsync -avP --partial SOURCE:/data/uav/datasets/VisDrone/ /data/uav/datasets/VisDrone/
rsync -avP --partial SOURCE:/data/uav/weights/matched_baseline/ /data/uav/weights/matched_baseline/
sha256sum /data/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt
```

复制完成后必须重新计算数据集签名和文件 SHA256，不能只比较文件名或大小。

## 5. GitHub 凭据

配置模板位于 `deploy/itber/publication.env.template`。token 只允许通过交互式隐藏输入写入指定文件：

```bash
install -d -m 700 /data/uav/HANDOFFS/secrets
read -rsp "GitHub token: " token_value; printf '\n'
umask 077
printf '%s' "$token_value" > /data/uav/HANDOFFS/secrets/github_token
unset token_value
chmod 600 /data/uav/HANDOFFS/secrets/github_token
```

日志和报告只能记录 token 文件路径，不能记录 token 内容。

## 6. 主机审计和 bootstrap

```bash
bash /data/uav/source/uav-detection-baselines/deploy/itber/verify_host.sh
bash /data/uav/source/uav-detection-baselines/deploy/itber/bootstrap_ubuntu.sh \
  /data/uav/source/uav-detection-baselines
```

冻结环境要求 RTX 4090、驱动 550.142、Python 3.10.12、PyTorch 2.5.1+cu121、Torchvision 0.20.1+cu121、CUDA 12.1 和 Ultralytics 8.4.90。任一字段不符，Gate 0 状态只能是 `engineering_invalid`，不得启动 Probe 或训练。

## 7. Gate 0

每次使用新的服务器、重新传输 baseline/data、重建 venv 或切换源码 commit，都要创建一个新的不可变 Gate 0 报告路径：

```bash
/data/uav/venvs/itber-v1.1/bin/python \
  /data/uav/source/uav-detection-baselines/scripts/run_itber_canary.py \
  --baseline-checkpoint /data/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt \
  --dataset-root /data/uav/datasets/VisDrone \
  --output /data/uav/runs/itber-v1.1/gate0/attempt-001.json \
  --device 0
```

只有报告为 `passed` 才能继续。失败报告永久保留，修复后使用 `attempt-002.json`，不得覆盖旧报告。

## 8. P0-P3 Probe 与筛选顺序

Gate 0 通过后才生成固定 evidence cache，并按同容量、同私有初始化运行 P0、P1、P2、P3，各 12 epoch。Probe 只用于信息量判断；通过后必须在固定 647 张子集上 fresh 运行 Gate 2，不能把 cache checkpoint 续训成正式模型。Gate 2 通过后，再在完整 6471/548 数据上 fresh 运行私有 30 epoch；同 checkpoint 的 stock/refined 是主比较。

实际命令由 Probe 训练实现提供，启动前必须再次检查 `docs/superpowers/specs/2026-08-01-i-tber-v1-1-design.md` 中的 P0-P3、Gate 1、Gate 2 和 formal 阈值没有漂移。

## 9. 每 epoch 发布、监控和恢复

每个 epoch 都必须先完成 checkpoint、轻量指标、manifest 的远端上传与 SHA 验证，再把 publication ledger 记为 `verified=true`。本地和 GitHub 滚动保留最近 3 个私有 checkpoint，所有轻量历史永久保留。训练日志、GPU 状态和 publication ledger 分开写入 `/data/uav/logs` 与对应 run 目录。

监控至少检查：进程仍存活、GPU 利用率与显存、epoch 连续性、loss 有限、detector SHA 不变、publication ledger 无缺口、最近 checkpoint 能独立加载评估。

恢复命令只接受与 baseline SHA、cache manifest、probe/stage、seed、设计版本完全一致的最高已验证 checkpoint：

```bash
/data/uav/venvs/itber-v1.1/bin/python \
  /data/uav/source/uav-detection-baselines/scripts/restore_itber_checkpoint.py \
  --run-dir /data/uav/runs/itber-v1.1/RUN_ID \
  --token-file /data/uav/HANDOFFS/secrets/github_token
```

恢复后先单独评估 checkpoint，再从下一个 epoch 继续。任何缺失 epoch、SHA 不一致或 metadata 漂移都按工程失败处理，不得跳过后解释为科学结果。
