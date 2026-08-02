# IBER-BE v1.0 裸服务器部署与运行指南

本指南只适用于 trajectory-free 的 IBER-BE v1.0。它不授权复用、覆盖或解释任何 I-TBER 路径和结果。所有服务器资产统一位于 `/data/uav`；源代码、缓存、运行目录和结果目录都必须绑定已推送并核验的 source commit。

## 1. 冻结的执行合同

唯一允许的执行环境如下：

| 项目 | 固定值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| 上报显存 | 49140 MiB |
| baseline 历史驱动 | 550.142 |
| 本次执行驱动 | 570.133.07 |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| CUDA | 12.1 |
| Ultralytics | 8.4.90 |

驱动和上报显存是已批准的 runtime amendment；主机与运行库全部吻合时状态必须是 `passed_with_runtime_amendment`。除此以外的偏差一律记为 `engineering_invalid`，不能降级成警告后继续。

科学资产必须同时匹配：

- baseline SHA256：`54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B`；
- VisDrone train/val SHA256：`FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`，6471/548 张、10 类；
- 固定 647 张子集 SHA256：`52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`；
- GitHub 上 `codex/iber-be` 的 40 字符 commit 必须与服务器 checkout 的 `git rev-parse HEAD` 完全一致。

## 2. SSH 与凭据边界

首次连接先人工核对 SSH host key 指纹，再写入专用 `known_hosts`。禁止用关闭 host-key 校验的方式换取自动化。

GitHub token 只允许保存在：

```text
/data/uav/HANDOFFS/secrets/github_token
```

执行 `chmod 600 /data/uav/HANDOFFS/secrets/github_token`，并用 `stat -c %a` 确认结果严格为 `600`。实际运行用的 `/data/uav/config/iber-be-v1/publication-screen.json` 同样必须 `chmod 600`；它只能保存 token 文件路径，不能保存 token 内容。token 不能出现在 Git remote、命令参数、JSON、日志、checkpoint、manifest 或 shell 历史中。代码仓库和部署模板不包含任何凭据。

该服务器的 GitHub API 预检为 HTTP 200，但 Git smart HTTP 必须固定 `http.version=HTTP/1.1`。bootstrap 对公开 source remote 的 clone/fetch 显式使用 `git -c http.version=HTTP/1.1`；这是网络兼容回退，不是鉴权绕过。它还生成 mode-600 的 `/data/uav/config/iber-be-v1/git-http.env`，内容仅为 `GIT_CONFIG_COUNT`、`GIT_CONFIG_KEY_0=http.version` 和 `GIT_CONFIG_VALUE_0=HTTP/1.1`，不含凭据。禁止把 token 拼进 remote URL，发布仍只从 mode-600 token 文件读取。

## 3. 本地审计与 bundle 校验

在传输前执行：

```bash
python scripts/audit_iber_deployment.py \
  --root /path/to/source \
  --source-commit 0123456789abcdef0123456789abcdef01234567
```

本地通过只会返回 `ready_waiting_for_server`；远端字段仍应为 `unresolved`。这不是主机通过证明。

将 `artifact-manifest.template.json` 复制为运行 manifest，替换 source commit、补齐每个文件的字节数和 SHA256 后执行：

```bash
python deploy/iber/verify_bundle.py \
  --root /data/uav/staging/iber-be-v1-bundle \
  --manifest /data/uav/staging/iber-be-v1-bundle/artifact-manifest.json \
  --source-commit 0123456789abcdef0123456789abcdef01234567
```

校验器拒绝目录穿越、符号链接、哈希漂移、source commit 漂移，以及任何旧 I-TBER 结果路径。

## 4. mirror-first 裸机安装

先运行只读主机检查。如果虚拟环境尚未创建，先执行 bootstrap，再重新执行主机检查以验证 Python 包：

```bash
bash deploy/iber/bootstrap_ubuntu.sh \
  0123456789abcdef0123456789abcdef01234567
bash deploy/iber/verify_host.sh
```

bootstrap 优先使用阿里云 PyPI 镜像，PyTorch CUDA 12.1 wheel 使用官方索引作为受控补充/回退。可预先在前台生成 wheelhouse：

```bash
bash deploy/iber/build_wheelhouse.sh
```

脚本采用内容哈希 marker，重复执行不会把另一份依赖或另一 commit 冒充为已完成。若要远程长时间执行，可由外层 `nohup` 包装，并把日志写到 `/data/uav/logs`；脚本本身保持 foreground，便于检查退出码。

## 5. immutable source checkout 与唯一运行根

部署 commit 的前 12 位记作 `<shortsha>`。唯一允许的路径是：

```text
/data/uav/source/uav-detection-baselines-<shortsha>
/data/uav/cache/iber-be-v1-<shortsha>
/data/uav/runs/iber-be-v1/<shortsha>-seed0-amended
/data/uav/results/iber-be-v1-<shortsha>
/data/uav/logs/iber-be-v1-<shortsha>-pipeline.log
```

源目录是 detached、只读的 immutable source checkout。运行根不得复用；若对应路径已有不一致 manifest、PID 或 checkpoint，必须保留现场并改用新的已验证 commit，不能清空旧目录后重跑。不得创建 `/data/uav/cache/itber-*`、`/data/uav/runs/itber-*` 或 `/data/uav/results/itber-*`。

## 6. 启动前门禁

按顺序完成：

1. `verify_host.sh` 返回 `passed_with_runtime_amendment`；
2. source commit、baseline、数据集、子集和类别映射全部匹配；
3. `scripts/run_iber_canary.py` 的 Gate-0 工程证明通过；
4. 三次 stock authority 逐值完全一致；
5. evidence cache 绑定相同 source commit；
6. B0–B3 Probe 完整结束，并由冻结规则产生 Gate-1 decision。

**禁止绕过 Gate-1。** 不得手改 `pipeline-state.json`、伪造 decision、直接调用训练脚本启动 screen30，也不得把 Probe checkpoint 续训成正式模型。Gate-1 为 `scientific_failed` 时，必须发布证据并停止；只有通过时 supervisor 才能 fresh 启动 30 epoch。

准备 `publication-screen.json` 时以 `publication-screen.template.json` 为基础，仅替换已验证的 source commit 与 `<shortsha>`。随后由状态机启动：

```bash
export YOLO_CONFIG_DIR=/data/uav/config/Ultralytics
source /data/uav/config/iber-be-v1/git-http.env
nohup /data/uav/venvs/iber-be-v1/bin/python \
  scripts/run_iber_pipeline.py \
  --baseline-checkpoint /data/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt \
  --dataset-root /data/uav/datasets/VisDrone \
  --run-root /data/uav/runs/iber-be-v1/<shortsha>-seed0-amended \
  --cache-root /data/uav/cache/iber-be-v1-<shortsha> \
  --publication-config /data/uav/config/iber-be-v1/publication-screen.json \
  --device 0 \
  > /data/uav/logs/iber-be-v1-<shortsha>-pipeline.log 2>&1 &
printf '%s\n' "$!" > /data/uav/runs/iber-be-v1/<shortsha>-seed0-amended/pipeline.pid
```

具体参数必须以 `scripts/run_iber_pipeline.py --help` 和已冻结的运行 manifest 为准；不要猜测或省略门禁参数。

## 7. 监控与每 epoch 发布

服务器付费期间持续按状态而不是固定 sleep 检查：

```bash
cat /data/uav/runs/iber-be-v1/<shortsha>-seed0-amended/pipeline-state.json
tail -n 200 /data/uav/logs/iber-be-v1-<shortsha>-pipeline.log
nvidia-smi
ps -fp "$(cat /data/uav/runs/iber-be-v1/<shortsha>-seed0-amended/pipeline.pid)"
```

必须依次看到 `authority -> gate0 -> stock_authority -> cache -> probe -> screen30 -> screen_decision`。每个 epoch 结束后由 `scripts/publish_iber_epoch.py` 上传 checkpoint/manifest 配对，远端逐字节与 SHA 验证、结果分支 commit 验证和 append-only ledger 成功后，才能进入下一 epoch。至少保留最近 3 个可恢复 checkpoint；远端验证失败属于工程故障，不能继续训练制造发布缺口。

不要启动第二个 supervisor。PID 存活且 `/proc/<pid>` 对应命令仍为当前 source/run root 时，只监控现有进程。

## 8. 恢复与故障分流

### 工程故障

`engineering_invalid`、网络中断、上传未验证、进程意外退出或本地 checkpoint 损坏时：

1. 保留日志、状态、ledger 和损坏文件，不删除现场；
2. 用 `scripts/restore_iber_checkpoint.py` 从最后一个远端已验证 checkpoint/manifest 原子恢复；
3. 复核 source/data/baseline/runtime authority；
4. 用同一个 run root 恢复未完成的连续 epoch，不重跑已发布 epoch；
5. 修复报告上传并验证后再继续。

### 科学停止

Gate-1 或 Gate-2 返回 `scientific_failed` 时不是工程故障。发布完整证据后停止，不调阈值、不挑最好 epoch、不绕过门禁。Gate-2 只评 epoch30，同 checkpoint 比较 stock/refined，并要求三次精确复评。

### 终态检查

screen30 结束后确认：30 条连续逐 epoch 记录、epoch30 checkpoint 身份、三次完全一致评估、Gate-2 decision、所有 authority 哈希、远端结果分支 commit，以及不存在遗留训练进程。无论通过或科学失败，都归档到独立的 `iber-be-v1-results`，绝不写入或删除旧 I-TBER 结果。
