# EQuAL Formal100 实验与论文材料交接文档

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-21T11:51:38+08:00
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1
- Upstream Dependencies: integrated implementation at `fca77636`; paper-facing rename at `65fe582b`

> 本文是执行交接，不是结果报告。`UNVERIFIED` 表示 EQuAL 仍在训练，尚未取得
> 100-epoch best-checkpoint val/test 终评结果。

## 1. 一句话目标

完成 EQuAL（Edge-adaptive Query-Aligned Localization）的 100-epoch seed0
训练，用 `best.pt` 在同一协议下分别评估 VisDrone val/test，并与已登记的
AP-FDR 做逐项比较；只有达到既定门槛，才用 EQuAL 替换论文中的 AP-FDR。

## 2. 当前状态（交接起点）

| 项目 | 当前值 | 作用 |
|---|---|---|
| 论文方法名 | `EQuAL` | 正文、图表和后续配置使用的规范名称 |
| 运行时旧标识 | `ACE-FDR` / `ace_fdr` | 训练启动后不可改写的原始证据标识 |
| 运行服务器 | `36.103.234.61:22`，用户 `ubuntu` | 仅记录连接目标；密码/PAT 不进入仓库 |
| 训练 PID | `83869` | 只用于查看该进程是否仍存活 |
| 启动时间 | 2026-08-21 10:56:16 +08:00 | 估算耗时与排查异常 |
| 最近复核（11:50 CST） | 已完成 7/100 epoch；GPU 正常；0 个 fatal pattern | 只是进度，不是最终结果 |
| 训练源码 | `fca7763679b6e10ed68f98971a362a054ecd4853` | 当前 checkpoint 的真实源码身份 |
| 规范命名分支 | `codex/ap-fdr-integrated-redesign` | EQuAL 配置、入口、设计和映射所在分支 |
| 最低规范命名提交 | `65fe582bacf941407b104911a2804589e64604df` | 含 EQuAL 接口与 alias；后续交接提交位于其上 |
| 代码回归 | `97 passed` | 证明实现与既有 FDR/BPDD 接口兼容；不代表效果有效 |

训练目录：

```text
/data/uav/runs/ace-fdr-formal100-20260821/formal-seed0-ace-fdr-v1
```

关键文件：

```text
/home/ubuntu/ace-fdr/ace-fdr-formal100.pid
/home/ubuntu/ace-fdr/logs/ace-fdr-formal100.log
/data/uav/runs/ace-fdr-formal100-20260821/authority/formal-seed0-ace-fdr-v1.json
/data/uav/runs/ace-fdr-formal100-20260821/formal-seed0-ace-fdr-v1/results.csv
/data/uav/runs/ace-fdr-formal100-20260821/formal-seed0-ace-fdr-v1/weights/best.pt
/data/uav/runs/ace-fdr-formal100-20260821/formal-seed0-ace-fdr-v1/weights/last.pt
```

关键权威哈希：

| 对象 | SHA-256 |
|---|---|
| 训练源码树 | `2B62ABF58676B331A026C092E1EB65DAC2FEA5084EEF0B726CB0552F348BB4644` |
| 运行时配置 | `B38EB6932F1CE115C780C0F2777FA6DF465013E70D417AE996B482BE8A96963E` |
| 初始状态 | `51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D` |
| VisDrone 数据集 | `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB` |

## 3. 名称映射规则

当前进程启动时方法还叫 ACE-FDR，后来才确定论文名称 EQuAL。两者是同一套
算法语义，不是两个实验：

```text
raw runtime method: ace_fdr / ACE-FDR
paper method:       equal / EQuAL
```

必须遵守：

1. 不为改名重启训练。
2. 不重命名服务器 run 目录、原始 authority、checkpoint 或日志。
3. 导入材料时同时保存 raw runtime id 和 `paper_method=equal`。
4. 论文只显示 EQuAL；审计清单保留映射文件
   `docs/evidence/equal-runtime-alias.json`。

## 4. 后续步骤总览

| 步骤 | 做什么 | 为什么要做 | 主要产物 | 当前状态 |
|---:|---|---|---|---|
| 1 | 监控训练 | 及时发现崩溃、卡死、显存异常 | 进度与异常记录 | 进行中 |
| 2 | 补自动收尾器 | 训练完成后自动验收、评估和私密上传 | watcher/finalizer 状态文件 | 待做 |
| 3 | 验收 100 epoch | 排除早停、不连续和残缺 checkpoint | completion manifest | 待训练完成 |
| 4 | 冻结原始证据 | 防止之后误覆盖或混用权重 | SHA-256 清单、归档 | 待训练完成 |
| 5 | 用 best.pt 跑 val | 得到模型选择划分的完整精确指标 | `equal-best-val.json` | 待训练完成 |
| 6 | 用同一 best.pt 跑 test | 检查独立划分上的泛化与 AP75 | `equal-best-test.json` | 待 val 完成 |
| 7 | 与 AP-FDR 对比 | 判断 EQuAL 是否真的值得替换原方案 | 对比表和门检结论 | 待评估完成 |
| 8 | 决定是否重跑组合 | EQuAL 改变 FDR 训练路径，旧组合不能直接改名 | go/no-go 记录 | 待门检 |
| 9 | 测复杂度 | 证明新设计不增加推理负担 | Params/GFLOPs/Latency 表 | 待采用后 |
| 10 | 更新 material/GitHub | 形成可审计论文材料 | Release、JSON、Markdown | 待终评 |
| 11 | 更新论文叙事 | 只把通过门检的结果写入正文 | 方法、消融、结论段 | 最后执行 |

## 5. 步骤 1：训练期间监控

### 这一步干什么

只观察 PID、GPU、`results.csv` 和错误日志，确认训练持续前进。不要因为早期
mAP 很低而停止；从统一初始状态训练时，前几轮数值不代表最终性能。

### 登录后检查命令

```bash
ace_pid=$(cat /home/ubuntu/ace-fdr/ace-fdr-formal100.pid)
ps -p "$ace_pid" -o pid=,lstart=,etime=,stat=,%cpu=,%mem=
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
tail -5 /data/uav/runs/ace-fdr-formal100-20260821/formal-seed0-ace-fdr-v1/results.csv
grep -Ein 'Traceback|CUDA out of memory|RuntimeError' /home/ubuntu/ace-fdr/logs/ace-fdr-formal100.log | tail
```

### 正常标准

- PID `83869` 存活；
- GPU 上存在该 PID；
- `results.csv` 的 epoch 行数持续增加；
- 没有 Traceback、CUDA OOM 或 RuntimeError；
- 训练 loss 为有限数。

### 异常时怎么办

- 只记录退出时间、最后 100 行日志、最后完成 epoch 和现存 checkpoint。
- 不自动重试，不从不明 checkpoint 续训，不删除残留目录。
- 先判断 `last.pt` 是否可恢复，再由负责人决定是否续训。

按当前 7 轮约 54 分钟的速度，100 轮粗略需要 12–14 小时；该 ETA 只用于
安排检查，不是完成保证。

## 6. 步骤 2：补自动收尾器

### 这一步干什么

当前训练已经后台运行，但 EQuAL 专用的“训练完成后自动 val/test + 私密上传”
尚未登记为已启用。下一执行者应在训练结束前补一个独立 watcher，不能修改
训练进程。

### watcher 必须做的事

1. 每 60 秒检查一次 PID 和 `results.csv` 行数。
2. 只有连续 100 个 epoch、`best.pt`/`last.pt` 均存在时才进入收尾。
3. 若进程异常退出，写 `equal-finalization-failed.json`，不自动重跑。
4. 依次执行步骤 3–6。
5. 将制品上传到私密仓库
   `kkc236/icassp2027-fdr-bpdd-fia-material` 的独立 Release。
6. 上传后用 GitHub API 复核每个资产的名称和字节数。
7. 不执行 `shutdown` 或 `poweroff`。

建议 Release tag：

```text
equal-seed0-formal100-20260821
```

## 7. 步骤 3：100-epoch 完成验收

### 这一步干什么

区分“训练进程结束”和“实验完整完成”。只有完整训练才允许进入论文终评。

### 必须同时满足

- `results.csv` 恰好 100 行数据；
- epoch 序列连续，为 `1..100` 或框架等价的 `0..99`；
- `best.pt`、`last.pt`、`args.yaml` 和训练日志都存在且非空；
- `last.pt` 对应最终轮或框架 strip 后的合法最终 checkpoint；
- 训练日志无未处理异常；
- authority 中的源码、配置、initial-state、dataset 与本文件一致。

### 产物

生成 `equal-completion-manifest.json`，记录完成轮数、best epoch、last epoch、
训练结束时间和所有输入身份。

## 8. 步骤 4：冻结原始证据

### 这一步干什么

在评估和上传前先计算哈希，防止后续同名文件被覆盖后仍被误认为原实验。

### 需要冻结的文件

- `best.pt`
- `last.pt`
- `results.csv`
- `args.yaml`
- 训练日志
- launch authority JSON
- 运行时 YAML
- EQuAL 名称映射 JSON
- 收尾/评估脚本自身

### 规则

- 使用 SHA-256；
- manifest 中保存相对路径、字节数和哈希；
- 原文件只读保存；
- 后续修订通过新增 amendment，不覆盖原 manifest。

## 9. 步骤 5：用 best.pt 统一评估 val

### 这一步干什么

训练日志中的 best-val mAP 用于 checkpoint 选择，但论文表需要一次独立、统一
评估来补齐 Precision、Recall、AP50、AP75、mAP50-95、分尺度和逐类别结果。

### 固定协议

```text
split=val
images=548
imgsz=640
batch=8
workers=8
conf=0.001
max_det=300
nms=False
cache=False
half=False
rect=False
checkpoint=best.pt
```

### 产物

`equal-best-val.json`，至少包含：

- P、R、F1、AP50、AP75、mAP50-95；
- tiny/small/medium/large；
- VisDrone 10 类逐类指标；
- checkpoint SHA-256；
- evaluator 参数和处理图片数 548。

## 10. 步骤 6：用同一个 best.pt 统一评估 test

### 这一步干什么

检查 EQuAL 的提升是否只出现在 val。不能重新按 test 选择 checkpoint，必须沿用
步骤 5 的同一个 `best.pt`。

### 固定协议

除 `split=test` 和图片数外，其余参数与 val 完全一致：

```text
split=test
images=1610
instances=75102
imgsz=640
batch=8
workers=8
conf=0.001
max_det=300
nms=False
cache=False
half=False
rect=False
checkpoint=the_same_best.pt
```

### 产物

`equal-best-test.json`，字段与 val 对齐，并明确记录处理图片数 1610。test 只跑
一次，不能根据 test 指标换权重。

## 11. 步骤 7：与 AP-FDR 逐项比较

### 这一步干什么

回答本轮唯一科学问题：EQuAL 作为一个整体，是否优于原 AP-FDR。

### 正式参考基准

Val：

| 方法 | P | R | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| AP-FDR | 0.56911 | 0.49278 | 0.48468 | 0.29253 | 0.28966 |

Test：

| 方法 | P | R | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| AP-FDR | 0.503401 | 0.431677 | 0.398340 | 0.228903 | 0.228001 |

每个差值统一计算为：

```text
delta_pp = (EQuAL - AP-FDR) * 100
```

### 已冻结的采用门槛

1. EQuAL val mAP50-95 必须严格高于 `0.2896597491`；
2. EQuAL val AP75 不得低于 `0.29253`；
3. test 作为泛化核验单列，不得与 val 混成一张“统一增益”；
4. 若 test mAP/AP75 反向，不能写“稳定提升”。

当前比较属于同训练协议、同 seed、同 initial-state 的登记历史 authority 对照；
它不是两臂在同一新 source commit 下同时重跑的 fresh paired Formal100。论文材料
必须保留这个证据级别说明。

## 12. 步骤 8：Go/No-Go 与后续组合实验

### 情况 A：EQuAL 通过 val 门槛，test 也正向

采用 EQuAL 作为新的第一方法。接下来依优先级重跑：

1. `EQuAL + BPDD`：确认主要原创训练模块与 EQuAL 兼容；
2. `EQuAL + FIA`：确认推理期高分辨率增强与 EQuAL 兼容；
3. `EQuAL + BPDD + FIA`：生成最终三模块主结果。

原因：旧 `AP-FDR+BPDD/FIA` checkpoint 是在旧 FDR 训练路径下得到的，不能只改
名字当成 EQuAL 组合结果。

### 情况 B：val 正向但 test 持平或反向

EQuAL 只作为探索结果，不立即替换 AP-FDR。先检查分尺度/逐类退化和评估身份，
不要马上投入三条组合 Formal100。

### 情况 C：val mAP 不升或 AP75 下降

停止 EQuAL 路线，保留原 AP-FDR。这样既有 FDR+BPDD、FDR+FIA 和三模块结果
仍可继续使用，不为失败设计重跑整套组合。

## 13. 步骤 9：统一复杂度测试

### 这一步干什么

证明 EQuAL 的提升不是通过新增推理参数或分支换来的。

### 最少报告

| 方法 | Params | GFLOPs@640 | FP32 Latency/FPS | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| RT-DETR-L | 待统一复测 | 待统一复测 | 待统一复测 | 已有 | 已有 |
| AP-FDR | 33,156,614 | 108.2291 | 待统一复测 | 已有 | 已有 |
| EQuAL | 待统一复测 | 待统一复测 | 待统一复测 | 本轮 | 本轮 |
| EQuAL+BPDD | 仅通过后测 | 仅通过后测 | 仅通过后测 | 仅通过后测 | 仅通过后测 |
| Full | 仅通过后测 | 仅通过后测 | 仅通过后测 | 仅通过后测 | 仅通过后测 |

EQuAL 的 edge-adaptive weighting 仅在训练损失中工作，理论上不新增推理参数和
FLOPs；仍必须用实际模型摘要和同机 latency 复核后才能写入论文。

## 14. 步骤 10：更新 material 仓库与 GitHub

### Release 必须包含

- `best.pt`、`last.pt`；
- `results.csv`、`args.yaml`、训练日志；
- runtime authority、运行配置和 EQuAL alias；
- completion manifest；
- `equal-best-val.json`、`equal-best-test.json`；
- 评估脚本及其哈希；
- publication manifest/status。

### 材料文件更新顺序

1. 先更新 machine-readable JSON；
2. 再更新 `RESULTS_ZH.md`；
3. 再更新 `MAIN_TABLE_ZH.md`；
4. 更新 split gain、消融表和论文 outline；
5. 最后运行完整 material integrity tests；
6. 通过后推送私密仓库。

不得在评估完成前用早期 epoch 数字覆盖正式结果，也不得把旧 ACE-FDR 路径从
原始证据中删除。

## 15. 步骤 11：论文写作落点

### EQuAL 的整体描述

```text
EQuAL uses the native RT-DETR decoder reference as a stable distribution
anchor, applies fine-grained distribution supervision only to clean matched
queries, and reallocates localization gradients according to the learning
difficulty of individual box edges.
```

### 贡献边界

- 明确写 `built upon / inspired by D-FINE`；
- D-FINE 的离散边界表示、non-uniform Integral 和基础 adjacent-bin FGL 要引用；
- EQuAL 的个人贡献写成：原生 query/reference 对齐、clean-query 监督组织和逐边
  自适应梯度分配构成的统一定位方法；
- 不把已经去掉的 preliminary reference 和额外 DN-FDR 监督写成有效创新；
- 不把方法改名本身当成原创证据，原创性由算法差异和实验结果支撑。

## 16. 完成定义

只有以下全部完成，EQuAL 任务才算闭环：

- [ ] 100 epochs 连续完成；
- [ ] best/last、日志、authority 和配置均已哈希冻结；
- [ ] 同一 best.pt 完成 val 和 test 独立评估；
- [ ] P/R/F1/AP50/AP75/mAP、尺度和类别指标齐全；
- [ ] 与 AP-FDR 的差值和证据级别已登记；
- [ ] Go/No-Go 已明确；
- [ ] 私密 GitHub Release 资产已逐项验证；
- [ ] material JSON、Markdown 和论文表格一致；
- [ ] 若采用 EQuAL，相关 BPDD/FIA 组合已重新训练，未挪用旧权重；
- [ ] 复杂度和延迟已在统一环境下复测；
- [ ] 完整材料测试通过。

## 17. 交接时最容易犯的五个错误

1. 为了改名重启训练，浪费当前有效 run。
2. 用 `last.pt` 代替 `best.pt` 做论文 val/test。
3. 看 test 后重新选 checkpoint，造成 test 泄漏。
4. 把旧 AP-FDR+BPDD/FIA 数字直接改名成 EQuAL 组合结果。
5. 只看 mAP，忽略 EQuAL 已冻结的 AP75 不退化门槛。
