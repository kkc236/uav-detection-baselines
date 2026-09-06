# LRS 系统 v2：本地验证与服务器交接

当前代码分支：`codex/no-server-correctness-v2`。
方法修订：`v2-fp32-extent-logspace-ac-bpdd`。
本页覆盖此前 v1 三臂运行手册的当前操作说明；旧文档只作版本追踪。

## 本地先验收

在仓库根目录执行：

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python -m pytest tests/test_no_server_correctness.py tests/test_fdr_feasible_geometry.py tests/test_bpdd_loss.py tests/test_lrs_system_local_execution.py tests/test_lrs_system_launcher.py tests/test_uavdt_full_launcher.py -q -p no:cacheprovider --junitxml=local-validation/local.xml
```

本机验证环境为 Python 3.10、PyTorch 2.13.0+cpu、Ultralytics 8.4.90。
这条命令执行合成数值测试、完整图 FP32/CPU BF16 前反向和 launcher 单元测试。
本机 CPU 纯 FP16 的 stock grid_sample 已复现有限输入产生 NaN，当前 predict
入口明确拒绝该模式，使用 FP32 或 CPU BF16 autocast；不能把这项保护测试写成
“FP16 推理通过”。CUDA FP16 不受此 CPU 保护影响，但尚未经本机实测。
正式 dataset authority 没有被降低；launcher 单元测试的临时输入不能替代真实
VisDrone SHA-256 或真实初始化权重检查。

## 四臂定义

| arm | 定义 | 默认运行名 |
| --- | --- | --- |
| f | LRS-FDR + v2 geometry | formal-seed0-lrs_fdr_feasible-v2 |
| g | f + AC-BPDD | formal-seed0-lrs_fdr_ac_bpdd-v2 |
| h | f + FIA | formal-seed0-lrs_fdr_fia-v2 |
| i | f + AC-BPDD + FIA | formal-seed0-lrs_fdr_ac_bpdd_fia-v2 |

`g-f`、`h-f`、`i-h`、`i-g` 分别给出条件效应；交互是 `(i-h)-(g-f)`。
四臂同 commit、同 data signature、同共享初始化、同正式配置、同评估器。
f 是新代码基线，旧 LRS Formal100 不可直接填入 f。若暂时只跑一对，f/g 或
h/i 只能回答对应条件问题，不构成完整系统或交互证据。

## 服务器取得算力后

沿用被比较 baseline 的环境与正式协议，先验证 CUDA/AMP 与真实数据管线。
不要根据本次 CPU 环境擅自更换正式 PyTorch/CUDA 版本。使用下列入口；路径变量
分别指实际 VisDrone 根目录、已验证初始权重文件和新的实验输出目录：

```bash
git clone --branch codex/no-server-correctness-v2 https://github.com/kkc236/uav-detection-baselines.git
cd uav-detection-baselines
git rev-parse HEAD
python scripts/train_visdrone_lrs_system.py --arm f --dataset-root "$VISDRONE_ROOT" --initial-state "$INITIAL_STATE" --output-root "$RUNS_ROOT" --dry-run
```

对 f/g/h/i 分别执行 dry-run，核对 authority 中的 `method_revision`、source、
config、initial_state 和 dataset SHA。完成实际 CUDA 前反向预检后，去掉
`--dry-run` 才会启动 Formal100。默认 v2 名称区分历史 v1；入口不支持 resume，
F/G/H/I trainer 均拒绝 checkpoint weights 作为重新训练的捷径。

每个 run 产生 `full-runtime.jsonl`，记录 BPDD coverage、损失均值、raw extent
非法计数和最终最小宽高。null 表示没有取得观测，不能当成零。BPDD 的均值是
按观测 batch 平均，不能解释为按全部目标加权的总体比例；梯度字段来自最后一次
可用 trainer 观测，不能单独证明全程梯度安全。

UAVDT 入口仍为 `scripts/train_uavdt_full.py`，继承该数据集的 baseline args；
默认 Full 名称升级到 v2，并写入同一方法修订。它只支持 Baseline/Full 的整体
比较，不声称验证了全部子模块泛化。

## 证据边界

本次交付证明本地执行与数值反例已测试；Formal100、CUDA 和数据集精度仍待验证。
没有 server 时可以整理现有论文和复现包，不填入推测 AP，不新增 PR-IRA/FIA
结构变体，也不把极小单 seed 正差值称为统计显著。
