# RA-GLGM v1.1 Full100 预注册

## 实验身份

- 基线：RT-DETR-L + FDR。
- 方法：RT-DETR-L + FDR + RA-GLGM v1.1；模块结构、损失、P3 融合位置和参数量不变。
- 目的：检验 v1.1 在完整训练集和更长训练周期下的收敛轨迹，不继承 Explore50 或 v1.2 权重。
- 证据属性：探索性。官方验证集在训练和每五轮锁定评估中重复使用，不作为一次性确认性检验。

## 数据与配对

- VisDrone train：6471 张。
- VisDrone val：官方 548 张，38,759 个目标。
- 两臂均从同一份字节完全一致的公开初始状态开始，seed=0。
- 单张物理 RTX 4090，先完整运行 Baseline，再完整运行 RA-GLGM；不使用 DDP，不跨 GPU 配对。

## 训练规格

- 输入：640x640。
- epoch：Baseline 100 + RA-GLGM 100。
- batch=8，nbs=64，梯度累积 8 次，有效 batch=64。该规格与既有 FDR/v1.1 协议一致，并经完整数据高密度样本审计后冻结。
- workers=8，AMP 开启，cache=false，deterministic=true。
- 禁用框架内部 OOM 自动降批；OOM 必须显式失败并建立新的实验 authority，不允许一臂内静默改变 batch。
- 优化器、学习率、增强、FDR/BPDD、Query 和 Hungarian 匹配保持既有冻结协议不变。
- 先对两臂各运行 Smoke2；任一臂显存、AMP、梯度或有限数值检查失败，则不得启动 Full100。改变 batch 必须形成新的预注册和新实验 authority，不能在一臂中途变更。

## 记录与评估

- 每个 epoch 记录训练验证指标、梯度、AMP、峰值显存、检查点哈希和本地队列记录。
- 独立锁定评估 epoch：5, 10, ..., 100。
- 每个锁定点对比 Precision、Recall、AP50、AP75、mAP50-95、AP-tiny、AP-small 和 10 类 AP。
- epoch100 为主要单点；最后五轮训练验证均值和 80-100 的五轮间隔轨迹用于判断后期稳定性。
- Best checkpoint 仅作补充，不替代 epoch100 结论。

## 工程门槛

- 两臂必须为同 GPU、同初始化、同数据、同 seed 和同训练协议。
- 每轮指标必须有限；AMP 不允许跳步；公共/FDR 梯度必须有限且非零，方法臂 RA 私有梯度还必须非零。
- 每个完成 epoch 恰有一个检查点和一个哈希绑定的本地队列记录。
- RA-GLGM 峰值显存必须低于 22 GiB；磁盘低于 20 GiB 可用空间时停止并审计，不自动删除其他实验。
- `.pt` 文件仅保留在服务器本地，未经明确批准不得发布。
