import warnings, os, sys
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
warnings.filterwarnings('ignore')
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 自定义模块 FIA / FDRRTDETRDecoder 在 src/ 中，通过该 trainer 注册进 ultralytics。
# initial_state_path=None 即不加载任何预训练/初始权重，完全从头训练。
from src.rtdetr_fdr_bpdd_fia import FDRBPDDFIATrainer


if __name__ == '__main__':
    trainer = FDRBPDDFIATrainer(
        overrides={
            'model': 'configs/rtdetr-l-fdr-bpdd-fia.yaml',
            'data': 'configs/VisDrone.yaml',
            'cache': False,
            'imgsz': 640,
            'epochs': 100,
            'batch': 8,
            'workers': 4,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
            # 'device': '0,1',  # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
            # 'resume': '',  # last.pt path
            'patience': 0,  # 设置0代表不早提供，设置30代表精度持续30epoch没有比之前最高的高就早停
            'pretrained': False,  # 从头训练
            'optimizer': 'MuSGD',  # 该 trainer 强制要求 MuSGD（lr0=0.01, momentum=0.937, weight_decay=0.0005）
            'project': 'runs/train',
            'name': 'exp',
        },
        experiment_seed=0,
        initial_state_path=None,
    )
    trainer.train()
