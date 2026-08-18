import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.rtdetr_fdr import FDRControlTrainer, register_fdr_module


# Only register FIA for YAML parsing. FDRControlTrainer keeps the stock decoder
# while sharing the same MuSGD and fixed-AMP experiment protocol as FDRTrainer.
register_fdr_module()


if __name__ == "__main__":
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trainer = FDRControlTrainer(
        overrides={
            "model": "configs/rtdetr-l-fia.yaml",
            "data": "configs/VisDrone.yaml",
            "cache": False,
            "imgsz": 640,
            "epochs": 100,
            "batch": 8,
            "workers": 4,
            "device": "0",
            # "resume": "",
            "patience": 0,
            "pretrained": False,
            "optimizer": "MuSGD",
            "seed": 0,
            "deterministic": True,
            "save": True,
            "project": str(ROOT / "runs"),
            "name": f"fia-p3-only-seed0-{run_stamp}",
            "exist_ok": False,
        },
        initial_state_path=None,
    )
    trainer.train()
