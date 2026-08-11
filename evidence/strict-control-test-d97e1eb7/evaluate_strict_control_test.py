from __future__ import annotations
import json
import os
import sys
from pathlib import Path

SOURCE = Path('/data/uav/source/uav-detection-baselines-2997705c')
sys.path.insert(0, str(SOURCE))
from ultralytics.nn.tasks import RTDETRDetectionModel
from src.bpdd_formal_evaluation import (
    CachedScaleRTDETRValidator,
    load_exact_final_checkpoint,
    summarize_native_box_metrics,
    summarize_scale_metrics,
)
from src.lpr_protocol import CATEGORY_NAMES

CHECKPOINT = Path('/data/uav/runs/fdr-formal-control-d97e1eb7/formal-seed0-control-fdr-v1/weights/epoch99.pt')
EXPECTED_SHA = '9C242711F44B7E68B360AF904AB7C44F64505C7136B7E7F90481092AE3308AF7'
DATA = Path('/data/uav/evidence/strict-control-test-d97e1eb7/test-data.yaml')
TEST_AUTHORITY = Path('/data/uav/evidence/strict-control-test-d97e1eb7/test-authority.json')
OUT = Path('/data/uav/evidence/strict-control-test-d97e1eb7/strict-control-test-eval.json')
SAVE_DIR = Path('/data/uav/runs/strict-control-test-d97e1eb7/validator')
if OUT.exists():
    raise FileExistsError(f'create-only report already exists: {OUT}')
loaded = load_exact_final_checkpoint(
    CHECKPOINT,
    expected_sha256=EXPECTED_SHA,
    model_factory=lambda nc: RTDETRDetectionModel('rtdetr-l.yaml', nc=nc, ch=3, verbose=False),
)
if type(loaded.model) is not RTDETRDetectionModel:
    raise TypeError(f'expected pure stock RTDETRDetectionModel, got {type(loaded.model)!r}')
if loaded.metadata['kind'] != 'exact-final-ema':
    raise ValueError('strict Control evaluation requires epoch100 EMA')
protocol = {
    'imgsz': 640, 'batch': 8, 'workers': 8, 'conf': 0.001,
    'max_det': 300, 'nms': False, 'device': '0', 'cache': False,
    'half': False, 'rect': False, 'plots': False, 'save_json': False,
    'save_txt': False, 'verbose': True, 'split': 'test',
}
validator = CachedScaleRTDETRValidator(
    save_dir=SAVE_DIR,
    args={
        'model': str(CHECKPOINT), 'data': str(DATA), 'task': 'detect', 'mode': 'val',
        **protocol,
    },
)
validator(model=loaded.model)
processed = len(validator.scale_targets)
if processed != 1610 or len(validator.scale_predictions) != processed:
    raise RuntimeError(f'test processed {processed} images instead of 1610')
native = summarize_native_box_metrics(validator.metrics.box, CATEGORY_NAMES)
scales = summarize_scale_metrics(
    validator.scale_predictions, validator.scale_targets, class_count=len(CATEGORY_NAMES)
)
test_authority = json.loads(TEST_AUTHORITY.read_text(encoding='utf-8'))
if test_authority['images'] != 1610 or test_authority['instances'] != 75102:
    raise ValueError('test authority mismatch')
report = {
    'format_version': 1,
    'identity': {
        'variant': 'control', 'model': 'Ultralytics RT-DETR-L',
        'fdr_enabled': False, 'bpdd_enabled': False, 'ira_enabled': False,
        'training_source_commit': 'd97e1eb7f98414752a1c1f38287697db3f2a0679',
        'training_protocol_sha256': '2545F68302639EEB217EC8B53FDB229681B2DAFC91463C61F6B5CEC9B5486302',
        'initial_state_sha256': '51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D',
        'seed': 0, 'completed_epoch': 100,
    },
    'environment': {
        'gpu': 'NVIDIA GeForce RTX 4090', 'driver': '550.142',
        'python': '3.10.12', 'torch': '2.5.1+cu121',
        'torchvision': '0.20.1+cu121', 'cuda': '12.1', 'ultralytics': '8.4.90',
    },
    'checkpoint': loaded.metadata,
    'test_authority': test_authority,
    'evaluation_protocol': protocol,
    **native,
    **scales,
    'processed_images': processed,
    'prediction_passes': 1,
}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
print('STRICT_CONTROL_TEST_COMPLETE')
print(json.dumps({'metrics': report['metrics'], 'scales': report['scales'], 'processed_images': processed}, indent=2))
