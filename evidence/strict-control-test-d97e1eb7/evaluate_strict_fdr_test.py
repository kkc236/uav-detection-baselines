from __future__ import annotations
import json,sys
from pathlib import Path
SOURCE=Path('/data/uav/source/uav-detection-baselines-2997705c');sys.path.insert(0,str(SOURCE))
from src.bpdd_formal_evaluation import CachedScaleRTDETRValidator,load_exact_final_checkpoint,summarize_native_box_metrics,summarize_scale_metrics
from src.lpr_protocol import CATEGORY_NAMES
from src.rtdetr_fdr import FDRRTDETRDetectionModel
CK=Path('/data/uav/runs/fdr-formal-d97e1eb7/formal-seed0-fdr-fdr-v1/weights/epoch99.pt')
SHA='C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2'
DATA=Path('/data/uav/evidence/strict-control-test-d97e1eb7/test-data.yaml')
AUTH=Path('/data/uav/evidence/strict-control-test-d97e1eb7/test-authority.json')
OUT=Path('/data/uav/evidence/strict-control-test-d97e1eb7/strict-fdr-test-eval.json')
if OUT.exists():raise FileExistsError(OUT)
loaded=load_exact_final_checkpoint(CK,expected_sha256=SHA)
if type(loaded.model) is not FDRRTDETRDetectionModel:raise TypeError(type(loaded.model))
protocol={'imgsz':640,'batch':8,'workers':8,'conf':0.001,'max_det':300,'nms':False,'device':'0','cache':False,'half':False,'rect':False,'plots':False,'save_json':False,'save_txt':False,'verbose':True,'split':'test'}
v=CachedScaleRTDETRValidator(save_dir=Path('/data/uav/runs/strict-control-test-d97e1eb7/fdr-validator'),args={'model':str(CK),'data':str(DATA),'task':'detect','mode':'val',**protocol})
v(model=loaded.model)
processed=len(v.scale_targets)
if processed!=1610 or len(v.scale_predictions)!=1610:raise RuntimeError(processed)
report={'format_version':1,'identity':{'variant':'fdr','model':'Ultralytics RT-DETR-L + FDR','training_source_commit':'d97e1eb7f98414752a1c1f38287697db3f2a0679','training_protocol_sha256':'2545F68302639EEB217EC8B53FDB229681B2DAFC91463C61F6B5CEC9B5486302','initial_state_sha256':'51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D','seed':0,'completed_epoch':100},'checkpoint':loaded.metadata,'test_authority':json.loads(AUTH.read_text()),'evaluation_protocol':protocol,**summarize_native_box_metrics(v.metrics.box,CATEGORY_NAMES),**summarize_scale_metrics(v.scale_predictions,v.scale_targets,class_count=10),'processed_images':processed,'prediction_passes':1}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print('STRICT_FDR_TEST_COMPLETE');print(json.dumps({'metrics':report['metrics'],'scales':report['scales']},indent=2))
