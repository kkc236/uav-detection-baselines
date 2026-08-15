from __future__ import annotations

DEFAULT_YOLO_MODEL = "yolo11n.yaml"
DEFAULT_RTDETR_MODEL = "rtdetr-l.yaml"

SCIENTIFIC_PROBLEMS = [
    {
        "key": "background_interference",
        "name": "背景干扰问题",
        "description": "小目标在复杂背景中响应弱，容易被道路、建筑、植被等背景纹理干扰。",
    },
    {
        "key": "dense_objects",
        "name": "目标分布密集问题",
        "description": "车辆、行人等目标分布密集，相邻目标边界不清，容易漏检或重复检测。",
    },
    {
        "key": "scale_variation",
        "name": "多尺度变化问题",
        "description": "无人机飞行高度和拍摄视角变化导致目标尺度不稳定。",
    },
]


def build_train_settings(
    *,
    model: str,
    epochs: int,
    imgsz: int,
    name: str,
    batch: int = 4,
    device: str = "0",
) -> dict[str, str | int]:
    return {
        "model": model,
        "data": "VisDrone.yaml",
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": "runs/baselines",
        "name": name,
        "pretrained": False,
    }
