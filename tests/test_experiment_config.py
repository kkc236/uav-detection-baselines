from src.experiment_config import (
    DEFAULT_RTDETR_MODEL,
    DEFAULT_YOLO_MODEL,
    SCIENTIFIC_PROBLEMS,
    build_train_settings,
)


def test_scientific_problems_match_uav_detection_focus():
    keys = [item["key"] for item in SCIENTIFIC_PROBLEMS]

    assert keys == ["background_interference", "dense_objects", "scale_variation"]


def test_build_train_settings_uses_visdrone_and_project_name():
    settings = build_train_settings(model="yolo11n.pt", epochs=1, imgsz=640, name="smoke-yolo")

    assert settings["data"] == "VisDrone.yaml"
    assert settings["project"] == "runs/baselines"
    assert settings["name"] == "smoke-yolo"
    assert settings["epochs"] == 1
    assert settings["imgsz"] == 640


def test_train_settings_disable_pretrained_weights():
    settings = build_train_settings(model=DEFAULT_YOLO_MODEL, epochs=1, imgsz=640, name="scratch-yolo")

    assert settings["pretrained"] is False


def test_default_models_are_yaml_configs_not_weight_files():
    assert DEFAULT_YOLO_MODEL == "yolo11n.yaml"
    assert DEFAULT_RTDETR_MODEL == "rtdetr-l.yaml"
    assert not DEFAULT_YOLO_MODEL.endswith(".pt")
    assert not DEFAULT_RTDETR_MODEL.endswith(".pt")
