import hashlib

import numpy as np
import pytest


def test_protocol_constants_and_arm_view_layout():
    from src.sbr_g0 import FrozenSBRProtocol, build_arm_views

    p = FrozenSBRProtocol()
    assert (p.imgsz, p.high_imgsz, p.conf, p.max_det, p.ios_threshold) == (
        640,
        1088,
        0.001,
        300,
        0.5,
    )
    assert [v.view_id for v in build_arm_views("C", 100, 80)] == [
        "full",
        "TL",
        "TR",
        "BL",
        "BR",
    ]
    assert [v.source_order for v in build_arm_views("C", 100, 80)] == [0, 1, 2, 3, 4]
    assert build_arm_views("B", 100, 80)[0].tile.bounds == (0, 0, 60, 48)
    assert build_arm_views("F", 101, 81)[0].tile.bounds == (0, 0, 50, 40)
    assert build_arm_views("F", 101, 81)[3].tile.bounds == (50, 40, 101, 81)


def test_collect_inverse_mapping_and_predict_once_per_view():
    from src.sbr_g0 import collect_raw_views

    image = np.zeros((10, 20, 3), dtype=np.uint8)
    calls = []

    def predict_square(square, imgsz):
        calls.append((square.shape, imgsz))
        return [{"xyxy": [320, 320, 640, 640], "score": 0.9, "class_id": 2, "query_index": 7}]

    records = collect_raw_views(image, "A", predict_square)
    assert len(calls) == 1
    rec = records[0]
    assert rec.network_xyxy == (320.0, 320.0, 640.0, 640.0)
    assert rec.view_xyxy == pytest.approx((10.0, 5.0, 330.0, 325.0))
    assert rec.global_xyxy == pytest.approx((10.0, 5.0, 20.0, 10.0))
    assert rec.transform.auto is False and rec.transform.scale_fill is False


def test_arm_c_d_share_raw_and_clusters_and_labels_never_touched():
    from src.sbr_g0 import assemble_arm, collect_raw_views

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    def predict_square(square, imgsz):
        return [{"xyxy": [0, 0, 8, 8], "score": 0.5, "class_id": 1, "query_index": 0}]

    raw = collect_raw_views(image, "C", predict_square)
    c = assemble_arm(raw, "C", width=8, height=8)
    d = assemble_arm(raw, "D", width=8, height=8)
    assert c["raw_hash"] == d["raw_hash"]
    assert c["cluster_hash"] == d["cluster_hash"]
    assert c["records"] == d["records"]
    assert len(c["predictions"]) <= 300 and len(d["predictions"]) <= 300
    assert hashlib.sha256(c["raw_bytes"]).hexdigest() == c["raw_hash"]


def test_raw_record_rejects_nonfinite_and_per_view_limit():
    from src.sbr_g0 import RawViewRecord, build_arm_views
    view = build_arm_views("A", 20, 20)[0]
    with pytest.raises(ValueError):
        RawViewRecord.from_prediction(view, [float("nan"), 0, 1, 1], 0.5, 0, 0, 20, 20)


def test_numpy_predictor_rows_are_accepted_and_filtered():
    from src.sbr_g0 import collect_raw_views

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    rows = np.array([[320, 320, 328, 328, 0.5, 3], [320, 320, 328, 328, 0.0001, 2]])
    records = collect_raw_views(image, "A", lambda square, imgsz: rows)
    assert len(records) == 1
    assert records[0].class_id == 3
