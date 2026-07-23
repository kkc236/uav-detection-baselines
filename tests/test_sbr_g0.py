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
    from src.sbr_g0 import assemble_arm, assemble_paired_arms, collect_raw_views

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    def predict_square(square, imgsz):
        return [{"xyxy": [318, 318, 320, 320], "score": 0.5, "class_id": 1, "query_index": 0}]

    raw = collect_raw_views(image, "C", predict_square)
    c = assemble_arm(raw, "C", width=8, height=8)
    d = assemble_paired_arms(raw, width=8, height=8)["D"]
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


def test_letterbox_matches_ultralytics_when_available():
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox
    from src.sbr_g0 import _letterbox

    image = np.arange(11 * 17 * 3, dtype=np.uint8).reshape(11, 17, 3)
    expected = LetterBox(
        new_shape=(640, 640), auto=False, scale_fill=False, scaleup=False,
        center=True, padding_value=114,
    )(image=image)
    assert np.array_equal(_letterbox(image, 640), expected)


def test_invalid_predictor_rows_fail_closed():
    from src.sbr_g0 import collect_raw_views

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        collect_raw_views(image, "A", lambda square, imgsz: [{"xyxy": [0, 0, 1], "score": 0.5, "class_id": 1}])
    with pytest.raises(ValueError):
        collect_raw_views(image, "A", lambda square, imgsz: [{"xyxy": [320, 320, 328, 328], "score": float("nan"), "class_id": 1}])


def test_assemble_rejects_overflow_and_out_of_tile_records():
    from src.sbr_g0 import RawViewRecord, assemble_arm, build_arm_views

    view = build_arm_views("B", 100, 100)[0]
    bad = RawViewRecord(
        image_id="i", width=100, height=100, arm="B", view_id="TL", source_order=1,
        query_index=0, tile_bounds=view.tile.bounds,
        transform=RawViewRecord.from_prediction(view, [300, 300, 320, 320], .5, 1, 0, 100, 100).transform,
        network_xyxy=(300., 300., 320., 320.), view_xyxy=(0., 0., 1000., 1000.),
        global_xyxy=(1., 1., 2., 2.), score=.5, class_id=1,
    )
    with pytest.raises(ValueError):
        assemble_arm([bad], "B", width=100, height=100)


def test_assemble_paired_arms_requires_identical_c_d_inputs():
    from src.sbr_g0 import assemble_paired_arms, collect_raw_views

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    raw = collect_raw_views(image, "C", lambda square, imgsz: [{"xyxy": [318, 318, 320, 320], "score": .5, "class_id": 1}])
    pair = assemble_paired_arms(raw, width=8, height=8)
    assert pair["C"]["raw_hash"] == pair["D"]["raw_hash"]
    assert pair["C"]["cluster_hash"] == pair["D"]["cluster_hash"]


def test_collect_manifest_and_missing_view_fail_closed():
    from src.sbr_g0 import assemble_arm, collect_raw_views

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    raw, manifest = collect_raw_views(image, "C", lambda square, imgsz: [], return_manifest=True)
    assert len(manifest) == 5 and all(item["executed"] for item in manifest)
    with pytest.raises(ValueError):
        assemble_arm(raw, "C", width=8, height=8, view_manifest=manifest[:-1])
