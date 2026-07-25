from __future__ import annotations

import pytest

from scripts.cache_saded_endpoint import (
    _same_sha,
    _view_manifest_is_complete,
)
from src.saded_stage import route_paired_caches, screen_seed0_gate
from src.saded_adjudicator import (
    _attribution_three_seed_gate,
    _formal_primary_failures,
)


def _prediction(box, score, *, source=0, query=0):
    return {
        "box": list(box),
        "global_xyxy": list(box),
        "score": score,
        "class_id": 0,
        "source_order": source,
        "query_index": query,
    }


def _row(*, full, local):
    return {
        "image_id": "image.jpg",
        "width": 640,
        "height": 640,
        "full_predictions": full,
        "local_fused_predictions": local,
    }


def test_paired_route_emits_one_gt_free_prediction_set_per_system():
    protected = _prediction((0, 0, 40, 40), 0.8)
    tiny = _prediction((100, 100, 110, 110), 0.4, query=1)
    control_local = _prediction(
        (100, 100, 111, 111),
        0.5,
        source=1,
        query=2,
    )
    treatment_local = _prediction(
        (100, 100, 112, 112),
        0.9,
        source=1,
        query=3,
    )

    rows, invariants = route_paired_caches(
        [
            _row(
                full=[protected, tiny],
                local=[control_local],
            )
        ],
        [
            _row(
                full=[protected, tiny],
                local=[treatment_local],
            )
        ],
    )

    assert invariants["passed"] is True
    assert set(rows[0]["arms"]) == {
        "A",
        "route_control",
        "route_treatment",
    }
    assert rows[0]["arms"]["A"][0] == protected
    assert rows[0]["arms"]["route_control"][0] == protected
    assert rows[0]["arms"]["route_treatment"][0] == protected
    assert rows[0]["arms"]["route_control"][1]["box"] == control_local["box"]
    assert (
        rows[0]["arms"]["route_treatment"][1]["box"]
        == treatment_local["box"]
    )
    assert not {
        "gt_boxes",
        "gt_classes",
        "ignore_boxes",
        "annotations",
    }.intersection(rows[0])


def test_paired_route_rejects_cache_identity_drift():
    baseline = [_row(full=[], local=[])]
    treatment = [_row(full=[], local=[])]
    treatment[0]["image_id"] = "other.jpg"

    with pytest.raises(ValueError, match="identity"):
        route_paired_caches(baseline, treatment)


def test_screen_seed0_gate_uses_only_frozen_attribution_thresholds():
    control = {
        "mAP50-95": 0.1,
        "AP-tiny-SBR": 0.2,
        "tiny_recall": 0.3,
        "AP75": 0.15,
        "AP-large-SBR": 0.25,
    }
    treatment = {
        "mAP50-95": 0.100001,
        "AP-tiny-SBR": 0.2,
        "tiny_recall": 0.3,
        "AP75": 0.148,
        "AP-large-SBR": 0.245,
    }

    passing = screen_seed0_gate(
        route_control=control,
        route_treatment=treatment,
        invariants_passed=True,
    )
    treatment["mAP50-95"] = 0.1
    failing = screen_seed0_gate(
        route_control=control,
        route_treatment=treatment,
        invariants_passed=True,
    )

    assert passing["decision"] == "TASCV_SCREEN_SEED0_GO"
    assert passing["failures"] == []
    assert passing["deltas"]["AP75"] == pytest.approx(-0.002)
    assert failing["decision"] == "TASCV_STOP"
    assert failing["failures"] == ["mAP50-95_delta<=0"]


def test_screen_seed0_runtime_failure_is_invalid():
    decision = screen_seed0_gate(
        route_control={},
        route_treatment={},
        invariants_passed=False,
    )
    assert decision["decision"] == "INVALID"


def test_endpoint_cache_requires_exact_five_view_execution_manifest():
    complete = [
        {"view_id": "full", "source_order": 0, "executed": True},
        {"view_id": "TL", "source_order": 1, "executed": True},
        {"view_id": "TR", "source_order": 2, "executed": True},
        {"view_id": "BL", "source_order": 3, "executed": True},
        {"view_id": "BR", "source_order": 4, "executed": True},
    ]

    assert _view_manifest_is_complete(complete) is True
    assert _view_manifest_is_complete(complete[:-1]) is False
    assert _view_manifest_is_complete(complete + [complete[-1]]) is False
    incomplete = [dict(record) for record in complete]
    incomplete[-1]["executed"] = False
    assert _view_manifest_is_complete(incomplete) is False


def test_endpoint_cache_accepts_uppercase_training_sha(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"sealed")
    import hashlib

    expected = hashlib.sha256(b"sealed").hexdigest().upper()
    assert _same_sha(artifact, expected) is True


def test_three_seed_attribution_gate_uses_frozen_sign_and_mean_rules():
    def verified(delta):
        return {
            "deltas": {
                "route_treatment_vs_route_control": delta,
            }
        }

    base = {
        "mAP50-95": 0.001,
        "AP-tiny-SBR": 0.0,
        "tiny_recall": 0.0,
        "AP75": -0.002,
        "AP-large-SBR": -0.005,
    }
    failures, record = _attribution_three_seed_gate(
        {
            0: verified(base),
            1: verified(base),
            2: verified({**base, "mAP50-95": -0.001}),
        }
    )

    assert failures == []
    assert record["counts"]["mAP_positive"] == 2
    assert record["mean"]["mAP50-95"] > 0.0


def test_formal_primary_gate_reapplies_original_five_thresholds():
    passing = {
        "AP-tiny-SBR": 0.010,
        "mAP50-95": 0.003,
        "tiny_recall": 0.020,
        "AP75": -0.002,
        "AP-large-SBR": -0.005,
    }
    assert _formal_primary_failures(passing) == []
    assert _formal_primary_failures(
        {**passing, "AP-tiny-SBR": 0.009}
    ) == ["AP-tiny-SBR<0.01"]
