from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.adjudicate_saded_stock_fresh import (
    decide,
    exit_code_for_decision,
)
from scripts.evaluate_saded_stock_single import (
    create_evaluation_claim,
    evaluation_invariants_passed,
    metric_row,
)
from scripts.route_saded_stock_single import _verify_checksums
from src.saded_single_model_adjudicator import FORMAL_THRESHOLDS
from src.sbr_artifacts import sha256_file, write_checksums
from src.saded_stock_postprocess import route_single_cache


def _prediction(box, score, *, source=0, query=0):
    return {
        "box": list(box),
        "global_xyxy": list(box),
        "score": score,
        "class_id": 0,
        "source_order": source,
        "query_index": query,
    }


def _cache_row():
    return {
        "image_id": "image.jpg",
        "width": 640,
        "height": 640,
        "full_predictions": [
            _prediction((0, 0, 40, 40), 0.8),
            _prediction((100, 100, 110, 110), 0.4, query=1),
        ],
        "local_fused_predictions": [
            _prediction(
                (100, 100, 111, 111),
                0.5,
                source=1,
                query=2,
            )
        ],
    }


def test_single_cache_route_emits_only_one_baseline_and_candidate() -> None:
    rows, invariants = route_single_cache([_cache_row()])
    assert invariants["passed"] is True
    assert set(rows[0]["arms"]) == {"A", "route_control"}
    assert rows[0]["arms"]["A"][0]["box"] == [0, 0, 40, 40]
    assert rows[0]["arms"]["route_control"][0]["box"] == [0, 0, 40, 40]
    assert rows[0]["arms"]["route_control"][1]["box"] == [
        100,
        100,
        111,
        111,
    ]
    assert {
        "protected_baseline",
        "remaining_tiny_slots",
        "accepted_local",
        "capacity_rejected",
    }.issubset(rows[0]["coverage"])


def test_single_cache_route_rejects_empty_or_identity_drift() -> None:
    with pytest.raises(ValueError, match="row count"):
        route_single_cache([])
    bad = deepcopy(_cache_row())
    bad["image_id"] = ""
    with pytest.raises(ValueError, match="identity"):
        route_single_cache([bad])


def test_single_cache_route_contains_no_gt_or_duplicate_arm() -> None:
    rows, invariants = route_single_cache([_cache_row()])
    assert invariants["single_endpoint_exact"] is True
    assert invariants["gt_fields_absent"] is True
    assert "route_treatment" not in rows[0]["arms"]
    assert not {
        "gt_boxes",
        "gt_classes",
        "ignore_boxes",
        "annotations",
    }.intersection(rows[0])


def test_route_checksum_reader_accepts_writer_output(tmp_path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    write_checksums(
        tmp_path / "checksums.sha256",
        [artifact],
        root=tmp_path,
    )

    observed = _verify_checksums(tmp_path, {"artifact.json"})

    assert observed["artifact.json"] == sha256_file(artifact)


def test_evaluation_claim_is_exclusive(tmp_path) -> None:
    claim = tmp_path / "claim.json"

    create_evaluation_claim(claim, {"state": "CONSUMED"})

    with pytest.raises(FileExistsError):
        create_evaluation_claim(claim, {"state": "CONSUMED"})


def test_metric_row_preserves_prediction_provenance() -> None:
    image = {
        "relative_path": "image.jpg",
        "width": 640,
        "height": 320,
        "gt_boxes": [[1.0, 2.0, 3.0, 4.0]],
        "gt_classes": [2],
        "ignore_boxes": [[5.0, 6.0, 7.0, 8.0]],
    }
    predictions = [
        {
            "box": [10.0, 20.0, 30.0, 40.0],
            "score": 0.75,
            "class_id": 2,
            "source_order": 0,
            "query_index": 7,
        }
    ]

    row = metric_row(image, predictions)

    assert row["image_id"] == "image.jpg"
    assert row["pred_source"] == [0]
    assert row["pred_query"] == [7]
    assert row["gt_boxes"] == image["gt_boxes"]
    assert row["effective_gain"] == 1.0


def test_fresh_adjudicator_recomputes_five_frozen_gates() -> None:
    baseline = {key: 0.1 for key in FORMAL_THRESHOLDS}
    candidate = {
        **baseline,
        "AP-tiny-SBR": 0.111,
        "mAP50-95": 0.104,
        "tiny_recall": 0.121,
    }

    result = decide(
        arm_a=baseline,
        route_control=candidate,
        invariants_passed=True,
    )

    assert result["decision"] == "SADED_SINGLE_SEED_GO"
    assert set(result["gates"]) == set(FORMAL_THRESHOLDS)


def test_fresh_adjudicator_fails_closed_on_bad_closure() -> None:
    metrics = {key: 0.1 for key in FORMAL_THRESHOLDS}

    result = decide(
        arm_a=metrics,
        route_control=metrics,
        invariants_passed=False,
    )

    assert result["decision"] == "INVALID"


def test_evaluation_invariants_use_positive_retry_policy() -> None:
    assert evaluation_invariants_passed(
        {
            "route_snapshot_unchanged": True,
            "retry_forbidden": True,
        }
    )
    assert not evaluation_invariants_passed(
        {
            "route_snapshot_unchanged": False,
            "retry_forbidden": True,
        }
    )


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("SADED_SINGLE_SEED_GO", 0),
        ("SADED_SINGLE_SEED_STOP", 1),
        ("INVALID", 2),
    ],
)
def test_adjudicator_exit_code_distinguishes_invalid(
    decision: str,
    expected: int,
) -> None:
    assert exit_code_for_decision(decision) == expected
