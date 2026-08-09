from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import src.rtdetr_complementarity_oracle as complementarity_module
from src.rtdetr_complementarity_oracle import (
    ComplementarityOracleCacheViolation,
    build_matched_quality_arm,
    candidate_iou_matrix,
    coverage_summary,
    decide_complementarity,
    load_paired_cache,
    one_to_one_same_class_assignment,
    visdrone_size_bucket,
    write_paired_cache,
)


def test_candidate_iou_matrix_uses_normalized_cxcywh() -> None:
    boxes = torch.tensor(
        [[0.50, 0.50, 0.40, 0.40], [0.25, 0.25, 0.20, 0.20]],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[0.50, 0.50, 0.20, 0.20], [0.80, 0.80, 0.10, 0.10]],
        dtype=torch.float64,
    )

    actual = candidate_iou_matrix(boxes, targets)

    assert actual.dtype == torch.float64
    assert torch.allclose(actual, torch.tensor([[0.25, 0.0], [0.0, 0.0]], dtype=torch.float64))


def test_candidate_iou_allows_finite_out_of_frame_decoder_boxes() -> None:
    actual = candidate_iou_matrix(
        torch.tensor([[-0.01, 0.5, 0.2, 0.2]]),
        torch.tensor([[0.05, 0.5, 0.1, 0.1]]),
    )

    assert actual.shape == (1, 1)
    assert 0.0 < actual.item() < 1.0


def test_assignment_is_same_class_one_to_one_and_maximizes_iou() -> None:
    predictions = torch.tensor(
        [
            [0.50, 0.50, 0.40, 0.40],
            [0.52, 0.50, 0.40, 0.40],
            [0.20, 0.20, 0.10, 0.10],
        ],
        dtype=torch.float32,
    )
    classes = torch.tensor([0, 0, 1])
    targets = torch.tensor(
        [[0.50, 0.50, 0.40, 0.40], [0.20, 0.20, 0.10, 0.10]],
        dtype=torch.float32,
    )
    target_classes = torch.tensor([0, 1])

    matrix = candidate_iou_matrix(predictions, targets)
    assignment = one_to_one_same_class_assignment(matrix, classes, target_classes)

    assert assignment.prediction_indices.tolist() == [0, 2]
    assert assignment.target_indices.tolist() == [0, 1]
    assert torch.allclose(assignment.ious, torch.ones(2))
    assert assignment.prediction_indices.device == matrix.device
    assert assignment.ious.dtype == matrix.dtype


def test_assignment_excludes_cross_class_and_zero_iou_pairs() -> None:
    result = one_to_one_same_class_assignment(
        torch.tensor([[1.0, 0.8], [0.9, 0.0]]),
        torch.tensor([0, 1]),
        torch.tensor([1, 1]),
    )

    assert result.prediction_indices.tolist() == [1]
    assert result.target_indices.tolist() == [0]
    assert result.ious.tolist() == pytest.approx([0.9])


def test_assignment_breaks_equal_iou_ties_by_prediction_index() -> None:
    iou = torch.tensor([[0.8], [0.8]])

    result = one_to_one_same_class_assignment(
        iou, torch.tensor([0, 0]), torch.tensor([0])
    )

    assert result.prediction_indices.tolist() == [0]


def test_assignment_never_lets_tie_breaking_outweigh_real_iou() -> None:
    iou = torch.tensor([[0.5], [0.5 + 5e-13]], dtype=torch.float64)

    result = one_to_one_same_class_assignment(
        iou, torch.tensor([0, 0]), torch.tensor([0])
    )

    assert result.prediction_indices.tolist() == [1]
    assert result.ious.item() == iou.max().item()


def test_assignment_is_repeatable_and_ordered_by_target_then_prediction() -> None:
    iou = torch.tensor([[0.5, 0.9], [0.8, 0.4], [0.7, 0.6]])
    prediction_classes = torch.tensor([0, 0, 0])
    target_classes = torch.tensor([0, 0])

    results = [
        one_to_one_same_class_assignment(iou, prediction_classes, target_classes)
        for _ in range(5)
    ]

    assert all(result.prediction_indices.tolist() == [1, 0] for result in results)
    assert all(result.target_indices.tolist() == [0, 1] for result in results)


@pytest.mark.parametrize(
    ("iou", "prediction_classes", "target_classes"),
    [
        (torch.empty((0, 0)), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        (torch.empty((3, 0)), torch.tensor([0, 1, 2]), torch.empty(0, dtype=torch.long)),
        (torch.empty((0, 2)), torch.empty(0, dtype=torch.long), torch.tensor([0, 1])),
    ],
)
def test_assignment_returns_empty_tensors_for_empty_inputs(
    iou: torch.Tensor,
    prediction_classes: torch.Tensor,
    target_classes: torch.Tensor,
) -> None:
    result = one_to_one_same_class_assignment(iou, prediction_classes, target_classes)

    assert result.prediction_indices.shape == (0,)
    assert result.target_indices.shape == (0,)
    assert result.ious.shape == (0,)
    assert result.prediction_indices.dtype == torch.long
    assert result.target_indices.dtype == torch.long


@pytest.mark.parametrize(
    ("boxes", "targets", "error", "message"),
    [
        ([[0.5, 0.5, 0.2, 0.2]], torch.empty((0, 4)), TypeError, "boxes.*tensor"),
        (torch.zeros(4), torch.empty((0, 4)), ValueError, r"boxes.*\[N, 4\]"),
        (torch.zeros((1, 5)), torch.empty((0, 4)), ValueError, r"boxes.*\[N, 4\]"),
        (torch.zeros((1, 4), dtype=torch.long), torch.empty((0, 4)), TypeError, "floating"),
        (torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]), torch.empty((0, 4)), ValueError, "finite"),
        (torch.tensor([[0.5, 0.5, -0.2, 0.2]]), torch.empty((0, 4)), ValueError, "non-negative width"),
        (torch.empty((0, 4)), torch.tensor([[0.5, 0.5, -0.1, 0.2]]), ValueError, "normalized"),
    ],
)
def test_candidate_iou_matrix_strictly_validates_inputs(
    boxes: object,
    targets: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        candidate_iou_matrix(boxes, targets)  # type: ignore[arg-type]


def test_candidate_iou_matrix_rejects_device_mismatch_before_computation() -> None:
    with pytest.raises(ValueError, match="device"):
        candidate_iou_matrix(
            torch.empty((0, 4)), torch.empty((0, 4), device="meta")
        )


@pytest.mark.parametrize(
    ("iou", "prediction_classes", "target_classes", "error", "message"),
    [
        ([[0.5]], torch.tensor([0]), torch.tensor([0]), TypeError, "IoU.*tensor"),
        (torch.tensor([0.5]), torch.tensor([0]), torch.tensor([0]), ValueError, "shape"),
        (torch.tensor([[0.5, 0.4]]), torch.tensor([0]), torch.tensor([0]), ValueError, "shape"),
        (torch.tensor([[1]], dtype=torch.long), torch.tensor([0]), torch.tensor([0]), TypeError, "floating"),
        (torch.tensor([[float("nan")]]), torch.tensor([0]), torch.tensor([0]), ValueError, "finite"),
        (torch.tensor([[1.1]]), torch.tensor([0]), torch.tensor([0]), ValueError, r"\[0, 1\]"),
        (torch.tensor([[0.5]]), torch.tensor([0], dtype=torch.int32), torch.tensor([0]), TypeError, "torch.long"),
        (torch.tensor([[0.5]]), torch.tensor([[0]]), torch.tensor([0]), ValueError, r"shape \[N\]"),
        (torch.tensor([[0.5]]), torch.tensor([-1]), torch.tensor([0]), ValueError, "class range"),
        (torch.tensor([[0.5]]), torch.tensor([10]), torch.tensor([0]), ValueError, "class range"),
    ],
)
def test_assignment_strictly_validates_inputs(
    iou: object,
    prediction_classes: torch.Tensor,
    target_classes: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        one_to_one_same_class_assignment(  # type: ignore[arg-type]
            iou, prediction_classes, target_classes
        )


def test_assignment_rejects_device_mismatch_before_computation() -> None:
    with pytest.raises(ValueError, match="device"):
        one_to_one_same_class_assignment(
            torch.empty((0, 0)),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long, device="meta"),
        )

    with pytest.raises(ValueError, match="device"):
        one_to_one_same_class_assignment(
            torch.ones((1, 1)),
            torch.tensor([0]),
            torch.tensor([0], dtype=torch.long, device="meta"),
        )


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (15.0, 15.0, "tiny"),
        (16.0, 16.0, "small"),
        (32.0, 32.0, "medium"),
        (96.0, 96.0, "large"),
        (0.0, 100.0, "tiny"),
    ],
)
def test_visdrone_size_bucket_uses_frozen_pixel_area_boundaries(
    width: float, height: float, expected: str
) -> None:
    assert visdrone_size_bucket(width, height) == expected


@pytest.mark.parametrize(
    ("width", "height", "error"),
    [
        (True, 1.0, TypeError),
        ("1", 1.0, TypeError),
        (-1.0, 1.0, ValueError),
        (float("inf"), 1.0, ValueError),
        (1.0, float("nan"), ValueError),
    ],
)
def test_visdrone_size_bucket_rejects_invalid_dimensions(
    width: object, height: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        visdrone_size_bucket(width, height)  # type: ignore[arg-type]


def test_union_coverage_counts_only_new_same_class_targets() -> None:
    summary = coverage_summary(
        torch.tensor([0.8, 0.2]),
        torch.tensor([0.7, 0.6]),
        thresholds=(0.5, 0.75),
    )

    assert summary["iou50"]["fdr_only"] == 1
    assert summary["iou50"]["frequencycm_only"] == 1
    assert summary["iou50"]["both"] == 0
    assert summary["iou50"]["neither"] == 0
    assert summary["iou50"]["union_gain"] == 0
    assert summary["iou75"]["fdr_only"] == 1
    assert summary["iou75"]["both"] == 0
    assert summary["raw"]["overall"]["iou50"]["both"] == 1
    assert summary["raw"]["overall"]["iou50"]["fdr_only"] == 0


def test_coverage_reports_raw_and_one_to_one_by_overall_scale_and_class() -> None:
    summary = coverage_summary(
        torch.tensor([0.8, 0.2, 0.8, 0.1]),
        torch.tensor([0.7, 0.6, 0.1, 0.1]),
        fdr_matched_iou=torch.tensor([0.8, 0.0, 0.0, 0.0]),
        frequencycm_matched_iou=torch.tensor([0.0, 0.6, 0.0, 0.0]),
        union_matched_iou=torch.tensor([0.8, 0.6, 0.0, 0.0]),
        target_scales=("tiny", "small", "tiny", "large"),
        target_classes=torch.tensor([0, 0, 1, 1]),
    )

    raw = summary["raw"]
    matched = summary["one_to_one"]
    assert raw["overall"]["iou50"]["fdr"] == 2
    assert raw["overall"]["iou50"]["frequencycm"] == 2
    assert raw["overall"]["iou50"]["union"] == 3
    assert raw["overall"]["iou50"]["union_gain"] == 1
    assert raw["by_scale"]["tiny"]["iou50"]["fdr"] == 2
    assert raw["by_scale"]["small"]["iou50"]["frequencycm_only"] == 1
    assert raw["by_class"][0]["iou50"]["union"] == 2
    assert matched["overall"]["iou50"]["fdr"] == 1
    assert matched["overall"]["iou50"]["frequencycm"] == 1
    assert matched["overall"]["iou50"]["union"] == 2
    assert matched["by_class"][1]["iou50"]["union"] == 0


def test_coverage_handles_no_targets_without_division_errors() -> None:
    summary = coverage_summary(torch.empty(0), torch.empty(0))

    assert summary["iou50"]["total"] == 0
    assert summary["iou50"]["union"] == 0
    assert summary["iou50"]["union_rate"] == 0.0
    assert summary["raw"]["by_scale"] == {}
    assert summary["raw"]["by_class"] == {}
    assert summary["one_to_one"] is None


def test_coverage_rejects_device_mismatch_before_value_checks() -> None:
    with pytest.raises(ValueError, match="device"):
        coverage_summary(torch.zeros(1), torch.zeros(1, device="meta"))


@pytest.mark.parametrize(
    ("fdr", "frequencycm", "kwargs", "error", "message"),
    [
        ([0.5], torch.tensor([0.5]), {}, TypeError, "fdr_best_iou.*tensor"),
        (torch.tensor([[0.5]]), torch.tensor([0.5]), {}, ValueError, r"shape \[N\]"),
        (torch.tensor([0.5]), torch.tensor([0.5], dtype=torch.long), {}, TypeError, "floating"),
        (torch.tensor([0.5]), torch.tensor([0.5, 0.4]), {}, ValueError, "same shape"),
        (torch.tensor([float("nan")]), torch.tensor([0.5]), {}, ValueError, "finite"),
        (torch.tensor([1.1]), torch.tensor([0.5]), {}, ValueError, r"\[0, 1\]"),
        (torch.tensor([0.5]), torch.tensor([0.5]), {"thresholds": (0.5,)}, ValueError, "frozen"),
        (
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            {"fdr_matched_iou": torch.tensor([0.5])},
            ValueError,
            "matched.*together",
        ),
        (
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            {"target_scales": ("unknown",)},
            ValueError,
            "scale",
        ),
        (
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            {"target_classes": torch.tensor([10])},
            ValueError,
            "class range",
        ),
    ],
)
def test_coverage_strictly_validates_inputs(
    fdr: object,
    frequencycm: object,
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        coverage_summary(fdr, frequencycm, **kwargs)  # type: ignore[arg-type]


def test_matched_quality_arm_assigns_query_class_pairs_and_orders_deterministically() -> None:
    arm = build_matched_quality_arm(
        boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]),
        probabilities=torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
        source_ranks=torch.tensor([1, 0]),
        target_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        target_classes=torch.tensor([1]),
        max_det=4,
    )

    assert arm.shape == (4, 6)
    assert arm[0].tolist() == pytest.approx([0.5, 0.5, 0.2, 0.2, 1.0, 1.0])
    assert arm[1].tolist() == pytest.approx([0.2, 0.2, 0.1, 0.1, 0.0, 0.0])
    assert arm[2].tolist() == pytest.approx([0.2, 0.2, 0.1, 0.1, 0.0, 1.0])
    assert arm[3].tolist() == pytest.approx([0.5, 0.5, 0.2, 0.2, 0.0, 0.0])


def test_duplicated_detector_has_exactly_neutral_oracle_candidates() -> None:
    arm = build_matched_quality_arm(
        boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        probabilities=torch.tensor([[0.9, 0.1]]),
        source_ranks=torch.tensor([0]),
        target_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        target_classes=torch.tensor([0]),
        max_det=2,
    )
    duplicated = build_matched_quality_arm(
        boxes=torch.tensor(
            [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]
        ),
        probabilities=torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
        source_ranks=torch.tensor([0, 1]),
        target_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        target_classes=torch.tensor([0]),
        max_det=2,
    )

    assert torch.equal(arm[:1], duplicated[:1])
    assert duplicated[1, 4].item() == 0.0


def test_duplicated_detector_cannot_reuse_one_box_for_two_same_class_targets() -> None:
    boxes = torch.tensor([[0.50, 0.50, 0.20, 0.20]])
    probabilities = torch.tensor([[0.9, 0.1]])
    targets = torch.tensor(
        [[0.50, 0.50, 0.20, 0.20], [0.54, 0.50, 0.20, 0.20]]
    )
    target_classes = torch.tensor([0, 0])
    original = build_matched_quality_arm(
        boxes,
        probabilities,
        torch.tensor([0]),
        targets,
        target_classes,
        max_det=2,
    )
    duplicated = build_matched_quality_arm(
        torch.cat((boxes, boxes)),
        torch.cat((probabilities, probabilities)),
        torch.tensor([0, 1]),
        targets,
        target_classes,
        max_det=2,
    )

    assert torch.equal(duplicated, original)


def test_matched_quality_arm_handles_empty_candidates_and_targets() -> None:
    empty = build_matched_quality_arm(
        boxes=torch.empty((0, 4)),
        probabilities=torch.empty((0, 2)),
        source_ranks=torch.empty(0, dtype=torch.long),
        target_boxes=torch.empty((0, 4)),
        target_classes=torch.empty(0, dtype=torch.long),
        max_det=300,
    )
    background = build_matched_quality_arm(
        boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        probabilities=torch.tensor([[0.9, 0.1]]),
        source_ranks=torch.tensor([0]),
        target_boxes=torch.empty((0, 4)),
        target_classes=torch.empty(0, dtype=torch.long),
        max_det=1,
    )

    assert empty.shape == (0, 6)
    assert empty.dtype == torch.float32
    assert background.shape == (1, 6)
    assert background[0, 4].item() == 0.0
    assert background[0, 5].item() == 0.0


def test_matched_quality_arm_rejects_target_device_mismatch_before_value_checks() -> None:
    with pytest.raises(ValueError, match="device"):
        build_matched_quality_arm(
            boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            probabilities=torch.tensor([[0.9, 0.1]]),
            source_ranks=torch.tensor([0]),
            target_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            target_classes=torch.tensor([0], dtype=torch.long, device="meta"),
            max_det=2,
        )


@pytest.mark.parametrize(
    ("replacement", "error", "message"),
    [
        ({"boxes": torch.zeros(4)}, ValueError, r"boxes.*\[Q, 4\]"),
        ({"probabilities": torch.zeros(1, 2, 1)}, ValueError, r"probabilities.*\[Q, C\]"),
        ({"probabilities": torch.tensor([[float("nan"), 0.0]])}, ValueError, "finite"),
        ({"probabilities": torch.tensor([[1.1, 0.0]])}, ValueError, r"\[0, 1\]"),
        ({"source_ranks": torch.tensor([0], dtype=torch.int32)}, TypeError, "torch.long"),
        ({"source_ranks": torch.tensor([-1])}, ValueError, "non-negative"),
        ({"target_classes": torch.tensor([2])}, ValueError, "class range"),
        ({"max_det": True}, ValueError, "positive integer"),
    ],
)
def test_matched_quality_arm_strictly_validates_inputs(
    replacement: dict[str, object], error: type[Exception], message: str
) -> None:
    arguments: dict[str, object] = {
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "probabilities": torch.tensor([[0.9, 0.1]]),
        "source_ranks": torch.tensor([0]),
        "target_boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "target_classes": torch.tensor([0]),
        "max_det": 2,
    }
    arguments.update(replacement)

    with pytest.raises(error, match=message):
        build_matched_quality_arm(**arguments)  # type: ignore[arg-type]


def _valid_authority() -> dict[str, str]:
    return {
        "fdr_sha256": "a" * 64,
        "frequencycm_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "evaluator_sha256": "d" * 64,
        "source_commit": "e" * 40,
    }


def _synthetic_record(image_id: str = "000001") -> dict[str, object]:
    return {
        "image_id": image_id,
        "original_shape": (1080, 1920),
        "resized_shape": (640, 640),
        "fdr_boxes": torch.full((300, 4), 0.5),
        "fdr_logits": torch.linspace(-2.0, 2.0, 3000).reshape(300, 10),
        "frequencycm_boxes": torch.full((300, 4), 0.25),
        "frequencycm_logits": torch.linspace(2.0, -2.0, 3000).reshape(300, 10),
        "target_boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "target_classes": torch.tensor([9], dtype=torch.long),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_paired_cache_is_create_only_authority_bound_and_hashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    authority = _valid_authority()

    manifest = write_paired_cache(root, [_synthetic_record()], authority)
    loaded = load_paired_cache(root, authority)

    assert manifest["record_count"] == 1
    assert manifest["complete"] is True
    assert manifest["authority"] == {
        name: value.upper() for name, value in authority.items()
    }
    assert manifest["artifact"]["sha256"] == _sha256(root / "records.pt")
    assert manifest["artifact"]["bytes"] == (root / "records.pt").stat().st_size
    assert {path.name for path in root.iterdir()} == {"manifest.json", "records.pt"}
    assert (root / "manifest.json").read_bytes() == (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    assert isinstance(loaded, tuple)
    assert loaded[0]["image_id"] == "000001"
    assert loaded[0]["original_shape"] == (1080, 1920)
    assert loaded[0]["resized_shape"] == (640, 640)
    assert torch.equal(loaded[0]["fdr_logits"], _synthetic_record()["fdr_logits"])
    with pytest.raises(FileExistsError, match="cache root"):
        write_paired_cache(root, [_synthetic_record("changed")], authority)


def test_paired_cache_accepts_empty_records(tmp_path: Path) -> None:
    root = tmp_path / "cache"

    manifest = write_paired_cache(root, [], _valid_authority())

    assert manifest["record_count"] == 0
    assert load_paired_cache(root, _valid_authority()) == ()


def test_paired_cache_preserves_out_of_frame_decoder_coordinates(
    tmp_path: Path,
) -> None:
    record = _synthetic_record()
    record["fdr_boxes"][0, 0] = -0.004

    write_paired_cache(tmp_path / "cache", [record], _valid_authority())
    loaded = load_paired_cache(tmp_path / "cache", _valid_authority())

    assert loaded[0]["fdr_boxes"][0, 0].item() == pytest.approx(-0.004)


def test_paired_cache_rejects_corrupted_payload_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    write_paired_cache(root, [_synthetic_record()], _valid_authority())
    (root / "records.pt").write_bytes(b"corrupt")
    load_calls: list[object] = []

    def forbidden_load(*args: object, **kwargs: object) -> object:
        load_calls.append(args[0])
        raise AssertionError("deserialized before SHA verification")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="SHA-256"):
        load_paired_cache(root, _valid_authority())
    assert load_calls == []


def test_paired_cache_deserializes_the_verified_open_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    write_paired_cache(root, [_synthetic_record()], _valid_authority())
    real_load = torch.load
    load_sources: list[object] = []

    def tracked_load(source: object, *args: object, **kwargs: object) -> object:
        load_sources.append(source)
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    loaded = load_paired_cache(root, _valid_authority())

    assert loaded[0]["image_id"] == "000001"
    assert len(load_sources) == 1
    assert not isinstance(load_sources[0], (str, Path))


def test_paired_cache_requires_exact_authority(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    authority = _valid_authority()
    write_paired_cache(root, [_synthetic_record()], authority)

    changed = dict(authority)
    changed["fdr_sha256"] = "0" * 64
    with pytest.raises(ComplementarityOracleCacheViolation, match="authority.*fdr_sha256"):
        load_paired_cache(root, changed)

    missing = dict(authority)
    missing.pop("dataset_sha256")
    with pytest.raises(ComplementarityOracleCacheViolation, match="authority schema"):
        load_paired_cache(root, missing)

    invalid = dict(authority)
    invalid["dataset_sha256"] = "not-a-hash"
    with pytest.raises(ComplementarityOracleCacheViolation, match="dataset_sha256"):
        load_paired_cache(root, invalid)


def test_paired_cache_revalidates_rehashed_payload_schema(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    authority = _valid_authority()
    manifest = write_paired_cache(root, [_synthetic_record()], authority)
    artifact_path = root / "records.pt"
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    payload["records"][0]["unexpected"] = torch.tensor(1)
    torch.save(payload, artifact_path)
    manifest["artifact"]["bytes"] = artifact_path.stat().st_size
    manifest["artifact"]["sha256"] = _sha256(artifact_path)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ComplementarityOracleCacheViolation, match="record schema"):
        load_paired_cache(root, authority)


def test_paired_cache_rejects_duplicate_image_ids_and_existing_empty_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ComplementarityOracleCacheViolation, match="unique"):
        write_paired_cache(
            tmp_path / "duplicates",
            [_synthetic_record(), _synthetic_record()],
            _valid_authority(),
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="cache root"):
        write_paired_cache(existing, [], _valid_authority())


@pytest.mark.parametrize(
    ("field", "unsafe", "message"),
    [
        ("original_shape", (1080,), "original_shape"),
        ("original_shape", (True, 1920), "original_shape"),
        ("resized_shape", (640,), "resized_shape"),
        ("resized_shape", (640, 0), "resized_shape"),
        ("fdr_boxes", torch.zeros((299, 4)), "production tensor shape"),
        ("fdr_boxes", torch.zeros((300, 4), dtype=torch.float64), "dtype"),
        ("fdr_boxes", torch.zeros((300, 4), requires_grad=True), "detached"),
        ("fdr_boxes", torch.zeros((4, 300)).T, "contiguous"),
        (
            "frequencycm_boxes",
            torch.tensor([[0.5, 0.5, -0.1, 0.1]]).repeat(300, 1),
            "non-negative width",
        ),
        ("fdr_logits", torch.full((300, 10), float("nan")), "finite"),
        ("frequencycm_logits", torch.zeros((300, 9)), "production tensor shape"),
        ("target_boxes", torch.zeros((1, 4), dtype=torch.float64), "dtype"),
        ("target_classes", torch.tensor([0], dtype=torch.int32), "torch.int64"),
        ("target_classes", torch.tensor([10]), "class range"),
    ],
)
def test_paired_cache_rejects_unsafe_records(
    tmp_path: Path, field: str, unsafe: object, message: str
) -> None:
    record = {**_synthetic_record(), field: unsafe}

    with pytest.raises(ComplementarityOracleCacheViolation, match=message):
        write_paired_cache(tmp_path / field, [record], _valid_authority())


def test_paired_cache_interrupted_creation_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"

    def interrupted_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="interrupted"):
        write_paired_cache(root, [_synthetic_record()], _valid_authority())

    assert not root.exists()
    assert not tuple(tmp_path.glob(".cache.staging-*"))


def test_paired_cache_rejects_symlink_or_reparse_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setattr(
        complementarity_module,
        "_is_symlink_or_reparse",
        lambda path: Path(path) == root,
        raising=False,
    )

    with pytest.raises(ComplementarityOracleCacheViolation, match="root.*symlink|reparse"):
        write_paired_cache(root, [], _valid_authority())


@pytest.mark.parametrize(
    ("map_delta", "recall_delta", "expected"),
    [
        (0.0029, 0.0099, "red"),
        (0.0030, 0.0, "yellow"),
        (0.0, 0.0100, "yellow"),
        (0.0100, 0.0, "green"),
        (0.0, 0.0200, "green"),
        (0.0030, 0.0200, "green"),
    ],
)
def test_decision_boundaries_are_exact(
    map_delta: float, recall_delta: float, expected: str
) -> None:
    result = decide_complementarity(map_delta, recall_delta)

    assert result["decision"] == expected
    assert float(result["observed"]["map_delta"]) == map_delta
    assert float(result["observed"]["tiny_small_recall_delta"]) == recall_delta


@pytest.mark.parametrize(
    ("map_delta", "recall_delta", "error"),
    [
        (True, 0.0, TypeError),
        ("0.01", 0.0, TypeError),
        (float("nan"), 0.0, ValueError),
        (0.0, float("inf"), ValueError),
    ],
)
def test_decision_rejects_non_numeric_or_nonfinite_inputs(
    map_delta: object, recall_delta: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        decide_complementarity(map_delta, recall_delta)  # type: ignore[arg-type]
