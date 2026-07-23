from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from PIL import Image
import pytest

from src.sbr_artifacts import (
    atomic_write_json,
    atomic_write_jsonl_gz,
    load_dataset,
    protocol_signature,
    sha256_file,
    write_checksums,
)
from src.sbr_g0 import FrozenSBRProtocol
from src.sbr_fusion import Detection
from src.sbr_metrics import evaluate_dataset


REQUIRED_OUTPUTS = {
    "audit_manifest.json",
    "attribution_events.jsonl.gz",
    "attribution_summary.json",
    "upper_bound_metrics.json",
    "invariants.json",
    "primary_gate.json",
    "checksums.sha256",
}


def _raw(
    image_id: str,
    arm: str,
    source: int,
    query: int,
    box: tuple[float, float, float, float],
    *,
    index: int,
    score: float = 0.9,
) -> dict[str, object]:
    tile = None if source == 0 else [0, 0, 384, 384]
    view_id = ("full", "TL", "TR", "BL", "BR")[source]
    manifest_sources = (0,) if arm == "A" else (0, 1, 2, 3, 4)
    return {
        "image_id": image_id,
        "width": 640,
        "height": 640,
        "arm": arm,
        "view_id": view_id,
        "source_order": source,
        "query_index": query,
        "tile_bounds": tile,
        "network_xyxy": list(box),
        "view_xyxy": list(box),
        "global_xyxy": list(box),
        "score": score,
        "class_id": 0,
        "view_manifest": [
            {
                "view_id": ("full", "TL", "TR", "BL", "BR")[item],
                "source_order": item,
                "executed": True,
            }
            for item in manifest_sources
        ],
    }


def _make_input(
    tmp_path: Path, *, recoverable: bool = True
) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "val").mkdir(parents=True)
    (dataset_root / "labels" / "val").mkdir(parents=True)
    Image.new("RGB", (640, 640)).save(
        dataset_root / "images" / "val" / "one.jpg"
    )
    (dataset_root / "labels" / "val" / "one.txt").write_text(
        "0 0.15625 0.15625 0.3125 0.3125\n", encoding="utf-8"
    )
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {dataset_root.as_posix()}\nval: images/val\n",
        encoding="utf-8",
    )
    dataset = load_dataset(dataset_yaml)

    evidence = tmp_path / "g0-evidence"
    evidence.mkdir()
    image_list = ["one.jpg"]
    atomic_write_json(evidence / "image_list.json", image_list)
    a_full = _raw("one.jpg", "A", 0, 0, (0.0, 0.0, 200.0, 200.0), index=0)
    c_full = _raw("one.jpg", "C", 0, 0, (0.0, 0.0, 200.0, 200.0), index=1)
    rows = [a_full, c_full]
    if recoverable:
        rows.append(
            _raw(
                "one.jpg",
                "C",
                1,
                1,
                (80.0, 0.0, 280.0, 200.0),
                index=2,
                score=0.95,
            )
        )
    atomic_write_jsonl_gz(evidence / "raw_views.jsonl.gz", rows)
    c_box = (
        [
            80.0 * 0.95 / 1.85,
            0.0,
            (200.0 * 0.9 + 280.0 * 0.95) / 1.85,
            200.0,
        ]
        if recoverable
        else [
            0.0,
            0.0,
            200.0,
            200.0,
        ]
    )

    def record(row: dict[str, object]) -> dict[str, object]:
        return {
            key: value
            for key, value in row.items()
            if key != "view_manifest"
        }

    def prediction(
        box: list[float],
        *,
        global_box: list[float] | None = None,
        score: float = 0.9,
        source: int = 0,
        query: int = 0,
    ) -> dict[str, object]:
        return {
            "box": box,
            "global_xyxy": (
                [0.0, 0.0, 200.0, 200.0]
                if global_box is None
                else global_box
            ),
            "score": score,
            "class_id": 0,
            "source_order": source,
            "query_index": query,
        }

    c_records = [record(c_full)]
    if recoverable:
        c_records.append(record(rows[-1]))
    atomic_write_jsonl_gz(
        evidence / "arm_predictions.jsonl.gz",
        [
            {
                "image_id": "one.jpg",
                "records": [record(a_full)],
                "predictions": [prediction([0.0, 0.0, 200.0, 200.0])],
            },
            {"image_id": "one.jpg", "records": [], "predictions": []},
            {
                "image_id": "one.jpg",
                "records": c_records,
                "predictions": [
                    prediction(
                        c_box,
                        global_box=(
                            [80.0, 0.0, 280.0, 200.0]
                            if recoverable
                            else None
                        ),
                        score=0.95 if recoverable else 0.9,
                        source=1 if recoverable else 0,
                        query=1 if recoverable else 0,
                    )
                ],
            },
            {"image_id": "one.jpg", "records": [], "predictions": []},
            {"image_id": "one.jpg", "records": [], "predictions": []},
            {"image_id": "one.jpg", "records": [], "predictions": []},
        ],
    )
    source_commit = "a" * 40
    source_tree = "b" * 64
    protocol = dict(FrozenSBRProtocol().__dict__)
    protocol_hash = protocol_signature(protocol)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    g0_manifest = {
        "mode": "g0-a",
        "source": {
            "commit": source_commit,
            "source_tree_hash": source_tree,
        },
        "source_hash": source_commit,
        "checkpoint_hash": checkpoint_hash,
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "dataset_signature": dataset["dataset_signature"],
        "image_count": 1,
        "image_list": image_list,
    }
    atomic_write_json(evidence / "g0_manifest.json", g0_manifest)
    def metric_row(
        box: list[float],
        *,
        score: float = 0.9,
        source: int = 0,
        query: int = 0,
    ) -> dict[str, object]:
        return {
            "image_id": "one.jpg",
            "width": 640,
            "height": 640,
            "pred_boxes": [box],
            "pred_scores": [score],
            "pred_classes": [0],
            "pred_source": [source],
            "pred_query": [query],
            "gt_boxes": [[0.0, 0.0, 200.0, 200.0]],
            "gt_classes": [0],
            "ignore_boxes": [],
            "effective_gain": 1.0,
        }

    g0_metrics = {
        "A": evaluate_dataset(
            [metric_row([0.0, 0.0, 200.0, 200.0])]
        ),
        "C": evaluate_dataset(
            [
                metric_row(
                    (
                        [80.0, 0.0, 280.0, 200.0]
                        if recoverable
                        else c_box
                    ),
                    score=0.95 if recoverable else 0.9,
                    source=1 if recoverable else 0,
                    query=1 if recoverable else 0,
                )
            ]
        ),
    }

    def json_ready(value):
        if isinstance(value, dict):
            return {str(key): json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_ready(item) for item in value]
        return value.item() if hasattr(value, "item") else value

    atomic_write_json(
        evidence / "g0_metrics.json", json_ready(g0_metrics)
    )
    atomic_write_json(
        evidence / "g0_gate.json",
        {
            "status": "SBR_G0A_FAIL",
            "source_hash": source_commit,
            "checkpoint_hash": checkpoint_hash,
            "protocol_hash": protocol_hash,
            "dataset_signature": dataset["dataset_signature"],
        },
    )
    atomic_write_json(
        evidence / "independent_adjudication.json",
        {
            "status": "SBR_G0A_INDEPENDENT_FAIL",
            "decision": "FAIL",
            "independent_gate": "SBR_G0A_FAIL",
            "runner_status": "SBR_G0A_FAIL",
            "checksums_verified": True,
            "source_hash": source_commit,
            "checkpoint_hash": checkpoint_hash,
            "protocol_hash": protocol_hash,
            "dataset_signature": dataset["dataset_signature"],
        },
    )
    write_checksums(
        evidence / "checksums.sha256",
        [p for p in evidence.iterdir() if p.name != "checksums.sha256"],
        root=evidence,
    )

    manifest_dir = tmp_path / "portable"
    manifest_dir.mkdir()

    def entry(path: Path) -> dict[str, str]:
        return {
            "uri": Path(os.path.relpath(path, manifest_dir)).as_posix(),
            "sha256": sha256_file(path),
        }

    manifest = {
        "schema_version": "sbr-v2-audit-input/v1",
        "protocol_hash": protocol_hash,
        "source": {"commit": source_commit, "tree": source_tree},
        "original_evidence_root": {
            "uri": Path(os.path.relpath(evidence, manifest_dir)).as_posix()
        },
        "files": {
            "g0_manifest": entry(evidence / "g0_manifest.json"),
            "raw_views": entry(evidence / "raw_views.jsonl.gz"),
            "arm_predictions": entry(evidence / "arm_predictions.jsonl.gz"),
            "g0_metrics": entry(evidence / "g0_metrics.json"),
            "g0_gate": entry(evidence / "g0_gate.json"),
            "independent_adjudication": entry(
                evidence / "independent_adjudication.json"
            ),
            "original_checksums": entry(evidence / "checksums.sha256"),
            "checkpoint": entry(checkpoint),
            "image_list": entry(evidence / "image_list.json"),
            "dataset_yaml": entry(dataset_yaml),
        },
        "dataset": {
            "root": {
                "uri": Path(os.path.relpath(dataset_root, manifest_dir)).as_posix(),
                "sha256": dataset["dataset_signature"],
            },
            "split": "val",
        },
    }
    manifest_path = manifest_dir / "audit_input.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path, evidence, dataset_root


def _clean_provenance() -> dict[str, object]:
    return {
        "commit": "d" * 40,
        "branch": "codex/test",
        "clean_tracked": True,
        "untracked": False,
        "source_tree_hash": "e" * 64,
    }


def _reseal_changed_input(
    manifest: Path, evidence: Path, key: str, changed_path: Path
) -> None:
    write_checksums(
        evidence / "checksums.sha256",
        [p for p in evidence.iterdir() if p.name != "checksums.sha256"],
        root=evidence,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][key]["sha256"] = sha256_file(changed_path)
    payload["files"]["original_checksums"]["sha256"] = sha256_file(
        evidence / "checksums.sha256"
    )
    atomic_write_json(manifest, payload)


def _gzip_rows(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_parser_exposes_only_operational_arguments():
    from scripts.audit_sbr_v2 import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["--input-manifest", "in.json", "--output", "out"]
    )
    assert args.workers == 0
    assert vars(args) == {
        "input_manifest": Path("in.json"),
        "output": Path("out"),
        "workers": 0,
    }
    for forbidden in (
        "--ios",
        "--conf",
        "--max-det",
        "--large-threshold",
        "--mechanism-share",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--input-manifest",
                    "in.json",
                    "--output",
                    "out",
                    forbidden,
                    "1",
                ]
            )


def test_manifest_checksum_mismatch_fails_closed(tmp_path: Path):
    from scripts.audit_sbr_v2 import validate_input_manifest

    manifest, _, _ = _make_input(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"]["raw_views"]["sha256"] = "0" * 64
    atomic_write_json(manifest, payload)
    with pytest.raises(ValueError, match="checksum"):
        validate_input_manifest(manifest, tmp_path / "out")


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda m: m.pop("protocol_hash"), "protocol"),
        (lambda m: m["source"].pop("tree"), "source"),
        (
            lambda m: m["files"]["raw_views"].update(
                {"uri": "../../outside.jsonl.gz"}
            ),
            "escape|outside|evidence",
        ),
    ],
)
def test_manifest_missing_provenance_or_path_escape_fails_closed(
    tmp_path: Path, mutation, match: str
):
    from scripts.audit_sbr_v2 import validate_input_manifest

    manifest, _, _ = _make_input(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(payload)
    atomic_write_json(manifest, payload)
    with pytest.raises(ValueError, match=match):
        validate_input_manifest(manifest, tmp_path / "out")


def test_output_must_be_outside_original_evidence_and_inputs(tmp_path: Path):
    from scripts.audit_sbr_v2 import validate_input_manifest

    manifest, evidence, _ = _make_input(tmp_path)
    with pytest.raises(ValueError, match="output"):
        validate_input_manifest(manifest, evidence / "v2")
    with pytest.raises(ValueError, match="output"):
        validate_input_manifest(manifest, manifest)


def test_synthetic_end_to_end_eligible_writes_exact_atomic_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())

    manifest, _, _ = _make_input(tmp_path)
    output = tmp_path / "out"
    args = cli.build_parser().parse_args(
        ["--input-manifest", str(manifest), "--output", str(output)]
    )
    assert cli.run(args) == 0
    assert {p.name for p in output.iterdir()} == REQUIRED_OUTPUTS
    gate = json.loads((output / "primary_gate.json").read_text())
    assert gate["status"] == "SBR_V2_AUDIT_ELIGIBLE"
    summary = json.loads((output / "attribution_summary.json").read_text())
    assert summary["primary_ap75"]["denominator"] == 1
    assert summary["primary_ap75"]["mixed_cluster_localization"] == 1
    invariants = json.loads((output / "invariants.json").read_text())
    assert invariants["passed"] is True
    audit_manifest = json.loads((output / "audit_manifest.json").read_text())
    assert audit_manifest["original_g0_decision"]["gate_status"] == "SBR_G0A_FAIL"
    assert len(
        audit_manifest["original_g0_decision"]["checksums_sha256"]
    ) == 64
    assert audit_manifest["limitations"]
    assert audit_manifest["primary_command"]["effective_workers"] == 0
    with gzip.open(
        output / "attribution_events.jsonl.gz", "rt", encoding="utf-8"
    ) as fh:
        events = [json.loads(line) for line in fh]
    assert len({(e["image_id"], e["gt_index"], e["iou_threshold"]) for e in events}) == len(events)


def test_zero_denominator_stops_and_repeat_deterministic_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())

    manifest, _, _ = _make_input(tmp_path, recoverable=False)
    hashes = []
    for name in ("out-a", "out-b"):
        output = tmp_path / name
        args = cli.build_parser().parse_args(
            ["--input-manifest", str(manifest), "--output", str(output)]
        )
        assert cli.run(args) == 0
        gate = json.loads((output / "primary_gate.json").read_text())
        assert gate["status"] == "SBR_V2_AUDIT_STOP"
        audit_manifest = json.loads((output / "audit_manifest.json").read_text())
        hashes.append(audit_manifest["deterministic_evidence_hash"])
    assert hashes[0] == hashes[1]


def test_false_invariant_forces_stop():
    from scripts.audit_sbr_v2 import primary_gate_status

    report = {
        "mechanism_gate": "PASS",
        "recoverable_upper_bound_gate": "PASS",
    }
    assert primary_gate_status(report, {"passed": False}) == "SBR_V2_AUDIT_STOP"
    assert (
        primary_gate_status(report, {"passed": True})
        == "SBR_V2_AUDIT_ELIGIBLE"
    )


@pytest.mark.parametrize("corruption", ["truncated", "malformed"])
def test_bad_gzip_or_row_fails_without_complete_output(
    tmp_path: Path,
    corruption: str,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.audit_sbr_v2 as cli

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())

    manifest, evidence, _ = _make_input(tmp_path)
    raw = evidence / "raw_views.jsonl.gz"
    if corruption == "truncated":
        raw.write_bytes(raw.read_bytes()[:-6])
    else:
        with raw.open("wb") as file_handle:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=file_handle, mtime=0
            ) as gzip_handle:
                gzip_handle.write(b'{"image_id":"one.jpg"\n')
    write_checksums(
        evidence / "checksums.sha256",
        [p for p in evidence.iterdir() if p.name != "checksums.sha256"],
        root=evidence,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"]["raw_views"]["sha256"] = sha256_file(raw)
    atomic_write_json(manifest, payload)
    output = tmp_path / "out"
    args = cli.build_parser().parse_args(
        ["--input-manifest", str(manifest), "--output", str(output)]
    )
    with pytest.raises(ValueError):
        cli.run(args)
    assert not output.exists()
    assert not list(tmp_path.glob(".out.audit-tmp-*"))


def test_original_g0_must_be_sealed_immutable_fail(tmp_path: Path):
    from scripts.audit_sbr_v2 import validate_input_manifest

    manifest, evidence, _ = _make_input(tmp_path)
    gate_path = evidence / "g0_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["status"] = "SBR_G0A_PASS"
    atomic_write_json(gate_path, gate)
    write_checksums(
        evidence / "checksums.sha256",
        [p for p in evidence.iterdir() if p.name != "checksums.sha256"],
        root=evidence,
    )
    portable = json.loads(manifest.read_text(encoding="utf-8"))
    portable["files"]["g0_gate"]["sha256"] = sha256_file(gate_path)
    portable["files"]["original_checksums"]["sha256"] = sha256_file(
        evidence / "checksums.sha256"
    )
    atomic_write_json(manifest, portable)
    with pytest.raises(ValueError, match="FAIL"):
        validate_input_manifest(manifest, tmp_path / "out")


def test_run_requires_clean_stable_audit_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    manifest, _, _ = _make_input(tmp_path)
    dirty = _clean_provenance()
    dirty["clean_tracked"] = False
    monkeypatch.setattr(cli, "git_provenance", lambda _path: dirty)
    args = cli.build_parser().parse_args(
        ["--input-manifest", str(manifest), "--output", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="clean"):
        cli.run(args)


def test_incomplete_c_view_manifest_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())
    manifest, evidence, _ = _make_input(tmp_path)
    raw_path = evidence / "raw_views.jsonl.gz"
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        if row["arm"] == "C":
            row["view_manifest"] = row["view_manifest"][:-1]
    atomic_write_jsonl_gz(raw_path, rows)
    write_checksums(
        evidence / "checksums.sha256",
        [p for p in evidence.iterdir() if p.name != "checksums.sha256"],
        root=evidence,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"]["raw_views"]["sha256"] = sha256_file(raw_path)
    payload["files"]["original_checksums"]["sha256"] = sha256_file(
        evidence / "checksums.sha256"
    )
    atomic_write_json(manifest, payload)
    args = cli.build_parser().parse_args(
        ["--input-manifest", str(manifest), "--output", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="view_manifest"):
        cli.run(args)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "sealed_name",
    [
        "g0_gate.json",
        "independent_adjudication.json",
        "checksums.sha256",
    ],
)
def test_manifest_directly_pins_original_seals(
    tmp_path: Path, sealed_name: str
):
    from scripts.audit_sbr_v2 import validate_input_manifest

    manifest, evidence, _ = _make_input(tmp_path)
    target = evidence / sealed_name
    if sealed_name == "checksums.sha256":
        target.write_bytes(target.read_bytes() + b"\n")
    else:
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["untrusted_reseal"] = True
        atomic_write_json(target, payload)
        write_checksums(
            evidence / "checksums.sha256",
            [
                path
                for path in evidence.iterdir()
                if path.name != "checksums.sha256"
            ],
            root=evidence,
        )
    with pytest.raises(ValueError, match="checksum"):
        validate_input_manifest(manifest, tmp_path / "out")


def test_checkpoint_bytes_must_match_all_frozen_provenance(tmp_path: Path):
    from scripts.audit_sbr_v2 import validate_input_manifest

    manifest, _, _ = _make_input(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = (manifest.parent / payload["files"]["checkpoint"]["uri"]).resolve()
    checkpoint.write_bytes(b"different-checkpoint")
    payload["files"]["checkpoint"]["sha256"] = sha256_file(checkpoint)
    atomic_write_json(manifest, payload)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_input_manifest(manifest, tmp_path / "out")


def test_protocol_payload_is_recomputed_and_frozen_exact(tmp_path: Path):
    from scripts.audit_sbr_v2 import validate_input_manifest

    manifest, evidence, _ = _make_input(tmp_path)
    g0_path = evidence / "g0_manifest.json"
    g0 = json.loads(g0_path.read_text(encoding="utf-8"))
    g0["protocol"]["max_det"] = 301
    atomic_write_json(g0_path, g0)
    write_checksums(
        evidence / "checksums.sha256",
        [p for p in evidence.iterdir() if p.name != "checksums.sha256"],
        root=evidence,
    )
    portable = json.loads(manifest.read_text(encoding="utf-8"))
    portable["files"]["g0_manifest"]["sha256"] = sha256_file(g0_path)
    portable["files"]["original_checksums"]["sha256"] = sha256_file(
        evidence / "checksums.sha256"
    )
    atomic_write_json(manifest, portable)
    with pytest.raises(ValueError, match="protocol"):
        validate_input_manifest(manifest, tmp_path / "out")


def test_dataset_image_order_must_match_frozen_list_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    manifest, _, _ = _make_input(tmp_path)
    real_loader = cli.load_dataset

    def reordered(*args, **kwargs):
        dataset = real_loader(*args, **kwargs)
        dataset["image_list"] = ["unexpected.jpg", *dataset["image_list"]]
        return dataset

    monkeypatch.setattr(cli, "load_dataset", reordered)
    with pytest.raises(ValueError, match="order"):
        cli.validate_input_manifest(manifest, tmp_path / "out")


def test_arm_prediction_records_are_canonical_bound_to_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())
    manifest, evidence, _ = _make_input(tmp_path)
    arms = evidence / "arm_predictions.jsonl.gz"
    rows = _gzip_rows(arms)
    rows[0]["records"][0]["score"] = 0.8
    atomic_write_jsonl_gz(arms, rows)
    _reseal_changed_input(manifest, evidence, "arm_predictions", arms)
    args = cli.build_parser().parse_args(
        ["--input-manifest", str(manifest), "--output", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="records|raw"):
        cli.run(args)


def test_frozen_a_c_predictions_are_exactly_reproduced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())
    manifest, evidence, _ = _make_input(tmp_path)
    arms = evidence / "arm_predictions.jsonl.gz"
    rows = _gzip_rows(arms)
    rows[2]["predictions"][0]["box"][0] = 1.0
    atomic_write_jsonl_gz(arms, rows)
    _reseal_changed_input(manifest, evidence, "arm_predictions", arms)
    args = cli.build_parser().parse_args(
        ["--input-manifest", str(manifest), "--output", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="prediction"):
        cli.run(args)


def test_arm_prediction_requires_exact_six_block_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())
    manifest, evidence, _ = _make_input(tmp_path)
    arms = evidence / "arm_predictions.jsonl.gz"
    rows = _gzip_rows(arms)
    atomic_write_jsonl_gz(arms, rows[:-1])
    _reseal_changed_input(manifest, evidence, "arm_predictions", arms)
    args = cli.build_parser().parse_args(
        ["--input-manifest", str(manifest), "--output", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="6|block|row"):
        cli.run(args)


def test_recomputed_a_c_metrics_must_equal_sealed_g0_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())
    manifest, evidence, _ = _make_input(tmp_path)
    metrics_path = evidence / "g0_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["A"]["mAP50-95"] += 0.01
    atomic_write_json(metrics_path, metrics)
    _reseal_changed_input(manifest, evidence, "g0_metrics", metrics_path)
    args = cli.build_parser().parse_args(
        ["--input-manifest", str(manifest), "--output", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="metrics"):
        cli.run(args)


def test_primary_reconstructs_c_clusters_exactly_once_per_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli
    import src.sbr_v2_audit as audit

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())
    calls = 0
    real_reconstruct = audit.reconstruct_c_clusters

    def counted(raw):
        nonlocal calls
        calls += 1
        return real_reconstruct(raw)

    monkeypatch.setattr(audit, "reconstruct_c_clusters", counted)
    manifest, _, _ = _make_input(tmp_path)
    args = cli.build_parser().parse_args(
        ["--input-manifest", str(manifest), "--output", str(tmp_path / "out")]
    )
    assert cli.run(args) == 0
    assert calls == 1


def test_nonzero_workers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())
    manifest, _, _ = _make_input(tmp_path)
    args = cli.build_parser().parse_args(
        [
            "--input-manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out"),
            "--workers",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="workers"):
        cli.run(args)


def test_zero_detection_a_c_rows_are_proven_by_empty_frozen_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.audit_sbr_v2 as cli

    monkeypatch.setattr(cli, "git_provenance", lambda _path: _clean_provenance())
    manifest, evidence, _ = _make_input(tmp_path)
    raw_path = evidence / "raw_views.jsonl.gz"
    arms_path = evidence / "arm_predictions.jsonl.gz"
    metrics_path = evidence / "g0_metrics.json"
    atomic_write_jsonl_gz(raw_path, [])
    arm_rows = _gzip_rows(arms_path)
    for index in (0, 2):
        arm_rows[index]["records"] = []
        arm_rows[index]["predictions"] = []
    atomic_write_jsonl_gz(arms_path, arm_rows)
    empty_row = {
        "image_id": "one.jpg",
        "width": 640,
        "height": 640,
        "pred_boxes": [],
        "pred_scores": [],
        "pred_classes": [],
        "pred_source": [],
        "pred_query": [],
        "gt_boxes": [[0.0, 0.0, 200.0, 200.0]],
        "gt_classes": [0],
        "ignore_boxes": [],
        "effective_gain": 1.0,
    }
    empty_metrics = evaluate_dataset([empty_row])
    json_metrics = json.loads(
        json.dumps(empty_metrics, default=lambda value: value.item())
    )
    atomic_write_json(
        metrics_path, {"A": json_metrics, "C": json_metrics}
    )
    write_checksums(
        evidence / "checksums.sha256",
        [p for p in evidence.iterdir() if p.name != "checksums.sha256"],
        root=evidence,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for key, path in (
        ("raw_views", raw_path),
        ("arm_predictions", arms_path),
        ("g0_metrics", metrics_path),
        ("original_checksums", evidence / "checksums.sha256"),
    ):
        payload["files"][key]["sha256"] = sha256_file(path)
    atomic_write_json(manifest, payload)
    args = cli.build_parser().parse_args(
        ["--input-manifest", str(manifest), "--output", str(tmp_path / "out")]
    )
    assert cli.run(args) == 0
    gate = json.loads(
        (tmp_path / "out" / "primary_gate.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "SBR_V2_AUDIT_STOP"


def test_frozen_g0_metrics_use_global_xyxy_not_fused_box():
    import scripts.audit_sbr_v2 as cli

    image = {
        "relative_path": "one.jpg",
        "width": 640,
        "height": 640,
        "gt_boxes": [],
        "gt_classes": [],
        "ignore_boxes": [],
    }
    prediction = Detection(
        box=(40.0, 0.0, 240.0, 200.0),
        global_xyxy=(80.0, 0.0, 280.0, 200.0),
        score=0.9,
        class_id=0,
        source_order=1,
        query_index=7,
    )

    row = cli._metric_row(
        image, [prediction], frozen_global_xyxy=True
    )

    assert row["pred_boxes"] == [[80.0, 0.0, 280.0, 200.0]]


@pytest.mark.parametrize(
    ("arm", "source", "network_box", "view_box", "global_box"),
    [
        (
            "C",
            2,
            [-0.4688, 0.0, 20.0, 20.0],
            [0.0, 0.0, 20.0, 20.0],
            [256.0, 0.0, 276.0, 20.0],
        ),
        (
            "A",
            0,
            [630.0, 630.0, 641.25, 641.0],
            [630.0, 630.0, 640.0, 640.0],
            [630.0, 630.0, 640.0, 640.0],
        ),
    ],
)
def test_raw_parser_accepts_legacy_network_overflow_after_view_clamp(
    arm, source, network_box, view_box, global_box
):
    import scripts.audit_sbr_v2 as cli

    row = _raw(
        "one.jpg",
        arm,
        source,
        0,
        tuple(global_box),
        index=0,
    )
    row["network_xyxy"] = network_box
    row["view_xyxy"] = view_box
    if source == 2:
        row["tile_bounds"] = [256, 0, 640, 384]
    row["_audit_original_index"] = 0

    detection = cli._parse_raw_detection(
        row, expected_image_id="one.jpg"
    )

    assert detection.network_xyxy == tuple(network_box)
    assert detection.view_xyxy == tuple(view_box)
    assert detection.global_xyxy == tuple(global_box)


@pytest.mark.parametrize(
    "network_box",
    [
        [float("nan"), 0.0, 20.0, 20.0],
        [0.0, 0.0, float("inf"), 20.0],
        [10.0, 0.0, 10.0, 20.0],
    ],
)
def test_raw_parser_still_rejects_invalid_network_geometry(network_box):
    import scripts.audit_sbr_v2 as cli

    row = _raw(
        "one.jpg",
        "A",
        0,
        0,
        (0.0, 0.0, 20.0, 20.0),
        index=0,
    )
    row["network_xyxy"] = network_box
    row["_audit_original_index"] = 0

    with pytest.raises(ValueError, match="network_xyxy"):
        cli._parse_raw_detection(row, expected_image_id="one.jpg")
