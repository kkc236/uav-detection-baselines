from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate_bpdd_formal as cli


def _inputs(tmp_path: Path, variant: str = "fdr_bpdd") -> tuple[Path, Path, Path]:
    run = tmp_path / variant
    run.mkdir()
    manifest = {
        "format_version": 1,
        "run_identity": {
            "source_sha256": "S" * 64,
            "protocol_sha256": "P" * 64,
            "fdr_protocol_sha256": "F" * 64,
            "initial_state_sha256": "I" * 64,
            "run_id": f"{variant}-formal-seed0-authority",
            "stage": "formal",
            "variant": variant,
            "seed": 0,
        },
        "data": str((tmp_path / "formal.yaml").resolve()),
    }
    (run / "bpdd-run.json").write_text(json.dumps(manifest), "utf-8")
    checkpoint = run / "weights" / "epoch99.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"exact-final")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    publication = tmp_path / "publication.json"
    publication.write_text(
        json.dumps(
            {
                "format_version": 1,
                "run_id": manifest["run_identity"]["run_id"],
                "variant": variant,
                "stage": "formal",
                "completed_epoch": 100,
                "checkpoint": {
                    "asset_id": 100,
                    "asset_name": f"{variant}-epoch-0100.pt",
                    "bytes": checkpoint.stat().st_size,
                    "sha256": digest,
                },
                "release_url": "https://example.invalid/release/final",
                "verified": True,
            }
        ),
        "utf-8",
    )
    return run, checkpoint, publication


def test_parser_exposes_only_fixed_protocol_inputs() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--run-dir",
            "run",
            "--checkpoint",
            "epoch99.pt",
            "--publication-manifest",
            "published.json",
            "--dataset-root",
            "VisDrone",
            "--output",
            "evaluation.json",
        ]
    )
    assert args.run_dir == Path("run")
    assert args.checkpoint == Path("epoch99.pt")
    assert args.publication_manifest == Path("published.json")
    assert args.dataset_root == Path("VisDrone")
    assert args.output == Path("evaluation.json")
    assert not any(
        option in parser._option_string_actions
        for option in ("--imgsz", "--batch", "--workers", "--conf", "--max-det", "--nms")
    )


def test_evaluation_constants_are_frozen() -> None:
    assert cli.EVALUATION_PROTOCOL == {
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "conf": 0.001,
        "max_det": 300,
        "nms": False,
    }


def test_publication_input_accepts_the_real_contiguous_epoch100_ledger(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "publication-ledger.jsonl"
    rows = [
        {
            "completed_epoch": epoch,
            "verified": True,
            "checkpoint": {"sha256": f"{epoch:064x}"[-64:]},
        }
        for epoch in range(1, 101)
    ]
    ledger.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assert cli._read_publication(ledger) == rows[-1]

    rows[50]["completed_epoch"] = 999
    ledger.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="continuous|ledger"):
        cli._read_publication(ledger)


def test_formal_evaluation_binds_run_data_publication_and_same_inference_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, checkpoint, publication = _inputs(tmp_path)
    dataset_root = tmp_path / "VisDrone"
    dataset_root.mkdir()
    output = tmp_path / "evaluation.json"
    loaded = SimpleNamespace(
        model=object(),
        metadata={
            "kind": "exact-final-ema",
            "completed_epoch": 100,
            "raw_epoch": 99,
            "sha256": json.loads(publication.read_text("utf-8"))["checkpoint"]["sha256"],
            "sha256_verified": True,
            "source_field": "ema",
            "ema_state_sha256": "E" * 64,
            "strict_fdr_inference_graph": True,
        },
    )
    monkeypatch.setattr(cli, "load_exact_final_checkpoint", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(
        cli,
        "dataset_signature",
        lambda path: {
            "file_count": 14038,
            "sha256": cli.BPDD_PROTOCOL["dataset"]["sha256"],
        },
    )
    calls = []

    def fake_validation(model, *, data, save_dir):
        calls.append((model, Path(data), Path(save_dir)))
        return {
            "metrics": {
                "precision": 0.5,
                "recall": 0.4,
                "f1": 4 / 9,
                "map50": 0.48,
                "map75": 0.29,
                "map": 0.30,
            },
            "classes": {name: 0.2 for name in cli.CATEGORY_NAMES},
            "class_details": {
                name: {"id": index, "map50": 0.3, "map75": 0.2, "map": 0.2}
                for index, name in enumerate(cli.CATEGORY_NAMES)
            },
            "scales": {name: 0.2 for name in cli.SCALE_NAMES},
            "scale_details": {
                name: {"gt": 1, "map50": 0.3, "map75": 0.2, "map": 0.2}
                for name in cli.SCALE_NAMES
            },
            "processed_images": 548,
            "prediction_passes": 1,
        }

    monkeypatch.setattr(cli, "run_official_validation", fake_validation)

    report = cli.evaluate_formal_checkpoint(
        run_dir=run,
        checkpoint=checkpoint,
        publication_manifest=publication,
        dataset_root=dataset_root,
        output=output,
    )

    run_manifest = json.loads((run / "bpdd-run.json").read_text("utf-8"))
    assert calls == [(loaded.model, Path(run_manifest["data"]), output.parent / "validator")]
    assert report["evaluation_identity"] == {
        **run_manifest["run_identity"],
        "data": run_manifest["data"],
        "dataset_sha256": cli.BPDD_PROTOCOL["dataset"]["sha256"],
    }
    assert report["checkpoint"]["remote_published"] is True
    assert report["checkpoint"]["remote_asset"].endswith("fdr_bpdd-epoch-0100.pt")
    assert report["checkpoint"]["ema_state_sha256"] == "E" * 64
    assert report["evaluation_protocol"] == cli.EVALUATION_PROTOCOL
    assert report["prediction_passes"] == 1
    assert report["processed_images"] == 548
    assert json.loads(output.read_text("utf-8")) == report


@pytest.mark.parametrize("field", ["completed_epoch", "run_id", "verified", "sha256"])
def test_publication_or_identity_drift_fails_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    run, checkpoint, publication = _inputs(tmp_path)
    payload = json.loads(publication.read_text("utf-8"))
    if field == "completed_epoch":
        payload[field] = 99
    elif field == "run_id":
        payload[field] = "wrong"
    elif field == "verified":
        payload[field] = False
    else:
        payload["checkpoint"][field] = "0" * 64
    publication.write_text(json.dumps(payload), "utf-8")
    monkeypatch.setattr(
        cli,
        "run_official_validation",
        lambda *args, **kwargs: pytest.fail("invalid authority must fail before val"),
    )

    with pytest.raises(ValueError, match="publication|SHA256|identity|epoch"):
        cli.evaluate_formal_checkpoint(
            run_dir=run,
            checkpoint=checkpoint,
            publication_manifest=publication,
            dataset_root=tmp_path,
            output=tmp_path / "evaluation.json",
        )
