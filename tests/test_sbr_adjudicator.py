import gzip
import json
from pathlib import Path

from src.sbr_artifacts import atomic_write_json, atomic_write_jsonl_gz, write_checksums


def _evidence(tmp_path: Path) -> Path:
    root = tmp_path / "evidence"
    root.mkdir()
    digest = "a" * 40  # Git commit IDs are SHA-1 in the production runner.
    images = [f"{i:04d}.jpg" for i in range(548)]
    manifest = {
        "mode": "g0-a",
        "source": {"commit": digest},
        "source_hash": digest,
        "checkpoint_hash": "b" * 64,
        "dataset_signature": "c" * 64,
        "protocol_hash": "d" * 64,
        "image_count": 548,
        "image_list": images,
    }
    metrics = {
        "A": {
            "AP-tiny-SBR": 0.10,
            "mAP50-95": 0.20,
            "AP75": 0.50,
            "AP-large-SBR": 0.60,
            "tiny_recall": 0.30,
        },
        "C": {
            "AP-tiny-SBR": 0.110001,
            "mAP50-95": 0.203001,
            "AP75": 0.498001,
            "AP-large-SBR": 0.595001,
            "tiny_recall": 0.320001,
        },
    }
    deltas = {
        "AP-tiny-SBR": 0.010001,
        "mAP50-95": 0.003001,
        "AP75": -0.001999,
        "AP-large-SBR": -0.004999,
        "tiny_recall": 0.020001,
    }
    gate = {
        "status": "SBR_G0A_FAIL",  # runner status is intentionally ignored
        "source_hash": digest,
        "checkpoint_hash": "b" * 64,
        "dataset_signature": "c" * 64,
        "protocol_hash": "d" * 64,
    }
    atomic_write_json(root / "g0_manifest.json", manifest)
    atomic_write_json(root / "g0_metrics.json", metrics)
    atomic_write_json(root / "g0_deltas.json", deltas)
    atomic_write_json(root / "g0_gate.json", gate)
    raw = [{"image_id": images[0], "arm": "A", "view_id": "full", "source_order": 0}]
    arms = [{"image_id": images[0], "arm": "A", "predictions": []}]
    atomic_write_jsonl_gz(root / "raw_views.jsonl.gz", raw)
    atomic_write_jsonl_gz(root / "arm_predictions.jsonl.gz", arms)
    files = [p for p in root.iterdir() if p.is_file()]
    write_checksums(root / "checksums.sha256", files, root=root)
    return root


def test_adjudicator_recomputes_gate_without_trusting_runner_status(tmp_path):
    from scripts.adjudicate_sbr_g0 import adjudicate_evidence

    report = adjudicate_evidence(_evidence(tmp_path))
    assert report["decision"] == "PASS"
    assert report["independent_gate"] == "SBR_G0A_PASS"
    assert report["runner_status"] == "SBR_G0A_FAIL"
    assert report["checksums_verified"] is True
    assert report["gate_updated"] is True
    assert json.loads((tmp_path / "evidence" / "g0_gate.json").read_text())["status"] == "SBR_G0A_PASS"
    assert json.loads((tmp_path / "evidence" / "independent_adjudication.json").read_text())["decision"] == "PASS"


def test_adjudicator_fails_on_tampered_metrics(tmp_path):
    from scripts.adjudicate_sbr_g0 import adjudicate_evidence

    root = _evidence(tmp_path)
    metrics_path = root / "g0_metrics.json"
    payload = json.loads(metrics_path.read_text())
    payload["C"]["AP75"] = 0.2
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    report = adjudicate_evidence(root)
    assert report["decision"] == "FAIL"
    assert report["checksums_verified"] is False
    assert "checksum" in report["error"]
