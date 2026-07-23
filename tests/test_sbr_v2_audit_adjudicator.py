import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = "sbr-v2-audit-evidence/v1"
SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "required_artifacts": [
        "audit_manifest.json",
        "attribution_events.jsonl.gz",
        "attribution_summary.json",
        "upper_bound_metrics.json",
        "invariants.json",
        "primary_gate.json",
        "checksums.sha256",
    ],
    "primary_event_id": ["image_id", "gt_index", "iou_threshold"],
    "primary_gate_inputs": [
        "mechanism_gate",
        "recoverable_upper_bound_gate",
        "invariants.passed",
    ],
}
CATEGORIES = (
    "mixed_cluster_localization",
    "final_300_truncation",
    "matching_competition",
    "class_or_candidate_loss",
    "other",
)
THRESHOLDS = tuple(round(0.50 + index * 0.05, 2) for index in range(10))


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def _write_events(path: Path, events) -> None:
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as gz:
        for event in events:
            gz.write(_canonical(event) + b"\n")


def _reseal(root: Path) -> None:
    paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    text = "".join(f"{_sha(path)}  {path.name}\n" for path in paths)
    (root / "checksums.sha256").write_text(text, encoding="utf-8")


def _events() -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        for gt_index in range(5):
            rows.append(
                {
                    "image_id": "images/0001.jpg",
                    "gt_index": gt_index,
                    "iou_threshold": threshold,
                    "category": (
                        "mixed_cluster_localization"
                        if gt_index < 3
                        else "other"
                    ),
                    "counterfactual_recovers": gt_index < 3,
                }
            )
    return rows


def _summarize(events: list[dict]) -> dict:
    secondary = {}
    for threshold in THRESHOLDS:
        rows = [
            event
            for event in events
            if event["iou_threshold"] == threshold
        ]
        counts = Counter(event["category"] for event in rows)
        secondary[f"{threshold:.2f}"] = {
            "denominator": len(rows),
            "category_counts": {
                category: counts.get(category, 0) for category in CATEGORIES
            },
        }
    primary = [
        event for event in events if event["iou_threshold"] == 0.75
    ]
    counts = Counter(event["category"] for event in primary)
    return {
        "primary_ap75": {
            "unique_event_key": ["image_id", "gt_index", "iou_threshold"],
            "denominator": len(primary),
            "mixed_cluster_localization": counts[
                "mixed_cluster_localization"
            ],
            "mechanism_share": 0.6,
            "category_counts": {
                category: counts.get(category, 0) for category in CATEGORIES
            },
        },
        "secondary_repeated_measures": {
            "note": "The ten thresholds are pooled repeated measures, not independent samples.",
            "thresholds": secondary,
        },
    }


def _eligible_primary_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "primary"
    root.mkdir()
    events = _events()
    _write_events(root / "attribution_events.jsonl.gz", events)
    _write_json(root / "attribution_summary.json", _summarize(events))
    invariants = {
        "raw_hash_equal": True,
        "cluster_hash_equal": True,
        "cluster_count_equal": True,
        "scores_equal": True,
        "classes_equal": True,
        "selected_cluster_ids_equal": True,
        "singleton_preservation": 1.0,
        "passed": True,
        "singleton_total": 2,
        "singleton_preserved": 2,
        "image_count": 1,
        "per_image": [
            {
                "image_id": "images/0001.jpg",
                "raw_hash_equal": True,
                "cluster_hash_equal": True,
                "cluster_count_equal": True,
                "scores_equal": True,
                "classes_equal": True,
                "selected_cluster_ids_equal": True,
                "singleton_preservation": 1.0,
                "passed": True,
                "singleton_total": 2,
                "singleton_preserved": 2,
            }
        ],
    }
    upper = {
        "mechanism_share_ap75": 0.6,
        "mechanism_gate": "PASS",
        "a_metrics": {"AP-large-SBR": 0.5},
        "c_metrics": {"AP-large-SBR": 0.45},
        "v2_metrics": {"AP-large-SBR": 0.495},
        "v2_minus_a": {"AP-large-SBR": -0.005},
        "v2_minus_c": {"AP-large-SBR": 0.045},
        "recoverable_upper_bound_gate": "PASS",
        "invariants": {
            key: value
            for key, value in invariants.items()
            if key
            in {
                "raw_hash_equal",
                "cluster_hash_equal",
                "cluster_count_equal",
                "scores_equal",
                "classes_equal",
                "selected_cluster_ids_equal",
                "singleton_preservation",
                "passed",
            }
        },
    }
    _write_json(root / "upper_bound_metrics.json", upper)
    _write_json(root / "invariants.json", invariants)
    _write_json(
        root / "primary_gate.json",
        {
            "status": "SBR_V2_AUDIT_ELIGIBLE",
            "mechanism_gate": "PASS",
            "recoverable_upper_bound_gate": "PASS",
            "invariants_passed": True,
        },
    )
    deterministic_names = (
        "attribution_events.jsonl.gz",
        "attribution_summary.json",
        "upper_bound_metrics.json",
        "invariants.json",
        "primary_gate.json",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA,
        "schema_hash": hashlib.sha256(_canonical(SCHEMA)).hexdigest(),
        "frozen_constants": {
            "mechanism_share_threshold": 0.60,
            "large_ap_tolerance": -0.005,
            "primary_iou_threshold": 0.75,
            "secondary_iou_thresholds": list(THRESHOLDS),
        },
        "input_manifest": {
            "uri": "/frozen/input_manifest.json",
            "sha256": "1" * 64,
        },
        "audit_source": {
            "commit": "2" * 40,
            "source_tree_hash": "3" * 64,
            "script_sha256": "4" * 64,
        },
        "image_count": 1,
        "image_order_hash": "5" * 64,
        "deterministic_artifact_hashes": {
            name: _sha(root / name) for name in deterministic_names
        },
    }
    manifest["deterministic_evidence_hash"] = hashlib.sha256(
        _canonical(manifest["deterministic_artifact_hashes"])
    ).hexdigest()
    _write_json(root / "audit_manifest.json", manifest)
    _reseal(root)
    return root


def _load_events(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_adjudicator_recomputes_eligible_gate_without_primary_import(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    root = _eligible_primary_fixture(tmp_path)
    report = adjudicate_evidence(root)

    assert report["decision"] == "PASS"
    assert report["status"] == "SBR_V2_AUDIT_INDEPENDENT_PASS"
    assert report["independent_gate"] == "SBR_V2_AUDIT_ELIGIBLE"
    assert report["checksums_verified"] is True
    assert report["checksums_regenerated"] is True
    assert report["primary_gate_agrees"] is True
    assert report["event_count"] == 50
    assert report["input_manifest_sha256"] == "1" * 64


def test_adjudicator_imports_only_stdlib_and_numpy():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "adjudicate_sbr_v2_audit.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert imported <= {
        "__future__",
        "argparse",
        "collections",
        "gzip",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "platform",
        "subprocess",
        "sys",
        "tempfile",
        "typing",
        "numpy",
    }
    assert not {"src", "scripts"} & imported


def test_adjudicator_fails_on_checksum_tampering_without_resealing(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    root = _eligible_primary_fixture(tmp_path)
    summary = json.loads(
        (root / "attribution_summary.json").read_text(encoding="utf-8")
    )
    summary["primary_ap75"]["denominator"] = 99
    _write_json(root / "attribution_summary.json", summary)

    report = adjudicate_evidence(root)

    assert report["decision"] == "FAIL"
    assert report["checksums_verified"] is False
    assert report["checksums_regenerated"] is False
    assert "checksum" in report["error"].lower()


def test_adjudicator_fails_on_resealed_event_summary_tampering(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    root = _eligible_primary_fixture(tmp_path)
    events = _load_events(root / "attribution_events.jsonl.gz")
    events[0]["category"] = "other"
    _write_events(root / "attribution_events.jsonl.gz", events)
    manifest = json.loads((root / "audit_manifest.json").read_text())
    manifest["deterministic_artifact_hashes"][
        "attribution_events.jsonl.gz"
    ] = _sha(root / "attribution_events.jsonl.gz")
    manifest["deterministic_evidence_hash"] = hashlib.sha256(
        _canonical(manifest["deterministic_artifact_hashes"])
    ).hexdigest()
    _write_json(root / "audit_manifest.json", manifest)
    _reseal(root)

    report = adjudicate_evidence(root)

    assert report["decision"] == "FAIL"
    assert "summary" in report["error"].lower()


def test_adjudicator_rejects_checksum_path_escape(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    root = _eligible_primary_fixture(tmp_path)
    with (root / "checksums.sha256").open("a", encoding="utf-8") as fh:
        fh.write(f"{'a' * 64}  ../escape\n")

    report = adjudicate_evidence(root)

    assert report["decision"] == "FAIL"
    assert "unsafe" in report["error"].lower()


def test_adjudicator_rejects_nonfinite_event_even_if_resealed(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    root = _eligible_primary_fixture(tmp_path)
    events = _load_events(root / "attribution_events.jsonl.gz")
    events[0]["iou_threshold"] = float("nan")
    with gzip.GzipFile(
        filename=str(root / "attribution_events.jsonl.gz"),
        mode="wb",
        mtime=0,
    ) as gz:
        for event in events:
            gz.write(json.dumps(event, allow_nan=True).encode() + b"\n")
    manifest = json.loads((root / "audit_manifest.json").read_text())
    manifest["deterministic_artifact_hashes"][
        "attribution_events.jsonl.gz"
    ] = _sha(root / "attribution_events.jsonl.gz")
    manifest["deterministic_evidence_hash"] = hashlib.sha256(
        _canonical(manifest["deterministic_artifact_hashes"])
    ).hexdigest()
    _write_json(root / "audit_manifest.json", manifest)
    _reseal(root)

    report = adjudicate_evidence(root)

    assert report["decision"] == "FAIL"
    assert "non-finite" in report["error"].lower()


def test_adjudicator_rejects_duplicate_event_id(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    root = _eligible_primary_fixture(tmp_path)
    events = _load_events(root / "attribution_events.jsonl.gz")
    events.append(dict(events[0]))
    _write_events(root / "attribution_events.jsonl.gz", events)
    manifest = json.loads((root / "audit_manifest.json").read_text())
    manifest["deterministic_artifact_hashes"][
        "attribution_events.jsonl.gz"
    ] = _sha(root / "attribution_events.jsonl.gz")
    manifest["deterministic_evidence_hash"] = hashlib.sha256(
        _canonical(manifest["deterministic_artifact_hashes"])
    ).hexdigest()
    _write_json(root / "audit_manifest.json", manifest)
    _reseal(root)

    report = adjudicate_evidence(root)

    assert report["decision"] == "FAIL"
    assert "duplicate" in report["error"].lower()


def test_adjudicator_recomputes_mechanism_and_upper_bound_gates(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    root = _eligible_primary_fixture(tmp_path)
    upper = json.loads((root / "upper_bound_metrics.json").read_text())
    upper["v2_metrics"]["AP-large-SBR"] = 0.494
    upper["v2_minus_a"]["AP-large-SBR"] = -0.006
    _write_json(root / "upper_bound_metrics.json", upper)
    manifest = json.loads((root / "audit_manifest.json").read_text())
    manifest["deterministic_artifact_hashes"][
        "upper_bound_metrics.json"
    ] = _sha(root / "upper_bound_metrics.json")
    manifest["deterministic_evidence_hash"] = hashlib.sha256(
        _canonical(manifest["deterministic_artifact_hashes"])
    ).hexdigest()
    _write_json(root / "audit_manifest.json", manifest)
    _reseal(root)

    report = adjudicate_evidence(root)

    assert report["decision"] == "FAIL"
    assert "upper-bound" in report["error"].lower()


def test_adjudicator_fails_on_gate_or_invariant_disagreement(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    for mutation in ("gate", "invariant"):
        case = tmp_path / mutation
        case.mkdir()
        root = _eligible_primary_fixture(case)
        if mutation == "gate":
            path = root / "primary_gate.json"
            payload = json.loads(path.read_text())
            payload["status"] = "SBR_V2_AUDIT_STOP"
        else:
            path = root / "invariants.json"
            payload = json.loads(path.read_text())
            payload["scores_equal"] = False
        _write_json(path, payload)
        manifest = json.loads((root / "audit_manifest.json").read_text())
        manifest["deterministic_artifact_hashes"][path.name] = _sha(path)
        manifest["deterministic_evidence_hash"] = hashlib.sha256(
            _canonical(manifest["deterministic_artifact_hashes"])
        ).hexdigest()
        _write_json(root / "audit_manifest.json", manifest)
        _reseal(root)

        report = adjudicate_evidence(root)

        assert report["decision"] == "FAIL"
        assert mutation in report["error"].lower()


def test_adjudicator_rejects_false_per_image_invariant(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    root = _eligible_primary_fixture(tmp_path)
    path = root / "invariants.json"
    payload = json.loads(path.read_text())
    payload["per_image"][0]["scores_equal"] = False
    _write_json(path, payload)
    manifest = json.loads((root / "audit_manifest.json").read_text())
    manifest["deterministic_artifact_hashes"][path.name] = _sha(path)
    manifest["deterministic_evidence_hash"] = hashlib.sha256(
        _canonical(manifest["deterministic_artifact_hashes"])
    ).hexdigest()
    _write_json(root / "audit_manifest.json", manifest)
    _reseal(root)

    report = adjudicate_evidence(root)

    assert report["decision"] == "FAIL"
    assert "per-image invariant" in report["error"].lower()


def test_adjudicator_rejects_boolean_summary_count_or_gate_flag(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    for mutation in ("summary", "gate"):
        case = tmp_path / mutation
        case.mkdir()
        root = _eligible_primary_fixture(case)
        if mutation == "summary":
            path = root / "attribution_summary.json"
            payload = json.loads(path.read_text())
            payload["primary_ap75"]["category_counts"][
                "final_300_truncation"
            ] = False
        else:
            path = root / "primary_gate.json"
            payload = json.loads(path.read_text())
            payload["invariants_passed"] = 1
        _write_json(path, payload)
        manifest = json.loads((root / "audit_manifest.json").read_text())
        manifest["deterministic_artifact_hashes"][path.name] = _sha(path)
        manifest["deterministic_evidence_hash"] = hashlib.sha256(
            _canonical(manifest["deterministic_artifact_hashes"])
        ).hexdigest()
        _write_json(root / "audit_manifest.json", manifest)
        _reseal(root)

        report = adjudicate_evidence(root)

        assert report["decision"] == "FAIL"
        assert mutation in report["error"].lower()


def test_output_hash_has_explicit_non_self_referential_semantics(tmp_path):
    from scripts.adjudicate_sbr_v2_audit import adjudicate_evidence

    root = _eligible_primary_fixture(tmp_path)
    report = adjudicate_evidence(root)
    written = json.loads(
        (root / "independent_adjudication.json").read_text(encoding="utf-8")
    )
    assert written == report
    assert (
        report["output_hash_semantics"]
        == "sha256(canonical-json(report without output_hash))"
    )
    basis = dict(report)
    output_hash = basis.pop("output_hash")
    assert hashlib.sha256(_canonical(basis)).hexdigest() == output_hash
    assert len(output_hash) == 64
    assert math.isfinite(report["mechanism_share"])
