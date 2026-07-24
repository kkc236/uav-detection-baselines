import ast
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import runpy

import pytest


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def imported_top_level_modules(tree):
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name.split(".", 1)[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    return imported


def reseal_primary_checksums(primary):
    primary = Path(primary)
    for path in sorted(
        primary.rglob("*"),
        key=lambda item: len(item.parts),
    ):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    paths = sorted(
        path
        for path in primary.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    text = "".join(
        f"{sha256_file(path)}  {path.name}\n"
        for path in paths
    )
    (primary / "checksums.sha256").write_text(
        text, encoding="utf-8"
    )
    for path in sorted(
        primary.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(primary, 0o555)


def rewrite_gzip_rows(path, rows):
    path = Path(path)
    os.chmod(path.parent, 0o755)
    os.chmod(path, 0o644)
    with gzip.GzipFile(
        filename=str(path), mode="wb", mtime=0
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )


def mutate_primary_json(root, name, mutation):
    path = Path(root) / "primary" / name
    os.chmod(path.parent, 0o755)
    os.chmod(path, 0o644)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    reseal_primary_checksums(path.parent)


@pytest.fixture
def primary_oracle_fixture(tmp_path, monkeypatch):
    import scripts.run_sbr_score_oracle as runner
    from scripts.prepare_sbr_score_oracle_protocol import (
        prepare_protocol,
    )

    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_sbr_v2_audit_cli.py"))
    )
    upstream, _, _ = namespace["_make_input"](
        tmp_path, recoverable=True
    )
    spec = tmp_path / "design.md"
    spec.write_text("# approved\n", encoding="utf-8")
    wrapper = tmp_path / "oracle-input.json"
    wrapper.write_text(
        json.dumps(
            prepare_protocol(
                upstream=upstream,
                spec=spec,
                commit="a" * 40,
                tree="b" * 40,
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "oracle-output"
    monkeypatch.setattr(runner, "EXPECTED_IMAGE_COUNT", 1)
    monkeypatch.setattr(
        runner,
        "capture_clean_source",
        lambda _repo: {
            "commit": "a" * 40,
            "tree": "b" * 40,
        },
    )
    args = runner.build_parser().parse_args(
        [
            "--input-manifest",
            str(wrapper),
            "--spec",
            str(spec),
            "--output",
            str(output),
        ]
    )
    assert runner.run(args) == 0
    return output


@pytest.fixture
def primary_oracle_no_groups_fixture(tmp_path, monkeypatch):
    import scripts.run_sbr_score_oracle as runner
    from scripts.prepare_sbr_score_oracle_protocol import (
        prepare_protocol,
    )

    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_sbr_v2_audit_cli.py"))
    )
    upstream, _, _ = namespace["_make_input"](
        tmp_path, recoverable=False
    )
    spec = tmp_path / "design.md"
    spec.write_text("# approved\n", encoding="utf-8")
    wrapper = tmp_path / "oracle-input.json"
    wrapper.write_text(
        json.dumps(
            prepare_protocol(
                upstream=upstream,
                spec=spec,
                commit="a" * 40,
                tree="b" * 40,
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "oracle-output"
    monkeypatch.setattr(runner, "EXPECTED_IMAGE_COUNT", 1)
    monkeypatch.setattr(
        runner,
        "capture_clean_source",
        lambda _repo: {
            "commit": "a" * 40,
            "tree": "b" * 40,
        },
    )
    assert (
        runner.run(
            runner.build_parser().parse_args(
                [
                    "--input-manifest",
                    str(wrapper),
                    "--spec",
                    str(spec),
                    "--output",
                    str(output),
                ]
            )
        )
        == 0
    )
    return output


@pytest.fixture(autouse=True)
def stable_clean_adjudicator_source(monkeypatch):
    try:
        import scripts.adjudicate_sbr_score_oracle as adjudicator
    except ModuleNotFoundError:
        yield
        return
    state = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "clean": True,
        "script_sha256": sha256_file(Path(adjudicator.__file__)),
        "repo_root": str(
            Path(adjudicator.__file__).resolve().parents[1]
        ),
    }
    monkeypatch.setattr(
        adjudicator,
        "_capture_self_state",
        lambda *_args: dict(state),
    )
    yield


def test_adjudicator_imports_only_stdlib_and_numpy():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "adjudicate_sbr_score_oracle.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = imported_top_level_modules(tree)

    assert imported <= {
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
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
        "urllib",
        "numpy",
    }
    assert not {"src", "scripts"} & imported


def test_adjudicator_replays_joint_and_agrees(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
        replay_primary_evidence,
    )

    root = primary_oracle_fixture
    anchor = sha256_file(
        root / "primary" / "checksums.sha256"
    )
    replay = replay_primary_evidence(root)

    report = adjudicate_evidence(root, anchor)

    assert replay["metrics"]["C"]["AP-large-SBR"] == 0.0
    assert (
        replay["metrics"]["joint"]["AP-large-SBR"]
        == 0.9949999999999999
    )
    assert replay["selected_units"] == 1
    assert replay["status"] == "SBR_SCORE_ORACLE_STOP"
    assert report["decision"] == "PASS"
    assert report["primary_gate_agrees"] is True
    assert report["joint_metrics_agree"] is True
    assert report["unit_labels_agree"] is True
    assert report["authoritative_status"] in {
        "SBR_SCORE_ORACLE_GO",
        "SBR_SCORE_ORACLE_STOP",
    }
    assert (root / "final_status.json").is_file()
    assert (root / "checksums.sha256").is_file()
    assert {path.name for path in root.iterdir()} == {
        "primary",
        "independent_adjudication.json",
        "final_status.json",
        "checksums.sha256",
    }


def test_resealed_metric_tampering_still_fails(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    path = root / "primary" / "oracle_metrics.json"
    os.chmod(root / "primary", 0o755)
    os.chmod(path, 0o644)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["joint"]["AP-large-SBR"] += 0.1
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    reseal_primary_checksums(root / "primary")

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert report["authoritative_status"] == (
        "SBR_SCORE_ORACLE_INVALID"
    )
    assert "metric" in report["error"].lower()
    assert not (root / "checksums.sha256").exists()


def test_resealed_unit_label_flip_still_fails(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    path = root / "primary" / "unit_events.jsonl.gz"
    os.chmod(root / "primary", 0o755)
    os.chmod(path, 0o644)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert rows
    rows[0]["selected"] = not rows[0]["selected"]
    with gzip.GzipFile(
        filename=str(path), mode="wb", mtime=0
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
    reseal_primary_checksums(root / "primary")

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "label" in report["error"].lower()


def test_writable_primary_is_rejected_before_replay(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    os.chmod(root / "primary", 0o755)

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "writable" in report["error"].lower()
    assert not (root / "checksums.sha256").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0]["group"].update(
            {"full_anchor_index": 999999}
        ),
        lambda rows: rows[0]["group"].update(
            {"aggressor_indices": [999999]}
        ),
    ],
)
def test_resealed_forged_group_or_anchor_fails(
    primary_oracle_fixture, mutation
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    path = root / "primary" / "unit_events.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    mutation(rows)
    rewrite_gzip_rows(path, rows)
    reseal_primary_checksums(root / "primary")

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "label" in report["error"].lower()


@pytest.mark.parametrize("operation", ["increase", "omit", "add"])
def test_resealed_joint_patch_mutation_fails(
    primary_oracle_fixture, operation
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    path = root / "primary" / "score_patches.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert rows
    if operation == "increase":
        rows[0]["new_score"] = rows[0]["old_score"]
    elif operation == "omit":
        rows.pop()
    else:
        rows.append({**rows[0], "original_index": 999999})
    rewrite_gzip_rows(path, rows)
    reseal_primary_checksums(root / "primary")

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "patch" in report["error"].lower()


def test_resealed_gate_threshold_edit_fails(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    mutate_primary_json(
        root,
        "primary_gate.json",
        lambda payload: payload["thresholds"].update(
            {"AP-large-SBR": -0.5}
        ),
    )

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "gate" in report["error"].lower()


def test_resealed_negative_peak_memory_fails(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    mutate_primary_json(
        root,
        "runtime.json",
        lambda payload: payload.update({"peak_rss_bytes": -1}),
    )

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "runtime" in report["error"].lower()


@pytest.mark.parametrize("field", ["source", "approved_spec"])
def test_resealed_source_or_spec_hash_edit_fails(
    primary_oracle_fixture, field
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture

    def mutation(payload):
        if field == "source":
            payload["source"]["commit"] = "c" * 40
        else:
            payload["approved_spec"]["sha256"] = "c" * 64

    mutate_primary_json(root, "oracle_manifest.json", mutation)

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert (
        "source" in report["error"].lower()
        if field == "source"
        else "spec" in report["error"].lower()
    )


def test_zero_eligible_and_selected_groups_reproduce_valid_stop(
    primary_oracle_no_groups_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_no_groups_fixture
    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "PASS"
    assert report["authoritative_status"] == (
        "SBR_SCORE_ORACLE_STOP"
    )
    assert report["eligible_units"] == 0
    assert report["selected_units"] == 0


def test_checksum_path_escape_is_rejected(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    checksum = root / "primary" / "checksums.sha256"
    os.chmod(root / "primary", 0o755)
    os.chmod(checksum, 0o644)
    checksum.write_text(
        checksum.read_text(encoding="utf-8")
        + f"{'a' * 64}  ../escape\n",
        encoding="utf-8",
    )
    os.chmod(checksum, 0o444)
    os.chmod(root / "primary", 0o555)

    report = adjudicate_evidence(
        root, sha256_file(checksum)
    )

    assert report["decision"] == "FAIL"
    assert "unsafe" in report["error"].lower()


def test_duplicate_unit_id_is_rejected(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    path = root / "primary" / "unit_events.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    rows.append(dict(rows[0]))
    rewrite_gzip_rows(path, rows)
    reseal_primary_checksums(root / "primary")

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "duplicate" in report["error"].lower()


def test_raw_identity_collision_is_rejected_directly():
    from scripts.adjudicate_sbr_score_oracle import (
        Raw,
        _reconstruct,
    )

    one = Raw(
        "i.jpg",
        "C",
        640,
        640,
        0,
        0,
        0,
        0.9,
        (0.0, 0.0, 10.0, 10.0),
        (0.0, 0.0, 10.0, 10.0),
        (0.0, 0.0, 10.0, 10.0),
        None,
        7,
    )
    two = Raw(
        "i.jpg",
        "C",
        640,
        640,
        1,
        1,
        0,
        0.8,
        (0.0, 0.0, 10.0, 10.0),
        (0.0, 0.0, 10.0, 10.0),
        (0.0, 0.0, 10.0, 10.0),
        (0, 0, 20, 20),
        7,
    )

    with pytest.raises(ValueError, match="collision"):
        _reconstruct((one, two))


def test_static_group_membership_and_unit_id_are_hand_fixed():
    from scripts.adjudicate_sbr_score_oracle import Raw, _groups

    full = Raw(
        "i.jpg",
        "C",
        640,
        640,
        0,
        0,
        0,
        0.8,
        (0.0, 0.0, 200.0, 200.0),
        (0.0, 0.0, 200.0, 200.0),
        (0.0, 0.0, 200.0, 200.0),
        None,
        10,
    )
    local = Raw(
        "i.jpg",
        "C",
        640,
        640,
        1,
        1,
        0,
        0.9,
        (50.0, 0.0, 250.0, 200.0),
        (50.0, 0.0, 250.0, 200.0),
        (50.0, 0.0, 250.0, 200.0),
        (0, 0, 384, 384),
        11,
    )

    assert _groups("i.jpg", (full, local)) == [
        {
            "image_id": "i.jpg",
            "unit_id": "i.jpg:dcf91eb175847a74875ab2a1",
            "stock_cluster_position": 0,
            "stock_member_indices": [11, 10],
            "full_anchor_index": 10,
            "aggressor_indices": [11],
            "anchor_score": 0.8,
        }
    ]


def test_raw_parser_allows_frozen_negative_network_coordinates():
    from scripts.adjudicate_sbr_score_oracle import _parse_raw

    row = {
        "image_id": "i.jpg",
        "arm": "C",
        "width": 640,
        "height": 640,
        "source_order": 0,
        "query_index": 1,
        "class_id": 0,
        "score": 0.9,
        "view_id": "full",
        "tile_bounds": None,
        "network_xyxy": [-1.0, -2.0, 10.0, 20.0],
        "view_xyxy": [0.0, 0.0, 10.0, 20.0],
        "global_xyxy": [0.0, 0.0, 10.0, 20.0],
        "view_manifest": [
            {
                "view_id": "full",
                "source_order": 0,
                "executed": True,
            },
            {
                "view_id": "TL",
                "source_order": 1,
                "executed": True,
            },
            {
                "view_id": "TR",
                "source_order": 2,
                "executed": True,
            },
            {
                "view_id": "BL",
                "source_order": 3,
                "executed": True,
            },
            {
                "view_id": "BR",
                "source_order": 4,
                "executed": True,
            },
        ],
    }

    parsed = _parse_raw(row, 7)

    assert parsed.network_xyxy == (-1.0, -2.0, 10.0, 20.0)


def test_all_five_gate_boundaries_are_inclusive_and_bare_float():
    from scripts.adjudicate_sbr_score_oracle import (
        GATES,
        _gate_metrics,
    )

    a = {
        "AP-tiny-SBR": 0.0,
        "mAP50-95": 0.0,
        "tiny_recall": 0.0,
        "AP75": 0.002,
        "AP-large-SBR": 0.005,
    }
    joint = {
        "AP-tiny-SBR": 0.010,
        "mAP50-95": 0.003,
        "tiny_recall": 0.020,
        "AP75": 0.0,
        "AP-large-SBR": 0.0,
    }

    status, _, gates = _gate_metrics(a, joint, 1)

    assert status == "SBR_SCORE_ORACLE_GO"
    assert all(gates.values())
    for name, threshold in GATES.items():
        changed_a = dict(a)
        changed_joint = dict(joint)
        if threshold >= 0:
            changed_joint[name] = math.nextafter(
                changed_joint[name], -math.inf
            )
        else:
            changed_a[name] = math.nextafter(
                changed_a[name], math.inf
            )
        failed, _, failed_gates = _gate_metrics(
            changed_a, changed_joint, 1
        )
        assert failed == "SBR_SCORE_ORACLE_STOP"
        assert failed_gates[name] is False


def test_selected_units_can_interact_into_a_joint_gate_stop():
    from scripts.adjudicate_sbr_score_oracle import _gate_metrics

    a = {
        "AP-tiny-SBR": 0.10,
        "mAP50-95": 0.10,
        "tiny_recall": 0.10,
        "AP75": 0.10,
        "AP-large-SBR": 0.10,
    }
    joint = {
        "AP-tiny-SBR": 0.12,
        "mAP50-95": 0.11,
        "tiny_recall": 0.13,
        "AP75": 0.10,
        "AP-large-SBR": 0.09,
    }

    status, _, gates = _gate_metrics(a, joint, selected_units=2)

    assert status == "SBR_SCORE_ORACLE_STOP"
    assert gates["AP-large-SBR"] is False


def test_dataset_metric_uses_frozen_numpy_arange_threshold_bits():
    from scripts.adjudicate_sbr_score_oracle import (
        Image,
        Prediction,
        _evaluate_dataset,
    )

    image = Image(
        "i.jpg",
        640,
        640,
        ((0.0, 0.0, 10.0, 10.0),),
        (0,),
        (),
    )
    prediction = Prediction(
        (0.0, 0.0, 6.0, 10.0),
        (0.0, 0.0, 6.0, 10.0),
        0.9,
        0,
        0,
        0,
        (0,),
    )

    metrics = _evaluate_dataset([(image, (prediction,))])

    assert metrics["mAP50-95"] == 0.199


def test_resealed_test_dev_uri_is_rejected(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    manifest_path = root / "primary" / "oracle_manifest.json"
    os.chmod(root / "primary", 0o755)
    os.chmod(manifest_path, 0o644)
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    wrapper_path = Path(manifest["wrapper"]["uri"])
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    original_spec = Path(manifest["approved_spec"]["uri"])
    test_spec = original_spec.with_name("test-dev-design.md")
    test_spec.write_bytes(original_spec.read_bytes())
    wrapper["approved_spec"] = {
        "uri": str(test_spec),
        "sha256": sha256_file(test_spec),
    }
    wrapper_path.write_text(
        json.dumps(
            wrapper,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest["wrapper"]["sha256"] = sha256_file(wrapper_path)
    manifest["approved_spec"] = {
        "uri": str(test_spec),
        "sha256": sha256_file(test_spec),
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    reseal_primary_checksums(root / "primary")

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "test-dev" in report["error"].lower()


def test_primary_snapshot_is_identical_before_and_after_replay(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        _primary_snapshot,
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    before = _primary_snapshot(root / "primary")
    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )
    after = _primary_snapshot(root / "primary")

    assert report["decision"] == "PASS"
    assert before == after == report["primary_snapshot"]


def test_git_subprocess_argv_is_exactly_allowlisted(
    monkeypatch,
):
    namespace = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "scripts"
            / "adjudicate_sbr_score_oracle.py"
        )
    )
    calls = []

    class Completed:
        def __init__(self, output):
            self.stdout = output

    def fake_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        output = ""
        if argv[-1] == "HEAD":
            output = "a" * 40 + "\n"
        elif argv[-1] == "HEAD^{tree}":
            output = "b" * 40 + "\n"
        return Completed(output)

    monkeypatch.setattr("subprocess.run", fake_run)
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "adjudicate_sbr_score_oracle.py"
    )

    state = namespace["_capture_self_state"](script)

    repo = str(Path(__file__).parents[1].resolve())
    allowed = {
        ("git", "-C", repo, "rev-parse", "HEAD"),
        ("git", "-C", repo, "rev-parse", "HEAD^{tree}"),
        (
            "git",
            "-C",
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
    }
    assert state["clean"] is True
    assert {argv for argv, _ in calls} <= allowed
    assert len(calls) == 3
    assert all(
        kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "shell": False,
        }
        for _, kwargs in calls
    )


def test_realistic_ultralytics_yaml_ignores_download_block(
    tmp_path,
):
    from scripts.adjudicate_sbr_score_oracle import (
        _yaml_scalar_mapping,
    )

    path = tmp_path / "VisDrone.yaml"
    path.write_text(
        "path: ../datasets/VisDrone\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: pedestrian\n"
        "  1: people\n"
        "download: |\n"
        "  import os\n"
        "  from pathlib import Path\n"
        "  os.system('echo ignored')\n",
        encoding="utf-8",
    )

    values = _yaml_scalar_mapping(path)

    assert values["val"] == "images/val"
    assert "import os" not in values


def test_adjudicator_parser_uses_only_frozen_anchor_name():
    from scripts.adjudicate_sbr_score_oracle import build_parser

    args = build_parser().parse_args(
        [
            "--evidence",
            "evidence",
            "--primary-checksums-sha256",
            "a" * 64,
        ]
    )

    assert vars(args) == {
        "evidence": Path("evidence"),
        "primary_checksums_sha256": "a" * 64,
    }
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--evidence",
                "evidence",
                "--expected-primary-checksums-sha256",
                "a" * 64,
            ]
        )


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_unsealed_extra_primary_node_is_rejected(
    primary_oracle_fixture, kind
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    primary = root / "primary"
    os.chmod(primary, 0o755)
    extra = primary / (
        "extra.bin" if kind == "file" else "extra-dir"
    )
    if kind == "file":
        extra.write_bytes(b"unsealed")
        os.chmod(extra, 0o444)
    else:
        extra.mkdir()
        os.chmod(extra, 0o555)
    os.chmod(primary, 0o555)

    report = adjudicate_evidence(
        root,
        sha256_file(primary / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "artifact set" in report["error"].lower()
    assert not (root / "checksums.sha256").exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_unsealed_extra_root_node_is_rejected(
    primary_oracle_fixture, kind
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    extra = root / (
        "extra.bin" if kind == "file" else "extra-dir"
    )
    if kind == "file":
        extra.write_bytes(b"unsealed")
    else:
        extra.mkdir()

    report = adjudicate_evidence(
        root,
        sha256_file(root / "primary" / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "root artifact set" in report["error"].lower()
    assert not (root / "checksums.sha256").exists()


def test_required_primary_symlink_is_rejected(
    primary_oracle_fixture,
):
    from scripts.adjudicate_sbr_score_oracle import (
        adjudicate_evidence,
    )

    root = primary_oracle_fixture
    primary = root / "primary"
    runtime = primary / "runtime.json"
    external = root.parent / "runtime-copy.json"
    external.write_bytes(runtime.read_bytes())
    os.chmod(primary, 0o755)
    os.chmod(runtime, 0o644)
    runtime.unlink()
    try:
        runtime.symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    os.chmod(primary, 0o555)

    report = adjudicate_evidence(
        root,
        sha256_file(primary / "checksums.sha256"),
    )

    assert report["decision"] == "FAIL"
    assert "artifact set" in report["error"].lower()
