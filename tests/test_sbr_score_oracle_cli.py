import hashlib
import gzip
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "prepare_sbr_score_oracle_protocol.py",
        "run_sbr_score_oracle.py",
        "adjudicate_sbr_score_oracle.py",
    ],
)
def test_documented_script_entrypoints_support_direct_execution(
    name,
):
    repo = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / name),
            "--help",
        ],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        shell=False,
    )

    assert completed.returncode == 0, completed.stderr


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def test_protocol_wrapper_binds_upstream_spec_commit_tree_and_rule(
    tmp_path,
):
    from scripts.prepare_sbr_score_oracle_protocol import (
        prepare_protocol,
    )

    upstream = tmp_path / "upstream.json"
    upstream.write_text(
        json.dumps(
            {
                "schema_version": "sbr-v2-audit-input/v1",
                "dataset": {"split": "val"},
            }
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "design.md"
    spec.write_text("# approved\n", encoding="utf-8")

    wrapper = prepare_protocol(
        upstream=upstream,
        spec=spec,
        commit="a" * 40,
        tree="b" * 40,
    )

    assert wrapper["schema_version"] == (
        "sbr-score-oracle-input/v1"
    )
    assert wrapper["upstream_input"]["sha256"] == sha256_file(
        upstream
    )
    assert wrapper["approved_spec"]["sha256"] == sha256_file(spec)
    assert wrapper["expected_source"] == {
        "commit": "a" * 40,
        "tree": "b" * 40,
    }
    assert wrapper["frozen_rule"]["conf"] == 0.001
    assert wrapper["frozen_rule"]["max_det"] == 300
    assert wrapper["forbidden_inputs"] == [
        "test-dev",
        "external-dataset",
    ]


def test_protocol_wrapper_rejects_wrong_upstream_schema(tmp_path):
    from scripts.prepare_sbr_score_oracle_protocol import (
        prepare_protocol,
    )

    upstream = tmp_path / "upstream.json"
    upstream.write_text(
        '{"schema_version":"wrong"}', encoding="utf-8"
    )
    spec = tmp_path / "design.md"
    spec.write_text("# approved\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        prepare_protocol(
            upstream=upstream,
            spec=spec,
            commit="a" * 40,
            tree="b" * 40,
        )


def test_capture_clean_source_uses_git_commit_and_tree(
    tmp_path, monkeypatch
):
    import scripts.prepare_sbr_score_oracle_protocol as module

    monkeypatch.setattr(
        module,
        "git_provenance",
        lambda _repo: {
            "commit": "a" * 40,
            "clean_tracked": True,
            "untracked": False,
        },
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="b" * 40 + "\n"
        ),
    )

    source = module.capture_clean_source(tmp_path)

    assert source == {"commit": "a" * 40, "tree": "b" * 40}


def _wrapper(tmp_path, *, split="val"):
    from scripts.prepare_sbr_score_oracle_protocol import (
        prepare_protocol,
    )

    upstream = tmp_path / "upstream.json"
    upstream.write_text(
        json.dumps(
            {
                "schema_version": "sbr-v2-audit-input/v1",
                "dataset": {"split": split},
            }
        ),
        encoding="utf-8",
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
    return wrapper, spec


def test_wrapper_validation_rejects_non_val_before_opening_data(
    tmp_path,
):
    from scripts.run_sbr_score_oracle import validate_oracle_wrapper

    wrapper, spec = _wrapper(tmp_path, split="test-dev")

    with pytest.raises(ValueError, match="val"):
        validate_oracle_wrapper(
            wrapper,
            spec,
            source={"commit": "a" * 40, "tree": "b" * 40},
        )


def test_wrapper_validation_rejects_source_mismatch(tmp_path):
    from scripts.run_sbr_score_oracle import validate_oracle_wrapper

    wrapper, spec = _wrapper(tmp_path)

    with pytest.raises(ValueError, match="source"):
        validate_oracle_wrapper(
            wrapper,
            spec,
            source={"commit": "c" * 40, "tree": "b" * 40},
        )


def _make_v2_input(tmp_path, *, recoverable=True):
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_sbr_v2_audit_cli.py"))
    )
    return namespace["_make_input"](
        tmp_path, recoverable=recoverable
    )


def _sealed_wrapper(tmp_path, upstream, spec):
    from scripts.prepare_sbr_score_oracle_protocol import (
        prepare_protocol,
    )

    path = tmp_path / "oracle-input.json"
    path.write_text(
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
    return path


def test_synthetic_end_to_end_primary_writes_sealed_contract(
    tmp_path, monkeypatch
):
    import scripts.run_sbr_score_oracle as runner

    upstream, _, _ = _make_v2_input(tmp_path)
    spec = tmp_path / "design.md"
    spec.write_text("# approved\n", encoding="utf-8")
    wrapper = _sealed_wrapper(tmp_path, upstream, spec)
    output = tmp_path / "oracle-output"
    monkeypatch.setattr(runner, "EXPECTED_IMAGE_COUNT", 1)
    monkeypatch.setattr(
        runner,
        "capture_clean_source",
        lambda _repo: {"commit": "a" * 40, "tree": "b" * 40},
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

    primary = output / "primary"
    assert {path.name for path in primary.iterdir()} == {
        "oracle_manifest.json",
        "unit_events.jsonl.gz",
        "score_patches.jsonl.gz",
        "coverage.json",
        "oracle_metrics.json",
        "invariants.json",
        "primary_gate.json",
        "runtime.json",
        "checksums.sha256",
    }
    gate = json.loads(
        (primary / "primary_gate.json").read_text(encoding="utf-8")
    )
    assert gate["proposed_status"] == "SBR_SCORE_ORACLE_STOP"
    assert gate["independent_adjudication"] == "PENDING"
    invariants = json.loads(
        (primary / "invariants.json").read_text(encoding="utf-8")
    )
    assert invariants["passed"] is True
    assert invariants["image_count"] == 1
    runtime = json.loads(
        (primary / "runtime.json").read_text(encoding="utf-8")
    )
    assert (
        isinstance(runtime["peak_rss_bytes"], int)
        and not isinstance(runtime["peak_rss_bytes"], bool)
        and runtime["peak_rss_bytes"] >= 0
    )
    assert (
        isinstance(runtime["parent_peak_rss_bytes"], int)
        and runtime["parent_peak_rss_bytes"] >= 0
    )
    assert (
        isinstance(runtime["max_worker_peak_rss_bytes"], int)
        and runtime["max_worker_peak_rss_bytes"] >= 0
    )
    with gzip.open(
        primary / "unit_events.jsonl.gz", "rt", encoding="utf-8"
    ) as handle:
        event_rows = [json.loads(line) for line in handle]
    if event_rows:
        assert "before_profile" in event_rows[0]
        assert "after_profile" in event_rows[0]
    manifest = json.loads(
        (primary / "oracle_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["primary_script_sha256"] == sha256_file(
        Path(runner.__file__)
    )
    assert len(manifest["frozen_rule_hash"]) == 64
    assert all(
        path.stat().st_mode & 0o222 == 0
        for path in primary.rglob("*")
    )


def test_authoritative_runner_requires_exact_548_images(
    tmp_path, monkeypatch
):
    import scripts.run_sbr_score_oracle as runner

    upstream, _, _ = _make_v2_input(tmp_path)
    spec = tmp_path / "design.md"
    spec.write_text("# approved\n", encoding="utf-8")
    wrapper = _sealed_wrapper(tmp_path, upstream, spec)
    monkeypatch.setattr(
        runner,
        "capture_clean_source",
        lambda _repo: {"commit": "a" * 40, "tree": "b" * 40},
    )
    args = runner.build_parser().parse_args(
        [
            "--input-manifest",
            str(wrapper),
            "--spec",
            str(spec),
            "--output",
            str(tmp_path / "oracle-output"),
        ]
    )

    with pytest.raises(ValueError, match="548"):
        runner.run(args)


def test_shard_schema_and_payload_hash_fail_closed():
    from scripts.run_sbr_score_oracle import (
        build_shard,
        validate_shard,
    )

    payload = {"events": [], "metrics": {"x": 1}}
    shard = build_shard(
        run_identity="a" * 64,
        image_order=7,
        image_id="images/0007.jpg",
        input_image_hash="b" * 64,
        payload=payload,
    )

    assert validate_shard(
        shard,
        run_identity="a" * 64,
        image_order=7,
        image_id="images/0007.jpg",
        input_image_hash="b" * 64,
    ) == payload
    shard["payload"]["metrics"]["x"] = 2
    with pytest.raises(ValueError, match="payload hash"):
        validate_shard(
            shard,
            run_identity="a" * 64,
            image_order=7,
            image_id="images/0007.jpg",
            input_image_hash="b" * 64,
        )


def test_complete_shards_require_unique_continuous_manifest_order():
    from scripts.run_sbr_score_oracle import validate_complete_shards

    entries = [
        {
            "image_order": 0,
            "image_id": "a.jpg",
            "input_image_hash": "a" * 64,
        },
        {
            "image_order": 0,
            "image_id": "a.jpg",
            "input_image_hash": "a" * 64,
        },
    ]

    with pytest.raises(ValueError, match="duplicate"):
        validate_complete_shards(
            entries,
            image_ids=("a.jpg", "b.jpg"),
            input_hashes=("a" * 64, "b" * 64),
        )


def test_scientific_primary_artifacts_are_worker_deterministic(
    tmp_path, monkeypatch
):
    import scripts.run_sbr_score_oracle as runner

    upstream, _, _ = _make_v2_input(tmp_path)
    spec = tmp_path / "design.md"
    spec.write_text("# approved\n", encoding="utf-8")
    wrapper = _sealed_wrapper(tmp_path, upstream, spec)
    monkeypatch.setattr(runner, "EXPECTED_IMAGE_COUNT", 1)
    monkeypatch.setattr(
        runner,
        "capture_clean_source",
        lambda _repo: {"commit": "a" * 40, "tree": "b" * 40},
    )
    outputs = []
    for workers in (0, 8):
        output = tmp_path / f"oracle-output-{workers}"
        args = runner.build_parser().parse_args(
            [
                "--input-manifest",
                str(wrapper),
                "--spec",
                str(spec),
                "--output",
                str(output),
                "--workers",
                str(workers),
            ]
        )
        assert runner.run(args) == 0
        outputs.append(output / "primary")

    deterministic = (
        "unit_events.jsonl.gz",
        "score_patches.jsonl.gz",
        "coverage.json",
        "oracle_metrics.json",
        "invariants.json",
        "primary_gate.json",
    )
    for name in deterministic:
        assert (outputs[0] / name).read_bytes() == (
            outputs[1] / name
        ).read_bytes()
    assert (
        outputs[0] / "unit_events.jsonl.gz"
    ).read_bytes()[4:8] == b"\x00\x00\x00\x00"


def test_existing_final_output_is_never_overwritten(
    tmp_path, monkeypatch
):
    import scripts.run_sbr_score_oracle as runner

    upstream, _, _ = _make_v2_input(tmp_path)
    spec = tmp_path / "design.md"
    spec.write_text("# approved\n", encoding="utf-8")
    wrapper = _sealed_wrapper(tmp_path, upstream, spec)
    output = tmp_path / "oracle-output"
    output.mkdir()
    monkeypatch.setattr(
        runner,
        "capture_clean_source",
        lambda _repo: {"commit": "a" * 40, "tree": "b" * 40},
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

    with pytest.raises(FileExistsError):
        runner.run(args)


def test_identical_run_resumes_valid_shard_without_recomputation(
    tmp_path, monkeypatch
):
    import scripts.run_sbr_score_oracle as runner

    upstream, _, _ = _make_v2_input(tmp_path)
    spec = tmp_path / "design.md"
    spec.write_text("# approved\n", encoding="utf-8")
    wrapper = _sealed_wrapper(tmp_path, upstream, spec)
    output = tmp_path / "oracle-output"
    monkeypatch.setattr(runner, "EXPECTED_IMAGE_COUNT", 1)
    monkeypatch.setattr(
        runner,
        "capture_clean_source",
        lambda _repo: {"commit": "a" * 40, "tree": "b" * 40},
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
    real_writer = runner._write_primary
    monkeypatch.setattr(
        runner,
        "_write_primary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated infrastructure interruption")
        ),
    )

    with pytest.raises(RuntimeError, match="interruption"):
        runner.run(args)

    staging = tmp_path / ".oracle-output.oracle-staging"
    assert (
        staging / "shards" / "000000.json.gz"
    ).is_file()
    monkeypatch.setattr(runner, "_write_primary", real_writer)
    monkeypatch.setattr(
        runner,
        "_evaluate_image_task",
        lambda _task: (_ for _ in ()).throw(
            AssertionError("valid shard was recomputed")
        ),
    )

    assert runner.run(args) == 0
    assert output.is_dir()
    assert not staging.exists()


def test_runner_parser_exposes_only_operational_arguments():
    from scripts.run_sbr_score_oracle import build_parser

    args = build_parser().parse_args(
        [
            "--input-manifest",
            "input.json",
            "--spec",
            "design.md",
            "--output",
            "evidence",
        ]
    )

    assert vars(args) == {
        "input_manifest": Path("input.json"),
        "spec": Path("design.md"),
        "output": Path("evidence"),
        "workers": 0,
    }


@pytest.mark.parametrize(
    "name",
    [
        "--conf",
        "--max-det",
        "--ios",
        "--demotion",
        "--size-threshold",
        "--subset",
        "--split",
    ],
)
def test_runner_parser_rejects_scientific_overrides(name):
    from scripts.run_sbr_score_oracle import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--input-manifest",
                "i",
                "--spec",
                "s",
                "--output",
                "o",
                name,
                "1",
            ]
        )
