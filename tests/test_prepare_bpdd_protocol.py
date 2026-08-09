from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_bpdd_protocol.py"


def _load_module():
    assert SCRIPT.is_file(), "BPDD protocol preparer has not been implemented"
    spec = importlib.util.spec_from_file_location("prepare_bpdd_protocol", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_manifest_binds_current_source_and_frozen_fdr_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    state = tmp_path / "initial-state.pt"
    state.write_bytes(b"test-state")
    output = tmp_path / "bpdd-protocol.json"
    monkeypatch.setattr(
        module,
        "_file_sha256",
        lambda _path: module.FDR_INITIAL_STATE_SHA256,
    )
    monkeypatch.setattr(module, "_validate_initial_state", lambda _path: None)

    manifest = module.prepare_manifest(
        source_commit="a" * 40,
        source_tree_sha256="B" * 64,
        initial_state=state,
        output=output,
    )

    assert output.is_file()
    assert manifest["source"] == {
        "git_commit": "a" * 40,
        "tree_sha256": "B" * 64,
    }
    assert manifest["initial_state"]["sha256"] == module.FDR_INITIAL_STATE_SHA256
    assert set(manifest["run_identities"]) == {
        "fdr_screen",
        "fdr_formal",
        "fdr_bpdd_screen",
        "fdr_bpdd_formal",
    }
    with pytest.raises(FileExistsError):
        module.prepare_manifest(
            source_commit="a" * 40,
            source_tree_sha256="B" * 64,
            initial_state=state,
            output=output,
        )

