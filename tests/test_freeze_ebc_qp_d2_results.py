from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.freeze_ebc_qp_d2_results import build_freeze_manifest, write_freeze_manifest


def test_freeze_manifest_hashes_artifacts_and_refuses_changed_overwrite(tmp_path: Path):
    a0 = tmp_path / "a0-last.pt"
    a2 = tmp_path / "a2-last.pt"
    a0.write_bytes(b"stock-control")
    a2.write_bytes(b"fusion-gamma")

    payload = build_freeze_manifest(
        variant="learnable-fusion-gamma-v1.1",
        protocol_signature="B1226E32",
        artifacts={"a0_last": a0, "a2_last": a2},
    )

    assert payload["format_version"] == 1
    assert payload["variant"] == "learnable-fusion-gamma-v1.1"
    assert payload["protocol_signature"] == "B1226E32"
    assert payload["artifacts"]["a0_last"] == {
        "path": str(a0.resolve()),
        "bytes": len(b"stock-control"),
        "sha256": hashlib.sha256(b"stock-control").hexdigest().upper(),
    }

    destination = tmp_path / "frozen.json"
    write_freeze_manifest(destination, payload)
    write_freeze_manifest(destination, payload)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload

    changed = json.loads(json.dumps(payload))
    changed["protocol_signature"] = "CHANGED"
    with pytest.raises(FileExistsError, match="refusing to replace changed freeze manifest"):
        write_freeze_manifest(destination, changed)


def test_freeze_manifest_rejects_missing_artifact(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_freeze_manifest(
            variant="gamma",
            protocol_signature="signature",
            artifacts={"missing": tmp_path / "missing.pt"},
        )
