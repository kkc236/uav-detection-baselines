from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from src.full_transfer_state import (
    ARTIFACT_ROLE,
    build_full_transfer_artifact,
    load_full_transfer_file,
    load_full_transfer_initial_state,
    validate_full_transfer_artifact,
)


class _TinyFull(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.head = nn.Linear(4, 10)


def _artifact() -> dict:
    torch.manual_seed(17)
    model = _TinyFull()
    return build_full_transfer_artifact(
        model.state_dict(),
        source_checkpoint_sha256="A" * 64,
        source_checkpoint_bytes=67_871_120,
        nc=10,
        channels=3,
    )


def test_build_validate_and_strict_load_preserve_every_tensor() -> None:
    artifact = _artifact()
    assert artifact["artifact_role"] == ARTIFACT_ROLE
    assert artifact["metadata"]["tensor_count"] == 4

    target = _TinyFull()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
    report = load_full_transfer_initial_state(target, artifact, expected_nc=10)

    assert report["tensor_count"] == 4
    assert report["tensor_mismatch_count"] == 0
    for name, expected in artifact["state_dict"].items():
        torch.testing.assert_close(target.state_dict()[name], expected, rtol=0, atol=0)


def test_file_loader_is_weights_only_and_rejects_pickle_execution(tmp_path: Path) -> None:
    state = tmp_path / "valid.pt"
    torch.save(_artifact(), state)
    assert load_full_transfer_file(state)["artifact_role"] == ARTIFACT_ROLE

    marker = tmp_path / "executed.txt"

    class _WriteMarker:
        def __reduce__(self):
            return (marker.write_text, ("unsafe",))

    malicious = tmp_path / "malicious.pt"
    torch.save(_WriteMarker(), malicious)
    with pytest.raises(Exception):
        load_full_transfer_file(malicious)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("role", "role"),
        ("nc", "class count"),
        ("fingerprint", "fingerprint"),
        ("source_hash", "SHA-256"),
        ("tensor_count", "tensor count"),
    ],
)
def test_validation_rejects_corrupted_schema(mutation: str, error: str) -> None:
    artifact = deepcopy(_artifact())
    if mutation == "role":
        artifact["artifact_role"] = "paired_scratch"
    elif mutation == "nc":
        artifact["metadata"]["nc"] = 3
    elif mutation == "fingerprint":
        artifact["state_dict"]["head.bias"][0] += 1
    elif mutation == "source_hash":
        artifact["metadata"]["source_checkpoint_sha256"] = "not-a-hash"
    elif mutation == "tensor_count":
        artifact["metadata"]["tensor_count"] += 1
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=error):
        validate_full_transfer_artifact(artifact, expected_nc=10)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing", "keys"),
        ("extra", "keys"),
        ("shape", "shape"),
        ("dtype", "dtype"),
    ],
)
def test_validation_rejects_target_incompatibility(mutation: str, error: str) -> None:
    artifact = _artifact()
    target = _TinyFull().state_dict()
    if mutation == "missing":
        target.pop("head.bias")
    elif mutation == "extra":
        target["extra"] = torch.zeros(1)
    elif mutation == "shape":
        target["head.bias"] = torch.zeros(11)
    elif mutation == "dtype":
        target["head.bias"] = target["head.bias"].double()
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=error):
        validate_full_transfer_artifact(
            artifact,
            target_state=target,
            expected_nc=10,
        )

