from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import src.acr_eg_release as release


class FakeACREGDetectionModel(nn.Module):
    def __init__(self, *, acr_key_count: int = 48) -> None:
        super().__init__()
        self.acr_eg = nn.Module()
        for index in range(acr_key_count):
            self.acr_eg.register_parameter(
                f"weight_{index}", nn.Parameter(torch.tensor(float(index)))
            )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module | None = None,
    raw_epoch: int = 9,
) -> None:
    torch.save(
        {
            "ema": model or FakeACREGDetectionModel(),
            "optimizer": {"state": {0: {"momentum_buffer": torch.ones(1)}}},
            "scaler": {"scale": 128.0, "growth_interval": 2**31 - 1},
            "epoch": raw_epoch,
            "updates": 1700,
        },
        path,
    )


def test_inspection_proves_integrated_checkpoint_identity_and_continuity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "epoch9.pt"
    save_checkpoint(checkpoint)
    monkeypatch.setattr(
        release,
        "_is_acr_eg_model",
        lambda model: isinstance(model, FakeACREGDetectionModel),
    )

    metadata = release.inspect_acr_eg_checkpoint(
        checkpoint,
        expected_completed_epoch=10,
    )

    assert metadata.checkpoint_epoch == 9
    assert metadata.completed_epoch == 10
    assert metadata.model_type == "FakeACREGDetectionModel"
    assert metadata.acr_eg_key_count == 48
    assert metadata.optimizer_state_entries == 1
    assert metadata.scaler_scale == 128.0
    assert metadata.scaler_growth_interval == 2**31 - 1
    assert metadata.updates == 1700
    assert metadata.bytes == checkpoint.stat().st_size
    assert len(metadata.sha256) == 64


def test_inspection_rejects_stock_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "stock.pt"
    save_checkpoint(checkpoint, model=nn.Linear(1, 1))
    monkeypatch.setattr(
        release,
        "_is_acr_eg_model",
        lambda model: isinstance(model, FakeACREGDetectionModel),
    )

    with pytest.raises(ValueError, match="ACR_EG_RELEASE_MODEL_IDENTITY_MISMATCH"):
        release.inspect_acr_eg_checkpoint(checkpoint, expected_completed_epoch=10)


def test_inspection_rejects_incomplete_acr_eg_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "incomplete.pt"
    save_checkpoint(checkpoint, model=FakeACREGDetectionModel(acr_key_count=47))
    monkeypatch.setattr(release, "_is_acr_eg_model", lambda _model: True)

    with pytest.raises(ValueError, match="ACR_EG_RELEASE_STATE_IDENTITY_MISMATCH"):
        release.inspect_acr_eg_checkpoint(checkpoint, expected_completed_epoch=10)


def test_inspection_rejects_wrong_epoch_and_nonfrozen_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "epoch9.pt"
    save_checkpoint(checkpoint)
    monkeypatch.setattr(release, "_is_acr_eg_model", lambda _model: True)

    with pytest.raises(ValueError, match="ACR_EG_RELEASE_EPOCH_MISMATCH"):
        release.inspect_acr_eg_checkpoint(checkpoint, expected_completed_epoch=11)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["scaler"]["scale"] = 256.0
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="ACR_EG_RELEASE_SCALER_SCALE_MISMATCH"):
        release.inspect_acr_eg_checkpoint(checkpoint, expected_completed_epoch=10)


def test_release_coordinates_are_unique_per_epoch() -> None:
    coordinates = release.release_coordinates("a" * 40, completed_epoch=10)

    assert coordinates.tag == "gcte-acr-eg-aaaaaaaa-epoch-010"
    assert coordinates.asset_name == "epoch9.pt"
    assert coordinates.evidence_name == "epoch-010.json"


def test_remote_asset_requires_matching_bytes_and_digest() -> None:
    metadata = release.ACREGCheckpointMetadata(
        source=Path("epoch9.pt"),
        checkpoint_epoch=9,
        completed_epoch=10,
        model_type="ACREGDetectionModel",
        state_key_count=989,
        acr_eg_key_count=48,
        optimizer_state_entries=621,
        scaler_scale=128.0,
        scaler_growth_interval=2**31 - 1,
        updates=1700,
        bytes=205325084,
        sha256="b" * 64,
    )
    asset = {
        "id": 5,
        "name": "epoch9.pt",
        "size": metadata.bytes,
        "digest": f"sha256:{metadata.sha256}",
    }

    assert release.verify_remote_asset(None, asset=asset, metadata=metadata) == asset["digest"]

    with pytest.raises(RuntimeError, match="ACR_EG_RELEASE_REMOTE_DIGEST_MISMATCH"):
        release.verify_remote_asset(
            None,
            asset={**asset, "digest": f"sha256:{'c' * 64}"},
            metadata=metadata,
        )


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def get(self, *_args, **_kwargs) -> FakeResponse:
        return FakeResponse(self.responses.pop(0))


def test_release_tag_must_resolve_to_exact_source_commit() -> None:
    source_commit = "a" * 40
    session = FakeSession([{"object": {"type": "commit", "sha": source_commit}}])

    assert (
        release.verify_release_target(
            session,
            repo="kkc236/uav-detection-baselines",
            tag="gcte-acr-eg-aaaaaaaa-epoch-010",
            source_commit=source_commit,
        )
        == source_commit
    )

    mismatch = FakeSession([{"object": {"type": "commit", "sha": "b" * 40}}])
    with pytest.raises(RuntimeError, match="ACR_EG_RELEASE_TARGET_MISMATCH"):
        release.verify_release_target(
            mismatch,
            repo="kkc236/uav-detection-baselines",
            tag="gcte-acr-eg-aaaaaaaa-epoch-010",
            source_commit=source_commit,
        )
