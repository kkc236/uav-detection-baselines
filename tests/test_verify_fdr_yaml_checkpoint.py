from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts import verify_fdr_yaml_checkpoint as verifier


ROOT = Path(__file__).resolve().parents[1]


class TinyCheckpointModel(nn.Module):
    float_calls = 0

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))

    def float(self) -> TinyCheckpointModel:
        type(self).float_calls += 1
        return super().float()


class TinyHead(nn.Module):
    pass


class TinyFDRModel(nn.Module):
    instances: list[TinyFDRModel] = []
    expected_imgsz = 16

    def __init__(
        self,
        cfg: str | Path,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = Path(cfg)
        self.ch = ch
        self.nc = nc
        self.verbose = verbose
        self.weight = nn.Parameter(torch.zeros(1))
        self.model = nn.ModuleList([TinyHead()])
        self.loaded_strict: bool | None = None
        type(self).instances.append(self)

    def load_state_dict(self, state_dict, strict: bool = True):
        self.loaded_strict = strict
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, image: torch.Tensor):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert image.shape == (1, 3, self.expected_imgsz, self.expected_imgsz)
        assert torch.count_nonzero(image).item() == 0
        output = torch.ones((1, 2, 6), dtype=image.dtype) * self.weight
        return output, {"raw": output.clone()}


@pytest.fixture(autouse=True)
def reset_tiny_model_state():
    TinyCheckpointModel.float_calls = 0
    TinyFDRModel.instances.clear()


def test_verify_checkpoint_strictly_loads_and_writes_success_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = tmp_path / "fdr.yaml"
    cfg.write_text("nc: 10\n", encoding="utf-8")
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model": TinyCheckpointModel()}, checkpoint)
    output = tmp_path / "reports" / "verification.json"

    real_torch_load = torch.load
    load_calls: list[dict[str, object]] = []

    def recording_load(path, **kwargs):
        load_calls.append(kwargs)
        return real_torch_load(path, **kwargs)

    monkeypatch.setattr(verifier, "FDRRTDETRDetectionModel", TinyFDRModel)
    monkeypatch.setattr(verifier.torch, "load", recording_load)

    report = verifier.verify_checkpoint(
        cfg=cfg,
        checkpoint=checkpoint,
        output=output,
        nc=7,
        imgsz=16,
    )

    assert load_calls == [{"map_location": "cpu", "weights_only": False}]
    assert TinyCheckpointModel.float_calls == 1
    assert len(TinyFDRModel.instances) == 1
    model = TinyFDRModel.instances[0]
    assert model.cfg == cfg.resolve()
    assert (model.ch, model.nc, model.verbose) == (3, 7, False)
    assert model.loaded_strict is True
    torch.testing.assert_close(model.weight, torch.tensor([2.0]))

    expected = {
        "cfg": str(cfg.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "strict_load": True,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "state_tensor_count": 1,
        "finite_output": True,
        "output_shape": [1, 2, 6],
        "head_type": "TinyHead",
    }
    assert report == expected
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert list(output.parent.glob("*.tmp")) == []


def test_corrupt_checkpoint_key_exits_nonzero_without_success_json(tmp_path: Path):
    cfg = tmp_path / "fdr.yaml"
    cfg.write_text("nc: 10\n", encoding="utf-8")
    checkpoint = tmp_path / "corrupt.pt"
    torch.save({"model": nn.Linear(1, 1, bias=False)}, checkpoint)
    output = tmp_path / "verification.json"
    program = f"""
import torch
from scripts import verify_fdr_yaml_checkpoint as verifier

class Target(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(1, 1, bias=True)

verifier.FDRRTDETRDetectionModel = Target
raise SystemExit(verifier.main([
    '--cfg', {str(cfg)!r},
    '--checkpoint', {str(checkpoint)!r},
    '--output', {str(output)!r},
]))
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Missing key(s) in state_dict" in completed.stderr
    assert not output.exists()


def test_cli_argument_parsing_requires_paths_and_applies_defaults():
    parser = verifier.build_parser()

    defaults = parser.parse_args(
        ["--cfg", "model.yaml", "--checkpoint", "last.pt", "--output", "report.json"]
    )
    assert defaults.cfg == Path("model.yaml")
    assert defaults.checkpoint == Path("last.pt")
    assert defaults.output == Path("report.json")
    assert defaults.nc == 10
    assert defaults.imgsz == 128

    custom = parser.parse_args(
        [
            "--cfg",
            "model.yaml",
            "--checkpoint",
            "last.pt",
            "--output",
            "report.json",
            "--nc",
            "4",
            "--imgsz",
            "256",
        ]
    )
    assert custom.nc == 4
    assert custom.imgsz == 256


def test_atomic_json_keeps_destination_intact_until_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "report.json"
    output.write_text('{"status": "old"}\n', encoding="utf-8")
    payload = {"status": "complete", "strict_load": True}
    real_replace = os.replace
    observations: list[tuple[dict[str, object], str]] = []

    def observing_replace(source, destination):
        observations.append(
            (
                json.loads(Path(source).read_text(encoding="utf-8")),
                Path(destination).read_text(encoding="utf-8"),
            )
        )
        real_replace(source, destination)

    monkeypatch.setattr(verifier.os, "replace", observing_replace)

    verifier.write_json_atomic(output, payload)

    assert observations == [(payload, '{"status": "old"}\n')]
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob("*.tmp")) == []
