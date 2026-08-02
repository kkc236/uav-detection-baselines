from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.run_iber_canary import CanaryViolation, run_private_step_canary


class _TinyFrozenDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.requires_grad_(False)
        self.eval()


class _TinyRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(2, 4)
        self.f3_projection = nn.Linear(2, 4)
        self.rgb_encoder = nn.Linear(2, 4)
        for module in (self.base, self.f3_projection, self.rgb_encoder):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.f3_projection(value) + self.rgb_encoder(value)


class _TinyAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = _TinyFrozenDetector()
        self.refiner = _TinyRefiner()

    def forward_evidence(self, image: torch.Tensor):
        stock = torch.full((image.shape[0], 1, 4), 0.5)
        delta = self.refiner(torch.ones(image.shape[0], 1, 2))
        return SimpleNamespace(stock_boxes=stock, refined_boxes=stock + delta)

    def training_step(self, batch: dict[str, torch.Tensor]):
        output = self.forward_evidence(batch["img"])
        target = torch.full_like(output.refined_boxes, 0.6)
        total = (output.refined_boxes - target).abs().mean()
        return SimpleNamespace(total=total, box=total, gate=total * 0, noop=total * 0)


class _PreflightMatcher:
    def matcher(self, *_args, **_kwargs):
        return [(torch.tensor([9]), torch.tensor([0]))]


class _TrainingMatcherAdapter(_TinyAdapter):
    """Expose a deliberately different preflight matcher than the AMP train path."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = _PreflightMatcher()
        self.last_match_indices = None

    def forward_evidence(self, image: torch.Tensor):
        output = super().forward_evidence(image)
        output.stock_scores = torch.zeros(image.shape[0], 1, 1)
        return output

    def training_step(self, batch: dict[str, torch.Tensor]):
        losses = super().training_step(batch)
        self.last_match_indices = [(torch.tensor([0]), torch.tensor([0]))]
        return losses


class _UnnamedEvidenceRefiner(_TinyRefiner):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.base = nn.Linear(2, 4)
        self.semantic_path = nn.Linear(2, 4)
        self.color_path = nn.Linear(2, 4)
        for module in (self.base, self.semantic_path, self.color_path):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.semantic_path(value) + self.color_path(value)


def test_canary_proves_private_paths_checkpoint_and_detector_invariance() -> None:
    adapter = _TinyAdapter()
    report = run_private_step_canary(
        adapter,
        {"img": torch.zeros(1, 3, 8, 8)},
        use_amp=False,
    )

    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["checks"]["zero_init_identity"] is True
    assert report["checks"]["f3_gradient_present"] is True
    assert report["checks"]["rgb_gradient_present"] is True
    assert report["checks"]["detector_gradient_absent"] is True
    assert report["checks"]["checkpoint_roundtrip"] is True
    assert report["checks"]["checkpoint_mode_switching"] is True


def test_canary_compares_matcher_before_and_after_private_step_in_same_train_path() -> None:
    adapter = _TrainingMatcherAdapter()
    report = run_private_step_canary(
        adapter,
        {
            "img": torch.zeros(1, 3, 8, 8),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "batch_idx": torch.tensor([0.0]),
        },
        use_amp=False,
    )

    assert report["checks"]["matcher_indices_same"] is True


def test_canary_rejects_trainable_detector() -> None:
    adapter = _TinyAdapter()
    adapter.detector.projection.weight.requires_grad_(True)
    with pytest.raises(CanaryViolation, match="requires_grad"):
        run_private_step_canary(
            adapter,
            {"img": torch.zeros(1, 3, 8, 8)},
            use_amp=False,
        )


def test_canary_rejects_missing_named_f3_or_rgb_gradient_paths() -> None:
    adapter = _TinyAdapter()
    adapter.refiner = _UnnamedEvidenceRefiner()
    with pytest.raises(CanaryViolation, match="gradient_present"):
        run_private_step_canary(
            adapter,
            {"img": torch.zeros(1, 3, 8, 8)},
            use_amp=False,
        )


def test_canary_source_is_independent_iber_b3() -> None:
    source = __import__("pathlib").Path("scripts/run_iber_canary.py").read_text(
        encoding="utf-8"
    )
    assert "FrozenIBERAdapter" in source
    assert 'probe="b3"' in source
    assert "DESIGN_VERSION" in source
    assert "rtdetr_itber" not in source
    assert 'probe="p3"' not in source
