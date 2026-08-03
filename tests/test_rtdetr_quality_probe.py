from __future__ import annotations

from decimal import Decimal

import pytest
import torch

from src.rtdetr_quality_probe import (
    C1QualityProbe,
    QQualityProbe,
    PROBE_ALPHA,
    PROBE_GATE_MAP_GAIN,
    c1_features,
    evaluate_internal_probe_gate,
    quality_probe_loss,
    rerank_with_predicted_quality,
    top_pair_mask,
)


def _sample() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = torch.tensor(
        [[[0.25, 0.50, 0.10, 0.20], [0.75, 0.25, 0.40, 0.10]]],
        dtype=torch.float32,
    )
    logits = torch.tensor(
        [[[2.0, -1.0, 0.0], [-2.0, 1.0, 3.0]]], dtype=torch.float32
    )
    hidden = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 16
    return boxes, logits, hidden


def test_c1_features_are_class_conditional_finite_and_geometry_bound() -> None:
    boxes, logits, _ = _sample()
    features = c1_features(boxes, logits, num_classes=3)

    assert features.shape == (1, 2, 3, 13)
    assert torch.isfinite(features).all()
    assert torch.equal(features[0, 0, :, -3:], torch.eye(3))
    assert torch.equal(features[..., 0], logits.sigmoid())
    assert torch.equal(features[0, 0, 0, 2:6], boxes[0, 0])
    assert not features.requires_grad


def test_probe_heads_have_production_shapes_and_do_not_mutate_inputs() -> None:
    boxes, logits, hidden = _sample()
    features = c1_features(boxes, logits, num_classes=3)
    features_before = features.clone()
    hidden_before = hidden.clone()

    c1 = C1QualityProbe(feature_dim=features.shape[-1], width=16)
    q = QQualityProbe(
        feature_dim=features.shape[-1], hidden_dim=hidden.shape[-1], width=16
    )
    c1_logits = c1(features)
    q_logits = q(features, hidden)

    assert c1_logits.shape == q_logits.shape == (1, 2, 3)
    assert torch.isfinite(c1_logits).all() and torch.isfinite(q_logits).all()
    assert torch.equal(features, features_before)
    assert torch.equal(hidden, hidden_before)


def test_probe_heads_never_backpropagate_into_detector_evidence() -> None:
    boxes, logits, hidden = _sample()
    features = c1_features(boxes, logits, num_classes=3).requires_grad_(True)
    hidden = hidden.requires_grad_(True)
    c1 = C1QualityProbe(feature_dim=features.shape[-1], width=16)
    q = QQualityProbe(
        feature_dim=features.shape[-1], hidden_dim=hidden.shape[-1], width=16
    )

    (c1(features).mean() + q(features, hidden).mean()).backward()

    assert features.grad is None
    assert hidden.grad is None
    assert all(parameter.grad is not None for parameter in c1.parameters())
    assert all(parameter.grad is not None for parameter in q.parameters())


def test_top_pair_mask_uses_exact_flattened_stock_probability() -> None:
    _, logits, _ = _sample()
    mask = top_pair_mask(logits.sigmoid(), topk=3)

    assert mask.shape == logits.shape
    assert mask.sum().item() == 3
    expected = torch.zeros_like(mask)
    expected.flatten(1)[0, logits.sigmoid().flatten(1).topk(3).indices[0]] = True
    assert torch.equal(mask, expected)


def test_quality_probe_loss_prefers_aligned_soft_quality() -> None:
    target = torch.tensor([[[0.0, 0.2, 0.9]]])
    stock = torch.tensor([[[0.1, 0.5, 0.8]]])
    aligned = torch.logit(target.clamp(1e-4, 1 - 1e-4))
    reversed_logits = aligned.flip(-1)

    aligned_loss = quality_probe_loss(aligned, target, stock)
    reversed_loss = quality_probe_loss(reversed_logits, target, stock)

    assert aligned_loss.ndim == 0 and torch.isfinite(aligned_loss)
    assert aligned_loss < reversed_loss


def test_reranking_uses_fixed_alpha_and_exact_flattened_topk() -> None:
    boxes, logits, _ = _sample()
    quality_logits = torch.tensor(
        [[[8.0, -8.0, 0.0], [-8.0, 4.0, 8.0]]], dtype=torch.float32
    )
    output = rerank_with_predicted_quality(
        boxes, logits, quality_logits, num_classes=3, max_det=3
    )
    expected_scores = logits.sigmoid() * quality_logits.sigmoid().pow(PROBE_ALPHA)
    values, indices = expected_scores.flatten(1).topk(3)

    assert output.shape == (1, 3, 6)
    assert torch.equal(output[0, :, 4], values[0])
    assert torch.equal(output[0, :, 5].long(), indices[0] % 3)


def test_internal_gate_requires_q_to_beat_both_controls_without_rounding() -> None:
    assert PROBE_GATE_MAP_GAIN == Decimal("0.0050")
    controls = {
        "c0": {"map": 0.20, "ap75": 0.18},
        "c1": {"map": 0.21, "ap75": 0.19},
    }
    passed = evaluate_internal_probe_gate(
        controls=controls, q={"map": 0.215, "ap75": 0.190001}
    )
    assert passed["status"] == "passed"

    below_map = evaluate_internal_probe_gate(
        controls=controls, q={"map": 0.214999, "ap75": 0.20}
    )
    tied_ap75 = evaluate_internal_probe_gate(
        controls=controls, q={"map": 0.22, "ap75": 0.19}
    )
    assert below_map["status"] == "scientific_failed"
    assert tied_ap75["status"] == "scientific_failed"


def test_invalid_shapes_and_nonfinite_values_are_rejected() -> None:
    boxes, logits, hidden = _sample()
    with pytest.raises(ValueError):
        c1_features(boxes[..., :3], logits, num_classes=3)
    with pytest.raises(ValueError):
        top_pair_mask(logits.sigmoid(), topk=0)
    features = c1_features(boxes, logits, num_classes=3)
    with pytest.raises(ValueError):
        QQualityProbe(feature_dim=13, hidden_dim=8)(features, hidden[..., :7])
    bad = logits.clone()
    bad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        c1_features(boxes, bad, num_classes=3)
