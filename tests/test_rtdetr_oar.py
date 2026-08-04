from decimal import Decimal

import pytest
import torch

from src.rtdetr_oar import (
    OARRanker,
    apply_oar_r2,
    oracle_score_families,
    restrict_oracle,
    select_candidate_k,
    topk_per_class_mask,
)


def _restricted_maps(*, k20: float, k40: float, k60: float, k100: float):
    return {20: k20, 40: k40, 60: k60, 100: k100}


def _oracle_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]),
        torch.zeros(2, 3),
        torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        torch.tensor([1]),
    )


def test_oracle_families_are_class_conditional() -> None:
    boxes, logits, target_boxes, target_classes = _oracle_inputs()

    scores = oracle_score_families(
        boxes,
        logits,
        target_boxes,
        target_classes,
        num_classes=3,
    )

    assert set(scores) == {"stock", "presence", "query_iou", "same_class"}
    assert torch.equal(scores["stock"], torch.full((2, 3), 0.5))
    assert torch.equal(
        scores["presence"],
        torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]]),
    )
    assert torch.equal(
        scores["query_iou"],
        torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]),
    )
    assert torch.equal(
        scores["same_class"],
        torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_oracle_families_return_zero_oracles_for_empty_targets() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]])
    logits = torch.tensor([[0.0, 1.0, -1.0], [2.0, -2.0, 0.0]])

    scores = oracle_score_families(
        boxes,
        logits,
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.long),
        num_classes=3,
    )

    assert torch.equal(scores["stock"], logits.sigmoid())
    for family in ("presence", "query_iou", "same_class"):
        assert torch.equal(scores[family], torch.zeros_like(logits))


def test_oracle_families_zero_absent_classes() -> None:
    boxes, logits, target_boxes, target_classes = _oracle_inputs()

    scores = oracle_score_families(
        boxes,
        logits,
        target_boxes,
        target_classes,
        num_classes=3,
    )

    assert torch.count_nonzero(scores["presence"][:, [0, 2]]) == 0
    assert torch.count_nonzero(scores["same_class"][:, [0, 2]]) == 0
    assert torch.count_nonzero(scores["query_iou"][0, [0, 2]]) == 2


def test_oracle_families_detach_inputs_and_return_float32() -> None:
    boxes, logits, target_boxes, target_classes = _oracle_inputs()
    boxes.requires_grad_()
    logits = logits.to(dtype=torch.float64).requires_grad_()
    target_boxes.requires_grad_()

    scores = oracle_score_families(
        boxes,
        logits,
        target_boxes,
        target_classes,
        num_classes=3,
    )

    assert all(score.dtype == torch.float32 for score in scores.values())
    assert all(not score.requires_grad for score in scores.values())


def test_topk_per_class_mask_selects_exactly_k_queries_per_class() -> None:
    probabilities = torch.arange(3000, dtype=torch.float32).reshape(300, 10)

    mask = topk_per_class_mask(probabilities, 20)

    assert mask.dtype == torch.bool
    assert mask.device == probabilities.device
    assert torch.equal(mask.sum(dim=0), torch.full((10,), 20))
    assert torch.all(mask[-20:])
    assert not bool(mask[:-20].any())


def test_topk_per_class_mask_breaks_boundary_ties_by_lower_query_index() -> None:
    probabilities = torch.zeros(300, 2)
    for class_index in range(2):
        higher = torch.arange(class_index, class_index + 38, 2)
        tied = torch.arange(class_index + 38, class_index + 68, 4)
        probabilities[higher, class_index] = 2
        probabilities[tied, class_index] = 1

    mask = topk_per_class_mask(probabilities, 20)

    for class_index in range(2):
        higher = torch.arange(class_index, class_index + 38, 2)
        tied = torch.arange(class_index + 38, class_index + 68, 4)
        assert bool(mask[higher, class_index].all())
        assert bool(mask[tied[0], class_index])
        assert not bool(mask[tied[1:], class_index].any())


@pytest.mark.parametrize("k", [20, 40, 60, 100])
def test_topk_per_class_mask_accepts_every_frozen_k(k: int) -> None:
    probabilities = torch.rand(300, 3)

    mask = topk_per_class_mask(probabilities, k)

    assert torch.equal(mask.sum(dim=0), torch.full((3,), k))


def test_topk_per_class_mask_validates_shape_type_finiteness_and_grid() -> None:
    probabilities = torch.ones(300, 10)

    with pytest.raises(TypeError, match="tensor"):
        topk_per_class_mask([[1.0]], 20)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        topk_per_class_mask(torch.ones(300, 10, dtype=torch.int64), 20)
    with pytest.raises(ValueError, match="shape"):
        topk_per_class_mask(torch.ones(1, 300, 10), 20)
    with pytest.raises(ValueError, match="shape"):
        topk_per_class_mask(torch.empty(300, 0), 20)
    with pytest.raises(ValueError, match="finite"):
        invalid = probabilities.clone()
        invalid[0, 0] = float("nan")
        topk_per_class_mask(invalid, 20)
    with pytest.raises(ValueError, match="OAR_K_GRID"):
        topk_per_class_mask(probabilities, 19)
    with pytest.raises(ValueError, match="OAR_K_GRID"):
        topk_per_class_mask(probabilities, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="queries"):
        topk_per_class_mask(torch.ones(20, 10), 40)


def test_oracle_score_families_validates_shapes_types_devices_and_finiteness() -> None:
    boxes, logits, target_boxes, target_classes = _oracle_inputs()

    with pytest.raises(TypeError, match="boxes must be a tensor"):
        oracle_score_families(
            boxes.tolist(), logits, target_boxes, target_classes, num_classes=3
        )
    with pytest.raises(TypeError, match="logits.*floating-point"):
        oracle_score_families(
            boxes,
            logits.to(dtype=torch.int64),
            target_boxes,
            target_classes,
            num_classes=3,
        )
    with pytest.raises(ValueError, match="boxes must have shape"):
        oracle_score_families(
            boxes.unsqueeze(0),
            logits,
            target_boxes,
            target_classes,
            num_classes=3,
        )
    with pytest.raises(ValueError, match="logits must have shape"):
        oracle_score_families(
            boxes,
            torch.zeros(3, 3),
            target_boxes,
            target_classes,
            num_classes=3,
        )
    with pytest.raises(ValueError, match="same number of targets"):
        oracle_score_families(
            boxes,
            logits,
            torch.empty(0, 4),
            target_classes,
            num_classes=3,
        )
    with pytest.raises(TypeError, match="integer class indices"):
        oracle_score_families(
            boxes,
            logits,
            target_boxes,
            target_classes.float(),
            num_classes=3,
        )
    with pytest.raises(ValueError, match="share a device"):
        oracle_score_families(
            boxes,
            logits,
            torch.empty(0, 4, device="meta"),
            torch.empty(0, dtype=torch.long),
            num_classes=3,
        )
    with pytest.raises(ValueError, match="finite"):
        invalid_logits = logits.clone()
        invalid_logits[0, 0] = float("inf")
        oracle_score_families(
            boxes,
            invalid_logits,
            target_boxes,
            target_classes,
            num_classes=3,
        )
    with pytest.raises(ValueError, match="num_classes"):
        oracle_score_families(
            boxes, logits, target_boxes, target_classes, num_classes=0
        )


def test_restrict_oracle_preserves_every_outside_pool_score_exactly() -> None:
    stock = torch.tensor([[0.12345679, 0.25], [0.5, 0.98765433]])
    oracle = torch.tensor([[0.9, 0.8], [0.7, 0.6]])
    mask = torch.tensor([[True, False], [False, True]])

    restricted = restrict_oracle(stock, oracle, mask)

    assert torch.equal(restricted[~mask], stock[~mask])
    assert torch.equal(restricted[mask], oracle[mask])


def test_restrict_oracle_validates_shapes_types_devices_and_finiteness() -> None:
    stock = torch.ones(2, 3)
    oracle = torch.zeros(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(TypeError, match="stock must be a tensor"):
        restrict_oracle(stock.tolist(), oracle, mask)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        restrict_oracle(stock.to(dtype=torch.int64), oracle, mask)
    with pytest.raises(TypeError, match="boolean"):
        restrict_oracle(stock, oracle, mask.float())
    with pytest.raises(ValueError, match="same shape"):
        restrict_oracle(stock, oracle[:, :2], mask)
    with pytest.raises(ValueError, match="share a device"):
        restrict_oracle(stock, oracle, torch.ones(2, 3, dtype=torch.bool, device="meta"))
    with pytest.raises(ValueError, match="finite"):
        invalid_oracle = oracle.clone()
        invalid_oracle[0, 0] = float("nan")
        restrict_oracle(stock, invalid_oracle, mask)


def test_select_candidate_k_accepts_exact_ninety_percent_recovery() -> None:
    result = select_candidate_k(
        stock_map=0.20,
        full_map=0.30,
        restricted_map=_restricted_maps(k20=0.29, k40=0.30, k60=0.30, k100=0.30),
    )

    assert result == {"status": "passed", "selected_k": 20, "recovered": "0.9"}
    assert Decimal(result["recovered"]) == Decimal("0.90")


def test_select_candidate_k_chooses_smallest_passing_grid_value() -> None:
    restricted = {
        100: 0.30,
        60: 0.295,
        40: 0.291,
        20: 0.2899,
    }

    result = select_candidate_k(
        stock_map=0.20,
        full_map=0.30,
        restricted_map=restricted,
    )

    assert result["status"] == "passed"
    assert result["selected_k"] == 40
    assert Decimal(result["recovered"]) == Decimal("0.91")


@pytest.mark.parametrize("full_map", [0.20, 0.19])
def test_select_candidate_k_fails_when_full_gain_is_nonpositive(full_map: float) -> None:
    result = select_candidate_k(
        stock_map=0.20,
        full_map=full_map,
        restricted_map=_restricted_maps(k20=0.29, k40=0.29, k60=0.29, k100=0.29),
    )

    assert result == {"status": "scientific_failed", "selected_k": None}


def test_select_candidate_k_fails_when_no_k_recovers_ninety_percent() -> None:
    result = select_candidate_k(
        stock_map=0.20,
        full_map=0.30,
        restricted_map=_restricted_maps(
            k20=0.21,
            k40=0.25,
            k60=0.28,
            k100=0.289999999,
        ),
    )

    assert result == {"status": "scientific_failed", "selected_k": None}


def test_select_candidate_k_validates_metrics_and_exact_k_grid() -> None:
    valid = _restricted_maps(k20=0.21, k40=0.22, k60=0.23, k100=0.24)

    with pytest.raises(TypeError, match="mapping"):
        select_candidate_k(stock_map=0.2, full_map=0.3, restricted_map=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="OAR_K_GRID"):
        select_candidate_k(
            stock_map=0.2,
            full_map=0.3,
            restricted_map={20: 0.21, 40: 0.22, 60: 0.23},
        )
    with pytest.raises(ValueError, match="OAR_K_GRID"):
        select_candidate_k(
            stock_map=0.2,
            full_map=0.3,
            restricted_map={**valid, 80: 0.24},
        )
    with pytest.raises(ValueError, match="OAR_K_GRID"):
        select_candidate_k(
            stock_map=0.2,
            full_map=0.3,
            restricted_map={20.0: 0.21, 40: 0.22, 60: 0.23, 100: 0.24},
        )
    with pytest.raises(ValueError, match="finite"):
        select_candidate_k(stock_map=float("nan"), full_map=0.3, restricted_map=valid)
    with pytest.raises(ValueError, match="finite"):
        invalid = dict(valid)
        invalid[60] = float("inf")
        select_candidate_k(stock_map=0.2, full_map=0.3, restricted_map=invalid)


def test_oar_r2_adjusts_all_pairs_and_starts_as_exact_stock() -> None:
    model = OARRanker()
    features = torch.randn(2, 300, 10, 276)
    logits = torch.randn(2, 300, 10)

    adjusted, residual = apply_oar_r2(model, features, logits)

    assert adjusted.shape == residual.shape == (2, 300, 10)
    assert residual[0].numel() == 3000
    assert residual[1].numel() == 3000
    assert torch.equal(adjusted, logits.sigmoid())
    assert torch.count_nonzero(residual) == 0


def test_oar_r2_has_exact_frozen_architecture_parameter_count() -> None:
    model = OARRanker()

    assert sum(parameter.numel() for parameter in model.parameters()) == 17_793
    assert model.network[0].in_features == 276
    assert model.network[0].out_features == 64
    assert isinstance(model.network[1], torch.nn.SiLU)
    assert model.network[2].in_features == 64
    assert model.network[2].out_features == 1


@pytest.mark.parametrize("final_bias", [-1_000.0, 1_000.0])
def test_oar_r2_residual_is_bounded(final_bias: float) -> None:
    model = OARRanker()
    with torch.no_grad():
        model.network[-1].bias.fill_(final_bias)

    residual = model(torch.randn(1, 300, 10, 276))

    assert residual.shape == (1, 300, 10)
    assert bool((residual >= -2.0).all())
    assert bool((residual <= 2.0).all())


def test_oar_r2_detaches_every_evidence_input_and_trains_model() -> None:
    features = torch.randn(1, 300, 10, 276, requires_grad=True)
    logits = torch.randn(1, 300, 10, requires_grad=True)
    model = OARRanker()

    adjusted, residual = apply_oar_r2(model, features, logits)
    (adjusted.sum() + residual.sum()).backward()

    assert features.grad is None
    assert logits.grad is None
    assert all(parameter.grad is not None for parameter in model.parameters())
