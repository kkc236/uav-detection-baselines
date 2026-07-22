from pathlib import Path

import pytest
import torch
from ultralytics.nn.modules.head import RTDETRDecoder

from src.ebc_qp_config import EBCQPConfig
from src.ebc_qp_decoder import EBCQPDecoder
from src.rtdetr_ebc_qp import EBCQPDetectionModel


CONFIG = Path(__file__).parents[1] / "configs" / "rtdetr-l-ebc-qp.yaml"


def test_yaml_passes_c2_p3_p4_p5_to_one_custom_decoder():
    model = EBCQPDetectionModel(CONFIG, ch=3, nc=10, verbose=False)
    head = model.model[-1]

    assert isinstance(head, EBCQPDecoder)
    assert head.f == [1, 21, 24, 27]
    assert len(head.input_proj) == 3
    assert head.p2_adapter[0].in_channels == 128
    assert head.num_queries == 300


def test_disabled_path_matches_stock_indices_and_outputs():
    stock, ebc = _elementwise_identical_small_heads()
    inputs = _small_inputs(requires_grad=False)
    stock.eval()
    ebc.eval()
    ebc.ebc_enabled = False

    with torch.no_grad():
        expected_indices = _stock_topk_indices(stock, inputs[1:])
        stock_output = stock(inputs[1:])
        ebc_output = ebc(inputs)

    torch.testing.assert_close(ebc_output[0], stock_output[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ebc_output[1][0], stock_output[1][0], rtol=1e-5, atol=1e-6)
    assert torch.equal(ebc.last_state.stock_topk_indices, expected_indices)
    assert ebc_output[0].dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP equivalence")
def test_disabled_path_matches_stock_under_cuda_amp():
    stock, ebc = _elementwise_identical_small_heads()
    stock = stock.cuda().eval()
    ebc = ebc.cuda().eval()
    ebc.ebc_enabled = False
    inputs = [tensor.cuda() for tensor in _small_inputs(requires_grad=False)]

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        stock_output = stock(inputs[1:])
        ebc_output = ebc(inputs)

    torch.testing.assert_close(ebc_output[0], stock_output[0], rtol=1e-4, atol=1e-5)


def test_p2_only_backward_isolates_stock_parameters_and_side_inputs():
    head = _small_ebc_head()
    head.train()
    inputs = _small_inputs(requires_grad=True)

    state = head.forward_with_state(inputs, _single_tiny_batch())
    state.p2_loss.backward()

    assert _grad_nonzero(head.p2_adapter)
    assert _grad_nonzero(head.p2_bbox_head)
    assert not _grad_nonzero(head.enc_output)
    assert not _grad_nonzero(head.enc_score_head)
    assert inputs[0].grad is None
    assert inputs[1].grad is None


def test_warmup_and_active_query_integrity():
    head = _small_ebc_head()
    head.train()
    inputs = _small_inputs(requires_grad=False)
    batch = _single_tiny_batch()

    head.set_progress(epoch=2)
    warm = head.forward_with_state(inputs, batch)
    assert warm.p2_entry_count == 0
    assert warm.ebc_active is False

    head.set_progress(epoch=3)
    active = head.forward_with_state(inputs, batch)
    assert active.ordinary_query_count == head.num_queries
    assert 0 <= active.p2_entry_count <= head.ebc_config.p2_candidates
    assert active.encoder_aux_source_is_stock


def test_fusion_gamma_adds_one_trainable_scalar_and_receives_p2_gradient():
    config = EBCQPConfig(query_budget=8, p2_candidates=4, learnable_fusion_gamma=True)
    head = EBCQPDecoder(
        nc=3,
        ch=(4, 8, 8, 8),
        hd=16,
        nq=8,
        ndp=2,
        nh=4,
        ndl=1,
        d_ffn=32,
        nd=0,
        ebc_config=config,
    )
    head.train()

    state = head.forward_with_state(_small_inputs(requires_grad=False), _single_tiny_batch())
    state.p2_loss.backward()

    assert head.p2_fusion_gamma.shape == torch.Size([])
    assert head.p2_fusion_gamma.item() == 1.0
    assert head.p2_fusion_gamma.grad is not None
    assert torch.count_nonzero(head.p2_fusion_gamma.grad) == 1


def test_a1_and_a2_start_with_identical_fixed_budget_query_injection():
    a1 = _small_ebc_head_with_config(
        EBCQPConfig(
            query_budget=8,
            p2_candidates=4,
            lambda_ebc=0.0,
            learnable_fusion_gamma=True,
        )
    )
    a2 = _small_ebc_head_with_config(
        EBCQPConfig(
            query_budget=8,
            p2_candidates=4,
            lambda_ebc=0.05,
            learnable_fusion_gamma=True,
        )
    )
    a2.load_state_dict(a1.state_dict(), strict=True)
    a1.train()
    a2.train()
    a1.set_progress(epoch=3)
    a2.set_progress(epoch=3)
    inputs = _small_inputs(requires_grad=False)
    batch = _single_tiny_batch()

    torch.manual_seed(41)
    a1_state = a1.forward_with_state(inputs, batch)
    torch.manual_seed(41)
    a2_state = a2.forward_with_state(inputs, batch)

    assert a1_state.ordinary_query_count == a2_state.ordinary_query_count == 8
    assert a1_state.p2_entry_count == a2_state.p2_entry_count
    assert torch.equal(a1_state.stock_topk_indices, a2_state.stock_topk_indices)
    assert torch.equal(a1_state.p2_topk_indices, a2_state.p2_topk_indices)
    assert torch.equal(a1_state.final_sources, a2_state.final_sources)
    assert torch.equal(a1_state.final_source_indices, a2_state.final_source_indices)


def test_a1_no_injection_trains_p2_but_keeps_all_final_queries_stock():
    head = _small_ebc_head_with_config(
        EBCQPConfig(
            query_budget=8,
            p2_candidates=4,
            lambda_ebc=0.0,
            learnable_fusion_gamma=True,
            query_injection_enabled=False,
        )
    )
    head.train()
    head.set_progress(epoch=3)

    state = head.forward_with_state(_small_inputs(requires_grad=False), _single_tiny_batch())
    state.p2_loss.backward()

    assert state.ordinary_query_count == 8
    assert state.p2_entry_count == 0
    assert state.competition_active is False
    assert torch.count_nonzero(state.final_sources) == 0
    assert torch.equal(state.final_source_indices, state.stock_topk_indices)
    assert _grad_nonzero(head.p2_adapter)
    assert _grad_nonzero(head.p2_bbox_head)
    assert head.p2_fusion_gamma.grad is not None
    assert torch.count_nonzero(head.p2_fusion_gamma.grad) == 1


def test_qg_p2_zero_initializes_quality_head_and_isolates_quality_gradient():
    head = _small_ebc_head_with_config(
        EBCQPConfig(
            query_budget=8,
            p2_candidates=4,
            lambda_ebc=0.0,
            learnable_fusion_gamma=True,
            quality_gated_p2=True,
        )
    )
    head.train()
    head.set_progress(epoch=2)
    warm = head.forward_with_state(_small_inputs(requires_grad=False), _single_tiny_batch())

    assert torch.count_nonzero(head.p2_quality_head.weight) == 0
    assert torch.count_nonzero(head.p2_quality_head.bias) == 0
    assert warm.p2_entry_count == 0
    assert torch.count_nonzero(warm.final_sources) == 0

    head.set_progress(epoch=3)
    active = head.forward_with_state(_small_inputs(requires_grad=False), _single_tiny_batch())
    active.quality_loss.backward()

    assert active.ordinary_query_count == 8
    assert 0 <= active.p2_entry_count <= 4
    assert _grad_nonzero(head.p2_quality_head)
    assert not _grad_nonzero(head.p2_adapter)
    assert not _grad_nonzero(head.p2_bbox_head)
    assert head.p2_fusion_gamma.grad is None


def _small_ebc_head() -> EBCQPDecoder:
    config = EBCQPConfig(query_budget=8, p2_candidates=4)
    return _small_ebc_head_with_config(config)


def _small_ebc_head_with_config(config: EBCQPConfig) -> EBCQPDecoder:
    return EBCQPDecoder(
        nc=3,
        ch=(4, 8, 8, 8),
        hd=16,
        nq=8,
        ndp=2,
        nh=4,
        ndl=1,
        d_ffn=32,
        nd=0,
        ebc_config=config,
    )


def _elementwise_identical_small_heads() -> tuple[RTDETRDecoder, EBCQPDecoder]:
    torch.manual_seed(7)
    stock = RTDETRDecoder(nc=3, ch=(8, 8, 8), hd=16, nq=8, ndp=2, nh=4, ndl=1, d_ffn=32, nd=0)
    ebc = _small_ebc_head()
    missing, unexpected = ebc.load_state_dict(stock.state_dict(), strict=False)
    assert not unexpected
    assert all(name.startswith(("p2_adapter.", "p2_bbox_head.")) for name in missing)
    return stock, ebc


def _small_inputs(requires_grad: bool) -> list[torch.Tensor]:
    shapes = [(4, 16, 16), (8, 8, 8), (8, 4, 4), (8, 2, 2)]
    return [torch.randn(1, channels, height, width, requires_grad=requires_grad) for channels, height, width in shapes]


def _single_tiny_batch() -> dict:
    return {
        "img": torch.zeros(1, 3, 64, 64),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        "cls": torch.tensor([[1.0]]),
        "batch_idx": torch.tensor([0.0]),
        "gt_groups": [1],
    }


def _stock_topk_indices(head: RTDETRDecoder, inputs: list[torch.Tensor]) -> torch.Tensor:
    feats, shapes = head._get_encoder_input(inputs)
    if head.dynamic or head.shapes != shapes:
        head.anchors, head.valid_mask = head._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        head.shapes = shapes
    features = head.enc_output(head.valid_mask * feats)
    scores = head.enc_score_head(features)
    return torch.topk(scores.max(-1).values, head.num_queries, dim=1).indices


def _grad_nonzero(module: torch.nn.Module) -> bool:
    return any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in module.parameters())
