from __future__ import annotations

from copy import deepcopy
import io
import threading

import pytest
import torch
from torch import nn
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.modules.transformer import MLP

import src.fdr_head as fdr_head
from src.fdr_head import (
    DistributionConditionedFeedback,
    FDRDeformableTransformerDecoder,
    build_distribution_heads,
    cumulative_distribution_logits,
)


def test_distribution_conditioned_feedback_is_zero_initialized_and_shape_safe() -> None:
    module = DistributionConditionedFeedback(16, private_seed=10_000)
    logits = torch.randn(2, 7, 132)
    output = module(logits)
    assert output.shape == (2, 7, 16)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    with pytest.raises(ValueError, match="last dimension 132"):
        module(torch.randn(2, 7, 131))


def test_distribution_conditioned_feedback_detaches_history_and_preserves_public_rng() -> None:
    torch.manual_seed(731)
    before = torch.random.get_rng_state().clone()
    module = DistributionConditionedFeedback(16, private_seed=10_000)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    module.output.weight.data.fill_(0.1)
    logits = torch.randn(1, 3, 132, requires_grad=True)
    module(logits).sum().backward()
    assert logits.grad is None
    assert module.output.weight.grad is not None


def test_dcf_decoder_uses_one_shared_adapter_after_layer_one() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=3)
    for head in dist:
        head.layers[-1].weight.data.fill_(0.01)
    adapter = DistributionConditionedFeedback(16, private_seed=10_001)
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(), pre_bbox_head=pre, distribution_feedback=adapter
    )
    decoder.train()
    calls: list[torch.Size] = []
    hook = adapter.register_forward_hook(lambda _m, args, _out: calls.append(args[0].shape))
    decoder(embed, refs, feats, shapes, dist, scores, _QueryPos(16))
    zero_corners = decoder.last_corner_logits.detach().clone()
    adapter.output.weight.data.fill_(0.1)
    decoder(embed, refs, feats, shapes, dist, scores, _QueryPos(16))
    changed_corners = decoder.last_corner_logits.detach().clone()
    hook.remove()
    assert len(calls) == 10  # five calls for each of two decoder forwards
    assert all(shape == torch.Size([1, 3, 132]) for shape in calls)
    assert not torch.equal(changed_corners, zero_corners)


def test_dcf_scale_zero_skips_adapter_and_matches_no_adapter() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=3)
    for head in dist:
        head.layers[-1].weight.data.fill_(0.01)
    adapter = DistributionConditionedFeedback(16, private_seed=10_001)
    adapter.output.weight.data.fill_(0.1)
    with_adapter = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=deepcopy(pre),
        distribution_feedback=adapter,
    )
    without_adapter = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(), pre_bbox_head=deepcopy(pre)
    )
    without_adapter.load_state_dict(
        {
            name: value
            for name, value in with_adapter.state_dict().items()
            if not name.startswith("distribution_feedback.")
        },
        strict=True,
    )
    calls: list[bool] = []
    hook = adapter.register_forward_hook(lambda *_: calls.append(True))
    with_adapter.set_distribution_feedback_scale(0.0)
    with_adapter.train()
    without_adapter.train()
    pos_mlp = _QueryPos(16)

    actual = with_adapter(embed, refs, feats, shapes, dist, scores, pos_mlp)
    expected = without_adapter(embed, refs, feats, shapes, dist, scores, pos_mlp)

    hook.remove()
    assert calls == []
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_dcf_scale_one_preserves_persistent_feedback_behavior() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=3)
    for head in dist:
        head.layers[-1].weight.data.fill_(0.01)
    adapter = DistributionConditionedFeedback(16, private_seed=10_001)
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(), pre_bbox_head=pre, distribution_feedback=adapter
    )
    adapter.output.weight.data.fill_(0.1)
    decoder.set_distribution_feedback_scale(1.0)
    decoder.train()

    decoder(embed, refs, feats, shapes, dist, scores, _QueryPos(16))

    assert torch.count_nonzero(decoder.last_corner_logits) > 0


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_dcf_feedback_scale_rejects_invalid_values(value: float) -> None:
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(), pre_bbox_head=nn.Linear(16, 4)
    )
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        decoder.set_distribution_feedback_scale(value)


def _old_global_seed_distribution_heads(
    hidden_dim: int,
    num_layers: int = 6,
    *,
    private_seed: int,
) -> nn.ModuleList:
    """Reference the exact pre-isolation construction sequence."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(private_seed)
        heads = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 132, num_layers=3) for _ in range(num_layers)]
        )
    for head in heads:
        nn.init.zeros_(head.layers[-1].weight)
        nn.init.zeros_(head.layers[-1].bias)
    return heads


def test_yaml_instantiable_fdr_head_is_a_real_rtdetr_decoder_module() -> None:
    assert hasattr(fdr_head, "FDRRTDETRDecoder")
    assert issubclass(fdr_head.FDRRTDETRDecoder, RTDETRDecoder)


def _legacy_injected_head(seed: int = 0, private_seed: int = 10_000) -> RTDETRDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = RTDETRDecoder(nc=10, ch=(256, 256, 256))
    stock_pre_bbox_head = head.dec_bbox_head[0]
    distribution_heads = build_distribution_heads(
        head.hidden_dim,
        head.num_decoder_layers,
        private_seed=private_seed,
    )
    head.decoder = FDRDeformableTransformerDecoder.from_stock(
        head.decoder,
        pre_bbox_head=stock_pre_bbox_head,
    )
    head.dec_bbox_head = distribution_heads
    head.decoder.reg_max = 32
    head.decoder.final_layers = [module.layers[-1] for module in distribution_heads]
    return head


def test_declarative_head_matches_legacy_state_contract_exactly() -> None:
    expected = _legacy_injected_head()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        actual = fdr_head.FDRRTDETRDecoder(
            nc=10,
            ch=(256, 256, 256),
            options={
                "hidden_dim": 256,
                "num_queries": 300,
                "num_decoder_layers": 6,
                "reg_max": 32,
                "reg_scale": 4.0,
                "up": 0.5,
                "cumulative": True,
                "preliminary_box": True,
                "private_seed": 10_000,
            },
        )

    assert expected.state_dict().keys() == actual.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], tensor, rtol=0, atol=0)


class _FakeLayer(nn.Module):
    def forward(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        feats: torch.Tensor,
        shapes: list[list[int]],
        padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        del reference, feats, shapes, padding_mask, attn_mask
        return output + 0.01 * query_pos


class _FakeStockDecoder(nn.Module):
    def __init__(self, layers: int = 6, hidden: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(layers)])
        self.hidden_dim = hidden
        self.num_layers = layers
        self.eval_idx = layers - 1


def test_legacy_pickled_decoder_restores_pinned_behavior_flags() -> None:
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(), pre_bbox_head=nn.Linear(16, 4)
    )
    del decoder.cumulative
    del decoder.preliminary_box
    stream = io.BytesIO()
    torch.save(decoder, stream)
    stream.seek(0)

    restored = torch.load(stream, map_location="cpu", weights_only=False)

    assert restored.cumulative is True
    assert restored.preliminary_box is True


class _QueryPos(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(4, hidden, bias=False)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        return self.proj(boxes)


def _inputs(batch: int = 2, queries: int = 7, hidden: int = 16):
    torch.manual_seed(7)
    embed = torch.randn(batch, queries, hidden, requires_grad=True)
    reference_logits = torch.randn(batch, queries, 4)
    feats = torch.randn(batch, 3, hidden)
    shapes = [[1, 3]]
    pre_head = MLP(hidden, hidden, 4, 3)
    dist_heads = build_distribution_heads(hidden, 6, private_seed=10_000)
    score_heads = nn.ModuleList([nn.Linear(hidden, 10) for _ in range(6)])
    return embed, reference_logits, feats, shapes, pre_head, dist_heads, score_heads


def test_cumulative_distribution_logits_is_exact_cumsum() -> None:
    deltas = torch.stack(
        [torch.full((1, 2, 132), float(index + 1)) for index in range(6)]
    )
    torch.testing.assert_close(
        cumulative_distribution_logits(deltas),
        deltas.cumsum(dim=0),
        rtol=0,
        atol=0,
    )


def test_private_distribution_initialization_does_not_advance_public_rng() -> None:
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    heads = build_distribution_heads(16, 6, private_seed=10_000)
    after = torch.random.get_rng_state()
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert len(heads) == 6
    for head in heads:
        assert head.layers[-1].out_features == 132
        torch.testing.assert_close(
            head.layers[-1].weight,
            torch.zeros_like(head.layers[-1].weight),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            head.layers[-1].bias,
            torch.zeros_like(head.layers[-1].bias),
            rtol=0,
            atol=0,
        )


def test_private_distribution_initialization_is_bit_exact_with_legacy_seed_sequence() -> None:
    expected = _old_global_seed_distribution_heads(16, private_seed=10_000)
    actual = build_distribution_heads(16, 6, private_seed=10_000)

    assert expected.state_dict().keys() == actual.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], tensor, rtol=0, atol=0)


def test_concurrent_private_initialization_is_rng_isolated_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force nested construction without timing assumptions or sleeps."""

    original_mlp = fdr_head.MLP
    entered = {name: threading.Event() for name in ("fdr-head-a", "fdr-head-b")}
    release = {name: threading.Event() for name in entered}
    completed = {name: threading.Event() for name in entered}
    first_call_seen: set[str] = set()
    first_call_lock = threading.Lock()
    results: dict[str, nn.ModuleList] = {}
    failures: list[BaseException] = []

    def controlled_mlp(*args, **kwargs):
        name = threading.current_thread().name
        if name in entered:
            with first_call_lock:
                first_call = name not in first_call_seen
                first_call_seen.add(name)
            if first_call:
                entered[name].set()
                if not release[name].wait(timeout=10):
                    raise TimeoutError(f"timed out releasing {name}")
        return original_mlp(*args, **kwargs)

    def construct() -> None:
        name = threading.current_thread().name
        try:
            results[name] = build_distribution_heads(16, 6, private_seed=10_000)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            completed[name].set()

    monkeypatch.setattr(fdr_head, "MLP", controlled_mlp)
    torch.manual_seed(314_159)
    public_rng_before = torch.random.get_rng_state().clone()

    first = threading.Thread(target=construct, name="fdr-head-a")
    second = threading.Thread(target=construct, name="fdr-head-b")
    first.start()
    assert entered["fdr-head-a"].wait(timeout=10)
    second.start()
    assert entered["fdr-head-b"].wait(timeout=10)

    release["fdr-head-a"].set()
    assert completed["fdr-head-a"].wait(timeout=10)
    release["fdr-head-b"].set()
    assert completed["fdr-head-b"].wait(timeout=10)
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not failures
    torch.testing.assert_close(
        torch.random.get_rng_state(), public_rng_before, rtol=0, atol=0
    )
    assert results.keys() == entered.keys()
    for left, right in zip(
        results["fdr-head-a"].state_dict().values(),
        results["fdr-head-b"].state_dict().values(),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_private_distribution_seed_is_reproducible_and_distinct() -> None:
    first = build_distribution_heads(16, 6, private_seed=10_000)
    second = build_distribution_heads(16, 6, private_seed=10_000)
    third = build_distribution_heads(16, 6, private_seed=10_001)
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), third.state_dict().values())
    )


def test_training_decoder_exposes_six_layer_fdr_evidence() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs()
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=deepcopy(pre),
    )
    decoder.train()
    boxes, classes = decoder(
        embed,
        refs,
        feats,
        shapes,
        dist,
        scores,
        _QueryPos(16),
    )
    assert boxes.shape == (6, 2, 7, 4)
    assert classes.shape == (6, 2, 7, 10)
    assert decoder.last_corner_logits is not None
    assert decoder.last_corner_logits.shape == (6, 2, 7, 132)
    assert decoder.last_references is not None
    assert decoder.last_references.shape == (6, 2, 7, 4)
    assert decoder.last_pre_bboxes is not None
    assert decoder.last_pre_bboxes.shape == (2, 7, 4)
    assert not decoder.last_references.requires_grad
    assert torch.isfinite(boxes).all()
    assert torch.isfinite(classes).all()
    assert torch.isfinite(decoder.last_corner_logits).all()


def test_cumulative_logits_keep_prior_distribution_gradients() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=2)
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=pre,
    )
    decoder.train()
    boxes, _ = decoder(embed, refs, feats, shapes, dist, scores, _QueryPos(16))
    boxes[-1].sum().backward()
    for head in dist:
        grad = head.layers[-1].weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_no_cumulative_ablation_uses_only_each_layers_distribution() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=2)
    for index, head in enumerate(dist):
        head.layers[-1].bias.data.fill_(float(index + 1))
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=pre,
    )
    decoder.cumulative = False
    decoder.train()
    decoder(embed, refs, feats, shapes, dist, scores, _QueryPos(16))
    assert decoder.last_corner_logits is not None
    for index, logits in enumerate(decoder.last_corner_logits):
        torch.testing.assert_close(
            logits,
            torch.full_like(logits, float(index + 1)),
            rtol=0,
            atol=0,
        )


def test_no_prebox_ablation_routes_original_reference_but_keeps_prebox_evidence() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=2)
    pre.layers[-1].weight.data.zero_()
    pre.layers[-1].bias.data.fill_(2.0)
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=pre,
    )
    decoder.preliminary_box = False
    decoder.train()
    decoder(embed, refs, feats, shapes, dist, scores, _QueryPos(16))

    assert decoder.last_pre_bboxes is not None
    assert decoder.last_references is not None
    expected_reference = refs.sigmoid().detach()
    for reference in decoder.last_references:
        torch.testing.assert_close(reference, expected_reference, rtol=0, atol=0)
    assert not torch.equal(decoder.last_pre_bboxes, expected_reference)


def test_eval_decoder_returns_only_eval_layer_but_keeps_stock_signature() -> None:
    embed, refs, feats, shapes, pre, dist, scores = _inputs(batch=1, queries=3)
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=pre,
    )
    decoder.eval()
    with torch.no_grad():
        boxes, classes = decoder(
            embed,
            refs,
            feats,
            shapes,
            dist,
            scores,
            _QueryPos(16),
        )
    assert boxes.shape == (1, 1, 3, 4)
    assert classes.shape == (1, 1, 3, 10)
    assert decoder.last_corner_logits is not None
    assert decoder.last_corner_logits.shape == (1, 1, 3, 132)


def test_from_stock_rejects_non_six_layer_decoder() -> None:
    _, _, _, _, pre, _, _ = _inputs()
    with pytest.raises(ValueError, match="six decoder layers"):
        FDRDeformableTransformerDecoder.from_stock(
            _FakeStockDecoder(layers=5),
            pre_bbox_head=pre,
        )


def test_fdr_head_contains_no_excluded_components() -> None:
    _, _, _, _, pre, _, _ = _inputs()
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _FakeStockDecoder(),
        pre_bbox_head=pre,
    )
    forbidden = ("ddf", "teacher", "lqe", "go_lsd", "target_gate")
    assert not any(
        token in name.lower()
        for name, _ in decoder.named_modules()
        for token in forbidden
    )
