from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.rtdetr_sqda_sgc import (
    SQDASGCDetectionModel,
    adapt_decoder_inputs,
    load_mature_baseline,
    sha256_file,
)
from src.sqda_sgc import SQDASGCAdapter


def test_custom_model_exposes_native_loss_class_count() -> None:
    model = SQDASGCDetectionModel("rtdetr-l.yaml", nc=10, verbose=False)

    assert model.nc == 10
    criterion = model.init_criterion()
    assert criterion.nc == 10


class _RecordingAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recorded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool] | None = None

    def forward(
        self,
        queries: torch.Tensor,
        boxes: torch.Tensor,
        c2: torch.Tensor,
        *,
        identity_override: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.recorded = (queries, boxes, c2, identity_override)
        return queries if identity_override else queries + 1.0, {"ok": torch.tensor(True)}


class _FakeDecoder(nn.Module):
    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list[list[int]],
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple:
        return embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp, attn_mask


class _FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _FakeDecoder()
        self.num_queries = 300


class _TinyDetection(nn.Module):
    def __init__(self, out_features: int = 4) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(3, out_features))


def _decoder_args(
    *,
    batch: int = 2,
    prefix: int = 40,
) -> tuple[tuple, dict]:
    torch.manual_seed(11)
    embed = torch.randn(batch, prefix + 300, 256)
    refer_bbox = torch.randn(batch, prefix + 300, 4)
    feats = torch.randn(batch, 30, 256)
    shapes = [[5, 6]]
    bbox_head = nn.ModuleList([nn.Linear(256, 4)])
    score_head = nn.ModuleList([nn.Linear(256, 10)])
    pos_mlp = nn.Linear(4, 256)
    attn_mask = torch.zeros(prefix + 300, prefix + 300, dtype=torch.bool)
    return (
        embed,
        refer_bbox,
        feats,
        shapes,
        bbox_head,
        score_head,
        pos_mlp,
    ), {"attn_mask": attn_mask}


def test_adaptation_changes_only_native_query_embeddings() -> None:
    adapter = _RecordingAdapter()
    args, kwargs = _decoder_args()
    original_args = tuple(args)
    original_kwargs = dict(kwargs)
    raw_c2 = torch.randn(2, 128, 40, 40)

    new_args, new_kwargs, diagnostics, reference_boxes = adapt_decoder_inputs(
        adapter,
        args,
        kwargs,
        raw_c2,
        query_count=300,
        identity_override=False,
    )

    assert torch.equal(new_args[0][:, :40], original_args[0][:, :40])
    assert torch.equal(new_args[0][:, 40:], original_args[0][:, 40:] + 1)
    assert all(new_args[index] is original_args[index] for index in range(1, len(args)))
    assert new_kwargs == original_kwargs
    assert new_kwargs["attn_mask"] is original_kwargs["attn_mask"]
    assert torch.equal(reference_boxes, original_args[1][:, -300:].sigmoid())
    assert reference_boxes.requires_grad is False
    assert diagnostics["ok"]
    assert adapter.recorded is not None
    assert adapter.recorded[0].shape == (2, 300, 256)
    assert adapter.recorded[2] is raw_c2


def test_adaptation_aligns_amp_c2_to_native_query_dtype() -> None:
    adapter = _RecordingAdapter()
    args, kwargs = _decoder_args(batch=1, prefix=0)
    raw_c2 = torch.randn(1, 128, 40, 40, dtype=torch.float16)

    output_args, _, _, _ = adapt_decoder_inputs(
        adapter,
        args,
        kwargs,
        raw_c2,
        query_count=300,
        identity_override=False,
    )

    assert adapter.recorded is not None
    assert adapter.recorded[0].dtype == torch.float32
    assert adapter.recorded[2].dtype == adapter.recorded[0].dtype
    assert raw_c2.dtype == torch.float16
    assert output_args[0].dtype == args[0].dtype


def test_identity_override_is_bitwise_exact_with_denoising_prefix() -> None:
    adapter = SQDASGCAdapter()
    args, kwargs = _decoder_args(batch=1, prefix=17)
    raw_c2 = torch.randn(1, 128, 40, 40)
    new_args, new_kwargs, _, _ = adapt_decoder_inputs(
        adapter,
        args,
        kwargs,
        raw_c2,
        query_count=300,
        identity_override=True,
    )
    assert torch.equal(new_args[0], args[0])
    assert all(new_args[index] is args[index] for index in range(1, len(args)))
    assert new_kwargs == kwargs


@pytest.mark.parametrize(
    "args,c2,message",
    [
        (_decoder_args(prefix=-1)[0], torch.randn(2, 128, 40, 40), "at least 300"),
        (_decoder_args()[0], None, "C2"),
    ],
)
def test_adaptation_fails_closed_on_malformed_inputs(
    args: tuple,
    c2: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        adapt_decoder_inputs(
            _RecordingAdapter(),
            args,
            {},
            c2,
            query_count=300,
            identity_override=False,
        )


def test_predict_uses_transient_hooks_and_cannot_reuse_stale_c2(monkeypatch: pytest.MonkeyPatch) -> None:
    model = SQDASGCDetectionModel.__new__(SQDASGCDetectionModel)
    nn.Module.__init__(model)
    model.model = nn.ModuleList(
        [
            nn.Identity(),
            nn.Conv2d(3, 128, kernel_size=1),
            _FakeHead(),
        ]
    )
    model.sqda_sgc = _RecordingAdapter()
    model.identity_override = False
    model.last_sqda_diagnostics = None
    model.last_sqda_reference_boxes = None

    def fake_stock_predict(
        instance: SQDASGCDetectionModel,
        image: torch.Tensor,
        profile: bool = False,
        visualize: bool = False,
        batch: dict | None = None,
        augment: bool = False,
        embed: list[int] | None = None,
    ) -> tuple:
        c2 = instance.model[1](image)
        decoder = instance.model[-1].decoder
        args, kwargs = _decoder_args(batch=image.shape[0], prefix=0)
        return decoder(*args, **kwargs), c2

    monkeypatch.setattr(
        "ultralytics.nn.tasks.RTDETRDetectionModel.predict",
        fake_stock_predict,
    )
    result, captured_c2 = model.predict(torch.randn(1, 3, 20, 20))
    decoder_result = result
    assert torch.equal(decoder_result[0], _decoder_args(batch=1, prefix=0)[0][0] + 1)
    assert model.sqda_sgc.recorded is not None
    assert model.sqda_sgc.recorded[2] is captured_c2
    assert len(model.model[1]._forward_hooks) == 0
    assert len(model.model[-1].decoder._forward_pre_hooks) == 0

    direct_args, direct_kwargs = _decoder_args(batch=1, prefix=0)
    direct_result = model.model[-1].decoder(*direct_args, **direct_kwargs)
    assert direct_result[0] is direct_args[0]


def test_adapter_is_outside_stock_sequential_and_in_state_dict() -> None:
    model = SQDASGCDetectionModel.__new__(SQDASGCDetectionModel)
    nn.Module.__init__(model)
    model.model = nn.ModuleList([nn.Identity()])
    model.sqda_sgc = SQDASGCAdapter()
    assert all(module is not model.sqda_sgc for module in model.model)
    assert any(key.startswith("sqda_sgc.") for key in model.state_dict())
    assert not any(key.startswith("model.1") for key in model.state_dict())


def test_strict_mature_baseline_loading_and_sha(tmp_path: Path) -> None:
    torch.manual_seed(3)
    source = _TinyDetection()
    checkpoint = tmp_path / "baseline.pt"
    torch.save({"ema": source}, checkpoint)
    expected_sha = sha256_file(checkpoint)

    target = _TinyDetection()
    with torch.no_grad():
        target.model[0].weight.zero_()
    metadata = load_mature_baseline(target, checkpoint, expected_sha256=expected_sha)

    assert metadata["sha256"] == expected_sha
    assert metadata["source_key"] == "ema"
    assert torch.equal(target.model[0].weight, source.model[0].weight)

    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA256"):
        load_mature_baseline(target, checkpoint, expected_sha256=expected_sha)


def test_strict_mature_baseline_rejects_stock_shape_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.pt"
    torch.save({"model": _TinyDetection(out_features=5)}, checkpoint)
    with pytest.raises(RuntimeError, match="stock state"):
        load_mature_baseline(_TinyDetection(out_features=4), checkpoint)
