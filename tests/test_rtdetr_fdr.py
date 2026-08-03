from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

from src.fdr_head import FDR_OUTPUT_DIM, FDRDeformableTransformerDecoder
from src.rtdetr_fdr import FDRRTDETRDetectionModel, split_fdr_evidence


EXCLUDED = ("ddf", "teacher", "lqe", "go_lsd", "target_gate")


def _stock(seed: int = 0) -> RTDETRDetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return RTDETRDetectionModel("rtdetr-l.yaml", nc=10, verbose=False)


def _fdr(seed: int = 0, private_seed: int = 10_000) -> FDRRTDETRDetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return FDRRTDETRDetectionModel(
            "rtdetr-l.yaml",
            nc=10,
            verbose=False,
            private_seed=private_seed,
        )


@pytest.fixture(scope="module")
def fdr_model() -> FDRRTDETRDetectionModel:
    return _fdr()


def _assert_state_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for name in left_state:
        torch.testing.assert_close(left_state[name], right_state[name], rtol=0, atol=0)


def _targets(batch_size: int, *, empty: bool = False) -> dict[str, object]:
    if empty:
        return {
            "cls": torch.empty((0,), dtype=torch.long),
            "bboxes": torch.empty((0, 4), dtype=torch.float32),
            "batch_idx": torch.empty((0,), dtype=torch.long),
            "gt_groups": [0] * batch_size,
        }
    return {
        "cls": torch.arange(batch_size, dtype=torch.long) % 10,
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32).repeat(batch_size, 1),
        "batch_idx": torch.arange(batch_size, dtype=torch.long),
        "gt_groups": [1] * batch_size,
    }


def test_model_replaces_only_decoder_box_contract_and_preserves_public_state():
    stock = _stock()
    method = _fdr()
    stock_head = stock.model[-1]
    method_head = method.model[-1]

    assert method_head.num_queries == stock_head.num_queries == 300
    assert isinstance(method_head.decoder, FDRDeformableTransformerDecoder)
    assert len(method_head.dec_bbox_head) == 6
    assert all(head.layers[-1].out_features == FDR_OUTPUT_DIM for head in method_head.dec_bbox_head)
    _assert_state_equal(stock_head.dec_bbox_head[0], method_head.decoder.pre_bbox_head)
    _assert_state_equal(stock_head.dec_score_head, method_head.dec_score_head)
    _assert_state_equal(stock_head.enc_score_head, method_head.enc_score_head)
    _assert_state_equal(stock_head.enc_bbox_head, method_head.enc_bbox_head)
    _assert_state_equal(stock_head.query_pos_head, method_head.query_pos_head)
    _assert_state_equal(stock_head.decoder.layers, method_head.decoder.layers)


def test_private_head_construction_does_not_advance_public_rng():
    torch.manual_seed(731)
    state = torch.random.get_rng_state()
    RTDETRDetectionModel("rtdetr-l.yaml", nc=10, verbose=False)
    expected = torch.random.get_rng_state()

    torch.random.set_rng_state(state)
    FDRRTDETRDetectionModel(
        "rtdetr-l.yaml", nc=10, verbose=False, private_seed=10_731
    )
    actual = torch.random.get_rng_state()
    assert torch.equal(actual, expected)


def test_eval_forward_keeps_stock_postprocess_shape(fdr_model: FDRRTDETRDetectionModel):
    fdr_model.eval()
    # 128 px yields 336 encoder locations, enough for the frozen Top-300.
    # no_grad (instead of inference_mode) keeps cached anchors reusable in train.
    with torch.no_grad():
        output, raw = fdr_model(torch.zeros(1, 3, 128, 128))
    assert output.shape == (1, 300, 6)
    dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta = raw
    assert dec_boxes.shape == (1, 1, 300, 4)
    assert dec_scores.shape == (1, 1, 300, 10)
    assert enc_boxes.shape == (1, 300, 4)
    assert enc_scores.shape == (1, 300, 10)
    assert dn_meta is None
    assert torch.isfinite(output).all()


def test_top300_postprocess_is_exact_for_identical_boxes_and_scores():
    stock = _stock()
    method = _fdr()
    generator = torch.Generator().manual_seed(941)
    boxes = torch.rand(2, 300, 4, generator=generator)
    scores = torch.rand(2, 300, 10, generator=generator)
    expected = stock.model[-1].postprocess(boxes, scores)
    actual = method.model[-1].postprocess(boxes, scores)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_training_forward_keeps_six_layers_and_splits_dn_evidence(
    fdr_model: FDRRTDETRDetectionModel,
):
    fdr_model.train()
    images = torch.zeros(2, 3, 128, 128)
    outputs = fdr_model.predict(images, batch=_targets(2))
    dec_boxes, dec_scores, _, _, dn_meta = outputs
    assert dec_boxes.shape[:2] == (6, 2)
    assert dec_scores.shape[:2] == (6, 2)
    assert dn_meta is not None
    assert dn_meta["dn_num_split"][1] == 300

    evidence = fdr_model.last_fdr_evidence
    assert evidence is not None
    assert evidence.corner_logits.shape == (6, 2, 300, FDR_OUTPUT_DIM)
    assert evidence.references.shape == (6, 2, 300, 4)
    assert evidence.pre_boxes.shape == (2, 300, 4)
    assert evidence.dn_corner_logits is not None
    assert evidence.dn_references is not None
    assert evidence.dn_pre_boxes is not None
    assert evidence.dn_corner_logits.shape[2] == dn_meta["dn_num_split"][0]
    assert torch.isfinite(evidence.corner_logits).all()
    assert evidence.references.requires_grad is False
    assert evidence.pre_boxes.requires_grad is True


def test_empty_gt_training_keeps_normal_evidence_and_touches_dn_embedding(
    fdr_model: FDRRTDETRDetectionModel,
):
    fdr_model.train()
    outputs = fdr_model.predict(
        torch.zeros(2, 3, 128, 128), batch=_targets(2, empty=True)
    )
    dec_boxes, dec_scores, _, _, dn_meta = outputs
    assert dec_boxes.shape == (6, 2, 300, 4)
    assert dec_scores.shape == (6, 2, 300, 10)
    assert dn_meta is None
    evidence = fdr_model.last_fdr_evidence
    assert evidence is not None
    assert evidence.corner_logits.shape == (6, 2, 300, FDR_OUTPUT_DIM)
    assert evidence.dn_corner_logits is None


def test_split_fdr_evidence_rejects_inconsistent_dn_partition():
    corners = torch.zeros(6, 1, 10, FDR_OUTPUT_DIM)
    references = torch.zeros(6, 1, 10, 4)
    pre_boxes = torch.zeros(1, 10, 4)
    with pytest.raises(ValueError, match="partition"):
        split_fdr_evidence(
            corners,
            references,
            pre_boxes,
            {"dn_num_split": [3, 8]},
        )


def test_finite_backward_reaches_private_distribution_heads(
    fdr_model: FDRRTDETRDetectionModel,
):
    fdr_model.zero_grad(set_to_none=True)
    fdr_model.train()
    outputs = fdr_model.predict(
        torch.zeros(2, 3, 128, 128), batch=_targets(2, empty=True)
    )
    dec_boxes, dec_scores, _, _, _ = outputs
    evidence = fdr_model.last_fdr_evidence
    assert evidence is not None
    loss = dec_boxes.square().mean() + dec_scores.square().mean()
    loss = loss + evidence.corner_logits.square().mean() + evidence.pre_boxes.square().mean()
    loss.backward()
    final_layers = [head.layers[-1] for head in fdr_model.model[-1].dec_bbox_head]
    assert all(layer.weight.grad is not None for layer in final_layers)
    assert all(torch.isfinite(layer.weight.grad).all() for layer in final_layers)


def test_no_excluded_modules_and_installed_ultralytics_is_not_modified():
    source = Path(inspect.getsourcefile(RTDETRDetectionModel) or "")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    model = _fdr(private_seed=20_000)
    module_names = tuple(name.lower() for name, _ in model.named_modules())
    assert not any(token in name for token in EXCLUDED for name in module_names)
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    assert after == before
