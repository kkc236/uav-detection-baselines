from __future__ import annotations

import hashlib
import inspect
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.nn.tasks import RTDETRDetectionModel, yaml_model_load

from src.fdr_head import (
    FDR_OUTPUT_DIM,
    FDRDeformableTransformerDecoder,
    FDRRTDETRDecoder,
    build_distribution_heads,
)
from src.fdr_loss import FDRDetectionLoss, stock_loss_subtotal
from src.rtdetr_fdr import (
    FDRControlTrainer,
    FDRRTDETRDetectionModel,
    FDRTrainer,
    FDRTrainingEvidence,
    split_fdr_evidence,
)
from src.rtdetr_lpr import FixedPairedProtocolMixin


EXCLUDED = ("ddf", "teacher", "lqe", "go_lsd", "target_gate")
FDR_CONFIG_ROOT = Path("configs")


def _declarative_cfg(
    *,
    private_seed: int = 10_000,
    cumulative: bool = True,
    preliminary_box: bool = True,
    fgl_weight: float = 0.15,
    supervise_pre_boxes: bool = True,
) -> dict:
    cfg = deepcopy(yaml_model_load("rtdetr-l.yaml"))
    cfg["head"][-1] = [
        [21, 24, 27],
        1,
        "FDRRTDETRDecoder",
        [
            "nc",
            [256, 256, 256],
            {
                "hidden_dim": 256,
                "num_queries": 300,
                "num_decoder_layers": 6,
                "reg_max": 32,
                "reg_scale": 4.0,
                "up": 0.5,
                "cumulative": cumulative,
                "preliminary_box": preliminary_box,
                "private_seed": private_seed,
            },
        ],
    ]
    cfg["fdr_loss"] = {
        "fgl_weight": fgl_weight,
        "supervise_pre_boxes": supervise_pre_boxes,
    }
    return cfg


def _stock(seed: int = 0) -> RTDETRDetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return RTDETRDetectionModel("rtdetr-l.yaml", nc=10, verbose=False)


def _legacy_fdr(seed: int = 0, private_seed: int = 10_000) -> RTDETRDetectionModel:
    model = _stock(seed)
    head = model.model[-1]
    stock_pre_bbox_head = head.dec_bbox_head[0]
    distribution_heads = build_distribution_heads(
        int(head.hidden_dim),
        int(head.num_decoder_layers),
        private_seed=private_seed,
    )
    head.decoder = FDRDeformableTransformerDecoder.from_stock(
        head.decoder,
        pre_bbox_head=stock_pre_bbox_head,
    )
    head.dec_bbox_head = distribution_heads
    head.decoder.reg_max = 32
    head.decoder.final_layers = [module.layers[-1] for module in distribution_heads]
    return model


def _fdr(seed: int = 0, private_seed: int = 10_000) -> FDRRTDETRDetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return FDRRTDETRDetectionModel(
            _declarative_cfg(private_seed=private_seed),
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


def test_declarative_yaml_builds_the_fdr_head_without_post_build_replacement() -> None:
    model = _fdr()
    assert isinstance(model.model[-1], FDRRTDETRDecoder)
    assert isinstance(model.fdr, FDRDeformableTransformerDecoder)


def test_declarative_model_matches_legacy_state_contract_exactly() -> None:
    expected = _legacy_fdr()
    actual = _fdr()
    assert expected.state_dict().keys() == actual.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], tensor, rtol=0, atol=0)


def test_declarative_model_matches_legacy_eval_output_exactly() -> None:
    expected_model = _legacy_fdr()
    actual_model = _fdr()
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    expected_model.eval()
    actual_model.eval()
    image = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        expected_output, expected_raw = expected_model(image)
        actual_output, actual_raw = actual_model(image)

    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    for actual, expected in zip(actual_raw[:-1], expected_raw[:-1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_raw[-1] is expected_raw[-1] is None


def test_fdr_yaml_parser_registration_is_restored_after_construction() -> None:
    stock_decoder_type = ultralytics_tasks.RTDETRDecoder
    _fdr()
    assert ultralytics_tasks.RTDETRDecoder is stock_decoder_type
    assert type(_stock().model[-1]) is stock_decoder_type


def test_fdr_model_parsing_is_serialized_across_threads(monkeypatch) -> None:
    """Two FDR builders must never overlap while the parser alias is installed."""

    import src.rtdetr_fdr as integration

    native_decoder = ultralytics_tasks.RTDETRDecoder
    first_inside = threading.Event()
    release_first = threading.Event()
    second_inside = threading.Event()
    failures: list[BaseException] = []
    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    class FakeFDRHead:
        num_queries = 300
        num_decoder_layers = 6
        fdr_options = {"private_seed": 10_000}

    def fake_parent_init(self, *args, **kwargs):
        del args, kwargs
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            assert ultralytics_tasks.RTDETRDecoder is FakeFDRHead
            if threading.current_thread().name == "fdr-first":
                first_inside.set()
                assert release_first.wait(2)
            else:
                second_inside.set()
            self.model = [FakeFDRHead()]
            self.yaml = {"nc": 10, "fdr_loss": {}}
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(integration, "FDRRTDETRDecoder", FakeFDRHead)
    monkeypatch.setattr(RTDETRDetectionModel, "__init__", fake_parent_init)

    def build() -> None:
        try:
            integration.FDRRTDETRDetectionModel(_declarative_cfg(), verbose=False)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    first = threading.Thread(target=build, name="fdr-first")
    second = threading.Thread(target=build, name="fdr-second")
    first.start()
    assert first_inside.wait(2)
    second.start()
    try:
        assert not second_inside.wait(0.1)
    finally:
        release_first.set()
    first.join(2)
    second.join(2)

    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert second_inside.is_set()
    assert max_active == 1
    assert ultralytics_tasks.RTDETRDecoder is native_decoder


def test_fdr_and_control_model_parsing_share_one_lock(monkeypatch) -> None:
    """A stock control must not parse while the FDR decoder alias is active."""

    import src.rtdetr_fdr as integration

    native_decoder = ultralytics_tasks.RTDETRDecoder
    fdr_inside = threading.Event()
    release_fdr = threading.Event()
    control_inside = threading.Event()
    failures: list[BaseException] = []
    observed_control_alias: list[object] = []

    class FakeFDRHead:
        num_queries = 300
        num_decoder_layers = 6
        fdr_options = {"private_seed": 10_000}

    def fake_parent_init(self, *args, **kwargs):
        del args, kwargs
        assert ultralytics_tasks.RTDETRDecoder is FakeFDRHead
        fdr_inside.set()
        assert release_fdr.wait(2)
        self.model = [FakeFDRHead()]
        self.yaml = {"nc": 10, "fdr_loss": {}}

    class FakeStockModel:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            observed_control_alias.append(ultralytics_tasks.RTDETRDecoder)
            control_inside.set()

    monkeypatch.setattr(integration, "FDRRTDETRDecoder", FakeFDRHead)
    monkeypatch.setattr(RTDETRDetectionModel, "__init__", fake_parent_init)
    monkeypatch.setattr(integration, "RTDETRDetectionModel", FakeStockModel)

    control_trainer = object.__new__(FDRControlTrainer)
    control_trainer.data = {"nc": 10, "channels": 3}
    control_trainer.initial_state_path = None

    def build_fdr() -> None:
        try:
            integration.FDRRTDETRDetectionModel(_declarative_cfg(), verbose=False)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def build_control() -> None:
        try:
            control_trainer.get_model(verbose=False)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    fdr_thread = threading.Thread(target=build_fdr)
    control_thread = threading.Thread(target=build_control)
    fdr_thread.start()
    assert fdr_inside.wait(2)
    control_thread.start()
    try:
        assert not control_inside.wait(0.1)
    finally:
        release_fdr.set()
    fdr_thread.join(2)
    control_thread.join(2)

    assert not fdr_thread.is_alive() and not control_thread.is_alive()
    assert failures == []
    assert observed_control_alias == [native_decoder]
    assert ultralytics_tasks.RTDETRDecoder is native_decoder


def test_fdr_criterion_reads_loss_options_from_model_yaml() -> None:
    model = FDRRTDETRDetectionModel(
        _declarative_cfg(fgl_weight=0.0, supervise_pre_boxes=False),
        nc=10,
        verbose=False,
    )
    criterion = model.init_criterion()
    assert criterion.fgl_weight == 0.0
    assert criterion.supervise_pre_boxes is False


@pytest.mark.parametrize(
    ("filename", "cumulative", "preliminary_box", "fgl_weight", "pre_loss"),
    (
        ("rtdetr-l-fdr.yaml", True, True, 0.15, True),
        ("rtdetr-l-fdr-no-fgl.yaml", True, True, 0.0, True),
        ("rtdetr-l-fdr-no-prebox-loss.yaml", True, True, 0.15, False),
        ("rtdetr-l-fdr-no-cumulative.yaml", False, True, 0.15, True),
        ("rtdetr-l-fdr-no-prebox.yaml", True, False, 0.15, False),
    ),
)
def test_each_standalone_ablation_yaml_builds_one_compatible_functional_unit(
    filename: str,
    cumulative: bool,
    preliminary_box: bool,
    fgl_weight: float,
    pre_loss: bool,
    fdr_model: FDRRTDETRDetectionModel,
) -> None:
    method = FDRRTDETRDetectionModel(
        FDR_CONFIG_ROOT / filename,
        nc=10,
        verbose=False,
    )
    assert fdr_model.state_dict().keys() == method.state_dict().keys()
    for key in fdr_model.state_dict():
        assert fdr_model.state_dict()[key].shape == method.state_dict()[key].shape
    assert method.fdr.cumulative is cumulative
    assert method.fdr.preliminary_box is preliminary_box
    criterion = method.init_criterion()
    assert criterion.fgl_weight == fgl_weight
    assert criterion.supervise_pre_boxes is pre_loss


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
        _declarative_cfg(private_seed=10_731),
        nc=10,
        verbose=False,
        private_seed=10_731,
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


def test_real_batch_criterion_loss_and_backward_cover_fdr_and_pre_heads():
    model = _fdr(private_seed=30_000)
    model.train()
    batch = {
        "img": torch.zeros(2, 3, 128, 128),
        "cls": torch.tensor([[1], [2]], dtype=torch.float32),
        "bboxes": torch.tensor(
            [[0.50, 0.50, 0.20, 0.20], [0.35, 0.40, 0.15, 0.10]],
            dtype=torch.float32,
        ),
        "batch_idx": torch.tensor([0, 1], dtype=torch.float32),
    }
    assert isinstance(model.init_criterion(), FDRDetectionLoss)
    total, displayed = model.loss(batch)
    assert total.ndim == 0 and torch.isfinite(total)
    assert displayed.shape == (3,) and torch.isfinite(displayed).all()
    losses = model.last_fdr_losses
    assert {
        "loss_fgl",
        "loss_fgl_aux",
        "loss_fgl_dn",
        "loss_fgl_aux_dn",
        "loss_bbox_pre",
        "loss_giou_pre",
        "loss_bbox_pre_dn",
        "loss_giou_pre_dn",
    }.issubset(losses)
    torch.testing.assert_close(
        displayed,
        torch.stack(
            [losses[name].detach() for name in ("loss_giou", "loss_class", "loss_bbox")]
        ),
        rtol=0,
        atol=0,
    )
    assert model.criterion.fgl_extra_match_calls == 0
    assert model.criterion.stock_match_calls == 7

    total.backward()
    private_parameters = [
        parameter
        for head in model.model[-1].dec_bbox_head
        for parameter in head.parameters()
    ]
    pre_parameters = list(model.model[-1].decoder.pre_bbox_head.parameters())
    assert private_parameters and pre_parameters
    assert all(parameter.grad is not None for parameter in private_parameters)
    assert all(parameter.grad is not None for parameter in pre_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in private_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in pre_parameters)


def test_real_batch_fgl_zero_preserves_exact_stock_subtotal_without_rematching():
    model = _fdr(private_seed=31_000)
    model.train()
    targets = _targets(2)
    predictions = model.predict(torch.zeros(2, 3, 128, 128), batch=targets)
    dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta = predictions
    evidence = model.last_fdr_evidence
    assert evidence is not None and dn_meta is not None
    dn_boxes, normal_boxes = torch.split(dec_boxes, dn_meta["dn_num_split"], dim=2)
    dn_scores, normal_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
    stock_predictions = (
        torch.cat([enc_boxes.unsqueeze(0), normal_boxes]),
        torch.cat([enc_scores.unsqueeze(0), normal_scores]),
    )

    stock = RTDETRDetectionLoss(nc=10, use_vfl=True)
    expected = stock(
        stock_predictions,
        targets,
        dn_bboxes=dn_boxes,
        dn_scores=dn_scores,
        dn_meta=dn_meta,
    )
    fdr = FDRDetectionLoss(
        nc=10, use_vfl=True, fgl_weight=0.0, supervise_pre_boxes=False
    )
    actual = fdr(
        stock_predictions,
        targets,
        dn_bboxes=dn_boxes,
        dn_scores=dn_scores,
        dn_meta=dn_meta,
        corner_logits=evidence.corner_logits,
        pre_boxes=evidence.pre_boxes,
        dn_corner_logits=evidence.dn_corner_logits,
        dn_pre_boxes=evidence.dn_pre_boxes,
    )
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(
        stock_loss_subtotal(actual), stock_loss_subtotal(expected), rtol=0, atol=0
    )
    assert actual["loss_fgl"].item() == 0.0
    assert actual["loss_fgl_dn"].item() == 0.0
    assert fdr.fgl_extra_match_calls == 0


@pytest.mark.parametrize("preliminary_box", (True, False))
def test_fgl_reference_follows_the_enabled_box_representation(preliminary_box: bool):
    model = FDRRTDETRDetectionModel(
        _declarative_cfg(preliminary_box=preliminary_box),
        nc=10,
        verbose=False,
    )
    model.train()
    normal_queries, denoising_queries = 2, 1
    references = torch.full((6, 1, normal_queries, 4), 0.20)
    pre_boxes = torch.full((1, normal_queries, 4), 0.80)
    dn_references = torch.full((6, 1, denoising_queries, 4), 0.30)
    dn_pre_boxes = torch.full((1, denoising_queries, 4), 0.70)
    model.last_fdr_evidence = FDRTrainingEvidence(
        corner_logits=torch.zeros(6, 1, normal_queries, FDR_OUTPUT_DIM),
        references=references,
        pre_boxes=pre_boxes,
        dn_corner_logits=torch.zeros(6, 1, denoising_queries, FDR_OUTPUT_DIM),
        dn_references=dn_references,
        dn_pre_boxes=dn_pre_boxes,
    )
    recorded: dict[str, object] = {}

    class RecordingCriterion:
        def stock_plus_fgl(self, predictions, targets, **kwargs):
            del predictions, targets
            recorded.update(kwargs)
            zero = torch.zeros((), requires_grad=True)
            return {"loss_giou": zero, "loss_class": zero, "loss_bbox": zero}

    model.criterion = RecordingCriterion()
    total_queries = normal_queries + denoising_queries
    predictions = (
        torch.zeros(6, 1, total_queries, 4),
        torch.zeros(6, 1, total_queries, 10),
        torch.zeros(1, normal_queries, 4),
        torch.zeros(1, normal_queries, 10),
        {"dn_num_split": [denoising_queries, normal_queries]},
    )
    batch = {
        "img": torch.zeros(1, 3, 128, 128),
        "cls": torch.zeros((0, 1)),
        "bboxes": torch.zeros((0, 4)),
        "batch_idx": torch.zeros((0,)),
    }

    total, _displayed = model.loss(batch, predictions)

    assert torch.isfinite(total)
    expected = pre_boxes if preliminary_box else references[0]
    expected_dn = dn_pre_boxes if preliminary_box else dn_references[0]
    torch.testing.assert_close(recorded["pre_boxes"], expected, rtol=0, atol=0)
    torch.testing.assert_close(recorded["dn_pre_boxes"], expected_dn, rtol=0, atol=0)
    assert recorded["pre_boxes"].data_ptr() == expected.data_ptr()
    assert recorded["dn_pre_boxes"].data_ptr() == expected_dn.data_ptr()


def test_validation_loss_accepts_stock_eval_prediction_wrapper():
    model = _fdr(private_seed=31_001)
    model.eval()
    image = torch.zeros(2, 3, 128, 128)
    targets = {**_targets(2), "img": image}
    with torch.inference_mode():
        predictions = model.predict(image)
        total, displayed = model.loss(targets, predictions)

    assert torch.isfinite(total)
    assert displayed.shape == (3,)
    assert torch.isfinite(displayed).all()


def test_no_excluded_modules_and_installed_ultralytics_is_not_modified():
    source = Path(inspect.getsourcefile(RTDETRDetectionModel) or "")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    model = _fdr(private_seed=20_000)
    module_names = tuple(name.lower() for name, _ in model.named_modules())
    assert not any(token in name for token in EXCLUDED for name in module_names)
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    assert after == before


def test_fdr_trainers_inherit_fixed_paired_musgd_amp_contract():
    assert issubclass(FDRTrainer, FixedPairedProtocolMixin)
    assert issubclass(FDRControlTrainer, FixedPairedProtocolMixin)
    assert FDRTrainer.controlled_amp_scale == 128.0
    assert FDRControlTrainer.controlled_amp_scale == 128.0


def test_fdr_setup_skips_only_the_network_dependent_generic_amp_probe(monkeypatch):
    import ultralytics.engine.trainer as engine_trainer

    original = lambda _model: False
    observed: list[bool] = []
    monkeypatch.setattr(engine_trainer, "check_amp", original)
    monkeypatch.setattr(
        FixedPairedProtocolMixin,
        "_setup_train",
        lambda _self: observed.append(engine_trainer.check_amp(object())),
    )

    trainer = object.__new__(FDRTrainer)
    trainer._setup_train()

    assert observed == [True]
    assert engine_trainer.check_amp is original


def test_fdr_gradient_groups_partition_common_and_all_private_parameters():
    model = _fdr(private_seed=40_000)
    trainer = object.__new__(FDRTrainer)
    trainer.model = model
    groups = trainer.gradient_parameter_groups()
    assert set(groups) == {"gradient_norm", "fdr_gradient_norm"}
    common = {id(parameter) for parameter in groups["gradient_norm"]}
    private = {id(parameter) for parameter in groups["fdr_gradient_norm"]}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert common and private
    assert common.isdisjoint(private)
    assert common | private == expected

    named = dict(model.named_parameters())
    expected_private = {
        id(parameter)
        for name, parameter in named.items()
        if ".dec_bbox_head." in name or ".decoder.pre_bbox_head." in name
    }
    assert private == expected_private


def test_fdr_and_control_get_model_build_matched_variants():
    method_trainer = object.__new__(FDRTrainer)
    method_trainer.data = {"nc": 10, "channels": 3}
    method_trainer.experiment_seed = 0
    method_trainer.initial_state_path = None
    method = method_trainer.get_model(verbose=False)
    assert isinstance(method, FDRRTDETRDetectionModel)
    assert method.private_seed == 10_000

    control_trainer = object.__new__(FDRControlTrainer)
    control_trainer.data = {"nc": 10, "channels": 3}
    control_trainer.initial_state_path = None
    control = control_trainer.get_model(verbose=False)
    assert type(control) is RTDETRDetectionModel


def test_trainers_load_protocol_initial_state_with_exact_variant(monkeypatch, tmp_path):
    artifact = {"authority": "sentinel"}
    state_path = tmp_path / "initial-state.pt"
    state_path.write_bytes(b"sentinel")
    monkeypatch.setattr("src.rtdetr_fdr.torch.load", lambda *args, **kwargs: artifact)
    calls: list[tuple[object, object, str]] = []

    def fake_load(model, loaded, *, variant):
        calls.append((model, loaded, variant))

    monkeypatch.setattr("src.rtdetr_fdr.load_fdr_initial_state", fake_load)

    method_trainer = object.__new__(FDRTrainer)
    method_trainer.data = {"nc": 10, "channels": 3}
    method_trainer.experiment_seed = 0
    method_trainer.initial_state_path = state_path
    method = method_trainer.get_model(verbose=False)

    control_trainer = object.__new__(FDRControlTrainer)
    control_trainer.data = {"nc": 10, "channels": 3}
    control_trainer.initial_state_path = state_path
    control = control_trainer.get_model(verbose=False)
    assert calls == [(method, artifact, "fdr"), (control, artifact, "control")]


def test_fdr_resume_applies_only_runtime_overrides(monkeypatch):
    trainer = object.__new__(FDRTrainer)
    trainer.args = SimpleNamespace(epochs=50, workers=8)

    def fake_parent(self, overrides):
        del overrides
        self.resume = True

    monkeypatch.setattr(
        "src.rtdetr_fdr.RTDETRTrainer.check_resume", fake_parent
    )
    recorded: list[dict] = []
    monkeypatch.setattr(
        "src.rtdetr_fdr.apply_resume_runtime_overrides",
        lambda args, overrides: recorded.append(dict(overrides)),
    )
    overrides = {"epochs": 100, "workers": 8}
    trainer.check_resume(overrides)
    assert recorded == [overrides]
    assert trainer.args.epochs == 100


def test_legacy_fdr_resume_normalizes_stock_yaml_and_preserves_checkpoint_state(
    monkeypatch,
):
    import ultralytics.engine.trainer as engine_trainer

    legacy = _legacy_fdr(private_seed=10_000)
    legacy.yaml = deepcopy(yaml_model_load("rtdetr-l.yaml"))
    optimizer_state = {"sentinel": "optimizer"}
    scaler_state = {"sentinel": "scaler"}
    checkpoint = {
        "epoch": 4,
        "model": legacy,
        "ema": legacy,
        "optimizer": optimizer_state,
        "scaler": scaler_state,
        "updates": 17,
    }
    monkeypatch.setattr(
        engine_trainer,
        "load_checkpoint",
        lambda _path: (legacy, checkpoint),
    )

    trainer = object.__new__(FDRTrainer)
    trainer.model = "legacy-formal-last.pt"
    trainer.args = SimpleNamespace(pretrained=False)
    trainer.resume = True
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 0
    trainer.initial_state_path = None

    returned = trainer.setup_model()

    assert returned is checkpoint
    assert returned["optimizer"] is optimizer_state
    assert returned["scaler"] is scaler_state
    assert returned["ema"] is legacy
    assert isinstance(trainer.model, FDRRTDETRDetectionModel)
    assert isinstance(trainer.model.model[-1], FDRRTDETRDecoder)
    assert trainer.model.yaml["head"][-1][2] == "FDRRTDETRDecoder"
    assert trainer.model.state_dict().keys() == legacy.state_dict().keys()


def test_fresh_stock_model_is_not_misclassified_as_legacy_fdr() -> None:
    trainer = object.__new__(FDRTrainer)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 0
    trainer.initial_state_path = None
    stock = _stock()

    with pytest.raises(TypeError, match="must end with FDRRTDETRDecoder"):
        trainer.get_model(cfg=stock.yaml, weights=stock, verbose=False)
