from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.fdr_math import cxcywh_to_xyxy
from src.rtdetr_lrs_system import MODEL_TYPES


@pytest.mark.parametrize("arm", list("fghi"))
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "cpu-bf16"])
def test_current_arm_real_forward_backward_and_eval(arm, autocast):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9100)
        model = MODEL_TYPES[arm](nc=10, verbose=False).train()
        batch = {
            "img": torch.rand(1, 3, 128, 128),
            "cls": torch.tensor([[0.0], [1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.05, 0.05]]),
            "batch_idx": torch.zeros(2),
        }
        state_keys = tuple(model.state_dict())
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            total, _ = model.loss(batch)
        assert torch.isfinite(total)
        total.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert model.fdr.last_geometry_statistics["minimum_decoded_width"] > 0
        if arm in "gi":
            assert "loss_bpdd" in model.last_fdr_losses
        else:
            assert "loss_bpdd" not in model.last_fdr_losses
        torch.optim.SGD(model.parameters(), lr=1e-5).step()
        model.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            prediction, _ = model(batch["img"])
        assert torch.isfinite(prediction).all()
        boxes = prediction[..., :4]
        assert boxes.dtype == torch.float32
        xyxy = cxcywh_to_xyxy(boxes)
        assert torch.all(xyxy[..., 2:] > xyxy[..., :2])
        assert tuple(model.state_dict()) == state_keys


@pytest.mark.parametrize("arm", ["f", "i"])
def test_full_half_cpu_model_fails_explicitly_before_unreliable_grid_sample(arm):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        model = MODEL_TYPES[arm](nc=10, verbose=False).eval().half()
        with torch.no_grad(), pytest.raises(ValueError, match="CPU FP16.*FP32.*BF16"):
            model(torch.rand(1, 3, 128, 128).half())


def test_shared_runtime_recorder_retains_decoded_extents_and_absence(tmp_path):
    from src import rtdetr_lrs_system
    import importlib.util
    assert importlib.util.find_spec("src.lrs_runtime_evidence") is not None
    from src.lrs_runtime_evidence import RuntimeEvidenceRecorder
    trainer = SimpleNamespace(
        model=SimpleNamespace(fdr=SimpleNamespace(last_geometry_statistics={
            "total": torch.tensor(1), "horizontal_infeasible": torch.tensor(1),
            "vertical_infeasible": torch.tensor(0), "minimum_raw_horizontal": torch.tensor(-1.),
            "minimum_raw_vertical": torch.tensor(4.), "minimum_extent": torch.tensor(1e-3),
            "minimum_decoded_width": torch.tensor(1e-6), "minimum_decoded_height": torch.tensor(0.2),
        })), epoch=0, save_dir=tmp_path,
    )
    recorder = RuntimeEvidenceRecorder()
    recorder.capture(trainer)
    record = recorder.write(trainer)
    assert record["method_revision"] == rtdetr_lrs_system.SYSTEM_REVISION
    assert record["geometry_minimum_decoded_width"] == pytest.approx(1e-6)
    assert record["geometry_minimum_decoded_height"] == pytest.approx(0.2)
    assert record["bpdd_observations"] == 0
    assert record["bpdd_active_edge_ratio_mean"] is None
    assert record["gradients_finite"] is None
