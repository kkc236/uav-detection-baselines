from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = Path(os.environ.get("GLGM_TEST_REPO", ""))


@pytest.fixture(scope="module", autouse=True)
def require_audited_checkout():
    if not REPO_DIR.is_dir():
        pytest.skip(
            "set GLGM_TEST_REPO to an Ultralytics checkout with the v2 overlay installed"
        )


def run_isolated(source: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_DIR.resolve())
    environment["YOLO_AUTOINSTALL"] = "false"
    subprocess.run([sys.executable, "-c", source], check=True, env=environment)


def test_glgm_lite_initial_gate_residual_and_parameter_budget():
    run_isolated(
        """
import torch
from ultralytics.nn.modules import GLGMLite
module = GLGMLite(384, 384, 96, 16, True, 0.01)
inputs = torch.randn(2, 384, 20, 20, requires_grad=True)
outputs = module(inputs)
outputs.square().mean().backward()
assert outputs.shape == inputs.shape
assert torch.isfinite(outputs).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
assert module.gate_weights is not None
assert torch.equal(module.gate_weights, torch.full_like(module.gate_weights, 0.5))
assert sum(p.numel() for p in module.parameters()) <= 1_327_815
assert (outputs.detach() - inputs.detach()).square().mean() < inputs.detach().square().mean() * 0.01
"""
    )


def test_all_rtdetr_variant_configs_parse():
    configs = {
        "rtdetr-x-glgm-control.yaml": 0,
        "rtdetr-x-glgm-lite-equal-p5.yaml": 1,
        "rtdetr-x-glgm-lite-gated-p5.yaml": 1,
        "rtdetr-x-glgm-control-p4.yaml": 0,
        "rtdetr-x-glgm-lite-gated-p4.yaml": 1,
        "rtdetr-x-glgm-control-p3.yaml": 0,
        "rtdetr-x-glgm-lite-gated-p3.yaml": 1,
    }
    run_isolated(
        f"""
import gc
from pathlib import Path
from ultralytics import RTDETR
root = Path({str(PACKAGE_ROOT)!r})
configs = {configs!r}
for name, expected in configs.items():
    wrapper = RTDETR(str(root / 'configs' / name))
    actual = sum(m.__class__.__name__ == 'GLGMLite' for m in wrapper.model.modules())
    assert actual == expected, (name, actual, expected)
    del wrapper
    gc.collect()
"""
    )
