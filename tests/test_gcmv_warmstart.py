from __future__ import annotations

import math

import pytest
import torch

from src.gcmv_warmstart import (
    EXPECTED_BASELINE_SHA256,
    PLEC_EXTRA_PREFIXES,
    build_module_artifact,
    load_baseline_detector_state,
    load_module_artifact,
    open_residual_scalar,
    split_warmstart_optimizer_groups,
)


class FakeStock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stock = torch.nn.Linear(2, 2)


class FakeGCMV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stock = torch.nn.Linear(2, 2)
        self.plec = torch.nn.Linear(2, 2)
        self.gcmv_injector = torch.nn.Module()
        self.gcmv_injector.gglf = torch.nn.Linear(2, 2)
        self.gcmv_injector.peg = torch.nn.Module()
        self.gcmv_injector.peg.rho = torch.nn.Parameter(torch.zeros(()))


class FakeGCMVWithBatchNorm(FakeGCMV):
    def __init__(self):
        super().__init__()
        self.stock_bn = torch.nn.BatchNorm1d(2)


def test_expected_baseline_hash_is_the_published_rtx4090_artifact():
    assert EXPECTED_BASELINE_SHA256 == (
        "54CE60289DD34C6750B8BA5F7516EEF"
        "CF3AFEF6C174C6E4F3B1EF810C883099B"
    )


def test_baseline_loader_replaces_only_detector_state():
    model = FakeGCMV()
    stock = FakeStock()
    for parameter in stock.parameters():
        parameter.data.fill_(3.0)
    before_extra = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if name.startswith(PLEC_EXTRA_PREFIXES)
    }

    load_baseline_detector_state(model, {"model": stock})

    assert torch.equal(
        model.state_dict()["stock.weight"],
        stock.state_dict()["stock.weight"],
    )
    for name, value in before_extra.items():
        assert torch.equal(model.state_dict()[name], value)


def test_module_artifact_round_trip_and_residual_opening():
    source = FakeGCMV()
    for name, parameter in source.named_parameters():
        if name.startswith(PLEC_EXTRA_PREFIXES):
            parameter.data.fill_(0.25)
    source.gcmv_injector.peg.rho.data.zero_()
    artifact = build_module_artifact(source)
    target = FakeGCMV()

    load_module_artifact(target, artifact)
    open_residual_scalar(target, gamma=0.02)

    for name, value in artifact["module_state"].items():
        if name.endswith("peg.rho"):
            continue
        assert torch.equal(target.state_dict()[name], value)
    assert math.isclose(
        torch.tanh(target.gcmv_injector.peg.rho).item(),
        0.02,
        rel_tol=0.0,
        abs_tol=1e-7,
    )


def test_module_artifact_rejects_detector_keys():
    model = FakeGCMV()
    artifact = build_module_artifact(model)
    artifact["module_state"]["stock.weight"] = model.stock.weight.detach()

    with pytest.raises(ValueError, match="module-only"):
        load_module_artifact(model, artifact)


def test_module_artifact_load_accepts_batchnorm_tracking_buffers():
    source = FakeGCMVWithBatchNorm()
    target = FakeGCMVWithBatchNorm()

    load_module_artifact(target, build_module_artifact(source))

    assert target.stock_bn.num_batches_tracked.item() == 0


def test_optimizer_groups_separate_detector_module_and_rho_lrs():
    model = FakeGCMV()
    optimizer = torch.optim.SGD(
        [{"params": list(model.parameters()), "lr": 1e-4}],
        lr=1e-4,
    )

    split_warmstart_optimizer_groups(
        optimizer,
        model=model,
        detector_lr=1e-4,
        module_lr=1e-3,
        rho_lr=1e-3,
        include_module=True,
    )

    groups = {group["gcmv_role"]: group for group in optimizer.param_groups}
    assert set(groups) == {"detector", "module", "rho"}
    assert groups["detector"]["lr"] == 1e-4
    assert groups["module"]["lr"] == 1e-3
    assert groups["rho"]["lr"] == 1e-3
    assert groups["rho"]["weight_decay"] == 0.0
    all_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(all_ids) == len(set(all_ids))
