from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.bpdd_capacity import LocalBoundaryExpert, boundary_grid, sample_local_features


def test_boundary_geometry_and_invalid_mask():
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.2], [2., 2., .1, .1]]])
    grid, valid = boundary_grid(boxes)
    assert grid.shape == (1, 2, 41, 2)
    torch.testing.assert_close(grid[0, 0, -1], torch.zeros(2))
    assert valid[0, 0].all() and not valid[0, 1].any()
    feature = torch.arange(16.).reshape(1, 1, 4, 4)
    tokens = sample_local_features(feature, grid)
    torch.testing.assert_close(tokens[0, 0, -1], torch.tensor([7.5]))


def test_expert_initial_identity_and_rng_isolation():
    torch.manual_seed(432)
    before = torch.random.get_rng_state().clone()
    module = LocalBoundaryExpert(channels=(8, 8, 8), hidden=32, nc=3, heads=4, ff=64)
    assert torch.equal(before, torch.random.get_rng_state())
    h = torch.randn(2, 4, 32)
    z = torch.randn(2, 4, 132)
    c = torch.randn(2, 4, 3)
    b = torch.tensor([.5, .5, .3, .2]).expand(2, 4, 4)
    features = [torch.randn(2, 8, s, s) for s in (16, 8, 4)]
    nz, nc = module(h, z, c, b, features)
    torch.testing.assert_close(nz, z, rtol=0, atol=0)
    torch.testing.assert_close(nc, c, rtol=0, atol=0)
    assert all(torch.equal(v, deepcopy(module).state_dict()[k]) for k,v in module.state_dict().items())


def test_expert_learns_visual_features_but_not_sampling_boxes():
    module = LocalBoundaryExpert(channels=(8, 8, 8), hidden=32, nc=3, heads=4, ff=64)
    torch.nn.init.normal_(module.box_out.weight, std=.01)
    h = torch.randn(1, 2, 32, requires_grad=True)
    z = torch.randn(1, 2, 132, requires_grad=True)
    c = torch.randn(1, 2, 3, requires_grad=True)
    boxes = torch.tensor([[[.5,.5,.2,.2],[2.,2.,.1,.1]]], requires_grad=True)
    features = [torch.randn(1, 8, s, s, requires_grad=True) for s in (16,8,4)]
    out, _ = module(h,z,c,boxes,features)
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert boxes.grad is None
    assert h.grad.abs().sum() > 0
    assert all(f.grad is not None and f.grad.abs().sum() > 0 for f in features)


def test_expert_empty_queries():
    module = LocalBoundaryExpert(channels=(8,8,8), hidden=32, nc=3, heads=4, ff=64)
    z,c = module(torch.zeros(1,0,32),torch.zeros(1,0,132),torch.zeros(1,0,3),
                 torch.zeros(1,0,4),[torch.zeros(1,8,4,4)]*3)
    assert z.shape == (1,0,132) and c.shape == (1,0,3)
