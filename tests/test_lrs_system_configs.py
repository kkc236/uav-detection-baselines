from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "rtdetr-l-lrs-fdr.yaml"
CONFIGS = {
    "g": ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd.yaml",
    "h": ROOT / "configs" / "rtdetr-l-lrs-fdr-fia.yaml",
    "i": ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd-fia.yaml",
}
BPDD_OPTIONS = {
    "enabled": True,
    "weight": 0.5,
    "temperature": 0.5,
    "margin": 0.02,
    "eps": 1.0e-6,
    "matched_layer": "final",
    "include_dn": False,
}
LRS_LOSS = {
    "fgl_weight": 0.15,
    "supervise_pre_boxes": False,
    "supervise_dn_fdr": False,
    "edge_adaptive_fgl": False,
    "reliability_shrinkage_alpha": 0.25,
}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def base() -> dict:
    return _load(BASE_CONFIG)


@pytest.mark.parametrize("arm", ["g", "h", "i"])
def test_all_arm_configs_keep_lrs_contract(arm: str, base: dict) -> None:
    payload = _load(CONFIGS[arm])
    options = payload["head"][-1][3][-1]

    assert payload["nc"] == base["nc"]
    assert options["num_queries"] == 300
    assert options["num_decoder_layers"] == 6
    assert options["cumulative"] is True
    assert options["preliminary_box"] is False
    assert options["distribution_feedback"] is False
    assert payload["fdr_loss"] == LRS_LOSS


def test_bpdd_is_present_only_in_g_and_i() -> None:
    payloads = {arm: _load(path) for arm, path in CONFIGS.items()}

    assert payloads["g"]["bpdd_loss"] == BPDD_OPTIONS
    assert "bpdd_loss" not in payloads["h"]
    assert payloads["i"]["bpdd_loss"] == BPDD_OPTIONS


def test_g_is_the_base_lrs_graph_plus_bpdd(base: dict) -> None:
    payload = _load(CONFIGS["g"])
    payload.pop("bpdd_loss")

    assert payload == base
    assert all(layer[2] != "FIA" for layer in _load(CONFIGS["g"])["head"])


def test_fia_is_p3_only_in_h_and_i(base: dict) -> None:
    for arm in ("h", "i"):
        payload = _load(CONFIGS[arm])
        head = payload["head"]

        # Backbone has indices 0-9, so global model index 22 is head entry 12.
        assert head[:12] == base["head"][:12]
        assert head[12] == [21, 1, "FIA", [256]]
        assert head[13] == [21, 1, "Conv", [256, 3, 2]]
        assert head[-1][0] == [22, 25, 28]
        assert sum(layer[2] == "FIA" for layer in head) == 1


def test_h_has_no_bpdd_and_i_only_adds_bpdd_to_h() -> None:
    h = _load(CONFIGS["h"])
    i = _load(CONFIGS["i"])
    i.pop("bpdd_loss")

    assert "bpdd_loss" not in h
    assert i == h
