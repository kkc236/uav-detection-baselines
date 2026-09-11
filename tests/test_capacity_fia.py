from __future__ import annotations

from pathlib import Path

import yaml

from src.rtdetr_bpdd_capacity_fia import (
    CAPACITY_BPDD_FIA_CFG,
    FIA_MODEL_INDEX,
    remap_capacity_fia_shared_key,
)


def test_capacity_fia_yaml_preserves_four_scale_decoder_and_p3_bypass() -> None:
    payload = yaml.safe_load(Path(CAPACITY_BPDD_FIA_CFG).read_text(encoding="utf-8"))
    head = payload["head"]
    assert head[12][2] == "FIA"
    assert head[12][0] == 21
    assert head[13][0] == 21  # stock P4 bypasses FIA
    assert head[-1][0] == [1, 22, 25, 28]
    assert head[-1][2] == "FDRRTDETRDecoder"
    assert head[-1][3][1] == [128, 256, 256, 256]
    assert head[-1][3][2]["local_expert_preserve_base"] is True


def test_capacity_fia_state_remap_shifts_inserted_suffix_only() -> None:
    assert FIA_MODEL_INDEX == 22
    assert remap_capacity_fia_shared_key("model.21.foo") == "model.21.foo"
    assert remap_capacity_fia_shared_key("model.22.foo") == "model.23.foo"
    assert remap_capacity_fia_shared_key("model.28.decoder.layers.0.self_attn.weight") == (
        "model.29.decoder.layers.0.self_attn.weight"
    )
