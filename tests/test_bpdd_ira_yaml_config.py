from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BPDD = ROOT / "configs" / "rtdetr-l-fdr-bpdd.yaml"
COMBINED = ROOT / "configs" / "rtdetr-l-fdr-bpdd-ira.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_combined_yaml_preserves_backbone_fdr_and_bpdd_contracts() -> None:
    bpdd = _load(BPDD)
    combined = _load(COMBINED)

    assert combined["nc"] == bpdd["nc"]
    assert combined["scales"] == bpdd["scales"]
    assert combined["backbone"] == bpdd["backbone"]
    assert combined["fdr_loss"] == bpdd["fdr_loss"]
    assert combined["bpdd_loss"] == bpdd["bpdd_loss"]


def test_combined_yaml_inserts_one_ira_only_on_decoder_p3() -> None:
    combined = _load(COMBINED)
    head = combined["head"]

    assert len(head) == 20
    assert sum(row[2] == "IRA" for row in head) == 1
    assert head[12] == [21, 1, "IRA", [256]]

    # Absolute model indices 23--28 rebuild stock P4/P5.  P4 must explicitly
    # read the unmodified P3 at index 21, never the new IRA output at index 22.
    assert head[13][0] == 21
    assert head[14][0] == [23, 17]
    assert head[16][0] == 25
    assert head[17][0] == [26, 12]
    assert head[19][0] == [22, 25, 28]
    assert head[19][2] == "FDRRTDETRDecoder"
    assert head[19][3] == _load(BPDD)["head"][-1][3]
