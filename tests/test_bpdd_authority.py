from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "research" / "bpdd" / "authority.json"


def test_bpdd_authority_freezes_upstream_and_fdr_evidence() -> None:
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))

    assert payload["format_version"] == 1
    assert payload["status"] == "research_candidate"
    assert payload["name"] == "Best-Progressive Distribution Distillation"
    assert payload["fdr_authority"] == {
        "source_commit": "d97e1eb7f98414752a1c1f38287697db3f2a0679",
        "protocol_sha256": "2545F68302639EEB217EC8B53FDB229681B2DAFC91463C61F6B5CEC9B5486302",
        "initial_state_sha256": "51AAB2EB3FB7D123501C69C7B8DC90FF3EA0B9344A108EDEEF2C7D6DCDBB742D",
        "epoch100_checkpoint_sha256": "C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2",
    }
    assert payload["dfine_authority"]["commit"] == (
        "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"
    )
    assert payload["dataset_sha256"] == (
        "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
    )


def test_bpdd_authority_records_collision_and_frozen_candidate() -> None:
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    origins = {item["name"]: item for item in payload["prior_work"]}

    assert set(origins) == {
        "D-FINE GO-LSD",
        "Localization Distillation",
        "Teacher-bounded Regression",
        "DETRDistill",
        "KD-DETR",
        "BYOT",
    }
    assert all(item["url"].startswith("https://") for item in origins.values())
    assert payload["candidate"] == {
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1e-6,
        "matching": "final_stock_assignment_only",
        "queries": "matched_normal_only",
        "teacher": "gt_proper_score_softmin_future_layers",
        "gate": "actual_mixture_strictly_better",
        "include_dn": False,
    }
    forbidden = set(payload["forbidden_claims"])
    assert {
        "invented_self_distillation",
        "invented_distribution_distillation",
        "invented_better_teacher_gate",
        "exact_go_lsd_reproduction",
        "guaranteed_accuracy_gain",
    }.issubset(forbidden)

