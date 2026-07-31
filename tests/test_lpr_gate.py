from __future__ import annotations

import csv
import json

from scripts.evaluate_lpr_gate import ArmScreenMetrics, evaluate_paired_screen, evaluate_pairs_manifest


def _arm(final, tail, l1, giou, *, gate=True):
    return ArmScreenMetrics(
        final_map=final,
        tail3_map=tail,
        tail3_l1=l1,
        tail3_giou=giou,
        finite=True,
        optimizer_valid=True,
        gate_active=gate,
        run_path="fixture",
    )


def test_three_seed_pair_passes_frozen_lpr_screen() -> None:
    controls = {
        0: _arm(0.00008, 0.00007, 0.30, 1.5),
        1: _arm(0.00005, 0.00004, 0.31, 1.6),
        2: _arm(0.00001, 0.00001, 0.32, 1.7),
    }
    methods = {
        0: _arm(0.00009, 0.00008, 0.29, 1.4),
        1: _arm(0.00006, 0.00005, 0.30, 1.5),
        2: _arm(0.000009, 0.000009, 0.31, 1.6),
    }

    report = evaluate_paired_screen(controls, methods)

    assert report["passed"]
    assert report["checks"]["final_map"]
    assert report["checks"]["tail3_map"]
    assert report["checks"]["localization"]
    assert report["recommendation"] == "fresh_full_data_seed0_pair"


def test_pair_rejects_one_seed_win_or_inactive_gate() -> None:
    controls = {seed: _arm(0.10, 0.10, 0.2, 1.0) for seed in (0, 1, 2)}
    methods = {
        0: _arm(0.11, 0.11, 0.19, 0.9),
        1: _arm(0.09, 0.09, 0.19, 0.9),
        2: _arm(0.09, 0.09, 0.19, 0.9, gate=False),
    }

    report = evaluate_paired_screen(controls, methods)

    assert not report["passed"]
    assert not report["checks"]["final_map"]
    assert not report["checks"]["lpr_evidence"]
    assert report["recommendation"] == "alpha_lr_multiplier_10"


def _write_run(path, maps, l1s, gious, *, lpr):
    path.mkdir(parents=True)
    with (path / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "metrics/mAP50-95(B)", "val/l1_loss", "val/giou_loss"),
        )
        writer.writeheader()
        for index, (map_value, l1, giou) in enumerate(zip(maps, l1s, gious), start=1):
            writer.writerow(
                {
                    "epoch": index,
                    "metrics/mAP50-95(B)": map_value,
                    "val/l1_loss": l1,
                    "val/giou_loss": giou,
                }
            )
    evidence = [
        {
            "optimizer_attempt": index,
            "amp_scale_before": 128.0,
            "amp_scale_after": 128.0,
            "amp_step_skipped": False,
            "gradient_norm_finite": True,
        }
        for index in range(1, 146)
    ]
    (path / "optimizer-evidence.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in evidence),
        encoding="utf-8",
    )
    if lpr:
        diagnostics = [
            {
                "epoch": epoch,
                "map75": 0.0,
                "gates": [0.001] * 6,
                "residual_mean": 0.1,
                "residual_max": 0.4,
                "lpr_grad_norm": 0.03,
                "cuda_peak_mib": 9000,
            }
            for epoch in range(1, 11)
        ]
        (path / "lpr_diagnostics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in diagnostics),
            encoding="utf-8",
        )


def test_manifest_parser_requires_ten_rows_and_145_fixed_amp_attempts(tmp_path) -> None:
    pairs = {}
    for seed in (0, 1, 2):
        control = tmp_path / f"control-{seed}"
        method = tmp_path / f"lpr-{seed}"
        _write_run(control, [0.01] * 10, [0.3] * 10, [1.5] * 10, lpr=False)
        _write_run(method, [0.011] * 10, [0.29] * 10, [1.4] * 10, lpr=True)
        pairs[str(seed)] = {"control": str(control), "lpr": str(method)}
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")

    report = evaluate_pairs_manifest(manifest)

    assert report["passed"]
    assert set(report["pairs"]) == {"0", "1", "2"}
    assert all(pair["delta"]["final_map"] > 0 for pair in report["pairs"].values())
