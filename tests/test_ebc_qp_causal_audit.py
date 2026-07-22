import json
from copy import deepcopy

import pytest
import torch
from torch import nn

from scripts.audit_ebc_qp_aux_causality import (
    _clip_coefficient_from_norm,
    _json_safe,
    batch_fingerprint,
    build_audit_ebc_config,
    build_audit_settings,
    build_parser,
    controlled_amp_config,
    gradient_alignment,
    optimizer_common_manifest,
    resolved_audit_steps,
    resolved_run_name,
    tensor_structure_fingerprint,
    validate_controlled_amp_runtime,
)
from src.ebc_qp_causal_audit import (
    capture_grouped_gradients,
    capture_grouped_parameter_deltas,
    capture_parameter_signatures,
    capture_tensor_sha256,
    classify_stock_parameter,
    classify_tsgr_gradient_boundary,
    clone_named_parameters,
    compare_a0_repeats,
    compare_audit_runs,
    compare_tsgr_audit_runs,
    validate_audit_attempt,
)


class _TinyStock(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([nn.Linear(2, 2) for _ in range(29)])
        head = self.model[28]
        head.enc_score_head = nn.Linear(2, 2)
        head.enc_bbox_head = nn.Linear(2, 4)
        head.decoder = nn.Linear(2, 2)


def test_classify_stock_parameter_covers_required_audit_boundaries():
    assert classify_stock_parameter("model.0.conv.weight") == "backbone_pre_c2"
    assert classify_stock_parameter("model.3.m.0.weight") == "backbone_c2"
    assert classify_stock_parameter("model.7.m.0.weight") == "backbone_post_c2"
    assert classify_stock_parameter("model.21.cv1.weight") == "p3_neck"
    assert classify_stock_parameter("model.28.input_proj.0.0.weight") == "encoder"
    assert classify_stock_parameter("model.28.enc_score_head.weight") == "stock_score_head"
    assert classify_stock_parameter("model.28.enc_bbox_head.layers.0.weight") == "stock_box_head"
    assert classify_stock_parameter("model.28.decoder.layers.0.weight") == "decoder"


def test_classify_tsgr_gradient_boundary_allows_only_model_zero_and_one():
    assert classify_tsgr_gradient_boundary("model.0.conv.weight") == "routed_shallow"
    assert classify_tsgr_gradient_boundary("model.1.block.weight") == "routed_shallow"
    assert classify_tsgr_gradient_boundary("model.2.conv.weight") == "forbidden_common"
    assert classify_tsgr_gradient_boundary("model.28.decoder.weight") == "forbidden_common"
    assert classify_tsgr_gradient_boundary("model.28.p2_adapter.0.weight") == "auxiliary_private"


def test_gradient_and_delta_summaries_detect_group_local_changes():
    model = _TinyStock()
    common = dict(model.named_parameters())
    before = clone_named_parameters(common)
    loss = model.model[3](torch.ones(1, 2)).sum() + model.model[28].enc_score_head(torch.ones(1, 2)).sum()
    loss.backward()

    gradients = capture_grouped_gradients(common)
    with torch.no_grad():
        model.model[3].weight.add_(0.25)
    deltas = capture_grouped_parameter_deltas(before, common)

    assert gradients["backbone_c2"]["l2"] > 0.0
    assert gradients["stock_score_head"]["l2"] > 0.0
    assert gradients["decoder"]["l2"] == 0.0
    assert deltas["backbone_c2"]["l2"] > 0.0
    assert deltas["stock_score_head"]["l2"] == 0.0


def _run(arm: str, *, clip: float, stock_delta: float, batch: str = "B", skipped: bool = False) -> dict:
    effective_delta = 0.0 if skipped else stock_delta
    return {
        "arm": arm,
        "target_optimizer_steps": 2,
        "attempted_optimizer_steps": 2,
        "completed_successful_updates": 2,
        "common_initial_fingerprint": "COMMON",
        "optimizer_common_manifest": {"model.0.weight": {"param_group": "muon"}},
        "p2_only_stock_grad_l2": 0.0,
        "p2_only_aux_private_grad_l2": 1.0 if arm == "aux-audit" else 0.0,
        "initial_probe": {
            "batch_fingerprint": "INITIAL-BATCH",
            "rng_before_forward": "INITIAL-RNG",
            "stock_topk_fingerprint": "TOPK",
            "decoder_output_fingerprint": "DECODER",
            "stock_output_fingerprint": "STOCK",
        },
        "steps": [
            {
                "optimizer_step": step,
                "optimizer_attempt": step,
                "successful_update": not skipped,
                "successful_update_index": None if skipped else step,
                "batch_fingerprints": [f"{batch}{step}"],
                "rng_before_forward": [f"R{step}"],
                "clip_coefficient": None if skipped else clip,
                "stock_only_clip_coefficient": None if skipped else 1.0,
                "stock_grad_total_norm": 3.0,
                "aux_private_grad_total_norm": 20.0 if arm == "aux-audit" else 0.0,
                "clip_norm_partition_relative_error": "NaN" if skipped else 0.0,
                "amp_step_skipped": skipped,
                "amp_scale_before": 65536.0,
                "amp_scale_after": 32768.0 if skipped else 65536.0,
                "stock_grad_preclip_finite": not skipped,
                "aux_private_grad_finite": not skipped,
                "all_grad_preclip_finite": not skipped,
                "stock_grad_postclip_finite": not skipped,
                "clip_total_norm_finite": not skipped,
                "loss_finite": True,
                "loss_items_finite": True,
                "model_parameters_finite": True,
                "stock_bn_finite": True,
                "stock_ema_finite": True,
                "optimizer_state_finite": True,
                "stock_delta_finite": True,
                "stock_delta_zero": skipped,
                "stock_grad_preclip": {"backbone_c2": {"l2": 3.0, "max_abs": 2.0}},
                "stock_grad_preclip_parameters": {"model.3.weight": {"l2": 3.0, "sum": 1.0, "max_abs": 2.0}},
                "stock_grad_preclip_sha256": {"model.3.weight": "GRAD"},
                "stock_delta": {"backbone_c2": {"l2": effective_delta, "max_abs": effective_delta}},
                "stock_delta_parameters": {
                    "model.3.weight": {
                        "l2": effective_delta,
                        "sum": effective_delta,
                        "max_abs": effective_delta,
                    }
                },
                "stock_delta_sha256": {"model.3.weight": f"DELTA-{effective_delta}"},
                "stock_bn": {"l2": 1.0, "sum": 1.0},
                "stock_bn_parameters": {"model.3.running_mean": {"l2": 1.0, "sum": 1.0, "max_abs": 1.0}},
                "stock_bn_sha256": {"model.3.running_mean": "BN"},
                "stock_ema": {"l2": 2.0, "sum": 2.0},
                "stock_ema_parameters": {"model.3.weight": {"l2": 2.0, "sum": 2.0, "max_abs": 2.0}},
                "optimizer_groups": [{"param_group": "weight", "lr": 0.1}],
                "optimizer_state_parameters": {
                    "model.3.weight.momentum_buffer": {"l2": 1.0, "sum": 1.0, "max_abs": 1.0}
                },
            }
            for step in (1, 2)
        ],
    }


def test_compare_audit_runs_classifies_global_clip_coupling():
    result = compare_audit_runs(
        _run("a0", clip=1.0, stock_delta=0.1),
        _run("aux-audit", clip=0.5, stock_delta=0.05),
    )

    assert result["pairing_valid"] is True
    assert result["first_stock_divergence_step"] == 1
    assert result["classification"] == "GLOBAL_CLIP_COUPLING"
    assert "global_clip" in result["mechanisms_detected"]


def test_compare_audit_runs_preserves_first_clip_cause_after_later_gradient_divergence():
    control = _run("a0", clip=1.0, stock_delta=0.1)
    auxiliary = _run("aux-audit", clip=0.5, stock_delta=0.05)
    auxiliary["steps"][1]["stock_grad_preclip_parameters"]["model.3.weight"]["sum"] = 1.5
    auxiliary["steps"][1]["stock_grad_preclip_sha256"]["model.3.weight"] = "GRAD-DIFFERENT"

    result = compare_audit_runs(control, auxiliary)

    assert result["first_clip_divergence_step"] == 1
    assert result["first_preclip_gradient_divergence_step"] == 2
    assert result["classification"] == "GLOBAL_CLIP_COUPLING"


def test_compare_audit_runs_uses_within_aux_counterfactual_clip_evidence():
    control = _run("a0", clip=1.0, stock_delta=0.1)
    auxiliary = _run("aux-audit", clip=0.5, stock_delta=0.05)
    auxiliary["steps"][0]["stock_grad_preclip_parameters"]["model.3.weight"]["sum"] = 1.25
    auxiliary["steps"][0]["stock_grad_preclip_sha256"]["model.3.weight"] = "GRAD-DIFFERENT"

    result = compare_audit_runs(control, auxiliary)

    assert result["first_preclip_gradient_divergence_step"] == 1
    assert result["first_counterfactual_clip_effect_step"] == 1
    assert result["classification"] == "GLOBAL_CLIP_COUPLING"
    assert "global_clip_counterfactual" in result["mechanisms_detected"]


def test_compare_audit_runs_rejects_invalid_gradient_norm_partition():
    auxiliary = _run("aux-audit", clip=0.5, stock_delta=0.05)
    auxiliary["steps"][0]["clip_norm_partition_relative_error"] = 0.01
    with pytest.raises(ValueError, match="gradient norm partition"):
        compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), auxiliary)


def test_compare_audit_runs_rejects_aux_probe_without_private_gradient():
    auxiliary = _run("aux-audit", clip=0.5, stock_delta=0.05)
    auxiliary["p2_only_aux_private_grad_l2"] = 0.0
    with pytest.raises(ValueError, match="AUX probe produced no auxiliary-private gradient"):
        compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), auxiliary)


def test_trace_json_marks_nonfinite_amp_attempts_without_emitting_invalid_json():
    safe = _json_safe({"nan": float("nan"), "positive": float("inf"), "negative": float("-inf")})
    assert safe == {"nan": "NaN", "positive": "+Infinity", "negative": "-Infinity"}
    json.dumps(safe, allow_nan=False)
    assert _clip_coefficient_from_norm(float("nan")) is None
    assert _clip_coefficient_from_norm(float("inf")) is None


def test_controlled_amp_runtime_rejects_silent_fp32_fallback():
    class Scaler:
        def __init__(self, enabled: bool, scale: float):
            self.enabled = enabled
            self.scale = scale

        def is_enabled(self):
            return self.enabled

        def get_scale(self):
            return self.scale

    config = {"enabled": True, "init_scale": 256.0}
    validate_controlled_amp_runtime(True, Scaler(True, 256.0), config)
    with pytest.raises(RuntimeError, match="requires AMP to remain enabled"):
        validate_controlled_amp_runtime(False, Scaler(False, 1.0), config)
    with pytest.raises(RuntimeError, match="scale mismatch"):
        validate_controlled_amp_runtime(True, Scaler(True, 1.0), config)


def test_compare_audit_runs_only_allows_nonfinite_partition_on_skipped_attempt():
    auxiliary = _run("aux-audit", clip=0.5, stock_delta=0.05, skipped=True)
    auxiliary["steps"][0]["clip_norm_partition_relative_error"] = "NaN"
    assert compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), auxiliary)["classification"] == "AMP_STEP_COUPLING"

    auxiliary["steps"][0]["amp_step_skipped"] = False
    auxiliary["steps"][0]["amp_scale_after"] = auxiliary["steps"][0]["amp_scale_before"]
    with pytest.raises(ValueError, match="non-finite gradient norm partition on successful update"):
        compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), auxiliary)


def test_validate_audit_attempt_rejects_nonfinite_state_or_nonzero_skipped_delta():
    skipped = _run("aux-audit", clip=0.0, stock_delta=0.0, skipped=True)["steps"][0]
    assert validate_audit_attempt(skipped) == []

    broken_state = deepcopy(skipped)
    broken_state["stock_bn_finite"] = False
    assert "stock_bn_finite must be true" in validate_audit_attempt(broken_state)

    broken_delta = deepcopy(skipped)
    broken_delta["stock_delta_zero"] = False
    assert "skipped AMP attempt changed stock parameters" in validate_audit_attempt(broken_delta)


def test_compare_audit_runs_classifies_amp_when_attempt_counts_differ():
    control = _run("a0", clip=1.0, stock_delta=0.1)
    auxiliary = _run("aux-audit", clip=1.0, stock_delta=0.1)
    third = deepcopy(auxiliary["steps"][1])
    auxiliary["steps"][1] = _run("aux-audit", clip=0.0, stock_delta=0.0, skipped=True)["steps"][1]
    third.update(
        optimizer_step=3,
        optimizer_attempt=3,
        successful_update=True,
        successful_update_index=2,
        amp_step_skipped=False,
        batch_fingerprints=["B3"],
        rng_before_forward=["R3"],
    )
    auxiliary["steps"].append(third)
    auxiliary["attempted_optimizer_steps"] = 3

    result = compare_audit_runs(control, auxiliary)
    assert result["classification"] == "AMP_STEP_COUPLING"
    assert result["first_amp_divergence_step"] == 2
    assert result["optimizer_attempt_count_mismatch"] is True


def test_compare_audit_runs_rejects_batch_or_optimizer_mismatch():
    with pytest.raises(ValueError, match="batch sequence mismatch"):
        compare_audit_runs(
            _run("a0", clip=1.0, stock_delta=0.1),
            _run("aux-audit", clip=1.0, stock_delta=0.1, batch="X"),
        )

    broken = _run("aux-audit", clip=1.0, stock_delta=0.1)
    broken["optimizer_common_manifest"] = {"model.0.weight": {"param_group": "weight"}}
    with pytest.raises(ValueError, match="optimizer common-parameter manifest mismatch"):
        compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), broken)

    broken_probe = _run("aux-audit", clip=1.0, stock_delta=0.1)
    broken_probe["initial_probe"]["decoder_output_fingerprint"] = "DIFFERENT"
    with pytest.raises(ValueError, match="initial stock forward/query mismatch"):
        compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), broken_probe)

    broken_controlled_amp = _run("aux-audit", clip=1.0, stock_delta=0.1)
    broken_controlled_amp["controlled_amp"] = {"enabled": True, "init_scale": 256.0}
    with pytest.raises(ValueError, match="controlled AMP configuration mismatch"):
        compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), broken_controlled_amp)


def test_compare_audit_runs_prioritizes_direct_gradient_or_amp_coupling():
    direct = _run("aux-audit", clip=1.0, stock_delta=0.1)
    direct["p2_only_stock_grad_l2"] = 0.25
    assert compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), direct)["classification"] == "DIRECT_GRADIENT_PATH"

    amp = _run("aux-audit", clip=1.0, stock_delta=0.0, skipped=True)
    assert compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), amp)["classification"] == "AMP_STEP_COUPLING"


def _tsgr_run(arm: str) -> dict:
    is_auxiliary = arm in {"h0", "h1"}
    eta = 0.0 if arm == "h0" else 0.1
    p2_signatures = {
        "model.0.weight": {"l2": 0.03 if arm == "h1" else 0.0, "max_abs": 0.03 if arm == "h1" else 0.0},
        "model.1.weight": {"l2": 0.04 if arm == "h1" else 0.0, "max_abs": 0.04 if arm == "h1" else 0.0},
        "model.2.weight": {"l2": 0.0, "max_abs": 0.0},
    }
    return {
        "arm": arm,
        "target_optimizer_steps": 1,
        "completed_successful_updates": 1,
        "common_initial_fingerprint": "COMMON",
        "initial_state_sha256": "INITIAL",
        "optimizer_common_manifest": {"model.0.weight": {"param_group": "muon"}},
        "controlled_amp": {"enabled": True},
        "ebc_config": (
            {
                "p2_c2_grad_scale": eta,
                "lambda_p2": 0.1,
                "lambda_ebc": 0.0,
                "query_injection_enabled": False,
                "contribution_separated_aux_gradients": True,
            }
            if is_auxiliary
            else None
        ),
        "p2_only_stock_grad_parameters": p2_signatures if is_auxiliary else {},
        "p2_only_aux_private_grad_l2": 1.0 if is_auxiliary else 0.0,
        "initial_probe": {
            "batch_fingerprint": "B",
            "rng_before_forward": "R",
            "stock_topk_fingerprint": "T",
            "decoder_output_fingerprint": "D",
            "stock_output_fingerprint": "S",
            "p2_loss": 2.0 if is_auxiliary else None,
            "p2_entry_count": 0,
            "ordinary_query_count": 300,
        },
        "steps": [
            {
                "amp_step_skipped": False,
                "gradient_clipping_mode": "contribution_separated" if is_auxiliary else "legacy_combined",
                "clip_coefficient": 1.0,
                "stock_only_clip_coefficient": 1.0,
                "routed_shallow_grad_total_norm": 0.05 if arm == "h1" else 0.0,
                "routed_shallow_clip_coefficient": 1.0,
                "p2_entry_count": 0,
                "ordinary_query_count": 300,
                "stock_grad_preclip_parameters": {
                    "model.0.weight": {"l2": 0.6},
                    "model.1.weight": {"l2": 0.8},
                    "model.2.weight": {"l2": 2.0},
                },
                "stock_delta_sha256": {
                    "model.0.weight": "SHALLOW-H1" if arm == "h1" else "SHALLOW-H0",
                    "model.1.weight": "SHALLOW-H1" if arm == "h1" else "SHALLOW-H0",
                    "model.2.weight": "DEEP",
                },
            }
        ],
    }


def test_tsgr_e0b_comparator_enforces_exact_gradient_boundary_and_ratio():
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), _tsgr_run("h0"), _tsgr_run("h1"))
    assert result["passed"] is True
    assert result["h1_shallow_applied_ratio_median"] == pytest.approx(0.05)

    leaked = _tsgr_run("h1")
    leaked["p2_only_stock_grad_parameters"]["model.2.weight"] = {"l2": 0.1, "max_abs": 0.1}
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), _tsgr_run("h0"), leaked)
    assert result["passed"] is False
    assert any("escaped" in error for error in result["errors"])


def test_audit_cli_freezes_100_steps_and_shared_training_settings(tmp_path):
    parser = build_parser()
    common = [
        "--initial-state",
        str(tmp_path / "initial.pt"),
        "--protocol-manifest",
        str(tmp_path / "protocol.json"),
        "--data",
        str(tmp_path / "data.yaml"),
        "--output",
        str(tmp_path / "audit.json"),
    ]
    a0 = parser.parse_args(["--arm", "a0", *common])
    auxiliary = parser.parse_args(["--arm", "aux-audit", *common])
    h0 = parser.parse_args(["--arm", "h0", *common])
    h1 = parser.parse_args(["--arm", "h1", *common])

    assert a0.steps == auxiliary.steps == 100
    assert resolved_audit_steps(a0) == 100
    assert build_audit_settings(a0) == build_audit_settings(auxiliary)
    assert build_audit_settings(a0)["amp"] is True
    assert build_audit_settings(a0)["batch"] == 8
    assert build_audit_settings(a0)["optimizer"] == "auto"
    assert resolved_run_name(h0) == "e0-h0-seed0"
    assert resolved_run_name(h1) == "e0-h1-seed0"
    assert build_audit_ebc_config(h0.arm).p2_c2_grad_scale == 0.0
    assert build_audit_ebc_config(h1.arm).p2_c2_grad_scale == 0.1
    assert build_audit_ebc_config(h1.arm).lambda_p2 == 0.1
    assert build_audit_ebc_config(h1.arm).query_injection_enabled is False
    assert build_audit_ebc_config(h1.arm).contribution_separated_aux_gradients is True

    smoke = parser.parse_args(["--arm", "a0", *common, "--smoke"])
    assert resolved_audit_steps(smoke) == 1
    assert resolved_run_name(smoke) == "e0-a0-seed0-smoke"

    repeat = parser.parse_args(["--arm", "a0-repeat", *common])
    assert build_audit_settings(repeat) == build_audit_settings(a0)
    assert resolved_run_name(repeat) == "e0-a0-repeat-seed0"

    controlled = parser.parse_args(["--arm", "a0", *common, "--controlled-amp-scale", "256"])
    assert resolved_audit_steps(controlled) == 32
    assert resolved_run_name(controlled) == "e0-a0-seed0-controlled-amp256-32step"
    assert controlled_amp_config(controlled) == {
        "enabled": True,
        "init_scale": 256.0,
        "growth_interval": 1000,
        "require_zero_skips": True,
    }
    controlled_100 = parser.parse_args(
        ["--arm", "h1", *common, "--controlled-amp-scale", "256", "--controlled-amp-steps", "100"]
    )
    assert resolved_audit_steps(controlled_100) == 100
    assert resolved_run_name(controlled_100) == "e0-h1-seed0-controlled-amp256-100step"


def test_gradient_alignment_uses_the_applied_contributions():
    result = gradient_alignment(
        {"model.0.weight": torch.tensor([3.0, 0.0])},
        {"model.0.weight": torch.tensor([0.0, 4.0])},
        stock_coefficient=0.5,
        routed_coefficient=0.25,
    )
    assert result["route_stock_cosine"] == pytest.approx(0.0)
    assert result["combined_to_stock_cosine"] == pytest.approx(1.5 / (3.25**0.5))


def test_batch_and_optimizer_manifests_are_stable_and_common_only():
    first = {"img": torch.arange(8).reshape(1, 2, 2, 2), "cls": torch.tensor([[1.0]])}
    second = {"cls": torch.tensor([[1.0]]), "img": torch.arange(8).reshape(1, 2, 2, 2)}
    changed = {"img": torch.arange(8).reshape(1, 2, 2, 2), "cls": torch.tensor([[2.0]])}
    assert batch_fingerprint(first) == batch_fingerprint(second)
    assert batch_fingerprint(first) != batch_fingerprint(changed)

    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    named = dict(model.named_parameters())
    optimizer = torch.optim.SGD(
        [
            {"params": [named["0.weight"]], "param_group": "weight", "lr": 0.1},
            {"params": [named["0.bias"]], "param_group": "bias", "lr": 0.2},
        ]
    )
    manifest = optimizer_common_manifest(optimizer, {"0.weight": named["0.weight"]})
    assert list(manifest) == ["0.weight"]
    assert manifest["0.weight"]["param_group"] == "weight"
    assert manifest["0.weight"]["lr"] == pytest.approx(0.1)


def test_parameter_signatures_distinguish_equal_norm_and_max_different_directions():
    first = capture_parameter_signatures({"model.3.weight": torch.tensor([1.0, -1.0, 0.0])})
    second = capture_parameter_signatures({"model.3.weight": torch.tensor([1.0, 0.0, -1.0])})

    assert first["model.3.weight"]["l2"] == pytest.approx(second["model.3.weight"]["l2"])
    assert first["model.3.weight"]["max_abs"] == pytest.approx(second["model.3.weight"]["max_abs"])
    assert first != second


def test_tensor_structure_fingerprint_covers_nested_decoder_outputs():
    first = (torch.tensor([[1.0, 2.0]]), [torch.tensor([3])], {"meta": None})
    same = (torch.tensor([[1.0, 2.0]]), [torch.tensor([3])], {"meta": None})
    changed = (torch.tensor([[1.0, 4.0]]), [torch.tensor([3])], {"meta": None})

    assert tensor_structure_fingerprint(first) == tensor_structure_fingerprint(same)
    assert tensor_structure_fingerprint(first) != tensor_structure_fingerprint(changed)


def test_full_tensor_sha256_detects_changes_outside_the_64_point_sample():
    first = torch.arange(1000, dtype=torch.float32)
    second = first.clone()
    second[101], second[102] = second[102].clone(), second[101].clone()

    assert capture_parameter_signatures({"model.3.weight": first}) == capture_parameter_signatures(
        {"model.3.weight": second}
    )
    assert capture_tensor_sha256({"model.3.weight": first}) != capture_tensor_sha256(
        {"model.3.weight": second}
    )


def test_a0_repeatability_report_calibrates_first_exact_divergence():
    reference = _run("a0", clip=1.0, stock_delta=0.1)
    repeat = _run("a0", clip=1.0, stock_delta=0.1)
    repeat["arm"] = "a0-repeat"
    exact = compare_a0_repeats(reference, repeat)
    assert exact["first_preclip_hash_divergence_step"] is None
    assert exact["first_delta_hash_divergence_step"] is None

    repeat["steps"][1]["stock_grad_preclip_sha256"]["model.3.weight"] = "NOISE"
    noisy = compare_a0_repeats(reference, repeat)
    assert noisy["first_preclip_hash_divergence_step"] == 2

    repeat = _run("a0", clip=1.0, stock_delta=0.1)
    repeat["arm"] = "a0-repeat"
    repeat["steps"][0] = _run("a0", clip=0.0, stock_delta=0.0, skipped=True)["steps"][0]
    with pytest.raises(ValueError, match="AMP skip-pattern mismatch"):
        compare_a0_repeats(reference, repeat)
