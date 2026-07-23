import hashlib
import json
import platform
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.audit_ebc_qp_aux_causality import (
    _clip_coefficient_from_norm,
    _json_safe,
    _validate_protocol,
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
    validate_audit_cli_args,
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
                "clip_total_norm": 3.0,
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

    config = {"enabled": True, "init_scale": 128.0}
    validate_controlled_amp_runtime(True, Scaler(True, 128.0), config)
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
    broken_controlled_amp["controlled_amp"] = {"enabled": True, "init_scale": 128.0}
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
    base_step = deepcopy(_run("a0", clip=1.0, stock_delta=0.1)["steps"][0])
    base_step.update(
        amp_scale_before=128.0,
        amp_scale_after=128.0,
        gradient_clipping_mode="contribution_separated" if is_auxiliary else "legacy_combined",
        aux_private_grad_total_norm=1.0 if is_auxiliary else 0.0,
        aux_private_clip_coefficient=1.0,
        routed_shallow_grad_total_norm=0.05 if arm == "h1" else 0.0,
        routed_shallow_grad_parameters=deepcopy(p2_signatures) if arm == "h1" else {},
        routed_shallow_grad_finite=True,
        routed_shallow_clip_coefficient=1.0,
        p2_entry_count=0,
        ordinary_query_count=300,
        stock_grad_preclip_parameters={
            "model.0.weight": {"l2": 0.6},
            "model.1.weight": {"l2": 0.8},
            "model.2.weight": {"l2": 2.0},
        },
        stock_delta_sha256={
            "model.0.weight": "SHALLOW-H1" if arm == "h1" else "SHALLOW-H0",
            "model.1.weight": "SHALLOW-H1" if arm == "h1" else "SHALLOW-H0",
            "model.2.weight": "DEEP",
        },
    )
    steps = []
    for index in range(1, 101):
        step = deepcopy(base_step)
        step.update(
            optimizer_step=index,
            optimizer_attempt=index,
            successful_update=True,
            successful_update_index=index,
            epoch=(index - 1) // 20,
            batch_fingerprints=[f"B{index}"],
            rng_before_forward=[f"R{index}"],
        )
        steps.append(step)
    return {
        "format_version": 1,
        "evidence": {
            "git_commit": "RUN-COMMIT",
            "sources": {"audit.py": "AUDIT-SHA", "trainer.py": "TRAINER-SHA"},
            "settings": {
                "epochs": 10,
                "fraction": 1.0,
                "imgsz": 640,
                "batch": 8,
                "workers": 8,
                "device": "0",
                "seed": 0,
                "optimizer": "auto",
                "amp": True,
                "model": "rtdetr-l.yaml" if arm == "a0" else "rtdetr-l-ebc-qp.yaml",
                "name": f"e0-{arm}",
            },
            "protocol_manifest_path": "/protocol.json",
            "protocol_manifest_sha256": "PROTOCOL-SHA",
            "protocol_signature": "PROTOCOL-SIGNATURE",
            "experiment_signature": "EXPERIMENT-SIGNATURE",
            "data_path": "/data.yaml",
            "data_sha256": "DATA-SHA",
            "python": "3.11.0",
            "platform": "Linux",
            "torch": "2.7.0",
            "cuda_runtime": "12.8",
            "ultralytics": "8.4.90",
            "cuda_available": True,
            "gpu": "GPU",
        },
        "arm": arm,
        "target_optimizer_steps": 100,
        "attempted_optimizer_steps": 100,
        "completed_successful_updates": 100,
        "common_initial_fingerprint": "COMMON",
        "initial_state_sha256": "INITIAL",
        "optimizer_common_manifest": {"model.0.weight": {"param_group": "muon"}},
        "controlled_amp": {
            "enabled": True,
            "init_scale": 128.0,
            "growth_interval": 2**31 - 1,
            "require_zero_skips": True,
        },
        "ebc_config": build_audit_ebc_config(arm).as_dict() if is_auxiliary else None,
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
        "steps": steps,
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


def test_tsgr_e0b_comparator_rejects_non_100_step_or_non_amp128_traces():
    short = _tsgr_run("a0")
    short["target_optimizer_steps"] = 32
    short["attempted_optimizer_steps"] = 32
    short["completed_successful_updates"] = 32
    short["steps"] = short["steps"][:32]
    result = compare_tsgr_audit_runs(short, _tsgr_run("h0"), _tsgr_run("h1"))
    assert result["passed"] is False
    assert any("exactly 100" in error for error in result["errors"])

    wrong_config = _tsgr_run("h0")
    wrong_config["controlled_amp"]["init_scale"] = 256.0
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), wrong_config, _tsgr_run("h1"))
    assert result["passed"] is False
    assert any("fixed AMP128" in error for error in result["errors"])

    drift = _tsgr_run("h1")
    drift["steps"][73]["amp_scale_after"] = 64.0
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), _tsgr_run("h0"), drift)
    assert result["passed"] is False
    assert any("optimizer attempt 74" in error and "AMP scale" in error for error in result["errors"])

    invalid_attempt = _tsgr_run("a0")
    invalid_attempt["steps"][4]["model_parameters_finite"] = False
    result = compare_tsgr_audit_runs(invalid_attempt, _tsgr_run("h0"), _tsgr_run("h1"))
    assert result["passed"] is False
    assert any("optimizer attempt 5" in error and "model_parameters_finite" in error for error in result["errors"])


def test_tsgr_e0b_comparator_rejects_identity_and_per_attempt_pairing_drift():
    wrong_commit = _tsgr_run("h1")
    wrong_commit["evidence"]["git_commit"] = "OTHER-COMMIT"
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), _tsgr_run("h0"), wrong_commit)
    assert result["passed"] is False
    assert any("evidence.git_commit" in error for error in result["errors"])

    wrong_batch = _tsgr_run("h0")
    wrong_batch["steps"][73]["batch_fingerprints"] = ["OTHER-BATCH"]
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), wrong_batch, _tsgr_run("h1"))
    assert result["passed"] is False
    assert any("batch sequence" in error and "74" in error for error in result["errors"])

    missing_rng = _tsgr_run("h1")
    missing_rng["steps"][73].pop("rng_before_forward")
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), _tsgr_run("h0"), missing_rng)
    assert result["passed"] is False
    assert any("random-state sequence" in error and "74" in error for error in result["errors"])

    wrong_optimizer = _tsgr_run("h1")
    wrong_optimizer["steps"][73]["optimizer_groups"][0]["lr"] = 0.25
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), _tsgr_run("h0"), wrong_optimizer)
    assert result["passed"] is False
    assert any("optimizer runtime-group sequence" in error and "74" in error for error in result["errors"])


def test_tsgr_e0b_comparator_rejects_config_nonfinite_signature_and_unverified_clipping():
    toxic_config = _tsgr_run("h0")
    toxic_config["ebc_config"]["learnable_fusion_gamma"] = True
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), toxic_config, _tsgr_run("h1"))
    assert result["passed"] is False
    assert any("frozen minimal P2-only config" in error for error in result["errors"])

    nonfinite = _tsgr_run("h0")
    nonfinite["p2_only_stock_grad_parameters"]["model.0.weight"]["l2"] = "NaN"
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), nonfinite, _tsgr_run("h1"))
    assert result["passed"] is False
    assert any("non-finite" in error for error in result["errors"])

    bad_clip = _tsgr_run("h1")
    bad_clip["steps"][73]["routed_shallow_clip_coefficient"] = 0.5
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), _tsgr_run("h0"), bad_clip)
    assert result["passed"] is False
    assert any("routed shallow clip coefficient" in error and "74" in error for error in result["errors"])

    missing_private = _tsgr_run("h0")
    missing_private["steps"][73]["aux_private_grad_total_norm"] = 0.0
    result = compare_tsgr_audit_runs(_tsgr_run("a0"), missing_private, _tsgr_run("h1"))
    assert result["passed"] is False
    assert any("no auxiliary-private gradient" in error and "74" in error for error in result["errors"])


def test_tsgr_e0b_a0_validates_each_independently_reduced_clip_norm():
    a0 = _tsgr_run("a0")
    for step in a0["steps"]:
        step["clip_total_norm"] = 3.000001
        step["clip_coefficient"] = min(1.0, 10.0 / (3.000001 + 1e-6))
    result = compare_tsgr_audit_runs(a0, _tsgr_run("h0"), _tsgr_run("h1"))
    assert result["passed"] is True

    a0["steps"][73]["clip_coefficient"] = 0.5
    result = compare_tsgr_audit_runs(a0, _tsgr_run("h0"), _tsgr_run("h1"))
    assert result["passed"] is False
    assert any("applied clip coefficient" in error and "74" in error for error in result["errors"])


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _json_sha256(payload):
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest().upper()


def test_audit_protocol_validation_recomputes_manifest_and_data_hashes(tmp_path, monkeypatch):
    from src.ebc_qp_config import SOURCE_SHA256
    import scripts.audit_ebc_qp_aux_causality as audit_script

    initial = tmp_path / "initial.pt"
    data = tmp_path / "data.yaml"
    protocol = tmp_path / "protocol.json"
    initial.write_bytes(b"initial")
    data.write_text("path: /dataset\n", encoding="utf-8")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": "8.4.90",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    monkeypatch.setattr(
        audit_script,
        "_runtime_environment_record",
        lambda: runtime_environment,
        raising=False,
    )
    manifest = {
        "format_version": 1,
        "seed": 0,
        "experiment_signature": "EXPERIMENT",
        "data": {"path": str(data.resolve()), "sha256": _sha256(data)},
        "initial_state": {"path": str(initial.resolve()), "sha256": _sha256(initial)},
        "source_sha256": SOURCE_SHA256,
        "git_commit": commit,
        "environment": runtime_environment,
    }
    manifest["signature"] = _json_sha256(manifest)
    protocol.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(initial_state=initial, protocol_manifest=protocol, data=data, seed=0)
    assert _validate_protocol(args)["signature"] == manifest["signature"]

    data.write_text("path: /changed\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="data hash mismatch"):
        _validate_protocol(args)

    data.write_text("path: /dataset\n", encoding="utf-8")
    changed = deepcopy(manifest)
    changed["source_sha256"] = {**SOURCE_SHA256, "head.py": "CHANGED"}
    protocol.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SystemExit, match="signature mismatch"):
        _validate_protocol(args)

    changed["signature"] = _json_sha256({key: value for key, value in changed.items() if key != "signature"})
    protocol.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SystemExit, match="source lock mismatch"):
        _validate_protocol(args)


def test_audit_cli_freezes_amp128_100_steps_for_a0_h0_h1(tmp_path):
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

    for arm in ("a0", "h0", "h1"):
        controlled = parser.parse_args(["--arm", arm, *common, "--controlled-amp-scale", "128"])
        assert resolved_audit_steps(controlled) == 100
        assert resolved_run_name(controlled) == f"e0-{arm}-seed0-controlled-amp128-100step"
        assert controlled_amp_config(controlled) == {
            "enabled": True,
            "init_scale": 128.0,
            "growth_interval": 2**31 - 1,
            "require_zero_skips": True,
        }

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--arm", "h1", *common, "--controlled-amp-scale", "128", "--controlled-amp-steps", "32"]
        )


def test_audit_cli_rejects_or_accepts_controlled_options_as_one_unit(tmp_path):
    parser = build_parser()
    common = [
        "--arm",
        "a0",
        "--initial-state",
        str(tmp_path / "initial.pt"),
        "--protocol-manifest",
        str(tmp_path / "protocol.json"),
        "--data",
        str(tmp_path / "data.yaml"),
        "--output",
        str(tmp_path / "audit.json"),
    ]

    validate_audit_cli_args(parser.parse_args(common))
    validate_audit_cli_args(parser.parse_args([*common, "--smoke"]))
    validate_audit_cli_args(parser.parse_args([*common, "--controlled-amp-scale", "128"]))

    with pytest.raises(SystemExit, match="requires --controlled-amp-scale"):
        validate_audit_cli_args(parser.parse_args([*common, "--controlled-amp-steps", "100"]))
    with pytest.raises(SystemExit, match="mutually exclusive"):
        validate_audit_cli_args(parser.parse_args([*common, "--smoke", "--controlled-amp-scale", "128"]))


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
