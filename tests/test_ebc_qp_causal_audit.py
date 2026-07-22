import pytest
import torch
from torch import nn

from scripts.audit_ebc_qp_aux_causality import (
    batch_fingerprint,
    build_audit_settings,
    build_parser,
    optimizer_common_manifest,
    resolved_audit_steps,
    resolved_run_name,
    tensor_structure_fingerprint,
)
from src.ebc_qp_causal_audit import (
    capture_grouped_gradients,
    capture_grouped_parameter_deltas,
    capture_parameter_signatures,
    capture_tensor_sha256,
    classify_stock_parameter,
    clone_named_parameters,
    compare_a0_repeats,
    compare_audit_runs,
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
                "clip_coefficient": clip,
                "stock_only_clip_coefficient": 1.0,
                "stock_grad_total_norm": 3.0,
                "aux_private_grad_total_norm": 20.0 if arm == "aux-audit" else 0.0,
                "clip_norm_partition_relative_error": 0.0,
                "amp_step_skipped": skipped,
                "amp_scale_before": 65536.0,
                "amp_scale_after": 32768.0 if skipped else 65536.0,
                "stock_grad_preclip": {"backbone_c2": {"l2": 3.0, "max_abs": 2.0}},
                "stock_grad_preclip_parameters": {"model.3.weight": {"l2": 3.0, "sum": 1.0, "max_abs": 2.0}},
                "stock_grad_preclip_sha256": {"model.3.weight": "GRAD"},
                "stock_delta": {"backbone_c2": {"l2": stock_delta, "max_abs": stock_delta}},
                "stock_delta_parameters": {
                    "model.3.weight": {"l2": stock_delta, "sum": stock_delta, "max_abs": stock_delta}
                },
                "stock_delta_sha256": {"model.3.weight": f"DELTA-{stock_delta}"},
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


def test_compare_audit_runs_prioritizes_direct_gradient_or_amp_coupling():
    direct = _run("aux-audit", clip=1.0, stock_delta=0.1)
    direct["p2_only_stock_grad_l2"] = 0.25
    assert compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), direct)["classification"] == "DIRECT_GRADIENT_PATH"

    amp = _run("aux-audit", clip=1.0, stock_delta=0.0, skipped=True)
    assert compare_audit_runs(_run("a0", clip=1.0, stock_delta=0.1), amp)["classification"] == "AMP_STEP_COUPLING"


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

    assert a0.steps == auxiliary.steps == 100
    assert resolved_audit_steps(a0) == 100
    assert build_audit_settings(a0) == build_audit_settings(auxiliary)
    assert build_audit_settings(a0)["amp"] is True
    assert build_audit_settings(a0)["batch"] == 8
    assert build_audit_settings(a0)["optimizer"] == "auto"

    smoke = parser.parse_args(["--arm", "a0", *common, "--smoke"])
    assert resolved_audit_steps(smoke) == 1
    assert resolved_run_name(smoke) == "e0-a0-seed0-smoke"

    repeat = parser.parse_args(["--arm", "a0-repeat", *common])
    assert build_audit_settings(repeat) == build_audit_settings(a0)
    assert resolved_run_name(repeat) == "e0-a0-repeat-seed0"


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
