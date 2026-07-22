from __future__ import annotations

import hashlib
from math import isclose, isfinite
from statistics import median
from typing import Mapping

import torch


STOCK_GROUPS = (
    "backbone_pre_c2",
    "backbone_c2",
    "backbone_post_c2",
    "p3_neck",
    "encoder",
    "stock_score_head",
    "stock_box_head",
    "decoder",
    "stock_other",
)

_AUXILIARY_PARTS = ("p2_adapter", "p2_bbox_head", "p2_fusion_gamma", "p2_quality_head")


def classify_stock_parameter(name: str) -> str:
    """Map a common RT-DETR parameter to the E0 causal-audit boundary it belongs to."""
    if any(part in name.split(".") for part in _AUXILIARY_PARTS):
        return "auxiliary"
    parts = name.split(".")
    layer = _model_layer_index(parts)
    if layer is not None:
        if layer <= 2:
            return "backbone_pre_c2"
        if layer == 3:
            return "backbone_c2"
        if layer <= 9:
            return "backbone_post_c2"
        if layer <= 27:
            return "p3_neck"
        if layer == 28:
            suffix = ".".join(parts[2:])
            if suffix.startswith(("input_proj.", "enc_output.")):
                return "encoder"
            if suffix.startswith("enc_score_head."):
                return "stock_score_head"
            if suffix.startswith("enc_bbox_head."):
                return "stock_box_head"
            if suffix.startswith(
                (
                    "decoder.",
                    "dec_score_head.",
                    "dec_bbox_head.",
                    "query_pos_head.",
                    "tgt_embed.",
                    "denoising_class_embed.",
                )
            ):
                return "decoder"
    return "stock_other"


def classify_tsgr_gradient_boundary(name: str) -> str:
    if any(part in name.split(".") for part in _AUXILIARY_PARTS):
        return "auxiliary_private"
    layer = _model_layer_index(name.split("."))
    return "routed_shallow" if layer in {0, 1} else "forbidden_common"


def clone_named_parameters(parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {name: value.detach().float().clone() for name, value in parameters.items()}


def capture_grouped_gradients(parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, dict[str, float]]:
    tensors = {name: parameter.grad for name, parameter in parameters.items() if parameter.grad is not None}
    return _capture_grouped_tensors(tensors)


def capture_grouped_tensor_mapping(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    return _capture_grouped_tensors(tensors)


def capture_grouped_parameter_deltas(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.nn.Parameter],
) -> dict[str, dict[str, float]]:
    if set(before) != set(after):
        raise ValueError("parameter snapshot names changed")
    tensors = {name: after[name].detach().float() - before[name] for name in before}
    return _capture_grouped_tensors(tensors)


def capture_grouped_values(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    return _capture_grouped_tensors({name: value.detach().float() for name, value in tensors.items()})


def capture_parameter_signatures(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    """Capture signed per-parameter evidence with a single device synchronization."""
    names = sorted(tensors)
    if not names:
        return {}
    rows = []
    counts = []
    for name in names:
        flat = tensors[name].detach().float().reshape(-1)
        counts.append(flat.numel())
        if flat.numel():
            sample_count = min(flat.numel(), 64)
            indices = torch.linspace(0, flat.numel() - 1, steps=sample_count, device=flat.device).long().unique()
            sample = flat[indices]
            weights = torch.arange(1, sample.numel() + 1, device=flat.device, dtype=flat.dtype)
            projection = (sample * weights).sum()
            rows.append(torch.stack((flat.square().sum().sqrt(), flat.sum(), flat.abs().max(), projection)))
        else:
            rows.append(torch.zeros(4, device=flat.device, dtype=flat.dtype))
    values = torch.stack(rows).cpu().tolist()
    return {
        name: {
            "l2": float(row[0]),
            "sum": float(row[1]),
            "max_abs": float(row[2]),
            "sample_projection": float(row[3]),
            "count": counts[index],
        }
        for index, (name, row) in enumerate(zip(names, values))
    }


def capture_parameter_delta_signatures(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.nn.Parameter],
) -> dict[str, dict[str, float | int]]:
    if set(before) != set(after):
        raise ValueError("parameter snapshot names changed")
    return capture_parameter_signatures(
        {name: after[name].detach().float() - before[name] for name in sorted(before)}
    )


def capture_tensor_sha256(tensors: Mapping[str, torch.Tensor]) -> dict[str, str]:
    result = {}
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[name] = digest.hexdigest().upper()
    return result


def capture_parameter_delta_sha256(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.nn.Parameter],
) -> dict[str, str]:
    if set(before) != set(after):
        raise ValueError("parameter snapshot names changed")
    return capture_tensor_sha256(
        {name: after[name].detach().float() - before[name] for name in sorted(before)}
    )


def validate_audit_attempt(record: dict) -> list[str]:
    """Return invariant violations for one optimizer attempt without discarding its evidence."""
    errors: list[str] = []
    skipped = bool(record.get("amp_step_skipped"))
    scale_before = _finite_float(record.get("amp_scale_before"))
    scale_after = _finite_float(record.get("amp_scale_after"))
    scale_decreased = scale_before is not None and scale_after is not None and scale_after < scale_before

    if skipped != scale_decreased:
        errors.append("amp_step_skipped must exactly match a decreasing AMP scale")

    required_finite = (
        "loss_finite",
        "loss_items_finite",
        "model_parameters_finite",
        "stock_bn_finite",
        "stock_ema_finite",
        "optimizer_state_finite",
        "stock_delta_finite",
    )
    for field in required_finite:
        if not bool(record.get(field)):
            errors.append(f"{field} must be true")

    gradients_finite = all(
        bool(record.get(field))
        for field in (
            "stock_grad_preclip_finite",
            "aux_private_grad_finite",
            "all_grad_preclip_finite",
            "stock_grad_postclip_finite",
            "clip_total_norm_finite",
        )
    )
    if not gradients_finite and not (skipped and scale_decreased):
        errors.append("non-finite gradients are only valid on a confirmed skipped AMP attempt")
    if skipped and not bool(record.get("stock_delta_zero")):
        errors.append("skipped AMP attempt changed stock parameters")

    partition_error = _finite_float(record.get("clip_norm_partition_relative_error"))
    if skipped:
        if record.get("clip_coefficient") is not None:
            errors.append("clip_coefficient must be null on a skipped AMP attempt")
        if record.get("stock_only_clip_coefficient") is not None:
            errors.append("stock_only_clip_coefficient must be null on a skipped AMP attempt")
        if partition_error is not None:
            errors.append("clip norm partition must be not-applicable on a skipped AMP attempt")
    else:
        if partition_error is None:
            errors.append("non-finite gradient norm partition on successful update")
        elif partition_error > 1e-4:
            errors.append("gradient norm partition mismatch")
        if record.get("clip_coefficient") is None or record.get("stock_only_clip_coefficient") is None:
            errors.append("clip coefficients missing on successful update")
    return errors


def compare_audit_runs(control: dict, auxiliary: dict, *, tolerance: float = 1e-8) -> dict:
    """Validate two E0 traces and identify the earliest supported stock-coupling mechanism."""
    if control.get("arm") != "a0" or auxiliary.get("arm") != "aux-audit":
        raise ValueError("audit arms must be a0 and aux-audit")
    if control.get("target_optimizer_steps") != auxiliary.get("target_optimizer_steps"):
        raise ValueError("target optimizer-step count mismatch")
    if control.get("common_initial_fingerprint") != auxiliary.get("common_initial_fingerprint"):
        raise ValueError("common initial-state fingerprint mismatch")
    if control.get("optimizer_common_manifest") != auxiliary.get("optimizer_common_manifest"):
        raise ValueError("optimizer common-parameter manifest mismatch")
    if control.get("controlled_amp") != auxiliary.get("controlled_amp"):
        raise ValueError("controlled AMP configuration mismatch")
    probe_fields = (
        "batch_fingerprint",
        "rng_before_forward",
        "stock_topk_fingerprint",
        "decoder_output_fingerprint",
        "stock_output_fingerprint",
    )
    control_probe = control.get("initial_probe", {})
    auxiliary_probe = auxiliary.get("initial_probe", {})
    if any(control_probe.get(field) != auxiliary_probe.get(field) for field in probe_fields):
        raise ValueError("initial stock forward/query mismatch")

    control_steps = control.get("steps", [])
    auxiliary_steps = auxiliary.get("steps", [])
    target = int(control["target_optimizer_steps"])
    control_successful = int(control.get("completed_successful_updates", len(control_steps)))
    auxiliary_successful = int(auxiliary.get("completed_successful_updates", len(auxiliary_steps)))
    if control_successful != target or auxiliary_successful != target:
        raise ValueError("audit trace does not contain the target successful-update count")
    optimizer_attempt_count_mismatch = len(control_steps) != len(auxiliary_steps)
    for label, records in (("control", control_steps), ("auxiliary", auxiliary_steps)):
        for expected_step, record in enumerate(records, start=1):
            if record.get("optimizer_step") != expected_step:
                raise ValueError(f"{label} optimizer-step sequence mismatch")
            violations = validate_audit_attempt(record)
            if violations:
                raise ValueError(f"{label} optimizer step {expected_step}: {'; '.join(violations)}")

    first_divergence = None
    clip_divergence = None
    amp_divergence = None
    bn_divergence = None
    ema_divergence = None
    preclip_divergence = None
    amp_scale_divergence = None
    optimizer_state_divergence = None
    counterfactual_clip_effect = None
    for expected_step, (left, right) in enumerate(zip(control_steps, auxiliary_steps), start=1):
        if left.get("batch_fingerprints") != right.get("batch_fingerprints"):
            raise ValueError(f"batch sequence mismatch at optimizer step {expected_step}")
        if left.get("rng_before_forward") != right.get("rng_before_forward"):
            raise ValueError(f"random-state sequence mismatch at optimizer step {expected_step}")
        if left.get("optimizer_groups") != right.get("optimizer_groups"):
            raise ValueError(f"optimizer runtime-group mismatch at optimizer step {expected_step}")

        left_skipped = bool(left.get("amp_step_skipped"))
        right_skipped = bool(right.get("amp_step_skipped"))
        if left_skipped != right_skipped:
            amp_divergence = expected_step
            amp_scale_divergence = expected_step
            first_divergence = expected_step
            break

        if not _nested_close(
            {"before": left.get("amp_scale_before"), "after": left.get("amp_scale_after")},
            {"before": right.get("amp_scale_before"), "after": right.get("amp_scale_after")},
            tolerance,
        ):
            amp_scale_divergence = amp_scale_divergence or expected_step
        if not _nested_close(_detail(left, "stock_bn"), _detail(right, "stock_bn"), tolerance):
            bn_divergence = bn_divergence or expected_step
        if not _nested_close(_detail(left, "stock_ema"), _detail(right, "stock_ema"), tolerance):
            ema_divergence = ema_divergence or expected_step
        if not _nested_close(_detail(left, "optimizer_state"), _detail(right, "optimizer_state"), tolerance):
            optimizer_state_divergence = optimizer_state_divergence or expected_step
        if not _nested_close(_detail(left, "stock_delta"), _detail(right, "stock_delta"), tolerance):
            first_divergence = first_divergence or expected_step

        if left_skipped:
            continue

        if not _nested_close(_detail(left, "stock_grad_preclip"), _detail(right, "stock_grad_preclip"), tolerance):
            preclip_divergence = preclip_divergence or expected_step
        if not isclose(
            float(left.get("clip_coefficient", 1.0)),
            float(right.get("clip_coefficient", 1.0)),
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            clip_divergence = clip_divergence or expected_step
        if not isclose(
            float(right.get("clip_coefficient", 1.0)),
            float(right.get("stock_only_clip_coefficient", right.get("clip_coefficient", 1.0))),
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            counterfactual_clip_effect = counterfactual_clip_effect or expected_step
    if optimizer_attempt_count_mismatch and amp_divergence is None:
        raise ValueError("optimizer-attempt count mismatch without a preceding AMP skip divergence")

    p2_only_stock_grad = float(auxiliary.get("p2_only_stock_grad_l2", 0.0))
    p2_only_aux_private_grad = float(auxiliary.get("p2_only_aux_private_grad_l2", 0.0))
    if p2_only_aux_private_grad <= tolerance:
        raise ValueError("AUX probe produced no auxiliary-private gradient")
    mechanisms = []
    if p2_only_stock_grad > tolerance:
        mechanisms.append("direct_gradient")
    if amp_divergence is not None:
        mechanisms.append("amp_step")
    if optimizer_attempt_count_mismatch:
        mechanisms.append("amp_attempt_count")
    if amp_scale_divergence is not None:
        mechanisms.append("amp_scale")
    if clip_divergence is not None:
        mechanisms.append("global_clip")
    if counterfactual_clip_effect is not None:
        mechanisms.append("global_clip_counterfactual")
    if bn_divergence is not None:
        mechanisms.append("bn_state")
    if preclip_divergence is not None:
        mechanisms.append("preclip_gradient")
    if optimizer_state_divergence is not None:
        mechanisms.append("optimizer_state")
    if ema_divergence is not None:
        mechanisms.append("ema_state")

    if p2_only_stock_grad > tolerance:
        classification = "DIRECT_GRADIENT_PATH"
    elif amp_divergence is not None:
        classification = "AMP_STEP_COUPLING"
    elif amp_scale_divergence is not None and (
        preclip_divergence is None or amp_scale_divergence < preclip_divergence
    ):
        classification = "AMP_SCALE_COUPLING"
    elif counterfactual_clip_effect is not None:
        classification = "GLOBAL_CLIP_COUPLING"
    elif clip_divergence is not None and first_divergence is not None and (
        preclip_divergence is None or clip_divergence < preclip_divergence
    ):
        classification = "GLOBAL_CLIP_COUPLING"
    elif bn_divergence is not None and first_divergence is not None and (
        preclip_divergence is None or bn_divergence <= preclip_divergence
    ):
        classification = "BN_STATE_COUPLING"
    elif preclip_divergence is not None:
        classification = "UNEXPECTED_STOCK_GRADIENT_DIVERGENCE"
    elif first_divergence is None:
        classification = "NO_STOCK_DIVERGENCE"
    else:
        classification = "UNRESOLVED_STOCK_UPDATE_DIVERGENCE"

    return {
        "pairing_valid": True,
        "classification": classification,
        "mechanisms_detected": mechanisms,
        "first_stock_divergence_step": first_divergence,
        "first_preclip_gradient_divergence_step": preclip_divergence,
        "first_clip_divergence_step": clip_divergence,
        "first_counterfactual_clip_effect_step": counterfactual_clip_effect,
        "first_amp_divergence_step": amp_divergence,
        "first_amp_scale_divergence_step": amp_scale_divergence,
        "first_bn_divergence_step": bn_divergence,
        "first_ema_divergence_step": ema_divergence,
        "first_optimizer_state_divergence_step": optimizer_state_divergence,
        "optimizer_attempt_count_mismatch": optimizer_attempt_count_mismatch,
        "p2_only_stock_grad_l2": p2_only_stock_grad,
        "p2_only_aux_private_grad_l2": p2_only_aux_private_grad,
    }


def compare_tsgr_audit_runs(a0: dict, h0: dict, h1: dict, *, tolerance: float = 1e-12) -> dict:
    """Apply the preregistered E0b gate to contribution-separated A0/H0/H1 traces."""
    errors: list[str] = []
    if (a0.get("arm"), h0.get("arm"), h1.get("arm")) != ("a0", "h0", "h1"):
        errors.append("E0b arms must be a0, h0, and h1")
    targets = {trace.get("target_optimizer_steps") for trace in (a0, h0, h1)}
    if len(targets) != 1:
        errors.append("E0b target optimizer-step counts differ")
    for trace in (a0, h0, h1):
        if trace.get("completed_successful_updates") != trace.get("target_optimizer_steps"):
            errors.append(f"{trace.get('arm')} did not complete its successful-update target")
        if any(step.get("amp_step_skipped") for step in trace.get("steps", [])):
            errors.append(f"{trace.get('arm')} contains an AMP skip")
        if not trace.get("controlled_amp", {}).get("enabled"):
            errors.append(f"{trace.get('arm')} is not a controlled-AMP trace")

    paired_fields = ("common_initial_fingerprint", "initial_state_sha256", "optimizer_common_manifest")
    for field in paired_fields:
        if not (a0.get(field) == h0.get(field) == h1.get(field)):
            errors.append(f"E0b pairing mismatch: {field}")
    probe_fields = (
        "batch_fingerprint",
        "rng_before_forward",
        "stock_topk_fingerprint",
        "decoder_output_fingerprint",
        "stock_output_fingerprint",
    )
    for field in probe_fields:
        values = [trace.get("initial_probe", {}).get(field) for trace in (a0, h0, h1)]
        if values[0] is None or len(set(values)) != 1:
            errors.append(f"E0b initial probe mismatch: {field}")
    if not _nested_close(
        h0.get("initial_probe", {}).get("p2_loss"),
        h1.get("initial_probe", {}).get("p2_loss"),
        tolerance,
    ):
        errors.append("H0/H1 initial P2 loss differs")

    h0_signatures = h0.get("p2_only_stock_grad_parameters", {})
    h1_signatures = h1.get("p2_only_stock_grad_parameters", {})
    h0_nonzero = [name for name, record in h0_signatures.items() if float(record.get("max_abs", 0.0)) > tolerance]
    if h0_nonzero:
        errors.append(f"H0 has common P2-only gradients: {h0_nonzero[:5]}")
    h1_allowed = [
        name
        for name, record in h1_signatures.items()
        if float(record.get("max_abs", 0.0)) > tolerance
        and classify_tsgr_gradient_boundary(name) == "routed_shallow"
    ]
    h1_forbidden = [
        name
        for name, record in h1_signatures.items()
        if float(record.get("max_abs", 0.0)) > tolerance
        and classify_tsgr_gradient_boundary(name) != "routed_shallow"
    ]
    allowed_layers = {_model_layer_index(name.split(".")) for name in h1_allowed}
    if not {0, 1}.issubset(allowed_layers):
        errors.append("H1 did not produce finite nonzero P2-only gradients in both model.0 and model.1")
    if h1_forbidden:
        errors.append(f"H1 P2-only gradient escaped model.0/1: {h1_forbidden[:5]}")
    h0_aux = float(h0.get("p2_only_aux_private_grad_l2", 0.0))
    h1_aux = float(h1.get("p2_only_aux_private_grad_l2", 0.0))
    if h0_aux <= tolerance or h1_aux <= tolerance:
        errors.append("H0/H1 auxiliary-private P2-only gradient is missing")
    elif not isclose(h0_aux, h1_aux, rel_tol=1e-6, abs_tol=tolerance):
        errors.append("H0/H1 auxiliary-private P2-only gradients differ")

    for trace in (h0, h1):
        config = trace.get("ebc_config") or {}
        expected_eta = 0.0 if trace.get("arm") == "h0" else 0.1
        if config.get("p2_c2_grad_scale") != expected_eta:
            errors.append(f"{trace.get('arm')} eta mismatch")
        if config.get("lambda_p2") != 0.1 or not config.get("contribution_separated_aux_gradients"):
            errors.append(f"{trace.get('arm')} is not the frozen contribution-separated TSGR config")
        if config.get("query_injection_enabled") or config.get("lambda_ebc") != 0.0:
            errors.append(f"{trace.get('arm')} enables a forbidden stock-coupling feature")
        if trace.get("initial_probe", {}).get("p2_entry_count") != 0:
            errors.append(f"{trace.get('arm')} initial probe injected P2 queries")
        if trace.get("initial_probe", {}).get("ordinary_query_count") != 300:
            errors.append(f"{trace.get('arm')} initial query count is not 300")
        for step in trace.get("steps", []):
            if step.get("gradient_clipping_mode") != "contribution_separated":
                errors.append(f"{trace.get('arm')} did not use contribution-separated clipping")
                break
            if step.get("p2_entry_count") != 0 or step.get("ordinary_query_count") != 300:
                errors.append(f"{trace.get('arm')} query integrity failed")
                break
            if not _nested_close(
                step.get("clip_coefficient"), step.get("stock_only_clip_coefficient"), tolerance
            ):
                errors.append(f"{trace.get('arm')} auxiliary gradients changed the stock clip coefficient")
                break

    h1_first = (h1.get("steps") or [{}])[0]
    shallow_p2_norm = _signature_subset_l2(
        h1_signatures,
        lambda name: classify_tsgr_gradient_boundary(name) == "routed_shallow",
    )
    initial_shallow_stock_norm = _signature_subset_l2(
        h1_first.get("stock_grad_preclip_parameters", {}),
        lambda name: classify_tsgr_gradient_boundary(name) == "routed_shallow",
    )
    initial_probe_preclip_ratio = shallow_p2_norm / (initial_shallow_stock_norm + 1e-12)
    preclip_ratios: list[float] = []
    applied_ratios: list[float] = []
    for step in h1.get("steps", []):
        stock_shallow_norm = _signature_subset_l2(
            step.get("stock_grad_preclip_parameters", {}),
            lambda name: classify_tsgr_gradient_boundary(name) == "routed_shallow",
        )
        route_norm = float(step.get("routed_shallow_grad_total_norm", 0.0))
        stock_coefficient = float(step.get("stock_only_clip_coefficient", 0.0))
        route_coefficient = float(step.get("routed_shallow_clip_coefficient", 0.0))
        preclip_ratios.append(route_norm / (stock_shallow_norm + 1e-12))
        applied_ratios.append(
            route_norm * route_coefficient / (stock_shallow_norm * stock_coefficient + 1e-12)
        )
    applied_ratio_median = median(applied_ratios) if applied_ratios else 0.0
    preclip_ratio_median = median(preclip_ratios) if preclip_ratios else 0.0
    if not 0.01 <= applied_ratio_median <= 0.25:
        errors.append(
            f"H1 applied shallow gradient ratio median={applied_ratio_median:.6g} is outside [0.01, 0.25]"
        )
    if any(float(step.get("routed_shallow_grad_total_norm", 0.0)) > tolerance for step in h0.get("steps", [])):
        errors.append("H0 produced a routed shallow gradient during training")

    if a0.get("steps") and h0.get("steps"):
        a0_delta = a0["steps"][0].get("stock_delta_sha256", {})
        h0_delta = h0["steps"][0].get("stock_delta_sha256", {})
        divergent = [name for name in sorted(set(a0_delta).intersection(h0_delta)) if a0_delta[name] != h0_delta[name]]
        if divergent:
            errors.append(f"H0 first update differs from A0: {divergent[:5]}")
    if h0.get("steps") and h1.get("steps"):
        h0_delta = h0["steps"][0].get("stock_delta_sha256", {})
        h1_delta = h1["steps"][0].get("stock_delta_sha256", {})
        deep_names = {
            name
            for name in set(h0_delta).intersection(h1_delta)
            if classify_tsgr_gradient_boundary(name) == "forbidden_common"
        }
        divergent = [name for name in sorted(deep_names) if h0_delta[name] != h1_delta[name]]
        if divergent:
            errors.append(f"H1 first update changed forbidden deep parameters: {divergent[:5]}")

    return {
        "passed": not errors,
        "classification": "TSGR_E0B_PASS" if not errors else "TSGR_E0B_FAIL",
        "errors": errors,
        "h1_initial_probe_preclip_ratio": initial_probe_preclip_ratio,
        "h1_shallow_preclip_ratio_median": preclip_ratio_median,
        "h1_shallow_applied_ratio_median": applied_ratio_median,
        "h1_shallow_applied_ratio_min": min(applied_ratios) if applied_ratios else 0.0,
        "h1_shallow_applied_ratio_max": max(applied_ratios) if applied_ratios else 0.0,
        "h1_routed_nonzero_parameter_count": len(h1_allowed),
        "h1_forbidden_nonzero_parameter_count": len(h1_forbidden),
    }


def compare_a0_repeats(reference: dict, repeat: dict) -> dict:
    if reference.get("arm") != "a0" or repeat.get("arm") != "a0-repeat":
        raise ValueError("repeatability arms must be a0 and a0-repeat")
    for field, label in (
        ("target_optimizer_steps", "target optimizer-step count"),
        ("completed_successful_updates", "successful-update count"),
        ("common_initial_fingerprint", "common initial-state fingerprint"),
        ("optimizer_common_manifest", "optimizer common-parameter manifest"),
        ("controlled_amp", "controlled AMP configuration"),
        ("initial_probe", "initial forward/query probe"),
    ):
        if reference.get(field) != repeat.get(field):
            raise ValueError(f"A0 repeat {label} mismatch")
    left_steps = reference.get("steps", [])
    right_steps = repeat.get("steps", [])
    if len(left_steps) != len(right_steps):
        raise ValueError("A0 repeat optimizer-attempt count mismatch")

    first_preclip = None
    first_delta = None
    first_bn = None
    for index, (left, right) in enumerate(zip(left_steps, right_steps), start=1):
        for label, record in (("reference", left), ("repeat", right)):
            violations = validate_audit_attempt(record)
            if violations:
                raise ValueError(f"A0 {label} optimizer step {index}: {'; '.join(violations)}")
        if left.get("batch_fingerprints") != right.get("batch_fingerprints"):
            raise ValueError(f"A0 repeat batch sequence mismatch at optimizer step {index}")
        if left.get("rng_before_forward") != right.get("rng_before_forward"):
            raise ValueError(f"A0 repeat random-state sequence mismatch at optimizer step {index}")
        if left.get("optimizer_groups") != right.get("optimizer_groups"):
            raise ValueError(f"A0 repeat optimizer runtime-group mismatch at optimizer step {index}")
        if left.get("amp_step_skipped") != right.get("amp_step_skipped"):
            raise ValueError(f"A0 repeat AMP skip-pattern mismatch at optimizer step {index}")
        if (
            left.get("amp_scale_before") != right.get("amp_scale_before")
            or left.get("amp_scale_after") != right.get("amp_scale_after")
        ):
            raise ValueError(f"A0 repeat AMP scale trajectory mismatch at optimizer step {index}")
        if (
            not left.get("amp_step_skipped")
            and left.get("stock_grad_preclip_sha256") != right.get("stock_grad_preclip_sha256")
        ):
            first_preclip = first_preclip or index
        if left.get("stock_delta_sha256") != right.get("stock_delta_sha256"):
            first_delta = first_delta or index
        if left.get("stock_bn_sha256") != right.get("stock_bn_sha256"):
            first_bn = first_bn or index
    return {
        "pairing_valid": True,
        "first_preclip_hash_divergence_step": first_preclip,
        "first_delta_hash_divergence_step": first_delta,
        "first_bn_hash_divergence_step": first_bn,
        "bitwise_repeatable": first_preclip is None and first_delta is None and first_bn is None,
    }


def _model_layer_index(parts: list[str]) -> int | None:
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def _capture_grouped_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    squared = {group: None for group in STOCK_GROUPS}
    maxima = {group: None for group in STOCK_GROUPS}
    counts = {group: 0 for group in STOCK_GROUPS}
    for name, value in tensors.items():
        group = classify_stock_parameter(name)
        if group == "auxiliary":
            continue
        tensor = value.detach().float()
        component = tensor.square().sum()
        squared[group] = component if squared[group] is None else squared[group] + component
        if tensor.numel():
            maximum = tensor.abs().max()
            maxima[group] = maximum if maxima[group] is None else torch.maximum(maxima[group], maximum)
        counts[group] += tensor.numel()
    return {
        group: {
            "l2": 0.0 if squared[group] is None else float(squared[group].sqrt().item()),
            "max_abs": 0.0 if maxima[group] is None else float(maxima[group].item()),
            "count": counts[group],
        }
        for group in STOCK_GROUPS
    }


def _signature_subset_l2(signatures: Mapping[str, Mapping[str, object]], predicate) -> float:
    return sum(float(record.get("l2", 0.0)) ** 2 for name, record in signatures.items() if predicate(name)) ** 0.5


def _nested_close(left: object, right: object, tolerance: float) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_nested_close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _detail(step: dict, field: str) -> object:
    return step.get(f"{field}_sha256", step.get(f"{field}_parameters", step.get(field, {})))
