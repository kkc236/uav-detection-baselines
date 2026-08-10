"""Frozen authority and scientific gates for FDR versus FDR+RA-GLGM."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.fdr_protocol import FDR_PROTOCOL, canonical_json_bytes, public_state_sha256


RA_VARIANTS = ("baseline", "ra_glgm")
RA_STAGES = ("smoke", "screen", "formal", "explore50")
BASELINE_PARAMETERS = 33_156_614
MAX_PARAMETER_INCREASE_RATIO = 0.10
MAX_PEAK_VRAM_MIB = 22 * 1024
SCALE_PRIOR_RELATIVE_PATH = Path("configs") / "ra-glgm-v12-scale-prior.json"

RA_EXPERIMENT_PROTOCOL: dict[str, Any] = {
    "design": "ra-glgm-on-fdr-v1.2-continuous-scale-modulation",
    "baseline": "Ultralytics RT-DETR-L + FDR",
    "method": "Ultralytics RT-DETR-L + FDR + RA-GLGM-v1.2(P3-only continuous scale modulation)",
    "seed": 0,
    "device": "0",
    "pairing": {
        "single_physical_gpu": True,
        "sequential_arms": True,
        "ddp": False,
        "scratch": True,
        "shared_public_initialization": "byte-identical",
        "private_seed": 20_000,
    },
    "dataset": {
        **FDR_PROTOCOL["dataset"],
        "screen_train_images": 6471,
        "ignore_sidecar": {
            "files": {"train": 6471, "val": 548},
            "boxes": {"train": 10_343, "val": 1_410},
            "raw_score_zero_rows": {"train": 10_345, "val": 1_410},
            "invalid_zero_area_rows_excluded": {"train": 2, "val": 0},
            "source_rule": (
                "VisDrone source confidence/score field equals zero; "
                "non-positive width or height rows are excluded before sidecar materialization"
            ),
        },
        "selection_set": {
            "source": "VisDrone train images excluding the frozen Screen647 subset",
            "images": 548,
            "algorithm": "ascending salted SHA-256 rank of dataset-relative paths",
            "salt": "ra-glgm-v1.1-selection-v1\\0",
            "role": "Smoke2 and optional Explore50 diagnostic validation only",
            "official_val_used": False,
            "frozen_in_manifest": ["path_list", "path_list_sha256", "relative_subset_sha256"],
        },
        "screen30": {
            "train": "all 6471 authoritative VisDrone train images",
            "validation": "official VisDrone val (548 images, 38,759 objects)",
            "official_val_used_for_exploratory_gate": True,
            "confirmatory_status": "not pristine after Screen30 selection; Formal100 is fresh paired evidence",
        },
    },
    "training": {
        **FDR_PROTOCOL["training"],
        "smoke_epochs": 2,
        "screen_schedule_epochs": 50,
        "screen_cutoff_epoch": 30,
        "explore50_schedule_epochs": 50,
        "explore50_cutoff_epoch": 50,
        "formal_schedule_epochs": 100,
        "save_period": 1,
    },
    "augmentation": FDR_PROTOCOL["augmentation"],
    "module": {
        "private_parameters": 813_018,
        "input": {
            "source": "FDR decoder P3 only",
            "shape": "[B,256,H,W]",
            "private_branch_input": "x.detach()",
        },
        "hidden_channels": 192,
        "reduction": {
            "operator": "1x1 Conv-BN-SiLU",
            "channels": "256->192",
            "bias": False,
        },
        "local_expert": {
            "operators": [
                "3x3 Conv-BN-SiLU",
                "3x3 Conv-BN-SiLU",
            ],
            "bias": False,
            "residual_source": "reduced",
        },
        "global_expert": {
            "operators": [
                "depthwise 7x7 Conv-BN-SiLU",
                "depthwise dilated 3x3 Conv-BN-SiLU",
            ],
            "dilated_kernel_dilation": 3,
            "pool_projection": {
                "operator": "1x1 Conv",
                "channels": "192->192",
                "bias": True,
                "batch_norm": False,
            },
        },
        "router": {
            "operator": "1x1 Conv",
            "channels": "192->16",
            "groups": 8,
            "bias": True,
            "initialization": "zeros",
            "input": "shared reduced feature",
            "competition": "per-position grouped two-expert softmax",
        },
        "support": {
            "operator": "1x1 Conv",
            "channels": "192->1",
            "bias": True,
            "activation": "sigmoid",
        },
        "scale_gate": {
            "operator": "1x1 Conv",
            "channels": "192->1",
            "condition": (
                "continuous percentile of log(current post-augmentation 640-canvas box area), "
                "calibrated by the frozen natural-image letterbox prior"
            ),
            "activation": "per-position sigmoid",
            "initialization": "zero weight and zero bias",
            "initial_value": 0.5,
            "modulation": "zero-initialized per-router-group antisymmetric local/global logit bias",
            "groups": 8,
            "bounded_bias": "tanh(scale_expert_slopes[g])*(2*s-1)",
            "inference_inputs": ["shared reduced feature"],
            "forbidden_inference_inputs": ["ground_truth", "IoU", "Hungarian_assignment"],
        },
        "scale_prior": {
            "population": "343,204 valid unaugmented VisDrone train instances",
            "area": "box area after centered aspect-preserving 640x640 source-image letterbox",
            "role": (
                "fixed natural-image reference calibration; training targets use the current "
                "post-augmentation canvas area so perceived scale follows unchanged FDR augmentation"
            ),
            "quantiles": [
                0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
            ],
            "log_area_knots": [
                -1.0939053609460274, 2.407853606211777,
                2.911793703975937, 3.2581411529031463,
                3.5404331464502383, 3.7883982013862365,
                4.025328833331064, 4.233540793865287,
                4.433455103796916, 4.620838915786072,
                4.812900798763111, 5.0039862478828026,
                5.196857416822457, 5.400227674165377,
                5.624407461981047, 5.872858322100196,
                6.155378738116352, 6.482825175640593,
                6.880816055611381, 7.4580074436281985,
                11.153424678643981,
            ],
            "interpolation": "piecewise linear with endpoint clipping",
            "supervision_balance": "each target instance contributes unit total Gaussian weight before averaging instances",
            "audit_sha256": "598487AD96F59D1E4B01DE8AA026D4C9D90251785BFE9D98016CE8A5785A2454",
        },
        "output_projection": {
            "operator": "1x1 Conv",
            "channels": "192->256",
            "bias": False,
        },
        "alpha": {
            "shape": "[1,256,1,1]",
            "initialization": "zeros",
        },
        "output_equation": "X + 0.5*tanh(alpha)*O*tanh(Wo(U))",
        "residual_difficulty": {
            "prediction_source": "final ordinary decoder Query only",
            "excluded_predictions": ["encoder Query", "denoising Query"],
            "assignment": "reuse existing Hungarian assignment; no second matcher",
            "matched_equation": "clamp(0.7*(1-p)+0.3*(1-IoU),0.25,1.0)",
            "probability": "sigmoid(final target-class logit)",
            "unmatched_gt": 1.0,
            "target_generation": "FP32 detached",
        },
        "gaussian_target": {
            "sigma": "max(1,box_size_on_P3/8)",
            "truncate_sigma": 3,
            "overlap_reduction": "pixelwise maximum",
        },
        "auxiliary_focal": {
            "objective": "soft binary focal BCE",
            "alpha": 0.25,
            "gamma": 2.0,
            "reduction": "valid-pixel mean",
            "weight": 0.05,
        },
        "auxiliary_scale": {
            "objective": "FP32 SmoothL1 continuous scale regression",
            "target": (
                "current post-augmentation 640-canvas log-area percentile interpolated "
                "against frozen source-letterbox prior knots"
            ),
            "supervised_pixels": "per-instance Gaussian support",
            "reduction": "unit-weight each instance, then mean over valid instances",
            "target_generation": "FP32 detached; no second matcher",
            "weight": 0.05,
        },
        "epoch_diagnostics": [
            "loss_ra_scale",
            "scale_instances",
            "scale_mae",
            "scale_rmse",
            "scale_prediction_mean",
            "scale_prediction_std",
            "scale_target_mean",
            "scale_target_std",
            "scale_pearson",
            "scale_spearman",
            "route_entropy",
            "route_global_mean",
            "route_global_std",
            "route_load_min",
            "route_load_max",
            "scale_route_correlation_mean_abs",
            "scale_route_correlation_max_abs",
            "scale_slope_rms",
            "scale_slope_max_abs",
            "scale_modulation_route_delta_mean",
            "scale_modulation_route_delta_max",
        ],
        "ignore_boxes": {
            "class_id": -1,
            "detection": "excluded",
            "target_generation": "excluded",
            "auxiliary_negative_supervision": "masked",
            "overlapping_positive_gaussian_pixels": "valid",
        },
        "identity_initialization": True,
        "parameter_budget_ratio": MAX_PARAMETER_INCREASE_RATIO,
        "peak_vram_mib_limit": MAX_PEAK_VRAM_MIB,
    },
    "evaluation": {
        "imgsz": 640,
        "max_det": 300,
        "nms": False,
        "conf": 0.001,
        "half": False,
        "coordinate_system": "source boxes mapped to a centered aspect-preserving 640x640 letterbox canvas",
        "tiny": "letterboxed box area below 16^2 pixels",
        "small": "letterboxed box area in [16^2,32^2) pixels",
        "ignore_regions": {
            "source": "frozen labels_ignore sidecars",
            "rule": "exclude predictions with intersection-over-detection-area >= 0.5",
            "coordinate_system": "same centered 640x640 letterbox canvas as GT and predictions",
        },
        "screen_evaluated_epochs": [26, 27, 28, 29, 30],
        "formal_evaluated_epochs": [98, 99, 100],
        "explore50_evaluated_epochs": list(range(5, 51, 5)),
    },
    "screen_gate": {
        "tail5_map_delta_min": 0.002,
        "tail5_ap_tiny_delta_min": 0.0015,
        "tail5_ap_small_delta_min": 0.0015,
        "positive_same_epoch_map_deltas_min": 4,
        "tail5_ap75_delta_min": 0.0,
        "class_ap_non_decreasing_min": 7,
        "class_ap_delta_floor": -0.0025,
        "precision_delta_floor": -0.003,
        "recall_delta_floor": -0.003,
        "scale_pearson_min": 0.40,
        "scale_spearman_min": 0.40,
        "route_entropy": "finite mechanism evidence; not a performance threshold",
        "route_group_load_min": 0.05,
        "route_group_load_max": 0.95,
        "scale_slope_rms_min": 0.0001,
        "scale_modulation_route_delta_mean_min": 0.0001,
        "amp_skipped_steps_max": 0,
        "gradient_p99_ratio_max": 1.5,
        "classes": 10,
        "parameter_increase_ratio_max": MAX_PARAMETER_INCREASE_RATIO,
        "peak_vram_mib_max": MAX_PEAK_VRAM_MIB,
    },
    "advancement": {
        "screen30_requires_smoke2": True,
        "screen30_requires_screen10_gate": False,
        "screen30_initialization": "fresh paired scratch artifact; never Smoke2 checkpoint",
        "formal_requires_screen_gate": True,
        "formal_initialization": "fresh paired scratch artifact; never Screen checkpoint",
        "screen30_validation": "official val (exploratory advancement use)",
        "formal_validation": "official val; fresh paired run but no longer pristine after Screen30 selection",
        "formal_evidence_status": "selection-conditioned exploratory evidence; external untouched test evidence required for confirmation",
        "primary_formal_evidence": ["epoch100", "tail3_mean"],
        "best_checkpoint": "supplemental only",
    },
    "exploration": {
        "explore50_role": "post-hoc trajectory evidence only; never confirmatory",
        "fresh_paired_scratch": True,
        "validation": "frozen train-derived selection_set; official val remains isolated",
        "report_every_epochs": 5,
        "no_advancement_gate": True,
    },
    "publication": {
        "checkpoint_scope": "local-only",
        "publish_pt": False,
    },
}

RA_EXPERIMENT_PROTOCOL_SHA256 = hashlib.sha256(
    canonical_json_bytes(RA_EXPERIMENT_PROTOCOL)
).hexdigest().upper()


def build_ra_run_identity(
    source_identity: Mapping[str, Any],
    *,
    stage: str,
    variant: str,
    seed: int = 0,
    pair_id: str,
) -> dict[str, Any]:
    """Bind one arm to the frozen RA protocol and its paired launch."""

    if stage not in RA_STAGES:
        raise ValueError(f"unknown RA stage: {stage}")
    if variant not in RA_VARIANTS:
        raise ValueError(f"unknown RA variant: {variant}")
    if seed != 0:
        raise ValueError("RA-GLGM v1.2 is frozen to seed0")
    if not pair_id or any(character.isspace() for character in pair_id):
        raise ValueError("pair_id must be a non-empty token")
    source_sha256 = public_state_sha256(source_identity)
    run_id = (
        f"{variant}-{stage}-seed0-{source_sha256[:12].lower()}-"
        f"{RA_EXPERIMENT_PROTOCOL_SHA256[:12].lower()}"
    )
    return {
        "source_sha256": source_sha256,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "run_id": run_id,
        "pair_id": pair_id,
        "stage": stage,
        "variant": variant,
        "seed": seed,
    }


def finite_number(value: Any) -> bool:
    """Return true only for finite, non-boolean real values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def continuous_epochs(rows: Sequence[Mapping[str, Any]], expected: int) -> bool:
    try:
        epochs = [int(row["completed_epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return len(rows) == expected and epochs == list(range(1, expected + 1))


def validate_runtime_identity(
    manifest: Mapping[str, Any], *, variant: str, stage: str
) -> dict[str, Any]:
    """Fail closed on stage/arm/protocol identity drift."""
    if variant not in RA_VARIANTS or stage not in RA_STAGES:
        raise ValueError("unknown RA run identity")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("RA runtime manifest is missing run_identity")
    expected = {
        "variant": variant,
        "stage": stage,
        "seed": 0,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise ValueError(
                f"RA runtime identity mismatch for {field}: expected={value!r}, actual={identity.get(field)!r}"
            )
    if not isinstance(identity.get("run_id"), str) or not identity["run_id"]:
        raise ValueError("RA runtime run_id is missing")
    return dict(identity)


def validate_runtime_authority(
    manifest: Mapping[str, Any],
    *,
    variant: str,
    stage: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Bind a runtime manifest back to current source, initialization, and stage authority."""

    identity = validate_runtime_identity(manifest, variant=variant, stage=stage)
    if manifest.get("format_version") != 1:
        raise ValueError("RA runtime manifest format must be 1")
    if manifest.get("protocol_sha256") != RA_EXPERIMENT_PROTOCOL_SHA256:
        raise ValueError("RA runtime protocol SHA256 differs from frozen authority")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or dict(source) != current_source_identity(repository_root):
        raise ValueError("RA runtime source differs from current source authority")
    if identity.get("source_sha256") != public_state_sha256(source):
        raise ValueError("RA runtime identity is not bound to its source")
    expected_identity = build_ra_run_identity(
        source,
        stage=stage,
        variant=variant,
        seed=0,
        pair_id=str(identity.get("pair_id", "")),
    )
    if identity != expected_identity:
        raise ValueError("RA runtime identity differs from reconstructed authority")

    initial_state = manifest.get("initial_state")
    if not isinstance(initial_state, Mapping):
        raise ValueError("RA runtime initial-state authority is missing")
    initial_path = Path(str(initial_state.get("path", ""))).resolve()
    if initial_path.is_symlink() or not initial_path.is_file():
        raise FileNotFoundError("RA runtime initial-state artifact is missing")
    if file_sha256(initial_path) != str(initial_state.get("sha256", "")).upper():
        raise ValueError("RA runtime initial-state SHA256 mismatch")

    schedule = {"smoke": 2, "screen": 50, "formal": 100, "explore50": 50}[stage]
    cutoff = {"smoke": None, "screen": 30, "formal": None, "explore50": 50}[stage]
    if manifest.get("schedule_epochs") != schedule or manifest.get("cutoff_epoch") != cutoff:
        raise ValueError("RA runtime schedule/cutoff differs from frozen stage authority")
    if manifest.get("initialization_mode") != "fresh_paired_scratch":
        raise ValueError("RA runtime was not initialized from fresh paired scratch")
    if manifest.get("parent_checkpoint") is not None:
        raise ValueError("RA runtime illegally inherits a parent checkpoint")

    evaluator = Path(repository_root).resolve() / "scripts" / "evaluate_ra_glgm_checkpoints.py"
    if evaluator.is_symlink() or not evaluator.is_file():
        raise FileNotFoundError("RA locked evaluator is missing from current source")
    if manifest.get("locked_evaluator_sha256") != file_sha256(evaluator):
        raise ValueError("RA runtime locked-evaluator SHA256 differs from current source")

    screen_sha = manifest.get("screen_gate_sha256")
    hexadecimal = set("0123456789ABCDEF")

    def digest(value: Any) -> bool:
        normalized = str(value).upper()
        return len(normalized) == 64 and set(normalized) <= hexadecimal

    if stage in {"smoke", "screen", "explore50"}:
        if screen_sha is not None:
            raise ValueError(f"{stage} runtime may not inherit an upstream gate")
    elif not digest(screen_sha):
        raise ValueError("Formal100 runtime is not bound only to one Screen30 gate")
    return identity


def paired_manifests(
    baseline: Mapping[str, Any], method: Mapping[str, Any], *, stage: str
) -> bool:
    """Require both arms to share every authority except variant and run ID."""
    try:
        base_identity = validate_runtime_identity(baseline, variant="baseline", stage=stage)
        method_identity = validate_runtime_identity(method, variant="ra_glgm", stage=stage)
    except ValueError:
        return False
    shared_manifest_fields = (
        "format_version",
        "protocol_sha256",
        "source",
        "initial_state",
        "data",
        "dataset_authority",
        "learnability_report_sha256",
        "gpu_uuid",
        "schedule_epochs",
        "cutoff_epoch",
        "locked_evaluator_sha256",
        "initialization_mode",
        "parent_checkpoint",
        "screen_gate_sha256",
    )
    shared_identity_fields = ("source_sha256", "protocol_sha256", "stage", "seed", "pair_id")
    return (
        all(baseline.get(field) == method.get(field) for field in shared_manifest_fields)
        and all(base_identity.get(field) == method_identity.get(field) for field in shared_identity_fields)
        and base_identity["run_id"] != method_identity["run_id"]
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_scale_prior_authority(repository_root: str | Path) -> dict[str, Any]:
    """Verify the executable full-precision scale prior against the protocol."""

    path = Path(repository_root).resolve() / SCALE_PRIOR_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("RA v1.2 scale-prior authority is missing")
    payload = read_json(path)
    recorded = str(payload.get("content_sha256", "")).upper()
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()
    expected = RA_EXPERIMENT_PROTOCOL["module"]["scale_prior"]
    if recorded != actual or recorded != expected["audit_sha256"]:
        raise ValueError("RA v1.2 scale-prior content SHA256 mismatch")
    if (
        payload.get("format_version") != 1
        or payload.get("images") != 6471
        or payload.get("instances") != 343_204
        or payload.get("letterbox_imgsz") != 640
        or payload.get("log_area_knots") != expected["log_area_knots"]
        or len(payload.get("quantile_probabilities", [])) != 21
    ):
        raise ValueError("RA v1.2 scale-prior payload differs from frozen protocol")
    return payload


def current_source_identity(
    repository_root: str | Path, *, require_clean: bool = False
) -> dict[str, str]:
    """Fingerprint tracked plus runtime-source bytes; optionally require a clean checkout."""

    root = Path(repository_root).resolve()
    if require_clean:
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty.strip():
            raise ValueError("RA source checkout must be clean before authority preparation")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    untracked_runtime = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src",
            "scripts",
            "configs",
        ],
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256()
    paths = sorted({raw for raw in (tracked + untracked_runtime).split(b"\0") if raw})
    for raw in paths:
        path = root / raw.decode("utf-8")
        digest.update(raw + b"\0")
        digest.update(path.read_bytes())
    return {"git_commit": commit, "tree_sha256": digest.hexdigest().upper()}


def load_ra_authority(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Load and fully validate one immutable RA experiment authority."""

    validate_scale_prior_authority(repository_root)
    manifest = read_json(Path(path).resolve())
    if manifest.get("format_version") != 1:
        raise ValueError("RA protocol manifest format must be 1")
    recorded = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()
    if recorded != actual:
        raise ValueError("RA protocol manifest SHA256 mismatch")
    if manifest.get("protocol") != RA_EXPERIMENT_PROTOCOL:
        raise ValueError("RA protocol payload differs from frozen authority")
    if manifest.get("protocol_sha256") != RA_EXPERIMENT_PROTOCOL_SHA256:
        raise ValueError("RA protocol SHA256 differs from frozen authority")
    initial_state = manifest.get("initial_state")
    if not isinstance(initial_state, Mapping):
        raise ValueError("RA initial-state authority is missing")
    initial_path = Path(str(initial_state.get("path", ""))).resolve()
    if initial_path.is_symlink() or not initial_path.is_file():
        raise FileNotFoundError("RA initial-state artifact is missing")
    if file_sha256(initial_path) != str(initial_state.get("sha256", "")).upper():
        raise ValueError("RA initial-state SHA256 mismatch")
    source = manifest.get("source")
    identities = manifest.get("run_identities")
    if not isinstance(source, Mapping) or not isinstance(identities, Mapping):
        raise ValueError("RA source/run identities are missing")
    if manifest.get("source_sha256") != public_state_sha256(source):
        raise ValueError("RA source identity hash mismatch")
    source_sha = public_state_sha256(source)
    for stage in RA_STAGES:
        pair_id = f"ra-glgm-{stage}-seed0-{source_sha[:12].lower()}"
        for variant in RA_VARIANTS:
            key = f"{variant}_{stage}"
            expected = build_ra_run_identity(
                source,
                stage=stage,
                variant=variant,
                seed=0,
                pair_id=pair_id,
            )
            if identities.get(key) != expected:
                raise ValueError(f"RA run identity mismatch: {key}")
    evaluator = manifest.get("locked_evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("locked evaluator authority is missing")
    evaluator_path = Path(str(evaluator.get("path", ""))).resolve()
    if evaluator_path.is_symlink() or not evaluator_path.is_file():
        raise FileNotFoundError("locked evaluator is missing")
    if file_sha256(evaluator_path) != str(evaluator.get("sha256", "")).upper():
        raise ValueError("locked evaluator SHA256 mismatch")
    dataset_authority = manifest.get("dataset_authority")
    if not isinstance(dataset_authority, Mapping):
        raise ValueError("RA v1.2 dataset authority is missing")
    selection = dataset_authority.get("selection_set")
    if not isinstance(selection, Mapping):
        raise ValueError("RA v1.2 diagnostic selection authority is missing")
    selection_path = Path(str(selection.get("path", ""))).resolve()
    if selection_path.is_symlink() or not selection_path.is_file():
        raise FileNotFoundError("RA v1.2 diagnostic selection list is missing")
    if file_sha256(selection_path) != str(selection.get("sha256", "")).upper():
        raise ValueError("RA v1.2 diagnostic selection SHA256 mismatch")
    if int(selection.get("images", -1)) != int(
        RA_EXPERIMENT_PROTOCOL["dataset"]["selection_set"]["images"]
    ):
        raise ValueError("RA v1.2 diagnostic selection image count differs from authority")
    if int(selection.get("objects", 0)) <= 0:
        raise ValueError("RA v1.2 diagnostic selection object count is invalid")
    validate_ra_source_authority(manifest, repository_root=repository_root)
    return manifest


def validate_ra_source_authority(
    manifest: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, str]:
    """Fail closed when tracked source bytes drift from the RA authority."""

    expected = manifest.get("source")
    if not isinstance(expected, Mapping):
        raise ValueError("RA source authority is missing")
    actual = current_source_identity(repository_root)
    if actual != dict(expected):
        raise ValueError("checked-out source differs from RA authority")
    return actual


def ignore_sidecar_signature(dataset_root: str | Path) -> dict[str, Any]:
    """Hash every transformed VisDrone ignore box consumed by RA supervision."""

    root = Path(dataset_root).resolve()
    digest = hashlib.sha256()
    splits: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        directory = root / "labels_ignore" / split
        if not directory.is_dir():
            raise FileNotFoundError(f"required ignore sidecar directory is missing: {directory}")
        files = sorted(path for path in directory.glob("*.txt") if path.is_file())
        expected_names = {
            path.with_suffix(".txt").name
            for path in (root / "images" / split).glob("*.jpg")
            if path.is_file()
        }
        actual_names = {path.name for path in files}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(
                f"ignore sidecar/image mismatch for {split}: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        boxes = 0
        nonempty_files = 0
        for path in files:
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(content).hexdigest().upper().encode("ascii"))
            digest.update(b"\n")
            rows = [line for line in content.decode("ascii").splitlines() if line.strip()]
            nonempty_files += bool(rows)
            for line_number, line in enumerate(rows, 1):
                fields = line.split()
                if len(fields) != 4:
                    raise ValueError(f"invalid ignore sidecar row at {path}:{line_number}")
                values = [float(value) for value in fields]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"non-finite ignore sidecar row at {path}:{line_number}")
                cx, cy, width, height = values
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(f"invalid normalized ignore box at {path}:{line_number}")
                boxes += 1
        splits[split] = {
            "files": len(files),
            "boxes": boxes,
            "nonempty_files": nonempty_files,
            "empty_files": len(files) - nonempty_files,
        }
    return {
        "files": sum(value["files"] for value in splits.values()),
        "boxes": sum(value["boxes"] for value in splits.values()),
        "nonempty_files": sum(value["nonempty_files"] for value in splits.values()),
        "empty_files": sum(value["empty_files"] for value in splits.values()),
        "splits": splits,
        "sha256": digest.hexdigest().upper(),
    }


__all__ = [
    "BASELINE_PARAMETERS",
    "MAX_PARAMETER_INCREASE_RATIO",
    "MAX_PEAK_VRAM_MIB",
    "RA_EXPERIMENT_PROTOCOL",
    "RA_EXPERIMENT_PROTOCOL_SHA256",
    "RA_STAGES",
    "RA_VARIANTS",
    "build_ra_run_identity",
    "continuous_epochs",
    "file_sha256",
    "finite_number",
    "ignore_sidecar_signature",
    "paired_manifests",
    "read_json",
    "read_jsonl",
    "validate_runtime_identity",
    "validate_scale_prior_authority",
]
