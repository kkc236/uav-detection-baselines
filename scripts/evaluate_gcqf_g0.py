"""Evaluate Global, Fixed SADED, and learned GCQF from one sealed cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

from scripts.train_gcqf_g0 import MODULE_ARTIFACT_SCHEMA
from src.gcqf import GCQF
from src.gcqf_cache import VerifiedEvidenceCache
from src.gcqf_routing import route_gcqf_record
from src.gcqf_training import collate_evidence_records
from src.saded_stage import prediction_payload
from src.sbr_artifacts import atomic_write_json, load_dataset
from src.sbr_metrics import evaluate_dataset
from src.sr_peg_routing import SRPEGThresholds, route_sr_peg_record


STATES = [
    "Global",
    "Raw-Union",
    "Fixed-SADED",
    "Residual-Off",
    "Full-GCQF",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adjudicate one trained GCQF seed against Fixed SADED."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--anchor-reference", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--states", nargs="+", default=STATES)
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _stringify_mapping_keys(value: Any) -> Any:
    """Make evaluator mappings legal for strict canonical JSON."""

    if isinstance(value, Mapping):
        return {
            str(key): _stringify_mapping_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def metric_deltas(
    reference: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, float]:
    result = {}
    for key in sorted(set(reference) & set(method)):
        left, right = reference[key], method[key]
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            result[key] = float(right) - float(left)
    return result


def load_gcqf_module(
    path: str | Path,
    *,
    device: torch.device,
) -> GCQF:
    artifact = torch.load(
        Path(path).resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != MODULE_ARTIFACT_SCHEMA
    ):
        raise ValueError("GCQF module artifact schema mismatch")
    config = artifact.get("config")
    state = artifact.get("module_state")
    required_config = {
        "query_dim",
        "num_classes",
        "num_heads",
        "num_views",
        "residual_cap",
        "residual_eta",
    }
    if (
        not isinstance(config, dict)
        or set(config) != required_config
        or not isinstance(state, dict)
        or not state
    ):
        raise ValueError("GCQF module artifact payload drift")
    module = GCQF(**config)
    if set(state) != set(module.state_dict()):
        raise ValueError("GCQF module state keys drift")
    if any(
        not isinstance(value, torch.Tensor)
        or not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("GCQF module state is nonfinite")
    module.load_state_dict(state, strict=True)
    module.eval()
    return module.to(device)


def per_seed_gate(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    anchor_exact: bool,
    protected_exact: bool,
    max_det_exact: bool,
    residual_statistics: Mapping[str, float],
) -> dict[str, bool]:
    def at_least(value: float, threshold: float) -> bool:
        return float(value) + 1e-12 >= float(threshold)

    full_vs_fixed = metric_deltas(
        metrics["Fixed-SADED"],
        metrics["Full-GCQF"],
    )
    full_vs_global = metric_deltas(
        metrics["Global"],
        metrics["Full-GCQF"],
    )
    checks = {
        "anchor_reference_exact": bool(anchor_exact),
        "protected_global_exact": bool(protected_exact),
        "max_det_respected": bool(max_det_exact),
        "map_gain_vs_global": (
            at_least(full_vs_global.get("mAP50-95", -1.0), 0.005)
        ),
        "tiny_gain_vs_global": (
            at_least(full_vs_global.get("AP-tiny-SBR", -1.0), 0.010)
        ),
        "tiny_recall_gain_vs_global": (
            at_least(full_vs_global.get("tiny_recall", -1.0), 0.020)
        ),
        "medium_budget_vs_global": (
            at_least(full_vs_global.get("AP-medium-SBR", -1.0), -0.002)
        ),
        "large_budget_vs_global": (
            at_least(full_vs_global.get("AP-large-SBR", -1.0), -0.005)
        ),
        "medium_recovery_vs_fixed": (
            at_least(full_vs_fixed.get("AP-medium-SBR", -1.0), 0.008)
        ),
        "map_nonnegative_vs_fixed": (
            at_least(full_vs_fixed.get("mAP50-95", -1.0), 0.0)
        ),
    }
    diagnostics = {
        "residual_is_active": (
            residual_statistics.get("mean_abs", 0.0) >= 1e-4
        ),
        "residual_not_saturated": (
            residual_statistics.get("saturation_fraction", 1.0) < 0.5
        ),
    }
    checks.update(diagnostics)
    checks["passed"] = all(
        value
        for name, value in checks.items()
        if name not in diagnostics
    )
    return checks


def _metric_row(
    image: Mapping[str, Any],
    predictions: Sequence,
) -> dict[str, Any]:
    return {
        "image_id": image["relative_path"],
        "width": int(image["width"]),
        "height": int(image["height"]),
        "pred_boxes": [list(value.box) for value in predictions],
        "pred_scores": [float(value.score) for value in predictions],
        "pred_classes": [int(value.class_id) for value in predictions],
        "pred_source": [int(value.source_order) for value in predictions],
        "pred_query": [int(value.query_index) for value in predictions],
        "gt_boxes": [list(value) for value in image["gt_boxes"]],
        "gt_classes": [int(value) for value in image["gt_classes"]],
        "ignore_boxes": [list(value) for value in image["ignore_boxes"]],
        "effective_gain": min(
            640.0 / float(image["width"]),
            640.0 / float(image["height"]),
            1.0,
        ),
    }


def _dataset_key(image_id: str) -> str:
    path = Path(image_id)
    if path.parts and path.parts[0] in {"train", "val"}:
        return Path(*path.parts[1:]).as_posix()
    return path.as_posix()


def _prediction_json(predictions: Sequence) -> list[dict[str, Any]]:
    return [prediction_payload(value) for value in predictions]


def _load_anchor_reference(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("anchor reference rows are invalid")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(
            row.get("image_id"),
            str,
        ):
            raise ValueError("anchor reference row schema drift")
        arms = row.get("arms", {})
        predictions = (
            arms.get("route_control")
            if isinstance(arms, dict)
            else None
        )
        if predictions is None:
            predictions = row.get("predictions")
        if not isinstance(predictions, list):
            raise ValueError("anchor reference prediction schema drift")
        result[row["image_id"]] = predictions
    return result


def _tensor_statistics(values: torch.Tensor) -> dict[str, Any]:
    vector = values.detach().float().reshape(-1)
    if vector.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    return {
        "count": int(vector.numel()),
        "mean": float(vector.mean()),
        "mean_abs": float(vector.abs().mean()),
        "std": float(vector.std(unbiased=False)),
        "min": float(vector.min()),
        "max": float(vector.max()),
        "saturation_fraction": float(
            (vector.abs() >= 0.99).float().mean()
        ),
        "histogram": {
            "[0.0,0.4)": int((vector < 0.4).sum()),
            "[0.4,0.5)": int(((vector >= 0.4) & (vector < 0.5)).sum()),
            "[0.5,0.6)": int(((vector >= 0.5) & (vector < 0.6)).sum()),
            "[0.6,1.0]": int((vector >= 0.6).sum()),
        },
    }


def evaluate(args: argparse.Namespace) -> Path:
    if (
        args.device != "0"
        or args.batch != 8
        or list(args.states) != STATES
    ):
        raise ValueError("GCQF evaluation protocol drift")
    if not torch.cuda.is_available():
        raise RuntimeError("GCQF evaluation requires CUDA")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    cache = VerifiedEvidenceCache(args.cache)
    records = list(cache.iter_records())
    if len(records) != 548:
        raise RuntimeError("GCQF val cache must contain 548 records")
    dataset = load_dataset(args.data, split="val")
    if dataset["image_count"] != len(records):
        raise RuntimeError("GCQF cache/dataset image count drift")
    if (
        dataset["dataset_signature"].upper()
        != cache.manifest["dataset_signature"].upper()
    ):
        raise RuntimeError("GCQF cache/dataset signature drift")
    image_by_id = {
        value["relative_path"]: value for value in dataset["images"]
    }
    calibration = json.loads(
        args.calibration.resolve().read_text(encoding="utf-8")
    )
    selected_thresholds = calibration.get("selected_thresholds")
    if (
        calibration.get("schema_version")
        != "gcte-sr-peg-calibration/v1"
        or not isinstance(selected_thresholds, dict)
        or set(selected_thresholds)
        != {"tiny_utility", "non_tiny_risk", "global_retain"}
        or calibration.get("module_sha256")
        != _sha256_file(Path(args.module))
    ):
        raise ValueError("SR-PEG calibration authority mismatch")
    thresholds = SRPEGThresholds(**selected_thresholds)
    module = load_gcqf_module(
        args.module,
        device=torch.device("cuda:0"),
    )
    outputs_by_id: dict[str, dict[str, torch.Tensor]] = {}
    residual_values: list[torch.Tensor] = []
    probability_values: dict[str, list[torch.Tensor]] = {
        "tiny_utility": [],
        "non_tiny_risk": [],
        "anchor_admission": [],
        "global_retain": [],
    }
    loader = torch.utils.data.DataLoader(
        records,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda values: values,
    )
    for values in loader:
        batch = collate_evidence_records(values).to("cuda:0")
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            prediction = module(
                batch.global_evidence,
                batch.local_evidence,
                batch.geometry,
                anchor_mask=batch.anchor_mask,
                residual_enabled=True,
            )
        for index, image_id in enumerate(batch.image_ids):
            residual = prediction.score_residual[
                index : index + 1
            ].detach().cpu()
            utility = prediction.tiny_utility_logits[
                index : index + 1
            ].sigmoid().detach().cpu()
            risk = prediction.non_tiny_risk_logits[
                index : index + 1
            ].sigmoid().detach().cpu()
            admission = prediction.anchor_admission_logits[
                index : index + 1
            ].sigmoid().detach().cpu()
            retain = prediction.global_retain_logits[
                index : index + 1
            ].sigmoid().detach().cpu()
            outputs_by_id[image_id] = {
                "score_residual": residual,
                "tiny_utility": utility,
                "non_tiny_risk": risk,
                "anchor_admission": admission,
                "global_retain": retain,
            }
            eligible = batch.anchor_mask[index].detach().cpu().squeeze(-1)
            valid = batch.geometry.valid_mask[index].detach().cpu()
            residual_values.append(residual[0, eligible])
            probability_values["tiny_utility"].append(utility[0, valid])
            probability_values["non_tiny_risk"].append(risk[0, valid])
            probability_values["anchor_admission"].append(
                admission[0, valid]
            )
            probability_values["global_retain"].append(retain[0])
    residual_vector = torch.cat(residual_values).float()
    residual_statistics = _tensor_statistics(residual_vector)
    gate_statistics = {
        name: _tensor_statistics(torch.cat(values))
        for name, values in probability_values.items()
    }
    rows = {state: [] for state in STATES}
    protected_exact = True
    max_det_exact = True
    anchor_outputs = {}
    coverage = {
        "anchor": {},
        "residual_off": {},
        "method": {},
    }
    for record in records:
        key = _dataset_key(record.image_id)
        if key not in image_by_id:
            raise RuntimeError(f"cache image missing from dataset: {key}")
        anchor = route_gcqf_record(record, score_residual=None)
        prediction = outputs_by_id[record.image_id]
        residual_off = route_sr_peg_record(
            record,
            **prediction,
            thresholds=thresholds,
            residual_enabled=False,
        )
        method = route_sr_peg_record(
            record,
            **prediction,
            thresholds=thresholds,
            residual_enabled=True,
        )
        image = image_by_id[key]
        rows["Global"].append(_metric_row(image, anchor.control))
        rows["Raw-Union"].append(_metric_row(image, anchor.raw_union))
        rows["Fixed-SADED"].append(_metric_row(image, anchor.output))
        rows["Residual-Off"].append(
            _metric_row(image, residual_off.output)
        )
        rows["Full-GCQF"].append(_metric_row(image, method.output))
        anchor_outputs[record.image_id] = _prediction_json(anchor.output)
        protected_exact = protected_exact and bool(
            anchor.invariants.get("protected_identity_exact")
            and residual_off.invariants.get("protected_identity_exact")
            and method.invariants.get("protected_identity_exact")
        )
        max_det_exact = max_det_exact and bool(
            residual_off.invariants.get("max_det_respected")
            and method.invariants.get("max_det_respected")
        )
        for label, routed in (
            ("anchor", anchor),
            ("residual_off", residual_off),
            ("method", method),
        ):
            for name, value in routed.coverage.items():
                coverage[label][name] = (
                    coverage[label].get(name, 0) + int(value)
                )
    if args.anchor_reference is None:
        anchor_exact = False
        anchor_reference_sha256 = None
    else:
        reference = _load_anchor_reference(args.anchor_reference)
        anchor_exact = reference == anchor_outputs
        anchor_reference_sha256 = _sha256_file(args.anchor_reference)
    metrics = {
        state: evaluate_dataset(state_rows)
        for state, state_rows in rows.items()
    }
    deltas = {
        "full_minus_global": metric_deltas(
            metrics["Global"],
            metrics["Full-GCQF"],
        ),
        "anchor_minus_global": metric_deltas(
            metrics["Global"],
            metrics["Fixed-SADED"],
        ),
        "full_minus_anchor": metric_deltas(
            metrics["Fixed-SADED"],
            metrics["Full-GCQF"],
        ),
        "full_minus_raw_union": metric_deltas(
            metrics["Raw-Union"],
            metrics["Full-GCQF"],
        ),
        "residual_off_minus_anchor": metric_deltas(
            metrics["Fixed-SADED"],
            metrics["Residual-Off"],
        ),
    }
    gate = per_seed_gate(
        metrics,
        anchor_exact=anchor_exact,
        protected_exact=protected_exact,
        max_det_exact=max_det_exact,
        residual_statistics=residual_statistics,
    )
    result = {
        "schema_version": "gcte-gcqf-five-state/v2",
        "cache": {
            "path": Path(args.cache).resolve().as_posix(),
            "manifest_sha256": _sha256_file(Path(args.cache)),
            "baseline_sha256": cache.manifest["baseline_sha256"],
        },
        "module": {
            "path": Path(args.module).resolve().as_posix(),
            "sha256": _sha256_file(Path(args.module)),
        },
        "calibration": {
            "path": args.calibration.resolve().as_posix(),
            "sha256": _sha256_file(args.calibration),
            "thresholds": selected_thresholds,
        },
        "dataset": {
            "yaml": Path(args.data).resolve().as_posix(),
            "signature": dataset["dataset_signature"],
            "image_count": dataset["image_count"],
        },
        "anchor_reference": {
            "path": (
                None
                if args.anchor_reference is None
                else args.anchor_reference.resolve().as_posix()
            ),
            "sha256": anchor_reference_sha256,
            "exact": anchor_exact,
        },
        "metrics": metrics,
        "deltas": deltas,
        "residual_statistics": residual_statistics,
        "gate_probability_statistics": gate_statistics,
        "protected_global_exact": protected_exact,
        "max_det_respected": max_det_exact,
        "coverage": coverage,
        "per_seed_gate": gate,
    }
    atomic_write_json(output, _stringify_mapping_keys(result))
    print(
        f"GCQF_EVALUATION_COMPLETE passed={gate['passed']} "
        f"anchor_exact={anchor_exact} output={output}",
        flush=True,
    )
    return output


def main() -> None:
    print(evaluate(build_parser().parse_args()))


if __name__ == "__main__":
    main()
