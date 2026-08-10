"""Create the immutable RA-GLGM-on-FDR paired experiment authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fdr_protocol import (  # noqa: E402
    canonical_json_bytes,
    public_state_sha256,
    write_create_only_manifest,
)
from src.ra_experiment_protocol import (  # noqa: E402
    RA_EXPERIMENT_PROTOCOL,
    RA_EXPERIMENT_PROTOCOL_SHA256,
    RA_STAGES,
    RA_VARIANTS,
    build_ra_run_identity,
    current_source_identity,
    file_sha256,
    ignore_sidecar_signature,
)
from src.lpr_protocol import dataset_signature  # noqa: E402
from src.ra_glgm_protocol import validate_ra_glgm_initial_state  # noqa: E402
from src.ra_v11_selection import build_ra_v11_selection_authority  # noqa: E402


def _hex(value: str, length: int, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != length or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be exactly {length} hexadecimal characters")
    return normalized


def prepare_manifest(
    *,
    source_commit: str,
    source_tree_sha256: str,
    gpu_uuid: str,
    initial_state: str | Path,
    locked_evaluator: str | Path,
    dataset_root: str | Path,
    output: str | Path,
    selection_list: str | Path | None = None,
    screen30_selection_list: str | Path | None = None,
) -> dict:
    if not gpu_uuid.startswith("GPU-") or any(character.isspace() for character in gpu_uuid):
        raise ValueError("gpu_uuid must be one NVIDIA GPU UUID token")
    source = {
        "git_commit": _hex(source_commit, 40, "source_commit"),
        "tree_sha256": _hex(source_tree_sha256, 64, "source_tree_sha256").upper(),
    }
    actual_source = current_source_identity(ROOT, require_clean=True)
    if source != actual_source:
        raise ValueError(
            f"supplied source identity differs from clean checkout: expected={actual_source}, actual={source}"
        )
    state_path = Path(initial_state).resolve()
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("initial_state must be a regular existing file")
    artifact = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_ra_glgm_initial_state(artifact)
    evaluator_path = Path(locked_evaluator).resolve()
    if evaluator_path.is_symlink() or not evaluator_path.is_file():
        raise ValueError("locked_evaluator must be a regular existing file")
    dataset_path = Path(dataset_root).resolve()
    positive_signature = dataset_signature(dataset_path)
    expected_dataset = RA_EXPERIMENT_PROTOCOL["dataset"]
    if positive_signature.get("sha256") != expected_dataset["sha256"]:
        raise ValueError("VisDrone positive dataset differs from frozen FDR authority")
    ignore_signature = ignore_sidecar_signature(dataset_path)
    expected_ignore = expected_dataset["ignore_sidecar"]
    for split in ("train", "val"):
        actual_split = ignore_signature["splits"][split]
        if actual_split["files"] != int(expected_ignore["files"][split]):
            raise ValueError(f"VisDrone {split} ignore sidecar file count is incomplete")
        if actual_split["boxes"] != int(expected_ignore["boxes"][split]):
            raise ValueError(f"VisDrone {split} ignore box count differs from authority")
    output_path = Path(output).resolve()
    selection_path = (
        Path(selection_list).resolve()
        if selection_list is not None
        else output_path.parent / "selection-dev.txt"
    )
    screen30_selection_path = (
        Path(screen30_selection_list).resolve()
        if screen30_selection_list is not None
        else output_path.parent / "screen30-dev.txt"
    )
    selection_report = build_ra_v11_selection_authority(
        dataset_path, selection_path, screen30_selection_path
    )
    selected = selection_report["selection"]
    selected_list = selected["absolute_list"]
    selection_authority = {
        "path": selected_list["path"],
        "sha256": selected_list["sha256"],
        "images": selected_list["count"],
        "objects": selected["objects"],
        "relative_subset_sha256": selected["relative_path_sha256"],
        "image_manifest_sha256": selected["image_manifest_sha256"],
        "label_manifest_sha256": selected["label_manifest_sha256"],
        "report": selection_report,
    }
    screen30_selected = selection_report["screen30_selection"]
    screen30_selected_list = screen30_selected["absolute_list"]
    screen30_selection_authority = {
        "path": screen30_selected_list["path"],
        "sha256": screen30_selected_list["sha256"],
        "images": screen30_selected_list["count"],
        "objects": screen30_selected["objects"],
        "relative_subset_sha256": screen30_selected["relative_path_sha256"],
        "image_manifest_sha256": screen30_selected["image_manifest_sha256"],
        "label_manifest_sha256": screen30_selected["label_manifest_sha256"],
        "report_sha256": hashlib.sha256(
            canonical_json_bytes(selection_report)
        ).hexdigest().upper(),
    }
    source_sha = public_state_sha256(source)
    identities = {}
    for stage in RA_STAGES:
        pair_id = f"ra-glgm-{stage}-seed0-{source_sha[:12].lower()}"
        for variant in RA_VARIANTS:
            identities[f"{variant}_{stage}"] = build_ra_run_identity(
                source,
                stage=stage,
                variant=variant,
                seed=0,
                pair_id=pair_id,
            )
    manifest = {
        "format_version": 1,
        "source": source,
        "source_sha256": source_sha,
        "protocol": RA_EXPERIMENT_PROTOCOL,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "gpu_uuid": gpu_uuid,
        "migration": artifact["migration"],
        "initial_state": {
            "path": str(state_path),
            "sha256": file_sha256(state_path),
            "fingerprints": artifact["fingerprints"],
        },
        "locked_evaluator": {
            "path": str(evaluator_path),
            "sha256": file_sha256(evaluator_path),
        },
        "dataset_authority": {
            "root": str(dataset_path),
            "positive": positive_signature,
            "ignore": ignore_signature,
            "selection_set": selection_authority,
            "screen30_selection_set": screen30_selection_authority,
        },
        "run_identities": identities,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest().upper()
    write_create_only_manifest(output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--locked-evaluator", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-list", type=Path)
    parser.add_argument("--screen30-selection-list", type=Path)
    args = parser.parse_args()
    manifest = prepare_manifest(
        source_commit=args.source_commit,
        source_tree_sha256=args.source_tree_sha256,
        gpu_uuid=args.gpu_uuid,
        initial_state=args.initial_state,
        locked_evaluator=args.locked_evaluator,
        dataset_root=args.dataset_root,
        output=args.output,
        selection_list=args.selection_list,
        screen30_selection_list=args.screen30_selection_list,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
