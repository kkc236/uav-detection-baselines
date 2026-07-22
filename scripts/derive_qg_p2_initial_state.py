from __future__ import annotations

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import torch
from ultralytics.data.utils import check_det_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ebc_qp_config import EBCQPConfig
from src.ebc_qp_protocol import state_fingerprint
from src.rtdetr_ebc_qp import EBCQPDetectionModel


def derive_qg_initial_state(
    parent: Mapping[str, Any],
    quality_state: Mapping[str, torch.Tensor],
    *,
    parent_sha256: str,
) -> dict[str, Any]:
    common = parent["common_state"]
    innovation = parent["innovation_state"]
    fingerprints = parent["fingerprints"]
    if state_fingerprint(common) != fingerprints["common"]:
        raise ValueError("parent common-state fingerprint mismatch")
    if state_fingerprint(innovation) != fingerprints["innovation"]:
        raise ValueError("parent innovation-state fingerprint mismatch")
    if not quality_state or any("p2_quality_head" not in name.split(".") for name in quality_state):
        raise ValueError("QG derivation accepts only quality-head state")
    if set(quality_state) & set(innovation):
        raise ValueError("quality-head state overlaps the parent innovation state")
    if any(torch.count_nonzero(value).item() for value in quality_state.values()):
        raise ValueError("quality-head state must be zero initialized")

    derived_innovation = {
        name: value.detach().cpu().clone()
        for name, value in innovation.items()
    }
    derived_innovation.update(
        {
            name: value.detach().cpu().clone()
            for name, value in quality_state.items()
        }
    )
    metadata = {
        **dict(parent.get("metadata", {})),
        "variant": "qg-p2-v1",
        "parent_initial_state_sha256": parent_sha256,
        "quality_head_initialization": "zeros",
    }
    return {
        "format_version": 1,
        "common_state": {name: value.detach().cpu().clone() for name, value in common.items()},
        "innovation_state": derived_innovation,
        "metadata": metadata,
        "fingerprints": {
            "common": fingerprints["common"],
            "innovation": state_fingerprint(derived_innovation),
        },
    }


def derive_qg_protocol_manifest(
    parent: Mapping[str, Any],
    *,
    initial_state_path: Path,
    initial_state_sha256: str,
    common_fingerprint: str,
    git_commit: str,
) -> dict[str, Any]:
    required = ("seed", "dataset", "subset", "data", "initial_state", "signature")
    missing = [name for name in required if name not in parent]
    if missing:
        raise ValueError(f"parent protocol is missing fields: {missing}")
    manifest = {
        **{name: value for name, value in parent.items() if name not in {"signature", "initial_state", "git_commit"}},
        "format_version": 2,
        "variant": "qg-p2-v1",
        "git_commit": git_commit,
        "common_fingerprint": common_fingerprint,
        "initial_state": {
            "path": str(initial_state_path.resolve()),
            "sha256": initial_state_sha256,
        },
        "lineage": {
            "parent_protocol_signature": parent["signature"],
            "parent_initial_state_sha256": parent["initial_state"]["sha256"],
        },
    }
    manifest["signature"] = _json_sha256(manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive an immutable QG-P2 initial state from a frozen parent state.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-protocol", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    data = check_det_dataset(str(args.data))
    model = EBCQPDetectionModel(
        ROOT / "configs" / "rtdetr-l-ebc-qp.yaml",
        nc=data["nc"],
        ch=data["channels"],
        verbose=False,
        ebc_config=EBCQPConfig(
            lambda_ebc=0.0,
            learnable_fusion_gamma=True,
            quality_gated_p2=True,
        ),
    )
    model_state = model.state_dict()
    parent_names = set(parent["common_state"]) | set(parent["innovation_state"])
    quality_names = {name for name in model_state if "p2_quality_head" in name.split(".")}
    if set(model_state) - quality_names != parent_names:
        missing = sorted((set(model_state) - quality_names) - parent_names)
        unexpected = sorted(parent_names - (set(model_state) - quality_names))
        raise SystemExit(f"parent state does not match QG model outside quality head: missing={missing}, unexpected={unexpected}")

    parent_hash = _file_sha256(args.parent)
    derived = derive_qg_initial_state(
        parent,
        {name: model_state[name] for name in sorted(quality_names)},
        parent_sha256=parent_hash,
    )
    _write_locked_state(args.output, derived)
    parent_protocol = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    if parent_protocol.get("initial_state", {}).get("sha256") != parent_hash:
        raise SystemExit("parent protocol does not identify the supplied parent initial state")
    manifest = derive_qg_protocol_manifest(
        parent_protocol,
        initial_state_path=args.output,
        initial_state_sha256=_file_sha256(args.output),
        common_fingerprint=derived["fingerprints"]["common"],
        git_commit=_git_commit(),
    )
    _write_locked_text(args.output_protocol, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": _file_sha256(args.output),
                "parent_sha256": parent_hash,
                "common_fingerprint": derived["fingerprints"]["common"],
                "innovation_fingerprint": derived["fingerprints"]["innovation"],
                "quality_tensors": sorted(quality_names),
                "protocol": str(args.output_protocol.resolve()),
                "protocol_signature": manifest["signature"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _write_locked_state(path: Path, artifact: dict[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        current = torch.load(path, map_location="cpu", weights_only=False)
        if current.get("fingerprints") != artifact.get("fingerprints") or current.get("metadata") != artifact.get("metadata"):
            raise FileExistsError(f"refusing to replace changed initial state: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(path)


def _write_locked_text(path: Path, content: str) -> None:
    path = path.resolve()
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to replace changed protocol manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_sha256(payload: object) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(content).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    main()
