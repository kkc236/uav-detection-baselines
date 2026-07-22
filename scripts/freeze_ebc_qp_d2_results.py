from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def artifact_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": digest.hexdigest().upper(),
    }


def build_freeze_manifest(
    *,
    variant: str,
    protocol_signature: str,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    if not variant:
        raise ValueError("variant must not be empty")
    if not protocol_signature:
        raise ValueError("protocol signature must not be empty")
    return {
        "format_version": 1,
        "variant": variant,
        "protocol_signature": protocol_signature,
        "artifacts": {name: artifact_record(path) for name, path in sorted(artifacts.items())},
    }


def write_freeze_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed freeze manifest: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def _artifact_argument(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("artifact must use NAME=PATH")
    return name, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze EBC-QP D2 evidence by size and SHA256.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--protocol-signature", required=True)
    parser.add_argument("--artifact", type=_artifact_argument, action="append", default=[], required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = dict(args.artifact)
    if len(artifacts) != len(args.artifact):
        raise SystemExit("artifact names must be unique")
    payload = build_freeze_manifest(
        variant=args.variant,
        protocol_signature=args.protocol_signature,
        artifacts=artifacts,
    )
    write_freeze_manifest(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
