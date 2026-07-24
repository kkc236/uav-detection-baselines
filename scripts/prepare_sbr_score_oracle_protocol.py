"""Seal the score-oracle protocol without reading scientific results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import string
import subprocess
import sys
from typing import Any

from src.sbr_artifacts import (
    atomic_write_json,
    git_provenance,
    sha256_file,
)
from src.sbr_score_oracle import (
    CONF,
    GATES,
    IOS,
    MAX_DET,
    THRESHOLDS,
)


SCHEMA_VERSION = "sbr-score-oracle-input/v1"
UPSTREAM_SCHEMA_VERSION = "sbr-v2-audit-input/v1"


def _git_digest(value: Any, name: str) -> str:
    text = str(value).lower()
    if (
        len(text) != 40
        or any(character not in string.hexdigits for character in text)
    ):
        raise ValueError(f"{name} must be a 40-character Git digest")
    return text


def prepare_protocol(
    *,
    upstream: Path,
    spec: Path,
    commit: str,
    tree: str,
) -> dict[str, Any]:
    upstream = Path(upstream).resolve()
    spec = Path(spec).resolve()
    if not upstream.is_file():
        raise FileNotFoundError(upstream)
    if not spec.is_file():
        raise FileNotFoundError(spec)
    try:
        upstream_payload = json.loads(
            upstream.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError("upstream manifest is not valid JSON") from exc
    if (
        not isinstance(upstream_payload, dict)
        or upstream_payload.get("schema_version")
        != UPSTREAM_SCHEMA_VERSION
    ):
        raise ValueError("upstream manifest schema is not frozen V2")
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream_input": {
            "uri": str(upstream),
            "sha256": sha256_file(upstream),
        },
        "approved_spec": {
            "uri": str(spec),
            "sha256": sha256_file(spec),
        },
        "expected_source": {
            "commit": _git_digest(commit, "commit"),
            "tree": _git_digest(tree, "tree"),
        },
        "frozen_rule": frozen_rule_payload(),
        "forbidden_inputs": ["test-dev", "external-dataset"],
    }


def frozen_rule_payload() -> dict[str, Any]:
    return {
        "conf": CONF,
        "max_det": MAX_DET,
        "ios": IOS,
        "thresholds": list(THRESHOLDS),
        "gates": GATES,
        "group_rule": (
            "mixed-cluster-all-local-strictly-above-best-full"
        ),
        "demotion": (
            "float64-nextafter-anchor-toward-negative-infinity"
        ),
        "selection": (
            "all-threshold-all-tiny-large-nondecrease-"
            "and-large-sum-positive"
        ),
    }


def capture_clean_source(repo: Path) -> dict[str, str]:
    repo = Path(repo).resolve()
    provenance = git_provenance(repo)
    if (
        provenance.get("clean_tracked") is not True
        or provenance.get("untracked") is not False
    ):
        raise ValueError("protocol source worktree must be clean")
    commit = _git_digest(provenance.get("commit"), "commit")
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    tree = _git_digest(completed.stdout.strip(), "tree")
    return {"commit": commit, "tree": tree}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal the frozen SBR score-oracle protocol"
    )
    parser.add_argument("--upstream-input", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    source = capture_clean_source(args.repo)
    payload = prepare_protocol(
        upstream=args.upstream_input,
        spec=args.spec,
        commit=source["commit"],
        tree=source["tree"],
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except Exception as exc:
        print(
            f"SBR_SCORE_ORACLE_PROTOCOL_INVALID: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
