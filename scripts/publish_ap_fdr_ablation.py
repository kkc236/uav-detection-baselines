"""Gate, package, and publish completed AP-FDR Formal100 ablations."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests


SOURCE_COMMIT = "ebb349aeb2cf092d4880751e165e22614c3c9d8c"
DEFAULT_REPOSITORY = "kkc236/icassp2027-fdr-bpdd-fia-material"
DEFAULT_TAG = "ap-fdr-internal-ablation-seed0-20260820"
DEFAULT_TARGET = "main"
EXPECTED_EPOCHS = 100


class PublicationGateError(RuntimeError):
    """Raised before any remote mutation when evidence is incomplete."""


@dataclass(frozen=True)
class VariantSpec:
    name: str
    run_dir: Path
    train_log: Path
    dry_run: Path
    authority: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        staging = Path(stream.name)
    os.replace(staging, path)


def completed_epochs(results_csv: Path) -> int:
    path = Path(results_csv)
    if not path.is_file():
        raise PublicationGateError(f"missing results.csv: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(float(str(row["epoch"]).strip())) for row in rows]
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as error:
        raise PublicationGateError(f"unreadable results.csv: {path}: {error}") from error
    if len(epochs) != EXPECTED_EPOCHS:
        raise PublicationGateError(
            f"results.csv must contain exactly {EXPECTED_EPOCHS} epochs: "
            f"{path} has {len(epochs)}"
        )
    if epochs not in (list(range(EXPECTED_EPOCHS)), list(range(1, EXPECTED_EPOCHS + 1))):
        raise PublicationGateError(f"results.csv epochs are not continuous: {path}")
    return len(epochs)


def _relative(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError as error:
        raise PublicationGateError(f"artifact escapes publication root: {path}") from error


def variant_artifacts(spec: VariantSpec) -> dict[str, Path]:
    return {
        "args.yaml": spec.run_dir / "args.yaml",
        "authority.json": spec.authority,
        "best.pt": spec.run_dir / "weights" / "best.pt",
        "dry-run.json": spec.dry_run,
        "last.pt": spec.run_dir / "weights" / "last.pt",
        "results.csv": spec.run_dir / "results.csv",
        "train.log": spec.train_log,
    }


def validate_variant(spec: VariantSpec, *, base_dir: Path) -> dict[str, Any]:
    artifacts = variant_artifacts(spec)
    for label, path in artifacts.items():
        if path.is_symlink() or not path.is_file():
            raise PublicationGateError(f"missing {label}: {path}")
    epochs = completed_epochs(artifacts["results.csv"])
    return {
        "name": spec.name,
        "completed_epochs": epochs,
        "artifacts": {
            label: {
                "path": _relative(path, base_dir),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in sorted(artifacts.items())
        },
    }


def build_publication_manifest(
    variants: Iterable[VariantSpec],
    *,
    base_dir: Path,
    repository: str,
    tag: str,
) -> dict[str, Any]:
    validated = [validate_variant(spec, base_dir=base_dir) for spec in variants]
    return {
        "format_version": 1,
        "experiment": "ap_fdr_internal_ablation_seed0_formal100",
        "repository": repository,
        "tag": tag,
        "source_commit": SOURCE_COMMIT,
        "variants": sorted(validated, key=lambda item: str(item["name"])),
    }


def default_variants(base_dir: Path) -> list[VariantSpec]:
    base = Path(base_dir).resolve()
    return [
        VariantSpec(
            name="no-preliminary-reference",
            run_dir=base
            / "runs"
            / "formal-seed0-ap-fdr-no-preliminary-reference",
            train_log=base / "logs" / "no-preliminary-reference-train.log",
            dry_run=base / "logs" / "no-preliminary-reference-dry-run.json",
            authority=base
            / "runs"
            / "authority"
            / "formal-seed0-ap-fdr-no-preliminary-reference.json",
        ),
        VariantSpec(
            name="no-dn-fdr",
            run_dir=base / "runs" / "formal-seed0-ap-fdr-no-dn-fdr",
            train_log=base / "logs" / "no-dn-fdr-train.log",
            dry_run=base / "logs" / "no-dn-fdr-dry-run.json",
            authority=base
            / "runs"
            / "authority"
            / "formal-seed0-ap-fdr-no-dn-fdr.json",
        ),
    ]


def _normalized_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def build_variant_archive(
    spec: VariantSpec,
    manifest: Mapping[str, Any],
    *,
    destination: Path,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                info = tarfile.TarInfo(f"{spec.name}/artifact-manifest.json")
                info.size = len(manifest_bytes)
                info = _normalized_tarinfo(info)
                import io

                archive.addfile(info, io.BytesIO(manifest_bytes))
                for label, path in sorted(variant_artifacts(spec).items()):
                    archive.add(
                        path,
                        arcname=f"{spec.name}/{label}",
                        recursive=False,
                        filter=_normalized_tarinfo,
                    )
    return destination


def github_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return session


def checked(response: Any) -> Any:
    if response.ok:
        return response
    raise RuntimeError(
        f"GitHub API {response.status_code}: {str(response.text)[:500]}"
    )


def verify_private_repository(session: requests.Session, repository: str) -> None:
    response = checked(
        session.get(f"https://api.github.com/repos/{repository}", timeout=30)
    )
    if response.json().get("private") is not True:
        raise RuntimeError(f"publication target is not private: {repository}")


def get_or_create_release(
    session: requests.Session,
    *,
    repository: str,
    tag: str,
    target_commitish: str,
) -> dict[str, Any]:
    api = f"https://api.github.com/repos/{repository}"
    response = session.get(f"{api}/releases/tags/{tag}", timeout=30)
    if response.status_code == 404:
        response = session.post(
            f"{api}/releases",
            json={
                "tag_name": tag,
                "target_commitish": target_commitish,
                "name": "AP-FDR Internal Ablation Seed0 Formal100",
                "body": (
                    "Automatically staged evidence for the two AP-FDR internal "
                    "ablations. Assets require evidence review before paper use."
                ),
                "draft": False,
                "prerelease": True,
            },
            timeout=30,
        )
    return checked(response).json()


def upload_asset(
    session: Any,
    *,
    release: Mapping[str, Any],
    path: Path,
    asset_name: str | None = None,
) -> str:
    path = Path(path)
    name = asset_name or path.name
    assets = {str(item["name"]): item for item in release.get("assets", [])}
    existing = assets.get(name)
    action = "uploaded"
    if existing and int(existing["size"]) == path.stat().st_size:
        return "skipped"
    if existing:
        checked(session.delete(str(existing["url"]), timeout=30))
        action = "replaced"
    upload_url = str(release["upload_url"]).split("{")[0]
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(path.stat().st_size),
    }
    with path.open("rb") as stream:
        checked(
            session.post(
                upload_url,
                params={"name": name},
                headers=headers,
                data=stream,
                timeout=(30, 3600),
            )
        )
    return action


def verify_assets(release: Mapping[str, Any], expected: Mapping[str, int]) -> bool:
    actual = {str(item["name"]): int(item["size"]) for item in release.get("assets", [])}
    return all(actual.get(name) == size for name, size in expected.items())


def _variant_manifest(
    publication_manifest: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    for item in publication_manifest["variants"]:
        if item["name"] == name:
            return item
    raise KeyError(name)


def publish(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = args.base_dir.resolve()
    if not (base_dir / "all.completed").is_file():
        raise PublicationGateError(f"completion marker is absent: {base_dir / 'all.completed'}")
    variants = default_variants(base_dir)
    manifest = build_publication_manifest(
        variants,
        base_dir=base_dir,
        repository=args.repository,
        tag=args.tag,
    )
    staging = base_dir / "publication-staging"
    staging.mkdir(parents=True, exist_ok=True)
    archive_paths: list[Path] = []
    for spec in variants:
        archive_paths.append(
            build_variant_archive(
                spec,
                _variant_manifest(manifest, spec.name),
                destination=staging / f"ap-fdr-{spec.name}-seed0-formal100.tar.gz",
            )
        )
    manifest = {
        **manifest,
        "release_assets": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(archive_paths)
        },
    }
    manifest_path = staging / "publication-manifest.json"
    write_json_atomic(manifest_path, manifest)
    assets = [*archive_paths, manifest_path]
    if args.check_only:
        print(json.dumps({"ready": True, **manifest}, indent=2, sort_keys=True))
        return {"ready": True, **manifest}

    token_file = args.token_file.resolve()
    if token_file.is_symlink() or not token_file.is_file():
        raise FileNotFoundError(token_file)
    token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("GitHub token file is empty")
    session = github_session(token)
    verify_private_repository(session, args.repository)
    release = get_or_create_release(
        session,
        repository=args.repository,
        tag=args.tag,
        target_commitish=args.target_commitish,
    )
    for path in assets:
        upload_asset(session, release=release, path=path)
        release = checked(session.get(str(release["url"]), timeout=30)).json()
    expected = {path.name: path.stat().st_size for path in assets}
    if not verify_assets(release, expected):
        raise RuntimeError("GitHub Release asset size verification failed")
    status = {
        "format_version": 1,
        "published": True,
        "release_url": release["html_url"],
        "repository": args.repository,
        "tag": args.tag,
        "assets": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in assets
        },
    }
    write_json_atomic(base_dir / "publication-status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--target-commitish", default=DEFAULT_TARGET)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        publish(args)
    except Exception as error:
        print(f"publication failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
