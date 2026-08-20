from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.publish_ap_fdr_ablation import (
    SOURCE_COMMIT,
    PublicationGateError,
    VariantSpec,
    build_publication_manifest,
    completed_epochs,
    upload_asset,
    validate_variant,
)


def _write_results(path: Path, epochs: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["epoch", "metrics/mAP50-95(B)"],
        )
        writer.writeheader()
        for epoch in epochs:
            writer.writerow({"epoch": epoch, "metrics/mAP50-95(B)": epoch / 1000})


def _make_variant(base: Path, slug: str, epochs: list[int] | None = None) -> VariantSpec:
    run_dir = base / "runs" / slug
    _write_results(run_dir / "results.csv", epochs or list(range(100)))
    (run_dir / "args.yaml").write_text("epochs: 100\nseed: 0\n", encoding="utf-8")
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best-checkpoint")
    (weights / "last.pt").write_bytes(b"last-checkpoint")
    logs = base / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    train_log = logs / f"{slug}.log"
    dry_run = logs / f"{slug}-dry-run.json"
    train_log.write_text("training complete\n", encoding="utf-8")
    dry_run.write_text('{"dry_run": true}\n', encoding="utf-8")
    authority = base / "runs" / "authority" / f"{slug}.json"
    authority.parent.mkdir(parents=True, exist_ok=True)
    authority.write_text('{"authority": true}\n', encoding="utf-8")
    return VariantSpec(
        name=slug,
        run_dir=run_dir,
        train_log=train_log,
        dry_run=dry_run,
        authority=authority,
    )


def test_completed_epochs_accepts_exact_zero_and_one_based_sequences(tmp_path: Path) -> None:
    zero_based = tmp_path / "zero.csv"
    one_based = tmp_path / "one.csv"
    _write_results(zero_based, list(range(100)))
    _write_results(one_based, list(range(1, 101)))

    assert completed_epochs(zero_based) == 100
    assert completed_epochs(one_based) == 100


def test_completed_epochs_rejects_incomplete_or_discontinuous_results(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.csv"
    discontinuous = tmp_path / "discontinuous.csv"
    _write_results(incomplete, list(range(99)))
    _write_results(discontinuous, [*range(50), *range(51, 101)])

    with pytest.raises(PublicationGateError, match="exactly 100"):
        completed_epochs(incomplete)
    with pytest.raises(PublicationGateError, match="continuous"):
        completed_epochs(discontinuous)


def test_validate_variant_requires_every_artifact_and_hashes_bytes(tmp_path: Path) -> None:
    spec = _make_variant(tmp_path, "no-preliminary-reference")
    manifest = validate_variant(spec, base_dir=tmp_path)

    assert manifest["completed_epochs"] == 100
    assert manifest["name"] == spec.name
    best = manifest["artifacts"]["best.pt"]
    assert best["bytes"] == len(b"best-checkpoint")
    assert best["sha256"] == hashlib.sha256(b"best-checkpoint").hexdigest()
    assert not Path(best["path"]).is_absolute()

    spec.authority.unlink()
    with pytest.raises(PublicationGateError, match="missing authority"):
        validate_variant(spec, base_dir=tmp_path)


def test_publication_manifest_is_stable_and_binds_source(tmp_path: Path) -> None:
    first = _make_variant(tmp_path, "no-preliminary-reference")
    second = _make_variant(tmp_path, "no-dn-fdr")

    one = build_publication_manifest(
        [first, second],
        base_dir=tmp_path,
        repository="kkc236/icassp2027-fdr-bpdd-fia-material",
        tag="ap-fdr-internal-ablation-seed0-20260820",
    )
    two = build_publication_manifest(
        [second, first],
        base_dir=tmp_path,
        repository="kkc236/icassp2027-fdr-bpdd-fia-material",
        tag="ap-fdr-internal-ablation-seed0-20260820",
    )

    assert one == two
    assert one["source_commit"] == SOURCE_COMMIT
    assert [item["name"] for item in one["variants"]] == [
        "no-dn-fdr",
        "no-preliminary-reference",
    ]


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, *, assets: list[dict]) -> None:
        self.assets = assets
        self.deleted: list[str] = []
        self.uploaded: list[tuple[str, int]] = []

    def delete(self, url: str, timeout: int) -> FakeResponse:
        self.deleted.append(url)
        return FakeResponse(204)

    def post(self, url: str, *, params: dict, headers: dict, data, timeout) -> FakeResponse:
        payload = data.read()
        self.uploaded.append((params["name"], len(payload)))
        return FakeResponse(201, {"name": params["name"], "size": len(payload)})


def test_upload_asset_is_idempotent_and_replaces_wrong_size(tmp_path: Path) -> None:
    asset = tmp_path / "variant.tar.gz"
    asset.write_bytes(b"12345")
    release = {"upload_url": "https://uploads.example/{?name}", "assets": []}

    fresh = FakeSession(assets=[])
    assert upload_asset(fresh, release=release, path=asset) == "uploaded"
    assert fresh.uploaded == [(asset.name, 5)]

    same = FakeSession(assets=[])
    release["assets"] = [{"name": asset.name, "size": 5, "url": "asset/1"}]
    assert upload_asset(same, release=release, path=asset) == "skipped"
    assert same.deleted == []
    assert same.uploaded == []

    wrong = FakeSession(assets=[])
    release["assets"] = [{"name": asset.name, "size": 4, "url": "asset/2"}]
    assert upload_asset(wrong, release=release, path=asset) == "replaced"
    assert wrong.deleted == ["asset/2"]
    assert wrong.uploaded == [(asset.name, 5)]


def test_watcher_contract_waits_for_completion_and_never_shuts_down() -> None:
    root = Path(__file__).resolve().parents[1]
    watcher = root / "scripts" / "watch_and_publish_ap_fdr_ablation.sh"
    content = watcher.read_text(encoding="utf-8")

    assert "all.completed" in content
    assert "--token-file" in content
    assert "ap-fdr-internal-ablation-seed0-20260820" in content
    assert "kkc236/icassp2027-fdr-bpdd-fia-material" in content
    assert SOURCE_COMMIT in content
    assert "shutdown" not in content.lower()
    assert "poweroff" not in content.lower()
