"""Publish terminal evidence for the authoritative Fresh-100 seed-0 run.

The publisher is intentionally external to the training checkout. It never
writes to the remote server and never treats a driver exit code as scientific
success without replaying the frozen runtime validator against downloaded
bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

import paramiko
import psutil


def resolve_state_root(
    configured: str | None,
    default: Path,
) -> Path:
    return Path(configured).resolve() if configured else default.resolve()


RUN_ID = "final-saded-fresh100-c5c35374"
HOST = "36.103.177.186"
USER = "ubuntu"
TRAINER_PID = 417400
DRIVER_PID = 417396
EXPECTED_SOURCE_COMMIT = "c5c353744f0d07366350389bf8d6c5fe0f62b8f8"
EXPECTED_PROTOCOL_SHA256 = (
    "F3C89D0F36827079F8EF149FAF1D088FDF43F112ADDCC68391EA9B6564F27D64"
)
REMOTE_LOG_ROOT = Path("/home/ubuntu/saded-fresh100-logs") / RUN_ID
REMOTE_RUN_ROOT = Path("/home/ubuntu/saded-fresh100-runs") / RUN_ID / "seed0"
REMOTE_PROTOCOL = (
    Path("/home/ubuntu/saded-fresh100-protocols")
    / RUN_ID
    / "protocol_manifest.json"
)
PUBLISHER_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_ROOT = (
    PUBLISHER_ROOT.parent / "saded-fresh100-validator-c5c35374"
)
EVIDENCE_ROOT = (
    PUBLISHER_ROOT / "docs" / "evidence" / "saded_fresh100_seed0"
)
LOCAL_STATE_ROOT = resolve_state_root(
    os.environ.get("SBR_FRESH100_STATE_ROOT"),
    PUBLISHER_ROOT.parents[1]
    / "runtime"
    / "sbr-fresh100-publisher",
)
DOWNLOAD_ROOT = LOCAL_STATE_ROOT / "downloads"
STATE_PATH = LOCAL_STATE_ROOT / "status.json"
LOG_PATH = LOCAL_STATE_ROOT / "publisher.log"
LOCK_PATH = LOCAL_STATE_ROOT / "publisher.lock"
KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts"
BRANCH = "final-saded-fresh100-results"
BASE_TAG = "saded-fresh100-seed0-c5c35374"
POLL_SECONDS = 300
TERMINAL_SETTLE_SECONDS = 30


class ScientificValidationError(RuntimeError):
    """The downloaded scientific endpoint is not a valid frozen endpoint."""


class PublicationConflictError(RuntimeError):
    """Existing GitHub or worktree state conflicts with this publication."""


def _require_terminal_state(terminal_state: str) -> None:
    if terminal_state not in {"SUCCESS", "INVALID"}:
        raise ValueError("terminal_state must be SUCCESS or INVALID")


def classify_terminal_state(
    status: str | None,
    exit_code: str | None,
) -> str | None:
    """Map remote driver state to a fail-closed publication state."""

    if status == "TRAIN_COMPLETE" and exit_code == "0":
        return "SUCCESS_CANDIDATE"
    if status == "TRAIN_INVALID":
        return "INVALID"
    if exit_code not in (None, "", "0"):
        return "INVALID"
    return None


def build_terminal_manifest(
    *,
    run_id: str,
    terminal_state: str,
    exit_code: str | None,
    artifacts: Mapping[str, str],
    validation_passed: bool,
) -> dict[str, Any]:
    """Build a deterministic manifest that cannot mislabel invalid evidence."""

    _require_terminal_state(terminal_state)
    if terminal_state == "SUCCESS" and not validation_passed:
        raise ValueError("SUCCESS evidence must be independently validated")
    if terminal_state == "INVALID" and validation_passed:
        raise ValueError("INVALID evidence cannot claim validation passed")
    return {
        "schema_version": "saded-fresh100-publication/v1",
        "run_id": run_id,
        "terminal_state": terminal_state,
        "exit_code": exit_code,
        "publish_as_success": terminal_state == "SUCCESS",
        "validation_passed": validation_passed,
        "artifacts": dict(sorted(artifacts.items())),
    }


def terminal_directory_name(terminal_state: str) -> str:
    """Return the isolated evidence directory for a terminal state."""

    _require_terminal_state(terminal_state)
    return "terminal" if terminal_state == "SUCCESS" else "invalid"


def release_tag_for_state(base_tag: str, terminal_state: str) -> str:
    """Return a tag whose spelling cannot hide invalid evidence."""

    _require_terminal_state(terminal_state)
    return base_tag if terminal_state == "SUCCESS" else f"{base_tag}-invalid"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_success_candidate(
    *,
    summary: Mapping[str, Any],
    checkpoint: Path,
    protocol: Path,
    expected_protocol_sha256: str,
    runtime_validator: Callable[[dict[str, Any]], list[str]],
) -> dict[str, Any]:
    """Validate downloaded bytes with the frozen runtime validator."""

    protocol_sha = sha256_file(protocol)
    if protocol_sha != expected_protocol_sha256.upper():
        raise ScientificValidationError("protocol manifest checksum drift")
    rebound = copy.deepcopy(dict(summary))
    checkpoint_record = rebound.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise ScientificValidationError(
            "runtime validation failed: checkpoint absent"
        )
    local_checkpoint = checkpoint.resolve().as_posix()
    checkpoint_record["path"] = local_checkpoint
    checkpoint_record["expected_path"] = local_checkpoint
    failures = runtime_validator(rebound)
    if failures:
        raise ScientificValidationError(
            f"runtime validation failed: {failures}"
        )
    return {
        "passed": True,
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def validate_summary_bindings(
    summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_source = protocol.get("runtime_source", {})
    data = protocol.get("data", {})
    initial_state = protocol.get("initial_state", {})
    if not all(
        isinstance(value, Mapping)
        for value in (runtime_source, data, initial_state)
    ):
        raise ScientificValidationError("protocol binding sections are invalid")
    expected = {
        "protocol_source_commit": runtime_source.get("commit"),
        "source_repo_bundle_sha256": runtime_source.get(
            "repo_bundle_sha256"
        ),
        "source_upstream_bundle_sha256": runtime_source.get(
            "upstream_bundle_sha256"
        ),
        "data_sha256": data.get("sha256"),
        "initial_state_sha256": initial_state.get("sha256"),
        "initial_state_common_fingerprint": initial_state.get(
            "common_fingerprint"
        ),
    }
    drift = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if value is None or summary.get(key) != value
    }
    if drift:
        raise ScientificValidationError(f"summary binding drift: {drift}")
    return {"passed": True, "bindings": expected}


def validate_terminal_facts(
    status: str | None,
    exit_code: str | None,
    expected_state: str,
) -> None:
    actual = classify_terminal_state(status, exit_code)
    if actual != expected_state:
        raise ScientificValidationError(
            "downloaded terminal facts changed: "
            f"expected={expected_state}, actual={actual}, "
            f"status={status!r}, exit_code={exit_code!r}"
        )


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def claim_immutable_record(
    path: Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    record = dict(value)
    payload = (
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PublicationConflictError(
                f"immutable record is unreadable: {path}"
            ) from error
        if existing != record:
            raise PublicationConflictError(
                f"immutable record conflicts: {path}"
            )
        return existing
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)
    return record


def load_json_object(
    path: Path,
    *,
    scientific: bool,
) -> dict[str, Any]:
    error_type = ScientificValidationError if scientific else RuntimeError
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise error_type(f"JSON is unreadable: {path}: {error}") from error
    if not isinstance(value, dict):
        raise error_type(f"expected a JSON object: {path}")
    return value


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")


def write_state(status: str, **extra: Any) -> None:
    atomic_json(
        STATE_PATH,
        {
            "schema_version": "saded-fresh100-publisher-state/v1",
            "status": status,
            "run_id": RUN_ID,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **extra,
        },
    )


def connect() -> paramiko.SSHClient:
    password = os.environ.get("SBR_FRESH100_SSH_PASSWORD")
    if not password:
        raise RuntimeError("SBR_FRESH100_SSH_PASSWORD is not set")
    if not KNOWN_HOSTS.is_file():
        raise RuntimeError(f"known_hosts is missing: {KNOWN_HOSTS}")
    client = paramiko.SSHClient()
    client.load_host_keys(str(KNOWN_HOSTS))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        HOST,
        username=USER,
        password=password,
        port=22,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def read_remote_text(
    client: paramiko.SSHClient,
    path: Path,
) -> str | None:
    sftp = client.open_sftp()
    try:
        with sftp.open(path.as_posix(), "r") as handle:
            value = handle.read()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return value.strip()
    except FileNotFoundError:
        return None
    finally:
        sftp.close()


def remote_process_identity(
    client: paramiko.SSHClient,
    pid: int,
) -> dict[str, Any] | None:
    command = (
        f"if test -r /proc/{pid}/stat; then "
        f"printf 'BOOT='; cat /proc/sys/kernel/random/boot_id; "
        f"printf 'START='; awk '{{print $22}}' /proc/{pid}/stat; "
        f"printf 'CMD='; tr '\\0' ' ' < /proc/{pid}/cmdline; "
        "printf '\\n'; fi"
    )
    _, stdout, stderr = client.exec_command(command, timeout=20)
    payload = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace").strip()
    if error:
        raise RuntimeError(f"remote process identity failed: {error}")
    if not payload.strip():
        return None
    records: dict[str, str] = {}
    for line in payload.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            records[key] = value.strip()
    if not {"BOOT", "START", "CMD"} <= set(records):
        raise RuntimeError("remote process identity is incomplete")
    return {
        "pid": pid,
        "boot_id": records["BOOT"],
        "start_ticks": records["START"],
        "command": records["CMD"],
    }


def download_stable(
    sftp: paramiko.SFTPClient,
    remote: Path,
    local: Path,
) -> dict[str, Any]:
    before = sftp.stat(remote.as_posix())
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_suffix(local.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    sftp.get(remote.as_posix(), str(temporary))
    after = sftp.stat(remote.as_posix())
    if (
        before.st_size != after.st_size
        or before.st_mtime != after.st_mtime
        or temporary.stat().st_size != after.st_size
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"remote artifact changed during download: {remote}")
    os.replace(temporary, local)
    return {
        "size": after.st_size,
        "mtime": after.st_mtime,
        "sha256": sha256_file(local),
    }


def try_download(
    sftp: paramiko.SFTPClient,
    remote: Path,
    local: Path,
) -> dict[str, Any] | None:
    try:
        return download_stable(sftp, remote, local)
    except FileNotFoundError:
        return None


def run(command: list[str], cwd: Path = PUBLISHER_ROOT) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_frozen_runtime_validator(summary: dict[str, Any]) -> list[str]:
    validator_commit = run(
        ["git", "rev-parse", "HEAD"],
        cwd=VALIDATOR_ROOT,
    )
    validator_status = run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=VALIDATOR_ROOT,
    )
    if (
        validator_commit != EXPECTED_SOURCE_COMMIT
        or validator_status
    ):
        raise PublicationConflictError(
            "frozen runtime validator checkout drift"
        )
    summary_path = LOCAL_STATE_ROOT / "runtime-summary-rebound.json"
    atomic_json(summary_path, summary)
    program = (
        "import json,sys;"
        "from pathlib import Path;"
        "from scripts.train_rtdetr_saded_stock import "
        "validate_runtime_summary;"
        "summary=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));"
        "print(json.dumps(validate_runtime_summary(summary)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(summary_path)],
        cwd=VALIDATOR_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise RuntimeError("frozen runtime validator returned invalid data")
    return value


def git_status() -> str:
    return run(["git", "status", "--porcelain", "--untracked-files=all"])


def dirty_paths_are_allowed(
    status: str,
    allowed_relative: str,
) -> bool:
    prefix = allowed_relative.rstrip("/") + "/"
    paths = [
        line[3:].strip()
        for line in status.splitlines()
        if line.strip()
    ]
    return bool(paths) and all(path.startswith(prefix) for path in paths)


def assert_publish_worktree_clean(
    *,
    allowed_relative: str | None = None,
) -> None:
    branch = run(["git", "branch", "--show-current"])
    if branch != BRANCH:
        raise PublicationConflictError(
            f"publisher branch drift: expected {BRANCH}, got {branch}"
        )
    dirty = git_status()
    if dirty and (
        allowed_relative is None
        or not dirty_paths_are_allowed(dirty, allowed_relative)
    ):
        raise PublicationConflictError(
            f"publisher worktree is not clean before terminal collection: {dirty}"
        )


def write_checksum_closure(directory: Path) -> dict[str, str]:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    records = {path.name: sha256_file(path) for path in files}
    (directory / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in records.items()),
        encoding="ascii",
        newline="\n",
    )
    return records


def verify_checksum_closure(directory: Path) -> dict[str, str]:
    checksum_path = directory / "checksums.sha256"
    if not checksum_path.is_file():
        raise PublicationConflictError("checksum closure is missing")
    records: dict[str, str] = {}
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PublicationConflictError(
            "checksum closure is unreadable"
        ) from error
    for line in lines:
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or Path(parts[1]).name != parts[1]
            or parts[1] in records
        ):
            raise PublicationConflictError("checksum closure syntax is invalid")
        records[parts[1]] = parts[0].upper()
    actual_names = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    }
    if actual_names != set(records):
        raise PublicationConflictError("checksum closure file set mismatch")
    for name, digest in records.items():
        if sha256_file(directory / name) != digest:
            raise PublicationConflictError(
                f"checksum closure mismatch: {name}"
            )
    return records


def prepare_terminal_directory(terminal_state: str) -> tuple[Path, bool]:
    ensure_no_opposite_terminal(EVIDENCE_ROOT, terminal_state)
    target = EVIDENCE_ROOT / terminal_directory_name(terminal_state)
    if target.is_dir():
        return target, False
    staging = EVIDENCE_ROOT / (
        f".{terminal_directory_name(terminal_state)}-staging"
    )
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging, True


def ensure_no_opposite_terminal(
    evidence_root: Path,
    terminal_state: str,
) -> None:
    _require_terminal_state(terminal_state)
    opposite = "invalid" if terminal_state == "SUCCESS" else "terminal"
    if (evidence_root / opposite).exists():
        raise PublicationConflictError(
            f"opposite terminal evidence exists: {opposite}"
        )


def discard_terminal_staging(
    evidence_root: Path,
    terminal_state: str,
) -> None:
    staging = evidence_root / (
        f".{terminal_directory_name(terminal_state)}-staging"
    )
    if staging.exists():
        shutil.rmtree(staging)


def finalize_terminal_directory(
    staging: Path,
    terminal_state: str,
) -> Path:
    target = EVIDENCE_ROOT / terminal_directory_name(terminal_state)
    if staging == target:
        return target
    if target.exists():
        raise PublicationConflictError(f"terminal evidence exists: {target}")
    os.replace(staging, target)
    return target


def commit_terminal_evidence(directory: Path, message: str) -> str:
    relative = directory.relative_to(PUBLISHER_ROOT).as_posix()
    run(["git", "add", "--", relative])
    staged = run(["git", "diff", "--cached", "--name-only"])
    if staged:
        names = staged.splitlines()
        if any(not name.startswith(relative + "/") for name in names):
            raise PublicationConflictError(
                f"unexpected staged paths: {names}"
            )
        run(["git", "commit", "-m", message])
    run(["git", "push", "origin", BRANCH])
    return run(["git", "rev-parse", "HEAD"])


def release_info(tag: str) -> dict[str, Any] | None:
    completed = subprocess.run(
        [
            "gh",
            "release",
            "view",
            tag,
            "--json",
            "url,isDraft,isPrerelease,targetCommitish,assets",
        ],
        cwd=PUBLISHER_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if is_release_not_found(completed.stderr):
            return None
        raise RuntimeError(
            f"GitHub release lookup failed: {completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("GitHub release lookup returned a non-object")
    return value


def is_release_not_found(stderr: str) -> bool:
    normalized = stderr.lower()
    return (
        "release not found" in normalized
        or "http 404" in normalized
        or "404 not found" in normalized
    )


def verify_release_assets(
    release: Mapping[str, Any],
    assets: list[Path],
) -> None:
    remote = {
        item["name"]: {
            "size": int(item["size"]),
            "digest": item.get("digest"),
            "state": item.get("state"),
        }
        for item in release.get("assets", [])
    }
    expected = {
        path.name: {
            "size": path.stat().st_size,
            "digest": f"sha256:{sha256_file(path).lower()}",
            "state": "uploaded",
        }
        for path in assets
    }
    if set(remote) != set(expected) or any(
        name not in remote or remote[name] != record
        for name, record in expected.items()
    ):
        raise PublicationConflictError(
            f"release asset mismatch: expected={expected}, remote={remote}"
        )


def verify_release_identity(
    release: Mapping[str, Any],
    *,
    target_commit: str,
    prerelease: bool,
    allow_draft: bool,
) -> None:
    if release.get("targetCommitish") != target_commit:
        raise PublicationConflictError("release target commit mismatch")
    if release.get("isPrerelease") is not prerelease:
        raise PublicationConflictError("release prerelease state mismatch")
    if not allow_draft and release.get("isDraft") is not False:
        raise PublicationConflictError("release is still a draft")


def publish_release(
    *,
    tag: str,
    title: str,
    notes: str,
    assets: list[Path],
    prerelease: bool,
    target_commit: str,
) -> str:
    info = release_info(tag)
    if info is None:
        command = [
            "gh",
            "release",
            "create",
            tag,
            "--target",
            target_commit,
            "--title",
            title,
            "--notes",
            notes,
            "--draft",
        ]
        if prerelease:
            command.append("--prerelease")
        run(command)
        info = release_info(tag)
    if info is None:
        raise RuntimeError("draft release was not created")
    verify_release_identity(
        info,
        target_commit=target_commit,
        prerelease=prerelease,
        allow_draft=True,
    )
    if not info.get("isDraft"):
        verify_release_identity(
            info,
            target_commit=target_commit,
            prerelease=prerelease,
            allow_draft=False,
        )
        verify_release_assets(info, assets)
        return str(info["url"])
    existing = {
        item["name"]: int(item["size"])
        for item in info.get("assets", [])
    }
    for asset in assets:
        if asset.name in existing:
            if existing[asset.name] != asset.stat().st_size:
                raise PublicationConflictError(
                    f"existing draft asset conflicts: {asset.name}"
                )
            continue
        run(["gh", "release", "upload", tag, str(asset)])
    info = release_info(tag)
    if info is None:
        raise RuntimeError("draft release disappeared")
    verify_release_identity(
        info,
        target_commit=target_commit,
        prerelease=prerelease,
        allow_draft=True,
    )
    verify_release_assets(info, assets)
    run(["gh", "release", "edit", tag, "--draft=false"])
    final = release_info(tag)
    if final is None or final.get("isDraft"):
        raise RuntimeError("release did not leave draft state")
    verify_release_identity(
        final,
        target_commit=target_commit,
        prerelease=prerelease,
        allow_draft=False,
    )
    verify_release_assets(final, assets)
    return str(final["url"])


def collect_success_candidate(
    client: paramiko.SSHClient,
) -> tuple[Path, dict[str, Any], list[Path]]:
    discard_terminal_staging(EVIDENCE_ROOT, "SUCCESS")
    target = EVIDENCE_ROOT / terminal_directory_name("SUCCESS")
    allowed = (
        target.relative_to(PUBLISHER_ROOT).as_posix()
        if target.exists()
        else None
    )
    assert_publish_worktree_clean(allowed_relative=allowed)
    write_state("COLLECTING_SUCCESS_CANDIDATE")
    staging, should_collect = prepare_terminal_directory("SUCCESS")
    checkpoint = DOWNLOAD_ROOT / "saded-fresh100-seed0-last.pt"
    if should_collect:
        downloads = {
            "saded_stock_training_summary.json": (
                REMOTE_RUN_ROOT / "saded_stock_training_summary.json"
            ),
            "results.csv": REMOTE_RUN_ROOT / "results.csv",
            "args.yaml": REMOTE_RUN_ROOT / "args.yaml",
            "train.log": REMOTE_LOG_ROOT / "train.log",
            "protocol_manifest.json": REMOTE_PROTOCOL,
            "status": REMOTE_LOG_ROOT / "status",
            "exit_code": REMOTE_LOG_ROOT / "exit_code",
        }
        DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        transfer: dict[str, Any] = {}
        sftp = client.open_sftp()
        try:
            for name, remote in downloads.items():
                transfer[name] = download_stable(
                    sftp,
                    remote,
                    DOWNLOAD_ROOT / name,
                )
            transfer[checkpoint.name] = download_stable(
                sftp,
                REMOTE_RUN_ROOT / "weights" / "last.pt",
                checkpoint,
            )
        finally:
            sftp.close()
        validate_terminal_facts(
            (DOWNLOAD_ROOT / "status").read_text(
                encoding="utf-8"
            ).strip(),
            (DOWNLOAD_ROOT / "exit_code").read_text(
                encoding="utf-8"
            ).strip(),
            "SUCCESS_CANDIDATE",
        )
        summary = load_json_object(
            DOWNLOAD_ROOT / "saded_stock_training_summary.json",
            scientific=True,
        )
        if summary.get("protocol_source_commit") != EXPECTED_SOURCE_COMMIT:
            raise ScientificValidationError("source commit binding drift")
        protocol_payload = load_json_object(
            DOWNLOAD_ROOT / "protocol_manifest.json",
            scientific=True,
        )
        bindings = validate_summary_bindings(summary, protocol_payload)
        validation = validate_success_candidate(
            summary=summary,
            checkpoint=checkpoint,
            protocol=DOWNLOAD_ROOT / "protocol_manifest.json",
            expected_protocol_sha256=EXPECTED_PROTOCOL_SHA256,
            runtime_validator=run_frozen_runtime_validator,
        )
        for name in downloads:
            shutil.copy2(DOWNLOAD_ROOT / name, staging / name)
        artifacts = write_checksum_closure(staging)
        manifest = build_terminal_manifest(
            run_id=RUN_ID,
            terminal_state="SUCCESS",
            exit_code="0",
            artifacts=artifacts,
            validation_passed=True,
        )
        manifest.update(
            {
                "source_commit": EXPECTED_SOURCE_COMMIT,
                "protocol_sha256": validation["protocol_sha256"],
                "bindings": bindings,
                "checkpoint": {
                    "asset": checkpoint.name,
                    "sha256": validation["checkpoint_sha256"],
                    "size": checkpoint.stat().st_size,
                },
                "transfer": transfer,
            }
        )
        atomic_json(staging / "publication_manifest.json", manifest)
        write_checksum_closure(staging)
        staging = finalize_terminal_directory(staging, "SUCCESS")
    manifest = load_json_object(
        staging / "publication_manifest.json",
        scientific=False,
    )
    if (
        manifest.get("terminal_state") != "SUCCESS"
        or manifest.get("validation_passed") is not True
    ):
        raise PublicationConflictError("existing success evidence is invalid")
    closure = verify_checksum_closure(staging)
    recorded_artifacts = manifest.get("artifacts")
    closure_without_manifest = {
        name: digest
        for name, digest in closure.items()
        if name != "publication_manifest.json"
    }
    if recorded_artifacts != closure_without_manifest:
        raise PublicationConflictError(
            "success manifest does not match checksum closure"
        )
    checkpoint_record = manifest.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise PublicationConflictError("success checkpoint record is absent")
    expected_checkpoint_sha = checkpoint_record.get("sha256")
    expected_checkpoint_size = checkpoint_record.get("size")
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != expected_checkpoint_size
        or sha256_file(checkpoint) != expected_checkpoint_sha
    ):
        sftp = client.open_sftp()
        try:
            download_stable(
                sftp,
                REMOTE_RUN_ROOT / "weights" / "last.pt",
                checkpoint,
            )
        finally:
            sftp.close()
    if (
        checkpoint.stat().st_size != expected_checkpoint_size
        or sha256_file(checkpoint) != expected_checkpoint_sha
    ):
        raise PublicationConflictError(
            "local checkpoint does not match committed success manifest"
        )
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    for name in (
        "saded_stock_training_summary.json",
        "protocol_manifest.json",
    ):
        shutil.copy2(staging / name, DOWNLOAD_ROOT / name)
    validate_terminal_facts(
        (staging / "status").read_text(encoding="utf-8").strip(),
        (staging / "exit_code").read_text(encoding="utf-8").strip(),
        "SUCCESS_CANDIDATE",
    )
    rebound_summary = load_json_object(
        staging / "saded_stock_training_summary.json",
        scientific=True,
    )
    rebound_protocol = load_json_object(
        staging / "protocol_manifest.json",
        scientific=True,
    )
    validate_summary_bindings(rebound_summary, rebound_protocol)
    replay = validate_success_candidate(
        summary=rebound_summary,
        checkpoint=checkpoint,
        protocol=staging / "protocol_manifest.json",
        expected_protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        runtime_validator=run_frozen_runtime_validator,
    )
    if (
        replay["checkpoint_sha256"] != expected_checkpoint_sha
        or replay["protocol_sha256"] != manifest.get("protocol_sha256")
    ):
        raise PublicationConflictError(
            "success replay does not match committed manifest"
        )
    checksum_sidecar = DOWNLOAD_ROOT / f"{checkpoint.name}.sha256"
    checksum_sidecar.write_text(
        f"{sha256_file(checkpoint)}  {checkpoint.name}\n",
        encoding="ascii",
        newline="\n",
    )
    return (
        staging,
        manifest,
        [
            checkpoint,
            checksum_sidecar,
            DOWNLOAD_ROOT / "saded_stock_training_summary.json",
            DOWNLOAD_ROOT / "protocol_manifest.json",
        ],
    )


def collect_invalid(
    client: paramiko.SSHClient,
    *,
    exit_code: str | None,
    reason: str,
) -> tuple[Path, dict[str, Any], list[Path]]:
    discard_terminal_staging(EVIDENCE_ROOT, "INVALID")
    target = EVIDENCE_ROOT / terminal_directory_name("INVALID")
    allowed = (
        target.relative_to(PUBLISHER_ROOT).as_posix()
        if target.exists()
        else None
    )
    assert_publish_worktree_clean(allowed_relative=allowed)
    write_state("COLLECTING_INVALID", reason=reason, exit_code=exit_code)
    staging, should_collect = prepare_terminal_directory("INVALID")
    if should_collect:
        downloads = {
            "train.log": REMOTE_LOG_ROOT / "train.log",
            "driver.log": REMOTE_LOG_ROOT / "driver.log",
            "driver.sh": REMOTE_LOG_ROOT / "driver.sh",
            "status": REMOTE_LOG_ROOT / "status",
            "exit_code": REMOTE_LOG_ROOT / "exit_code",
            "protocol_manifest.json": REMOTE_PROTOCOL,
            "runtime_invalid.json": REMOTE_RUN_ROOT / "runtime_invalid.json",
        }
        DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        transfer: dict[str, Any] = {}
        sftp = client.open_sftp()
        try:
            for name, remote in downloads.items():
                record = try_download(sftp, remote, DOWNLOAD_ROOT / name)
                if record is not None:
                    transfer[name] = record
                    shutil.copy2(DOWNLOAD_ROOT / name, staging / name)
        finally:
            sftp.close()
        artifacts = write_checksum_closure(staging)
        manifest = build_terminal_manifest(
            run_id=RUN_ID,
            terminal_state="INVALID",
            exit_code=exit_code,
            artifacts=artifacts,
            validation_passed=False,
        )
        manifest.update(
            {
                "reason": reason,
                "source_commit": EXPECTED_SOURCE_COMMIT,
                "checkpoint_uploaded": False,
                "transfer": transfer,
            }
        )
        atomic_json(staging / "publication_manifest.json", manifest)
        write_checksum_closure(staging)
        staging = finalize_terminal_directory(staging, "INVALID")
    manifest = load_json_object(
        staging / "publication_manifest.json",
        scientific=False,
    )
    if (
        manifest.get("terminal_state") != "INVALID"
        or manifest.get("publish_as_success") is not False
    ):
        raise PublicationConflictError("existing invalid evidence is unsafe")
    verify_checksum_closure(staging)
    assets = [
        path
        for name in ("train.log", "runtime_invalid.json", "publication_manifest.json")
        if (path := staging / name).is_file()
    ]
    return staging, manifest, assets


def publish_success(client: paramiko.SSHClient) -> None:
    directory, manifest, assets = collect_success_candidate(client)
    claim_immutable_record(
        LOCAL_STATE_ROOT / "scientific_decision.json",
        {
            "terminal_state": "SUCCESS",
            "validation_passed": True,
            "manifest_sha256": sha256_file(
                directory / "publication_manifest.json"
            ),
        },
    )
    write_state(
        "SUCCESS_VALIDATED",
        checkpoint_sha256=manifest["checkpoint"]["sha256"],
    )
    commit = commit_terminal_evidence(
        directory,
        "final: publish validated fresh100 seed0 endpoint",
    )
    url = publish_release(
        tag=release_tag_for_state(BASE_TAG, "SUCCESS"),
        title="SADED fresh seed0 RT-DETR-L 100 epochs",
        notes=(
            "Validated Fresh-100 seed-0 endpoint. Scientific validation "
            f"passed before publication. Evidence commit: {commit}."
        ),
        assets=assets,
        prerelease=False,
        target_commit=commit,
    )
    write_state(
        "PUBLISHED",
        terminal_state="SUCCESS",
        branch=BRANCH,
        commit=commit,
        release_url=url,
    )
    log(f"published validated success: {url}")


def publish_invalid(
    client: paramiko.SSHClient,
    *,
    exit_code: str | None,
    reason: str,
) -> None:
    directory, _, assets = collect_invalid(
        client,
        exit_code=exit_code,
        reason=reason,
    )
    claim_immutable_record(
        LOCAL_STATE_ROOT / "scientific_decision.json",
        {
            "terminal_state": "INVALID",
            "validation_passed": False,
            "manifest_sha256": sha256_file(
                directory / "publication_manifest.json"
            ),
        },
    )
    commit = commit_terminal_evidence(
        directory,
        "evidence: publish invalid fresh100 run",
    )
    url = publish_release(
        tag=release_tag_for_state(BASE_TAG, "INVALID"),
        title="[INVALID] SADED fresh seed0 run",
        notes=(
            "This prerelease contains failure diagnostics only. It is not a "
            f"successful endpoint. Evidence commit: {commit}."
        ),
        assets=assets,
        prerelease=True,
        target_commit=commit,
    )
    write_state(
        "PUBLISHED_INVALID",
        terminal_state="INVALID",
        branch=BRANCH,
        commit=commit,
        release_url=url,
        reason=reason,
    )
    log(f"published invalid evidence: {url}")


def process_is_current(
    actual: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> bool:
    return (
        actual is not None
        and actual.get("pid") == expected.get("pid")
        and actual.get("boot_id") == expected.get("boot_id")
        and actual.get("start_ticks") == expected.get("start_ticks")
        and actual.get("command") == expected.get("command")
    )


def validate_process_baseline(
    baseline: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_pids: set[int],
) -> dict[int, dict[str, Any]]:
    records = baseline.get("processes")
    if (
        baseline.get("run_id") != expected_run_id
        or not isinstance(records, Mapping)
    ):
        raise PublicationConflictError("persisted process baseline is invalid")
    parsed: dict[int, dict[str, Any]] = {}
    required = {"pid", "boot_id", "start_ticks", "command"}
    for raw_pid, identity in records.items():
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError) as error:
            raise PublicationConflictError(
                "persisted process baseline PID is invalid"
            ) from error
        if (
            not isinstance(identity, Mapping)
            or not required <= set(identity)
            or identity.get("pid") != pid
        ):
            raise PublicationConflictError(
                "persisted process baseline identity is invalid"
            )
        parsed[pid] = dict(identity)
    if set(parsed) != expected_pids:
        raise PublicationConflictError(
            "persisted process baseline PID set is invalid"
        )
    return parsed


def classify_lock_owner(
    record: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_script_sha256: str,
) -> str:
    try:
        pid = int(record["pid"])
        recorded_create_time = float(record["create_time"])
        process = psutil.Process(pid)
        alive = (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and abs(process.create_time() - recorded_create_time) < 0.001
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        psutil.Error,
    ):
        return "STALE"
    if not alive:
        return "STALE"
    if (
        record.get("run_id") != expected_run_id
        or record.get("script_sha256") != expected_script_sha256
    ):
        return "CONFLICT"
    return "MATCH"


def acquire_lock() -> int | None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    script_sha = sha256_file(Path(__file__).resolve())
    process = psutil.Process(os.getpid())
    record = {
        "schema_version": "saded-fresh100-publisher-lock/v1",
        "pid": process.pid,
        "create_time": process.create_time(),
        "run_id": RUN_ID,
        "script_sha256": script_sha,
        "script_path": Path(__file__).resolve().as_posix(),
    }
    try:
        descriptor = os.open(
            LOCK_PATH,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError:
        try:
            existing = json.loads(
                LOCK_PATH.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            existing = {}
        lock_state = classify_lock_owner(
            existing,
            expected_run_id=RUN_ID,
            expected_script_sha256=script_sha,
        )
        if lock_state == "MATCH":
            return None
        if lock_state == "CONFLICT":
            raise PublicationConflictError(
                "live publisher lock belongs to a different identity"
            )
        try:
            LOCK_PATH.unlink(missing_ok=True)
            descriptor = os.open(
                LOCK_PATH,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except OSError as error:
            raise RuntimeError("failed to replace stale publisher lock") from error
    os.write(
        descriptor,
        (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"),
    )
    os.close(descriptor)
    return descriptor


def main() -> None:
    if acquire_lock() is None:
        return
    write_state("STARTING", poll_seconds=POLL_SECONDS)
    log("publisher started")
    expected_processes: dict[int, dict[str, Any]] = {}
    while True:
        try:
            client = connect()
            try:
                status = read_remote_text(
                    client,
                    REMOTE_LOG_ROOT / "status",
                )
                exit_code = read_remote_text(
                    client,
                    REMOTE_LOG_ROOT / "exit_code",
                )
                terminal = classify_terminal_state(status, exit_code)
                if terminal is None and not expected_processes:
                    baseline_path = (
                        LOCAL_STATE_ROOT / "process_baseline.json"
                    )
                    if baseline_path.is_file():
                        baseline = load_json_object(
                            baseline_path,
                            scientific=False,
                        )
                        expected_processes = validate_process_baseline(
                            baseline,
                            expected_run_id=RUN_ID,
                            expected_pids={DRIVER_PID, TRAINER_PID},
                        )
                    else:
                        captured: dict[int, dict[str, Any]] = {}
                        for pid in (DRIVER_PID, TRAINER_PID):
                            identity = remote_process_identity(
                                client,
                                pid,
                            )
                            if identity is not None:
                                captured[pid] = identity
                        if set(captured) != {DRIVER_PID, TRAINER_PID}:
                            missing_initial = sorted(
                                {DRIVER_PID, TRAINER_PID} - set(captured)
                            )
                            claim_immutable_record(
                                LOCAL_STATE_ROOT
                                / "terminal_observation.json",
                                {
                                    "status": status,
                                    "exit_code": exit_code,
                                    "classified_state": (
                                        "INVALID_MISSING_BASELINE"
                                    ),
                                    "missing_pids": missing_initial,
                                },
                            )
                            publish_invalid(
                                client,
                                exit_code=exit_code,
                                reason=(
                                    "RUNNING state lacks the complete original "
                                    f"process identity baseline: {missing_initial}"
                                ),
                            )
                            return
                        baseline = claim_immutable_record(
                            baseline_path,
                            {
                                "run_id": RUN_ID,
                                "processes": {
                                    str(pid): identity
                                    for pid, identity in captured.items()
                                },
                            },
                        )
                        expected_processes = validate_process_baseline(
                            baseline,
                            expected_run_id=RUN_ID,
                            expected_pids={DRIVER_PID, TRAINER_PID},
                        )
                if terminal == "SUCCESS_CANDIDATE":
                    claim_immutable_record(
                        LOCAL_STATE_ROOT / "terminal_observation.json",
                        {
                            "status": status,
                            "exit_code": exit_code,
                            "classified_state": terminal,
                        },
                    )
                    write_state("TERMINAL_OBSERVED_SUCCESS_CANDIDATE")
                    time.sleep(TERMINAL_SETTLE_SECONDS)
                    try:
                        publish_success(client)
                    except ScientificValidationError as error:
                        discard_terminal_staging(
                            EVIDENCE_ROOT,
                            "SUCCESS",
                        )
                        publish_invalid(
                            client,
                            exit_code=exit_code,
                            reason=f"success candidate rejected: {error}",
                        )
                    return
                if terminal == "INVALID":
                    claim_immutable_record(
                        LOCAL_STATE_ROOT / "terminal_observation.json",
                        {
                            "status": status,
                            "exit_code": exit_code,
                            "classified_state": terminal,
                        },
                    )
                    publish_invalid(
                        client,
                        exit_code=exit_code,
                        reason=(
                            f"driver terminal state status={status!r}, "
                            f"exit_code={exit_code!r}"
                        ),
                    )
                    return
                missing = []
                for pid, expected in expected_processes.items():
                    actual = remote_process_identity(client, pid)
                    if not process_is_current(actual, expected):
                        missing.append(pid)
                if expected_processes and missing:
                    claim_immutable_record(
                        LOCAL_STATE_ROOT / "terminal_observation.json",
                        {
                            "status": status,
                            "exit_code": exit_code,
                            "classified_state": "INVALID_MISSING_RECORD",
                            "missing_pids": missing,
                        },
                    )
                    publish_invalid(
                        client,
                        exit_code=exit_code,
                        reason=(
                            "training process identity disappeared before "
                            f"an atomic terminal record: {missing}"
                        ),
                    )
                    return
                write_state(
                    "WATCHING",
                    remote_status=status,
                    poll_seconds=POLL_SECONDS,
                    observed_pids=sorted(expected_processes),
                )
            finally:
                client.close()
        except PublicationConflictError:
            raise
        except Exception as error:
            log(f"retryable error: {type(error).__name__}: {error}")
            write_state(
                "RETRYING",
                error_type=type(error).__name__,
                error=str(error),
            )
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(traceback.format_exc())
        write_state(
            "FAILED",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
