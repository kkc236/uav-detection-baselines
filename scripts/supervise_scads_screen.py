"""Supervise the immutable 30-epoch FDR then SCADS screen sequence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_EPOCHS = 30


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    rows = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {number} is not an object: {source}")
        rows.append(value)
    return rows


def write_status(path: Path, *, state: str, **details: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def process_alive(pid_file: Path) -> bool:
    try:
        pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False
    return True


def arm_evidence(run: Path, variant: str) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(run) / "scads-epochs.jsonl")
    epochs = [int(row.get("completed_epoch", -1)) for row in rows]
    if len(rows) > EXPECTED_EPOCHS:
        raise ValueError(f"{variant} has more than {EXPECTED_EPOCHS} evidence rows")
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f"{variant} epoch evidence is not continuous: {epochs}")
    if any(row.get("variant") != variant or row.get("stage") != "screen" for row in rows):
        raise ValueError(f"{variant} evidence authority mismatch")
    return rows


def verified_epochs(ledger: Path, *, run_id: str) -> list[int]:
    epochs = sorted(
        int(row["completed_epoch"])
        for row in read_jsonl(ledger)
        if row.get("run_id") == run_id and row.get("status") == "published-verified"
    )
    if len(epochs) != len(set(epochs)):
        raise ValueError(f"duplicate verified ledger epoch for {run_id}")
    return epochs


def scads_command(args: argparse.Namespace) -> list[str]:
    experiment = args.experiment_root.resolve()
    return [
        str(args.python.resolve()),
        "-u",
        str((args.training_root / "scripts" / "train_rtdetr_scads.py").resolve()),
        "--variant",
        "scads",
        "--stage",
        "screen",
        "--protocol-manifest",
        str((experiment / "authority" / "scads-fdr-protocol.json").resolve()),
        "--initial-state",
        str((experiment / "authority" / "scads-fdr-seed0.pt").resolve()),
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--output-root",
        str((experiment / "runs").resolve()),
        "--publication-queue",
        str((experiment / "publication-queue.jsonl").resolve()),
        "--name",
        "screen-seed0-scads-scads-v1",
    ]


def _wait_for_publication(
    *,
    args: argparse.Namespace,
    run: Path,
    variant: str,
    run_id: str,
) -> None:
    deadline = time.monotonic() + args.publication_timeout
    expected = list(range(1, EXPECTED_EPOCHS + 1))
    while True:
        epochs = verified_epochs(args.ledger, run_id=run_id)
        if epochs == expected:
            return
        if not process_alive(args.publisher_pid):
            raise RuntimeError("checkpoint publisher stopped before the ledger was complete")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"publication timeout for {variant}: {epochs}")
        write_status(
            args.status_file,
            state=f"waiting-{variant}-publication",
            run=str(run),
            evidence_epochs=EXPECTED_EPOCHS,
            verified_epochs=len(epochs),
        )
        time.sleep(args.interval)


def _validate_complete_arm(run: Path, variant: str) -> tuple[list[dict[str, Any]], str]:
    rows = arm_evidence(run, variant)
    if len(rows) != EXPECTED_EPOCHS:
        raise RuntimeError(f"{variant} stopped with {len(rows)} of {EXPECTED_EPOCHS} epochs")
    results = run / "results.csv"
    if not results.is_file() or len(results.read_text(encoding="utf-8").splitlines()) != EXPECTED_EPOCHS + 1:
        raise RuntimeError(f"{variant} results.csv is incomplete")
    run_ids = {str(row.get("run_id", "")) for row in rows}
    if len(run_ids) != 1 or not next(iter(run_ids)):
        raise ValueError(f"{variant} evidence has no unique run_id")
    return rows, next(iter(run_ids))


def execute(args: argparse.Namespace) -> None:
    for name in (
        "training_root",
        "experiment_root",
        "dataset_root",
        "python",
        "fdr_pid",
        "scads_pid",
        "publisher_pid",
        "ledger",
        "status_file",
        "log",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve())
    fdr_run = args.experiment_root / "runs" / "screen-seed0-fdr-scads-v1"
    scads_run = args.experiment_root / "runs" / "screen-seed0-scads-scads-v1"
    while process_alive(args.fdr_pid):
        write_status(
            args.status_file,
            state="training-fdr",
            completed_epochs=len(arm_evidence(fdr_run, "fdr")),
        )
        time.sleep(args.interval)
    _fdr_rows, fdr_run_id = _validate_complete_arm(fdr_run, "fdr")
    _wait_for_publication(
        args=args, run=fdr_run, variant="fdr", run_id=fdr_run_id
    )

    if scads_run.exists():
        if process_alive(args.scads_pid):
            while process_alive(args.scads_pid):
                write_status(
                    args.status_file,
                    state="training-scads",
                    completed_epochs=len(arm_evidence(scads_run, "scads")),
                )
                time.sleep(args.interval)
        else:
            _validate_complete_arm(scads_run, "scads")
    else:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(
                scads_command(args),
                cwd=args.training_root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            args.scads_pid.write_text(f"{process.pid}\n", encoding="utf-8")
            write_status(
                args.status_file,
                state="training-scads",
                pid=process.pid,
                completed_epochs=0,
            )
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"SCADS training exited with status {return_code}")

    _scads_rows, scads_run_id = _validate_complete_arm(scads_run, "scads")
    _wait_for_publication(
        args=args, run=scads_run, variant="scads", run_id=scads_run_id
    )
    write_status(
        args.status_file,
        state="both-arms-published",
        fdr_epochs=EXPECTED_EPOCHS,
        scads_epochs=EXPECTED_EPOCHS,
        verified_epochs=2 * EXPECTED_EPOCHS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--fdr-pid", type=Path, required=True)
    parser.add_argument("--scads-pid", type=Path, required=True)
    parser.add_argument("--publisher-pid", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--publication-timeout", type=int, default=7200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        execute(args)
    except Exception as error:
        write_status(
            Path(args.status_file).resolve(),
            state="failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
