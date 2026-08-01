"""Orchestrate resumable seed0 control/LPR-G screening and formal training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import (
    prune_local_epoch_checkpoints,
    validate_token_file,
)
from src.github_checkpoint_sync import checkpoint_metadata, github_session
from src.lpr_g_publication import (
    PublicationConfig,
    PublicationLedger,
    pending_epoch_checkpoints,
    publish_with_retry,
)


SCREEN_ORDER = ((0, "control"), (0, "lprg"))
FORMAL_ORDER = SCREEN_ORDER
RELEASE_TAG = "lpr-g-v2-live"


def normalize_python_executable(path: Path) -> Path:
    """Make an executable path absolute without dereferencing a venv symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def run_name(stage: str, variant: str) -> str:
    return f"{stage}-seed0-{variant}-lpr-g-v2"


def asset_prefix(stage: str, variant: str, *, preflight: bool = False) -> str:
    prefix = f"{stage}-seed0-{variant}-lpr-g-v2"
    return f"{prefix}-preflight" if preflight else prefix


def build_arm_command(
    *,
    python: Path,
    protocol: Path,
    initial_state: Path,
    project: Path,
    stage: str,
    variant: str,
    token_file: Path = Path("/data/uav/secrets/github_token"),
    repo: str = "kkc236/uav-detection-baselines",
    repo_url: str = "https://github.com/kkc236/uav-detection-baselines.git",
    source_branch: str = "codex/lpr-rtdetr",
    results_branch: str = "training-results",
    results_repo: Path | None = None,
    resume: Path | None = None,
    preflight: bool = False,
) -> list[str]:
    """Build an operational command with no scientific hyperparameter switches."""
    if stage not in {"screen", "formal"} or variant not in {"control", "lprg"}:
        raise ValueError(f"invalid paired arm: stage={stage}, variant={variant}")
    results_repo = results_repo or project.parent / "uav-training-results-lpr-g"
    command = [
        str(python),
        "scripts/train_rtdetr_lpr_g.py",
        "--variant",
        variant,
        "--stage",
        stage,
        "--seed",
        "0",
        "--protocol-manifest",
        str(protocol),
        "--initial-state",
        str(initial_state),
        "--project",
        str(project),
        "--name",
        run_name(stage, variant),
        "--token-file",
        str(token_file),
        "--repo",
        repo,
        "--repo-url",
        repo_url,
        "--tag",
        RELEASE_TAG,
        "--source-branch",
        source_branch,
        "--results-branch",
        results_branch,
        "--results-repo",
        str(results_repo),
        "--asset-prefix",
        asset_prefix(stage, variant),
        "--retain",
        "3",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    if preflight:
        command.append("--preflight")
    return command


def next_stage(screen_report: dict[str, Any], *, through_formal: bool) -> str:
    status = screen_report.get("status")
    if status == "engineering_invalid":
        return "repair"
    if status == "passed" and through_formal:
        return "formal"
    return "stop"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _transition(path: Path, state: dict[str, Any], status: str, **details: Any) -> None:
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "status": status,
        **details,
    }
    state.setdefault("history", []).append(record)
    state.update(record)
    _atomic_json(path, state)


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + current if current else "")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("CUDA_MODULE_LOADING", "LAZY")
    return environment


def _run_logged(command: list[str], *, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\nCOMMAND {json.dumps(command)}\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=_child_environment(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"child command failed with exit code {result.returncode}; see {log}"
        )


def _expected_epochs(stage: str, *, preflight: bool = False) -> int:
    if preflight:
        return 1
    return 50 if stage == "screen" else 100


def _publication_config(
    args: argparse.Namespace,
    *,
    stage: str,
    variant: str,
    preflight: bool,
) -> PublicationConfig:
    name = run_name(stage, variant) + ("-preflight" if preflight else "")
    return PublicationConfig(
        repo=args.repo,
        repo_url=args.repo_url,
        source_branch=args.source_branch,
        results_branch=args.results_branch,
        tag=RELEASE_TAG,
        asset_prefix=asset_prefix(stage, variant, preflight=preflight),
        run_name=name,
        token_file=args.token_file,
        results_repo=args.results_repo,
        variant=variant,
        stage=stage,
        retain=3,
    )


def _arm_run_dir(
    project: Path,
    *,
    stage: str,
    variant: str,
    preflight: bool,
) -> Path:
    name = run_name(stage, variant) + ("-preflight" if preflight else "")
    return project / name


def _verified_local_checkpoint(run_dir: Path, ledger: PublicationLedger) -> Path | None:
    rows = ledger.records()
    if not rows:
        return None
    record = rows[-1]
    completed_epoch = int(record["completed_epoch"])
    expected_sha = str(record["checkpoint"]["sha256"])
    candidates = (
        run_dir / "weights" / f"epoch{completed_epoch - 1}.pt",
        run_dir / "weights" / f"restored-epoch{completed_epoch - 1}.pt",
        run_dir / "weights" / "last.pt",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            metadata = checkpoint_metadata(candidate)
        except Exception:
            continue
        if metadata.completed_epoch == completed_epoch and metadata.sha256 == expected_sha:
            return candidate.resolve()
    return None


def _restore_verified_checkpoint(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    stage: str,
    variant: str,
    preflight: bool,
    log: Path,
) -> Path:
    command = [
        str(args.python),
        "scripts/restore_lpr_g_checkpoint.py",
        "--variant",
        variant,
        "--stage",
        stage,
        "--run-dir",
        str(run_dir),
        "--token-file",
        str(args.token_file),
        "--repo",
        args.repo,
        "--tag",
        RELEASE_TAG,
        "--asset-prefix",
        asset_prefix(stage, variant, preflight=preflight),
    ]
    _run_logged(command, log=log)
    ledger = PublicationLedger(run_dir / "publication-ledger.jsonl")
    restored = _verified_local_checkpoint(run_dir, ledger)
    if restored is None:
        raise RuntimeError("restored checkpoint does not match the verified publication ledger")
    return restored


def _repair_publication(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    stage: str,
    variant: str,
    preflight: bool,
) -> None:
    if not run_dir.is_dir():
        return
    ledger = PublicationLedger(run_dir / "publication-ledger.jsonl")
    for _, checkpoint in pending_epoch_checkpoints(run_dir / "weights", ledger):
        publish_with_retry(
            run_dir,
            checkpoint,
            _publication_config(
                args,
                stage=stage,
                variant=variant,
                preflight=preflight,
            ),
        )
        prune_local_epoch_checkpoints(run_dir / "weights", retain=3)


def _validate_existing_runtime(run_dir: Path, *, stage: str, variant: str) -> None:
    if not run_dir.exists():
        return
    runtime_path = run_dir / "lpr_g_protocol.json"
    if not runtime_path.is_file():
        if any(run_dir.iterdir()):
            raise RuntimeError(f"existing run has no LPR-G runtime authority: {run_dir}")
        return
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    expected = {"variant": variant, "stage": stage, "seed": 0}
    actual = {key: runtime.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"existing run belongs to another arm: expected={expected}, actual={actual}"
        )


def _arm_complete(run_dir: Path, *, expected_epochs: int) -> bool:
    if not run_dir.is_dir():
        return False
    rows = PublicationLedger(run_dir / "publication-ledger.jsonl").records()
    return len(rows) == expected_epochs and int(rows[-1]["completed_epoch"]) == expected_epochs


def run_arm(
    args: argparse.Namespace,
    state_path: Path,
    state: dict[str, Any],
    *,
    protocol: Path,
    initial_state: Path,
    stage: str,
    variant: str,
    preflight: bool = False,
) -> Path:
    expected_epochs = _expected_epochs(stage, preflight=preflight)
    run_dir = _arm_run_dir(
        args.project,
        stage=stage,
        variant=variant,
        preflight=preflight,
    )
    _validate_existing_runtime(run_dir, stage=stage, variant=variant)
    _repair_publication(
        args,
        run_dir=run_dir,
        stage=stage,
        variant=variant,
        preflight=preflight,
    )
    if _arm_complete(run_dir, expected_epochs=expected_epochs):
        ledger = PublicationLedger(run_dir / "publication-ledger.jsonl")
        checkpoint = _verified_local_checkpoint(run_dir, ledger)
        if checkpoint is None:
            checkpoint = _restore_verified_checkpoint(
                args,
                run_dir=run_dir,
                stage=stage,
                variant=variant,
                preflight=preflight,
                log=args.project / "logs" / f"restore-{run_dir.name}.log",
            )
        _transition(
            state_path,
            state,
            "arm_verified",
            stage=stage,
            variant=variant,
            completed_epoch=expected_epochs,
        )
        return checkpoint

    if preflight and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"incomplete preflight is preserved for repair: {run_dir}")
    ledger = PublicationLedger(run_dir / "publication-ledger.jsonl")
    resume = _verified_local_checkpoint(run_dir, ledger)
    if ledger.last_completed_epoch and resume is None:
        resume = _restore_verified_checkpoint(
            args,
            run_dir=run_dir,
            stage=stage,
            variant=variant,
            preflight=preflight,
            log=args.project / "logs" / f"restore-{run_dir.name}.log",
        )

    _transition(
        state_path,
        state,
        "arm_running",
        stage=stage,
        variant=variant,
        resume=str(resume) if resume is not None else None,
    )
    command = build_arm_command(
        python=args.python,
        protocol=protocol,
        initial_state=initial_state,
        project=args.project,
        stage=stage,
        variant=variant,
        token_file=args.token_file,
        repo=args.repo,
        repo_url=args.repo_url,
        source_branch=args.source_branch,
        results_branch=args.results_branch,
        results_repo=args.results_repo,
        resume=resume,
        preflight=preflight,
    )
    log = args.project / "logs" / f"{run_dir.name}.log"
    try:
        _run_logged(command, log=log)
    except RuntimeError:
        _repair_publication(
            args,
            run_dir=run_dir,
            stage=stage,
            variant=variant,
            preflight=preflight,
        )
        _transition(
            state_path,
            state,
            "arm_interrupted",
            stage=stage,
            variant=variant,
            run_dir=str(run_dir),
        )
        raise
    _repair_publication(
        args,
        run_dir=run_dir,
        stage=stage,
        variant=variant,
        preflight=preflight,
    )
    if not _arm_complete(run_dir, expected_epochs=expected_epochs):
        raise RuntimeError(f"arm exited without complete verified evidence: {run_dir}")
    ledger = PublicationLedger(run_dir / "publication-ledger.jsonl")
    checkpoint = _verified_local_checkpoint(run_dir, ledger)
    if checkpoint is None:
        raise RuntimeError(f"completed arm has no matching local verified checkpoint: {run_dir}")
    _transition(
        state_path,
        state,
        "arm_complete",
        stage=stage,
        variant=variant,
        completed_epoch=expected_epochs,
    )
    return checkpoint


def run_model_canary(initial_state: Path) -> dict[str, Any]:
    """Check exact initial stock outputs/losses/common gradients on the target GPU."""
    import torch
    from ultralytics.nn.tasks import RTDETRDetectionModel

    from src.lpr_g_protocol import load_lpr_g_initial_state
    from src.rtdetr_lpr_g import LPRGRTDETRDetectionModel

    if not torch.cuda.is_available():
        raise RuntimeError("LPR-G model canary requires CUDA")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    artifact = torch.load(initial_state, map_location="cpu", weights_only=False)
    control = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    method = LPRGRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    load_lpr_g_initial_state(control, artifact, variant="control")
    load_lpr_g_initial_state(method, artifact, variant="lprg")
    control, method = control.to(device), method.to(device)
    generator = torch.Generator().manual_seed(123)
    image = torch.rand(1, 3, 160, 160, generator=generator).to(device)

    control.eval()
    method.eval()
    method.set_refinement_output("refined")
    with torch.inference_mode():
        stock_output = control.predict(image)[0]
        method_output = method.predict(image)[0]
    torch.testing.assert_close(method_output, stock_output, rtol=0, atol=0)

    batch = {
        "img": image,
        "cls": torch.tensor([[1.0]], device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
        "batch_idx": torch.tensor([0.0], device=device),
    }
    control.train()
    method.train()
    torch.manual_seed(77)
    torch.cuda.manual_seed_all(77)
    control_total, control_items = control.loss(batch)
    torch.manual_seed(77)
    torch.cuda.manual_seed_all(77)
    method_total, method_items = method.loss(batch)
    torch.testing.assert_close(method_items, control_items, rtol=0, atol=0)
    control_total.backward()
    method_total.backward()
    method_parameters = dict(method.named_parameters())
    common_gradient_max_abs = 0.0
    for name, parameter in control.named_parameters():
        method_gradient = method_parameters[name].grad
        if parameter.grad is None or method_gradient is None:
            if parameter.grad is not None or method_gradient is not None:
                raise RuntimeError(f"canary common gradient presence differs: {name}")
            continue
        difference = float((parameter.grad - method_gradient).abs().max().detach().cpu())
        common_gradient_max_abs = max(common_gradient_max_abs, difference)
        if difference != 0.0:
            raise RuntimeError(f"canary common gradient differs for {name}: {difference}")
    report = {
        "stock_output_exact": True,
        "stock_loss_items_exact": True,
        "common_gradients_exact": True,
        "common_gradient_max_abs": common_gradient_max_abs,
        "private_loss": float(method.last_lpr_g_loss_total.detach().float().cpu()),
    }
    del control, method, image, batch, control_total, method_total
    torch.cuda.empty_cache()
    return report


def _verify_preflight_pair(project: Path) -> None:
    rows = []
    for variant in ("control", "lprg"):
        path = (
            project
            / f"screen-seed0-{variant}-lpr-g-v2-preflight"
            / "common_state_audit.jsonl"
        )
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != 1:
            raise RuntimeError(f"preflight audit must have exactly one epoch: {path}")
        rows.append(records[0])
    fields = ("common_model_sha256", "common_optimizer_sha256")
    if any(rows[0][field] != rows[1][field] for field in fields):
        raise RuntimeError("preflight common model/optimizer fingerprints differ")


def _prepare_protocol(args: argparse.Namespace, *, log: Path) -> tuple[Path, Path]:
    protocol = args.protocol_dir / "protocol-seed0.json"
    initial_state = args.protocol_dir / "initial-state-seed0.pt"
    command = [
        str(args.python),
        "scripts/prepare_lpr_g_protocol.py",
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(args.protocol_dir),
        "--seed",
        "0",
    ]
    _run_logged(command, log=log)
    if not protocol.is_file() or not initial_state.is_file():
        raise RuntimeError("protocol preparation did not create seed0 authority artifacts")
    return protocol, initial_state


def _run_evaluation(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    stage: str,
    output: Path,
) -> dict:
    if not output.exists():
        command = [
            str(args.python),
            "scripts/evaluate_lpr_g.py",
            "--checkpoint",
            str(checkpoint),
            "--stage",
            stage,
            "--output",
            str(output),
        ]
        _run_logged(command, log=args.project / "logs" / f"evaluate-{stage}.log")
    return json.loads(output.read_text(encoding="utf-8"))


def _run_benchmark(
    args: argparse.Namespace,
    *,
    initial_state: Path,
    output: Path,
) -> dict:
    if not output.exists():
        _run_logged(
            [
                str(args.python),
                "scripts/benchmark_lpr_g.py",
                "--initial-state",
                str(initial_state),
                "--output",
                str(output),
            ],
            log=args.project / "logs" / "benchmark.log",
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _run_comparison(
    args: argparse.Namespace,
    *,
    stage: str,
    evaluation: Path,
    benchmark: Path,
    output: Path,
) -> dict:
    if not output.exists():
        _run_logged(
            [
                str(args.python),
                "scripts/compare_lpr_g.py",
                "--control-run",
                str(args.project / run_name(stage, "control")),
                "--method-run",
                str(args.project / run_name(stage, "lprg")),
                "--ablation",
                str(evaluation),
                "--benchmark",
                str(benchmark),
                "--stage",
                stage,
                "--output",
                str(output),
            ],
            log=args.project / "logs" / f"compare-{stage}.log",
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _audit_github(args: argparse.Namespace) -> None:
    token = validate_token_file(args.token_file)
    session = github_session(token)
    response = session.get(f"https://api.github.com/repos/{args.repo}", timeout=30)
    response.raise_for_status()
    branch = session.get(
        f"https://api.github.com/repos/{args.repo}/branches/{args.source_branch}", timeout=30
    )
    branch.raise_for_status()


def run_supervisor(args: argparse.Namespace) -> dict[str, Any]:
    args.dataset_root = args.dataset_root.resolve()
    args.protocol_dir = args.protocol_dir.resolve()
    args.project = args.project.resolve()
    args.python = normalize_python_executable(args.python)
    args.token_file = args.token_file.resolve()
    args.results_repo = args.results_repo.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    args.protocol_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.project / "lpr-g-v2-supervisor.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"design_version": "lpr-g-v2", "seed": 0, "history": []}
    )

    _transition(state_path, state, "auditing_authority")
    _audit_github(args)
    protocol, initial_state = _prepare_protocol(
        args, log=args.project / "logs" / "prepare-protocol.log"
    )
    _transition(
        state_path,
        state,
        "authority_ready",
        protocol=str(protocol),
        initial_state=str(initial_state),
    )

    canary_path = args.protocol_dir / "lpr-g-v2-canary.json"
    if canary_path.exists():
        canary = json.loads(canary_path.read_text(encoding="utf-8"))
    else:
        canary = run_model_canary(initial_state)
        _atomic_json(canary_path, canary)
    _transition(state_path, state, "model_canary_passed", canary=canary)

    for _, variant in SCREEN_ORDER:
        run_arm(
            args,
            state_path,
            state,
            protocol=protocol,
            initial_state=initial_state,
            stage="screen",
            variant=variant,
            preflight=True,
        )
    _verify_preflight_pair(args.project)
    _transition(state_path, state, "preflight_passed")
    if args.preflight:
        _transition(state_path, state, "preflight_complete")
        return state

    checkpoints: dict[str, Path] = {}
    for _, variant in SCREEN_ORDER:
        checkpoints[variant] = run_arm(
            args,
            state_path,
            state,
            protocol=protocol,
            initial_state=initial_state,
            stage="screen",
            variant=variant,
        )
    evidence = args.project / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    screen_evaluation_path = evidence / "screen-seed0-lprg-evaluation.json"
    _run_evaluation(
        args,
        checkpoint=checkpoints["lprg"],
        stage="screen",
        output=screen_evaluation_path,
    )
    benchmark_path = evidence / "lpr-g-v2-benchmark.json"
    _run_benchmark(args, initial_state=initial_state, output=benchmark_path)
    screen_report = _run_comparison(
        args,
        stage="screen",
        evaluation=screen_evaluation_path,
        benchmark=benchmark_path,
        output=evidence / "screen-seed0-comparison.json",
    )
    _transition(
        state_path,
        state,
        "screen_decided",
        screen_status=screen_report["status"],
    )
    action = next_stage(screen_report, through_formal=args.through_formal)
    if action == "repair":
        _transition(state_path, state, "repair_required", report=screen_report)
        return state
    if action == "stop":
        _transition(
            state_path,
            state,
            "screen_complete" if screen_report["status"] == "passed" else "scientific_failed",
            report=screen_report,
        )
        return state

    formal_checkpoints: dict[str, Path] = {}
    for _, variant in FORMAL_ORDER:
        formal_checkpoints[variant] = run_arm(
            args,
            state_path,
            state,
            protocol=protocol,
            initial_state=initial_state,
            stage="formal",
            variant=variant,
        )
    formal_evaluation_path = evidence / "formal-seed0-lprg-evaluation.json"
    _run_evaluation(
        args,
        checkpoint=formal_checkpoints["lprg"],
        stage="formal",
        output=formal_evaluation_path,
    )
    formal_report = _run_comparison(
        args,
        stage="formal",
        evaluation=formal_evaluation_path,
        benchmark=benchmark_path,
        output=evidence / "formal-seed0-comparison.json",
    )
    _transition(
        state_path,
        state,
        "formal_complete",
        formal_status=formal_report["status"],
        report=formal_report,
    )
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen resumable seed0 control/LPR-G v2 experiments."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", default="kkc236/uav-detection-baselines")
    parser.add_argument(
        "--repo-url", default="https://github.com/kkc236/uav-detection-baselines.git"
    )
    parser.add_argument("--source-branch", default="codex/lpr-rtdetr")
    parser.add_argument("--results-branch", default="training-results")
    parser.add_argument("--results-repo", type=Path, required=True)
    parser.add_argument("--through-formal", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> None:
    state = run_supervisor(build_parser().parse_args())
    print(json.dumps(state, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
