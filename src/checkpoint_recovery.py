from __future__ import annotations

import re
from pathlib import Path

import torch


EPOCH_CHECKPOINT_PATTERN = re.compile(r"epoch(\d+)\.pt$")


def validate_checkpoint(path: str | Path) -> tuple[bool, str]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return False, "missing"

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        return False, f"unreadable: {type(error).__name__}"

    if not isinstance(checkpoint, dict):
        return False, "not a checkpoint dictionary"

    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        return False, "missing completed epoch"
    if checkpoint.get("optimizer") is None:
        return False, "optimizer state was stripped"
    if checkpoint.get("ema") is None and checkpoint.get("model") is None:
        return False, "model state is missing"
    return True, f"epoch={epoch}"


def _epoch_number(path: Path) -> int:
    match = EPOCH_CHECKPOINT_PATTERN.match(path.name)
    return int(match.group(1)) if match else -1


def find_resume_checkpoint(run_dir: str | Path) -> Path | None:
    weights_dir = Path(run_dir) / "weights"
    candidates = [weights_dir / "last.pt"]
    candidates.extend(sorted(weights_dir.glob("epoch*.pt"), key=_epoch_number, reverse=True))

    for candidate in candidates:
        valid, _ = validate_checkpoint(candidate)
        if valid:
            return candidate.resolve()
    return None
