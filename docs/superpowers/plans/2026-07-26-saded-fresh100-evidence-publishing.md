# SADED Fresh-100 Evidence Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the live Fresh-100 run state now and automatically publish an integrity-checked success or clearly marked failure bundle when training terminates.

**Architecture:** A testable state classifier separates scientific run state
from GitHub publication behavior. The publisher runs on Windows, reads only
terminal status files until training ends, and uses a dedicated result branch
plus GitHub Release assets.

**Tech Stack:** Python 3.10, Paramiko, Git, GitHub CLI, pytest, PowerShell

---

### Task 1: Freeze terminal publication semantics

**Files:**
- Create: `tests/test_saded_fresh100_publisher.py`
- Create: `scripts/publish_saded_fresh100.py`

- [ ] **Step 1: Write the failing tests**

```python
from scripts.publish_saded_fresh100 import classify_terminal_state


def test_complete_zero_is_success():
    assert (
        classify_terminal_state("TRAIN_COMPLETE", "0")
        == "SUCCESS_CANDIDATE"
    )


def test_invalid_or_nonzero_is_invalid():
    assert classify_terminal_state("TRAIN_INVALID", "1") == "INVALID"
    assert classify_terminal_state("TRAIN_COMPLETE", "7") == "INVALID"


def test_running_is_not_terminal():
    assert classify_terminal_state("RUNNING", None) is None
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_saded_fresh100_publisher.py -q`

Expected: FAIL because `scripts.publish_saded_fresh100` does not exist.

- [ ] **Step 3: Implement the minimal state classifier**

```python
def classify_terminal_state(status: str | None, exit_code: str | None) -> str | None:
    if status == "TRAIN_COMPLETE" and exit_code == "0":
        return "SUCCESS_CANDIDATE"
    if status == "TRAIN_INVALID":
        return "INVALID"
    if exit_code not in (None, "", "0"):
        return "INVALID"
    return None
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_saded_fresh100_publisher.py -q`

Expected: `3 passed`.

### Task 2: Add deterministic evidence manifests

**Files:**
- Modify: `tests/test_saded_fresh100_publisher.py`
- Modify: `scripts/publish_saded_fresh100.py`

- [ ] **Step 1: Add a failing test**

```python
def test_invalid_manifest_cannot_claim_success():
    manifest = build_terminal_manifest(
        run_id="final-saded-fresh100-c5c35374",
        terminal_state="INVALID",
        exit_code="9",
        artifacts={"train.log": "abc"},
    )
    assert manifest["terminal_state"] == "INVALID"
    assert manifest["publish_as_success"] is False
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_saded_fresh100_publisher.py -q`

Expected: FAIL because `build_terminal_manifest` is missing.

- [ ] **Step 3: Implement the minimal manifest builder**

```python
def build_terminal_manifest(
    *,
    run_id,
    terminal_state,
    exit_code,
    artifacts,
    validation_passed,
):
    if terminal_state not in {"SUCCESS", "INVALID"}:
        raise ValueError("terminal_state must be SUCCESS or INVALID")
    if terminal_state == "SUCCESS" and not validation_passed:
        raise ValueError("SUCCESS evidence must be independently validated")
    return {
        "schema_version": "saded-fresh100-publication/v1",
        "run_id": run_id,
        "terminal_state": terminal_state,
        "exit_code": exit_code,
        "publish_as_success": terminal_state == "SUCCESS",
        "validation_passed": validation_passed,
        "artifacts": dict(sorted(artifacts.items())),
    }
```

- [ ] **Step 4: Verify GREEN and the focused suite**

Run: `python -m pytest tests/test_saded_fresh100_publisher.py -q`

Expected: all tests pass.

### Task 3: Publish the current non-terminal snapshot

**Files:**
- Create: `docs/evidence/saded_fresh100_seed0/progress/latest.json`
- Create: `docs/evidence/saded_fresh100_seed0/progress/README.md`

- [ ] **Step 1: Query only run progress and system health**

Run the read-only SSH collector and record epoch, batch, status, PIDs,
GPU/RSS/disk, anomaly counts, source commit, and collection timestamp.

- [ ] **Step 2: Verify the snapshot contains no validation metrics**

Run:
`rg -ni "map|ap50|ap75|recall|precision|fitness" docs/evidence/saded_fresh100_seed0/progress`

Expected: no metric payload matches.

- [ ] **Step 3: Commit and push**

Run:
`git add docs/superpowers docs/evidence/saded_fresh100_seed0/progress tests/test_saded_fresh100_publisher.py scripts/publish_saded_fresh100.py`

Run:
`git commit -m "ops: preserve fresh100 publication state"`

Run:
`git push origin final-saded-fresh100-results`

### Task 4: Deploy and verify the terminal watcher

**Files:**
- Execute in place:
  `C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\tmp\worktrees\sbr-fresh100-publisher\scripts\publish_saded_fresh100.py`
- Frozen validator checkout:
  `C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\tmp\worktrees\saded-fresh100-validator-c5c35374`
- Observe:
  `%LOCALAPPDATA%\Codex\SbrFresh100Publisher\status.json`

- [ ] **Step 1: Stop only the old local watcher**

Terminate the old Windows watcher PID after confirming its command line.
Do not signal the remote training PIDs.

- [ ] **Step 2: Start the tested watcher hidden**

Verify both worktrees are clean and at their expected commits. Supply the SSH
password only in the child process environment, then run the publisher from
the result worktree using `Start-Process -WindowStyle Hidden`. Do not copy the
script to `%TEMP%`, because its repository and validator paths are deliberately
derived from the checked-in script location.

- [ ] **Step 3: Verify live state**

Expected local status:

```json
{
  "status": "WATCHING",
  "remote_status": "RUNNING"
}
```

- [ ] **Step 4: Verify remote training continuity**

Confirm PID `417400` and driver PID `417396` remain alive, with remote status
`RUNNING` and unchanged command line.
