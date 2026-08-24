# Clean FDR / DCF-FDR Automatic Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy an unattended finalizer that publishes complete Clean FDR and DCF-FDR Formal100 evidence to the private material repository and verifies the Git commit and Release assets remotely.

**Architecture:** A pure Python evidence module validates both arms and derives deterministic comparisons. A publication CLI stages lightweight evidence, commits it through an isolated private-repository checkout, uploads four checkpoints to a dedicated Release, and verifies remote state. A detached Bash watcher waits for DCF completion and retries publication without touching training or deleting local evidence.

**Tech Stack:** Python 3.10, PyTorch, CSV/JSON/gzip/tarfile, requests, Git, Bash, pytest, GitHub REST API.

---

## File map

- Create `src/dcf_fdr_publication.py`: completion gates, metric derivation, hashing, deterministic evidence staging.
- Create `scripts/publish_dcf_fdr_results.py`: private-repository Git commit, Release upload, and remote verification.
- Create `scripts/watch_and_publish_dcf_fdr.sh`: detached completion watcher and bounded retry policy.
- Create `tests/test_dcf_fdr_publication.py`: pure evidence and gate tests.
- Create `tests/test_publish_dcf_fdr_results.py`: GitHub transaction and secret-safety tests.
- Modify this plan: mark executed steps.

### Task 1: Pure completion gates and comparison evidence

**Files:**
- Create: `tests/test_dcf_fdr_publication.py`
- Create: `src/dcf_fdr_publication.py`

- [x] **Step 1: Write failing gate and metric tests**

Create fixtures for two synthetic 100-row `results.csv` files, authorities,
arguments, logs, and checkpoints. Require `validate_arm()` to reject 99 rows,
epoch gaps, wrong source commit, wrong frozen-state hash, terminal error text, and
unreadable checkpoints. Require `build_comparison()` to select best mAP without
rounding and emit 100 aligned DCF-minus-Clean rows.

```python
def test_validate_arm_requires_exact_continuous_formal100(tmp_path: Path) -> None:
    spec = make_arm(tmp_path, "clean", epochs=range(99))
    with pytest.raises(PublicationGateError, match="exactly 100"):
        validate_arm(spec)


def test_comparison_uses_unrounded_best_map_and_aligns_all_epochs(tmp_path: Path) -> None:
    clean = validate_arm(make_arm(tmp_path, "clean", best_map="0.29696"))
    dcf = validate_arm(make_arm(tmp_path, "dcf", best_map="0.29697"))
    report, rows = build_comparison(clean, dcf)
    assert report["decision"] == "passed_nonnegative"
    assert report["best_delta"]["metrics/mAP50-95(B)"] == pytest.approx(0.00001)
    assert len(rows) == 100
```

- [x] **Step 2: Run the focused test and confirm red state**

Run: `pytest tests/test_dcf_fdr_publication.py -q`

Expected: collection fails because `src.dcf_fdr_publication` does not exist.

- [x] **Step 3: Implement the evidence module**

Implement immutable `ArmSpec`, `ValidatedArm`, and
`PublicationGateError` types, streaming SHA-256, strict CSV parsing, authority
checks against commit `ec4e2a463db7a53f7c4c9c4bc9edabdf5c39f40b` and
initial-state hash
`51aab2eb3fb7d123501c69c7b8dc90ff3ea0b9344a108edeef2c7d6dcdbb742d`,
CPU checkpoint deserialization, log failure scanning, best/latest/peak metric
extraction, aligned comparison rows, atomic JSON/CSV writes, gzip log copying,
and a deterministic lightweight tar bundle.

```python
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def decision(clean_map: float, dcf_map: float) -> str:
    return "passed_nonnegative" if dcf_map >= clean_map else "failed_negative"
```

- [x] **Step 4: Run focused tests and confirm green state**

Run: `pytest tests/test_dcf_fdr_publication.py -q`

Expected: all tests pass.

- [x] **Step 5: Commit Task 1**

```bash
git add src/dcf_fdr_publication.py tests/test_dcf_fdr_publication.py
git commit -m "feat: validate and compare Clean DCF evidence"
```

### Task 2: Idempotent private GitHub publication

**Files:**
- Create: `tests/test_publish_dcf_fdr_results.py`
- Create: `scripts/publish_dcf_fdr_results.py`

- [x] **Step 1: Write failing publication transaction tests**

Use fake HTTP responses and a temporary Git repository to require private-repo
verification, same-size asset skipping, wrong-size asset replacement, exact
asset-name verification, result commit verification, token-file mode rejection,
and absence of token content from commands, status JSON, and errors.

```python
def test_release_assets_are_idempotent(tmp_path: Path) -> None:
    asset = tmp_path / "clean-best.pt"
    asset.write_bytes(b"checkpoint")
    session = FakeSession(existing_size=asset.stat().st_size)
    assert upload_asset(session, release(), asset, asset.name) == "skipped"
    assert session.uploaded == []


def test_token_file_must_be_private(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("secret", encoding="utf-8")
    token.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        read_private_token(token)
```

- [x] **Step 2: Run the publication tests and confirm red state**

Run: `pytest tests/test_publish_dcf_fdr_results.py -q`

Expected: collection fails because `scripts.publish_dcf_fdr_results` does not
exist.

- [x] **Step 3: Implement the publication CLI**

Add `--clean-root`, `--dcf-root`, `--staging-root`,
`--material-checkout`, `--token-file`, `--repository`, `--branch`,
`--tag`, and `--check-only` options. Build evidence before remote mutation.
Use a mode-700 `GIT_ASKPASS` helper that reads the mode-600 token file through
an environment variable. Clone/fetch the private repository into an isolated
checkout, copy only the staged lightweight tree to
`experiments/clean-dcf-fdr-formal100-seed0-20260824`, create an idempotent
commit, push, and verify its SHA through the API. Create/reuse the private
prerelease, upload the four renamed checkpoint assets plus evidence bundle and
manifest, refetch the Release, and require exact remote sizes.

```python
REPOSITORY = "kkc236/icassp2027-fdr-bpdd-fia-material"
TAG = "clean-dcf-fdr-formal100-seed0-20260824"
EXPERIMENT_DIR = "experiments/clean-dcf-fdr-formal100-seed0-20260824"


def read_private_token(path: Path) -> str:
    if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError(f"GitHub token file must be a regular 0600 file: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("GitHub token file is empty")
    return value
```

- [x] **Step 4: Run publication tests and confirm green state**

Run: `pytest tests/test_publish_dcf_fdr_results.py -q`

Expected: all tests pass.

- [x] **Step 5: Commit Task 2**

```bash
git add scripts/publish_dcf_fdr_results.py tests/test_publish_dcf_fdr_results.py
git commit -m "feat: publish Clean DCF evidence to GitHub"
```

### Task 3: Detached watcher and failure-safe retry

**Files:**
- Create: `scripts/watch_and_publish_dcf_fdr.sh`
- Modify: `tests/test_publish_dcf_fdr_results.py`

- [x] **Step 1: Write the failing watcher contract test**

Require fixed Formal100 paths, a 60-second poll, DCF-process detection, exact
publication CLI arguments, success/failure markers, retries, no token text, no
shutdown command, and no deletion of run artifacts or shared credentials.

```python
def test_watcher_is_non_destructive_and_waits_for_dcf_completion() -> None:
    text = (ROOT / "scripts/watch_and_publish_dcf_fdr.sh").read_text()
    assert "publish_dcf_fdr_results.py" in text
    assert "sleep 60" in text
    assert "publication-succeeded.json" in text
    assert "shutdown" not in text.lower()
    assert "poweroff" not in text.lower()
    assert "rm -rf" not in text
```

- [x] **Step 2: Run the watcher contract and confirm red state**

Run: `pytest tests/test_publish_dcf_fdr_results.py -q`

Expected: failure because the watcher does not exist.

- [x] **Step 3: Implement the watcher**

Wait while the exact DCF output-root trainer exists. When it exits, invoke the
publisher `--check-only`; incomplete or invalid evidence writes a sanitized
failure JSON. For complete evidence, retry actual publication at bounded
60-second intervals, write the publisher JSON atomically as the success marker,
and retain all local artifacts and the shared token file.

```bash
while pgrep -f "train_dcf_fdr.py --arm dcf .*dcf-fdr-ec4e2a46-dcf" >/dev/null; do
  sleep 60
done

for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if "$PYTHON" "$PUBLISHER" "${PUBLISH_ARGS[@]}"; then
    exit 0
  fi
  sleep 60
done
exit 1
```

- [x] **Step 4: Run all new tests and shell syntax check**

Run: `pytest tests/test_dcf_fdr_publication.py tests/test_publish_dcf_fdr_results.py -q`

Run: `bash -n scripts/watch_and_publish_dcf_fdr.sh`

Expected: all tests pass and shell syntax exits zero.

- [x] **Step 5: Commit Task 3**

```bash
git add scripts/watch_and_publish_dcf_fdr.sh tests/test_publish_dcf_fdr_results.py
git commit -m "feat: watch and finalize Clean DCF publication"
```

### Task 4: Full verification, push, and server deployment

**Files:**
- Modify: `docs/superpowers/plans/2026-08-24-clean-dcf-fdr-auto-publication.md`

- [x] **Step 1: Run focused and regression tests**

Run:

```bash
pytest tests/test_dcf_fdr_publication.py \
       tests/test_publish_dcf_fdr_results.py \
       tests/test_fdr_head.py \
       tests/test_fdr_protocol.py \
       tests/test_rtdetr_fdr.py -q
```

Expected: zero failures.

- [x] **Step 2: Verify repository hygiene and push source branch**

Run:

```bash
git diff --check
git status --short
git push origin codex/ap-fdr-integrated-redesign
git ls-remote origin refs/heads/codex/ap-fdr-integrated-redesign
```

Expected: clean worktree and remote branch SHA equal to local `HEAD`.

- [x] **Step 3: Transfer the exact commit to the server**

Create a Git bundle from local `HEAD`, transfer it to `/data/uav/source`, and
clone an immutable publication checkout at that exact commit. Verify the commit
SHA, clean worktree, Python imports, token-file mode, current DCF PID, and disk
space before launch.

- [x] **Step 4: Exercise live dry-run gates**

Run the publisher with `--check-only` against the live paths while DCF is still
incomplete.

Expected: Clean validates, DCF is rejected as incomplete, no GitHub mutation
occurs, and the current DCF trainer remains alive.

- [x] **Step 5: Launch and verify the detached watcher**

Start with `nohup`, record PID and log under
`/data/uav/publication/clean-dcf-fdr-formal100-seed0-20260824`, and verify with
`ps` that PPID is 1 and the process is sleeping/waiting. Recheck DCF epoch count,
GPU process, and publication status paths.

- [x] **Step 6: Mark the plan execution and commit deployment record**

Update this plan's completed checkboxes, record the deployed source commit and
watcher PID in a sanitized deployment JSON, commit it, push it, and verify the
remote SHA. The deployment record must contain no host password or GitHub token.
