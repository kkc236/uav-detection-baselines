# SADED Fresh-100 Postprocess Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete and seal the missing fresh-stock runtime CLIs, then run the completed seed-0 `last.pt` through five-view cache, GT-free routing, sealed dev-val evaluation, and the unchanged five-gate adjudicator.

**Architecture:** Preserve the existing authority core and three uncommitted runtime drafts. Add regression tests before correcting their checksum bug, then add the missing sealed evaluator and standalone adjudicator test-first. Commit a clean source closure, deploy that exact commit to `/home/ubuntu`, freeze a fresh protocol, and execute each evidence stage serially while the already completed endpoint files remain read-only.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, pytest, gzip JSONL, SHA256 closures, NVIDIA RTX 4090.

---

### Task 1: Characterize and close the protocol, cache, and route drafts

**Files:**
- Modify: `tests/test_saded_stock_evaluation_protocol.py`
- Modify: `tests/test_saded_stock_postprocess.py`
- Modify: `src/saded_stock_evaluation_protocol.py`
- Add: `scripts/prepare_saded_stock_evaluation_protocol.py`
- Add: `scripts/cache_saded_stock_endpoint.py`
- Add: `scripts/route_saded_stock_single.py`

- [ ] **Step 1: Add protocol validation tests**

Add tests that build a minimal bound protocol and assert:

```python
def test_protocol_source_file_set_contains_every_runtime_cli() -> None:
    required = {
        "scripts/prepare_saded_stock_evaluation_protocol.py",
        "scripts/cache_saded_stock_endpoint.py",
        "scripts/route_saded_stock_single.py",
        "scripts/evaluate_saded_stock_single.py",
        "scripts/adjudicate_saded_stock_fresh.py",
    }
    assert required <= set(POSTPROCESS_SOURCE_FILES)
    assert all((REPO_ROOT / name).is_file() for name in required)
```

The file-existence assertion must fail until Tasks 2 and 3 add the missing
CLIs.

- [ ] **Step 2: Add the checksum regression test**

```python
def test_route_checksum_reader_accepts_writer_output(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    write_checksums(
        tmp_path / "checksums.sha256",
        [artifact],
        root=tmp_path,
    )
    observed = _verify_checksums(tmp_path, {"artifact.json"})
    assert observed["artifact.json"] == sha256_file(artifact)
```

- [ ] **Step 3: Run RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest `
  tests/test_saded_stock_evaluation_protocol.py `
  tests/test_saded_stock_postprocess.py -q
```

Expected: the runtime-file test reports the two missing CLIs, and the checksum
test reports an uppercase/lowercase digest mismatch.

- [ ] **Step 4: Apply the minimal checksum correction**

Keep checksum values canonical lowercase:

```python
digest, relative = line.split("  ", 1)
observed[relative] = digest.lower()
if sha256_file(root / relative) != digest:
    raise ValueError("fresh cache checksum closure drift")
```

Do not change any route or metric rule.

- [ ] **Step 5: Run the focused tests**

Run the same pytest command. Expected: only the deliberately missing evaluator
and adjudicator file assertion remains RED.

### Task 2: Add the one-shot sealed dev-val evaluator

**Files:**
- Add: `scripts/evaluate_saded_stock_single.py`
- Modify: `tests/test_saded_stock_postprocess.py`

- [ ] **Step 1: Write failing evaluator tests**

Test the desired helper API before creating the script:

```python
def test_evaluation_claim_is_exclusive(tmp_path: Path) -> None:
    claim = tmp_path / "claim.json"
    create_evaluation_claim(claim, {"state": "CONSUMED"})
    with pytest.raises(FileExistsError):
        create_evaluation_claim(claim, {"state": "CONSUMED"})


def test_metric_row_preserves_prediction_provenance() -> None:
    row = metric_row(DATASET_IMAGE, PREDICTIONS)
    assert row["pred_source"] == [0]
    assert row["pred_query"] == [7]
    assert row["gt_boxes"] == DATASET_IMAGE["gt_boxes"]
```

- [ ] **Step 2: Run RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest `
  tests/test_saded_stock_postprocess.py -q
```

Expected: import fails because `scripts.evaluate_saded_stock_single` does not
exist.

- [ ] **Step 3: Implement the evaluator**

The CLI accepts only `--evaluation-protocol`. It must:

1. call `validate_evaluation_protocol`;
2. verify the exact route artifact set, checksums, anchor, source binding,
   checkpoint binding, image order, invariants, and immutable snapshots;
3. atomically create the protocol-bound `evaluation_claim.json` with
   `O_CREAT | O_EXCL`;
4. only after the claim exists, import `load_dataset` and `evaluate_dataset`;
5. evaluate exactly arms `A` and `route_control` from one unified prediction
   file;
6. write `metrics.json`, `deltas.json`, `evaluation_invariants.json`,
   `evaluation_manifest.json`, and `checksums.sha256` to staging;
7. atomically rename staging and write `evaluation_anchor.json`.

The claim helper is:

```python
def create_evaluation_claim(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
```

`metric_row` mirrors the already frozen stage evaluator and copies box, score,
class, source, query, GT, ignore boxes, and `effective_gain`.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest `
  tests/test_saded_stock_postprocess.py -q
```

Expected: all tests pass.

### Task 3: Add the standalone five-gate adjudicator

**Files:**
- Add: `scripts/adjudicate_saded_stock_fresh.py`
- Modify: `tests/test_saded_stock_postprocess.py`

- [ ] **Step 1: Write failing adjudicator tests**

```python
def test_fresh_adjudicator_recomputes_five_frozen_gates() -> None:
    result = decide(
        arm_a=BASELINE,
        route_control=CANDIDATE,
        invariants_passed=True,
    )
    assert result["decision"] == "SADED_SINGLE_SEED_GO"
    assert set(result["gates"]) == set(FORMAL_THRESHOLDS)


def test_fresh_adjudicator_fails_closed_on_bad_closure() -> None:
    result = decide(
        arm_a=BASELINE,
        route_control=CANDIDATE,
        invariants_passed=False,
    )
    assert result["decision"] == "INVALID"
```

- [ ] **Step 2: Run RED**

Run the focused postprocess tests. Expected: import fails because
`scripts.adjudicate_saded_stock_fresh` does not exist.

- [ ] **Step 3: Implement the adjudicator**

The CLI accepts only `--evaluation-protocol`. It verifies the protocol, route,
evaluation, claim, checksum closures, anchors, source state, image count, and
input snapshots. It then calls the existing independent
`adjudicate_single_model`:

```python
def decide(*, arm_a, route_control, invariants_passed):
    return adjudicate_single_model(
        arm_a=arm_a,
        route_control=route_control,
        invariants_passed=invariants_passed,
    )
```

It writes `manifest.json`, `bindings.json`, `adjudication.json`, and
`checksums.sha256`, atomically renames staging, and writes a root
`adjudication_anchor.json`. Exit code is 0 for GO, 1 for scientific STOP, and
2 for INVALID.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest `
  tests/test_saded_stock_evaluation_protocol.py `
  tests/test_saded_stock_postprocess.py `
  tests/test_saded_single_model_adjudicator.py -q
```

Expected: all tests pass.

### Task 4: Freeze a clean source commit

**Files:**
- All files listed by `POSTPROCESS_SOURCE_FILES`
- Add: `docs/superpowers/plans/2026-07-26-saded-fresh100-postprocess-runtime.md`

- [ ] **Step 1: Run the complete test suite**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Verify source closure**

Run:

```powershell
C:\uav_env\Scripts\python.exe -c "from src.saded_stock_evaluation_protocol import postprocess_source_state; print(postprocess_source_state('.'))"
```

Expected: clean repository requirement cannot pass until the commit is made;
after committing, rerun and require a commit, per-file hashes, and bundle hash.

- [ ] **Step 3: Commit**

```powershell
git add docs/superpowers/plans/2026-07-26-saded-fresh100-postprocess-runtime.md `
  src/saded_stock_evaluation_protocol.py `
  tests/test_saded_stock_evaluation_protocol.py `
  tests/test_saded_stock_postprocess.py `
  scripts/prepare_saded_stock_evaluation_protocol.py `
  scripts/cache_saded_stock_endpoint.py `
  scripts/route_saded_stock_single.py `
  scripts/evaluate_saded_stock_single.py `
  scripts/adjudicate_saded_stock_fresh.py
git commit -m "final: close fresh100 postprocess runtime"
```

### Task 5: Deploy and execute without GitHub

**Files created on server:**
- `/home/ubuntu/repo-saded-fresh-postprocess-<commit>/`
- `/home/ubuntu/saded-fresh-eval-protocols/<run-id>/`
- `/home/ubuntu/saded-fresh-evidence/<run-id>/`
- `/home/ubuntu/saded-fresh-eval-logs/<run-id>/`

- [ ] **Step 1: Preflight**

Require:

- no GPU process;
- `/home/ubuntu` has enough free bytes and inodes with a 2 GB safety margin;
- `/mnt/uav` is read-only input only;
- exact `last.pt` SHA256
  `515674348D0FF542663FE6FB4317240FC167A71EA31FACC1DEFE6A7E91B521F8`;
- clean deployed commit and passing focused server tests;
- no target, staging, anchor, or evaluation claim already exists.

- [ ] **Step 2: Freeze the protocol**

Run the prepare CLI with the original training protocol, source repo, summary,
and fresh `/home/ubuntu` protocol/evidence parents. Verify the protocol,
endpoint, image authority, route contract, checksums, and external anchor.

- [ ] **Step 3: Start five-view cache**

Start only the cache CLI on GPU 0 under a background driver. Record driver and
worker PIDs, log, status, and exit code. Report only image progress, GPU, RSS,
disk, and exceptions.

- [ ] **Step 4: Continue serially**

On cache exit 0, run GT-free route. On route exit 0, run the one-shot sealed
dev-val evaluator. On evaluation exit 0, run the standalone adjudicator.
Never read partial metrics.

- [ ] **Step 5: Report the sealed result**

After adjudication closure exists, report absolute metrics for `A` and
`route_control`, five deltas, five gate decisions, final GO/STOP/INVALID,
runtime, artifact paths, and checksums. Leave test-dev unopened and GitHub
untouched.
