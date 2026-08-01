# I-TBER Runtime Driver Amendment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admit the approved `570.133.07` execution driver without erasing the historical `550.142` baseline identity, re-evaluate stock in the current environment, and start I-TBER only after amended Gate 0 evidence passes.

**Architecture:** `src/itber_protocol.py` owns two immutable environments and one canonical amendment hash. Gate 0 reports `passed_with_runtime_amendment`; the supervisor adds a mandatory current-environment stock-authority phase before cache generation. The amendment identity is copied into cache, private checkpoints, evaluations, benchmark reports, GitHub manifests, and recovery validation.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, Bash, pytest, GitHub REST, RTX 4090.

---

## File map

- Modify `src/itber_protocol.py`: baseline/execution environment constants, amendment payload/hash, environment capture and validation.
- Modify `scripts/run_itber_canary.py`: amended Gate 0 status and report identity.
- Modify `deploy/itber/verify_host.sh`: exact execution driver and explicit amended status.
- Modify `scripts/run_itber_pipeline.py`: accept amended Gate 0 and require stock authority before cache.
- Create `scripts/evaluate_itber_stock.py`: immutable three-repeat current-environment stock evaluation.
- Modify `src/itber_cache.py` and `scripts/cache_itber_evidence.py`: bind cache to amendment hash.
- Modify `scripts/train_itber.py`, `scripts/evaluate_itber.py`, `scripts/benchmark_itber.py`: bind private artifacts and reports to both environments and amendment hash.
- Modify `src/itber_publication.py` and `scripts/restore_itber_checkpoint.py`: publish and recover only the exact amended identity.
- Modify focused tests and `docs/ITBER_BARE_SERVER_GUIDE.md`.

### Task 1: Dual immutable environment authority

**Files:**
- Modify: `src/itber_protocol.py`
- Test: `tests/test_itber_protocol.py`

- [ ] **Step 1: Write failing tests for the two environments and exact amendment**

```python
def test_runtime_driver_amendment_preserves_baseline_reference() -> None:
    assert BASELINE_REFERENCE_ENVIRONMENT["driver"] == "550.142"
    assert EXECUTION_ENVIRONMENT["driver"] == "570.133.07"
    assert EXECUTION_ENVIRONMENT["reported_memory_mib"] == 49140
    assert RUNTIME_AMENDMENT["allowed_differences"] == ["driver", "reported_memory_mib"]
    assert len(RUNTIME_AMENDMENT_SHA256) == 64

def test_authority_accepts_only_the_approved_execution_environment() -> None:
    report = validate_authorities(**_authority())
    assert report["status"] == "passed_with_runtime_amendment"
    assert report["runtime_amendment_sha256"] == RUNTIME_AMENDMENT_SHA256
    changed = _authority()
    changed["environment"]["torch"] = "2.6.0"
    with pytest.raises(ProtocolViolation, match="environment.torch"):
        validate_authorities(**changed)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_itber_protocol.py -q`

Expected: collection or assertion failure because dual-environment symbols and amended status do not exist.

- [ ] **Step 3: Implement canonical constants and validation**

Add `BASELINE_REFERENCE_ENVIRONMENT`, `EXECUTION_ENVIRONMENT`, `RUNTIME_AMENDMENT`, and a SHA256 over canonical JSON. Add `current_execution_environment()` that records the seven existing runtime fields plus `nvidia-smi --query-gpu=memory.total`. Make `validate_authorities()` compare only against `EXECUTION_ENVIRONMENT` while returning both environments, the amendment payload/hash, and `passed_with_runtime_amendment`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_itber_protocol.py -q`

Expected: all protocol tests pass.

```bash
git add src/itber_protocol.py tests/test_itber_protocol.py
git commit -m "feat: record I-TBER runtime driver amendment"
```

### Task 2: Amended Gate 0 and supervisor semantics

**Files:**
- Modify: `scripts/run_itber_canary.py`
- Modify: `scripts/run_itber_pipeline.py`
- Modify: `deploy/itber/verify_host.sh`
- Test: `tests/test_itber_canary.py`
- Test: `tests/test_itber_pipeline.py`
- Test: `tests/test_itber_deploy_scripts.py`

- [ ] **Step 1: Write failing status-transition tests**

```python
def test_amended_authority_and_gate0_are_accepted() -> None:
    evidence = PipelineEvidence(
        authority="passed_with_runtime_amendment",
        gate0="passed_with_runtime_amendment",
        stock_authority=None,
        cache_complete=False,
        gate1=None,
        screen=None,
        formal=None,
    )
    assert next_pipeline_phase(evidence) == "stock_authority"

def test_unapproved_pass_like_status_is_rejected() -> None:
    evidence = _evidence(authority="passed_with_other_amendment")
    assert next_pipeline_phase(evidence) == "engineering_invalid"
```

Require the Bash verifier to contain both `baseline_reference_driver="550.142"`, `expected_driver="570.133.07"`, and `passed_with_runtime_amendment`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_itber_canary.py tests/test_itber_pipeline.py tests/test_itber_deploy_scripts.py -q`

Expected: failures on the new state, field, and shell markers.

- [ ] **Step 3: Implement minimal amended status handling**

Use the accepted status set exactly as:

```python
ACCEPTED_GATE_STATUSES = {"passed", "passed_with_runtime_amendment"}
```

The current I-TBER authority and canary must emit only `passed_with_runtime_amendment`; ordinary `passed` remains accepted solely for old immutable reports. Update the host verifier to require `570.133.07`, report the baseline reference driver separately, and return the amended status. No generic allow-list or CLI override is permitted.

- [ ] **Step 4: Verify GREEN and commit**

Run the same focused test command and require all pass.

```bash
git add scripts/run_itber_canary.py scripts/run_itber_pipeline.py deploy/itber/verify_host.sh tests/test_itber_canary.py tests/test_itber_pipeline.py tests/test_itber_deploy_scripts.py
git commit -m "feat: admit approved I-TBER runtime environment"
```

### Task 3: Mandatory current-environment stock authority

**Files:**
- Create: `scripts/evaluate_itber_stock.py`
- Modify: `scripts/run_itber_pipeline.py`
- Test: `tests/test_itber_stock_evaluation.py`
- Test: `tests/test_itber_pipeline.py`

- [ ] **Step 1: Write failing evaluator and phase tests**

Test a pure `build_stock_authority_report()` helper with three identical metric dictionaries. Require exact repeat equality, baseline/data/category SHA, both environments, amendment hash, fixed validation constants, and status `passed_with_runtime_amendment`. Reject non-identical repeats and any environment drift. Require the pipeline order `gate0 -> stock_authority -> cache`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_itber_stock_evaluation.py tests/test_itber_pipeline.py -q`

Expected: missing module/helper and state transition failures.

- [ ] **Step 3: Implement stock evaluation**

Reuse `_build_validation_loader`, `_evaluate_once`, `assert_repeated_evaluations`, and `write_immutable_report` from the existing I-TBER evaluator. Instantiate a zero-initialized frozen adapter, evaluate stock three times, assert stock/refined identity, and write `stock-authority.json` exactly once. The CLI accepts only baseline, dataset, and output paths. Add this command as the supervisor phase immediately before cache.

- [ ] **Step 4: Verify GREEN and commit**

Run the same focused tests and require all pass.

```bash
git add scripts/evaluate_itber_stock.py scripts/run_itber_pipeline.py tests/test_itber_stock_evaluation.py tests/test_itber_pipeline.py
git commit -m "feat: lock current-environment stock authority"
```

### Task 4: Bind every downstream artifact to the amendment

**Files:**
- Modify: `src/itber_cache.py`
- Modify: `scripts/cache_itber_evidence.py`
- Modify: `scripts/train_itber.py`
- Modify: `scripts/evaluate_itber.py`
- Modify: `scripts/benchmark_itber.py`
- Modify: `src/itber_publication.py`
- Modify: `scripts/restore_itber_checkpoint.py`
- Test: `tests/test_itber_cache.py`
- Test: `tests/test_itber_training.py`
- Test: `tests/test_itber_evaluation.py`
- Test: `tests/test_itber_benchmark.py`
- Test: `tests/test_itber_publication.py`
- Test: `tests/test_itber_restore.py`

- [ ] **Step 1: Write failing identity tests**

Require `runtime_amendment_sha256 == RUNTIME_AMENDMENT_SHA256` in cache authority, resume checkpoints, publication identity, download manifests, evaluation reports, and benchmark reports. Add one corruption case per artifact family that replaces the hash with `"F" * 64` and must fail before loading private state.

- [ ] **Step 2: Verify RED**

Run all six focused test files and require failures specifically on the missing amendment identity.

- [ ] **Step 3: Implement exact propagation**

Add the amendment hash to `_normalized_authority`, `validate_gate1_cache_manifest`, `validate_resume_checkpoint`, each saved checkpoint, `PublicationIdentity.as_dict()`, restore verification, evaluation reports, and benchmark reports. Include the two full environment dictionaries in human-readable reports; use only the hash in recovery identities. Do not add scientific CLI flags.

- [ ] **Step 4: Verify GREEN and commit**

Run the same six test files and require all pass.

```bash
git add src/itber_cache.py scripts/cache_itber_evidence.py scripts/train_itber.py scripts/evaluate_itber.py scripts/benchmark_itber.py src/itber_publication.py scripts/restore_itber_checkpoint.py tests/test_itber_cache.py tests/test_itber_training.py tests/test_itber_evaluation.py tests/test_itber_benchmark.py tests/test_itber_publication.py tests/test_itber_restore.py
git commit -m "feat: bind I-TBER artifacts to runtime amendment"
```

### Task 5: Verify, deploy, pass gates, and start

**Files:**
- Modify: `docs/ITBER_BARE_SERVER_GUIDE.md`
- Modify: `scripts/audit_itber_deployment.py`
- Test: `tests/test_itber_deployment_audit.py`

- [ ] **Step 1: Update documentation test first**

Require the guide to state both driver values, `passed_with_runtime_amendment`, current-environment stock authority, and prohibition on cross-environment AP subtraction. Verify RED, then update the guide and local readiness audit.

- [ ] **Step 2: Run complete local verification**

```bash
$itberTests = Get-ChildItem tests -Filter 'test_itber_*.py' | ForEach-Object { $_.FullName }
python -m pytest @itberTests tests/test_rtdetr_itber.py -q
python -m pytest -q
python -m compileall -q src scripts deploy/itber
git diff --check
python scripts/audit_itber_deployment.py --output tmp/itber-runtime-amendment-readiness.json
```

Expected: every test passes and readiness is `ready_waiting_for_server`.

- [ ] **Step 3: Commit and push exact source**

```bash
git add docs/ITBER_BARE_SERVER_GUIDE.md scripts/audit_itber_deployment.py tests/test_itber_deployment_audit.py
git commit -m "docs: operationalize I-TBER driver amendment"
git push origin codex/lpr-rtdetr
```

- [ ] **Step 4: Deploy immutable commit and verify server**

Mirror-first fetch the exact pushed commit into a new `/data/uav/source/uav-detection-baselines-<shortsha>` directory. Run all I-TBER tests, compileall, `verify_host.sh`, authority audit, and Gate 0. Require both host and Gate 0 status `passed_with_runtime_amendment`.

- [ ] **Step 5: Re-evaluate stock and start the supervised pipeline**

Run the supervisor with the immutable baseline, dataset, run root, cache root, and credential-free screen/formal publication configs. It must create `stock-authority.json` before cache. Start it in a hidden/background process with PID, log, and immutable source commit recorded. Verify process liveness, GPU utilization, pipeline phase, stock report status, and that no training phase begins before both amended gates pass.
