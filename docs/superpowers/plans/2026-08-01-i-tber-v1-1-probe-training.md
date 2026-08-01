# I-TBER v1.1 Probe, Training, and Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pre-registered P0-P3 Probe, 12-epoch frozen-detector screen, 30-epoch full-data private training, exact evaluation, and resumable per-epoch GitHub evidence pipeline.

**Architecture:** Gate 1 uses an immutable no-augmentation evidence cache so four equal-capacity probes share detector evidence. Gate 2 and formal training generate evidence on the fly from the frozen detector under the registered augmentation sequence. A state-machine supervisor advances only on immutable Gate reports and publishes every complete private checkpoint before the next epoch is accepted.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, NumPy/Pandas, GitHub REST, pytest, RTX 4090.

---

## File map

- Create `src/itber_cache.py`: immutable shard writer/reader and cache authority.
- Create `src/itber_probe.py`: equal-capacity P0-P3 trainer and Gate 1 comparator.
- Create `src/itber_metrics.py`: edge MAE, direction accuracy, matched IoU, correction safety, and area AP helpers.
- Create `src/itber_evaluation.py`: same-checkpoint stock/refined evaluation and Gate 2/formal decisions.
- Create `src/itber_publication.py`: per-epoch transactional GitHub publication ledger.
- Create `scripts/cache_itber_evidence.py`: fixed Gate 1 cache CLI.
- Create `scripts/run_itber_probe.py`: P0-P3 fresh training and immutable Gate 1 report.
- Create `scripts/train_itber.py`: screen/formal frozen-detector private trainer.
- Create `scripts/evaluate_itber.py`: repeatable stock/refined evaluator.
- Create `scripts/restore_itber_checkpoint.py`: exact asset-pair recovery.
- Create `scripts/benchmark_itber.py`: parameter/GFLOPs/latency benchmark.
- Create `scripts/run_itber_pipeline.py`: Gate 0 -> Gate 1 -> Gate 2 -> formal state machine.
- Add focused tests for every component.

### Task 1: Immutable Gate 1 evidence cache

**Files:**
- Create: `src/itber_cache.py`
- Create: `scripts/cache_itber_evidence.py`
- Test: `tests/test_itber_cache.py`

- [ ] Write tests that prove contiguous image IDs, train/val separation, relative safe paths, per-shard SHA/bytes, manifest authority, atomic completion, corruption rejection, baseline SHA rejection, and deterministic round-trip tensors.
- [ ] Run `python -m pytest tests/test_itber_cache.py -q` and verify RED because the module is absent.
- [ ] Implement `CacheManifest`, `EvidenceShard`, `write_evidence_cache`, and `load_evidence_cache`. The CLI accepts baseline/data/output paths plus operational batch/workers only, disables random augmentation, uses fixed letterbox 640, and writes a completion manifest last.
- [ ] Run `python -m pytest tests/test_itber_cache.py -q` and verify GREEN.
- [ ] Commit with `git commit -m "feat: cache immutable I-TBER probe evidence"`.

### Task 2: Metrics and P0-P3 Gate 1

**Files:**
- Create: `src/itber_metrics.py`
- Create: `src/itber_probe.py`
- Create: `scripts/run_itber_probe.py`
- Test: `tests/test_itber_probe.py`

- [ ] Write tests with synthetic P0-P3 reports that require P3 edge-MAE improvements of 5% over P0 and 1.5% over P2, matched-IoU delta at least 0.005, tiny/small direction gain at least 3 percentage points, P3 best on both primary Probe metrics, finite activity, and exact 12 epochs per arm.
- [ ] Run `python -m pytest tests/test_itber_probe.py -q` and verify RED.
- [ ] Implement fixed seed0 private initialization, AdamW `1e-3/1e-4/(0.9,0.999)`, fixed AMP scale 128, independent P0-P3 fresh arms, equal parameter fingerprints, 12 epochs, immutable per-arm reports, and one Gate 1 decision JSON. No best-epoch selection is permitted.
- [ ] Run `python -m pytest tests/test_itber_probe.py -q` and verify GREEN.
- [ ] Commit with `git commit -m "feat: add pre-registered I-TBER P0-P3 Probe"`.

### Task 3: Screen/formal trainer and diagnostics

**Files:**
- Create: `scripts/train_itber.py`
- Test: `tests/test_itber_training.py`

- [ ] Write tests that lock screen to fixed647/full-val/12 epochs, formal to full6471/full-val/30 epochs, seed0 only, fresh private initialization between stages, frozen detector eval/no-grad, on-the-fly evidence, unified VisDrone augmentation, AdamW private optimizer, batch8/workers8/imgsz640, AMP128, save-period1, and no scientific CLI overrides.
- [ ] Run `python -m pytest tests/test_itber_training.py -q` and verify RED.
- [ ] Implement a private training loop that records named v1.1 losses, gate/residual distributions, detector SHA before/after, matched/unmatched correction RMS, optimizer/scaler/scheduler state, and an atomic checkpoint after every complete epoch.
- [ ] Run `python -m pytest tests/test_itber_training.py -q` and verify GREEN.
- [ ] Commit with `git commit -m "feat: train frozen I-TBER private stages"`.

### Task 4: Same-checkpoint evaluation and decisions

**Files:**
- Create: `src/itber_evaluation.py`
- Create: `scripts/evaluate_itber.py`
- Test: `tests/test_itber_evaluation.py`

- [ ] Write tests that require three repeated evaluations to match exactly and compute stock/refined mAP50-95, AP50, AP75, AP-tiny, AP-small, matched IoU improvement/degradation counts, matched/unmatched RMS, and activity distributions.
- [ ] Write Gate 2 tests for `delta map>=0.002`, `delta AP75>=0.003`, `delta AP50>=-0.0005`, positive tiny/small, improvement count greater than degradation count, unmatched RMS ratio<=0.25, and active unsaturated gates/residuals. Write formal tests for `delta map>=0.003`, `delta AP75>=0.005`, positive tiny/small, positive tail5, and unchanged detector SHA.
- [ ] Run `python -m pytest tests/test_itber_evaluation.py -q` and verify RED.
- [ ] Implement immutable reports and strict boundary comparisons using exact decimal-to-float values without rounded decision inputs.
- [ ] Run `python -m pytest tests/test_itber_evaluation.py -q` and verify GREEN.
- [ ] Commit with `git commit -m "feat: evaluate I-TBER same-checkpoint gains"`.

### Task 5: Transactional publication and recovery

**Files:**
- Create: `src/itber_publication.py`
- Create: `scripts/restore_itber_checkpoint.py`
- Test: `tests/test_itber_publication.py`
- Test: `tests/test_itber_restore.py`

- [ ] Write tests for contiguous verified ledgers, exact stage/probe/seed/design/baseline/cache identities, checkpoint bytes/SHA/epoch, latest complete `.pt/.json` pair selection, cross-stage rejection, corruption rejection, retry exhaustion, and mode-600 token validation without token output.
- [ ] Run the focused tests and verify RED.
- [ ] Adapt the proven LPR-G transaction pattern under new release name/tag/prefixes, retaining three large private checkpoints and permanent lightweight history. Recovery must atomically download and validate before rename.
- [ ] Run `python -m pytest tests/test_itber_publication.py tests/test_itber_restore.py -q` and verify GREEN.
- [ ] Commit with `git commit -m "feat: publish and restore I-TBER checkpoints"`.

### Task 6: Truthful efficiency benchmark

**Files:**
- Create: `scripts/benchmark_itber.py`
- Test: `tests/test_itber_benchmark.py`

- [ ] Write tests for exact private parameter accounting, positive baseline denominator, alternating control/stock/refined timing order, synchronized CUDA, 50 warmups, 200 measurements, FP16 input `[1,3,640,640]`, and nonblocking targets `<1%/<1%/<3%`.
- [ ] Verify RED, implement benchmark, then verify GREEN with `python -m pytest tests/test_itber_benchmark.py -q`.
- [ ] Commit with `git commit -m "feat: benchmark I-TBER inference overhead"`.

### Task 7: Resumable scientific supervisor

**Files:**
- Create: `scripts/run_itber_pipeline.py`
- Test: `tests/test_itber_pipeline.py`

- [ ] Write state-machine tests for authority -> Gate0 -> cache -> P0-P3 -> Gate1 decision -> fresh Gate2 -> Gate2 decision -> fresh formal30. Failed gates stop; engineering-invalid gates enter repair; verified complete epochs resume; unverified epochs are republished before resume; formal never starts from a screen checkpoint.
- [ ] Run `python -m pytest tests/test_itber_pipeline.py -q` and verify RED.
- [ ] Implement atomic state/history JSON, subprocess logs, exact commands without scientific overrides, PID/process-group tracking, and terminal statuses `engineering_invalid`, `scientific_failed`, `screen_passed`, and `formal_complete`.
- [ ] Run `python -m pytest tests/test_itber_pipeline.py -q` and verify GREEN.
- [ ] Commit with `git commit -m "feat: supervise the I-TBER scientific pipeline"`.

### Task 8: Full pipeline verification

**Files:**
- Modify only concrete defects found by verification.

- [ ] Run all I-TBER tests: `python -m pytest tests/test_itber_*.py tests/test_rtdetr_itber.py -q`.
- [ ] Run the complete suite: `python -m pytest -q`.
- [ ] Run `git diff --check` and `python -m compileall -q src scripts deploy/itber`.
- [ ] Generate local readiness evidence with `python scripts/audit_itber_deployment.py --output tmp/itber-deployment-readiness.json` and require `ready_waiting_for_server`, not a false remote-ready status.
- [ ] Commit only evidence-backed corrections with `git commit -m "test: verify I-TBER deployment readiness"`; do not create an empty commit.
