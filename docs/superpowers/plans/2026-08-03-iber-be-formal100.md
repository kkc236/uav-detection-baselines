# IBER-BE Formal Full-Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable, per-epoch-published, seed0 100-epoch full-model IBER-BE training and independent epoch-100 baseline comparison path that exactly follows the frozen RT-DETR-L protocol.

**Architecture:** Keep the existing frozen-detector Gate-1/Gate-2 implementation untouched. Add a repository-owned RT-DETR model that preserves the stock forward/loss path and Hungarian assignment, while a detached final-query/F3/RGB side branch runs the current signed B3 `IBERRefiner`; optimize stock and private losses together with one MuSGD optimizer under fixed AMP128. Reuse the proven LPR-G runtime pattern for protocol authority, strict resume, diagnostics, and transactional publication, and add a separate evaluator that reconstructs both method and baseline models from exact epoch-100 checkpoints.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, Ultralytics 8.4.90, CUDA AMP, pytest, GitHub release/result publication helpers.

---

### Task 1: Freeze the formal protocol and initialization contract

**Files:**
- Create: `src/iber_formal_protocol.py`
- Test: `tests/test_iber_formal_protocol.py`

- [ ] Write tests that require the complete 100-epoch/full-data/MuSGD/augmentation/query contract, seed0 only, random-from-YAML initialization, and exact common-state/RNG parity with a stock RT-DETR-L model.
- [ ] Run `python -m pytest tests/test_iber_formal_protocol.py -q` and observe failure because the formal protocol module does not exist.
- [ ] Implement immutable settings and strict manifest/resume authority helpers without adding scientific CLI overrides.
- [ ] Re-run the test and require all assertions to pass.

### Task 2: Integrate signed IBER into full-model RT-DETR training

**Files:**
- Create: `src/rtdetr_iber_formal.py`
- Test: `tests/test_rtdetr_iber_formal.py`

- [ ] Write tests for exact stock initialization/output/loss parity, detached private-loss gradients, common stock gradients, 300 normal queries, stock/refined mode switching, and private/common optimizer partitioning.
- [ ] Run `python -m pytest tests/test_rtdetr_iber_formal.py -q` and observe the missing-module failure.
- [ ] Implement `IBERFullRTDETRDetectionModel` and `IBERFullTrainer` by reusing `IBERRecordingDecoder`, signed `IBERRefiner`, stock match recording, and `FixedPairedProtocolMixin`.
- [ ] Re-run the focused integration test and require it to pass.

### Task 3: Add the formal100 training entry point and publication hooks

**Files:**
- Create: `scripts/train_rtdetr_iber_formal.py`
- Create: `src/iber_formal_publication.py`
- Test: `tests/test_iber_formal_training.py`
- Test: `tests/test_iber_formal_publication.py`

- [ ] Write tests requiring no mutable scientific hyperparameters, exact full-data settings, strict same-run resume authority, per-epoch checkpoint publication before pruning, append-only epochs 1-100, and diagnostics for stock/private losses and signed evidence activity.
- [ ] Run both new test files and observe missing APIs.
- [ ] Implement the trainer CLI, runtime manifest, diagnostics/audit callbacks, verified publication transaction, and resume checks.
- [ ] Re-run both tests and require them to pass.

### Task 4: Add independent epoch-100 evaluation and comparison

**Files:**
- Create: `scripts/evaluate_rtdetr_iber_formal.py`
- Test: `tests/test_iber_formal_evaluation.py`

- [ ] Write tests requiring exact epoch100 method and baseline checkpoints, independent reconstruction, stock/refined/baseline metrics, finite deltas, immutable JSON output, and rejection of cross-protocol or non-epoch100 artifacts.
- [ ] Run the test and observe missing evaluator APIs.
- [ ] Implement frozen validation with `imgsz=640`, `batch=8`, `workers=8`, `max_det=300`, `nms=False`, and a single comparison artifact containing checkpoint hashes and all metric deltas.
- [ ] Re-run the evaluator tests and require them to pass.

### Task 5: Verify and commit

**Files:**
- Modify only files listed above and this plan.

- [ ] Run all new tests.
- [ ] Run the existing IBER and LPR-G integration/protocol/publication regressions.
- [ ] Run `git diff --check` and audit every frozen user requirement against tests/source.
- [ ] Commit the scoped implementation and report the commit hash, changed files, commands/results, and unresolved runtime assumptions.
