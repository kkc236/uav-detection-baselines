# EQuAL Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the integrated localization method from ACE-FDR to EQuAL without interrupting or mutating the active Formal100 experiment.

**Architecture:** The paper-facing source interface becomes `EQuAL`, while the running job retains its immutable ACE-FDR raw identifiers. A compatibility wrapper keeps the old launcher/config path usable, and launch evidence records both the canonical paper method and the raw runtime alias.

**Tech Stack:** Python 3.10, PyYAML, pytest, Git, Ultralytics 8.4.90.

---

### Task 1: Define the canonical EQuAL contract test-first

**Files:**
- Create: `tests/test_train_equal.py`
- Modify: `tests/test_fdr_yaml_configs.py`

- [ ] **Step 1: Write failing tests**

Assert that `scripts.train_equal.EQUAL_CONFIG` points to
`configs/rtdetr-l-equal.yaml`, the default run name is
`formal-seed0-equal-v1`, and the launch record contains
`method: equal` plus `runtime_alias: ace_fdr` only when supplied.

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/test_train_equal.py tests/test_fdr_yaml_configs.py -q`.
Expected: collection fails because `scripts.train_equal` and the EQuAL config
do not exist.

### Task 2: Add the EQuAL interface and compatibility mapping

**Files:**
- Create: `configs/rtdetr-l-equal.yaml`
- Create: `scripts/train_equal.py`
- Modify: `scripts/train_ace_fdr.py`

- [ ] **Step 1: Implement the minimal EQuAL config**

Copy the validated integrated configuration byte-for-byte except for its
paper-facing comment. Preserve `preliminary_box: false`,
`supervise_pre_boxes: false`, `supervise_dn_fdr: false`, and
`edge_adaptive_fgl: true`.

- [ ] **Step 2: Implement the canonical launcher**

Expose the same frozen Formal100 protocol under `train_equal.py`; emit
`method: equal`. Keep the old launcher as a compatibility wrapper so already
recorded commands remain replayable.

- [ ] **Step 3: Verify GREEN**

Run `python -m pytest tests/test_train_equal.py tests/test_train_ace_fdr.py tests/test_fdr_yaml_configs.py -q`.
Expected: all selected tests pass.

- [ ] **Step 4: Commit**

Commit with `feat: rename integrated localization method to EQuAL`.

### Task 3: Bind the active run without mutating evidence

**Files:**
- Create: `docs/evidence/equal-runtime-alias.json`
- Modify: `docs/superpowers/specs/2026-08-21-ap-fdr-integrated-redesign.md`

- [ ] **Step 1: Record the immutable mapping**

Store source commit `fca7763679b6e10ed68f98971a362a054ecd4853`, raw method
`ace_fdr`, raw run name `formal-seed0-ace-fdr-v1`, canonical paper method
`equal`, and a statement that checkpoint/log bytes are not renamed.

- [ ] **Step 2: Validate JSON and source status**

Run `python -m json.tool docs/evidence/equal-runtime-alias.json` and
`git diff --check`.
Expected: both exit zero.

- [ ] **Step 3: Commit and push private branch**

Commit with `docs: bind ACE-FDR runtime to EQuAL` and push
`codex/ap-fdr-integrated-redesign` to the private materials remote.

### Task 4: Final verification and live-run check

**Files:**
- Verify only.

- [ ] **Step 1: Run the focused regression suite**

Run the existing 93-test FDR/BPDD compatibility selection plus
`tests/test_train_equal.py`.
Expected: all pass.

- [ ] **Step 2: Confirm the server job remains live**

Confirm PID 83869 is alive, GPU memory is allocated, the log advances, and no
NaN/Inf or traceback appears. Do not restart or rename the remote run directory.

- [ ] **Step 3: Push verified HEAD**

Push the private branch and report the source commit, active epoch, and exact
runtime-to-paper mapping.
