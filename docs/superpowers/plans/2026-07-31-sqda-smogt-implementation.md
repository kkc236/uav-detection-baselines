# SMOGT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the strict SMGT threshold-metric regression with a semantic-preserving geometry gate and run it in an isolated, auditable namespace.

**Architecture:** Split the inherited geometry residual into semantic-aligned and semantic-orthogonal parts. Preserve the aligned part and apply the existing trainable scale-monotone geometry trust only to the orthogonal part. Fresh stage namespaces ensure that repaired evidence cannot be mixed with failed SMGT evidence.

**Tech Stack:** Python 3.10, PyTorch, Ultralytics 8.4.90, Bash, pytest.

---

### Task 1: Prove the residual split contract

**Files:**
- Modify: `tests/test_sqda_sgc.py`
- Modify: `src/sqda_sgc.py`

- [x] **Step 1: Write failing decomposition tests**

Test that the geometry split reconstructs `g`, that the orthogonal component
has zero dot product with `s`, and that a zero semantic vector leaves all
geometry in the orthogonal component.

- [x] **Step 2: Verify RED**

`/root/data/uav/venv/bin/python -m pytest tests/test_sqda_sgc.py -k semantic_orthogonal -q`
failed with the expected missing `split_geometry_residual` attribute.

- [x] **Step 3: Implement the minimal projection and apply it only in normal SMOGT forward**

Keep `full`, `semantic_only`, and `geometry_only` counterfactual behavior
unchanged. Record the two components in the diagnostic payload.

- [x] **Step 4: Verify GREEN**

The two decomposition tests passed on the remote validation environment.

### Task 2: Isolate repaired experiment stages

**Files:**
- Modify: `scripts/train_rtdetr_sqda_geometry_gate.py`
- Modify: `scripts/run_sqda_geometry_gate_server.sh`
- Modify: `tests/test_sqda_geometry_gate_training.py`

- [x] **Step 1: Write failing namespace and admission tests**

Require `g1r`, `g2r2`, and `formalr`, with G2R2 admitted only by G1R and
formalR admitted only by a strict G2R2 selected checkpoint.

- [x] **Step 2: Verify RED**

The remote test run failed because all three repaired namespaces and guards
were absent.

- [x] **Step 3: Add namespaced launch and synchronization guards**

Use 3, 10, and 100 epoch targets respectively; keep 10 G2R2 checkpoints so
the exact evaluator can inspect every updated snapshot.

- [x] **Step 4: Verify implementation**

Run Bash syntax validation and the focused test groups, then the complete suite
before deployment.

### Task 3: Deploy and stage

**Files:**
- Remote deploy: `/root/data/uav/sqda-sgc`
- Remote run: `/root/data/uav/runs/sqda-geometry-gate/sqda-geometry-smogt-g1r-seed0-3ep`

- [ ] **Step 1: Commit and push verified source changes**
- [ ] **Step 2: Confirm G2R1 inventory has synchronized to GitHub**
- [ ] **Step 3: Deploy the source commit, run GPU smoke checks, and start G1R**
- [ ] **Step 4: Require exact inventory evaluation before G2R2**
