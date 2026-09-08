# LRS-GFDR-FIA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and deploy one FIA extension of LRS-GFDR, then start it only after the current LRS-GFDR run completes.

**Architecture:** Reuse the validated FIA graph insertion and gradient partition. The only model change from LRS-GFDR is the P3 FIA branch; feasible geometry and LRS remain unchanged.

**Tech Stack:** Python, PyTorch, Ultralytics RT-DETR, YAML, pytest, SSH.

---

### Task 1: Implement the FIA configuration and trainer

**Files:**
- Create: `configs/rtdetr-l-lrs-gfdr-fia.yaml`
- Modify: `src/rtdetr_lrs_system.py`
- Create: `scripts/train_lrs_gfdr_fia.py`
- Create: `scripts/chain_lrs_gfdr_fia.sh`

- [ ] Add explicit `feasible_geometry: true` to the LRS-GFDR FIA graph.
- [ ] Add `LRSGFDRFIADetectionModel` and `LRSGFDRFIATrainer` with isolated FIA initialization.
- [ ] Add a launcher using the frozen Formal100 settings and an isolated output root.
- [ ] Add a completion-gated chain watcher that refuses to launch FIA after an incomplete or failed LRS-GFDR run.

### Task 2: Verify and document

**Files:**
- Create: `tests/test_lrs_gfdr_fia_module.py`
- Create: `docs/LRS_GFDR_FIA_METHOD_ZH.md`

- [ ] Verify one P3 FIA, P4/P5 bypass, explicit geometry, LRS alpha, and no BPDD.
- [ ] Run focused model/config tests.
- [ ] Commit and push the implementation and method documents.

### Task 3: Deploy and chain the run

**Files:**
- Create remotely: `/root/sj-tmp/runs/visdrone-v2-lrs-gfdr-fia-20260909/`

- [ ] Transfer the new config, source, launcher, and method metadata.
- [ ] Verify the current LRS-GFDR PID has exited successfully and has 100 result rows.
- [ ] Start the FIA run from the original initial state, never from the LRS-GFDR checkpoint.
