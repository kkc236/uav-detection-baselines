# AC-BPDD-R v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add one single-factor residual-gradient AC-BPDD candidate and start its strictly paired VisDrone Formal100 run.

**Architecture:** Reuse `LRSFDRBPDDTrainer` with an explicit candidate YAML. Add a dedicated launcher so the existing G-arm identity and files remain immutable. The launcher reuses the frozen protocol, runtime recorder, initial-state validator, clean-worktree guard, and authority record conventions.

**Tech Stack:** Python, PyTorch, Ultralytics RT-DETR, YAML, pytest, SSH.

---

### Task 1: Lock the candidate contract in tests

**Files:**
- Create: `tests/test_ac_bpdd_residual_candidate.py`
- Reference: `tests/test_bpdd_joint_mechanism.py`
- Reference: `tests/test_lrs_system_launcher.py`

- [x] Assert the candidate changes only `residual_gradient_only` relative to the G configuration.
- [x] Assert LRS alpha and feasible geometry remain enabled.
- [x] Assert same-seed initialized candidate and G model states are identical.
- [x] Assert launcher identity, config, epoch count, safe name, and frozen settings.
- [x] Run the new tests first and observe the expected missing-artifact failures.

### Task 2: Implement the isolated candidate

**Files:**
- Create: `configs/rtdetr-l-lrs-gfdr-ac-bpdd-residual.yaml`
- Create: `scripts/train_lrs_gfdr_ac_bpdd_residual.py`

- [x] Copy the effective G graph and make feasible geometry explicit.
- [x] Keep KL and existing AC-BPDD gate/settings; enable only residual-gradient isolation.
- [x] Build a dedicated Formal100 launcher and versioned launch-authority record.
- [x] Re-run candidate, BPDD mechanism, launcher, and LRS system tests.

### Task 3: Freeze, deploy, and launch

**Files:**
- Remote worktree: `/root/sj-tmp/lrs-v2/ac-bpdd-residual-v1-code`
- Remote output: `/root/sj-tmp/runs/visdrone-v2-ac-bpdd-residual-20260909/`

- [ ] Commit and push the reviewed source.
- [ ] Create an isolated clean remote worktree at the exact commit.
- [ ] Run a dry run and verify source/config/dataset/initial-state identities.
- [ ] Verify same-seed initialization equality against G on the server.
- [ ] Start the 100-epoch job from the original seed-0 initial state.
- [ ] Confirm the process remains alive, GPU work begins, and the authority/log paths exist.
