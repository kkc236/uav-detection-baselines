# AP-FDR Internal Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one isolated DN-side AP-FDR supervision switch and a reproducible two-run ablation launcher, then collect best-checkpoint val/test evidence for the AP-FDR paper table.

**Architecture:** Keep the model and stock RT-DETR denoising path unchanged. A loss-only boolean gates the extra DN FGL and DN preliminary-box losses; a standalone launcher selects either the existing no-preliminary-reference YAML or the new no-DN-FDR YAML while reusing the frozen FDR training settings.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics 8.4.90, PyYAML, pytest, Git, VisDrone.

---

### Task 1: Add the DN-side loss gate with TDD

**Files:**
- Modify: `tests/test_fdr_loss.py`
- Modify: `src/fdr_loss.py`

- [ ] **Step 1: Write the failing behavior test**

Add a test that constructs `FDRDetectionLoss(..., supervise_dn_fdr=False)`, supplies normal and DN predictions plus corner/pre-box evidence, and asserts:

```python
assert {"loss_fgl", "loss_fgl_aux", "loss_bbox_pre", "loss_giou_pre"}.issubset(losses)
assert not {
    "loss_fgl_dn",
    "loss_fgl_aux_dn",
    "loss_bbox_pre_dn",
    "loss_giou_pre_dn",
}.intersection(losses)
assert {"loss_class_dn", "loss_bbox_dn", "loss_giou_dn"}.issubset(losses)
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/test_fdr_loss.py -k disables_only_extra_dn_fdr_supervision -q`.

Expected: FAIL because DN FDR keys are still produced.

- [ ] **Step 3: Implement the minimal gate**

In `FDRDetectionLoss.__init__`, add `supervise_dn_fdr: bool = True` and store `self.supervise_dn_fdr`. Retain stock DN loss unchanged; wrap only the extra DN block with `if dn_meta is not None and self.supervise_dn_fdr:`.

- [ ] **Step 4: Verify GREEN**

Run `python -m pytest tests/test_fdr_loss.py -q`. Expected: all tests pass.

### Task 2: Make the option declarative and compatible

**Files:**
- Modify: `tests/test_fdr_yaml_configs.py`
- Modify: `tests/test_rtdetr_fdr.py`
- Modify: `tests/test_bpdd_fdr_integration.py`
- Modify: `src/rtdetr_fdr.py`
- Modify: `src/rtdetr_fdr_bpdd.py`
- Create: `configs/rtdetr-l-fdr-no-dn.yaml`
- Modify: all tracked `configs/rtdetr-l-fdr*.yaml` files that contain `fdr_loss`

- [ ] **Step 1: Write failing YAML/model tests**

Require the Full loss mapping to contain `supervise_dn_fdr: true`. Add the new YAML to the ablation set and require its only leaf difference from Full to be `{("fdr_loss", "supervise_dn_fdr"): (True, False)}`. Extend the model and BPDD integration tests to assert the criterion option.

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/test_fdr_yaml_configs.py tests/test_rtdetr_fdr.py tests/test_bpdd_fdr_integration.py -q`.

Expected: FAIL because the YAML key/config and criterion propagation do not yet exist.

- [ ] **Step 3: Implement declarative propagation**

Pass the option from `self.fdr_loss_options` in both FDR and BPDD `init_criterion()` methods. Add `supervise_dn_fdr: true` to every FDR-family YAML containing `fdr_loss`, then create the no-DN YAML as an exact Full copy with only that value set to `false`.

- [ ] **Step 4: Verify GREEN**

Run the same three-file pytest command. Expected: all tests pass.

### Task 3: Add a reproducible formal-ablation launcher

**Files:**
- Create: `tests/test_train_ap_fdr_ablation.py`
- Create: `scripts/train_ap_fdr_ablation.py`

- [ ] **Step 1: Write failing launcher tests**

Require two variants and exact config mapping:

```python
assert VARIANT_CONFIGS == {
    "no_preliminary_reference": ROOT / "configs" / "rtdetr-l-fdr-no-prebox.yaml",
    "no_dn_fdr": ROOT / "configs" / "rtdetr-l-fdr-no-dn.yaml",
}
```

Require `build_settings()` to preserve the frozen formal settings (`imgsz=640`, `batch=8`, `workers=8`, `epochs=100`, `seed=0`, `pretrained=False`, `cache=False`, `deterministic=True`, `optimizer="MuSGD"`) while changing only model path, project/name, and data path. Require `build_launch_record()` to include source identity, config SHA-256, initial-state SHA-256, dataset signature, and the complete settings mapping.

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/test_train_ap_fdr_ablation.py -q`.

Expected: collection FAIL because the launcher module does not exist.

- [ ] **Step 3: Implement the launcher**

Create a CLI with required `--variant`, `--dataset-root`, `--initial-state`, and `--output-root`, plus optional `--name` and `--dry-run`. It validates the VisDrone signature and FDR initial state, rejects dirty tracked source, writes an atomic launch record under `<output-root>/authority/<run-name>.json`, instantiates `FDRTrainer`, and trains unless dry-run is selected.

- [ ] **Step 4: Verify GREEN**

Run `python -m pytest tests/test_train_ap_fdr_ablation.py -q`. Expected: all tests pass.

### Task 4: Verify, commit, and push the source branch

- [ ] Run focused pytest for the five affected test files.
- [ ] Run `python -m compileall -q src scripts` and `git diff --check`.
- [ ] Confirm the two pre-existing untracked user files remain unadded.
- [ ] Commit only scoped files with `feat: add AP-FDR DN supervision ablation`.
- [ ] Push `codex/bpdd-ira-final-eval-d7200906`.

### Task 5: Launch the two formal experiments

- [ ] Preflight GPU availability, free disk, repository commit, Python environment, dataset signature, and initial-state artifact. Never stop another user's process or delete an existing run.
- [ ] Dry-run both variants.
- [ ] Launch `no_preliminary_reference` first and `no_dn_fdr` second in a success-gated sequential shell chain, with distinct append-only logs. Leave the server powered on.
- [ ] Monitor epoch, best epoch, P/R/AP50/AP75/mAP, finite losses, CUDA memory, and checkpoints. Resume only from the run's own `last.pt` and matching launch record.

### Task 6: Evaluate and integrate evidence

- [ ] For each run, select the epoch maximizing val mAP50-95 from its own `results.csv`; record exact row and checkpoint SHA-256.
- [ ] Evaluate each best checkpoint on val and test with `imgsz=640`, `batch=8`, `workers=8`, `conf=0.001`, `max_det=300`, and `nms/cache/half/rect=False`.
- [ ] Update machine JSON/CSV evidence before Markdown; include source commit, config hash, launch record, checkpoint hash, evaluator settings, P/R/AP50/AP75/mAP and per-class metrics.
- [ ] Add AP-FDR Full, w/o preliminary reference, and w/o DN FDR supervision to the internal ablation table, with deltas relative to Full and AP75 emphasized.
- [ ] Run material integrity tests and `git diff --check`, then commit and push private material `main`.

### Task 7: Adversarial paper audit

- [ ] Verify every AP-FDR originality sentence distinguishes repository-owned adaptations from D-FINE primitives.
- [ ] Verify the abstract and contribution list do not rely on incomplete evidence.
- [ ] Verify all val/test and strict/arithmetic deltas are labeled consistently.
- [ ] Verify BPDD remains training-only and FIA remains P3-only in method, tables, and figures.
- [ ] Produce a prioritized residual-risk list; only blocking evidence gaps may defer manuscript finalization.

