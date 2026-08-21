# ACE-FDR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and launch the parameter-free ACE-FDR Formal100 seed-0 experiment with native RT-DETR references, stock-only DN supervision, and normal-query edge-adaptive FGL.

**Architecture:** Extend the existing FDR criterion with one opt-in edge-weight calculation that reuses adjacent-bin targets and detached IoU.  A dedicated declarative YAML selects the complete ACE-FDR method, while a source-bound launcher records the exact config, initial state, dataset and frozen Formal100 settings before creating the existing `FDRTrainer`.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics 8.4.90, pytest, YAML, Git, Linux/RTX 4090 deployment.

---

### Task 1: Lock the Edge-Adaptive Weight Contract

**Files:**
- Modify: `tests/test_fdr_loss.py`
- Modify: `src/fdr_loss.py`

- [ ] **Step 1: Write failing tests for the missing weight function**

Add tests importing `edge_adaptive_fgl_weights` and asserting:

```python
weights = edge_adaptive_fgl_weights(
    logits,
    target_indices,
    left_weight,
    right_weight,
    matched_iou,
)
assert weights.shape == matched_iou.shape
assert weights.requires_grad is False
```

Use one matched box with four identical edge distributions to assert that all
four weights equal the repeated IoU.  Use another box where one edge assigns
less probability to its adjacent target bins and assert that this edge receives
the larger weight.  Assert every modulation ratio is within `[0.5, 2.0]`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_fdr_loss.py -k edge_adaptive -q
```

Expected: collection fails because `edge_adaptive_fgl_weights` is not exported.

- [ ] **Step 3: Implement the minimal detached edge-weight calculation**

In `src/fdr_loss.py`, add a pure function that computes:

```python
probabilities = corner_logits.detach().softmax(dim=-1)
left_index = target_indices.long().unsqueeze(-1)
right_index = (target_indices.long() + 1).unsqueeze(-1)
target_mass = (
    probabilities.gather(1, left_index).squeeze(1) * left_weight
    + probabilities.gather(1, right_index).squeeze(1) * right_weight
)
difficulty = (1.0 - target_mass).reshape(-1, 4)
mean_difficulty = difficulty.mean(dim=1, keepdim=True).clamp_min(1e-6)
modulation = (difficulty / mean_difficulty).clamp(0.5, 2.0)
return matched_iou.detach() * modulation.reshape(-1)
```

Validate aligned one-dimensional edge tensors and a number of edges divisible
by four.  Export the function through `__all__`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_fdr_loss.py -k edge_adaptive -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the pure loss behavior**

```powershell
git add src/fdr_loss.py tests/test_fdr_loss.py
git commit -m "feat: add ACE-FDR edge-adaptive localization weights"
```

### Task 2: Integrate the Weighting into the Existing FDR Criterion

**Files:**
- Modify: `tests/test_fdr_loss.py`
- Modify: `tests/test_rtdetr_fdr.py`
- Modify: `src/fdr_loss.py`
- Modify: `src/rtdetr_fdr.py`
- Modify: `src/rtdetr_fdr_bpdd.py`

- [ ] **Step 1: Write failing criterion and construction tests**

Add tests showing that `FDRDetectionLoss(edge_adaptive_fgl=True)` uses the new
weighting only for normal-query FGL, while `False` remains numerically identical
to the pinned primitive.  Extend the declarative model helper with
`edge_adaptive_fgl` and assert `model.init_criterion().edge_adaptive_fgl is True`.

- [ ] **Step 2: Run the new tests and verify RED**

```powershell
python -m pytest tests/test_fdr_loss.py tests/test_rtdetr_fdr.py -k "edge_adaptive or criterion_reads" -q
```

Expected: failures because the criterion and model do not accept the option.

- [ ] **Step 3: Wire the option with no inference changes**

Add `edge_adaptive_fgl: bool = False` to `FDRDetectionLoss.__init__`.  In
`_fgl_for_layer`, select the new edge weights only when the option is true;
otherwise retain `matched_iou.repeat_interleave(4)` exactly.  Pass the YAML
option from both `FDRDetectionModel` and `FDRBPDDetectionModel` criterion
constructors so later BPDD integration does not require another interface
change.

- [ ] **Step 4: Verify GREEN and existing FDR compatibility**

```powershell
python -m pytest tests/test_fdr_loss.py tests/test_rtdetr_fdr.py tests/test_bpdd_fdr_criterion.py tests/test_bpdd_fdr_integration.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit criterion integration**

```powershell
git add src/fdr_loss.py src/rtdetr_fdr.py src/rtdetr_fdr_bpdd.py tests/test_fdr_loss.py tests/test_rtdetr_fdr.py
git commit -m "feat: integrate edge-adaptive supervision into ACE-FDR"
```

### Task 3: Add the Indivisible ACE-FDR Configuration

**Files:**
- Create: `configs/rtdetr-l-ace-fdr.yaml`
- Modify: `tests/test_fdr_yaml_configs.py`

- [ ] **Step 1: Write a failing declarative-config test**

Assert that the new configuration preserves the exact stock graph before the
decoder and declares this complete contract:

```python
assert options["preliminary_box"] is False
assert loss == {
    "fgl_weight": 0.15,
    "supervise_pre_boxes": False,
    "supervise_dn_fdr": False,
    "edge_adaptive_fgl": True,
}
```

Also assert the original FDR YAML is byte-semantically unchanged.

- [ ] **Step 2: Run the YAML test and verify RED**

```powershell
python -m pytest tests/test_fdr_yaml_configs.py -k ace_fdr -q
```

Expected: failure because `configs/rtdetr-l-ace-fdr.yaml` does not exist.

- [ ] **Step 3: Create the YAML**

Copy the graph fields from `configs/rtdetr-l-fdr.yaml`; change only the native
reference selection and loss block specified above.  Keep `cumulative: true`,
`reg_max: 32`, `reg_scale: 4.0`, `up: 0.5`, and `private_seed: 10000`.

- [ ] **Step 4: Run YAML and construction tests**

```powershell
python -m pytest tests/test_fdr_yaml_configs.py tests/test_rtdetr_fdr.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the full method configuration**

```powershell
git add configs/rtdetr-l-ace-fdr.yaml tests/test_fdr_yaml_configs.py
git commit -m "feat: declare the integrated ACE-FDR method"
```

### Task 4: Add a Source-Bound Formal100 Launcher

**Files:**
- Create: `scripts/train_ace_fdr.py`
- Create: `tests/test_train_ace_fdr.py`

- [ ] **Step 1: Write failing launcher tests**

Test that the launcher has no method-selection flag, always uses
`configs/rtdetr-l-ace-fdr.yaml`, freezes seed 0 and 100 epochs, refuses dirty
tracked source, validates the FDR initial state, and writes an authority record
containing source/config/initial-state/dataset/settings hashes.

- [ ] **Step 2: Run the launcher tests and verify RED**

```powershell
python -m pytest tests/test_train_ace_fdr.py -q
```

Expected: collection fails because `scripts.train_ace_fdr` does not exist.

- [ ] **Step 3: Implement the minimal launcher**

Reuse `FROZEN_SETTINGS`, `FORMAL_EPOCHS`, `prepare_data_yaml`,
`current_source_identity`, `dataset_signature`, `validate_fdr_initial_state`,
and `FDRTrainer`.  Default the run name to `formal-seed0-ace-fdr-v1`; support
only `--dataset-root`, `--initial-state`, `--output-root`, optional `--name`,
and `--dry-run`.

- [ ] **Step 4: Run launcher and regression tests**

```powershell
python -m pytest tests/test_train_ace_fdr.py tests/test_train_ap_fdr_ablation.py tests/test_train_rtdetr_fdr_cli.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the launcher**

```powershell
git add scripts/train_ace_fdr.py tests/test_train_ace_fdr.py
git commit -m "feat: add source-bound ACE-FDR Formal100 launcher"
```

### Task 5: Verify, Publish Source, and Launch the Server Run

**Files:**
- Verify all files changed above.
- Remote run directory: `/data/uav/runs/ace-fdr-formal100-20260821`

- [ ] **Step 1: Run local verification**

```powershell
python -m pytest tests/test_fdr_loss.py tests/test_fdr_yaml_configs.py tests/test_rtdetr_fdr.py tests/test_bpdd_fdr_criterion.py tests/test_bpdd_fdr_integration.py tests/test_train_ace_fdr.py tests/test_train_ap_fdr_ablation.py tests/test_train_rtdetr_fdr_cli.py -q
python -m compileall -q src scripts
git diff --check
git status --short
```

Expected: all selected tests pass, compileall exits 0, diff check is clean, and
the worktree contains no uncommitted tracked changes.

- [ ] **Step 2: Push the source branch**

```powershell
git push -u origin codex/ap-fdr-integrated-redesign
```

Expected: the exact verified commit is available on GitHub.

- [ ] **Step 3: Inspect the server without changing existing jobs**

Check `nvidia-smi`, active Python processes, disk capacity, the VisDrone dataset,
the frozen FDR initial state and existing repository directories.  Do not stop
or overwrite unrelated runs.

- [ ] **Step 4: Deploy and run a dry launch**

Clone or fetch the pushed branch into a dedicated directory, install only
missing pinned requirements, and run:

```bash
python scripts/train_ace_fdr.py \
  --dataset-root /data/uav/datasets/VisDrone \
  --initial-state /data/uav/protocols/fdr-d97e1eb7/initial-state.pt \
  --output-root /data/uav/runs/ace-fdr-formal100-20260821 \
  --dry-run
```

Expected: authority validation succeeds and reports the ACE-FDR YAML, seed 0,
100 epochs, and the frozen settings.

- [ ] **Step 5: Launch and verify live progress**

Start the same command without `--dry-run` under a persistent background
session, redirect stdout/stderr to a dedicated log, and verify all of:

- the process remains alive after startup;
- the GPU is allocated by the ACE-FDR process;
- `fdr-run`/authority evidence identifies the pushed commit and config hash;
- the log reaches data scanning and the first training epoch without NaN/Inf.

- [ ] **Step 6: Report the exact run identity**

Return the pushed commit, remote checkout, PID/session, log path, run directory,
dataset path, initial-state SHA-256 and first observed training status.  Do not
claim an accuracy result before best-checkpoint evaluation completes.
