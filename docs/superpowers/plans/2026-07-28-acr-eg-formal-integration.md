# ACR-EG Formal RT-DETR Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Integrate ACR-EG as a registered query-level `nn.Module` in a YAML-configured RT-DETR wrapper and evaluate it against the sealed mature RT-DETR-L baseline.

**Architecture:** Keep the mature RT-DETR detector as the shared global/local evidence producer. Register `GCQF` under a top-level `ACREGIntegratedRTDETR` wrapper, invoke it from the wrapper's forward path, and expose explicit `ACR-EG-Off` and `GCTE-Off` branches. YAML is the single source for module dimensions, view count, and ablation switches.

**Tech Stack:** Python 3.10, PyTorch 2.5, Ultralytics RT-DETR-L 8.4.90, PyYAML, pytest.

---

### Task 1: YAML configuration and immutable parser

**Files:**
- Create: `configs/rtdetr-l-gcte.yaml`
- Create: `src/acr_eg_integration.py`
- Test: `tests/test_acr_eg_integration.py`

- [x] **Step 1: Write failing configuration and registration tests**

The tests require `load_acr_eg_config`, `ACREGConfig`, and `ACREGIntegratedRTDETR`; the initial run must fail with `ModuleNotFoundError`.

- [x] **Step 2: Confirm the expected red test**

Run:

```powershell
python -m pytest tests/test_acr_eg_integration.py -q
```

Expected: collection fails because `src.acr_eg_integration` does not exist.

- [ ] **Step 3: Implement the YAML parser and top-level wrapper**

The YAML must contain:

```yaml
model: rtdetr-l.yaml
gcte:
  enabled: true
  forward_integration: true
  query_dim: 256
  num_classes: 10
  num_heads: 8
  num_views: 4
  residual_eta: 0.2
  residual_enabled: true
  acr_eg_off: false
  gcte_off: false
```

`ACREGConfig` validates positive dimensions, `num_views == 4`, and bounded
`residual_eta`. `ACREGIntegratedRTDETR` must register both `detector` and
`acr_eg`, and its forward method must call `GCQF` when enabled. Disabled
mode returns the original global evidence object.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest tests/test_acr_eg_integration.py -q
```

Expected: all integration tests pass.

- [ ] **Step 5: Commit**

```powershell
git add configs/rtdetr-l-gcte.yaml src/acr_eg_integration.py tests/test_acr_eg_integration.py
git commit -m "Integrate ACR-EG through YAML-configured RT-DETR wrapper"
```

### Task 2: Formal entrypoint consumes YAML and mature baseline

**Files:**
- Modify: `scripts/train_gcte_formal.py`
- Test: `tests/test_gcte_formal_cli.py`

- [ ] **Step 1: Add a failing CLI assertion**

Assert that `--config configs/rtdetr-l-gcte.yaml` sets `gcte.enabled`,
`forward_integration`, and `acr_eg_off` in the protocol manifest.

- [ ] **Step 2: Run the focused CLI test and verify it fails**

```powershell
python -m pytest tests/test_gcte_formal_cli.py -q
```

Expected: parser rejects the missing `--config` argument.

- [ ] **Step 3: Implement config loading and baseline checkpoint mode**

Add `--config`, `--baseline-checkpoint`, and `--module-checkpoint`.
Reject `pretrained=True`; require the baseline checkpoint SHA recorded in the
protocol. The protocol must record the YAML SHA-256, ACR-EG config, detector
checkpoint SHA-256, and module checkpoint SHA-256.

- [ ] **Step 4: Run CLI tests**

```powershell
python -m pytest tests/test_gcte_formal_cli.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/train_gcte_formal.py tests/test_gcte_formal_cli.py
git commit -m "Load ACR-EG formal protocol from YAML and mature baseline"
```

### Task 3: Independent evaluation and checkpoint evidence

**Files:**
- Create: `scripts/evaluate_acr_eg_integrated.py`
- Create: `tests/test_evaluate_acr_eg_integrated.py`
- Modify: `docs/handoffs/2026-07-28-acr-eg-round1-progress.md`

- [ ] **Step 1: Write failing evaluator tests**

Cover Global, Fixed-SADED, ACR-EG-On, ACR-EG-Off, and residual-off states;
verify exact baseline and module SHA values are included.

- [ ] **Step 2: Run tests and verify the missing evaluator failure**

```powershell
python -m pytest tests/test_evaluate_acr_eg_integrated.py -q
```

Expected: import failure for the missing evaluator.

- [ ] **Step 3: Implement the evaluator**

Reuse `scripts/evaluate_gcqf_g0.py` route and metric functions. Require the
sealed validation manifest and mature baseline SHA. Write absolute metrics,
Global-relative deltas, Fixed-relative diagnostics, gate booleans, latency,
candidate counts, and checkpoint hashes.

- [ ] **Step 4: Run focused evaluator tests**

```powershell
python -m pytest tests/test_evaluate_acr_eg_integrated.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/evaluate_acr_eg_integrated.py tests/test_evaluate_acr_eg_integrated.py docs/handoffs/2026-07-28-acr-eg-round1-progress.md
git commit -m "Add independent ACR-EG integrated evaluation"
```

### Task 4: Regression and server run

**Files:**
- Modify: `docs/handoffs/2026-07-28-acr-eg-round1-progress.md`
- Create: `artifacts/acr-eg-integrated-<commit>/RESULTS.md`

- [ ] **Step 1: Run the complete local suite**

```powershell
python -m pytest -q
git diff --check
```

Expected: zero test failures and no whitespace errors.

- [ ] **Step 2: Package and deploy exact source**

Create a new source directory and output directory on `36.103.199.151`; do
not overwrite prior formal output. Upload the exact commit archive and verify
its SHA-256.

- [ ] **Step 3: Run one-epoch forward/backward smoke**

Require non-zero ACR-EG gradients, detector and module keys in the checkpoint,
and a YAML protocol manifest with the mature baseline SHA.

- [ ] **Step 4: Run fixed seed0 diagnostic**

Use the sealed 548-image validation set and the existing 10-epoch module
protocol. Do not run seed1/seed2. Record all five states and unchanged gates.

- [ ] **Step 5: Publish evidence**

Download the evaluation JSON, protocol, logs, module checkpoint, and hashes
to `artifacts/acr-eg-integrated-<commit>`. Write `RESULTS.md` with the
primary Global comparison and separate Fixed-SADED internal diagnostics.
