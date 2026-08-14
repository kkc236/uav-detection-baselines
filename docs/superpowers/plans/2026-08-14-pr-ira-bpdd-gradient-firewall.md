# PR-IRA + BPDD Gradient Firewall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the YAML-pluggable PR-IRA P3 adapter and a mathematically verified BPDD-to-PR-IRA private-gradient firewall without changing the mature FDR/BPDD equations or the stock Ultralytics optimizer rule.

**Architecture:** `PRIRA` preserves stock P3 and adds a bounded, scheduled, channel-spatial gated local residual. The detector computes the unchanged main and BPDD losses in one forward pass; it captures the unscaled BPDD gradient on PR-IRA-private parameters, while the trainer subtracts the accumulated contribution after AMP unscale and before the existing one-shot global gradient clip. Protocol, checkpoint, evaluation and publication code remain separate from the module math.

**Tech Stack:** Python 3.10.12, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.90, pytest, YAML, RTX 4090.

---

## File map

- Create `src/pr_ira.py`: module math, schedule and runtime diagnostics.
- Modify `src/rtdetr_fdr.py`: register `PRIRA` with the existing YAML parser lock.
- Create `src/rtdetr_fdr_bpdd_pr_ira.py`: graph integration, isolated initial-state mapping, private-gradient capture/buffer/subtraction and trainer optimizer step.
- Create `configs/rtdetr-l-fdr-bpdd-pr-ira.yaml`: complete pluggable graph.
- Create `src/pr_ira_protocol.py`: immutable Screen30/Formal100 authority and gates.
- Create `scripts/prepare_pr_ira_protocol.py`: create-only manifests for paired screens and formal run.
- Create `scripts/train_rtdetr_pr_ira.py`: epoch evidence, checkpoint, resume and publication queue.
- Create `scripts/evaluate_pr_ira_gate.py`: fixed compatibility and independence Screen30 decisions.
- Create `scripts/evaluate_pr_ira_formal.py`: exact EMA validation and efficiency report.
- Create `tests/test_pr_ira.py`: math, bounds, schedule and RNG tests.
- Create `tests/test_bpdd_pr_ira_firewall.py`: analytical and real-graph gradient equivalence tests.
- Create `tests/test_bpdd_pr_ira_integration.py`: YAML, initial state, optimizer, checkpoint and inference contract tests.
- Create `tests/test_pr_ira_protocol.py`: immutable authority and gate tests.
- Create `tests/test_train_rtdetr_pr_ira_cli.py`: create-only epoch evidence and resume tests.

## Task 1: Freeze executable PR-IRA math

**Files:**
- Create: `tests/test_pr_ira.py`
- Create: `src/pr_ira.py`

- [ ] **Step 1: Write failing identity, bounds and schedule tests**

Test `BCHW` shape preservation, `a=0` bit-exact identity, gates in `[0,1]`, finite RMS-normalized residual, fixed `epsilon=1e-6`, and `alpha_max=0.20`. Test exact integer schedule milestones: Screen30 epochs 1–3 identity, 4–9 ramp, 10–30 fully open; Formal100 epochs 1–10 identity, 11–30 ramp, 31–100 fully open. Also reject boolean/non-positive channels, wrong rank, wrong channels and non-floating inputs.

- [ ] **Step 2: Run the focused tests and record the expected import failure**

Run:

```powershell
python -m pytest tests/test_pr_ira.py -q
```

Expected: collection fails because `src.pr_ira` does not exist.

- [ ] **Step 3: Implement the minimal module**

Implement these public contracts:

```python
def relative_open_ratio(epoch: int, epochs: int) -> float: ...

class PRIRA(nn.Module):
    def __init__(
        self,
        channels: int,
        alpha_max: float = 0.20,
        *,
        epsilon: float = 1e-6,
    ) -> None: ...

    def set_training_progress(self, epoch: int, epochs: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
```

Use two repository-owned local residual blocks, per-sample/per-channel spatial RMS, channel and spatial gates from `abs(d_raw)`, scalar `tanh(amplitude)`, and a non-persistent progress buffer. After private-seed initialization, zero both weight and bias of each gate's final layer so both gates are exactly 0.5 initially. Expose detached diagnostics for effective amplitude, gate mean/max and residual RMS ratio.

- [ ] **Step 4: Add RNG and round-trip tests**

Construct the module inside `torch.random.fork_rng`, verify public CPU/CUDA RNG states do not advance, and prove that save/load preserves output and progress exactly.

- [ ] **Step 5: Run and commit the module**

Run:

```powershell
python -m pytest tests/test_pr_ira.py -q
```

Expected: all PR-IRA tests pass.

Commit only `src/pr_ira.py` and `tests/test_pr_ira.py` with message `feat: add protected residual IRA module`.

## Task 2: Add the declarative graph without changing FDR/BPDD

**Files:**
- Modify: `src/rtdetr_fdr.py`
- Create: `configs/rtdetr-l-fdr-bpdd-pr-ira.yaml`
- Create: `src/rtdetr_fdr_bpdd_pr_ira.py`
- Create: `tests/test_bpdd_pr_ira_integration.py`

- [ ] **Step 1: Write graph and initialization tests**

Require exactly one `PRIRA` layer at model index 22 consuming stock P3 index 21; stock P4 must still consume index 21; the FDR Decoder must consume `[22, 25, 28]`. Load the mature FDR initial-state artifact with the same post-insertion key shift as IRA and prove every shared tensor is bit-exact while every missing key belongs to `model.22.`.

- [ ] **Step 2: Verify the tests fail before parser registration**

Run:

```powershell
python -m pytest tests/test_bpdd_pr_ira_integration.py -q
```

Expected: failure because `PRIRA` is not registered and the combined class is absent.

- [ ] **Step 3: Register and integrate PR-IRA**

Import `PRIRA` in `src/rtdetr_fdr.py` and assign `ultralytics_tasks.PRIRA = PRIRA` inside `register_fdr_module()`. Implement `FDRBPDDPRIRADetectionModel` as a subclass of `FDRBPDDDetectionModel`, use private seed `20000 + experiment_seed`, and expose a `.pr_ira` property.

- [ ] **Step 4: Add bit-exact identity and ablation tests**

With shared tensors loaded and amplitude zero, compare FDR+BPDD versus the combined model predictions and all raw decoder tensors using `rtol=0, atol=0`. Verify removing the YAML layer and restoring the P3 index gives the existing FDR+BPDD graph without editing FDR or BPDD source.

- [ ] **Step 5: Run focused and inherited tests**

Run:

```powershell
python -m pytest tests/test_pr_ira.py tests/test_bpdd_pr_ira_integration.py tests/test_bpdd_fdr_integration.py -q
```

Expected: all tests pass and mature FDR/BPDD outputs remain unchanged.

Commit with message `feat: integrate PR-IRA as a YAML P3 layer`.

## Task 3: Expose loss components without changing their values

**Files:**
- Modify: `src/rtdetr_fdr_bpdd_pr_ira.py`
- Modify: `tests/test_bpdd_pr_ira_integration.py`

- [ ] **Step 1: Write a failing loss-decomposition test**

On one deterministic tensor batch, require:

```python
total == main_loss + loss_bpdd
main_loss == sum(value for name, value in last_fdr_losses.items() if name != "loss_bpdd")
loss_bpdd is last_fdr_losses["loss_bpdd"]
```

Use exact tensor identity where possible and `rtol=0, atol=0` for recomposed totals.

- [ ] **Step 2: Implement a compatibility-only decomposition interface**

Do not modify `BPDDDetectionLoss` math. After the inherited loss call, read `last_fdr_losses`, validate exactly one `loss_bpdd`, and expose `last_main_loss` and `last_bpdd_loss` on the combined model for gradient capture.

- [ ] **Step 3: Prove mature BPDD numerics are unchanged**

Instantiate the old FDR+BPDD and new zero-gate model from the same state and batch. Assert each named loss, displayed loss and BPDD statistic is bit-exact.

- [ ] **Step 4: Run and commit**

Run:

```powershell
python -m pytest tests/test_bpdd_pr_ira_integration.py tests/test_bpdd_fdr_integration.py tests/test_bpdd_loss.py -q
```

Expected: all tests pass.

Commit with message `refactor: expose BPDD loss components for firewall`.

## Task 4: Implement and prove the private-gradient firewall

**Files:**
- Create: `tests/test_bpdd_pr_ira_firewall.py`
- Modify: `src/rtdetr_fdr_bpdd_pr_ira.py`

- [ ] **Step 1: Write the three-way analytical gradient test**

For identical model states and batch, compute `main-only`, `main+BPDD without firewall`, and `main+BPDD with firewall`. Require PR-IRA-private gradients from the firewall run to match main-only within `rtol=1e-5, atol=1e-7`, while all non-PR-IRA gradients match main+BPDD. Require at least one FDR distribution-head BPDD gradient to remain non-zero.

- [ ] **Step 2: Implement unscaled FP32 capture and accumulation**

Before returning total loss, call:

```python
torch.autograd.grad(
    loss_bpdd,
    tuple(pr_ira_private_parameters),
    retain_graph=True,
    allow_unused=True,
)
```

Accumulate detached FP32 contributions in a parameter-identity keyed buffer. Reject non-finite values, shape/dtype/device identity drift and capture while a previous optimizer step is half-complete.

- [ ] **Step 3: Override only the optimizer step**

The override must preserve the stock order:

```python
self.scaler.unscale_(self.optimizer)
model.subtract_pr_ira_firewall_buffer()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
self.scaler.step(self.optimizer)
self.scaler.update()
self.optimizer.zero_grad()
model.clear_pr_ira_firewall_buffer()
if self.ema:
    self.ema.update(self.model)
```

Do not use grouped clipping and do not change the optimizer or global max norm.

- [ ] **Step 4: Test eight-microbatch accumulation and AMP scale 128**

Compare one optimizer-window result against a reference that explicitly sums eight unscaled BPDD-private gradients. Require an empty buffer after step, after explicit reset and after checkpoint save. Simulate a failed microbatch and verify reset clears the pending contribution before retry.

- [ ] **Step 5: Run and commit**

Run:

```powershell
python -m pytest tests/test_bpdd_pr_ira_firewall.py tests/test_bpdd_pr_ira_integration.py -q
```

Expected: all gradient-equivalence tests pass.

Commit with message `feat: isolate BPDD gradients from PR-IRA`.

## Task 5: Freeze protocol, optimizer groups and resume authority

**Files:**
- Create: `src/pr_ira_protocol.py`
- Create: `scripts/prepare_pr_ira_protocol.py`
- Create: `tests/test_pr_ira_protocol.py`
- Modify: `src/rtdetr_fdr_bpdd_pr_ira.py`

- [ ] **Step 1: Write immutable protocol tests**

Freeze Ultralytics 8.4.90, dataset hashes, FDR initial-state hash, seed0, fixed 647-image subset, Screen30/Formal100 schedules, `alpha_max=0.20`, private LR multiplier `0.1`, private seed namespace, BPDD options and all gate thresholds from the design spec.

- [ ] **Step 2: Implement create-only manifests and run identities**

Generate distinct identities for `fdr_bpdd`, `fdr_bpdd_pr_ira`, `fdr`, and `fdr_pr_ira` Screen30 arms plus the one eligible Formal100 arm. Reject source, protocol, initial-state, dataset, variant or seed drift on resume.

- [ ] **Step 3: Implement the private optimizer group without changing public groups**

Assign PR-IRA private parameters the common LR multiplied by 0.1 while preserving MuSGD, momentum 0.937, weight decay 0.0005, warmup and all existing parameter grouping semantics. During identity phase, set private grads to `None` before the optimizer step.

- [ ] **Step 4: Prove state and optimizer round trips**

Save and reload exact combined checkpoints. Verify schedule epoch, optimizer state, scaler fixed at 128, EMA, private amplitude, firewall-empty invariant and all run-identity hashes.

- [ ] **Step 5: Run and commit**

Run:

```powershell
python -m pytest tests/test_pr_ira_protocol.py tests/test_bpdd_pr_ira_integration.py tests/test_bpdd_pr_ira_firewall.py -q
```

Expected: all tests pass.

Commit with message `feat: freeze PR-IRA experiment authority`.

## Task 6: Add training evidence and fixed Gate evaluators

**Files:**
- Create: `scripts/train_rtdetr_pr_ira.py`
- Create: `scripts/evaluate_pr_ira_gate.py`
- Create: `tests/test_train_rtdetr_pr_ira_cli.py`

- [ ] **Step 1: Write failing create-only evidence tests**

Require one immutable JSON row per completed epoch with metrics, named losses, common/FDR/PR-IRA gradient norms, BPDD activity, effective amplitude, gate statistics, residual RMS ratio, LR groups, checkpoint/EMA SHA256 and authority identity. Reject gaps, duplicates with changed content and non-finite values.

- [ ] **Step 2: Implement the training CLI**

Reuse the existing FDR/BPDD data preparation, fixed settings, callbacks and publication queue. Add no new dataset or augmentation option. Resume only exact combined checkpoints with an empty firewall buffer.

- [ ] **Step 3: Implement the frozen Screen30 evaluator**

For the compatibility screen, require final/tail3 mAP and AP75 strictly positive, AP50 delta `>= -0.0005`, Precision delta `>= -0.0020`, and tiny or small mAP positive. Apply the same conditions to the independent FDR versus FDR+PR-IRA screen. Emit machine-readable reasons for every pass/fail item.

- [ ] **Step 4: Run CLI and evaluator tests**

Run:

```powershell
python -m pytest tests/test_train_rtdetr_pr_ira_cli.py tests/test_pr_ira_protocol.py -q
```

Expected: all tests pass.

Commit with message `feat: add PR-IRA evidence and fixed gates`.

## Task 7: Pass real CUDA preflight before training

**Files:**
- Create: `scripts/run_pr_ira_preflight.py`
- Create: `tests/test_pr_ira_preflight.py`

- [ ] **Step 1: Add a real-batch preflight test harness**

Verify model graph, exact shared initialization, one VisDrone batch8 forward, main/BPDD decomposition, firewall capture, AMP128 backward, unscale, subtraction, original global clip, MuSGD step, EMA, save/reload and finite diagnostics.

- [ ] **Step 2: Run CPU regression first**

Run:

```powershell
python -m pytest tests/test_pr_ira.py tests/test_bpdd_pr_ira_firewall.py tests/test_bpdd_pr_ira_integration.py tests/test_pr_ira_protocol.py tests/test_train_rtdetr_pr_ira_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run RTX 4090 preflight**

Run the preflight with `device=0`, `batch=8`, AMP scale 128 and the frozen VisDrone authority. Expected report: all P0–P3 checks true, no skipped step, no non-finite value and firewall equivalence within tolerance.

- [ ] **Step 4: Benchmark P4 and freeze hashes**

Measure parameters, GFLOPs and FP16 batch1 latency using warmup50/runs200. Record source, protocol, initial-state and report SHA256. Do not enforce an unmeasured `<1%` overhead claim.

- [ ] **Step 5: Commit preflight evidence code**

Commit with message `test: validate PR-IRA on real CUDA batch`.

## Task 8: Run strict compatibility Screen30

**Files:**
- Generated evidence under a create-only run directory; no source edits are allowed after launch.

- [ ] **Step 1: Launch fresh FDR+BPDD control**

Use the fixed 647-image subset, seed0, 30 epochs, batch8, workers8, MuSGD and the formal initial-state. Publish every epoch checkpoint and JSON without blocking training.

- [ ] **Step 2: Launch fresh FDR+BPDD+PR-IRA method**

Use the identical shared state and public random sequence. Only PR-IRA private parameters may differ.

- [ ] **Step 3: Evaluate the fixed compatibility gate**

Run `scripts/evaluate_pr_ira_gate.py` after both arms reach 30/30 and independent epoch30 validation completes. Freeze the report before inspecting any alternative hyperparameter.

- [ ] **Step 4: Stop on failure or proceed on pass**

If any gate item fails, mark the candidate `scientific_failed` and do not launch Formal100. If all pass, continue to Task 9 without changing thresholds.

## Task 9: Prove independent third-module contribution

**Files:**
- Generated evidence only.

- [ ] **Step 1: Fresh train FDR and FDR+PR-IRA Screen30**

Use the same source, initial state, subset, public random sequence and training protocol as Task 8.

- [ ] **Step 2: Apply the same frozen gate**

If PR-IRA does not independently pass, classify it as a BPDD compatibility component rather than an independent third contribution and stop the three-innovation claim.

- [ ] **Step 3: Freeze the two-screen evidence matrix**

Publish all four arms, final/tail3 metrics, scale results, gate diagnostics and SHA256 values.

## Task 10: Run Formal100 and independent final evaluation

**Files:**
- Create: `scripts/evaluate_pr_ira_formal.py`
- Create: `tests/test_pr_ira_formal_evaluation.py`

- [ ] **Step 1: Test exact checkpoint and evaluator authority**

Accept only exact epoch100 EMA from the frozen graph and reject split, class mapping, preprocessing, source, protocol or checkpoint drift.

- [ ] **Step 2: Fresh train the eligible full model for 100 epochs**

Do not inherit Screen30 weights. Save and publish every epoch with exact resume support.

- [ ] **Step 3: Independently evaluate official val once**

Report P/R/F1/AP50/AP75/mAP, tiny/small/medium/large, ten-class AP/AP50/AP75, parameters, GFLOPs, FP16 median/P95/FPS, peak memory and all hashes.

- [ ] **Step 4: Compare with evidence-level labels**

Use the existing FDR+BPDD authority only as `preliminary_cross_run` until a fresh paired Formal100 control is rerun for the final paper ablation. Never label the cross-run comparison strict.

- [ ] **Step 5: Publish final report and commit**

Run the full regression suite, publish checkpoint/report assets, commit the lightweight evidence summary and record the final remote object IDs.
