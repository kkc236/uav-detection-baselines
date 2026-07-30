# SQDA-SMGT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the non-monotone five-feature geometry MLP with a scale-monotone geometry-trust module and evaluate every actual training snapshot before any stage promotion.

**Architecture:** SMGT separates three agreement features from a deterministic reference-scale coordinate. A constrained non-negative scale coefficient ensures geometry trust cannot decrease as reference area increases while leaving semantic residuals and all stock layers untouched. The evaluator gains a checkpoint inventory/selection layer so `best.pt` cannot substitute for an unexamined updated checkpoint.

**Tech Stack:** Python 3.10, PyTorch 2.5, Ultralytics 8.4, pytest, deterministic CUDA evaluation.

---

### Task 1: Lock the failed-G1 evidence into tests

**Files:**
- Modify: `tests/test_sqda_geometry_gate_training.py`
- Modify: `tests/test_sqda_geometry_gate_decision.py`

- [ ] **Step 1: Write failing SMGT tests**

```python
def test_smgt_keeps_semantic_budget_at_one_and_is_monotone_in_reference_scale():
    adapter = SQDASGCAdapter(query_count=2, hidden_dim=256)
    agreement = torch.zeros(1, 2, 3)
    log_size = torch.tensor([[[-4.0, -4.0], [-1.5, -1.5]]])
    budgets = adapter.geometry_trust_budget(agreement, log_size)
    assert budgets[0, 1] >= budgets[0, 0]
```

```python
def test_updated_checkpoint_inventory_excludes_an_initial_best_payload(tmp_path):
    save_geometry_checkpoint(tmp_path / "epoch0.pt", fill=0.0)
    save_geometry_checkpoint(tmp_path / "best.pt", fill=0.0)
    save_geometry_checkpoint(tmp_path / "epoch1.pt", fill=1.0)
    assert select_trainable_candidates(tmp_path) == [tmp_path / "epoch1.pt"]
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `pytest tests/test_sqda_geometry_gate_training.py tests/test_sqda_geometry_gate_decision.py -q`

Expected: failure because no SMGT budget method or trained-checkpoint inventory exists.

### Task 2: Implement the bounded scale-monotone module

**Files:**
- Modify: `src/sqda_sgc.py`
- Modify: `src/rtdetr_sqda_sgc.py`
- Test: `tests/test_sqda_geometry_gate_training.py`

- [ ] **Step 1: Add one focused module boundary**

```python
def geometry_trust_budget(self, agreement: Tensor, log_size: Tensor) -> Tensor:
    log_area = log_size.sum(dim=-1, keepdim=True)
    scale = torch.sigmoid((log_area - self.scale_anchor) / self.scale_temperature)
    logit = self.geometry_agreement(agreement) + F.softplus(self.scale_slope_raw) * scale
    return 0.80 + 0.20 * logit.sigmoid()
```

`geometry_agreement` consumes only the last three existing diagnostic features. `scale_slope_raw` is initialized so its softplus is positive and small. The old semantic budget remains one, and no residual mode other than the learned geometry branch changes.

- [ ] **Step 2: Strictly permit only new SMGT keys when loading the inherited G2 adapter**

```python
permitted_missing = {
    key
    for key in target_state
    if key.startswith("geometry_agreement.") or key == "scale_slope_raw"
}
```

Keep the trained-adapter loader strict: every SMGT key must be present for a post-training candidate.

- [ ] **Step 3: Run focused tests**

Run: `pytest tests/test_sqda_geometry_gate_training.py tests/test_sqda_sgc.py -q`

Expected: all pass, proving bounded output, monotonic scale order, unchanged residual modes, and frozen-scope optimizer grouping.

### Task 3: Evaluate all actual trainable checkpoints

**Files:**
- Create: `src/sqda_geometry_checkpoint_selection.py`
- Modify: `scripts/evaluate_sqda_geometry_gate.py`
- Modify: `scripts/run_sqda_geometry_gate_server.sh`
- Test: `tests/test_sqda_geometry_gate_decision.py`

- [ ] **Step 1: Add a deterministic payload fingerprint and inventory**

```python
def select_trainable_candidates(weights: Path) -> list[Path]:
    """Return updated `epoch*.pt`/`last.pt` payloads in chronological order."""
    # Fingerprint the SMGT state in epoch0.pt, omit equal states, and never select best.pt by name.
```

- [ ] **Step 2: Write one decision record per candidate and a summary**

```python
summary = {
    "candidates": [{"checkpoint": str(path), "passed": decision["passed"]} for path in paths],
    "selected": selected_checkpoint_or_none,
}
```

The selected snapshot is the earliest passing updated checkpoint in deterministic order; no passing checkpoint means G2 cannot launch.

- [ ] **Step 3: Run focused tests**

Run: `pytest tests/test_sqda_geometry_gate_decision.py tests/test_sqda_geometry_diagnosis.py -q`

Expected: all pass, including the initial-`best.pt` regression case.

### Task 4: Full verification, deploy, and repeat the stage gate

**Files:**
- Modify: `docs/superpowers/specs/2026-07-31-sqda-smgt-design.md` only if verification exposes a contradiction
- Modify: `docs/superpowers/plans/2026-07-31-sqda-smgt-implementation.md` only to mark completed tasks

- [ ] **Step 1: Run the full local suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Commit and push the implementation**

```bash
git add src scripts tests docs/superpowers
git commit -m "feat: add scale-monotone geometry trust"
git push origin codex/sqda-sgc
```

- [ ] **Step 3: Pull cleanly on the authorized server and start a new G1 namespace**

Run the existing fixed protocol with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, batch 8, image size 640, AMP fixed scale 128, max detections 300, NMS disabled, and seed 0. Do not overwrite the failed G1 run.

- [ ] **Step 4: After three epochs, run the candidate inventory evaluator**

Require unchanged frozen hashes, finite diagnostics, monotonically ordered scale probe, and every final criterion passing. Publish the run and evaluation summary to GitHub. Launch the independent G2 only if a selected checkpoint passes.
