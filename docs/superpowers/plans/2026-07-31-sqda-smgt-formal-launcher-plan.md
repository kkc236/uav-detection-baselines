# SQDA-SMGT Formal Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested, resumable 100-epoch formal-stage launcher that is impossible to invoke without strict G2 inventory passage.

**Architecture:** Reuse the established geometry-only training CLI and server runner, adding a `formal` stage with a fixed namespace and epoch budget. The server runner reads G2's strict `selected_checkpoint` rather than the exploratory feasibility field, then preserves existing manifest, resume and checkpoint-sync behavior.

**Tech Stack:** Python 3.10, Bash, pytest, Ultralytics 8.4, deterministic CUDA protocol.

---

### Task 1: Specify formal-stage settings in a failing test

**Files:**
- Modify: `tests/test_sqda_geometry_gate_training.py`
- Modify: `scripts/train_rtdetr_sqda_geometry_gate.py`

- [ ] **Step 1: Add the expected formal namespace and setting case**

```python
assert RUN_NAMES["formal"] == "sqda-geometry-smgt-formal-seed0-100ep"

@pytest.mark.parametrize(("gate", "epochs"), [("formal", 100)])
def test_formal_stage_is_fixed_to_one_hundred_epochs(...):
    ...
    assert build_settings(args)["epochs"] == epochs
```

- [ ] **Step 2: Run the focused test and observe failure**

Run: `pytest tests/test_sqda_geometry_gate_training.py -q`

Expected: fail because `formal` is not an accepted CLI stage or namespace.

- [ ] **Step 3: Add the minimal CLI mapping**

```python
RUN_NAMES = {
    "g1": "sqda-geometry-smgt-g1-seed0-3ep",
    "g2": "sqda-geometry-smgt-g2-seed0-10ep",
    "formal": "sqda-geometry-smgt-formal-seed0-100ep",
}
STAGE_EPOCHS = {"g1": 3, "g2": 10, "formal": 100}
```

Use `STAGE_EPOCHS[args.gate]` in `build_settings`.

- [ ] **Step 4: Run focused test green**

Run: `pytest tests/test_sqda_geometry_gate_training.py -q`

Expected: pass.

### Task 2: Guard the server formal launch with strict G2 evidence

**Files:**
- Modify: `tests/test_sqda_geometry_gate_training.py`
- Modify: `scripts/run_sqda_geometry_gate_server.sh`

- [ ] **Step 1: Add a static behavioral assertion**

```python
assert 'g2_inventory="$project/sqda-geometry-smgt-g2-seed0-10ep/evaluation-inventory/candidate-inventory.json"' in runner
assert 'get("selected_checkpoint")' in runner
assert 'g2_eligible_checkpoint' not in formal_guard_block
```

- [ ] **Step 2: Run focused test and observe failure**

Run: `pytest tests/test_sqda_geometry_gate_training.py -q`

Expected: fail because the runner has no formal guard.

- [ ] **Step 3: Add the minimal strict guard and formal metadata**

```bash
if [[ "$gate" == "formal" ]]; then
  g2_inventory="$project/sqda-geometry-smgt-g2-seed0-10ep/evaluation-inventory/candidate-inventory.json"
  # Require bool(json.load(...).get("selected_checkpoint")); never read the feasibility field here.
fi
```

Map the formal run name to 100 epochs, expand accepted gates, and retain the
existing distinct `smgt-${gate}` log/status/tag/asset naming.

- [ ] **Step 4: Run focused test green**

Run: `pytest tests/test_sqda_geometry_gate_training.py -q`

Expected: pass.

### Task 3: Verify, commit and deploy the dormant launcher

**Files:**
- Modify: `docs/superpowers/specs/2026-07-31-sqda-smgt-formal-launcher-design.md` only if verification exposes a contradiction
- Modify: `docs/superpowers/plans/2026-07-31-sqda-smgt-formal-launcher-plan.md` only to mark completed work

- [ ] **Step 1: Run full verification**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Commit and push**

```bash
git add scripts/train_rtdetr_sqda_geometry_gate.py scripts/run_sqda_geometry_gate_server.sh tests/test_sqda_geometry_gate_training.py docs/superpowers
git commit -m "feat: gate formal SMGT training on strict G2"
git push origin codex/sqda-sgc
```

- [ ] **Step 3: Deploy only the source branch**

Fetch/fast-forward the authorized server repository. Do not invoke the
launcher until the fresh G2 inventory has a non-empty `selected_checkpoint`.
