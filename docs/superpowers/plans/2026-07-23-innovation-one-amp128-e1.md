# Innovation One AMP128 E1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the shared AMP128-v2 numerical protocol, revalidate TSGR causality, and produce a valid three-seed E1 effectiveness decision without mixing AMP256 evidence.

**Architecture:** Centralize the E1 controlled scale as one constant used by the CLI, protocol generator, manifest validator, and numerical audit. Rebuild every signed artifact from a new commit, run A0/H0/H1 causal preflight, then run six alternating control/TSGR jobs into an immutable new project. Conditional redesign and the 30/100-epoch stages remain separate plans selected by the E1 decision.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, CUDA on RTX 4090, pytest, JSONL/SHA256 manifests, Bash runner.

---

## File Map

- Modify `scripts/train_rtdetr_ebc_qp.py`: define and enforce the shared E1 AMP128 constant and validate all evidence against it.
- Modify `scripts/prepare_ebc_qp_protocol.py`: sign AMP128 and the no-growth interval into new protocols.
- Modify `scripts/audit_ebc_qp_aux_causality.py`: permit the same AMP128 contract for A0/H0/H1 100-step audits.
- Modify `tests/test_ebc_qp_cli.py`: lock CLI/runtime scale to 128 and reject 256.
- Modify `tests/test_ebc_qp_protocol.py`: lock signed experiment payload to AMP128.
- Modify `tests/test_ebc_qp_causal_audit.py`: lock numerical-audit naming/config to AMP128.
- Modify `tests/test_ebc_qp_e1_artifacts.py`: reject manifests containing scale drift.
- Create runtime artifacts under new server paths only; do not modify old AMP256 evidence.

### Task 1: Centralize the AMP128-v2 Contract

**Files:**
- Modify: `tests/test_ebc_qp_cli.py`
- Modify: `tests/test_ebc_qp_e1_artifacts.py`
- Modify: `scripts/train_rtdetr_ebc_qp.py`

- [ ] **Step 1: Write failing CLI and evidence tests**

Add tests that parse E1 with `--controlled-amp-scale 128`, reject 256 in
`validate_protocol()`, and require every optimizer evidence record to contain
`amp_scale_before == amp_scale_after == 128.0`.

```python
def test_e1_requires_fixed_amp128():
    args = build_parser().parse_args(
        ["--stage", "e1", "--arm", "control", "--controlled-amp-scale", "128"]
    )
    validate_protocol(args)
    args.controlled_amp_scale = 256.0
    with pytest.raises(SystemExit, match="scale 128"):
        validate_protocol(args)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_ebc_qp_cli.py tests/test_ebc_qp_e1_artifacts.py -q
```

Expected: failure because the parser and manifest validator still require 256.

- [ ] **Step 3: Implement one shared constant**

In `scripts/train_rtdetr_ebc_qp.py`, define:

```python
E1_CONTROLLED_AMP_SCALE = 128.0
E1_CONTROLLED_AMP_GROWTH_INTERVAL = 2**31 - 1
```

Use the constant for the parser choice, `validate_protocol()`,
`write_e1_run_manifest()`, and the controlled-AMP manifest fields. Reject any
record whose before or after scale differs from 128.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same command. Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_rtdetr_ebc_qp.py tests/test_ebc_qp_cli.py tests/test_ebc_qp_e1_artifacts.py
git commit -m "Freeze E1 on shared AMP128"
```

### Task 2: Sign AMP128 Into Protocol Artifacts

**Files:**
- Modify: `tests/test_ebc_qp_protocol.py`
- Modify: `scripts/prepare_ebc_qp_protocol.py`

- [ ] **Step 1: Write the failing signed-payload test**

Assert the frozen payload contains:

```python
assert payload["training"]["controlled_amp_scale"] == 128.0
assert payload["training"]["controlled_amp_growth_interval"] == 2**31 - 1
assert payload["training"]["save_period"] == -1
assert payload["training"]["retained_zero_based_epoch_checkpoints"] == [7, 8, 9]
```

- [ ] **Step 2: Run the protocol test and verify RED**

Run `python -m pytest tests/test_ebc_qp_protocol.py -q`.

Expected: the old payload reports 256.

- [ ] **Step 3: Import and use the shared constants**

Replace the literal scale/growth fields in `prepare_ebc_qp_protocol.py` with
`E1_CONTROLLED_AMP_SCALE` and `E1_CONTROLLED_AMP_GROWTH_INTERVAL` imported from
the launcher. Do not change dataset, subset, seed, optimizer, augmentation,
TSGR configuration, or checkpoint retention.

- [ ] **Step 4: Run the protocol and CLI tests**

Run:

```bash
python -m pytest tests/test_ebc_qp_protocol.py tests/test_ebc_qp_cli.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_ebc_qp_protocol.py tests/test_ebc_qp_protocol.py
git commit -m "Sign AMP128 into E1 protocols"
```

### Task 3: Revalidate A0/H0/H1 at AMP128

**Files:**
- Modify: `tests/test_ebc_qp_causal_audit.py`
- Modify: `scripts/audit_ebc_qp_aux_causality.py`

- [ ] **Step 1: Write failing audit-contract tests**

Parse `--controlled-amp-scale 128 --controlled-amp-steps 100` for arms A0, H0,
and H1. Assert run names contain `controlled-amp128-100step`, the audit config
requires zero skips, and growth interval is `2**31 - 1`.

- [ ] **Step 2: Run the audit tests and verify RED**

Run `python -m pytest tests/test_ebc_qp_causal_audit.py -q`.

Expected: parser rejects 128 or config reports the old growth interval.

- [ ] **Step 3: Implement the matching audit contract**

Import the shared constants and set:

```python
parser.add_argument(
    "--controlled-amp-scale",
    type=float,
    choices=(E1_CONTROLLED_AMP_SCALE,),
)
```

Return the shared no-growth interval from `controlled_amp_config()`.

- [ ] **Step 4: Run focused and full EBC tests**

Run:

```bash
python -m pytest tests/test_ebc_qp_causal_audit.py tests/test_ebc_qp_*.py tests/test_rtdetr_ebc_qp_integration.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_ebc_qp_aux_causality.py tests/test_ebc_qp_causal_audit.py
git commit -m "Audit TSGR causality at AMP128"
```

### Task 4: Freeze Failure and Supersession Evidence

**Files:**
- Create only server-side JSON records under `/home/ubuntu/tsgr-p2-e1-8cb70c50/`.

- [ ] **Step 1: Hash the old evidence**

Record SHA256 and byte size for seed0 manifests/results/query diagnostics,
seed1 failure JSONL, supervisor log, protocol manifests, and Git commit.

- [ ] **Step 2: Write immutable status records**

Write `SUPERSEDED_AMP256.json` for seed0 and
`INVALID_OVERFLOW_ATTEMPT25.json` for seed1 using temporary files plus atomic
rename. Each record contains `eligible_for_comparison: false` and the reason.

- [ ] **Step 3: Verify exact failure fields**

Require attempt 25 to report scale `256 -> 128`, non-finite stock fields,
finite auxiliary gradients, `P2=0`, and `stock=300`.

- [ ] **Step 4: Reclaim only invalid large checkpoints**

After hashes and status records are verified, delete only explicitly enumerated
old AMP256 `.pt` files. Preserve manifests, CSV, diagnostics, logs, failure
JSONL, and status records. Report removed paths and bytes.

### Task 5: Verify the Exact Commit on RTX 4090

**Files:**
- No additional source changes.

- [ ] **Step 1: Run `git diff --check` and the full server EBC suite**

Run:

```bash
/mnt/uav/venv/bin/python -m pytest tests/test_ebc_qp_*.py tests/test_rtdetr_ebc_qp_integration.py -q
```

Expected: all tests pass on the exact clean commit.

- [ ] **Step 2: Run 100-step A0/H0/H1 AMP128 audits**

Use the same seed-0 initial state, protocol, hashed subset, batch, workers, and
device. Each JSON must report exactly 100 attempts, zero skips, constant 128,
finite gradients, and the expected H0/H1 boundary.

- [ ] **Step 3: Compare audits**

Run the existing causal comparator and require TSGR routing only to model 0/1,
nonzero H1 routed/private contributions, zero H0 routed contribution, stock
clipping independence, and no update-monitor abort.

### Task 6: Rebuild Immutable AMP128 Protocols

**Files:**
- Create a new server protocol directory named with the new commit.

- [ ] **Step 1: Generate seeds 0/1/2**

Run `prepare_ebc_qp_protocol.py` three times into one empty AMP128 directory.

- [ ] **Step 2: Validate every signature and fingerprint**

Require one shared experiment signature, dataset/subset/data hashes, environment,
and Git commit; require distinct seed-specific common/innovation fingerprints.

- [ ] **Step 3: Run a production one-step preflight**

For both control and TSGR, require scale 128, zero skip, finite gradients,
TSGR query 300/0, and nonzero routed contribution.

### Task 7: Run Six Fresh AMP128 E1 Arms

**Files:**
- Create one new root-filesystem project and supervisor log.

- [ ] **Step 1: Preflight storage and launch guards**

Require a clean server worktree, exact commit, empty project, matching protocol,
and at least 7.5 GB free. Stop before any run when free space falls below 2 GB.

- [ ] **Step 2: Launch alternating arms**

Run seed0 control/TSGR, seed1 TSGR/control, and seed2 control/TSGR. Each run must
produce exactly 10 result rows, exactly 145 optimizer attempts, zero skips,
exact epoch7/8/9 resumable checkpoints, and an immutable run manifest.

- [ ] **Step 3: Produce stock diagnostics**

For raw epochs 7/8/9 of every arm, generate fixed stock Top-300 tiny coverage
and normalized-rank JSON. Hash every output.

- [ ] **Step 4: Execute the frozen comparator**

Run `compare_tsgr_e1.py`. If any evidence/pairing gate fails, invalidate the
affected pair before interpretation. Otherwise emit `E1_PASS` or `E1_FAIL`
without changing thresholds.

### Task 8: Conditional Handoff

**Files:**
- Create the next design/plan only after E1 classification.

- [ ] **Step 1: Obtain independent B and C reviews**

Give both agents the same six manifests, comparison JSON, evidence JSONL, and
query diagnostics. B audits causality; C proposes the conditional experiment
tree. Neither edits production files.

- [ ] **Step 2: Select exactly one branch**

If `E1_FAIL`, write an attribution-matrix plan that identifies the earliest
reproducible uplift source before implementing a new candidate. If `E1_PASS`,
write a clean-path plan that removes/rejects toxic features while preserving
only the passing mechanism.

- [ ] **Step 3: Do not launch 100 epochs**

The selected candidate must first pass the separate 30-epoch accumulating-
advantage gate and stock-export equivalence specified in the design.
