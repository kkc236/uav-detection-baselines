# SQDA Geometry-Trust Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute every checked task in order.

**Goal:** Diagnose the retained SQDA-SGC G2 checkpoint and, only if supported, add a trainable geometry-only trust gate that loads and freezes the inherited adapter.

**Architecture:** Keep the old fusion Linear(2D,D) so its two input-column blocks define r_s and r_g. The new MLP_g receives five scalar geometry-reliability features and produces a_g in (0.80,1). Diagnostic residual modes are forward-local read-only overrides; official runs always use the learned gate. Legacy SQDA-SGC scripts/results remain untouched; new scripts use the sqda-geometry-gate namespace.

**Tech Stack:** Python, PyTorch, Ultralytics, pytest, pycocotools, tidecv when installed, Bash.

---

## Task 1: Finish the fixed threshold audit already started

**Files:**

- Create: src/sqda_error_audit.py
- Create: scripts/audit_sqda_regressions.py
- Create: tests/test_sqda_error_audit.py

- [ ] **Step 1: Verify the existing RED test for comparison deltas**

~~~powershell
python -m pytest tests/test_sqda_error_audit.py -q
~~~

Expected: import failure for compare_error_summaries.

- [ ] **Step 2: Implement compare_error_summaries**

~~~python
def compare_error_summaries(baseline, candidate):
    # Require the same bins and fields.
    # Counts tp/fp/fn use candidate-minus-baseline.
    # mean_tp_score, mean_fp_score and mean_tp_iou are None when either value is None.
~~~

- [ ] **Step 3: Add and test the CLI**

The CLI takes images, labels, baseline-predictions, candidate-predictions and output; it reuses build_coco_dataset, writes fixed confidence_threshold 0.25, iou_threshold 0.50, class-aware score-descending greedy matching and training_signal false. Add a temporary-file CLI test that asserts this exact protocol and a small count delta.

~~~powershell
python -m pytest tests/test_sqda_error_audit.py tests/test_sqda_small_ap.py -q
~~~

Expected: PASS.

- [ ] **Step 4: Commit**

~~~powershell
git add src/sqda_error_audit.py scripts/audit_sqda_regressions.py tests/test_sqda_error_audit.py
git commit -m "feat: audit SQDA precision and scale regressions"
~~~

## Task 2: Define single-sided geometry-gate behavior before code

**Files:**

- Modify: tests/test_sqda_sgc.py
- Modify: tests/test_sqda_sgc_training.py

- [ ] **Step 1: Replace double-budget tests by failing geometry-gate tests**

~~~python
def test_geometry_trust_gate_starts_near_one_and_has_strict_bounds() -> None:
    module = SQDASGCAdapter()
    assert module.geometry_trust[-1].out_features == 1
    assert module.geometry_trust[-1].bias.item() == pytest.approx(math.log(0.90 / 0.10))
    q, boxes, c2 = _inputs(batch=1, queries=4)
    _, d = module(q, boxes, c2)
    assert torch.all((d["geometry_budget"] > 0.80) & (d["geometry_budget"] < 1.0))
    assert d["geometry_budget"].mean().item() == pytest.approx(0.98, abs=0.01)
    assert torch.equal(d["semantic_budget"], torch.ones_like(d["semantic_budget"]))
~~~

- [ ] **Step 2: Add a failing inherited-fusion equivalence test**

Expose an internal helper or diagnostic override such that gate_override=1 uses the original fusion path. With the same q, boxes and C2, assert the pre-saturation residual equals fusion(concat(x_sem,x_geo)) within FP32 tolerance and all legacy fusion state keys are retained.

- [ ] **Step 3: Add failing counterfactual mode tests**

For full, semantic_only, geometry_only and identity overrides, assert no tensor shape/query count changes. Identity must remain bitwise query identity. Semantic-only has zero geometry residual; geometry-only has zero semantic residual. Overrides must be rejected outside the four fixed values.

- [ ] **Step 4: Prove RED**

~~~powershell
python -m pytest tests/test_sqda_sgc.py -q
~~~

Expected: failures because current code has two projectors/two gates and no approved diagnostic modes.

## Task 3: Implement the minimal inherited-fusion geometry gate

**Files:**

- Modify: src/sqda_sgc.py
- Modify: tests/test_sqda_sgc.py
- Modify: tests/test_sqda_sgc_training.py

- [ ] **Step 1: Restore the legacy fusion parameter**

Restore:

~~~python
self.fusion = nn.Linear(2 * dim, dim)
self.geometry_trust = nn.Sequential(
    nn.Linear(5, 16),
    nn.SiLU(),
    nn.Linear(16, 1),
)
~~~

Remove semantic_projector, geometry_projector, agreement norms and agreement_gate. Initialize fusion exactly as before; initialize geometry_trust final weight Normal(0,0.01) and final bias logit(0.90).

- [ ] **Step 2: Split the existing matrix mathematically without changing inherited weights**

~~~python
semantic_input = semantic_modulation.unsqueeze(-1) * semantic_gate * semantic
geometry_input = geometry_gate * geometry
r_s = F.linear(semantic_input, self.fusion.weight[:, :dim], self.fusion.bias)
r_g = F.linear(geometry_input, self.fusion.weight[:, dim:], None)
geometry_features = torch.stack((
    log_size[..., 0], log_size[..., 1],
    F.cosine_similarity(object_queries.detach(), r_g.detach(), dim=-1, eps=1e-6),
    r_g.detach().norm(dim=-1) / object_queries.detach().norm(dim=-1).clamp_min(1e-6),
    F.cosine_similarity(r_s.detach(), r_g.detach(), dim=-1, eps=1e-6),
), dim=-1)
a_g = 0.80 + 0.20 * self.geometry_trust(geometry_features).sigmoid()
f_raw = r_s + a_g * r_g
~~~

Use a_semantic multiplier 1 in normal forward. Retain the existing soft RMS saturation unchanged in this first controlled version. Persist f_raw RMS, saturated RMS, a_g min/max/mean and diagnostic branch residual norms.

- [ ] **Step 3: Implement forward-local diagnostic modes**

Pass residual_mode through adapt_decoder_inputs and SQDASGCDetectionModel only. The four retained-G2 counterfactual modes bypass the new, untrained gate: full uses the original unsplit fusion, semantic_only uses 1/0, geometry_only uses 0/1, and identity returns exactly the original native queries. Official training/evaluation has no diagnostic override and is the only path that uses learned a_g. No diagnostic override may be accepted by trainer settings or public CLI.

- [ ] **Step 4: Restrict the first optimization to the new gate**

Add freeze_inherited_sqda(model): freeze stock and every SQDA parameter except geometry_trust. Add build_geometry_trust_optimizer(model) that contains exactly the geometry_trust parameters. Add tests that an optimization step changes geometry_trust, leaves every inherited SQDA and stock tensor bitwise equal, and produces finite nonzero geometry_trust gradients.

- [ ] **Step 5: Prove GREEN**

~~~powershell
python -m pytest tests/test_sqda_sgc.py tests/test_sqda_sgc_training.py tests/test_rtdetr_sqda_sgc_integration.py -q
~~~

Expected: PASS.

- [ ] **Step 6: Commit**

~~~powershell
git add src/sqda_sgc.py tests/test_sqda_sgc.py tests/test_sqda_sgc_training.py
git commit -m "feat: add inherited SQDA geometry trust gate"
~~~

## Task 4: Load G2 adapter exactly and produce read-only artifacts

**Files:**

- Modify: src/rtdetr_sqda_sgc.py
- Create: scripts/diagnose_sqda_geometry_branches.py
- Modify: tests/test_rtdetr_sqda_sgc_integration.py
- Create: tests/test_sqda_geometry_diagnosis.py

- [ ] **Step 1: Write failing strict-inheritance tests**

Save a fixture checkpoint containing an old adapter state without geometry_trust. Assert loader accepts only missing geometry_trust keys, rejects all other missing/unexpected/mismatched keys, copies every inherited adapter tensor byte-for-byte and records source SHA256.

- [ ] **Step 2: Implement load_inherited_sqda_adapter**

Load source ema/model, extract source.sqda_sgc state, load strict=False and permit exactly geometry_trust missing keys. Return source path/SHA and inherited tensor count. Never load source stock weights through this function.

- [ ] **Step 3: Implement the read-only four-mode CLI**

For each fixed mode, run retained G2 adapter on the same validation images, serialize up to 300 prediction entries per image, run the existing COCO AP evaluator, error audit and Ultralytics PR/P/R/F1 curve export. Emit one manifest carrying checkpoint SHA, mode, metrics and prediction SHA. Invoke tidecv only if import succeeds; write unavailable reason otherwise.

- [ ] **Step 4: Verify and commit**

~~~powershell
python -m pytest tests/test_rtdetr_sqda_sgc_integration.py tests/test_sqda_geometry_diagnosis.py tests/test_sqda_error_audit.py -q
git add src/rtdetr_sqda_sgc.py scripts/diagnose_sqda_geometry_branches.py tests/test_rtdetr_sqda_sgc_integration.py tests/test_sqda_geometry_diagnosis.py
git commit -m "feat: diagnose SQDA geometry branch evidence"
~~~

## Task 5: Run isolated controlled G1 only after diagnostics

**Files:**

- Create: scripts/train_rtdetr_sqda_geometry_gate.py
- Create: scripts/run_sqda_geometry_gate_server.sh
- Create: scripts/continue_sqda_geometry_gate_if_pass.sh
- Create: tests/test_sqda_geometry_gate_training.py

- [ ] **Step 1: Test frozen gate-only settings**

Require adapter-checkpoint, set run name sqda-geometry-gate-g1-seed0-3ep, data order/seed 0, 640, batch 8, max_det 300, NMS false, AMP scale 128. Assert no command-line loss/query/optimizer mutations.

- [ ] **Step 2: Implement a separate trainer path**

Load immutable matched stock baseline plus the supplied retained G2 adapter. Freeze inherited SQDA and stock, train only geometry_trust. Write separate manifests, frozen-tensor audits, gate saturation diagnostics and GitHub results to sqda-geometry-gate-results. Do not change legacy SQDA-SGC scripts, directories or releases.

- [ ] **Step 3: Verify and deploy**

~~~powershell
python -m pytest -q
git diff --check
git push origin codex/sqda-sgc
~~~

On the authorized server: pull branch, run G0, run the four branch diagnostics and inspect their manifests. Start the three-epoch geometry-gate G1 only when the addendum's evidence condition supports it; a passing G1 is then eligible for the separate ten-epoch directional run. The strict gate uses both max-F1 Precision and P/R at the frozen baseline threshold, plus mAP50-95/AP-small tolerance and no lower-bound saturation.

## Plan review

- The plan directly implements the user-selected single geometry gate and preserves the original fusion weights.
- It treats the current reported difference as an unproven trigger, not as a confirmed geometry defect.
- It deliberately excludes a semantic gate, hard RMS-clip change, loss change, query changes, decoder changes, routing and post-processing.
