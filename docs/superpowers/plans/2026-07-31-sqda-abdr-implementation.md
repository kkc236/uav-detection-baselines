# SQDA-ABDR Double-Bounded Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Implement Agreement-Bounded Directional Residual (ABDR) with independent nonzero semantic/geometry budgets while preserving every frozen RT-DETR control.

**Architecture:** Retain all SQDA sampling, context modulation, 16-group two-way gate, 300 native queries, decoder hook, loss and post-processing. Replace only final fusion by independent projectors, a detached query-relative decomposition, an agreement MLP, RMS bounding and the existing single LayerScale. Add a read-only fixed-threshold error audit.

**Tech Stack:** Python, PyTorch, Ultralytics, pytest, pycocotools, Bash.

---

## File map

- src/sqda_sgc.py: core ABDR network layer and diagnostics.
- tests/test_sqda_sgc.py; tests/test_sqda_sgc_training.py: module, gradient and frozen-stock contracts.
- src/sqda_error_audit.py; scripts/audit_sqda_regressions.py; tests/test_sqda_error_audit.py: TP/FP/FN evidence-only audit.
- scripts/train_rtdetr_sqda_sgc.py and three server scripts: isolated ABDR names, metrics and publishing.

### Task 1: Test the approved network behavior first

**Files:**

- Modify: tests/test_sqda_sgc.py
- Modify: tests/test_sqda_sgc_training.py

- [ ] **Step 1: Add a failing budget/initialization test**

~~~python
def test_abdr_budget_initialization_and_strict_bounds() -> None:
    module = SQDASGCAdapter()
    assert module.agreement_gate[0].in_features == 5 * 256 + 5
    assert module.agreement_gate[-1].out_features == 2
    assert module.agreement_gate[-1].bias.tolist() == pytest.approx(
        [math.log(0.60 / 0.40), math.log(0.90 / 0.10)], abs=1e-6
    )
    q, boxes, c2 = _inputs(batch=1, queries=4)
    _, d = module(q, boxes, c2)
    assert torch.all((d["semantic_budget"] > 0.95) & (d["semantic_budget"] < 1.0))
    assert torch.all((d["geometry_budget"] > 0.80) & (d["geometry_budget"] < 1.0))
    assert d["semantic_budget"].mean().item() == pytest.approx(0.98, abs=0.01)
    assert d["geometry_budget"].mean().item() == pytest.approx(0.98, abs=0.01)
~~~

- [ ] **Step 2: Add a failing orthogonality test**

~~~python
def test_abdr_directional_factorization_is_orthogonal() -> None:
    module = SQDASGCAdapter()
    q, boxes, c2 = _inputs(batch=1, queries=5)
    _, d = module(q, boxes, c2)
    assert torch.allclose(
        (d["geometry_direction"] * d["query_direction"]).sum(-1),
        torch.zeros_like(d["geometry_direction"][..., 0]), atol=2e-5
    )
    assert torch.allclose(
        (d["semantic_direction"] * d["geometry_direction"]).sum(-1),
        torch.zeros_like(d["geometry_direction"][..., 0]), atol=2e-5
    )
~~~

- [ ] **Step 3: Extend the existing frozen-stock one-step test**

Replace fusion.weight in representative_branches with:

~~~python
"semantic_projector.weight",
"geometry_projector.weight",
"agreement_gate.0.weight",
"agreement_gate.2.weight",
~~~

- [ ] **Step 4: Prove RED**

~~~powershell
pytest tests/test_sqda_sgc.py::test_abdr_budget_initialization_and_strict_bounds tests/test_sqda_sgc.py::test_abdr_directional_factorization_is_orthogonal tests/test_sqda_sgc_training.py::test_one_step_changes_adapter_but_not_stock_parameters_or_buffers -q
~~~

Expected: FAIL because no ABDR layer exists.

### Task 2: Add ABDR without touching frozen controls

**Files:**

- Modify: src/sqda_sgc.py
- Test: tests/test_sqda_sgc.py
- Test: tests/test_sqda_sgc_training.py

- [ ] **Step 1: Replace only self.fusion in SQDASGCAdapter.__init__**

~~~python
self.semantic_projector = nn.Linear(dim, dim)
self.geometry_projector = nn.Linear(dim, dim)
self.agreement_query_norm = nn.LayerNorm(dim)
self.agreement_evidence_norm = nn.LayerNorm(dim)
self.agreement_gate = nn.Sequential(
    nn.Linear(5 * dim + 5, 64), nn.SiLU(), nn.Linear(64, 2),
)
~~~

Do not change self.gate, raw-C2 roles, query count/order, references, decoder hook, loss, NMS, input size or optimizer.

- [ ] **Step 2: Set the approved initialization**

~~~python
for projector in (self.semantic_projector, self.geometry_projector):
    nn.init.normal_(projector.weight, mean=0.0, std=0.01)
    nn.init.zeros_(projector.bias)
nn.init.normal_(self.agreement_gate[-1].weight, mean=0.0, std=0.01)
with torch.no_grad():
    self.agreement_gate[-1].bias.copy_(torch.tensor(
        [_inverse_sigmoid_probability(0.60), _inverse_sigmoid_probability(0.90)],
        dtype=self.agreement_gate[-1].bias.dtype,
    ))
~~~

- [ ] **Step 3: Replace the current fusion input/projection by the exact equations**

~~~python
semantic_evidence = semantic_modulation.unsqueeze(-1) * semantic_gate * semantic
geometry_evidence = geometry_gate * geometry
z_s = self.semantic_projector(semantic_evidence)
z_g = self.geometry_projector(geometry_evidence)
query_direction = F.normalize(object_queries.detach(), dim=-1, eps=1e-6)
semantic_direction = (z_s * query_direction).sum(-1, keepdim=True) * query_direction
geometry_direction = z_g - (z_g * query_direction).sum(-1, keepdim=True) * query_direction
aq = self.agreement_query_norm(object_queries)
a_s = self.agreement_evidence_norm(semantic)
a_g = self.agreement_evidence_norm(geometry)
agreement_input = torch.cat((
    aq, a_s, a_g, a_s * a_g, (a_s - a_g).abs(),
    semantic_similarity.unsqueeze(-1), geometry_similarity.unsqueeze(-1),
    context_similarity.unsqueeze(-1), log_size,
), dim=-1)
agreement = self.agreement_gate(agreement_input).sigmoid()
semantic_budget = 0.95 + 0.05 * agreement[..., :1]
geometry_budget = 0.80 + 0.20 * agreement[..., 1:]
fused = semantic_budget * semantic_direction + geometry_budget * geometry_direction
~~~

Keep the existing RMS bound, role-validity mask, single LayerScale and residual addition.

- [ ] **Step 4: Publish detached diagnostics**

~~~python
"query_direction": query_direction.detach(),
"semantic_direction": semantic_direction.detach(),
"geometry_direction": geometry_direction.detach(),
"semantic_budget": semantic_budget.detach(),
"geometry_budget": geometry_budget.detach(),
~~~

- [ ] **Step 5: Prove GREEN**

~~~powershell
pytest tests/test_sqda_sgc.py tests/test_sqda_sgc_training.py tests/test_rtdetr_sqda_sgc_integration.py -q
~~~

Expected: PASS, including bitwise G0 identity and frozen stock checks.

- [ ] **Step 6: Commit**

~~~powershell
git add src/sqda_sgc.py tests/test_sqda_sgc.py tests/test_sqda_sgc_training.py
git commit -m "feat: add ABDR dual bounded residual budgets"
~~~

### Task 3: Add the fixed evidence-only decline audit

**Files:**

- Create: src/sqda_error_audit.py
- Create: scripts/audit_sqda_regressions.py
- Create: tests/test_sqda_error_audit.py

- [ ] **Step 1: Write failing synthetic tests**

Build a 640-pixel dataset with one small, medium, large GT. Match small; create an unmatched medium-size detection; leave large unmatched:

~~~python
assert report["small"]["tp"] == 1
assert report["medium"]["fp"] == 1
assert report["large"]["fn"] == 1
assert report["small"]["mean_tp_score"] == pytest.approx(0.90)
assert report["small"]["mean_tp_iou"] == pytest.approx(1.0)
~~~

Add two same-class predictions on one GT and require higher score to become the only TP.

- [ ] **Step 2: Prove RED**

~~~powershell
pytest tests/test_sqda_error_audit.py -q
~~~

Expected: FAIL because the audit does not exist.

- [ ] **Step 3: Implement deterministic matching**

Implement summarize_detection_errors(dataset, predictions, confidence_threshold=0.25):

~~~python
# Per image/category, sort score >= 0.25 predictions descending.
# A prediction is TP iff it has IoU >= 0.50 with the best unmatched same-class GT.
# Others are FP; remaining GTs are FN.
# TP/FN use GT original-pixel area; FP uses prediction bbox area.
# all/small/medium/large report tp, fp, fn, mean_tp_score, mean_fp_score, mean_tp_iou.
# Empty means are None, never NaN.
~~~

The CLI reuses build_coco_dataset and emits:

~~~json
{"protocol":{"confidence_threshold":0.25,"iou_threshold":0.5,"matching":"class-aware score-descending greedy","training_signal":false},"baseline":{},"candidate":{},"delta":{}}
~~~

Counts use candidate minus baseline; means are null if either operand is null.

- [ ] **Step 4: Prove GREEN and commit**

~~~powershell
pytest tests/test_sqda_error_audit.py tests/test_sqda_small_ap.py -q
git add src/sqda_error_audit.py scripts/audit_sqda_regressions.py tests/test_sqda_error_audit.py
git commit -m "feat: audit SQDA precision and scale regressions"
~~~

### Task 4: Isolate ABDR runs and publish its diagnostics

**Files:**

- Modify: scripts/train_rtdetr_sqda_sgc.py
- Modify: scripts/run_sqda_sgc_server.sh
- Modify: scripts/supervise_sqda_sgc_server.sh
- Modify: scripts/continue_sqda_sgc_if_pass.sh
- Modify: tests/test_sqda_sgc_training.py

- [ ] **Step 1: Add failing stage/diagnostic tests**

Assert the g1 name is sqda-abdr-g1-seed0-3ep. Invoke record_epoch_diagnostics with ABR tensors and assert JSONL fields semantic_budget_mean, geometry_budget_mean, semantic_direction_norm_mean, geometry_direction_norm_mean.

- [ ] **Step 2: Prove RED**

~~~powershell
pytest tests/test_sqda_sgc_training.py -q
~~~

Expected: FAIL because SQDA-SGC stage names/fields remain.

- [ ] **Step 3: Implement isolated names and paths**

~~~python
RUN_NAMES = {
    "g1": "sqda-abdr-g1-seed0-3ep",
    "g1r": "sqda-abdr-g1r-seed0-3ep",
    "g2": "sqda-abdr-g2-seed0-10ep",
    "formal": "sqda-abdr-formal-seed0-100ep",
}
~~~

Persist budgets mean/max; persist directions through norm(dim=-1) mean/max. Preserve all prior diagnostics and epoch clamp. In Bash retain source checkout at /root/data/uav/sqda-sgc but use /root/data/uav/runs/sqda-abdr, /root/.config/sqda-abdr/github-token, sqda-abdr-<gate>-live, sqda-abdr-results and sqda-abdr-<gate> asset prefix. The continuation script reads sqda-abdr-g2-seed0-10ep/final-gate-decision.json and still demands strict_pass=true.

- [ ] **Step 4: Verify and commit**

~~~powershell
pytest tests/test_sqda_sgc_training.py tests/test_sqda_sgc.py tests/test_rtdetr_sqda_sgc_integration.py -q
bash -n scripts/run_sqda_sgc_server.sh scripts/supervise_sqda_sgc_server.sh scripts/continue_sqda_sgc_if_pass.sh
git add scripts/train_rtdetr_sqda_sgc.py scripts/run_sqda_sgc_server.sh scripts/supervise_sqda_sgc_server.sh scripts/continue_sqda_sgc_if_pass.sh tests/test_sqda_sgc_training.py
git commit -m "ops: isolate ABDR training stages and diagnostics"
~~~

### Task 5: Verify, deploy, and enforce gates

**Files:**

- Verify: all files above
- Verify: docs/superpowers/specs/2026-07-31-sqda-abdr-design.md

- [ ] **Step 1: Verify and push**

~~~powershell
pytest -q
git diff --check
git push origin codex/sqda-sgc
~~~

Expected: all tests pass, diff check has no output, branch pushes.

- [ ] **Step 2: Deploy and prove identity**

~~~bash
cd /root/data/uav/sqda-sgc
git fetch origin codex/sqda-sgc
git checkout codex/sqda-sgc
/root/data/uav/venv/bin/python scripts/verify_sqda_sgc_g0.py   --checkpoint /root/data/uav/checkpoints/matched-baseline-best-epoch-0100.pt   --data /root/data/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml   --device 0   --output /root/data/uav/runs/sqda-abdr/g0-equivalence.json
~~~

Expected: passed=true, 300 queries and max_abs_difference=0.0.

- [ ] **Step 3: Generate fixed audit before G1**

~~~bash
/root/data/uav/venv/bin/python scripts/audit_sqda_regressions.py   --images /root/data/uav/datasets/VisDrone/images/val   --labels /root/data/uav/datasets/VisDrone/labels/val   --baseline-predictions /root/data/uav/runs/sqda-sgc/baseline-predictions.json   --candidate-predictions /root/data/uav/runs/sqda-sgc/sqda-sgc-g2-seed0-10ep/predictions-completed_epoch9.json   --output /root/data/uav/runs/sqda-abdr/baseline-vs-g2-error-audit.json
~~~

Expected: fixed 0.25/0.50 evidence only. If an immutable prediction JSON is absent, find it via the prior manifest or GitHub release; do not regenerate predictions with changed weights.

- [ ] **Step 4: Smoke then G1**

~~~bash
/root/data/uav/venv/bin/python scripts/smoke_sqda_sgc_cuda.py   --checkpoint /root/data/uav/checkpoints/matched-baseline-best-epoch-0100.pt   --data /root/data/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml   --output /root/data/uav/runs/sqda-abdr/cuda-smoke.json
bash scripts/run_sqda_sgc_server.sh g1
~~~

Expected: fixed AMP 128, frozen stock, isolated run and automatic artifact sync.

- [ ] **Step 5: Gate**

Any NaN/Inf, frozen audit failure, or lower Precision, Recall, mAP50 or mAP50-95 fails. Only strict G1 pass starts G2; only strict G2 pass starts formal 100 epochs.

## Plan self-review

- Covers dual budgets, directional decomposition, initialization, diagnostics, audit, isolated staging, G0, AMP and strict gating.
- Does not change loss, query number/order, Top-300, decoder, NMS, input resolution, optimizer, routing, extra attention or post-processing.
- Production/test identifiers are consistent: semantic_projector, geometry_projector, agreement_gate, semantic_budget and geometry_budget.

