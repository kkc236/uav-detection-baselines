# RT-DETR Learnable Quality Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic hidden-aware quality probe that can advance only when Q beats both frozen controls internally and then beats stock on one unopened official-validation pass.

**Architecture:** Extract output-neutral final decoder hidden states into new immutable caches, train equal-parameter C1 and Q MLPs on the fixed 518-image partition, and select checkpoints only on the frozen 129-image internal development split. Publish the C0/C1/Q selection before opening official validation, then evaluate all three arms once from one new hidden-aware validation cache.

**Tech Stack:** Python 3.10.12, PyTorch 2.5.1+cu121, Ultralytics 8.4.90 MuSGD, pytest, CUDA 12.1, RTX 4090.

---

## File map

- Create `src/rtdetr_quality_probe.py`: frozen constants, features, equal-parameter MLP, loss, scoring, checkpoint selection, and Gates.
- Create `src/rtdetr_quality_probe_cache.py`: hidden-aware create-only sharded cache and safe resume validation.
- Create `scripts/run_rtdetr_quality_probe.py`: hook canary, extraction, deterministic training, delayed validation, reports, and state machine.
- Create `tests/test_rtdetr_quality_probe.py`: split, features, model, loss, scoring, selection, and Gate tests.
- Create `tests/test_rtdetr_quality_probe_cache.py`: schema, authority, corruption, and resume tests.
- Create `tests/test_run_rtdetr_quality_probe.py`: hook, detector isolation, deterministic training, delayed validation, and CLI tests.
- Modify `tests/test_rtdetr_quality_oracle.py` only to lock the 518-image complement hash beside the existing real 647/129 path fixture.
- Reuse unchanged `src/rtdetr_quality_oracle.py`, `src/iber_evaluation.py`, `src/iber_protocol.py`, and `src/lpr_protocol.py`.

### Task 1: Freeze split and feature contracts

**Files:**
- Create: `src/rtdetr_quality_probe.py`
- Create: `tests/test_rtdetr_quality_probe.py`
- Modify: `tests/test_rtdetr_quality_oracle.py`

- [ ] **Step 1: Write failing split and feature tests**

Assert constants and exact contracts:

```python
assert PROBE_TRAIN_COUNT == 518
assert DEV_COUNT == 129
assert OFFICIAL_VAL_COUNT == 548
assert PROBE_TRAIN_SHA256 == "1E46817FFFBDBCBA0BA1675CA6142ABABBD6147394AA1D0F10B57F0ECAF7236D"
assert ALPHA == 2.0
assert C1_FEATURES == 20
assert DECODER_HIDDEN == 256
assert MODEL_INPUT == 276

c1 = build_c1_features(boxes, logits)
q = build_q_features(boxes, logits, hidden)
assert c1.shape == q.shape == (2, 300, 10, 276)
assert torch.count_nonzero(c1[..., 20:]) == 0
torch.testing.assert_close(q[..., 20:], hidden[:, :, None, :].expand(-1, -1, 10, -1))
```

Extend the existing real-path test to assert that removing the frozen 129 paths from the
ordered 647 authority yields 518 unique/disjoint paths and the exact hash above.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rtdetr_quality_probe.py tests/test_rtdetr_quality_oracle.py -q`

Expected: FAIL because `src.rtdetr_quality_probe` does not exist.

- [ ] **Step 3: Implement frozen features**

Implement detached float32 features with probability clamp `1e-6`, normalized box
coordinates, `1/640` width/height floor, mean Bernoulli entropy, one-hot class, and exact
zero-padding for C1:

```python
def build_c1_features(boxes, logits):
    base = _class_conditional_features(boxes.detach().float(), logits.detach().float())
    zeros = base.new_zeros((*base.shape[:-1], 256))
    return torch.cat((base, zeros), -1).contiguous()

def build_q_features(boxes, logits, hidden):
    base = _class_conditional_features(boxes.detach().float(), logits.detach().float())
    expanded = hidden.detach().float()[:, :, None, :].expand(-1, -1, 10, -1)
    return torch.cat((base, expanded), -1).contiguous()
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_rtdetr_quality_probe.py tests/test_rtdetr_quality_oracle.py -q`

Expected: all selected tests pass.

```bash
git add src/rtdetr_quality_probe.py tests/test_rtdetr_quality_probe.py tests/test_rtdetr_quality_oracle.py
git commit -m "experiment: lock quality probe features"
```

### Task 2: Lock model, loss, scoring, selection, and Gates

**Files:**
- Modify: `src/rtdetr_quality_probe.py`
- Modify: `tests/test_rtdetr_quality_probe.py`

- [ ] **Step 1: Write failing behavioral tests**

Test identical C1/Q initial state bytes, architecture `276->64->1`, soft BCE over all
query/class rows, exact C0 stock reconstruction, alpha 2.0 reranking, unchanged boxes,
flattened Top-300, lexicographic `(map, ap75, ap50, -epoch)` selection, and these Gate
boundaries:

```python
assert decide_internal_gate(c0_map=.20, c0_ap75=.18, c1_map=.201,
    c1_ap75=.181, q_map=.206, q_ap75=.182)["status"] == "passed"
assert decide_internal_gate(c0_map=.20, c0_ap75=.18, c1_map=.201,
    c1_ap75=.182, q_map=.206, q_ap75=.182)["status"] == "scientific_failed"
assert decide_official_gate(c0_map=.24, c0_ap75=.23,
    q_map=.240001, q_ap75=.230001)["detector_screen_eligible"] is True
assert decide_official_gate(c0_map=.24, c0_ap75=.23,
    q_map=.24, q_ap75=.24)["detector_screen_eligible"] is False
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rtdetr_quality_probe.py -q`

Expected: FAIL because model/scoring/Gate symbols are absent.

- [ ] **Step 3: Implement minimal frozen core**

```python
class QualityProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(276, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)

def quality_loss(prediction_logits, target_quality):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        prediction_logits, target_quality.detach().float(), reduction="mean"
    )
```

Use `flattened_topk(boxes, logits.sigmoid())` for C0 and
`flattened_topk(boxes, logits.sigmoid() * probe_logits.sigmoid().pow(2.0))` for C1/Q.
Use `Decimal(str(value))` for all Gate arithmetic and JSON-safe canonical strings.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_rtdetr_quality_probe.py -q`

```bash
git add src/rtdetr_quality_probe.py tests/test_rtdetr_quality_probe.py
git commit -m "experiment: lock quality probe decisions"
```

### Task 3: Build the hidden-aware immutable cache

**Files:**
- Create: `src/rtdetr_quality_probe_cache.py`
- Create: `tests/test_rtdetr_quality_probe_cache.py`

- [ ] **Step 1: Write failing cache tests**

Require exact record fields `image_id/boxes/logits/hidden/quality/target_boxes/target_classes`,
float32/int64 dtypes, shapes `[300,4]/[300,10]/[300,256]/[300,10]`, CPU,
contiguity, finiteness, no gradients, exact target recomputation, 32-image shards,
ordered split identities, create-only atomic files, fsync, bytes/SHA-256, manifest-last,
safe `weights_only=True` loading, symlink/reparse rejection, corruption rejection, and
resume that writes only missing shards under an identical intent.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rtdetr_quality_probe_cache.py -q`

- [ ] **Step 3: Implement cache contracts**

Expose:

```python
write_cache_intent(root, *, stage, image_ids, authority) -> dict
write_cache_shard(root, *, index, records, intent_sha256) -> dict
publish_cache_manifest(root, *, intent, shard_inventory) -> dict
load_quality_probe_cache(root, *, authority, manifest_sha256) -> tuple[dict, ...]
missing_shard_indices(root, *, verified_intent) -> tuple[int, ...]
```

Use temporary sibling directories/files plus no-replace publication, canonical JSON with
one LF, and strict allowlists for root contents.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_rtdetr_quality_probe_cache.py -q`

```bash
git add src/rtdetr_quality_probe_cache.py tests/test_rtdetr_quality_probe_cache.py
git commit -m "experiment: cache hidden quality evidence"
```

### Task 4: Capture final hidden state without changing output

**Files:**
- Create: `scripts/run_rtdetr_quality_probe.py`
- Create: `tests/test_run_rtdetr_quality_probe.py`

- [ ] **Step 1: Write failing hook/isolation tests**

Use fake and pinned-head tests to require a pre-hook on
`head.dec_score_head[head.decoder.eval_idx]`, exactly one call, `None` return, cleanup on
success/error, hidden `[B,300,256]`, exact `score_head(hidden)==final_logits`, and
byte-for-byte equality of no-hook/hook stock output, boxes, and logits. Also require
`eval()`, inference mode, detector gradients all `None`, unchanged state fingerprint,
and no detector parameter in an optimizer group.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_run_rtdetr_quality_probe.py -q`

- [ ] **Step 3: Implement hook canary and extraction**

```python
def hook(module, args):
    if len(captures) != 0:
        raise RuntimeError("eval score hook fired more than once")
    captures.append(args[0].detach())
    return None

handle = head.dec_score_head[head.decoder.eval_idx].register_forward_pre_hook(hook)
try:
    with torch.inference_mode():
        stock, auxiliary = detector.predict(images)
finally:
    handle.remove()
```

Extract boxes/logits/hidden/targets, compute detached same-class quality, validate all
record contracts, and write only fixed missing shards.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_run_rtdetr_quality_probe.py tests/test_rtdetr_quality_probe_cache.py -q`

```bash
git add scripts/run_rtdetr_quality_probe.py tests/test_run_rtdetr_quality_probe.py
git commit -m "experiment: capture neutral decoder hidden"
```

### Task 5: Deterministic MuSGD training and checkpoint resume

**Files:**
- Modify: `src/rtdetr_quality_probe.py`
- Modify: `scripts/run_rtdetr_quality_probe.py`
- Modify: `tests/test_run_rtdetr_quality_probe.py`

- [ ] **Step 1: Write failing replay/checkpoint tests**

Lock seed0, deterministic algorithms, float32/no AMP, 20 epochs, image batch 8,
`manual_seed(epoch)` permutations, identical C1/Q initialization and order, MuSGD
`lr=.01/momentum=.937/nesterov=True/muon=.2/sgd=1.0`, weight decay `.0005` only on
2-D weights, no scheduler/warmup/clipping, create-only checkpoint/sidecar pairs,
contiguous safe resume, cross-arm rejection, and byte-identical replay on synthetic data.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_run_rtdetr_quality_probe.py -q`

- [ ] **Step 3: Implement training**

Construct separate MuSGD optimizers from the same seed0 model state. Train C1 then Q for
all 20 epochs over every `8*300*10` row batch. Save pure tensor/primitive checkpoint
dictionaries and load with `weights_only=True`. Evaluate each epoch on the ordered 129
records and publish metrics before accepting the checkpoint as resumable.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_rtdetr_quality_probe.py tests/test_run_rtdetr_quality_probe.py -q`

```bash
git add src/rtdetr_quality_probe.py scripts/run_rtdetr_quality_probe.py tests/test_run_rtdetr_quality_probe.py
git commit -m "experiment: train deterministic quality probes"
```

### Task 6: Freeze internal selection before validation access

**Files:**
- Modify: `scripts/run_rtdetr_quality_probe.py`
- Modify: `tests/test_run_rtdetr_quality_probe.py`

- [ ] **Step 1: Write failing state-machine tests**

Record events and prove the runner completes train-cache, C1/Q epoch reports, selected
checkpoint hash verification, `internal-selection-report.json`, and internal decision
before any validation path enumeration/loader/cache call. Internal failure must emit
`scientific_failed`, leave official-cache/report paths absent, and stop on the FDR-only
branch. Existing oracle validation cache paths must always be rejected.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_run_rtdetr_quality_probe.py -q`

- [ ] **Step 3: Implement internal orchestration**

Evaluate C0 once, select C1/Q by `(map, ap75, ap50, -epoch)`, verify checkpoint bytes and
SHA-256, apply the four-condition internal Gate, and publish canonical create-only
selection/decision reports. Pass a verified internal-selection value—not a boolean—to
the only function capable of constructing the official loader.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/test_run_rtdetr_quality_probe.py -q
git add scripts/run_rtdetr_quality_probe.py tests/test_run_rtdetr_quality_probe.py
git commit -m "experiment: gate quality probe validation"
```

### Task 7: One-shot official evaluation and final decision

**Files:**
- Modify: `scripts/run_rtdetr_quality_probe.py`
- Modify: `tests/test_run_rtdetr_quality_probe.py`

- [ ] **Step 1: Write failing official-stage tests**

Require exactly 548 new hidden-aware records and one detector inference per image after
internal pass; one shared cache for C0/C1/Q; no rerun when a complete cache exists; exact
AP stock authority; `1e-8` precision/recall diagnostic tolerance; strict Q mAP/AP75 over
C0 eligibility; C1 report-only status; no post-val tuning; and immutable final report,
decision, environment, and SHA inventory.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_run_rtdetr_quality_probe.py -q`

- [ ] **Step 3: Implement official stage and CLI**

Limit CLI arguments to baseline checkpoint, dataset root, oracle decision, cache root,
report root, and device. Revalidate all authority, extract or safely load the probe-owned
official cache, evaluate selected checkpoints once, apply the strict official Gate, and
return exit 0 for both valid `passed` and `scientific_failed` outcomes.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/test_rtdetr_quality_probe.py tests/test_rtdetr_quality_probe_cache.py tests/test_run_rtdetr_quality_probe.py tests/test_iber_evaluation.py tests/test_rtdetr_quality_oracle.py tests/test_run_rtdetr_quality_oracle.py -q
python -m compileall -q src scripts tests
git add scripts/run_rtdetr_quality_probe.py tests/test_run_rtdetr_quality_probe.py
git commit -m "experiment: evaluate learnable quality probe"
```

### Task 8: Full verification and immutable execution

**Files:**
- Modify only defects first reproduced by a failing test.

- [ ] Run `python -m pytest -q` and require zero failures, allowing only existing intentional skips.
- [ ] Run `git diff --check` and `python -m compileall -q src scripts tests`.
- [ ] Push `codex/iber-be`; require local and remote 40-character SHAs to match.
- [ ] Verify the pinned server fingerprint, RTX 4090/runtime, baseline/data/subset/oracle hashes, free space, and absence of competing training.
- [ ] Deploy a new immutable source/run/cache root keyed by source SHA; run focused CUDA hook-neutrality and C0 reconstruction canaries before the pipeline.
- [ ] Run the probe once. If internal Q fails, publish `scientific_failed` without opening validation and pivot to FDR-only migration. If internal Q passes, permit the single official stage; only a strict official Q-over-C0 mAP/AP75 pass authorizes a separately designed 30-epoch detector screen.
- [ ] Publish design, plan, source SHA, cache/checkpoint/report inventories, decisions, environment, and all SHA-256 evidence transactionally; verify the remote result SHA.
