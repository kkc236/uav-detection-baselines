# Ultralytics RT-DETR-L FDR-only Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate only the official D-FINE FDR/FGL mechanics into the frozen Ultralytics RT-DETR-L baseline, pass all engineering/representation Gates, then run the authorized fixed 10% seed0 30-epoch comparison and, only after unchanged Gate2 passes, the full-data seed0 100-epoch experiment.

**Architecture:** Keep Ultralytics 8.4.90 RT-DETR-L classification, query selection, backbone, encoder, decoder layers, matching, Top-300, and MuSGD protocol intact. Replace only decoder box regression with a traditional first-layer preliminary box plus six cumulative 132-logit distribution heads, decode with commit-pinned D-FINE weighting/Integral and box transforms, and add IoU-weighted adjacent-bin FGL on stock assignments.

**Tech Stack:** Python 3.10.12, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.90, pytest, CUDA 12.1, NVIDIA RTX 4090, canonical JSON/SHA256, Git/GitHub epoch publication.

---

## File map

- Create `third_party/dfine_7fe2f888/AUTHORITY.json`: immutable upstream commit, URLs, license, and vendored-file hashes.
- Create `third_party/dfine_7fe2f888/reference_fdr.py`: test-only official weighting/Integral/box/FGL formulas copied from the pinned source with attribution.
- Create `src/fdr_math.py`: production weighting, Integral, box transforms, and target interpolation.
- Create `src/fdr_head.py`: preliminary box head, distribution heads, cumulative decoder outputs, and diagnostics.
- Create `src/fdr_loss.py`: stock loss extension that reuses stock assignments and adds FGL only.
- Create `src/fdr_protocol.py`: frozen constants, public/private initialization, manifests, Gates, and resume authority.
- Create `src/rtdetr_fdr.py`: Ultralytics model/trainer integration and fixed AMP128/MuSGD enforcement.
- Create `scripts/prepare_fdr_protocol.py`: build paired seed0 initialization and immutable protocol evidence.
- Create `scripts/run_fdr_preflight.py`: CPU golden, model/loss isolation, shape, representation oracle, and 4090 one-step Gates.
- Create `scripts/train_rtdetr_fdr.py`: fixed screen/formal training, resume, and per-epoch publication.
- Create `scripts/evaluate_rtdetr_fdr.py`: independent paired evaluation using unchanged Gate2 authority.
- Create `tests/test_fdr_authority.py`, `tests/test_fdr_math.py`, `tests/test_fdr_head.py`, `tests/test_fdr_loss.py`, `tests/test_fdr_protocol.py`, `tests/test_rtdetr_fdr.py`, `tests/test_fdr_preflight.py`, and `tests/test_train_rtdetr_fdr.py`.
- Reuse unchanged `src/lpr_protocol.py`, `src/lpr_g_publication.py`, `src/checkpoint_recovery.py`, `scripts/sync_experiment_checkpoint.py`, and the existing detector Gate2 authority/evaluator.

## Task 0: Vendor and bind the exact official source

**Files:**
- Create: `third_party/dfine_7fe2f888/AUTHORITY.json`
- Create: `third_party/dfine_7fe2f888/reference_fdr.py`
- Create: `tests/test_fdr_authority.py`

- [ ] **Step 1: Write the failing authority test**

```python
import hashlib
import json
from pathlib import Path

PIN = "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"
ROOT = Path("third_party/dfine_7fe2f888")

def test_dfine_authority_is_commit_pinned_and_self_hashing():
    authority = json.loads((ROOT / "AUTHORITY.json").read_text("utf-8"))
    assert authority["repository"] == "https://github.com/Peterande/D-FINE"
    assert authority["commit"] == PIN
    assert set(authority["sources"]) == {
        "dfine_decoder.py", "dfine_utils.py", "dfine_criterion.py",
        "dfine_hgnetv2.yml",
    }
    data = (ROOT / "reference_fdr.py").read_bytes()
    assert hashlib.sha256(data).hexdigest() == authority["vendored_reference_sha256"]
    assert authority["usage"] == "test-only golden reference"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_fdr_authority.py -q`

Expected: FAIL because the pinned authority files do not exist.

- [ ] **Step 3: Fetch only from the immutable commit and vendor the required formulas**

Use these exact sources:

```text
https://raw.githubusercontent.com/Peterande/D-FINE/7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6/src/zoo/dfine/dfine_decoder.py
https://raw.githubusercontent.com/Peterande/D-FINE/7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6/src/zoo/dfine/dfine_utils.py
https://raw.githubusercontent.com/Peterande/D-FINE/7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6/src/zoo/dfine/dfine_criterion.py
https://raw.githubusercontent.com/Peterande/D-FINE/7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6/configs/dfine/include/dfine_hgnetv2.yml
```

Copy only `weighting_function`, `translate_gt`, `distance2bbox`,
`bbox2distance`, the Integral calculation, and adjacent-bin FGL into the
test-only reference module. Preserve the upstream copyright/license header.
Record SHA256 for every downloaded source and the final vendored reference.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_fdr_authority.py -q`

Expected: PASS with the exact commit and all hashes bound.

```bash
git add third_party/dfine_7fe2f888 tests/test_fdr_authority.py
git commit -m "test: pin official FDR authority"
```

## Task 1: Implement commit-exact FDR mathematics

**Files:**
- Create: `src/fdr_math.py`
- Create: `tests/test_fdr_math.py`

- [ ] **Step 1: Write failing golden tests**

```python
import pytest
import torch
from third_party.dfine_7fe2f888.reference_fdr import (
    bbox2distance as ref_bbox2distance,
    distance2bbox as ref_distance2bbox,
    translate_gt as ref_translate_gt,
    weighting_function as ref_weighting,
)
from src.fdr_math import distance2bbox, translate_gt, weighting_function

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weighting_and_box_transforms_match_pinned_reference(dtype):
    up = torch.tensor([0.5], dtype=dtype)
    project_ref = ref_weighting(32, up, torch.tensor([4.0], dtype=dtype))
    project = weighting_function(32, up, torch.tensor([4.0], dtype=dtype))
    torch.testing.assert_close(project, project_ref, rtol=0, atol=0)

    points = torch.tensor([[.5, .5, .2, .1], [.1, .9, .02, .03]], dtype=dtype)
    offsets = torch.tensor([[0., 0., 0., 0.], [-4., 2., 4., -2.]], dtype=dtype)
    torch.testing.assert_close(
        distance2bbox(points, offsets, 4.0),
        ref_distance2bbox(points, offsets, 4.0), rtol=0, atol=0,
    )

def test_bbox_target_interpolation_matches_reference_and_is_bounded():
    values = torch.tensor([-9., -8., -1e-6, 0., 1e-6, 8., 9.], dtype=torch.float64)
    up = torch.tensor([0.5], dtype=torch.float64)
    actual = translate_gt(values, reg_max=32, reg_scale=4.0, up=up)
    expected = ref_translate_gt(values, reg_max=32, reg_scale=4.0, up=up)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    indices, weight_right, weight_left = actual
    assert torch.all((indices >= 0) & (indices < 32))
    torch.testing.assert_close(weight_left + weight_right,
                               torch.ones_like(weight_left), rtol=0, atol=0)
```

The explicit boundary vector is mandatory; randomized-only coverage is not
sufficient.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_fdr_math.py -q`

Expected: FAIL because `src.fdr_math` does not exist.

- [ ] **Step 3: Implement the minimal production math**

Expose only the following constants and functions. Copy each function body
mechanically from the already vendored pinned reference, changing only local
box-conversion imports; the production module must not import the test vendor:

```python
REG_MAX = 32
REG_SCALE = 4.0
UP = 0.5

def integral(corner_logits: torch.Tensor, project: torch.Tensor) -> torch.Tensor:
    shape = corner_logits.shape
    probabilities = torch.softmax(corner_logits.reshape(-1, REG_MAX + 1), dim=1)
    offsets = torch.nn.functional.linear(probabilities, project.to(probabilities.device))
    return offsets.reshape(list(shape[:-1]) + [-1])
```

The module also exports `weighting_function`, `translate_gt`, `distance2bbox`,
and `bbox2distance` with the exact pinned signatures and bodies. Add no learned
parameters, approximations, alternate clipping, or tunable constants.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_fdr_authority.py tests/test_fdr_math.py -q`

Expected: all selected tests pass in float32 and float64.

```bash
git add src/fdr_math.py tests/test_fdr_math.py
git commit -m "feat: add pinned FDR math"
```

## Task 2: Build the FDR-only decoder box path

**Files:**
- Create: `src/fdr_head.py`
- Create: `tests/test_fdr_head.py`

- [ ] **Step 1: Write failing head and shape tests**

```python
def test_fdr_head_has_exact_outputs_and_no_excluded_components():
    decoder = build_test_fdr_decoder(layers=6, hidden=256, queries=300)
    out = decoder(test_hidden(batch=2), test_references(batch=2))
    assert out.corner_logits.shape == (6, 2, 300, 132)
    assert out.boxes.shape == (6, 2, 300, 4)
    assert out.pre_boxes.shape == (2, 300, 4)
    names = set(dict(decoder.named_modules()))
    assert not any(any(x in name.lower() for x in (
        "ddf", "teacher", "lqe", "go_lsd", "target_gate"
    )) for name in names)

def test_distribution_logits_are_cumulative_residuals():
    delta = torch.stack([torch.full((1, 2, 132), float(i + 1)) for i in range(6)])
    actual = cumulative_logits(delta)
    expected = delta.cumsum(dim=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

def test_neutral_distribution_decodes_to_preliminary_box():
    pre = torch.tensor([[[.5, .5, .2, .1]]], dtype=torch.float32)
    project = weighting_function(32, torch.tensor([.5]), torch.tensor([4.]))
    zero_bin = int(torch.nonzero(project == 0, as_tuple=False).item())
    logits = torch.full((1, 1, 4, 33), -100.0)
    logits[..., zero_bin] = 100.0
    decoded = distance2bbox(pre, integral(logits, project), 4.0)
    torch.testing.assert_close(decoded, pre, rtol=0, atol=1e-7)
```

Also assert with a backward test that the next-layer reference is detached.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_fdr_head.py -q`

Expected: FAIL because `src.fdr_head` is absent.

- [ ] **Step 3: Implement the minimum FDR head**

Implement focused types:

```python
@dataclass(frozen=True)
class FDROutput:
    boxes: torch.Tensor
    corner_logits: torch.Tensor
    references: torch.Tensor
    pre_boxes: torch.Tensor

class FDRBoxPath(nn.Module):
    reg_max = 32
    reg_scale = 4.0
    # pre_bbox_head: hidden -> hidden -> hidden -> 4
    # six dist heads: hidden -> hidden -> hidden -> 132
```

At layer 0 compute the traditional preliminary box. At every layer add the
132-logit residual to cumulative logits, run the pinned Integral, decode against
the detached preliminary box, and detach only the box used as the next decoder
reference. The classification path must not be present in this module.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_fdr_math.py tests/test_fdr_head.py -q`

Expected: all selected tests pass with exact six-layer production shapes.

```bash
git add src/fdr_head.py tests/test_fdr_head.py
git commit -m "feat: add FDR-only decoder box path"
```

## Task 3: Extend stock loss with FGL only

**Files:**
- Create: `src/fdr_loss.py`
- Create: `tests/test_fdr_loss.py`

- [ ] **Step 1: Write failing stock-isolation and FGL tests**

```python
def test_fgl_zero_is_exact_stock_loss():
    stock = stock_criterion()
    fdr = fdr_criterion(fgl_weight=0.0)
    predictions, batch = deterministic_stock_inputs()
    stock_loss = stock(predictions, batch)
    fdr_loss = fdr.stock_plus_fgl(predictions, batch, corner_logits=None)
    assert stock_loss.keys() == fdr_loss.keys()
    for key in stock_loss:
        torch.testing.assert_close(fdr_loss[key], stock_loss[key], rtol=0, atol=0)

def test_fgl_matches_pinned_adjacent_bin_reference():
    logits, indices, wl, wr, iou = deterministic_fgl_inputs()
    actual = adjacent_bin_fgl(logits, indices, wl, wr, iou, avg_factor=2.0)
    expected = pinned_reference_fgl(logits, indices, wl, wr, iou, avg_factor=2.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

def test_fgl_reuses_each_stock_assignment_without_second_match():
    criterion, matcher = recording_fdr_criterion()
    criterion(deterministic_aux_and_dn_predictions(), deterministic_batch())
    assert matcher.calls == criterion.expected_stock_match_calls
    assert criterion.fgl_extra_match_calls == 0
```

The helpers construct fixed small tensors in the same test file. Add explicit
empty-GT, mixed empty/non-empty, boundary-target, and no-`loss_ddf`/teacher-key
assertions.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_fdr_loss.py -q`

Expected: FAIL because the FDR criterion is missing.

- [ ] **Step 3: Implement stock-plus-FGL criterion**

Subclass the pinned Ultralytics 8.4.90 `RTDETRDetectionLoss`. Record the exact
stock match indices per normal, auxiliary, and DN prediction group, delegate all
VFL/L1/GIoU work to `super()`, and calculate only:

```python
losses["loss_fgl"] = 0.15 * adjacent_bin_ce(
    corner_logits=matched_corners,
    target_indices=target_indices,
    left_weight=left_weight,
    right_weight=right_weight,
    sample_weight=matched_iou.detach(),
    normalizer=num_boxes,
)
```

Do not implement cross-layer union, DDF, GO-LSD, teacher targets, LQE, or target
gating.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_fdr_loss.py tests/test_lpr_g_loss.py -q`

Expected: all selected stock-compatibility and FDR tests pass.

```bash
git add src/fdr_loss.py tests/test_fdr_loss.py
git commit -m "feat: add isolated FGL supervision"
```

## Task 4: Integrate with Ultralytics while preserving stock contracts

**Files:**
- Create: `src/rtdetr_fdr.py`
- Create: `tests/test_rtdetr_fdr.py`

- [ ] **Step 1: Write failing model integration tests**

Assert:

```python
model = FDRRTDETRDetectionModel("rtdetr-l.yaml", nc=10, verbose=False)
head = model.model[-1]
assert head.num_queries == 300
assert model.fdr.reg_max == 32
assert model.fdr.reg_scale == 4.0
assert all(layer.out_features == 132 for layer in model.fdr.final_layers)
assert classification_state(model) == classification_state(stock_model)
assert query_selection_state(model) == query_selection_state(stock_model)
assert postprocess_signature(model) == postprocess_signature(stock_model)
```

Add production forward tests for outputs `[6,B,300,132]`, boxes
`[6,B,300,4]`, scores `[6,B,300,10]`, Top-300 equality for identical final
boxes/scores, DN metadata, empty GT, and finite backward.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rtdetr_fdr.py -q`

Expected: FAIL because the repository-owned FDR model does not exist.

- [ ] **Step 3: Implement repository-owned model and trainer**

Follow the existing inheritance/replacement pattern in `src/rtdetr_lpr.py`.
Do not patch site-packages. Preserve the stock head's class heads, query
selection, DN builder, and `postprocess`; route only decoder box outputs through
`FDRBoxPath`, and return distribution/reference tensors only in the training
auxiliary tuple.

Reuse the fixed paired trainer behavior for:

```text
MuSGD(lr=0.01, momentum=0.937, weight_decay=0.0005)
AMP scale=128, no growth, no skipped step
gradient clipping/evidence
deterministic seed0
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_rtdetr_fdr.py tests/test_lpr_paired_trainer.py -q`

Expected: all selected tests pass; installed Ultralytics files remain unchanged.

```bash
git add src/rtdetr_fdr.py tests/test_rtdetr_fdr.py
git commit -m "feat: integrate FDR-only RT-DETR"
```

## Task 5: Freeze paired initialization and protocol authority

**Files:**
- Create: `src/fdr_protocol.py`
- Create: `scripts/prepare_fdr_protocol.py`
- Create: `tests/test_fdr_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

Test the exact environment/data constants, upstream commit, public/private key
partition, public-state byte equality, private RNG isolation, `pre_bbox_head`
copy from stock layer 0, zero-initialized distribution finals, optimizer group
coverage, create-only artifacts, and resume rejection across source/protocol/run
identities.

```python
assert FDR_PROTOCOL["dfine_commit"] == "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"
assert FDR_PROTOCOL["reg_max"] == 32
assert FDR_PROTOCOL["reg_scale"] == 4.0
assert FDR_PROTOCOL["loss_weights"] == {
    "vfl": 1.0, "bbox": 5.0, "giou": 2.0, "fgl": 0.15,
}
assert FDR_PROTOCOL["excluded"] == [
    "DDF", "GO-LSD", "teacher", "LQE", "target_gating",
]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_fdr_protocol.py -q`

Expected: FAIL because FDR protocol symbols and preparer are absent.

- [ ] **Step 3: Implement immutable preparation**

Build stock and FDR models from one seed0 public state. Hash every public tensor
before adding private heads, initialize private heads under `torch.random.fork_rng`,
restore public RNG state, and write create-only canonical JSON plus state files by
staging/fsync/atomic publish. Reject symlinks, partial roots, and unbound manifests.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_fdr_protocol.py tests/test_lpr_protocol.py -q`

Expected: all selected protocol tests pass.

```bash
git add src/fdr_protocol.py scripts/prepare_fdr_protocol.py tests/test_fdr_protocol.py
git commit -m "experiment: freeze paired FDR protocol"
```

## Task 6: Implement and pass pre-30-epoch Gates F0-F4

**Files:**
- Create: `scripts/run_fdr_preflight.py`
- Create: `tests/test_fdr_preflight.py`

- [ ] **Step 1: Write failing Gate state-machine tests**

Test that F0-F4 run in order; any failure prevents later gates and training
authorization. Test canonical create-only reports, representation statistics,
device enforcement `cuda:0`, real-batch one-step evidence schema, source/model
immutability, and a final `screen_eligible` value that is true only when every
gate passed.

```python
decision = decide_preflight({"F0": "passed", "F1": "passed",
                             "F2": "passed", "F3": "passed", "F4": "passed"})
assert decision == {"status": "passed", "screen_eligible": True}
for failed in ("F0", "F1", "F2", "F3", "F4"):
    states = {key: "passed" for key in ("F0", "F1", "F2", "F3", "F4")}
    states[failed] = "engineering_failed"
    assert decide_preflight(states)["screen_eligible"] is False
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_fdr_preflight.py -q`

Expected: FAIL because the preflight runner is absent.

- [ ] **Step 3: Implement the Gate runner**

The CLI accepts only authority paths and output roots; it exposes no reg, loss,
threshold, seed, device, batch, or subset tuning arguments. F0-F2 run on CPU.
F3 requires the exact RTX 4090 environment and one real `batch=8` step. F4 uses
frozen baseline references and matched GT, publishing reconstruction error and
bin saturation overall and by object scale.

- [ ] **Step 4: Verify locally, deploy immutably, and run on the 4090**

Run locally:

```bash
python -m pytest tests/test_fdr_authority.py tests/test_fdr_math.py \
  tests/test_fdr_head.py tests/test_fdr_loss.py tests/test_fdr_protocol.py \
  tests/test_rtdetr_fdr.py tests/test_fdr_preflight.py -q
python -m compileall -q src scripts tests
git diff --check
```

Expected: zero failures, compile exit 0, and no whitespace errors.

Deploy the exact committed source to a new immutable server directory. Run:

```bash
SOURCE_SHA="$(git rev-parse --short=12 HEAD)"
PROTOCOL_SHA="$(sha256sum /data/uav/protocols/fdr-seed0/protocol.json | cut -c1-12)"
RUN_ID="${SOURCE_SHA}-${PROTOCOL_SHA}-seed0"
python scripts/run_fdr_preflight.py \
  --protocol-manifest /data/uav/protocols/fdr-seed0/protocol.json \
  --baseline-checkpoint /data/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt \
  --dataset-root /data/uav/datasets/VisDrone \
  --report-root "/data/uav/runs/fdr-preflight/${RUN_ID}"
```

Expected: F0-F4 all `passed`, `screen_eligible=true`; otherwise repair the
engineering defect with a failing regression test and create a new immutable run.

- [ ] **Step 5: Commit the verified Gate runner and evidence schema**

```bash
git add scripts/run_fdr_preflight.py tests/test_fdr_preflight.py
git commit -m "experiment: gate FDR-only training"
```

## Task 7: Add resumable per-epoch screen/formal training

**Files:**
- Create: `scripts/train_rtdetr_fdr.py`
- Create: `tests/test_train_rtdetr_fdr.py`

- [ ] **Step 1: Write failing CLI, resume, and publication tests**

Assert exact frozen values, seed0 only, `screen=30`, `formal=100`, full-data
formal mode, fixed subset screen mode, `mosaic=1.0`, AMP128, MuSGD grouping,
per-epoch callback order, create-only publication, retry/remote verification,
latest verified checkpoint selection, partial-epoch rejection, and prohibition
on formal resume from screen.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_train_rtdetr_fdr.py -q`

Expected: FAIL because the training runner is absent.

- [ ] **Step 3: Implement the fixed runner**

Expose only:

```text
--variant control|fdr
--stage screen|formal
--seed 0
--protocol-manifest PATH
--initial-state PATH
--preflight-decision PATH
--resume PATH (optional)
--project PATH
--name IMMUTABLE_NAME
--token-file PATH
--tag IMMUTABLE_TAG
--results-repo PATH
```

All scientific parameters come from `FDR_PROTOCOL`. After every verified epoch,
publish metrics, losses, FGL diagnostics, checkpoint/hash, AMP/optimizer evidence,
environment/source/data/order authority, GPU timing/memory, and pipeline state.
Do not start epoch `n+1` until epoch `n` publication is remotely verified.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
python -m pytest tests/test_train_rtdetr_fdr.py tests/test_lpr_g_publication.py \
  tests/test_lpr_g_restore.py -q
python -m compileall -q src scripts tests
git diff --check
```

Expected: zero failures and clean static checks.

```bash
git add scripts/train_rtdetr_fdr.py tests/test_train_rtdetr_fdr.py
git commit -m "experiment: add resumable FDR training"
```

## Task 8: Run fixed 10% seed0 30 epochs and apply unchanged Gate2

**Files:**
- Create: `scripts/evaluate_rtdetr_fdr.py`
- Create: `tests/test_evaluate_rtdetr_fdr.py`

- [ ] **Step 1: Write failing evaluator authority tests**

Assert that the evaluator imports the existing frozen detector Gate2 thresholds,
never defines replacements, requires exactly 30 verified epochs, compares only
matching seed0/subset/source/protocol authorities, performs independent
evaluation, and emits either `passed` or `scientific_failed`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_evaluate_rtdetr_fdr.py -q`

Expected: FAIL because the evaluator is missing.

- [ ] **Step 3: Implement the authority-bound evaluator**

Load the existing baseline seed0 evidence only after every authority hash equals
the FDR screen. If any hash differs, require a new matched 30-epoch control; do
not compare unmatched runs. Select no favorable checkpoint on official val.

- [ ] **Step 4: Verify, commit, deploy, and launch the screen**

```bash
python -m pytest tests/test_evaluate_rtdetr_fdr.py tests/test_iber_evaluation.py -q
git add scripts/evaluate_rtdetr_fdr.py tests/test_evaluate_rtdetr_fdr.py
git commit -m "experiment: evaluate FDR Gate2"
```

After pushing and immutable deployment, start the FDR screen immediately:

```bash
SOURCE_SHA="$(git rev-parse --short=12 HEAD)"
PROTOCOL_SHA="$(sha256sum /data/uav/protocols/fdr-seed0/protocol.json | cut -c1-12)"
RUN_ID="${SOURCE_SHA}-${PROTOCOL_SHA}-seed0"
python scripts/train_rtdetr_fdr.py --variant fdr --stage screen --seed 0 \
  --protocol-manifest /data/uav/protocols/fdr-seed0/protocol.json \
  --initial-state /data/uav/protocols/fdr-seed0/initial-state.pt \
  --preflight-decision "/data/uav/runs/fdr-preflight/${RUN_ID}/decision.json" \
  --project /data/uav/runs/fdr-screen --name "${RUN_ID}" \
  --token-file /data/uav/secrets/github-token --tag "fdr-screen-${RUN_ID}" \
  --results-repo /data/uav/results
```

Monitor GPU/process/checkpoint/publication continuously. Repair engineering
failures through a new tested source/run identity and resume from the latest
verified checkpoint. At epoch 30, independently evaluate and publish the
unchanged Gate2 decision.

## Task 9: Run full-data seed0 100 epochs only after Gate2 passes

**Files:**
- Modify: none unless a verified engineering defect requires a separate TDD commit.

- [ ] **Step 1: Verify formal authorization**

Require a remotely verified screen decision containing:

```json
{"status":"passed","formal_eligible":true,"completed_epoch":30,"seed":0}
```

Verify its source/protocol/upstream/public-state hashes and unchanged Gate2
authority. If Gate2 failed, publish `scientific_failed` and stop this plan.

- [ ] **Step 2: Launch formal training from the formal initial state**

```bash
SOURCE_SHA="$(git rev-parse --short=12 HEAD)"
PROTOCOL_SHA="$(sha256sum /data/uav/protocols/fdr-seed0/protocol.json | cut -c1-12)"
RUN_ID="${SOURCE_SHA}-${PROTOCOL_SHA}-seed0"
FORMAL_ID="${RUN_ID}-formal100"
python scripts/train_rtdetr_fdr.py --variant fdr --stage formal --seed 0 \
  --protocol-manifest /data/uav/protocols/fdr-seed0/protocol.json \
  --initial-state /data/uav/protocols/fdr-seed0/initial-state.pt \
  --preflight-decision "/data/uav/runs/fdr-preflight/${RUN_ID}/decision.json" \
  --project /data/uav/runs/fdr-formal --name "${FORMAL_ID}" \
  --token-file /data/uav/secrets/github-token --tag "fdr-formal-${FORMAL_ID}" \
  --results-repo /data/uav/results
```

Expected: full 6471-image data, exactly 100 epochs, seed0, no screen checkpoint
inheritance, and each epoch remotely verified before the next proceeds.

- [ ] **Step 3: Complete independent evaluation and final evidence audit**

Verify all 100 epoch records, checkpoint chain, metrics, final/best checkpoint,
environment, data/order/augmentation hashes, source/upstream/protocol authority,
parameter/GFLOPs/latency evidence, and independent final evaluation. Publish the
final comparison and machine-readable manifest to GitHub.

- [ ] **Step 4: Run final repository verification**

```bash
python -m pytest -q
python -m compileall -q src scripts tests
git diff --check
git status --short
```

Expected: full suite passes, compilation succeeds, no whitespace errors, and no
uncommitted tracked changes. Only then report the 100-epoch result complete.

## Plan self-review checklist

- The upstream commit is exact and immutable: `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6`.
- FDR mechanics include 32/4/0.5, non-uniform Integral, preliminary box,
  cumulative 132-logit residuals, box transforms, and IoU-weighted adjacent-bin FGL.
- DDF, GO-LSD, teacher, LQE, target gating, matching union, and other D-FINE
  additions are explicitly absent.
- Stock classification/query/backbone/encoder/Top-300/NMS=False and the complete
  MuSGD experiment protocol remain frozen.
- F0-F4 block training until all pass.
- The 30-epoch screen is fixed 10% seed0; detector Gate2 thresholds are imported
  unchanged.
- The 100-epoch run is full-data seed0 and starts only after Gate2 passes.
- Resume, immutable evidence, independent evaluation, and every-epoch remote
  publication are required and testable.
