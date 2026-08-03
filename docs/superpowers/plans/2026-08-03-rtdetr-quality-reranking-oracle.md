# RT-DETR Quality-Reranking Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic ground-truth quality-reranking upper-bound oracle that decides whether RT-DETR-L has enough remaining Query-by-class score-ordering headroom to justify a learnable quality probe.

**Architecture:** Extract final-layer stock boxes and logits from one frozen mature baseline forward pass, cache detached development/validation evidence under full authority hashes, and compare exact Ultralytics flattened Query-by-class Top-300 with same-class-IoU reranking. Select one frozen exponent on a deterministic 129-image internal development split, evaluate the selected exponent once on the official 548-image validation split, and emit an immutable pass/scientific-failure decision.

**Tech Stack:** Python 3.10.12, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, pytest, CUDA 12.1, RTX 4090.

---

## File map

- Create `src/rtdetr_quality_oracle.py`: frozen constants, quality computation, exact Top-300, alpha selection, immutable cache, and decision.
- Create `tests/test_rtdetr_quality_oracle.py`: math, Top-300 equivalence, split, cache, decision, and CLI-contract tests.
- Create `scripts/run_rtdetr_quality_oracle.py`: authority validation, frozen extraction, paired metrics, immutable reports, and execution metadata.
- Reuse `src/iber_evaluation.py`: existing no-NMS AP metric implementation.
- Use the non-exported Ultralytics PyTorch model auxiliary tuple directly; do not patch
  the decoder or route this diagnostic through the IBER adapter.

### Task 1: Lock quality and flattened Top-300 semantics

**Files:**
- Create: `tests/test_rtdetr_quality_oracle.py`
- Create: `src/rtdetr_quality_oracle.py`

- [ ] **Step 1: Write failing public-contract tests**

Create tests that import the following symbols and assert the exact constants, same-class
maximum IoU, zero quality for absent classes, unchanged box gathering, duplicate query
selection for different classes, and byte-for-byte equality with
`RTDETRDecoder.postprocess` for stock probabilities:

```python
ALPHA_GRID = (0.25, 0.5, 1.0, 2.0)
DEV_COUNT = 129
MAP_GAIN_THRESHOLD = Decimal("0.0050")

quality = same_class_iou_quality(
    boxes,
    target_boxes,
    target_classes,
    num_classes=10,
)
stock = flattened_topk(boxes[None], logits[None].sigmoid(), num_classes=10)
oracle = oracle_topk(
    boxes[None],
    logits[None],
    quality[None],
    alpha=0.5,
    num_classes=10,
)
assert quality.shape == (300, 10)
assert stock.shape == oracle.shape == (1, 300, 6)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_rtdetr_quality_oracle.py -q`

Expected: collection fails because `src.rtdetr_quality_oracle` does not exist.

- [ ] **Step 3: Implement the minimal quality and Top-300 functions**

Use normalized `cxcywh` boxes and implement same-class quality without Hungarian matching:

```python
def same_class_iou_quality(boxes, target_boxes, target_classes, *, num_classes):
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [Q,4]")
    quality = boxes.new_zeros((len(boxes), num_classes), dtype=torch.float32)
    if not len(target_boxes):
        return quality
    iou = box_iou(cxcywh_to_xyxy(boxes.float()), cxcywh_to_xyxy(target_boxes.float()))
    for class_index in range(num_classes):
        selected = target_classes.long() == class_index
        if bool(selected.any()):
            quality[:, class_index] = iou[:, selected].amax(dim=1)
    return quality.clamp_(0, 1)
```

Reproduce Ultralytics 8.4.90 flattened Top-300 exactly:

```python
def flattened_topk(boxes, scores, *, num_classes, max_det=300):
    if scores.shape != (*boxes.shape[:2], num_classes):
        raise ValueError("scores and boxes disagree")
    selected_scores, index = scores.flatten(1).topk(max_det)
    query = torch.div(index, num_classes, rounding_mode="floor")
    selected_boxes = boxes.gather(1, query[..., None].expand(-1, -1, 4).long())
    classes = index - query * num_classes
    return torch.cat((selected_boxes, selected_scores[..., None], classes[..., None].float()), -1)

def oracle_topk(boxes, logits, qualities, *, alpha, num_classes, max_det=300):
    if alpha not in ALPHA_GRID:
        raise ValueError("alpha is outside the frozen grid")
    if qualities.shape != logits.shape:
        raise ValueError("quality and logits disagree")
    reranked = logits.sigmoid() * qualities.float().pow(alpha)
    return flattened_topk(boxes, reranked, num_classes=num_classes, max_det=max_det)
```

- [ ] **Step 4: Run focused and RT-DETR integration tests**

Run: `python -m pytest tests/test_rtdetr_quality_oracle.py tests/test_rtdetr_iber.py -q`

Expected: all selected tests pass and the stock Top-300 equivalence assertion is exact.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/rtdetr_quality_oracle.py tests/test_rtdetr_quality_oracle.py
git commit -m "experiment: lock RT-DETR quality reranking math"
```

### Task 2: Lock the development split, cache, alpha selection, and decision

**Files:**
- Modify: `src/rtdetr_quality_oracle.py`
- Modify: `tests/test_rtdetr_quality_oracle.py`

- [ ] **Step 1: Add failing split, cache, and decision tests**

Assert that the split selects exactly 129 paths from the already-authorized 647 paths,
is independent of input ordering, changes if any path changes, and has a canonical
uppercase SHA-256 equal to
`FCF8749BAADBA8BDDF5870F472BDE1E937156AFBCEEFDA9F96FED21FA6BB0514` when paths are
UTF-8 LF-delimited with one trailing LF. Assert create-only cache behavior, artifact SHA verification,
train/official-val disjointness, safe `weights_only=True` loading, exact authority
matching, alpha tie-breaking by `(map, ap75, ap50, -alpha)`, and these decision cases:

```python
assert decide_quality_oracle(stock_map=0.20, stock_ap75=0.18,
                             oracle_map=0.205, oracle_ap75=0.180001)["status"] == "passed"
assert decide_quality_oracle(stock_map=0.20, stock_ap75=0.18,
                             oracle_map=0.204999, oracle_ap75=0.20)["status"] == "scientific_failed"
assert decide_quality_oracle(stock_map=0.20, stock_ap75=0.18,
                             oracle_map=0.21, oracle_ap75=0.18)["status"] == "scientific_failed"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_rtdetr_quality_oracle.py -q`

Expected: tests fail because the split, cache, selection, and decision symbols are absent.

- [ ] **Step 3: Implement deterministic selection and exact decision logic**

Expose these exact typed contracts:

- `select_internal_dev(paths: Sequence[Path], *, root: Path) -> tuple[Path, ...]`
- `ordered_path_sha256(paths: Sequence[Path], *, root: Path) -> str`
- `write_quality_oracle_cache(root: Path, *, dev: list[dict], val: list[dict], authority: Mapping[str, str]) -> dict`
- `load_quality_oracle_cache(root: Path, *, authority: Mapping[str, str]) -> dict[str, tuple[dict, ...]]`
- `select_alpha(metrics_by_alpha: Mapping[float, Mapping[str, float]]) -> float`
- `decide_quality_oracle(*, stock_map: float, stock_ap75: float, oracle_map: float, oracle_ap75: float) -> dict`

Use exact decimal conversion for the decision:

```python
gain = Decimal(str(oracle_map)) - Decimal(str(stock_map))
ap75_gain = Decimal(str(oracle_ap75)) - Decimal(str(stock_ap75))
passed = gain >= MAP_GAIN_THRESHOLD and ap75_gain > Decimal("0")
```

Cache records have exactly `image_id`, `boxes`, `logits`, `target_boxes`, and
`target_classes`; every tensor is detached, finite, contiguous, on CPU, and shape checked.
Write split payloads first, fsync them, hash them, and publish canonical `manifest.json`
last with create-only semantics.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_rtdetr_quality_oracle.py -q`

Expected: all quality-oracle unit tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/rtdetr_quality_oracle.py tests/test_rtdetr_quality_oracle.py
git commit -m "experiment: add immutable quality oracle protocol"
```

### Task 3: Integrate frozen extraction and paired evaluation CLI

**Files:**
- Create: `scripts/run_rtdetr_quality_oracle.py`
- Modify: `tests/test_rtdetr_quality_oracle.py`

- [ ] **Step 1: Add failing CLI contract tests**

Import the CLI module without executing it and assert:

- public arguments are limited to baseline checkpoint, dataset root, cache root, report
  root, and device;
- constants are fixed to image size 640, batch 8, workers 8, confidence 0.001,
  `max_det=300`, `NMS=False`, 647 authorized train images, 129 development images, and
  548 official validation images;
- no argument can change alpha, alpha grid, split salt, threshold, confidence, class
  mapping, or official validation pass count;
- detector hashes before and after extraction must match and all gradients remain `None`;
- stock postprocess reconstructed from auxiliary decoder tensors equals the stock output
  returned by the same model call exactly;
- report and decision files use create-only writes and bind all authority hashes.
- official-validation stock metrics exactly equal the frozen same-server stock authority
  (`map=0.24164844987309864`, `ap50=0.4143946635382976`,
  `ap75=0.23916375458831637`, `ap_tiny=0.10314861659739166`,
  `ap_small=0.24166148504350557`, `precision=0.5119369275291381`,
  `recall=0.43525461908044843`).

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `python -m pytest tests/test_rtdetr_quality_oracle.py -q`

Expected: CLI import or contract assertions fail because the file is absent.

- [ ] **Step 3: Implement extraction using the unmodified PyTorch model**

Build RT-DETR validation loaders from an immutable image-list file for internal dev and
the unchanged official validation directory. Keep `head.export=False`. For each
preprocessed batch, use the standard model auxiliary return without hooks or monkeypatches:

```python
with torch.inference_mode():
    stock_output, auxiliary = detector.predict(batch["img"])
    decoder_boxes, decoder_logits, _, _, _ = auxiliary
    boxes = decoder_boxes[-1].detach().float()
    logits = decoder_logits[-1].detach().float()
    reconstructed = detector.model[-1].postprocess(boxes, logits.sigmoid())
    if not torch.equal(reconstructed, stock_output):
        raise RuntimeError("decoder reconstruction differs from stock RT-DETR output")
```

Convert batched labels using `batch_idx`, cache one CPU record per image, and assert
exactly 129/548 records. Hash the detector state before and after both splits. The CLI
must never call `backward`, create an optimizer, or enable detector gradients.

- [ ] **Step 4: Implement paired development and official-val metrics**

For development, compute stock metrics once and oracle metrics for every frozen alpha,
then call `select_alpha`. For official validation, compute paired stock and only the
selected oracle metrics from the same cached record sequence. Convert a postprocessed
tensor to the existing metric schema with:

```python
def prediction_record(postprocessed: torch.Tensor, *, conf: float = 0.001) -> dict[str, torch.Tensor]:
    selected = postprocessed.detach().float().cpu()
    selected = selected[selected[:, 4] > conf]
    return {
        "boxes": selected[:, :4],
        "scores": selected[:, 4],
        "classes": selected[:, 5].long(),
    }
```

Call `src.iber_evaluation.compute_detection_metrics` unchanged. Write canonical
`alpha-selection-report.json`, `quality-oracle-report.json`, and
`quality-oracle-decision.json` only after all checks pass.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
python -m pytest tests/test_rtdetr_quality_oracle.py tests/test_iber_evaluation.py tests/test_rtdetr_iber.py -q
python -m compileall -q src scripts tests
```

Expected: all selected tests and compilation pass.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add scripts/run_rtdetr_quality_oracle.py tests/test_rtdetr_quality_oracle.py
git commit -m "experiment: add frozen quality oracle runner"
```

### Task 4: Full verification and immutable server deployment

**Files:**
- No source file changes unless a test first reproduces an engineering failure.

- [ ] **Step 1: Run the full local suite**

Run: `python -m pytest -q`

Expected: zero failures; existing intentional skips may remain.

- [ ] **Step 2: Push and independently verify the source branch**

Run:

```bash
git push origin codex/iber-be
git rev-parse HEAD
git ls-remote origin refs/heads/codex/iber-be
```

Expected: local and remote 40-character commit SHAs are identical.

- [ ] **Step 3: Verify the current server before deployment**

Connect only after the direct SSH handshake reports the pinned ed25519 fingerprint
`SHA256:FPVBIMs2LoVe0RenG9xDN5KvN99tgIcdPP9rY8Ym+u8`. Verify one RTX 4090, driver
550.142, the pinned Python/PyTorch/Ultralytics runtime, dataset and baseline hashes,
at least 10 GiB free space, and no competing training process.

- [ ] **Step 4: Deploy a new immutable source and run root**

Create source `/data/uav/source/uav-detection-baselines-<sha12>` and run
`/data/uav/runs/rtdetr-quality-oracle/<sha12>-seed10000`. Reuse the verified virtual
environment and existing local assets; do not mutate an older source or run root.

- [ ] **Step 5: Run CUDA smoke and the oracle**

First run the focused test on CUDA and one real batch that asserts exact stock Top-300
equality. Then execute:

```bash
/data/uav/venvs/iber-be-v1/bin/python scripts/run_rtdetr_quality_oracle.py \
  --baseline-checkpoint /data/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt \
  --dataset-root /data/uav/datasets/VisDrone \
  --cache-root /data/uav/cache/rtdetr-quality-oracle-<sha12> \
  --report-root /data/uav/runs/rtdetr-quality-oracle/<sha12>-seed10000/report \
  --device 0
```

Expected: process exit 0 and decision `passed` or `scientific_failed`; both are valid
scientific outcomes.

- [ ] **Step 6: Publish evidence transactionally**

Publish the design, plan, source commit, environment, cache manifest, alpha-selection
report, official-validation report, decision, and SHA-256 inventory to the existing
results branch. If server HTTPS is unavailable, download over pinned SSH, verify every
SHA locally, commit locally, push from the desktop, and verify the remote result SHA.

### Task 5: Apply the frozen scientific branch

**Files:**
- Create a new design and plan only for the branch selected by the immutable decision.

- [ ] **Step 1: If the oracle passes**

Freeze the positive evidence and design the smallest learnable quality probe with three
controls: C0 stock probability, C1 detached probability/entropy/geometry, and Q detached
decoder hidden plus C1. Train only on the authorized internal training partition and
evaluate official validation once. A quality head is eligible for a 30-epoch detector
screen only if it beats C0 and C1 by the separately predeclared probe threshold.

- [ ] **Step 2: If the oracle fails**

Freeze `scientific_failed`, do not implement a quality head, and write a new exact fixed
FDR reproduction design based on the official D-FINE distribution-regression mechanics.
The fixed FDR reproduction must use the same baseline protocol and must pass its own
low-cost engineering/oracle checks before any 30-epoch detector screen.

- [ ] **Step 3: Continue to 30/100 epoch only after the selected candidate passes**

Run fixed 10% seed0 for 30 epochs with the unified baseline parameters and paired
independent evaluation. Only a Gate2-passing candidate advances to full-data seed0 100
epochs. Preserve strict resume authority, publish every completed epoch, recover saved
but unpublished epochs, and finish with independent stock/candidate evaluation and a
remote-verified final result commit.
