# RT-DETR Quality-Reranking Upper-Bound Oracle Design

Date: 2026-08-03

## Decision

The next scientific candidate is a diagnostic-only, ground-truth quality-reranking
oracle for the frozen mature Ultralytics RT-DETR-L baseline. It tests whether imperfect
Query-by-class score ordering is still a material AP bottleneck after RT-DETR's existing
IoU-aware Varifocal Loss. It does not train a detector, alter a box coordinate, start the
30-epoch screen, or weaken any existing Gate threshold.

The user approved this candidate and delegated execution decisions on 2026-08-03. If
the oracle fails its frozen decision, the score-quality direction stops and the next
candidate is an exact fixed-FDR reproduction. A failed oracle cannot be repaired by
tuning on the official validation split.

## Frozen authority

- Detector: the existing mature Ultralytics RT-DETR-L baseline checkpoint.
- Runtime: Ultralytics 8.4.90, Python 3.10.12, PyTorch 2.5.1+cu121, CUDA 12.1,
  NVIDIA GeForce RTX 4090 with driver 550.142.
- Data: the fixed hashed 647-image VisDrone training subset and the complete 548-image
  official validation split.
- Dataset SHA-256: `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`.
- Fixed-subset SHA-256: `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`.
- Input and evaluation: image size 640, batch 8, workers 8, device 0, cache disabled,
  confidence 0.001, `max_det=300`, and `NMS=False`.
- Detector parameters are frozen and must have no gradients before or after extraction.
- Stock boxes and stock class logits are taken from the final decoder layer without a
  second forward path or a modified matcher.

## Internal development split

The official 548-image validation split is never used to select the reranking exponent.
The fixed 647-image training subset is deterministically divided by sorting each relative
image path by

```text
SHA256("rtdetr-quality-oracle-dev-v1\0" + relative_image_path)
```

and assigning the first 129 images to the internal development split. The remaining
518 images are not needed by this diagnostic. The exact ordered development image list,
its canonical SHA-256, and the public subset authority are written to the report.

## Stock inference semantics

Ultralytics 8.4.90 RT-DETR inference applies sigmoid to the final decoder class logits,
flattens all `300 queries x 10 classes`, selects the global top 300 values, recovers the
query index by integer division and the class by modulo, and gathers the corresponding
unchanged query boxes. The oracle reproduces those semantics exactly. It must not reduce
each query to its maximum class before global Top-300 selection.

For each image, stock input tensors have these contracts:

```text
boxes:  [300, 4] normalized cx,cy,w,h
logits: [300, 10] pre-sigmoid class logits
target_boxes:   [N, 4] normalized cx,cy,w,h
target_classes: [N] integer class indices in [0, 9]
```

## Perfect same-class quality

For every query `q` and class `c`, compute

```text
quality[q,c] = max IoU(stock_box[q], target_box[i])
               over targets i whose class is c
```

and use zero when the image contains no target of class `c`. Quality is detached,
finite, bounded to `[0,1]`, and derived only from unchanged stock boxes and labels. It
does not use Hungarian assignment or one-to-one suppression because the intended upper
bound is the quality signal available to a learnable class-conditional IoU predictor.
The class-conditional definition is deliberate: this experiment asks whether any ideal
per-class quality signal can still repair final RT-DETR ranking after VFL. It is therefore
an optimistic upper bound, not a deployable scalar localization-quality head. Failure
also rules out the less-informed class-agnostic Query-quality variant under this scoring
family; passing only authorizes controls that test whether the extra signal is learnable.

For an exponent `alpha`, rerank with

```text
stock_probability[q,c] = sigmoid(logits[q,c])
oracle_score[q,c] = stock_probability[q,c] * quality[q,c] ** alpha
```

then apply the exact global Query-by-class Top-300 operation. Zero quality remains zero.
Boxes are gathered from the unchanged stock tensor; only the ranking score and therefore
the selected query/class pairs may change.

## Frozen alpha selection

The only candidates are `alpha in {0.25, 0.5, 1.0, 2.0}`. Evaluate stock and all four
oracle candidates on the 129-image internal development split using the same metric code.
Choose the candidate with the lexicographically greatest tuple

```text
(mAP50-95, AP75, AP50, -alpha)
```

so exact ties select the smaller exponent. No additional exponent, calibration function,
confidence threshold, class-specific weight, or image-size rule may be introduced after
seeing development or official-validation metrics.

## Official validation and decision

After selecting alpha, process the official 548-image validation split once and emit
paired stock and oracle predictions from the same decoder tensors. Compute the existing
no-NMS metrics: mAP50-95, AP50, AP75, AP-tiny, AP-small, precision, and recall.

The score-quality direction is viable only if all artifacts and metrics are finite and
both exact conditions hold:

- oracle mAP50-95 minus stock mAP50-95 is at least `0.0050`;
- oracle AP75 is strictly greater than stock AP75.

Passing authorizes a separately designed learnable quality probe. It does not authorize
the 30-epoch detector screen by itself. Failure freezes the diagnostic as
`scientific_failed`, stops quality-head work, and moves to an exact fixed-FDR reproduction.
The threshold, exponent grid, development split, official validation pass, and metric
implementation must not be changed after execution.

## Artifacts and isolation

- Write an immutable cache containing only detached CPU stock boxes, logits, targets,
  image identifiers, and split identity. Bind it to baseline, dataset, subset, runtime,
  source, schema, and ordered development-list hashes.
- Reject a non-empty cache or report root unless every authority and artifact hash matches
  a previously complete cache intended for resume.
- Publish canonical JSON for the manifest, alpha-selection report, official-validation
  report, decision, execution environment, and SHA-256 inventory.
- A report is valid only when it covers exactly 129 development images and 548 official
  validation images, all class indices are valid, detector hashes before and after are
  equal, and no detector parameter has a gradient.
- The non-exported PyTorch model auxiliary tuple supplies final decoder boxes and logits.
  Reconstructing stock Top-300 from those tensors must be exactly equal to the model's
  own returned stock output for every batch or the run is an engineering failure.
- Engineering failures may be corrected with tests and rerun under a new source commit
  and immutable run root. Scientific failure may not be relabeled or tuned away.

## Test strategy

Unit tests lock same-class maximum IoU, empty-class behavior, exact flattened Top-300
semantics, duplicate-query class handling, alpha tie-breaking, immutable cache authority,
the exact `+0.0050` and strict AP75 decision boundaries, and detector isolation. CLI
contract tests lock all public constants and forbid options that could tune the split,
alpha grid, threshold, confidence, or validation protocol. A CUDA smoke test compares
the stock path byte-for-byte with the Ultralytics 8.4.90 head postprocess before the full
oracle is permitted to run.
