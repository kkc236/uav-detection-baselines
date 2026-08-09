# FrequencyCM Complementarity Upper-Bound Oracle Design

## 1. Objective

Measure whether the completed FrequencyCM-v1 detector contains useful detections that the mature FDR detector does not contain. The experiment is diagnostic only: it does not train a model, tune a threshold, or claim a deployable mAP gain.

The result decides whether a protected, lightweight frequency-evidence side branch (CM-v2) has enough residual headroom to justify implementation.

## 2. Frozen authorities

- Dataset and evaluator: the same 548-image VisDrone validation set, class mapping, preprocessing, `imgsz=640`, `conf=0.001`, `NMS=false`, and `max_det=300` used by the strict FDR paper evaluation.
- FDR checkpoint: GitHub Release `fdr-formal-d97e1eb7-live`, asset `fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt`, SHA-256 `c2f638744508adfe7b6c4a1ef3e08c503273f628062e4650ad59ffff4c6588c2`.
- FrequencyCM checkpoint: GitHub Release `fdr-frequencycm-formal-d3655b14-live`, asset `fdr-frequencycm-formal-d3655b14-epoch-0100.pt`, SHA-256 `2bbcd6057fefed5792f786a18e603f8feca3ec426a6f68938f5f8ada1603a141`.
- Source authority: FrequencyCM integration commit `d3655b14c17a3c8ca14e1888517b6fde4e059766`; FDR mechanism authority remains `d97e1eb7`.

Any checkpoint, dataset, class-map, preprocessing, or evaluator mismatch stops the oracle and produces an engineering-failure report. It must not be repaired by silently substituting another artifact.

## 3. Non-goals

- Do not train or fine-tune FrequencyCM, CM-v2, a selector, or a quality head.
- Do not search alpha, confidence thresholds, IoU thresholds, NMS settings, or candidate counts.
- Do not use ground-truth oracle scores as deployable detector scores.
- Do not describe the result as a final CM-v2 gain.
- Do not overwrite the completed FrequencyCM-v1 negative result.

## 4. Exact prediction cache

Run each checkpoint once through the frozen validation loader and cache, per image:

- image identifier and original/resized geometry;
- all 300 decoder query boxes before flattened class selection or NMS;
- all ten class probabilities or logits for every query, yielding the exact 3,000 query-class pairs needed to reconstruct stock top-300 selection;
- query index and model source;
- all ground-truth boxes, classes, and size bucket labels.

Cache files are immutable and include checkpoint, data, evaluator, source, and content SHA-256 values. A stock reconstruction from each cache must reproduce that checkpoint's ordinary Precision, Recall, AP50, AP75, and mAP50-95 within the evaluator's frozen numerical tolerance before any oracle calculation is accepted.

## 5. Oracle family

### 5.1 Coverage oracle

For every ground-truth instance, compute the best same-class IoU available from:

1. FDR candidates only;
2. FrequencyCM candidates only;
3. the union of both candidate sets.

Report raw coverage and deterministic one-to-one matched recall at IoU 0.50 and 0.75. Break results down by tiny, small, medium, large, and the ten VisDrone classes.

The primary coverage quantity is:

```text
union coverage - max(FDR coverage, FrequencyCM coverage)
```

This measures genuinely complementary candidates rather than ordinary score reordering.

### 5.2 Matched class-conditional quality oracle

For each arm, perform deterministic one-to-one same-class assignment that maximizes total IoU. Assigned candidates receive their matched IoU as perfect diagnostic utility; unassigned duplicates and background candidates receive zero utility. Use the same assignment, utility, tie-breaking, and top-300 rule for four arms:

1. FDR only;
2. FrequencyCM only;
3. duplicated FDR control (`FDR + FDR`);
4. FDR and FrequencyCM union (`FDR + FrequencyCM`).

Run the unchanged evaluator on the top-300 oracle-ranked candidates. The duplicated-FDR arm must not gain beyond numerical tolerance; otherwise the union implementation is invalid.

The primary candidate-complementarity quantity is:

```text
oracle_mAP(FDR + FrequencyCM)
  - max(oracle_mAP(FDR), oracle_mAP(FrequencyCM))
```

The same difference is reported for AP50 and AP75. Absolute oracle mAP is secondary because it is non-deployable and includes perfect ground-truth ranking.

### 5.3 Image-level selector oracle

Select either the complete FDR prediction set or the complete FrequencyCM prediction set for each image using a frozen ground-truth utility: deterministic one-to-one matched same-class IoU sum. Preserve the selected detector's original scores and run the standard evaluator.

This is a weaker upper bound for a future image-level gate. It must be reported separately from the candidate-union oracle.

### 5.4 Missed-target audit

Enumerate ground-truth objects that are:

- covered by both models;
- covered only by FDR;
- covered only by FrequencyCM;
- covered by neither model.

Produce counts and rates at IoU 0.50 and 0.75 by scale and class. This identifies whether any complementarity is concentrated in tiny/small targets or is only random large-object variation.

## 6. Frozen decision boundary

Use the union candidate-complementarity delta and one-to-one tiny/small recall delta. Do not change these bands after inspecting results.

| Decision | Candidate-oracle mAP delta | Tiny/small recall@0.50 delta | Consequence |
|---|---:|---:|---|
| Green | at least `+0.010` | or at least `+0.020` | Design CM-v2 and run a learnability probe |
| Yellow | at least `+0.003` | or at least `+0.010` | Signal exists but expected deployable gain is small; perform probe before architecture work |
| Red | below both Yellow thresholds | below both Yellow thresholds | Freeze the frequency direction as scientifically unpromising |

The thresholds are intentionally larger than a desired deployable gain because a learned gate can capture only part of a perfect oracle advantage.

## 7. Required outputs

- `oracle-summary.json`: authorities, reproduction checks, all oracle metrics, decision, and hashes;
- `coverage-by-scale.csv` and `coverage-by-class.csv`;
- `missed-target-categories.csv`;
- `oracle-arms.csv`: stock and oracle metrics for every arm;
- `frequencycm-complementarity-report.md`: human-readable interpretation with explicit non-deployable caveat;
- immutable FDR and FrequencyCM prediction caches or cache manifests when the full cache is too large for Git;
- SHA-256 manifest for every report and cache artifact.

## 8. Validation and failure handling

- Unit tests cover same-class utility, deterministic tie-breaking, top-300 selection, one-to-one matching, duplicate-control neutrality, scale buckets, and empty-GT/empty-prediction images.
- Integration tests reconstruct both stock detector metrics from small synthetic and real cached samples.
- Full execution is single-pass and deterministic.
- Engineering failures are repaired test-first without altering the scientific definition.
- A Red scientific result is frozen and uploaded; it is not followed by threshold changes or a second oracle variant.

## 9. Interpretation boundary

A Green result means only that the completed FrequencyCM model contains complementary candidates and that a protected side-gate is worth studying. It does not prove that frequency statistics can learn the oracle selector.

A Yellow result supports only a low-cost learnability probe. A Red result rejects CM-v2 as a priority and redirects work to an orthogonal high-resolution recall module such as IRA-Lite.

Because the same official validation set has already informed prior project decisions, this oracle is explicitly design-selection evidence. A later CM-v2 result on that validation set cannot by itself serve as an untouched confirmatory test; final claims require a prospectively frozen protocol and independent evidence such as a new seed, split, or test evaluation.
