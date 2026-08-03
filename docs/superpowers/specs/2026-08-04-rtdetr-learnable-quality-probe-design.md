# RT-DETR Learnable Quality Probe Design

Date: 2026-08-04

Status: approved and frozen after the quality-reranking oracle passed

## 1. Decision

The next stage is a detached, class-conditional quality probe for the mature
Ultralytics RT-DETR-L baseline. The passed upper-bound oracle established that
same-class localization quality can materially improve the final flattened
Query-by-class ranking and selected `alpha=2.0`. This stage asks the narrower question:

> Can the final decoder hidden state predict enough of that quality signal to beat both
> the untouched stock ranking and a geometry/probability-only control under a frozen,
> validation-blind protocol?

This is an offline probe study. It does not alter detector weights, boxes, logits,
matching, training data, or inference topology; it does not yet authorize a detector
30-epoch screen. Only the hidden-aware arm `Q` is an eligible scientific candidate.
`C1` is an information control and can never advance by itself.

Three implementation approaches were considered:

1. Reuse the quality-oracle cache. This is rejected because that cache has boxes,
   logits, and targets but no decoder hidden state.
2. Run the frozen detector online during every probe epoch. This is rejected because it
   repeats expensive evidence extraction and makes exact resume and arm comparability
   harder to audit.
3. Build a new hidden-aware, create-only evidence cache, then train equal-parameter C1
   and Q arms offline. This is selected because both arms consume the same detector
   evidence, official validation can remain physically inaccessible until internal
   selection is frozen, and all later work can resume from verified hashes.

## 2. Frozen authority

- Detector: mature Ultralytics RT-DETR-L baseline checkpoint
  `/data/uav/weights/matched_baseline/matched-baseline-best-epoch-0100.pt`.
- Baseline checkpoint SHA-256:
  `54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B`.
- Dataset SHA-256:
  `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`.
- Fixed 647-image subset SHA-256:
  `52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0`.
- Runtime: Python 3.10.12, PyTorch 2.5.1+cu121, Ultralytics 8.4.90,
  CUDA 12.1, NVIDIA GeForce RTX 4090, driver 550.142.
- Detector input/evaluation: image size 640, extraction batch 8, workers 8,
  confidence `0.001`, `max_det=300`, and `NMS=False`.
- Seed: `0` only.
- Oracle authority: a canonical immutable oracle decision with `status="passed"` and
  selected `alpha=2.0`. Its bytes, SHA-256, source commit, cache manifest SHA-256, and
  report inventory SHA-256 are inputs to the probe authority.
- Probe implementation source commit, schema hash, environment hash, data hashes,
  baseline hash, and every selected checkpoint byte count/SHA-256 are recorded in all
  downstream manifests and decisions.

No scientific constant is exposed as a CLI override. Paths and device are operational
arguments; alpha, features, model width, loss, seed, optimizer, batch/chunk sizes,
epoch count, checkpoint selector, gates, confidence, and validation count are fixed in
source and tested.

## 3. Data partition and validation isolation

The already-authorized 647 training images are divided by the existing frozen oracle
split. The internal-development list is still the first 129 paths under

```text
SHA256("rtdetr-quality-oracle-dev-v1\0" + relative_image_path)
```

ranking. Its ordered UTF-8/LF path SHA-256 remains
`FCF8749BAADBA8BDDF5870F472BDE1E937156AFBCEEFDA9F96FED21FA6BB0514`.
The probe-training partition is the ordered 647-path authority with those 129 paths
removed. It contains exactly 518 unique images, is disjoint from internal development,
and has ordered path SHA-256
`1E46817FFFBDBCBA0BA1675CA6142ABABBD6147394AA1D0F10B57F0ECAF7236D`.

The 548 official validation images must remain unopened before internal selection is
immutable. Before that point the runner must not enumerate `images/val`, construct a
validation loader, inspect an oracle validation cache, compute validation identities,
or resolve validation records. The pre-validation authority check uses the frozen
baseline, 647-subset, 518-list, 129-list, runtime, oracle-decision, schema, and source
hashes only.

After all 20 epochs for both trainable arms are complete, one C0/C1/Q internal selection
report freezes:

- the exact C0 internal metrics;
- the selected C1 checkpoint epoch, bytes, SHA-256, and internal metrics;
- the selected Q checkpoint epoch, bytes, SHA-256, and internal metrics;
- the internal Gate inputs, exact decimal deltas, and decision;
- every authority hash needed to reproduce the selection.

Only a complete, canonical, create-only internal report with a passing Gate unlocks the
official-validation stage. The official 548 images then receive exactly one frozen
detector evidence-extraction pass. Resume may load the new probe validation cache after
verifying it, but it must never rerun completed validation inference. The earlier oracle
validation cache is forbidden because it does not contain hidden states.

## 4. Output-neutral hidden capture

The detector stays in `eval()` under `torch.inference_mode()`, and every detector
parameter has `requires_grad=False`. The final hidden tensor is captured from the input
to the classification head used by eval inference:

```text
head = detector.model[-1]
eval_index = head.decoder.eval_idx
module = head.dec_score_head[eval_index]
hidden = forward_pre_hook(module).args[0]
```

The hook only stores `args[0].detach()` and returns `None`; it cannot replace or mutate
the input. It is removed in a `finally` block and must fire exactly once per detector
call. For the pinned model the captured shape is `[B,300,256]`, and applying the hooked
`dec_score_head` to the captured hidden must reproduce the final auxiliary logits
exactly.

Before any cache write, a train-subset canary runs the same preprocessed batch once
without the hook and once with the hook. Stock postprocess output, final auxiliary boxes,
and final auxiliary logits must match byte-for-byte in shape, dtype, and contents.
The hooked run must also reconstruct the model's own stock output exactly through the
existing flattened Top-300 path. Hook neutrality is proven on training data; it does not
consume an extra official-validation pass.

Detector state fingerprints before and after each extraction stage must be identical,
all detector gradients must remain `None`, and no detector parameter may enter either
probe optimizer. Any violation is `engineering_invalid`, never a scientific result.

## 5. Hidden-aware evidence cache

The probe uses a new cache schema. Each record has exactly:

```text
image_id:       canonical relative POSIX path
boxes:          float32 [300,4] normalized cx,cy,w,h
logits:         float32 [300,10] pre-sigmoid class logits
hidden:         float32 [300,256] final eval-layer decoder hidden
quality:        float32 [300,10] detached same-class maximum IoU target
target_boxes:   float32 [N,4] normalized cx,cy,w,h
target_classes: int64   [N] values in [0,9]
```

Every tensor is detached, finite, contiguous, and on CPU. `quality` is recomputed from
`boxes`, `target_boxes`, and `target_classes` during validation and must match the cached
tensor exactly. Train evidence is stored as separate `probe_train` and `internal_dev`
shard sequences. Official evidence is stored under a distinct cache stage that cannot
exist before a passing internal selection.

Cache shards contain at most 32 images. Each stage writes an immutable intent first,
writes each shard with create-only atomic publication and fsync, records bytes and
SHA-256, and publishes canonical `manifest.json` last. A partial stage may resume only
when its intent, existing shard set, identities, ordering, schema, and all authority
hashes match exactly; only missing fixed shards may be written. A complete stage is
read-only. Cache loading uses `torch.load(..., map_location="cpu", weights_only=True)`,
rejects symlinks/reparse points and extra files, and revalidates every record.

## 6. Controls and feature contract

Let `p[q,c] = sigmoid(logits[q,c])`. All inputs below are explicitly detached and
computed in float32.

### 6.1 C0: untouched stock control

C0 has no learned model. Its score is exactly:

```text
score_C0[q,c] = p[q,c]
```

It then uses the existing global flattened `300 queries x 10 classes` Top-300. C0 is not
`p^3` and does not pass through the quality MLP. Its output must be byte-for-byte equal
to stock RT-DETR postprocess for every evaluated batch.

### 6.2 C1: no-hidden information control

For each query and candidate class, C1 receives these 20 values:

```text
p(class)                                                       1
mean Bernoulli class entropy over the query's 10 sigmoid scores 1
cx, cy, w, h                                                   4
log(max(w,1/640)), log(max(h,1/640))                            2
area = w*h                                                     1
aspect = w/max(h,1/640)                                        1
one-hot candidate class                                       10
                                                               --
total                                                          20
```

The entropy is

```text
-mean_c(p_c*log(p_c) + (1-p_c)*log(1-p_c))
```

after clamping probabilities to `[1e-6,1-1e-6]`. Geometry stays in normalized detector
coordinates. C1 receives no decoder hidden value, feature, statistic, projection, or
surrogate.

### 6.3 Q: hidden-aware candidate

Q receives the exact 20 C1 values plus the corresponding query's detached 256-value
final decoder hidden vector, repeated across the ten candidate classes.

To isolate information rather than parameter count, both trainable arms use a 276-value
input and the same architecture:

```text
Linear(276,64,bias=True) -> SiLU -> Linear(64,1,bias=True)
```

C1 fills the final 256 positions with exact zeros; Q fills them with the detached hidden
vector. Both models are initialized from the same seed0 state dictionary but are trained
independently. They therefore have identical parameter counts, initial bytes, optimizer
rules, sample order, and checkpoint schedule. The only arm difference is hidden-state
information.

The network emits one quality logit per query/class. `sigmoid` converts it to a bounded
predicted quality in `[0,1]` only for scoring and reporting.

## 7. Target, loss, and optimization

For each query `q` and class `c`, the detached target is exactly the oracle target:

```text
quality[q,c] = max IoU(boxes[q], target_boxes[i])
               over targets i with target_classes[i] == c
```

and zero when class `c` is absent. There is no Hungarian matching, positive mining,
negative subsampling, class weighting, focal term, auxiliary target, or detector loss.

The single training loss for both C1 and Q is soft-target binary cross entropy over all
`B*300*10` rows:

```text
binary_cross_entropy_with_logits(predicted_quality_logits, quality, reduction="mean")
```

Optimization is frozen as follows:

```text
seed: 0
epochs: exactly 20 unless interrupted by an engineering failure
precision: float32; AMP disabled
optimizer: pinned Ultralytics 8.4.90 MuSGD
lr0: 0.01, constant; no scheduler and no warmup
momentum: 0.937
weight_decay: 0.0005 on 2-D Linear weights; 0 on biases
nesterov: True
Muon/SGD mixture: pinned Ultralytics defaults used by MuSGD construction,
                  muon=0.2, sgd=1.0
training image batch: 8 (at most 24,000 query/class rows per step)
cache shard: 32 images
evaluation image chunk: 8
training workers: 0
gradient clipping: none
```

The per-epoch image permutation is `torch.randperm(518)` from a CPU generator seeded
with `seed + epoch` for one-based epochs. This makes each epoch's order independent of
resume history. C1 and Q consume the same permutation. All Python, NumPy, CPU, and CUDA
RNGs are seeded; deterministic algorithms and the required CUDA deterministic workspace
configuration are enabled before model construction.

Every epoch produces a create-only checkpoint and canonical sidecar containing model,
optimizer, epoch, arm, protocol, RNG/permutation authority, source, cache manifest,
environment, and bytes/SHA-256. Resume accepts only the highest contiguous verified
checkpoint/sidecar pair for the same arm and authority. Loading is CPU-safe with
`weights_only=True`, then model and optimizer state are restored before the next frozen
epoch permutation. A checkpoint cannot cross arms or stages.

## 8. Reranking and internal checkpoint selection

For C1 and Q, the frozen score is:

```text
predicted_quality[q,c] = sigmoid(mlp(features[q,c]))
score[q,c] = p[q,c] * predicted_quality[q,c] ** 2.0
```

The boxes remain unchanged. Scores use the existing exact flattened Query-by-class
Top-300, confidence `0.001`, and no NMS. The existing
`src.iber_evaluation.compute_detection_metrics` implementation computes mAP50-95,
AP50, AP75, AP-tiny, AP-small, precision, and recall.

C0 is evaluated once on the 129-image internal-development cache. After each of the 20
epochs, each trainable arm is evaluated on the same ordered internal records. The
selected checkpoint for each arm is the lexicographic maximum of:

```text
(mAP50-95, AP75, AP50, -epoch)
```

so an exact metric tie selects the earlier epoch. Selection uses finite full-precision
metrics, never rounded display values, training loss, official validation, or a combined
score. Both arms always finish all 20 epochs before selection; there is no early stopping.

## 9. Frozen internal Gate

Let C0, C1, and Q denote the internal metrics from the fixed stock control and selected
checkpoints. Q passes only when all four exact conditions hold:

```text
Q.map  - C0.map  >= 0.0050
Q.ap75 - C0.ap75 >  0
Q.map  - C1.map  >= 0.0050
Q.ap75 - C1.ap75 >  0
```

Gate arithmetic converts full-precision metric inputs with `Decimal(str(value))` and
emits canonical decimal strings. Equality passes the mAP thresholds and fails the strict
AP75 thresholds. NaN, infinity, missing checkpoints, incomplete epochs, authority drift,
or detector mutation is `engineering_invalid`.

If Q fails a scientific condition, the runner writes an immutable
`scientific_failed` decision, never opens official validation, does not promote C1, does
not weaken a threshold, and pivots to the already-audited FDR-only migration.

## 10. One-shot official validation and screen eligibility

Only after the passing internal report is verified does the runner create the new
hidden-aware official-validation cache. The one detector pass emits shared evidence for
C0, selected C1, and selected Q; all three are then evaluated from that same ordered
cache. No checkpoint, architecture, calibration, exponent, threshold, confidence,
feature, or optimizer choice may change after seeing official metrics.

The new C0 official metrics must reproduce the frozen native-auxiliary stock authority.
The AP metrics remain exact-authority checks. Precision and recall use only the existing
`1e-8` non-Gate tolerance from the 2026-08-04 quality-oracle floating-point amendment.
Any larger mismatch is engineering-invalid.

Q becomes eligible for the separate detector 30-epoch screen only when:

```text
Q.map  > C0.map
Q.ap75 > C0.ap75
```

Both comparisons are strict. C1 official metrics are reported for control interpretation
but never confer eligibility. Official failure freezes `scientific_failed`, stops the
quality-probe branch, and does not authorize tuning or another validation pass.

Passing this one-shot Gate authorizes only a new, separately reviewed detector-screen
design. It does not itself attach the probe to RT-DETR, train detector parameters, or
start the 30-epoch screen.

## 11. State machine, artifacts, and safe resume

The only legal stage order is:

```text
authority
-> hook-neutrality canary
-> hidden-aware 518/129 train cache
-> C1 epochs 1..20
-> Q epochs 1..20
-> immutable internal C0/C1/Q selection
-> internal decision
-> hidden-aware official-val cache (only if internal passed)
-> one-shot official C0/C1/Q report
-> final decision
```

Required create-only artifacts include:

- train-cache intent, shards, and final manifest;
- one checkpoint and canonical sidecar per arm/epoch;
- per-epoch internal metric reports;
- `internal-selection-report.json` and `internal-quality-probe-decision.json`;
- official-cache intent, shards, and final manifest only after internal pass;
- `official-quality-probe-report.json` and `quality-probe-decision.json`;
- execution environment and complete SHA-256 inventory.

A resume validates the stage prefix, every existing byte count/hash, exact authority,
contiguous epoch sequence, selected checkpoint, and state transition before doing work.
It skips complete immutable stages, fills only missing cache shards under a matching
intent, and never overwrites an artifact. A partial or conflicting report tree is
rejected. Engineering fixes require a new source commit and immutable run root unless a
separate written compatibility amendment proves that existing evidence can be reused.

## 12. Failure policy and scope boundary

- Hook mismatch, non-finite evidence, wrong shape/dtype, unsafe cache, detector state or
  gradient drift, nondeterministic replay, stock reconstruction mismatch, authority
  mismatch, or premature validation access is `engineering_invalid`. Fix the engineering
  defect under tests and use an allowed immutable resume/new run; do not interpret it.
- Internal Q Gate failure is `scientific_failed`; validation remains unopened and the
  next work is the FDR-only migration.
- Official Q eligibility failure is `scientific_failed`; do not tune on validation or
  rerun it.
- C1 can expose whether probability/geometry explains the signal, but it can never be
  substituted for Q as the candidate.
- No detector parameter, decoder layer, score head, matcher, box, query count, class
  logit, augmentation, official-validation protocol, or historical oracle artifact is
  modified by this stage.

## 13. Test strategy

Focused tests must lock:

- the exact 518/129 counts, ordered hashes, uniqueness, and disjointness;
- C1's 20-value semantic features, zero hidden slots, Q's 256 hidden values, and exact
  `[B,300,10,276]` feature/output shapes;
- hook placement at `dec_score_head[head.decoder.eval_idx]`, one-fire behavior, cleanup,
  final-logit identity, and byte-for-byte no-hook/hook output neutrality;
- detector `eval`/inference mode, no gradients, unchanged state fingerprint, and optimizer
  isolation;
- exact float32/int64 cache schema, target recomputation, create-only shard/manifest
  authority, corruption rejection, safe `weights_only=True` loading, and partial resume;
- seed0 replay, identical C1/Q initialization, identical epoch permutations, exact
  MuSGD groups, 20-epoch completion, checkpoint corruption/cross-arm rejection, and
  lexicographic checkpoint tie-breaking;
- exact C0 reconstruction, alpha `2.0`, unchanged boxes, flattened Top-300, and no NMS;
- delayed validation access and exactly one official detector pass after immutable
  internal selection;
- internal `+0.0050`/strict AP75 boundaries against both controls and strict official
  Q-over-C0 boundaries;
- terminal `scientific_failed` behavior and the prohibition on threshold weakening,
  C1 promotion, oracle-val reuse, or post-validation tuning.
