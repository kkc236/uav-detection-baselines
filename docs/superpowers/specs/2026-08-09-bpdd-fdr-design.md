# BPDD for FDR RT-DETR-L: Adversarially Audited Design

## 1. Decision and claim boundary

BPDD (Best-Progressive Distribution Distillation) is approved as a **training-only
research candidate** on top of the already validated FDR RT-DETR-L detector. It
does not change the deployed graph, classification scores, Query selection,
matching, post-processing, or prediction schema.

BPDD must not be described as inventing self-distillation. Its defensible claim
is the following narrow combination:

> For cumulative FDR distributions, a matched Query edge is supervised by a
> detached, GT-conditioned mixture of only its later decoder distributions, and
> only to the extent that this future mixture is more accurate than the current
> distribution.

The proposal is inspired by, but is not an exact reproduction of, D-FINE's
GO-LSD. The paper and code must cite:

- D-FINE and GO-LSD: <https://arxiv.org/abs/2410.13842>
- official D-FINE source authority, commit
  `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6`:
  <https://github.com/Peterande/D-FINE/tree/7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6>
- DETRDistill: <https://openaccess.thecvf.com/content/ICCV2023/html/Chang_DETRDistill_A_Universal_Knowledge_Distillation_Framework_for_DETR-families_ICCV_2023_paper.html>
- KD-DETR: <https://openaccess.thecvf.com/content/CVPR2024/html/Wang_KD-DETR_Knowledge_Distillation_for_Detection_Transformer_with_Consistent_Distillation_Points_CVPR_2024_paper.html>
- Localization Distillation: <https://arxiv.org/abs/2102.12252>
- Teacher-bounded Regression: <https://proceedings.neurips.cc/paper/2017/file/e1e32e235eee1f970470a3a6658dfdd5-Paper.pdf>
- BYOT: <https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Be_Your_Own_Teacher_Improve_the_Performance_of_Convolutional_Neural_ICCV_2019_paper.html>

The final paper may claim BPDD as an original module only after a complete
literature search and empirical ablation support. Until then, all reports call
it a candidate.

The individual ideas of later-to-earlier self-distillation, four-edge
distribution KL, and “teacher is better” gating already exist. The only
potentially defensible novelty is their concrete FDR-DETR combination: a
GT-proper-score future-layer mixture, an actual-mixture superiority check, and
single-final-assignment matched-only isolation. If BPDD does not outperform a
final-layer teacher with the same better-only gate, the novelty claim fails.

## 2. Why this candidate follows the failure evidence

The project has repeatedly observed early gains that disappear late:

- FDR is the mature positive control and improves strict Control by `+7.055 pp`
  mAP50-95 at full-data seed0/100 epochs.
- FrequencyCM was positive early but converged to flat/negative evidence.
- SCADS became negative and suffered a late gradient failure.
- PFCR failed its internal learnability gate.
- original GLGM changed AP75 slightly but degraded Precision, Recall, F1, and
  AP50.

These failures make another inference-time feature, score, or box branch a
high-risk choice. BPDD instead regularizes the already successful FDR path and
vanishes entirely at inference. It also avoids assuming that the last decoder
layer is the best teacher for every small-object edge.

## 3. Immutable baseline and single-variable scope

The comparison is:

```text
control: validated RT-DETR-L + FDR
candidate: the same RT-DETR-L + FDR + BPDD
```

The candidate retains the frozen FDR authority:

- Ultralytics `8.4.90`, RT-DETR-L, six decoder layers, 300 Queries;
- FDR `reg_max=32`, `reg_scale=4.0`, `up=0.5`, cumulative logits;
- FGL weight `0.15` and preliminary-box supervision unchanged;
- same public and FDR-private seed0 initialization;
- same VisDrone train/val, sample order, augmentation sequence, MuSGD,
  batch 8, AMP scale 128, and evaluation;
- no pretrained weights, NMS disabled, `max_det=300`.

BPDD is the only scientific variable. PS-FGL, GO-LSD, DDF, GLGM, SCADS,
FrequencyCM, PFCR, boundary evidence, trajectory evidence, LPR, and score
reranking are excluded from this experiment.

## 4. Inputs and assignment authority

For the six decoder layers, BPDD consumes:

```text
Z: cumulative FDR corner logits [6, B, Q, 4, 33]
B: decoded boxes                [6, B, Q, 4]
R: preliminary reference       [B, Q, 4]
M: final decoder stock matches per image
G: ground-truth boxes
```

Only `M`, the already computed assignment for the final decoder layer, is used.
BPDD must not call the matcher, construct a union of assignments, change target
cardinality, or supervise unmatched Queries. Denoising Queries are excluded in
v1. This isolates the experiment from GO-LSD's cross-layer matching-union and
matched/unmatched DDF weighting.

## 5. Target-relative future teacher

For each final-matched Query `q`, target `t`, edge `e`, and layer `l`, the
existing FDR `bbox2distance` authority maps the GT edge to two adjacent bins
`k` and `k+1` with interpolation weights `a` and `1-a`. Define the detached
target-relative edge error of layer `j` as:

```text
P_j = softmax(Z_j)
E_j = -a log(P_j[k]) - (1-a) log(P_j[k+1])
```

For student layers `l=0..4`, only future layers `j>l` are eligible. Their
mixture weights and teacher distribution are:

```text
pi_lj = softmax(-stop_gradient(E_j) / tau)
T_l   = stop_gradient(sum_j pi_lj P_j)
```

The frozen first candidate uses `tau=0.5`. The teacher is an edge-wise mixture:
different edges of one Query may prefer different future layers. No gradient
passes through `pi_lj`, `T_l`, GT-derived bin indices, or reliability weights.

## 6. Better-only reliability gate

The teacher is allowed to act only when it improves the GT-relative error:

```text
E_T = -a log(T_l[k]) - (1-a) log(T_l[k+1])
r_l = clamp((stop_gradient(E_l) - stop_gradient(E_T) - delta) /
            max(stop_gradient(E_l), eps), 0, 1)
```

`r_l=0` makes the edge an exact no-op. The frozen margin is `delta=0.02` nats;
it suppresses numerically trivial teacher advantages and cannot be retuned after
validation. Relative improvement gives a bounded weight. The implementation
publishes the active-edge ratio and mean teacher improvement; hidden all-zero
behavior is an engineering failure.

## 7. Distillation loss

The per-edge loss is teacher-to-student KL divergence:

```text
D_l = KL(T_l || softmax(Z_l))
L_BPDD = lambda_bpdd * mean_{matched q, edge e, l=0..4}(r_l D_l)
```

The frozen initial candidate uses `lambda_bpdd=0.5`, `tau=0.5`, `delta=0.02`,
and `eps=1e-6`. The loss is divided by all eligible matched student edges, not
only active edges, so a sparse gate cannot amplify one example. Empty-GT
batches and batches with no active edge return a finite, graph-connected zero.
When enabled, BPDD adds one loss key, `loss_bpdd`, to the training sum.
FGL, preliminary-box, VFL, L1, and GIoU values are otherwise unmodified.

The final decoder layer is never a student. Later distributions and gates are
detached, so BPDD gradients may flow only into student-layer cumulative logits
and their upstream training graph. Tests must demonstrate that future logits
receive no gradient through the teacher branch.

## 8. Modular and compatibility contract

BPDD is implemented in an independent `src/bpdd_loss.py` functional unit. A
separate `configs/rtdetr-l-fdr-bpdd.yaml` exposes only training options:

```yaml
bpdd_loss:
  enabled: true
  weight: 0.5
  temperature: 0.5
  margin: 0.02
  eps: 1.0e-6
  matched_layer: final
  include_dn: false
```

The FDR graph itself stays byte-identical. The ordinary FDR YAML contains no
BPDD block and remains the exact ablation. `enabled: false` or `weight: 0`
must return the original FDR loss dictionary without even an added zero key,
and must give identical predictions, state-dict keys, parameters, GFLOPs, and
inference latency within the frozen numeric protocol.

Checkpoint loading rules:

- a mature FDR checkpoint loads into the BPDD model with no missing or
  unexpected model keys because BPDD has no parameters;
- a BPDD-trained checkpoint loads into the ordinary FDR inference model;
- resume requires matching BPDD source/protocol/options and may not silently
  change weight or temperature.

## 9. Adversarial engineering gates

### B0: mathematical unit gates

- future softmin weights sum to one and use only `j>l`;
- an exact future teacher produces the expected mixture and zero teacher grad;
- a worse/equal teacher produces `r=0` and exact zero loss;
- a better teacher produces positive finite loss and student-only gradients;
- empty GT, one layer, one match, extreme logits, AMP float16, and invalid
  shapes are handled deterministically.

### B1: criterion isolation

- reuse the final decoder stock assignment with zero additional matcher calls;
- no unmatched or DN BPDD term exists;
- BPDD off/weight zero preserves the complete stock/FDR loss mapping exactly;
- BPDD on changes only `loss_bpdd`; FGL and pre-box losses are bit-exact for
  identical supplied predictions and matches.

### B2: model and checkpoint compatibility

- candidate and FDR have identical parameter names, shapes, count, and initial
  tensor SHA256 values;
- constructing the BPDD candidate does not advance global CPU/CUDA RNG state;
- train/eval prediction tensors and postprocess outputs are exact;
- FDR -> BPDD -> FDR checkpoint round trip succeeds;
- candidate inference contains no BPDD operation or output.

### B3: real RTX 4090 preflight

- one real VisDrone `batch=8`, `imgsz=640` forward/backward/MuSGD step with
  AMP scale 128 is finite;
- gradient norms, active-edge ratio, teacher improvement, and loss are finite;
- public/FDR-private initialization, dataset, subset, and environment hashes
  match the mature FDR authority.
- two continuous optimizer steps equal a save/resume boundary followed by the
  second step for model, optimizer, scaler, EMA, and next-batch authority.

### B4: Query-trajectory diagnostic

- publish final-assignment versus per-layer-assignment agreement;
- publish per-layer IoU of the final-matched Query against its assigned GT;
- stratify agreement and IoU by tiny/small/medium/large;
- do not reinterpret disagreement as a positive result.

B4 is diagnostic rather than a new matching mechanism. V1 still uses only the
final stock assignment. Severe disagreement is a scientific warning and must be
reported with the final result.

## 10. Scientific progression and anti-overfitting rules

The existing official validation set has already informed earlier module
selection, so no BPDD hyperparameter may be tuned after looking at its official
validation result. The first candidate is frozen above.

1. Complete B0-B3.
2. Run one paired fixed-10% seed0 Screen30 from the same initial state and
   schedule. Neither arm inherits another screen checkpoint.
3. Gate uses the predeclared final epoch and tail-three mean, never a best
   checkpoint selected from the official validation curve.
4. Advance only if final mAP50-95, tail-three mean mAP50-95, and final AP75 are
   all strictly above FDR, with finite training and nonzero BPDD activity.
5. If the gate passes, start a fresh full-data seed0 BPDD Formal100 from the
   same formal initial state. Do not inherit Screen30.
6. Historical FDR may be used only for online futility checks. A final
   publication-level comparison requires a fresh FDR Formal100 under the same
   BPDD source/protocol and initial state.
7. Formal success requires independent exact-final-EMA evaluation with
   `delta mAP50-95 >= +0.0030`, `delta AP75 >= +0.0010`, and
   `delta AP50 >= -0.0010`; tail10 mAP delta must be at least `+0.0020`, tail10
   AP75 must be positive, and at least eight of the last ten epochs must have a
   positive mAP delta. No scale may fall by more than `0.005` and no class by
   more than `0.010`.

The Screen30 result is development evidence, not an unbiased final estimate.
The seed0 Formal100 result is preliminary paper evidence; a publication claim
still requires additional seeds or an external test set.

Formal early checkpoints may only stop a failed candidate, never declare early
success. E10 checks engineering health only. At E50, continue only if tail10
mAP delta is positive and tail10 AP75 delta is at least `-0.001`. At E75,
continue only if tail20 mAP is positive, tail10 mAP is at least `+0.001`, tail10
AP75 is positive, and at least seven of the last ten mAP deltas are positive.
After E90, run to E100. These checks must use a source-compatible FDR curve;
otherwise they are advisory and may not terminate the run.

If Screen30 fails, freeze BPDD-v1 as `scientific_failed`. A later amendment may
change only one of `weight`, `temperature`, or teacher construction and must
receive a new protocol identity; thresholds may not be relaxed.

## 11. Evidence, resume, and publication

Every completed epoch creates immutable metrics, exact checkpoint SHA256,
gradient/activity statistics, runtime authority, and a publication-queue row.
Training never waits on GitHub availability. Upload retries are decoupled, and
resume is allowed only from the latest fully verified checkpoint under the
same source, protocol, dataset, optimizer, and BPDD options.

The independent evaluator must load the exact remotely published final-EMA
checkpoint and verify its SHA256; a different local `last.pt` is not acceptable.
Formal completion means all 100 epochs, independent final evaluation, scale and
per-class metrics, parameter/GFLOPs/latency audit, artifact hashes, and remote
publication are complete. Training completion alone is not final completion.

## 12. Expected benefit and risk

BPDD adds zero inference parameters and is less likely than another feature
branch to erase FDR's mature gain. Its realistic target is a small incremental
improvement, not a guaranteed `+0.5 pp`. The main remaining risks are:

- future layers may not contain useful complementary information;
- self-distillation can oversmooth FDR distributions;
- auxiliary-layer improvement may not transfer to the final layer;
- a single seed can produce a false positive;
- repeated use of the same validation set can overfit research decisions.

Those risks are why the better-only gate, exact FDR ablation, final-epoch rule,
fresh Formal100, and full negative reporting are mandatory.
