# EBC-QP v1.0 Frozen Design

## 1. Status and scope

- Chinese name: 固定查询预算下的淘汰感知边界校准查询保护机制
- English name: Eviction-Aware Boundary-Calibrated Query Preservation
- Status: frozen for D1/D2 screening
- Date: 2026-07-22
- Base: `origin/codex/matched-baseline`
- Branch: `codex/ebc-qp`
- Baseline: Ultralytics RT-DETR-L, VisDrone train/val, scratch, 640, batch 8

This is the only EBC-QP v1.0 design authority. It replaces earlier BTD-SE,
P2-fusion, P2-residual, NWD, and rank-guided drafts.

EBC-QP targets tiny GTs not represented in the stock Top-300 decoder
initialization set. It does not claim first use of P2, Top-K queries, fixed query
counts, dense positives, or high-resolution query initialization. Its claim is:

```text
stock coverage awareness
+ isolated P2 supplemental queries
+ target-selective Top-300 boundary calibration
+ fixed-budget global competition without a reserved P2 quota
```

## 2. Baseline source lock

Target Ultralytics version: `8.4.90`.

| Local source | SHA256 |
| --- | --- |
| `ultralytics/nn/modules/head.py` | `5701116D86881827AC9E1E7462DFAA44C33937BD68E23324763459685729E06F` |
| `ultralytics/nn/tasks.py` | `B00935C1851BB9CEA240985704C12E654E68B369F6C59DE20E45FA295CB79B92` |
| `ultralytics/cfg/models/rt-detr/rtdetr-l.yaml` | `85716F626769CB5DDF00D59FCF6CAFB5814AAD196328100BDC7C93306F650E83` |

Every run records the package version and source hashes. A mismatch requires a
new stock-path audit before comparison.

Audited stock behavior:

1. P3/P4/P5 tokens are scored by maximum raw class logit.
2. The highest 300 tokens initialize decoder content and reference boxes.
3. Query content and reference boxes are detached before decoder training.
4. Original stock encoder boxes/scores are prepended as an auxiliary prediction
   layer for the RT-DETR criterion.

## 3. Frozen data flow

```text
C2.detach() --1x1 Conv+BN--+
                            +--SiLU--P2 encoder transform--P2 score/box
stock P3 projected.detach()-+                 |
                                              +--P2 Top-50

stock P3/P4/P5 --> stock Top-300 ------------+
                                              |
                         global Top-300 <-----+
                                 |
                         unchanged decoder

encoder auxiliary criterion input: original stock Top-300 only
```

### 3.1 Isolated P2 side branch

`C2` is the layer-1 stride-4, 128-channel backbone output. `F3_stock` is the
actual stride-8, 256-channel feature after the stock decoder input projection.

```text
F2 = SiLU(
       BN(Conv1x1(stop_gradient(C2)))
       + nearest_upsample_2x(stop_gradient(F3_stock))
     )
```

No extra P3 projection is added. P2 then uses the current stock encoder-output
transform and encoder-score head with parameter-level stop-gradient:

```text
E2 = stock_enc_output(F2; stop_gradient(theta_enc_output))
Z2 = stock_enc_score(E2; stop_gradient(theta_enc_score))
```

Do not detach `E2` or `Z2`: gradients must pass to the P2 adapter while stock
head parameters remain unaffected by P2 losses. The stock path still updates
those parameters normally.

The P2 regression head is an independent copy of the stock encoder regression
head. Its base anchor size is `0.025`. P2 anchors reuse stock grid generation and
validity rules. Invalid border anchors are excluded from assignment and masked to
negative infinity before Top-50. Normalized anchor centers are retained separately
from inverse-sigmoid reference boxes.

P2 auxiliary/EBC gradients may update only the P2 Conv, BN, and independent box
head. They do not update C2, P3, stock input projection, stock encoder transform,
or stock score/box heads through the side branch.

## 4. Four mandatory definitions

### 4.1 Stock-uncovered GT

For augmented 640-pixel training boxes:

```text
r_g = sqrt(width_g * height_g)
tiny: r_g <= 16 pixels
report groups: r_g < 8 and 8 <= r_g <= 16
```

Build a category-independent bipartite graph between valid stock Top-300 anchor
centers and all tiny GTs. An edge exists when the anchor center is inside the GT.

```text
cost(j, g) = distance(anchor_center_j, gt_center_g)
             / (sqrt(width_g * height_g) + 1e-6)
```

Matching objectives, in order:

1. maximize matched GT count;
2. minimize total normalized distance;
3. break ties by GT index, then flattened token index.

Each query and GT match at most once. Define `u_g=0` for matched GTs and `u_g=1`
otherwise. `u_g` is detached. Regressed IoU is diagnostic and never defines
coverage.

### 4.2 Parameter stop-gradient

P2 calls use detached stock parameters, not detached outputs. Implement with
functional operations or a detached-parameter functional call, for example:

```python
p2_logits = F.linear(p2_embed, stock_weight.detach(), stock_bias.detach())
```

Required P2-loss-only backward result:

```text
P2 adapter grad != 0
P2 box-head grad != 0 when positives exist
stock encoder-transform grad == 0
stock score-head grad == 0
C2/P3 side-input grad == 0
```

A separate stock-loss backward must still update normal stock parameters.

### 4.3 Deterministic P2 local assignment

For every tiny GT, collect valid P2 cells in its clipped 3x3 center neighborhood.
Take the union and build GT-to-cell edges only within each GT's neighborhood.

```text
cost(g, j) = distance(p2_center_j, gt_center_g)
             / (sqrt(width_g * height_g) + 1e-6)
```

Use the same maximum-cardinality, minimum-distance, deterministic tie-breaking
objectives as stock coverage. Each GT and cell match at most once. Scores and
predicted IoU are never used. The rule does not change by epoch. Unassigned GTs
contribute neither P2 positive loss nor EBC loss.

```text
LocalAssignRate = assigned_tiny / max(tiny_count, 1)
```

v1.0 never changes the neighborhood to 5x5 inside a run.

### 4.4 Sparse P2 VFL

Classification positions are:

```text
C2_cls = unique(P2_Top50 union assigned_positive_positions)
```

For assigned positive `j_g` of class `y_g`:

```text
target[j_g, y_g] = stop_gradient(IoU(p2_box_j_g, gt_g))
```

Boxes use the stock normalized `xywh` convention. Other classes are zero. A P2
Top-50 position is negative only when unassigned and its anchor center is outside
every labeled GT. Unassigned positions inside any labeled GT are excluded. The
converted YOLO dataset has no retained VisDrone ignore regions, so v1.0 does not
implement ignore filtering.

L1/GIoU apply only to assigned positives. Per image, then batch mean:

```text
L_P2_image = (L_VFL + 5*L_L1 + 2*L_GIoU) / max(N_positive, 1)
L_P2 = mean_b(L_P2_image)
```

No-positive images may retain Top-50 negative VFL; box losses are differentiable
zero.

## 5. Boundary calibration

For image `b`:

```text
tau_b = stop_gradient(stock_rank_300_max_class_raw_logit)
```

For locally assigned, stock-uncovered tiny GTs only:

```text
L_EBC = sum_g u_g * relu(tau_b - Z2[j_g, y_g])
        / max(sum_g u_g, 1)
```

Frozen semantics:

- the positive uses its correct-class raw logit;
- `tau_b` uses the stock maximum-class raw-logit score;
- margin is zero;
- assignment, `u_g`, and `tau_b` are detached;
- crossing the boundary makes the term zero;
- no eligible target returns differentiable zero;
- no hard-negative or rank-251-to-350 loss exists.

## 6. Fixed-budget competition and loss isolation

P2 and stock use the same `max_class_raw_logit` ranking score:

```text
Q_decoder = TopK300(Q_stock_300 union Q_P2_50)
```

There is no explicit `P2 score > tau` filter. Equal scores prefer stock; remaining
ties use flattened source index. Merge the complete score, logits, query feature,
inverse-sigmoid reference box, source level, and source index tuple. Final ordinary
query count is exactly 300 and P2 count is 0-50. P2 uses stock decoder detach rules;
denoising order is unchanged.

The encoder auxiliary criterion always receives the original stock Top-300 boxes
and scores, never the mixed set. Decoder architecture, Hungarian matching, final
loss, and post-processing remain unchanged.

Global competition is an implementation consequence, not an independent claimed
module. Explicit conditional replacement and separate S2/S3 experiments are
forbidden because they are mathematically redundant.

## 7. Frozen schedule and parameters

Epochs 1-3:

- stock detector trains normally;
- P2 sparse loss trains the isolated side branch;
- P2 does not enter the decoder;
- EBC is zero.

Epoch 4 onward:

- P2 sparse loss continues;
- EBC activates;
- stock/P2 global competition activates;
- stock encoder auxiliary outputs remain isolated.

```text
L_total = L_stock + 0.25*L_P2 + 0.05*L_EBC
```

| Item | v1.0 |
| --- | ---: |
| Query budget | 300 |
| P2 candidate limit | 50 |
| Warm-up | 3 epochs |
| Tiny threshold | `r <= 16` |
| P2 anchor | `0.025` |
| EBC margin | `0` |
| `lambda_P2` | `0.25` |
| `lambda_EBC` | `0.05` |
| Local assignment | 3x3 |
| P2 NMS/local peaks | disabled |
| Hard-negative ranking | disabled |
| Reserved P2 quota | disabled |
| Decoder/Hungarian changes | none |

## 8. Minimal experiment chain

### D0: stock zero-training diagnosis

Run archived matched baseline seed-0 on full validation. Use anchor centers and
classification ranks for all stock tokens; do not use boxes below Top-300 as
primary evidence. Record all-token center coverage, Top-300 coverage, relevant
token rank, and final decoder failure.

### D1: 3-epoch P2 health probe

- start from archived matched baseline seed-0 best;
- use a fixed, hashed 10% training subset;
- freeze stock parameters and stock BN running statistics in eval mode;
- train only P2 Conv/BN/box head;
- do not inject P2 or compute EBC;
- D1 weights never initialize D2.

### D2: 10-epoch full screen

Run scratch v1.0 and scratch stock control on the same fixed 10% subset with
identical common initialization, seed, data order, and augmentation order. Warm-up
for three epochs; activate EBC/global competition at epoch 4.

Proceed only when P2 coverage, tiny recall, and effective P2 entries improve;
displaced effective stock count remains lower than gained coverage; P2 count does
not saturate at 50; final-three-epoch mean mAP50-95 is not below control; and no
sustained late decline appears.

### D3: no-EBC ablation

Only after D2 shows a positive trend, rerun the same scratch 10% protocol with
`lambda_EBC=0`.

| ID | P2 queries | EBC |
| --- | ---: | ---: |
| A0 | no | no |
| A1 | yes | no |
| A2 | yes | yes |

Only a frozen D2 configuration that passes may start 100-epoch seed-0 scratch
training. Additional baseline/EBC seeds follow only after positive seed-0 evidence.

## 9. Persistent logs

Keep only:

- AP-tiny and tiny Recall for resized `r<=16` GTs;
- stock Top-300 center coverage;
- LocalAssignRate;
- P2 entry count;
- effective P2 entry rate;
- displaced effective stock count;
- boundary-gap mean and positive ratio;
- `L_P2` and `L_EBC`;
- overall Precision, Recall, mAP50, and mAP50-95.

An entered P2 candidate is effective when its anchor center covers a tiny GT not
covered by stock Top-300. A displaced stock candidate is effective when it was in
the frozen stock-to-GT center matching. Both use the same detached one-to-one
coverage definition.

Do not persist full P2 maps, large percentile tables, or repeated per-stage
complexity traces. Gradient and finite-value diagnostics are test outputs only.

## 10. Required tests and abort rules

Tests must cover:

1. **Disabled equivalence:** exact Top-300 indices and matching stock outputs;
   FP32 `rtol=1e-5, atol=1e-6`, AMP `rtol=1e-4, atol=1e-5`.
2. **Gradient isolation:** parameter-level detach works and the stock path still
   trains normally in a separate stock-loss backward.
3. **Matching determinism:** maximum cardinality, minimum distance, unique cells,
   deterministic ties, and correct unassigned handling.
4. **Loss edge cases:** finite differentiable zeros, detached VFL IoU target,
   non-tiny GT locations excluded from negatives, EBC eligibility correct.
5. **Query integrity:** exactly 300 queries, P2 count 0-50, stock tie priority,
   complete tuple merge, stock-only encoder auxiliary outputs, unchanged denoising.
6. **Training state:** optimizer, EMA, checkpoints, resume, best/last, and warm-up
   state include EBC-QP correctly.

Abort instead of editing an active run when:

- NaN/Inf or disabled-path mismatch occurs;
- P2 adapter has no gradient;
- P2 auxiliary gradient exceeds stock gradient by over 10x in the first 200 steps;
- P2 scores completely dominate stock or entry count saturates at 50;
- matching is nondeterministic or encoder auxiliary outputs contain P2.

Any change creates v1.1 and starts from a clean checkpoint. v1.0/v1.1 trajectories
must not be spliced.

## 11. Reporting boundary

Measure parameters, GFLOPs, memory, latency, FPS, and epoch time after a method
passes. Before measurement, claim only that decoder architecture and its 300-query
budget are unchanged. P2 full-map scoring still costs computation; EBC-QP is not
zero-cost or fully sparse feature computation.

Frozen definition:

> EBC-QP uses an isolated stride-4 semantic side branch to generate supplemental
> queries for tiny objects not covered by stock Top-300, calibrates their correct-
> class scores against the actual per-image Top-300 boundary, and lets them compete
> for the unchanged 300-query decoder budget without a reserved P2 quota.
