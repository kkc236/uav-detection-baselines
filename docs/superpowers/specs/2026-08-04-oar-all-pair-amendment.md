# OAR All-Pair Amendment

Date: 2026-08-04

Status: approved under the user's delegated-autonomy instruction after the frozen sparse
D0 prerequisite failed

Amends: `2026-08-04-objective-aligned-reranker-design.md`

## 1. Frozen sparse-D0 result

The existing 129-image internal-development cache was verified shard-by-shard before a
read-only D0 pre-audit. It exactly reproduced C0 mAP `0.28628865801344866` and AP75
`0.2923640747621435`.

The oracle decomposition was:

| Score family | mAP | AP75 |
|---|---:|---:|
| C0 stock | 0.286288658013 | 0.292364074762 |
| class-presence oracle | 0.294608200682 | 0.300115294639 |
| class-agnostic Query-IoU oracle | 0.324185693386 | 0.344708985960 |
| full same-class oracle | 0.409733588907 | 0.413238330995 |

Restricting same-class oracle modifications to stock Top-K Queries per class produced:

| K per class | mAP | AP75 | full-gain recovery |
|---:|---:|---:|---:|
| 20 | 0.304549967436 | 0.328443354487 | 0.147931 |
| 40 | 0.335016751522 | 0.361708134040 | 0.394735 |
| 60 | 0.358000690130 | 0.381850344194 | 0.580923 |
| 100 | 0.385568106152 | 0.401433500185 | 0.804241 |

The frozen D0 prerequisite required the smallest K in `{20,40,60,100}` recovering at
least 90% of the full same-class-oracle mAP gain. No K passed. The sparse Top-K OAR design
is therefore `scientific_failed`. Its grid and threshold are not extended or weakened
after observing this result.

## 2. Interpretation

The failed prerequisite does not reject objective-aligned reranking. It rejects sparse
candidate truncation as the first deployable approximation. The evidence also shows:

- image-level class presence accounts for about `+0.00832` mAP;
- class-agnostic Query localization quality accounts for about `+0.03790` mAP;
- the full class-conditional signal reaches about `+0.12344` mAP;
- restricting updates to 1,000 of 3,000 Query-by-class pairs still loses about 19.58% of
  the available oracle gain.

The next candidate must therefore preserve all 3,000 class-conditional outputs without
using 3,000-token global self-attention.

## 3. OAR-R2: all-pair objective-only ranker

OAR-R2 keeps the 276-value Query-by-class representation and the same 17,793-parameter
MLP:

```text
Linear(276,64) -> SiLU -> Linear(64,1)
```

It evaluates all `[300,10]` pairs. There is no Top-K pool and no unmodifiable candidate.
The last layer is zero-initialized. For raw output `a[q,c]`:

```text
r[q,c] = 2 * tanh(a[q,c] / 2)
score[q,c] = sigmoid(stock_logit[q,c] + r[q,c])
```

Epoch-zero output is exactly stock. Inputs remain detached, detector losses remain
stock, and OAR gradients update only private OAR parameters.

The teacher utility and deterministic Top-300 boundary RankNet definition remain exactly
as frozen in the parent design. Removing the pool mask is the only scientific change
from sparse OAR-R.

## 4. OAR-QS2: 300-token class-conditional set ranker

OAR-QS2 is authorized only when OAR-R2 improves both mAP and AP75 over C0 but misses the
final `+0.0050` mAP Gate.

It does not create 3,000 tokens. Each of the 300 Query tokens contains:

```text
final decoder hidden                              256
stock box geometry: cx,cy,w,h,logw,logh,area,ratio  8
all ten stock class logits                         10
mean Bernoulli class entropy                        1
                                                   ---
total                                              275
```

The architecture is:

```text
Linear(275,64)
-> one TransformerEncoder layer, 4 heads, FFN width 128
-> Linear(64,10)
-> bounded residual [300,10]
```

The final `Linear(64,10)` is zero-initialized. The output is class-conditional while
self-attention cost scales with 300 Query tokens rather than 3,000 Query-by-class tokens.
There is no FDR, feature-map input, NMS, boundary evidence, or trajectory signal.

OAR-QS2 uses the same all-pair teacher, boundary pairs, optimization, data, checkpoint
selection, and stock-preserving score formula as OAR-R2. Set interaction is the only
added scientific variable.

## 5. Gates and execution

The internal Gate is unchanged:

```text
candidate.map  - C0.map  >= 0.0050
candidate.ap75 - C0.ap75 >  0
```

OAR-QS2 must also be strictly better than OAR-R2 in both metrics. Official validation
remains a one-shot release after the first passing immutable internal decision and
requires strict positive mAP and AP75 deltas over C0.

If OAR-R2 has a non-positive mAP or AP75 delta, OAR-QS2 is not authorized. If both
all-pair arms fail, objective-aligned reranking is frozen and the next branch must add a
new information source under a separate design rather than change these thresholds.

An official-positive arm proceeds to a separately reviewed last-decoder integration,
paired fixed-subset seed0 30-epoch screen, Gate2, and fresh full-data seed0 100 epochs.
