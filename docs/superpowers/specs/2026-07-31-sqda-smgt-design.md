# SQDA-SMGT Design

## Objective

Repair the failed G1 geometry-trust screen with a trainable network-layer module that preserves the stock RT-DETR-L, inherited SQDA-SGC tensors, semantic residual, decoder outputs, and all inference/post-processing settings. A candidate may advance only when the fixed-protocol evaluator reports no decline in every registered gate criterion.

## Evidence motivating the change

The fully evaluated G1 checkpoints did not pass. At the frozen confidence threshold, the error change was restricted to the small-object bin. The learned epoch-2 gate was inversely correlated with log-width/log-height and assigned mean geometry budgets of 0.98619, 0.98468, and 0.98319 to small, medium, and large reference boxes respectively. It therefore gives the most geometry weight to the population in which geometry-induced false positives were isolated. The latest snapshot kept threshold TP/FP unchanged but reduced precision at its own best-F1 point by 0.001168.

The original `best.pt` is not an admissible proxy for G1 selection in this run: it stores the initial module state, whereas `epoch1.pt`, `epoch2.pt`, and `last.pt` contain the actual updates. Candidate selection must enumerate evaluated saved snapshots rather than trusting a filename.

## Module: Scale-Monotone Geometry Trust (SMGT)

SMGT replaces only the new `geometry_trust` submodule. It receives the existing five detached, per-query diagnostic quantities but factorizes them:

1. An agreement MLP consumes only query--geometry cosine, geometry/query norm ratio, and semantic--geometry cosine.
2. A deterministic normalized reference-scale coordinate is derived from log area. A non-negative learned coefficient adds to the geometry-trust logit as scale increases.
3. The budget remains `0.80 + 0.20 * sigmoid(logit)`. With identical agreement evidence, a larger reference box can never receive less geometry trust than a smaller one. Semantic residuals remain exactly un-gated.

The scale direction is an inductive constraint, not a copied attention block. It is informed by scale-aware feature specialization and spatially conditioned DETR attention, but it does not reproduce their architectures, losses, or detector heads.

## Initialisation and training scope

The agreement MLP starts at the previous nominal geometry trust (0.98 budget). The non-negative scale coefficient starts at a small positive value, which makes small boxes retain the nominal budget while giving larger boxes only a bounded additional trust. Only SMGT parameters train; stock and inherited SQDA-SGC tensors must remain byte-identical. Existing fixed AMP scale, batch size, image size, query count, max detections, NMS setting, seed, and augmentation protocol are unchanged.

## Evaluation and advancement gate

Every completed G1/G2 epoch checkpoint (`epoch*.pt` and `last.pt`) is independently evaluated using the exact retained-G2 full branch, COCO AP/APsmall, class-aware fixed-threshold errors, PR/F1 curve, and gate diagnostics. `best.pt` is included only after its payload is fingerprinted; it cannot hide a newer trainable state. A run advances only if at least one actual trained snapshot passes every criterion. G1 remains a three-epoch screen; a passing SMGT G1 is followed by an independent ten-epoch G2. Only a passing G2 authorizes one evidence-driven refinement and a separate 100-epoch formal run.

## Safety and rejection conditions

Reject a candidate if any stock or inherited tensor changes, gate values become non-finite, the scale monotonicity invariant fails, the gate saturates at its lower bound, or any registered precision/recall/AP criterion declines. Do not reinterpret a small decline as a pass.
