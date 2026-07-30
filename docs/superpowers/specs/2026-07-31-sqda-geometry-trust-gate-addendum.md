# SQDA Geometry-Trust Gate Frozen Addendum

Date: 2026-07-31  
Status: user-directed replacement for the double-budget first round; not yet implemented or trained  
Supersedes: the two-budget mechanism in sections 3.1, 4, 5, 6, and 7 of 2026-07-31-sqda-abdr-design.md

## 1. Evidence boundary

The G2 candidate differs from its matched baseline by Precision -0.00053845, Recall +0.00014790, mAP50 +0.00005305, mAP50-95 +0.00003102, AP-medium -0.00006061, and AP-large -0.00061589. One seed and one retained checkpoint do not establish a stable Precision regression, a scale regression, or a geometry-residual cause.

Reported Ultralytics Precision is an operating-point metric, whereas mAP integrates the PR curve. Therefore the observed small disagreement is a diagnostic trigger, not a reason to pre-commit a semantic gate.

## 2. Frozen first-round network change

Retain the current learned fusion matrix and split it by its input columns:

~~~
W_f = [W_s | W_g]
r_s = W_s x_sem + b_f
r_g = W_g x_geo
~~~

where x_sem and x_geo are the existing context-modulated semantic and two-way-group-gated geometry inputs. The only new trainable layer is:

~~~
u_g = MLP_g[log(w), log(h), cos(q, r_g),
            ||r_g|| / (||q|| + eps), cos(r_s, r_g)]
a_g = 0.80 + 0.20 sigmoid(u_g)
f_raw = r_s + a_g r_g
~~~

a_s is exactly 1. The existing SQDA-SGC adapter, stock RT-DETR, original group gate, query count/order, Top-300 selection, decoder, loss, NMS, resolution, data, optimizer and post-processing remain unchanged.

The gate final bias is logit(0.90), so a_g starts at 0.98. All old SQDA adapter tensors are loaded from the same retained G2 checkpoint and frozen. Only MLP_g is trainable in the first geometry-gate run.

## 3. Fusion and bound compatibility

At a_g=1, f_raw is algebraically the old fusion output. The implementation must retain the existing fusion layer and load its state directly; it must not replace fusion by two newly initialized projectors.

The current code applies soft RMS saturation:

~~~
f = f_raw / sqrt(1 + mean(f_raw**2))
~~~

It is not a strict hard RMS clip. Replacing it with a hard clip would make a_g=1 no longer reproduce the current SQDA output, so hard clipping is excluded from this first controlled experiment. The first run logs pre-saturation and post-saturation RMS to test whether saturation materially reduces gate sensitivity. A hard clip is a separate future ablation, not a silent change.

## 4. Required zero-training diagnosis

Before any new checkpoint training, run the retained G2 checkpoint on the fixed validation set in four read-only modes:

1. Full: semantic plus geometry residual.
2. Semantic-only: geometry residual multiplier set to zero.
3. Geometry-only: semantic residual multiplier set to zero.
4. Identity: both residual multipliers zero.

These overrides are diagnostic-only and never appear in the official train/validation/predict configuration.

For every Full/Semantic-only/Geometry-only/Identity run, save the fixed 300-prediction output and evaluate:

- complete PR, Precision-confidence, Recall-confidence and F1-confidence curves;
- Precision and Recall at the baseline's frozen confidence threshold;
- COCO AP, AP50, AP75, AP-small, AP-medium, AP-large;
- class-aware fixed-threshold TP/FP/FN, confidence and IoU summaries by COCO size bin;
- TIDE error categories if the installed tidecv package can run against the same predictions.

The geometry gate is permitted only if the diagnostics consistently show that geometry suppression reduces localization or combined errors without a material AP-small/Recall loss. A single max-F1 Precision movement alone is insufficient evidence.

## 5. Controlled G1 comparison

Use the same retained G2 adapter checkpoint and fixed data order.

- Control: no geometry gate; all inherited SQDA parameters frozen; read-only G2 result serves as the parameter-identical control.
- Geometry-gate: load exactly the same inherited adapter tensors, freeze them, and train only MLP_g for three epochs.
- Both paths use seed 0, deterministic mode, imgsz 640, batch 8, AMP scale 128, max_det 300 and NMS false.

The gate must report its mean, min, max and fraction within 0.005 of the 0.80 lower bound. Large-scale saturation at the lower bound is a failure of the intended gentle correction.

## 6. First-round decision rule

The gate must satisfy all of the following against the fixed control:

- no NaN/Inf and stock plus inherited-SQDA tensor audit passes;
- Precision at maximum-F1 does not decrease;
- Precision at the frozen baseline confidence threshold does not decrease;
- mAP50-95 and AP-small do not decrease by more than 0.0002;
- no large lower-bound saturation pattern;
- PR/TIDE evidence must not indicate increased misses or duplicate errors.

This is a directional screening rule. It is not a claim that the gate is guaranteed to improve every seed or that semantics and geometry exactly equal classification and localization heads.

## 7. Paper wording

The paper may describe a trainable geometry-residual trust gate that conditionally attenuates the geometry contribution of an inherited semantic-geometry fusion. It must not claim that the [0.80, 1] range is theoretically derived from literature, that the gate corrects a confirmed Precision defect, or that the result generalizes until multi-seed results support it.

