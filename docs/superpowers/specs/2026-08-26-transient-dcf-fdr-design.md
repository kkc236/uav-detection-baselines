# Transient DCF-FDR Design

Date: 2026-08-26
Status: Author-approved design; implementation not started

## 1. Decision

Replace persistent Distribution-Conditioned Feedback (DCF) with one training-only
Transient DCF (T-DCF) module. T-DCF learns for the first two thirds of Formal100,
is frozen and smoothly withdrawn between two thirds and three quarters, and is
fully absent from the regression path for the final quarter and at inference.

This is the only FDR-internal mechanism added to Clean FDR. The preliminary-box
path, edge-adaptive FGL, extra DN-FDR supervision, score calibration, auxiliary
losses, and any second feedback branch remain disabled.

## 2. Evidence and Motivation

The completed pilot runs observed:

| Result | Clean | Persistent DCF | DCF - Clean |
|---|---:|---:|---:|
| Best Precision (%) | 59.476 | 59.411 | -0.065 pp |
| Best Recall (%) | 50.032 | 49.993 | -0.039 pp |
| Best AP50 (%) | 49.331 | 49.303 | -0.028 pp |
| Best mAP50-95 (%) | 29.696 | 29.661 | -0.035 pp |
| Epoch 90-100 mean Precision (%) | 58.990 | 58.162 | -0.829 pp |
| Epoch 90-100 mean Recall (%) | 49.463 | 49.767 | +0.305 pp |
| Epoch 90-100 mean mAP50-95 (%) | 29.612 | 29.478 | -0.134 pp |

The pilot therefore does not establish a positive persistent-DCF contribution.
It suggests that late feedback changes the Precision/Recall operating point and
can reduce mAP while validation GIoU remains slightly improved. The pilot is
design evidence only: the DCF run was non-exactly resumed after paper Epoch 63,
so its late deltas are not confirmatory results.

The first 65 epochs are used only to pre-register this schedule and its failure
conditions. Their metrics or optimizer evidence cannot reconstruct an Epoch-65
training state, and they will not be represented as a resumable checkpoint.

## 3. Goals and Non-goals

### Goals

- Preserve the FDR-specific information path: the preceding cumulative 4x33
  edge distribution conditions the next decoder-layer regression residual.
- Use DCF as a training scaffold without persistent late-training or inference
  dependence.
- Make Epochs 75-100 and inference exactly use the Clean regression path.
- Reuse the valid, uninterrupted Clean result only after proving that the new
  source leaves the Clean path byte- and output-equivalent.
- Produce a source-bound, non-resumed Formal100 seed-0 result with a frozen
  pass/fail rule.

### Non-goals

- No entropy gate, learned gate, quality-estimation branch, new loss, or new
  matcher/post-processing rule.
- No search over alternative switch points after the run starts.
- No claim that cosine withdrawal or training/inference decoupling is novel by
  itself.
- No claim of statistical significance from one seed.
- No use of the old non-exact DCF continuation as formal causal evidence.

## 4. Algorithm

For decoder layer `l > 1`, keep the existing detached full-distribution adapter:

1. Detach the preceding cumulative corner logits.
2. Reshape them to four 33-bin edge distributions and apply softmax.
3. Encode every edge with the one shared `Linear(33, 16) + SiLU` encoder.
4. Flatten the four edge encodings and project them through the shared
   zero-initialized `Linear(64, 256)` output.

Let this output be `F_DCF`. The regression input is:

```text
X_reg = X_clean + alpha(e, T) * F_DCF
X_clean = output + stop_gradient(previous_output)
```

For one-indexed paper epoch `e` and total epochs `T`:

```text
r = e / T

alpha(r) = 1                                      when r <= 2/3
alpha(r) = 0.5 * (1 + cos(pi * (r-2/3)/(3/4-2/3)))
                                                  when 2/3 < r < 3/4
alpha(r) = 0                                      when r >= 3/4
```

For Formal100 this is fixed as:

| Paper epochs | Adapter parameters | Feedback scale | Regression path |
|---|---|---:|---|
| 1-66 | trainable | 1.0 | persistent DCF |
| 67-74 | frozen | cosine 1 to 0 | withdrawing DCF |
| 75-100 | frozen and skipped | 0.0 | exact Clean path |

The adapter is frozen at the first epoch whose ratio exceeds `2/3`. Freezing is
mandatory: otherwise the adapter could increase its raw output while `alpha`
falls and partially defeat withdrawal. No optimizer state or momentum is reset
for the shared/base network.

When `alpha == 0.0`, the decoder must skip the adapter call rather than multiply
an evaluated adapter output by zero. This establishes zero DCF inference FLOPs
and exact Clean-path arithmetic.

## 5. Schedule State and EMA

The feedback scale is a plain Python float on the decoder, not a parameter or a
registered floating-point buffer. A floating buffer would be interpolated by
Ultralytics ModelEMA and would leave residual feedback after the nominal cutoff.

At every `on_train_epoch_start` event, the schedule controller must:

1. Compute the scale from `(trainer.epoch + 1) / trainer.epochs`.
2. Apply exactly the same float to the live model and `trainer.ema.ema`.
3. Freeze live DCF parameters once paper Epoch 67 begins.
4. Assert that both live and EMA scale values match the scheduled value.
5. Append a source-bound schedule evidence row.

The EMA's DCF weights may continue approaching the already-frozen live DCF
weights, but the EMA scale is assigned directly and is never EMA-smoothed. At
paper Epoch 75 both models must hold exact `0.0` and skip DCF execution.

## 6. Checkpoint Eligibility and Export

Metrics before paper Epoch 75 remain in `results.csv` for transparency but are
ineligible for the paper result. At the start of paper Epoch 75, reset only the
trainer's checkpoint-selection fitness so that the subsequently written
`best.pt` can come only from Epochs 75-100. Do not reset optimizer, scheduler,
base weights, EMA weights, or early-stopping history.

The authority record and final evidence must assert:

- selected best epoch is in `[75, 100]`;
- live and EMA feedback scales are exactly zero for every Epoch 75-100 row;
- the final checkpoint's decoder records scale zero;
- no DCF forward call occurs with scale zero.

For deployment, instantiate the Clean FDR configuration, copy every shared
trained tensor from the eligible T-DCF best checkpoint, reject any missing or
unexpected shared key, and discard only the declared
`decoder.distribution_feedback.*` tensors. The exported Clean-shaped checkpoint
must be validated against the scale-zero T-DCF checkpoint with exact FP32 output
comparison on a fixed batch before metrics are published.

## 7. Formal Experiment Protocol

- Dataset: the existing frozen VisDrone Formal data authority.
- Total epochs: 100.
- Seed: 0.
- Initial state: `/data/uav/protocols/fdr-d97e1eb7/initial-state.pt` with its
  recorded SHA-256.
- All optimizer, image size, batch, augmentation, worker, AMP, and loss settings
  remain identical to the completed Clean run.
- EAW, preliminary boxes, extra DN-FDR supervision, and every other optional FDR
  mechanism remain false.
- Start from Epoch 0 and run uninterrupted. If the process or host interrupts,
  discard that attempt and restart from Epoch 0; the current upstream checkpoint
  format is not accepted as an exact resume authority.
- Do not tune schedule fractions, loss weights, or selection rules after launch.

The existing uninterrupted Clean best result (`29.696%` mAP50-95 at Epoch 88)
may be reused only if preflight proves:

1. the Clean config and all frozen runtime settings are unchanged;
2. the initial-state and dataset hashes match;
3. a Clean model built from the new source has identical shared state and exact
   fixed-input outputs to the prior Clean source;
4. the source-difference manifest shows that all behavior changes are gated
   behind the DCF-present path.

If any preflight condition fails, the old Clean result is not an admissible
comparator and a matched Clean rerun is required.

## 8. Evidence and Monitoring

The run must retain:

- immutable launch authority with source, config, dataset, initial-state, and
  schedule hashes;
- `results.csv`, `args.yaml`, optimizer evidence, logs, best and last weights;
- `transient-dcf-schedule.jsonl` with paper epoch, ratio, scheduled scale, live
  scale, EMA scale, DCF frozen state, and checkpoint eligibility;
- the old persistent-DCF and Clean pilot comparison, labeled non-confirmatory;
- export key audit and scale-zero/output-equivalence report;
- process-alive, GPU, disk, non-finite loss, and time-reset monitoring.

The DCF parameters belong to the private FDR gradient-evidence group. Their
gradient norm must be observable through Epoch 66 and absent after they are
frozen. Shared/base gradient evidence must remain available through Epoch 100.

## 9. Tests and Preflight Gates

Implementation cannot launch Formal100 until all of the following pass:

1. Schedule boundary tests: Epoch 66 is 1.0, Epochs 67-74 are strictly
   decreasing and in `(0, 1)`, and Epoch 75 is exactly 0.0.
2. Adapter freeze test: parameters are trainable through Epoch 66 and frozen
   from Epoch 67 onward.
3. Live/EMA synchronization test: both receive identical schedule values.
4. EMA isolation test: schedule state is not present as an EMA-interpolated
   floating state-dict buffer.
5. Zero-path test: scale zero causes no adapter forward call and gives exact
   equality with the DCF-disabled decoder.
6. Full-path regression test: scale one reproduces the old persistent-DCF
   forward behavior.
7. Private-RNG and zero-initialization tests remain passing.
8. Tail-selection test: a larger pre-75 metric cannot remain the formal best.
9. Export test: exactly the declared DCF keys are removed; all shared keys load;
   outputs match exactly.
10. Clean-equivalence test and launch-authority dry run pass.
11. A bounded smoke forward/backward run confirms finite losses, DCF gradients
    before freeze, and base gradients after freeze.

## 10. Frozen Result Interpretation

Primary result: the best eligible Epoch 75-100 mAP50-95.

| Verdict | Frozen rule |
|---|---|
| Fail | eligible best mAP50-95 `< 29.696%` |
| Technical pass, weak paper contribution | eligible best `>= 29.696%` but gain `< 0.100 pp` |
| Strong pass | eligible best `>= 29.796%` |

Precision, Recall, AP50, F1, final-epoch metrics, and Epoch 91-100 means are
mandatory diagnostics but do not override the primary mAP gate. A Recall gain
cannot rescue a negative mAP result. No failed result may be repaired by
post-hoc checkpoint eligibility, schedule changes, or a new threshold.

## 11. Claim Boundary

The paper may claim a single FDR-internal training module: a preceding full
edge-distribution representation conditions the next-layer regression residual
during early optimization and is then frozen and withdrawn to leave an
inference-free Clean path.

The paper must not claim that auxiliary training branches, cosine schedules, or
training/inference decoupling are new. D-FINE already treats localization
distributions as transferable knowledge through GO-LSD; YOLOv7 uses auxiliary
training heads; RepVGG explicitly decouples training and inference structures.
The defensible distinction is the specific full-distribution-to-next-residual
information path and its evidence-backed transient use inside this FDR variant.

Primary references:

- D-FINE, ICLR 2025: https://openreview.net/pdf?id=MFZjrTFE7h
- YOLOv7, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html
- RepVGG, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html

## 12. Adversarial Audit Verdict

Conditional pass becomes implementation-authorized only with all four controls:

1. freeze DCF after two thirds;
2. synchronize schedule state to live and EMA models without EMA smoothing;
3. select the formal checkpoint only after complete shutdown;
4. export and verify the true Clean inference path.

Removing any control invalidates the approved design. The user approved these
controls and the Epoch-0 restart requirement on 2026-08-26.
