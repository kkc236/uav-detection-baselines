# IOQC-SA Standalone Module And Resilient Server Training Design

## Goal

Add Instance-Ownership-Aware Query Competition and Scale Alignment (IOQC-SA) to the unmodified RT-DETR-L baseline as a training-only module. The experiment must remain isolated from BTD-SE, preserve the baseline inference graph, keep the auxiliary objective numerically stable under mixed precision, and run to completion on a newly rented RTX 30/40/50-series server with automatic recovery and batch adaptation.

## Experimental Boundary

- Architecture: stock Ultralytics `rtdetr-l.yaml` plus a training-only sampling probe.
- Dataset: `VisDrone.yaml` using the same train and validation splits as the existing baseline.
- Initialization: scratch training with `pretrained=False`.
- Schedule: 100 epochs, image size 640, seed 0, deterministic mode, NMS disabled, and the original RT-DETR detection criterion unchanged.
- Isolation: IOQC-SA code must not import BTD-SE modules, use the BTD-SE YAML, or add background/saliency losses.
- Inference: the sampling probe and IOQC-SA loss are absent; exported predictions are the stock RT-DETR predictions.

The primary comparison is the existing RT-DETR-L baseline versus RT-DETR-L plus IOQC-SA. Every intentional difference must be present in the saved run arguments and diagnostics.

## Architecture

### Training-Only P3 Sampling Probe

The final decoder layer's deformable cross-attention receives the query embedding, reference box, flattened multiscale values, and feature shapes. A repository-owned forward pre-hook observes these inputs and applies the existing layer's `sampling_offsets` and `attention_weights` projections a second time inside an autocast-disabled FP32 region.

This observer approach has four properties:

1. it does not edit Ultralytics `site-packages`;
2. it does not replace or alter the cross-attention output used by RT-DETR;
3. its auxiliary computation remains differentiable with respect to the final decoder query and cross-attention projection weights;
4. removing the hook restores the exact baseline inference path.

For the P3 level, the probe stores the regular-query sampling center, extent, and original P3 attention mass. Denoising-query statistics remain in the captured tensor until the model loss receives `dn_meta`; the loss then removes the denoising prefix using `dn_num_split` and keeps only ordinary matching queries.

### Sampling Statistics

For query `i`, attention head `h`, and P3 point `k`, let `p_i,h,k` be the normalized sampling location and `a_i,h,k` the attention weight after the original softmax over all levels and points. Define P3 mass

```
m_i = sum(h,k, a_i,h,k).
```

Queries with non-finite values or `m_i <= 1e-6` are invalid for IOQC-SA. Valid P3 weights are renormalized within P3:

```
a_tilde_i,h,k = a_i,h,k / max(m_i, 1e-6).
```

The center and element-wise standard deviation are

```
mu_i = sum(h,k, a_tilde_i,h,k * p_i,h,k)
s_i  = sqrt(sum(h,k, a_tilde_i,h,k * (p_i,h,k - mu_i)^2) + 1e-6).
```

All values in this section are computed and retained as FP32 tensors.

## Instance Ownership

### Dense Ground Truth

For ground-truth object `g`, normalized center `c_g`, width `w_g`, and height `h_g`, define

```
rho_g = min(k != g) ||c_g - c_k||_2
        / (sqrt(w_g * h_g) + sqrt(w_k * h_k) + 1e-6).
d_g = 1[rho_g < r_d].
```

Images with fewer than two valid ground-truth objects have no dense targets. The default density threshold is `r_d = 1.0`.

### Owner Query

The last decoder layer's ordinary predictions are cast to FP32 and passed to the same Ultralytics Hungarian matcher and cost gains used by RT-DETR. The matched query `i_g` is the owner of ground truth `g`. Recomputing the assignment is deterministic because the matcher detaches its inputs and the model/version are pinned.

### Top-1 Duplicate Query

For unmatched query `j` and ground truth `g`, define the detached duplicate quality

```
Q_j,g = stop_gradient(sigmoid(logit_j[y_g]) * IoU(box_j, box_g)).
```

Each unmatched query is assigned to its single highest-quality ground truth. For each ground truth, only the highest-quality assigned query is retained. It is a valid duplicate when `Q_j,g > theta_dup`; the default is `theta_dup = 0.10`. Invalid P3-statistic queries are excluded before Top-1 selection.

This mechanism identifies same-instance ownership conflicts. It does not claim to redirect a duplicate query to an undetected neighboring object.

## Stable Auxiliary Objective

### Target Scale

The uniform-box reference extent is

```
t_g = (w_g / sqrt(12), h_g / sqrt(12)).
```

To prevent small targets from producing unstable division, the element-wise lower bound is one P3 cell:

```
t_bar_g = max(t_g, (1 / W3, 1 / H3)).
```

### One-Way Competition

For owner `i_g` and duplicate `j_g`, define

```
D_g = mean(concat(
    abs((mu_jg - stop_gradient(mu_ig)) / t_bar_g),
    abs((s_jg  - stop_gradient(s_ig))  / t_bar_g)
)).
L_comp = mean_g relu(1 - D_g).
```

Only dense ground truths with a valid owner and duplicate contribute. The owner statistics are detached, so the competition term updates only the duplicate side. An empty set returns a differentiable FP32 zero.

### Owner Center-Scale Alignment

Raw normalized-coordinate L1 gives weak gradients to small objects. IOQC-SA therefore uses target-scale-normalized Smooth-L1 with fixed beta 1:

```
z_center = (mu_ig - c_g) / t_bar_g
z_extent = (s_ig - t_bar_g) / t_bar_g
L_align = mean_g mean(smooth_l1(concat(z_center, z_extent), beta=1)).
```

Only dense ground truths with a valid owner statistic contribute. This change adds no tunable internal coefficient and limits the effect of large normalized residuals.

### Schedule And Weights

The default auxiliary weights are `lambda_comp = 0.05` and `lambda_align = 0.05`. Let `r = epoch / total_epochs`. The fixed activation factor is

```
phi(r) = 0                         when r < 0.10
         (r - 0.10) / 0.05        when 0.10 <= r < 0.15
         1                         when r >= 0.15.
```

The standalone objective is

```
L = L_det + phi(r) * (lambda_comp * L_comp + lambda_align * L_align).
```

The five-percent ramp replaces an abrupt switch and is fixed rather than exposed as another method hyperparameter.

## Numerical Safety

- Sampling projections, softmax, locations, IoU, Hungarian inputs, density calculations, divisions, Smooth-L1, reductions, and auxiliary loss summation run with autocast disabled and FP32 inputs.
- Widths, heights, denominators, and square-root inputs receive explicit positive lower bounds.
- Invalid individual sampling statistics are masked before owner/duplicate selection.
- Empty valid sets return graph-connected FP32 zero values.
- `nan_to_num` is prohibited in the loss path because it would conceal an invalid computation.
- The final detection loss, auxiliary losses, and total loss are checked with `torch.isfinite` before return. A non-finite aggregate raises a `NONFINITE_LOSS` error before optimizer update.
- Diagnostics record active weight, dense-target count, valid duplicate count, P3 mass range, both raw auxiliary losses, weighted auxiliary contribution, batch, AMP state, and peak CUDA memory.

The main RT-DETR forward remains AMP-capable. IOQC-SA itself never executes in FP16 or BF16.

## GPU-Adaptive Server Supervisor

### Detection And Batch Ladder

At startup, PyTorch and `nvidia-smi` provide GPU name, total memory, free memory, driver, and CUDA versions. The supported default ladders are:

| Detected VRAM | Batch levels | Initial level |
| --- | --- | --- |
| at least 20 and below 27 GiB | 2, 4, 6, 8 | 6 |
| at least 27 and below 36 GiB | 4, 6, 8, 10, 12 | 8 |
| at least 36 and below 55 GiB | 6, 8, 12, 16, 20 | 12 |
| at least 55 GiB | 8, 12, 16, 20, 24, 28 | 16 |

If startup free memory is below 85% of total memory, the initial level is reduced proportionally. The run uses a fixed nominal batch size `nbs=64` so Ultralytics gradient accumulation compensates for physical-batch changes as closely as its baseline trainer permits.

After each completed epoch:

- peak allocation at or above 94% of total VRAM causes one-level demotion;
- peak allocation below 82% for three consecutive epochs permits one-level promotion;
- any CUDA OOM causes immediate one-level demotion and a five-epoch promotion cooldown;
- the chosen level and reason are written atomically before restart.

### Recovery State Machine

Training saves `last.pt` and an independent epoch checkpoint after every completed epoch. Before every resume, checkpoint deserialization verifies the epoch, optimizer, EMA, and training state. A damaged `last.pt` falls back to the newest valid epoch checkpoint.

Supervisor exit handling is:

- planned batch transition: resume from the just-completed checkpoint with the new batch;
- CUDA OOM: lower batch and resume from the latest completed checkpoint;
- `NONFINITE_LOSS`: disable AMP permanently for this run, lower one batch level, and resume from the latest completed checkpoint;
- transient process failure: retry with capped delay while preserving current state;
- repeated unexplained failures: stop after three consecutive attempts and retain all local and remote artifacts;
- low disk space: stop safely before launch when less than 20 GiB remains.

The state, status, and manifest JSON files use write-to-temporary plus atomic replace. A process lock prevents duplicate supervisors and stale locks are recoverable after verifying that the recorded PID no longer exists.

### Remote Protection

- Source, tests, configuration, and operator documentation are pushed to `kkc236/uav-detection-baselines` on branch `codex/ioqc-sa`.
- The newest three validated resumable checkpoints are rolling GitHub Release assets under tag `ioqc-sa-rtdetr-l-live`.
- Metrics, arguments, IOQC-SA diagnostics, batch history, and SHA256 manifests are committed to the isolated `training-results` branch.
- Remote upload is verified before old remote assets are removed.
- GitHub upload failure never terminates local training; the watcher retries independently.
- Automatic server shutdown is disabled by default.

## Test Strategy

### Pure Loss Tests

Tests use synthetic tensors to prove:

- FP16 inputs produce finite FP32 auxiliary losses and finite gradients;
- identical owner/duplicate sampling yields maximum competition penalty;
- a one-target-scale difference reaches zero margin penalty;
- owner statistics receive no competition gradient;
- center and extent alignment respond in the expected direction;
- no GT, one GT, no dense GT, no valid duplicate, and zero P3 mass return exact differentiable zero;
- tiny boxes use the P3-cell scale floor;
- denoising query prefixes never enter owner or duplicate selection;
- any non-finite final loss raises the expected marker.

### Integration Tests

- Construct stock RT-DETR-L plus IOQC-SA and verify no BTD-SE modules exist.
- Run a real training forward/backward at batch one and verify all trainable gradients are finite.
- Disable the probe and compare inference output structure with stock RT-DETR-L.
- Verify the saved loss item names and diagnostics.

### Supervisor Tests

- GPU-memory bands produce the documented batch ladders.
- OOM, high peak, stable epochs, and numeric failure cause the expected transitions.
- state writes are atomic and round-trip complete.
- a corrupted `last.pt` falls back to the newest valid epoch snapshot.
- remote retention never deletes the previous checkpoint before the new upload is verified.

## Acceptance Criteria

The implementation is ready for server handoff only when:

1. all repository tests pass;
2. Python compilation and Bash syntax checks pass;
3. the real local forward/backward smoke test has finite total loss and gradients;
4. a forced half-input auxiliary-loss test remains FP32 and finite;
5. simulated OOM, numeric failure, and corrupt-checkpoint recovery tests pass;
6. the tested commit is pushed to `origin/codex/ioqc-sa`;
7. the clean-server guide contains clone, setup, launch, monitoring, recovery, migration, and result-download commands.
