# FDR Feasible-Geometry Repair and UAVDT Full Mapping

Date: 2026-08-30

## 1. Decision

Implement a parameter-free, straight-through pairwise extent projection between
the FDR Integral output and the pinned `distance2bbox` primitive.  The projection
must be an exact identity for every already-feasible FDR distance tuple and must
keep a useful localization gradient when a horizontal or vertical extent becomes
infeasible.  Do not clamp the GIoU loss, do not clamp each signed FDR edge, and do
not remove or relax the fixed AMP128 invariant.

The repair is developed from frozen source commit
`445120b7337dad0ccc41d18ad081b6b33580dcd2` on branch
`codex/fdr-feasible-geometry-uavdt`.

The branch also supplies a repository-owned UAVDT Full launcher.  `Full` maps
only to the existing `LRS-FDR+BPDD+FIA` graph:

- config: `configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml`;
- trainer: `src.rtdetr_lrs_system.TRAINER_TYPES["i"]`;
- method identity: `lrs_fdr_bpdd_fia`.

## 2. Observed failure and comparison evidence

The original UAVDT Full run reached epoch 14 with an epoch-running
`giou_loss=-5.821`.  The next non-finite optimizer attempt caused PyTorch AMP to
reduce its scale from 128 to 64, and the repository's fixed-scale invariant
correctly aborted the run.

The operator then tested a direct non-negative coordinate clamp.  That run no
longer showed the original early crash, but its saved best result was weak and
imbalanced relative to the completed baseline:

| UAVDT best | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.527 | 0.310 | 0.277 | 0.156 |
| Direct-clamp Full | 0.392 | 0.326 | 0.275 | 0.162 |

The direct clamp therefore cannot be accepted as a complete repair.  It removes
the sign failure at the loss boundary but creates degenerate zero-area boxes and
zero localization gradient for raw negative widths or heights.

VisDrone did not show the same systemic failure.  Its stored LRS evidence has
positive epoch-average GIoU losses, zero skipped AMP attempts, and zero
non-finite optimizer records.  The repair must consequently preserve the exact
valid path so that the existing VisDrone result is not gratuitously redefined.

## 3. Root-cause contract

For reference box width `w0`, height `h0`, scale `s=4`, and Integral distances
`dl, dt, dr, db`, the decoded extents are

```text
w = w0 * (s + dl + dr) / s
h = h0 * (s + dt + db) / s
```

The current independent edge distributions permit
`s + dl + dr <= 0` or `s + dt + db <= 0`.  The pinned Ultralytics GIoU path
assumes positive extents.  Once that assumption is violated, GIoU can exceed one,
the scaled GIoU loss can become negative, and optimization can reward increasingly
invalid geometry until FP16 gradients overflow.

The interface defect is therefore the absence of a joint feasibility contract
between the signed FDR edge representation and the stock box criterion.

## 4. Feasible-extent operator

Define the raw pair extents

```text
ex = s + dl + dr
ey = s + dt + db
```

and a fixed, dimensionless minimum extent `epsilon = 1e-3`.  Apply the
straight-through lower-bound operator

```python
def straight_through_lower_bound(raw, minimum):
    bounded = raw.clamp_min(minimum)
    return raw + (bounded - raw).detach()
```

to obtain `ex_safe` and `ey_safe`.  Correct the two edges in each pair equally:

```text
cx = (ex_safe - ex) / 2
cy = (ey_safe - ey) / 2

dl_safe = dl + cx
dr_safe = dr + cx
dt_safe = dt + cy
db_safe = db + cy
```

Properties required by the implementation:

1. If `ex >= epsilon` and `ey >= epsilon`, every output distance is bitwise equal
   to its input and the backward path is the ordinary identity.
2. If an extent is infeasible, its forward value is positive while the
   straight-through gradient with respect to the raw pair sum remains non-zero.
3. Adding equal corrections preserves `dr-dl` and `db-dt`; therefore the decoded
   center and the learned edge asymmetry do not move merely because feasibility
   was restored.
4. The operator changes neither corner logits nor the FGL target construction.
5. The pinned official `distance2bbox` function remains byte-for-byte unchanged.

The operator belongs in `src/fdr_math.py` as an explicit pure function and is
called in `src/fdr_head.py` immediately after Integral and immediately before
`distance2bbox`.  The same decoder path serves training, validation, inference,
normal queries, and denoising queries.

## 5. Runtime evidence

For every decoder forward, retain a plain, non-persistent evidence record with:

- total FDR query-layer elements;
- horizontal infeasible count;
- vertical infeasible count;
- minimum raw horizontal extent;
- minimum raw vertical extent;
- configured minimum extent.

This evidence must not become a parameter, buffer, checkpoint tensor, loss term,
or EMA state.  It exists to distinguish a rare numerical boundary event from a
continuing raw-distribution collapse.  Tests must show that state-dict keys are
unchanged.

## 6. UAVDT Full cross-server launcher

Add `scripts/train_uavdt_full.py`.  It is dataset-specific and must not call the
VisDrone dataset authority.  Its public inputs are:

- `--data-yaml`: an already validated UAVDT YAML;
- `--baseline-args`: the completed baseline run's `args.yaml`;
- `--initial-state`: the FDR initial-state artifact;
- `--output-root`;
- optional `--name`;
- optional `--dry-run`.

The launcher loads the baseline arguments, removes only baseline run identity and
path fields (`model`, `data`, `project`, `name`, `save_dir`, and `resume`), and
then binds the Full config, supplied UAVDT YAML, output root, safe run name, and
fresh-start semantics.  Dataset/training/evaluation settings otherwise come from
the baseline authority rather than being guessed from VisDrone defaults.

Before trainer construction it must:

1. verify that all three input files are ordinary files and resolve them;
2. validate the initial-state artifact with the existing weights-only loader;
3. parse the data YAML, require non-empty `train`, `val`, and `names`, require
   contiguous class ids when `names` is a mapping, and derive `nc` from `names`;
4. reject a conflicting declared `nc`;
5. write and print an immutable JSON launch record containing source identity,
   Full config hash, all input hashes, derived class count, method mapping, and
   final settings;
6. make `--dry-run` stop before model or trainer construction.

The launcher then dispatches exactly
`TRAINER_TYPES["i"](..., initial_state_path=..., experiment_seed=<baseline seed>)`.
No `g`, `h`, ad-hoc YAML, or stock trainer fallback is accepted by this entrypoint.

## 7. TDD and acceptance tests

Implementation begins only after the following tests are written and observed to
fail for the expected missing behavior:

### Geometry tests

1. Construct the known invalid case whose current decoded width and height are
   negative and whose scaled GIoU loss is approximately `-6`; require repaired
   widths/heights to be positive and the stock GIoU loss to be finite and
   non-negative.
2. Require exact tensor equality between raw and repaired distances for feasible
   random FP32 and FP16 inputs.
3. Backpropagate an L1 localization target through an infeasible pair and require
   finite, non-zero gradients whose optimizer direction increases the raw extent.
4. Require pair differences, decoded centers, tensor shapes, devices, and dtypes
   to be preserved.
5. Require both training and evaluation decoder paths to use the repair.
6. Require decoder state-dict keys to remain unchanged.

### Full mapping tests

1. Require the launcher to expose only the declared UAVDT Full contract.
2. Require `Full` to resolve to config arm `i`, trainer arm `i`, and method
   `lrs_fdr_bpdd_fia`.
3. Require baseline fields to survive except for the explicit path/identity
   replacements.
4. Require malformed data YAML, conflicting `nc`, unsafe names, missing inputs,
   and unsafe initial-state pickle payloads to fail before trainer construction.
5. Require dry-run authority bytes to be deterministic and conflict-safe.
6. Require non-dry-run to instantiate and train only the mapped Full trainer.

### Regression gates

- focused geometry tests;
- existing FDR math/head/loss/model tests;
- existing LRS system config/model/launcher tests;
- UAVDT Full launcher tests;
- full repository test suite if the focused gates pass.

## 8. Experiment acceptance boundary

The code repair is accepted when all automated gates pass and a UAVDT replay of
the former epoch-14 failure window has:

- no negative displayed GIoU loss;
- no non-finite gradient record;
- no AMP scale change;
- positive decoded widths and heights after the operator;
- recorded raw infeasible counts that do not grow without bound.

The replay is an engineering gate, not a final paper result.  Final UAVDT Full
evidence must be a fresh run using this single source branch.  Existing VisDrone
results may be retained only after the old VisDrone best checkpoint is evaluated
with the repaired decoder and reports zero feasibility activations; otherwise the
affected VisDrone arm must be re-evaluated and the activation disclosed.

## 9. Explicit non-goals

- Do not clamp negative GIoU losses to zero.
- Do not clamp each signed FDR distance independently.
- Do not add a new loss weight or trainable parameter.
- Do not change BPDD, FIA, LRS-FGL, matching, AMP scale, optimizer, or scheduler.
- Do not reuse the VisDrone launcher for UAVDT.
- Do not claim that code tests alone prove an accuracy improvement.
