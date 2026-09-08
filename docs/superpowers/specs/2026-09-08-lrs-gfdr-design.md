# LRS-GFDR Design Specification

## Goal

Package Layerwise Reliability Shrinkage (LRS) and the feasible-geometry FDR decoder into one auditable innovation module, then rerun the frozen VisDrone experiment without BPDD or FIA.

## Scope and non-goals

- In scope: one declarative model configuration, an explicit module identity, runtime metadata, regression tests, and a fresh 100-epoch server run.
- In scope: preserve the current FDR decoder and LRS mathematics; make the geometry constraint explicit rather than relying on a hidden default.
- Out of scope: BPDD, FIA, backbone changes, optimizer changes, dataset changes, initial-state changes, or rewriting historical results.

## Module definition

The module is named **LRS-GFDR** (Layerwise Reliability Shrinkage with Geometrically Feasible FDR).

Forward path:

1. FDR predicts distributional offsets and decodes them through the feasible-geometry projection.
2. The projection enforces positive, finite box extents before reference boxes are updated.
3. LRS applies the existing reliability shrinkage term to FGL supervision with `alpha=0.25`.

The module has no BPDD/FIA branch and adds no inference-only parameters beyond the existing LRS-FDR implementation.

## Frozen experiment contract

- Dataset: VisDrone authority split already used by the formal runs.
- Initial state: `/root/sj-tmp/protocols/lrs-v2-f56b210/initial-state-seed0.pt`.
- Seed: `0`.
- Epochs: `100`.
- Batch: `8`; image size: `640`; device: `0`; AMP: enabled; deterministic: enabled.
- Optimizer and all scheduler/loss settings remain identical to the existing LRS-FDR formal arm.
- New output root: `/root/sj-tmp/runs/visdrone-v2-lrs-gfdr-20260908`.

## Acceptance criteria

1. Local configuration/model construction succeeds and explicitly resolves `feasible_geometry=true` and `reliability_shrinkage_alpha=0.25`.
2. Regression tests confirm the module excludes BPDD/FIA options and produces finite positive decoded box extents.
3. The server run creates 100 result rows plus `best.pt`, `last.pt`, and epoch checkpoints.
4. The second-dataset validation path does not crash from invalid boxes.
5. The result report records best and final mAP50/mAP50-95 and compares against historical pure FDR, Clean-FDR+geometry, and LRS-FDR+BPDD without overwriting them.
