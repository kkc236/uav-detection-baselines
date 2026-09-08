# LRS-GFDR-FIA Design Specification

## Goal

After the in-progress LRS-GFDR run finishes, execute one isolated FIA extension under the same frozen protocol.

## Design

Use the explicit LRS-GFDR graph and insert the existing P3-only FIA layer at model index 22. P4 and P5 retain their stock bypass paths. The decoder keeps `feasible_geometry=true`; the loss keeps `reliability_shrinkage_alpha=0.25`. BPDD and FIA are not combined with any other adapter in this arm.

## Initialization and comparison

The shared FDR state loads from the same `initial-state-seed0.pt`; FIA private parameters use `20000 + seed` and zero residual scale. The arm starts fresh and never loads the completed LRS-GFDR checkpoint. The comparison is LRS-GFDR-FIA versus LRS-GFDR using identical data and settings.

## Acceptance criteria

- Model construction validates exactly one FIA at P3, stock P4/P5 bypass, and explicit feasible geometry.
- No BPDD options or BPDD loss are present.
- The post-GFDR run produces 100 result rows, best/last checkpoints, finite geometry statistics, and no second-dataset crash.
