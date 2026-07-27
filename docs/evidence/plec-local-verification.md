# PLEC Local Verification

Date: 2026-07-27

## Scope

This record covers only the standalone PLEC module:

**Phase-Preserving Local Evidence Canonicalizer**

It verifies local tensor geometry, trainable structure, gradients, masking,
serialization, mixed precision, and repository regression behavior. It does not
cover RT-DETR integration or detection accuracy.

## Revision and environment

```text
branch:       codex/gcmv-rtdetr
code commit:  cfdd245046b9efcd40d6450947bdd469ce42cebb
Python:       3.10.11
PyTorch:      2.5.1+cu121
Ultralytics:  8.4.90
CUDA runtime: 12.1
GPU:          NVIDIA GeForce RTX 4070 Laptop GPU
```

The evidence document itself is committed after the code revision above.

## Public contract

Inputs:

```text
four local P3 tensors:
    4 x [B, C, Hl, Wl]

PLECGeometry:
    sample_grid       [B, 4, 9, Hg, Wg, 2]
    sample_valid      [B, 4, 9, Hg, Wg]
    center_valid      [B, 4, 1, Hg, Wg]
    subcell_offset    [B, 4, 9, 2, Hg, Wg]
    magnification     [B, 4, 2, Hg, Wg]
    edge_distance     [B, 4, 1, Hg, Wg]
```

Outputs:

```text
canonical         [B, C, Hg, Wg]
valid_count       [B, 1, Hg, Wg]
edge_prior        [B, 1, Hg, Wg]
overlap_weights   [B, 4, 1, Hg, Wg]
```

The default paper configuration uses `C=256`, four views, and nine phase
samples.

## Trainable structure

Default PLEC trainable parameter count:

```text
122,497
```

This passes the frozen `<= 200,000` parameter gate.

Tested parameter families:

- view embedding;
- phase-offset MLP;
- magnification/boundary MLP;
- grouped phase reducer;
- depthwise spatial mixer;
- pointwise projection;
- learned overlap head;
- channel-only output LayerNorm.

`test_full_plec_backpropagates_to_every_parameter_family` proves that every
family above receives finite nonzero gradients. All four contributing local
feature tensors also receive finite nonzero gradients. Geometry tensors remain
non-trainable and detached.

## Geometry and masking evidence

The local tests verify:

- exact full-P3 to local-P3 coordinate composition;
- fixed row-major `3 x 3` phase order;
- explicit `grid_sample(..., align_corners=False)`;
- non-integer x/y magnification without rounding;
- TL/TR/BL/BR order and 60%-overlap coverage;
- normalized crop-edge distance;
- one vectorized `grid_sample` for all views/phases;
- channel-major grouping of nine phases;
- valid-view-only masked softmax;
- overlap weights summing to one only over valid views;
- exact-zero invalid samples;
- exact-zero canonical feature, count, edge prior, and weights at empty
  locations even after all available biases are forced to `2.0`;
- batch size two with different non-square local/global feature shapes;
- fail-closed errors for malformed geometry and feature contracts.

## Verification commands and results

Static verification:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m compileall -q `
  src/gcmv_geometry.py src/gcmv_plec.py
git diff --check
```

Result: exit code `0`.

Focused PLEC/geometry verification:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest `
  tests/test_gcmv_geometry.py `
  tests/test_gcmv_plec.py `
  tests/test_plec_innovation_isolation.py `
  tests/test_sbr_geometry.py -q
```

Result:

```text
54 passed in 4.25s
```

Full local repository regression:

```powershell
& 'C:\uav_env\Scripts\python.exe' -m pytest -q
```

Result:

```text
909 passed, 1 skipped in 120.74s
```

The single skip is an existing environment-conditional test; the command
reported zero failures.

CUDA autocast was available and executed. The PLEC autocast test produced finite
forward output, scalar loss, local-feature gradients, and module gradients.

## Isolation result

The standalone PLEC source imports no:

- RT-DETR model/trainer;
- GGLF or PEG;
- query logic;
- SADED post-processing;
- SBR output fusion.

The module has trainable PyTorch layers, consumes feature tensors during the
forward pass, and backpropagates into its inputs. It is therefore a network
structure module, not inference post-processing.

## Explicitly not claimed

This local gate does not show that:

- PLEC improves mAP, AP-tiny, or recall;
- the local semantic P3 extraction path is feasible at training batch scale;
- GCMV fits a 24 GiB server;
- RT-DETR Top-300 remains identical when a future integration is disabled;
- GGLF or PEG is implemented;
- the complete GCMV method is publication-ready.

No server, dataset training, RT-DETR integration, GGLF implementation, or PEG
implementation was started during this local verification.
