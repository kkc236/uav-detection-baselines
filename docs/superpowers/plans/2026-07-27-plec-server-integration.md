# PLEC RT-DETR Integration and Server Screen Plan

**Goal:** Connect the locally verified PLEC network module to RT-DETR-L,
prove the real multi-view training path on CUDA, and start a bounded server
screen without modifying the stock decoder, matcher, heads, P4, or P5.

**Frozen boundary:** The full source image is centered-letterboxed to 640.
Four ordered 60%-overlap source-resolution crops are centered-letterboxed to
1088. All five views share the backbone and Hybrid Encoder. Local views stop
at semantic P3 (layer 21). PLEC canonicalizes the four local P3 tensors and the
common pre-PEG reference adapter injects only the canonical tensor into global
P3 immediately before the stock decoder.

## Task 1: Add the paired source-resolution view data path

- Add `src/gcmv_data.py`.
- Preserve the raw source image until the four crops are created.
- Disable unpaired mosaic, mixup, cutmix, copy-paste, geometric, colour and
  flip augmentation for the first PLEC screen.
- Reuse the frozen `overlapping_tiles()` and centered `LetterBox` convention.
- Return stacked `local_views: [B,4,3,1088,1088]` and
  `source_shapes: [B,2]` alongside the stock full-view detection batch.
- Write and run failing tests before implementation.

## Task 2: Add the explicit model configuration and reference adapter

- Add `configs/rtdetr-l-gcmv-plec.yaml` with the complete RT-DETR-L graph and a
  top-level `gcmv` section that names PLEC, layer 21, the 640/1088 view sizes,
  and the zero-initialized reference adapter.
- Add `PLECReferenceAdapter`, one shared 1x1 projection and one scalar
  `gamma_ref` initialized to exactly zero.
- Test exact stock identity at zero gamma and live gradients at nonzero gamma.

## Task 3: Connect shared local Hybrid-Encoder passes to stock RT-DETR

- Add `src/rtdetr_gcmv_plec.py`.
- Run the global path once and preserve `G3`, `G4`, and `G5`.
- Run each ordered local view through the same layers 0--21 under activation
  checkpointing and frozen BatchNorm running statistics.
- Build exact PLEC geometry from the source shapes and actual P3 shapes.
- Replace only the decoder's P3 input with
  `G3 + gamma_ref * Project(L3c)`.
- Test layer boundaries, exact stock identity, local BatchNorm preservation,
  state loading, and gradient paths.

## Task 4: Add bounded preflight and training entry points

- Add `scripts/preflight_gcmv_plec.py` for one real CUDA batch.
- The preflight must verify gamma-zero identity, finite forward/loss/backward,
  finite nonzero PLEC/local-feature gradients with the audit gamma enabled,
  unchanged local BatchNorm buffers, and peak CUDA memory.
- Add `scripts/train_rtdetr_gcmv_plec.py` with explicit model, pretrained
  checkpoint, data, output, fraction, batch and epoch arguments.
- Internal Ultralytics validation is not a scientific PLEC result and remains
  disabled for the initial train-only screen.

## Task 5: Verify, deploy and start the server screen

- Run focused tests, static compilation, `git diff --check`, then the full
  local suite.
- Transfer the exact clean branch state into a new directory under
  `/home/ubuntu`; do not write to the full `/mnt/uav` filesystem.
- Run focused tests with `/mnt/uav/venv/bin/python`.
- Run CUDA preflight at batch 1, then batch 2 if memory permits.
- Start a 3-epoch, 3%-data PLEC screen in the background using the largest
  preflight-approved batch, and confirm PID, log growth, GPU use and checkpoint
  directory.

