# VisDrone Full Transfer Initial-State Design

## Objective

Convert the trusted `DFR-BDPP-FIA-best.pt` release checkpoint into a safe,
weights-only initialization artifact for the current VisDrone Full model. The
artifact targets `nc=10` and preserves all 969 checkpoint tensors exactly.

This artifact is a transfer-learning start, not the paired scratch
`initial-state.pt` used to establish causal baseline or module ablations.

## Source authority

- Source asset: `DFR-BDPP-FIA-best.pt`
- Source size: `67,871,120` bytes
- Source SHA-256:
  `FD972586E2833A28AA02D04AC9E460B0D7C8CFC4E6E8BCB48DA273F0E905593C`
- Source topology: P3-only FDR + BPDD + FIA, ten VisDrone classes
- Observed state contract: 969 tensors
- Current `nc=10` Full compatibility: 969 common keys, 969 shape-compatible
  tensors, zero missing keys, zero unexpected keys

The source checkpoint is stripped (`epoch=-1`) and contains no usable
optimizer state. The resulting training mode is weight-only warm start with a
new optimizer and schedule, never exact resume.

## Artifact contract

The generated file is named `initial-state.pt` under a VisDrone transfer
artifact directory. It contains only tensors and primitive metadata so it can
be deserialized with `torch.load(..., weights_only=True)`.

Required fields:

- an explicit transfer format identifier;
- `artifact_role=visdrone_full_transfer`, preventing confusion with the paired
  scratch FDR artifact;
- `nc=10`, `channels=3`, source checkpoint SHA-256 and source tensor count;
- the complete current-Full-compatible state dictionary;
- a deterministic fingerprint of the tensor state.

Generation must fail if the source hash differs, any source tensor is missing,
the source and current Full keys differ, or any shape/dtype differs. It must not
silently drop or reinitialize a tensor.

## Loading boundary

The transfer artifact is accepted only by the current Full topology
(`LRS-FDR + AC-BPDD + FIA`, arm `i`) with `nc=10`. It is rejected for UAVDT,
stock RT-DETR baseline, arms `f/g/h`, and exact-resume paths.

The existing paired scratch artifact and its loader remain unchanged. A
separate explicit transfer loader prevents a transfer state from being passed
accidentally to an experiment that claims matched scratch initialization.

## Training and paper semantics

Training starts with the transferred model tensors but creates a new optimizer,
EMA, epoch counter and learning-rate schedule. Runs must be labeled
`VisDrone Full warm-start` or equivalent. They cannot replace the completed
from-scratch baseline/FDR/module ablations and cannot be reported as a clean
causal comparison against a scratch baseline.

The intended use is continued VisDrone optimization or engineering reuse of
the trained detector. If the transferred run is compared scientifically, its
control must use an explicitly matched transfer protocol.

## Verification

Completion requires all of the following:

1. the source file hash equals the frozen source authority;
2. safe weights-only loading of the generated artifact succeeds;
3. metadata reports `nc=10`, 969 tensors and the exact source hash;
4. strict loading into the current Full model reports zero missing and zero
   unexpected keys;
5. every loaded tensor is byte-equal to the extracted source tensor;
6. the current Full model completes a finite CPU smoke forward after loading;
7. the final artifact size and SHA-256 are recorded for transfer to another
   server.

