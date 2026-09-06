# Historical FDR Four-Arm Transfer Design

## Decision

Use option A: derive one common four-arm warm-start state from the trusted
historical `DFR-BDPP-FIA-best.pt`, but discard every trained FIA tensor before
any new run. Freeze FDR to its historical implementation and hyperparameters;
do not add LRS, feasible-geometry projection, DCF, edge-adaptive FGL, or the
later AC-BPDD design to this experiment track.

The four arms are:

| Arm | Model identity | Added component |
| --- | --- | --- |
| F | Historical FDR | none |
| G | Historical FDR + BPDD | legacy parameter-free BPDD |
| H | Historical FDR + FIA | fresh P3-only FIA |
| I | Historical FDR + BPDD + FIA | legacy BPDD and fresh P3-only FIA |

This is a matched warm-start ablation. It is not an exact resume and must not
be described as a matched from-scratch reproduction of the older experiments.

## Frozen authority

The tensor source is the archived VisDrone Full checkpoint:

- asset: `DFR-BDPP-FIA-best.pt`;
- bytes: `67,871,120`;
- SHA-256:
  `FD972586E2833A28AA02D04AC9E460B0D7C8CFC4E6E8BCB48DA273F0E905593C`;
- graph: ten-class P3-only FDR + BPDD + FIA;
- observed state: `969` tensors;
- embedded best validation: epoch `96`, P `0.56996`, R `0.49848`, AP50
  `0.48982`, mAP50-95 `0.29618`.

The checkpoint manifest does not verify the exact training source commit.
Therefore the new artifact is checkpoint-derived. Historical source semantics
are reconstructed from the frozen FDR/BPDD evidence (`d97e1eb7` and
`848f00cb`) and the P3-only FIA graph evidence, not claimed as a byte-exact
reproduction of an unknown original checkout.

## Historical method semantics

The four configurations use the original FDR decoder settings:

- `reg_max=32`, `reg_scale=4.0`, `up=0.5`;
- cumulative residual distributions enabled;
- preliminary boxes enabled;
- FDR private seed `10000`;
- FGL weight `0.15`;
- preliminary-box supervision enabled;
- DN FDR supervision enabled, matching the established historical runtime
  default.

Historical decoding maps the integral distances through the original
`distance2bbox` path. The later feasible-geometry projection is not active.
The historical BPDD arms use weight `0.5`, temperature `0.5`, margin `0.02`,
epsilon `1e-6`, final-layer matching, and exclude denoising queries.

FIA is the P3-only graph: layer 22 consumes layer 21, layer 23 also consumes
the unmodified layer 21, and the FDR decoder consumes layers `[22, 25, 28]`.
Its private seed is `20000`, and its residual scale is initialized to zero.

## Common-state transformation

The source state has `969` tensors. Conversion must:

1. safely extract the model state and verify the source checkpoint hash;
2. remove exactly the `19` trained FIA tensors under `model.22.*`;
3. shift source keys `model.N.*`, where `N > 22`, to `model.(N-1).*`;
4. produce one canonical `950`-tensor Historical-FDR base state;
5. record a deterministic state fingerprint and the full provenance above.

The generated artifact role is
`visdrone_historical_fdr_four_arm_transfer`. It contains only tensors and
primitive metadata and must load with `torch.load(..., weights_only=True)`.

## Per-arm loading contract

F and G strictly load all `950` tensors with zero missing and zero unexpected
keys. BPDD is parameter-free, so their initial model tensors are identical.

H and I remap canonical keys at and after model index 22 by `+1`, loading the
same `950` shared tensors. The only allowed missing keys are exactly the 19 FIA
keys under `model.22.*`. H and I then initialize those keys with the same
private seed and zero residual scale. Their complete initial model tensors are
therefore identical to each other.

The loader must reject a different class count, artifact role, fingerprint,
source authority, tensor count, key set, shape, dtype, or unexpected private
key. It must also reject use by the current LRS track so the two experiment
families cannot be mixed accidentally.

## Software structure

The Historical-FDR track is isolated from the existing LRS code:

- four new explicit Historical-FDR YAML files;
- one Historical-FDR model/trainer router for F/G/H/I;
- one four-arm transfer-state builder/validator/loader;
- one Historical-FDR launcher accepting only `--arm`, `--dataset-root`,
  `--initial-state`, `--output-root`, optional `--name`, and `--dry-run`;
- one serial shell launcher for F -> G -> H -> I;
- one Chinese cross-server runbook with artifact hashes, dataset layout,
  environment, verification, commands, outputs, and paper-labeling rules.

Existing LRS configs, trainers, artifacts, and launchers remain available but
are not used by this historical experiment.

## Frozen training protocol

All four arms use one artifact and identical training settings except the
model YAML and run name:

- VisDrone train/val, `nc=10`, image size `640`;
- 100 epochs, seed `0`, deterministic mode;
- batch `8`, workers `4`, device `0`;
- no pretrained model argument and no resume;
- MuSGD, `lr0=0.01`, `lrf=0.01`, momentum `0.937`, weight decay `0.0005`;
- AMP enabled, nominal batch size `64`, mosaic closes for the last 10 epochs;
- checkpoint and validation behavior otherwise follow the frozen Formal100
  settings.

Each launch writes a conflict-safe authority JSON containing the Git identity,
config hash, artifact hash, dataset signature, arm identity, and full settings.

## Cross-server delivery

The runbook treats code and the weight artifact separately. Code is checked
out from the named branch/commit. The artifact is transferred as a binary
file or downloaded from a release asset, then verified by exact byte count and
SHA-256 before any deserialization. The repository must not contain access
tokens or passwords.

The runbook includes dry-run and smoke checks before the serial 100-epoch
queue. It explicitly states that a different artifact hash or dataset
signature invalidates direct comparison across the four new arms.

## Verification gates

Implementation is complete only after all of the following pass:

1. the converter rejects a wrong source hash and produces exactly 950 tensors;
2. the artifact is safe under weights-only loading and detects corruption;
3. real `nc=10` F/G models strict-load 950/950 tensors;
4. real `nc=10` H/I models load the same 950 shared tensors and leave only the
   same 19 FIA keys private;
5. H and I complete states are byte-equal at initialization;
6. legacy decode is selected for all four arms and the feasible-geometry path
   is not called;
7. F/G/H/I CPU smoke forwards are finite;
8. all four dry-runs emit authority records with identical protocol settings;
9. the cross-server commands and checksums match the generated artifact and
   committed launcher.

## Reporting boundary

The valid primary comparison is among F/G/H/I produced by this new matched
warm-start protocol. Prior scratch baseline/FDR numbers may be shown only as
historical context. Any paper table must label these new results as a shared
Historical-FDR warm-start ablation and must not merge them with strict scratch
rows without an explicit caveat.
