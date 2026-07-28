# GCTE ACR-EG Round 0 Handoff

Date: 2026-07-28 (Asia/Shanghai)

## Current authoritative conclusion

The previous `GCQF/SR-PEG` seed0 diagnostic is useful but is not accepted as
the final method. It passes every Global-relative safety gate; its only
shortfall is relative to the internal Fixed-SADED development anchor:
`map_nonnegative_vs_fixed`.

| Comparison | mAP50-95 | AP-tiny-SBR | Tiny recall | AP-medium-SBR | AP-large-SBR |
|---|---:|---:|---:|---:|---:|
| Full - Global | +0.010595 | +0.013412 | +0.054338 | -0.001640 | -0.000096 |
| Full - Fixed-SADED | -0.016579 | -0.030630 | -0.045411 | +0.009265 | +0.002036 |

The key coverage evidence is:

- Fixed-SADED accepted 120,326 local candidates and emitted 164,384
  predictions.
- The old learned gate accepted 23,283 local candidates and emitted 156,960
  predictions.
- 7,440 final prediction slots were left unused by the old learned route.

The failure is therefore over-rejection by the old third-stage hard gate, not
absence of useful multi-view evidence. This interpretation is supported by
the positive tiny and medium/large protection deltas against Global.

## Frozen correction

The only authorized correction before a formal run is:

`ACR-EG: Anchor-Conditioned Residual Evidence Gate`

The first two `GCQF` stages remain:

1. `GeometryQueryProjector`;
2. `GlobalLocalQueryInteraction`.

The third trainable stage adds the Fixed-SADED anchor bit to the local trunk,
adds a zero-initialized anchor-admission residual head, and uses a fixed
documented anchor prior. Safe local predictions then compete by learned
admission-weighted score and fill available capacity deterministically.
Non-tiny global protection, fragment rejection, provenance checks, and all
scientific gates remain unchanged. This is a network module, not a
post-processing-only fallback.

Estimated probability of passing the unchanged seed0 gate: 65–70%. If this
one-shot correction fails, local patching stops and the architecture is
reshaped.

## Repository state

- Repository: `kkc236/uav-detection-baselines`
- Branch: `codex/gcte-rtdetr-g0`
- Design commit: `46b5d98`
- Plan commit: `66bd54b`
- Previous diagnostic implementation commit: `3b1f43a`
- Current local state: TDD red tests have been added but implementation has
  not started.
- Focused red-test result before implementation: 9 failed, 22 passed, 1
  skipped. The failures are the expected missing ACR-EG interfaces.

The new design and implementation plan are:

- `docs/superpowers/specs/2026-07-28-anchor-conditioned-residual-evidence-gate-design.md`
- `docs/superpowers/plans/2026-07-28-anchor-conditioned-residual-evidence-gate.md`

## Sealed evidence references

- Dataset: VisDrone train 6,471 / validation 548
- Dataset signature:
  `A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A`
- Baseline checkpoint SHA256:
  `54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B`
- Previous seed0 evaluation SHA256:
  `7a4bab626aedf0b8e3b5c5aa3f09fc4bbcf4a6a094078413c218df11cbbb462b`
- Previous seed0 result:
  `/home/ubuntu/gcte-srpeg-seed0-output-3b1f43a4/seed0-evaluation.json`
- Previous module SHA256:
  `0AB9A949E838B1FFA462591BFB7934BACEBC252C94D0FF965CB516AD145CA88C`

The previous evaluation is retained as evidence, not as the ACR-EG result.
Fixed-SADED is an internal development anchor, not an external paper
baseline. Formal advancement is decided by the complete method versus the
original Global RT-DETR-L baseline, tiny improvement, medium/large
protection, and network/safety invariants. Fixed-relative deltas remain
reported for ablation and debugging.

Because ACR-EG uses utility and risk as learned rank features rather than
hard thresholds, calibration evaluates only the three effective
`global_retain` settings. This removes nine-fold duplicate CPU work without
changing any accepted candidate or scientific gate.

## Server continuity and cleanup

Replacement server: `36.103.199.151`, user `ubuntu`, SSH port `22`,
RTX 4090. No training process is currently running. The server has roughly
1.6 GB free on `/` and no free space on `/mnt/uav`.

Before the next GPU run, remove only explicitly abandoned GCMV/old-GCQF
outputs after path and process checks. Keep the VisDrone data, protocol
manifests, baseline checkpoint, current source snapshot, and authoritative
evaluation evidence. Each subsequent round must create:

1. a new source commit and server source directory;
2. a new output directory;
3. a tracked handoff containing metrics, checksums, logs, and resume paths.

Model weights are not stored in ordinary Git; their SHA256 and exact server
path are recorded in each handoff.

## Next action

Implement ACR-EG under TDD, run focused and full regression, deploy the exact
new commit to a new server directory, then run seed0 only. A formal
100-epoch RTX 4090 run is allowed only after every unchanged seed0 hard gate
passes.
