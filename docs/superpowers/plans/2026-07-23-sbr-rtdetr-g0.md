# SBR-RTDETR Zero-Training G0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fail-closed, zero-training six-arm SBR-RTDETR G0 evaluation on the complete VisDrone validation split.

**Architecture:** New SBR-only modules implement pure tile geometry, deterministic class-aware Greedy NMM, SP-BRF, evaluation, artifacts, and a strict runner around the unchanged Ultralytics RT-DETR predictor. Raw view predictions are cached once, then recombined offline so Arms C and D differ only in duplicate-cluster coordinates.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, OpenCV, NumPy, pytest, JSONL/GZip, SHA256, RTX 4090.

---

### Task 1: Freeze the Design and Test Contracts

**Files:**
- Create: `docs/superpowers/specs/2026-07-23-sbr-rtdetr-g0-design.md`
- Create: `docs/superpowers/plans/2026-07-23-sbr-rtdetr-g0.md`

- [ ] Verify the design contains no `TBD`, `TODO`, adjustable scientific
  threshold, or reference to BQP execution code.
- [ ] Run:

```powershell
git diff --check
rg -n "TBD|TODO|bqp_capture|bqp_g0_validator" docs/superpowers/specs/2026-07-23-sbr-rtdetr-g0-design.md docs/superpowers/plans/2026-07-23-sbr-rtdetr-g0.md
```

  Expected: `git diff --check` exits 0; the search only finds the explicit
  prohibition in the design.
- [ ] Commit only the two documentation files:

```powershell
git add docs/superpowers/specs/2026-07-23-sbr-rtdetr-g0-design.md docs/superpowers/plans/2026-07-23-sbr-rtdetr-g0.md
git commit -m "Freeze SBR-RTDETR zero-training G0"
```

### Task 2: Implement Fixed Tile Geometry

**Files:**
- Create: `src/sbr_geometry.py`
- Create: `tests/test_sbr_geometry.py`

- [ ] Write failing tests for `Tile`, `LetterboxTransform`,
  `overlapping_tiles(width,height)`, `non_overlapping_tiles(width,height)`,
  `inverse_letterbox_xyxy(boxes, transform)`, and
  `tile_to_global_xyxy(boxes,tile,width,height)`.
- [ ] Tests must cover odd/even landscape and portrait sizes, ordered four
  origins, half-open bounds, exact integer overlap, full coverage, the odd
  remainder in Arm F, padding on each axis, corner boxes, seam boxes, and
  clipping. Round-trip coordinate error must be at most 0.5 px.
- [ ] Run and verify RED:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_geometry.py -q
```

  Expected: import failure because `src.sbr_geometry` does not exist.
- [ ] Implement frozen dataclasses and pure functions with
  `tile_w=ceil(0.60*W)` and `tile_h=ceil(0.60*H)`.
- [ ] Run the same test and verify GREEN.
- [ ] Commit:

```powershell
git add src/sbr_geometry.py tests/test_sbr_geometry.py
git commit -m "Add fixed SBR tile geometry"
```

### Task 3: Implement Standard Greedy NMM

**Files:**
- Create: `src/sbr_fusion.py`
- Create: `tests/test_sbr_fusion.py`

- [ ] Write failing tests for immutable `Detection`, `intersection_over_smaller`,
  `greedy_ios_clusters`, and `fuse_standard`.
- [ ] Cover IoS versus IoU, strict `>0.5` including an exact `IoS==0.5`
  non-match, cross-class rejection, stable
  score/source/index ties, seed-only non-transitive clustering, singleton
  identity, hand-calculated score-weighted coordinates, max score, and final
  stable `max_det=300`.
- [ ] Run and verify RED:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_fusion.py -q
```

- [ ] Implement only standard clustering and standard fusion. Do not implement
  SP-BRF in this commit.
- [ ] Run geometry and fusion tests and verify GREEN:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_geometry.py tests/test_sbr_fusion.py -q
```

- [ ] Commit:

```powershell
git add src/sbr_fusion.py tests/test_sbr_fusion.py
git commit -m "Add deterministic SBR Greedy NMM"
```

### Task 4: Add SP-BRF as an Independent Coordinate Rule

**Files:**
- Modify: `src/sbr_fusion.py`
- Modify: `tests/test_sbr_fusion.py`

- [ ] Add failing tests for `border_reliability(detection,tile,full_shape)` and
  `fuse_sp_brf(cluster)`.
- [ ] Hand-calculate full-view reliability, each artificial edge, multiple-edge
  minimum, real-image edges, no-edge fallback, reliability bounds,
  two/three-member weighted coordinates, maximum score, same cluster
  membership as standard, and bitwise singleton preservation.
- [ ] Run the focused test and verify RED for missing SP-BRF functions.
- [ ] Implement the exact formula from the design without changing standard
  clustering or standard fusion.
- [ ] Run both focused test files and verify GREEN.
- [ ] Commit separately:

```powershell
git add src/sbr_fusion.py tests/test_sbr_fusion.py
git commit -m "Add singleton-preserving border fusion"
```

### Task 5: Build the Frozen Evaluator

**Files:**
- Create: `src/sbr_metrics.py`
- Create: `tests/test_sbr_metrics.py`

- [ ] Write failing synthetic tests for class-aware one-to-one matching at
  IoUs `0.50:0.05:0.95`, perfect predictions, empty predictions, wrong classes,
  duplicate predictions, AP75, `max_det=300`, ignore-region neutralization,
  out-of-bin GT/prediction neutralization at each current IoU threshold
  (including a prediction that is neutral at AP50 but FP at AP75), and GT bins at
  effective radii 16, 32, and 96.
- [ ] Require `AP-tiny`, `AP-small`, `AP-medium`, `AP-large`, AP50, AP75,
  mAP50-95, and tiny micro recall at IoU 0.50.
- [ ] Run and verify RED:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_metrics.py -q
```

- [ ] Implement an accumulator using the same `ultralytics.utils.metrics`
  matching/AP primitives where possible. Size bins must use Arm-A 640 gain,
  independent of the prediction arm.
- [ ] Add a parity fixture that feeds identical prepared targets/predictions to
  the SBR evaluator and Ultralytics metric code and requires overall AP fields
  to agree within `1e-6`. The full-image Arm-A smoke must also compare against
  a stock `model.val` run using the same explicit
  `LetterBox(scaleup=False, scale_fill=False, auto=False, center=True,
  padding_value=114)` preparation.
- [ ] Run the focused suite and verify GREEN.
- [ ] Commit:

```powershell
git add src/sbr_metrics.py tests/test_sbr_metrics.py
git commit -m "Add frozen SBR G0 metrics"
```

### Task 6: Implement Raw-View Inference and Six Arms

**Files:**
- Create: `src/sbr_g0.py`
- Create: `tests/test_sbr_g0.py`

- [ ] Write failing tests for the frozen Arm A--F view contract, per-view
  `max_det=300`, simultaneous network/view/global coordinate state, one-time raw-view execution,
  byte-identical C/D inputs and clusters, final maxDet, deterministic
  serialization, and no label access before prediction freeze.
- [ ] Run and verify RED:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_g0.py -q
```

- [ ] Implement `FrozenSBRProtocol`, raw-view records, and view prediction
  collection around stock Ultralytics `RTDETR.predict`. Preprocess every view
  with `LetterBox((imgsz,imgsz), auto=False, scale_fill=False, scaleup=False,
  center=True, padding_value=114)` before passing the already-square image to
  the predictor, then store network/view/global boxes. Implement offline arm assembly,
  runtime counters, provenance, and finite/legal-coordinate validation.
- [ ] Do not import or patch any decoder, trainer, BQP module, model YAML, loss,
  or optimizer.
- [ ] Run all SBR tests and verify GREEN.
- [ ] Commit:

```powershell
git add src/sbr_g0.py tests/test_sbr_g0.py
git commit -m "Add zero-training SBR arm pipeline"
```

### Task 7: Add Fail-Closed Artifacts and CLI

**Files:**
- Create: `scripts/run_sbr_g0.py`
- Create: `src/sbr_artifacts.py`
- Create: `tests/test_sbr_cli.py`
- Create: `tests/test_sbr_artifacts.py`
- Create: `docs/SBR_RTDETR_SERVER_GUIDE.md`

- [ ] Write failing tests for exact checkpoint/data/source hashes, clean tracked
  worktree, Ultralytics/Torch versions, CUDA device, 548-image full-G0 set,
  frozen constants, non-empty output rejection, finite values, atomic
  JSON/JSONL/GZip, checksums, and resumable raw-view cache validation.
- [ ] The CLI exposes operational paths, device, workers, and run mode only.
  It must not expose tile ratio, overlap, IoS, confidence, maxDet, size bins,
  SP-BRF weights, or gate thresholds.
- [ ] Run and verify RED:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_cli.py tests/test_sbr_artifacts.py -q
```

- [ ] Implement the strict runner and artifact writer. Every failure exits
  nonzero before interpreting metrics.
- [ ] Document exact local/server commands and expected artifacts.
- [ ] Run all SBR tests and verify GREEN.
- [ ] Commit:

```powershell
git add scripts/run_sbr_g0.py src/sbr_artifacts.py tests/test_sbr_cli.py tests/test_sbr_artifacts.py docs/SBR_RTDETR_SERVER_GUIDE.md
git commit -m "Add audited SBR G0 runner"
```

### Task 8: Add an Independent Adjudicator

**Files:**
- Create: `scripts/adjudicate_sbr_g0.py`
- Create: `tests/test_sbr_adjudicator.py`

- [ ] Write failing tests that recompute metrics, deltas, gates, singleton
  preservation, cluster membership, image identity, hashes, seam-band
  provenance, `1-D/C` reduction (including the `C=0` cases), candidate
  association at same-class IoU `>=0.50`, and the mandatory
  `g0_gate.json == SBR_G0A_PASS` prerequisite from frozen raw artifacts;
  corrupt one field at a time and require fail-closed behavior.
- [ ] Run and verify RED, implement the adjudicator without importing
  runner-side metric summaries, then verify GREEN.
- [ ] Run the complete local suite:

```powershell
C:\uav_env\Scripts\python.exe -m pytest -q
git diff --check
```

- [ ] Commit:

```powershell
git add scripts/adjudicate_sbr_g0.py tests/test_sbr_adjudicator.py
git commit -m "Add independent SBR G0 adjudication"
```

### Task 9: Run S0 and Freeze the Executable Commit

**Files:**
- Create only runtime evidence in a new server output directory.

- [ ] Push `codex/sbr-rtdetr-g0`, verify the server host key, fetch, and switch
  `/mnt/uav/repo` to the exact clean commit.
- [ ] Verify checkpoint SHA256
  `54ce60289dd34c6750b8ba5f7516eefcf3afef6c174c6e4f3b1ef810c883099b`.
  Generate the validation content signature from a sorted list of
  `sha256  relative_posix_path` records for `images/val`, `labels/val`, and
  `labels_ignore/val`, hash that canonical UTF-8 manifest, and bind the
  generated value to the run manifest; never hard-code an unverified data hash.
- [ ] Create a deterministic 16-image S0 manifest before inference. Run all six
  arms without interpreting research deltas.
- [ ] Require Arm-A stock parity, legal coordinates, constant hashes, identical
  rerun checksum, C/D raw identity, and independent adjudication.
- [ ] If any check fails, stop and fix through a failing test. Otherwise write
  `SBR_S0_PASS`.

### Task 10: Run the One-Shot Full G0-A

**Files:**
- Create only runtime evidence in a new server output directory.

- [ ] Reverify exact clean commit, checkpoint/data hashes, 548 images, free
  space, and idle GPU.
- [ ] Run Arms A--F on the same frozen raw-view cache. Do not inspect partial
  metrics or change any scientific constant.
- [ ] Run the independent adjudicator and `sha256sum -c checksums.sha256`.
- [ ] Emit exactly `SBR_G0A_PASS` or `SBR_G0A_FAIL` from the frozen gate.
- [ ] On failure, archive the evidence and stop the SBR route. Do not search
  tile counts, tile ratios, thresholds, or postprocessors.

### Task 11: Run G0-B/C Only After G0-A Passes

**Files:**
- Reuse frozen G0-A raw artifacts; add adjudicated mechanism/runtime reports.

- [ ] Compare D versus C with the pre-registered G0-B gates.
- [ ] Record area amplification, per-view and combined tiny coverage,
  candidates per tiny GT, seam FP/truncated FP, duplicates, singleton
  preservation, sequential/batched timing, peak memory, and throughput.
- [ ] Apply exactly the seam-band, IoU matching, duplicate, boundary-target,
  coverage, and candidate-multiplicity formulas in the design; do not invent
  an alternate denominator or matching rule.
- [ ] Require the CLI to load `g0_gate.json`, verify status
  `SBR_G0A_PASS`, and compare source/checkpoint/dataset/protocol hashes before
  allowing G0-B or G0-C; add a fail-closed test for each mismatch.
- [ ] Obtain independent B causal/protocol audit and C experiment/paper audit on
  the same manifests and adjudication output.
- [ ] Produce a paper-ready evidence README and tables. Do not create slice
  fine-tuning data or launch training unless G0-A passed; do not launch any
  100-epoch run in this plan.
