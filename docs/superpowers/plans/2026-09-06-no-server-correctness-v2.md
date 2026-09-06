# No-server correctness v2 implementation plan

**Goal:** repair numerical defects, expose the missing F baseline, verify local
execution, and publish source plus material evidence.

**Architecture:** retain pinned D-FINE primitives; place FP32 direct extent decode
in a separate wrapper; use masked logsumexp for AC-BPDD; reuse the FDR trainer for
arm F and one shared runtime recorder for VisDrone/UAVDT.

**Tech stack:** Python, PyTorch, Ultralytics 8.4.90, pytest, Git.

## Tasks and verification

- [x] Add adversarial regressions in `tests/test_no_server_correctness.py` for
  actual decoder BF16 autocast, tiny reference boxes after XYXY conversion,
  source NLL20/teacher NLL40, underflow, no teacher, detach, and the F launcher.
  Run `python -m pytest tests/test_no_server_correctness.py -q -p no:cacheprovider`
  before production edits and record the failing behavior.
- [x] Add `decode_feasible_fdr_boxes(points, distance)` in `src/fdr_math.py`.
  Center is `points_xy + (right_bottom-left_top)*points_wh/(2*scale)`;
  extent is `(scale+left_top+right_bottom)*points_wh/scale` bounded by the larger
  of the relative floor and detached machine-precision floor. Use
  `bounded.detach() + (raw-raw.detach())` for the explicit surrogate gradient.
  Promote half inputs; keep the official function unchanged. In `src/fdr_head.py`,
  disable autocast around Integral and decode; retain FP32 geometry and diagnostics.
- [x] Replace current AC-BPDD teacher clipping in `src/bpdd_loss.py` with detached
  log_softmax, masked logsumexp normalization, and logsumexp mixture. Inactive
  edges use a finite placeholder and exact zero gate. Keep targets, temperature,
  margin, candidate normalization, and the legacy loss contract unchanged.
- [x] Add F model/trainer in `src/rtdetr_lrs_system.py` using the existing LRS
  YAML and strict artifact loader. Extend VisDrone maps, version authorities to
  v2, reject checkpoint weights, update model/launcher tests, and persist runtime
  evidence in all arms through a shared recorder.
- [x] Run numerical, actual model/criterion, launcher, runtime, and pinned
  primitive parity tests. Record pass/fail/skip counts and environment versions.
  CPU BF16 autocast is local numerical evidence; CUDA and Formal100 remain
  separate unperformed gates. Save the audit and revised operating instructions.
- [ ] Commit source, create a tracked-source ZIP and SHA-256 manifest/test report
  in the material repository, update material entry documents, and verify archive
  correspondence. Push both branches and verify exact remote heads. Preserve
  historical results and unrelated changes.

Local acceptance: 176 passed, 1 CUDA-dependent skip. CPU pure FP16 was found to
produce nonfinite stock grid_sample output with finite inputs; an explicit CPU
FP16 rejection is tested, not reported as successful half inference. Publication
is recorded after this source snapshot in the material package authority/report;
the final checkbox above is a pre-publication snapshot, not a training gate.

## Audit corrections

Reject a fixed normalized 1e-3 width prior, standalone-gain early stopping,
universal AP significance thresholds, and dry-run-as-training-readiness claims.
User authorization already covers implementation and publication of tested work.
