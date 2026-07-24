# T-ASCV implementation and staged training plan

## Objective

Implement the independent tiny expert authorized by the sealed SADED `R0_GO`,
then execute the frozen state machine:

`PREFLIGHT_1(seed0) -> TINY_MECHANISM_500(seed1) -> SCREEN_10(seed0,1,2) -> FORMAL_100(seed0,1,2) -> one sealed test-dev adjudication`.

The stopped ASCV-Loc route remains immutable and is never resumed or relabelled.
All treatment endpoints start from their stage/seed common initial state.

## Frozen evidence input

- R0 source: `ada48a1f09e468138e70eaa4b20cd426de6157da`
- route anchor:
  `e3c3a391496774412c60c921bf2db11cdbc2de908a562e5ad173123f36fb077c`
- evaluation anchor:
  `cbcd803318f59372b3bb0feffc234f829e9cd7653399fecf58e2e48c46926cc8`
- decision: `R0_GO`

## Task 1: pure tiny-only asymmetric loss

Create `src/tascv.py` and `tests/test_tascv.py`.

- Reuse public crop-v2 identity and local-to-full geometry from `src.ascv_loc`.
- Define a new result record with loss, matched tiny count, excluded non-tiny
  count, advantage sum, and wins.
- Select pairs by the full target's frozen 640-frame effective size `<=16`.
- Treat mapped local predictions as detached teacher coordinates and full-view
  predictions as the student.
- Use FP32 `L1 + (1 - aligned GIoU)` only for selected tiny pairs.
- Empty or tiny-empty cases return a finite differentiable zero.
- Degenerate or non-finite geometry fails closed.
- Prove that local teacher and all non-tiny pairs receive zero auxiliary
  gradient.

Verification:

`python -m pytest tests/test_tascv.py tests/test_ascv_loc.py -q`

## Task 2: stage policy and mechanism diagnostics

Create:

- `src/tascv_stage.py`
- `src/tascv_diagnostics.py`
- `tests/test_tascv_stage_policy.py`
- `tests/test_tascv_diagnostics.py`

Freeze:

- preflight seed0: 1 successful batch / 1 optimizer step / tensor `{8}`;
- mechanism seed1: 500 / 106 / tensor `{7,8}`;
- screen seeds0/1/2: 810 / 145 / tensor `{7,8}`;
- formal seeds0/1/2: 80900 / 10556 / tensor `{7,8}`.

The mechanism accumulator adjudicates only tail batches 401--500, requires the
frozen tiny pair/batch/advantage/win gates, and makes any non-zero non-tiny
auxiliary contribution `INVALID`.

## Task 3: Ultralytics integration

Create `src/rtdetr_tascv.py` and
`tests/test_rtdetr_tascv_integration.py`.

- Keep the stock RT-DETR parameter and inference schema.
- Evaluate the stock full-image criterion exactly once.
- The local pass supplies detached teacher coordinates only; it receives no
  detection or DN loss.
- Add exactly one fourth loss component named `tascv_loss`.
- Preserve BN buffers, AMP scale 128, MuSGD, canaries, checkpoint recomputation,
  and no-validation training behavior.
- Evaluation mode never constructs crops.

## Task 4: protocol, state machine, and stock-control allowlist

Create:

- `src/tascv_protocol.py`
- `src/tascv_cli.py`
- `scripts/prepare_tascv_protocol.py`
- `scripts/resolve_saded_controls.py`
- `tests/test_tascv_protocol.py`
- `tests/test_tascv_cli.py`
- `tests/test_saded_control_allowlist.py`

The new manifest binds R0, fresh seed-specific common states, stage endpoints,
`save_period=-1`, source closure, predecessor gates, and the unique
provenance-only `B(stage,seed)` allowlist. Zero historical matches means
`RUN_FRESH`; more than one means `INVALID`. The resolver may not open metrics,
results, AP, mAP, deltas, gate, val annotations, or test-dev.

## Task 5: runtime and adjudication

Create:

- `scripts/train_rtdetr_tascv.py`
- `src/tascv_adjudicator.py`
- `scripts/adjudicate_tascv.py`
- `tests/test_tascv_training_cli.py`
- `tests/test_tascv_adjudicator.py`
- `tests/test_tascv_adjudicator_cli.py`

Replay predecessor evidence independently, rehash checkpoints, separate
runtime `INVALID` from scientific `STOP`, and prohibit scientific CLI knobs.

## Task 6: unified SADED endpoint cache, route, and evaluator

Create:

- `scripts/cache_saded_endpoint.py`
- `scripts/route_saded_pair.py`
- `scripts/evaluate_saded_stage.py`
- `tests/test_saded_pair_cli.py`

For each `(stage,seed)` seal one stock B cache and one treatment T cache.
Route-control is `B-full + B-five-view-tiny`; route-treatment is
`B-full + T-five-view-tiny`. Both emit one unified prediction JSON and use the
same frozen router. Prediction routing and GT-aware evaluation remain separate
processes with external anchors.

## Task 7: staged execution and rolling storage

1. Seal the new source commit and T-ASCV protocol.
2. Run paired seed0 preflight.
3. Run fresh seed1 mechanism endpoint and independent adjudication.
4. Run screen seed0; only after its gate passes run seeds1/2.
5. Freeze the screen-selected configuration without tuning router constants.
6. Run formal seed0 for 100 epochs; only after its attribution/five-item gate
   passes run seeds1/2 from fresh common states.
7. Freeze all nine formal JSONs before opening test-dev exactly once.
8. Report the arithmetic mean of per-seed treatment-minus-A deltas.

Before every training launch `/home` must have at least 2GB free. Each seed is
trained, cached, routed, evaluated, anchored, mirrored off-server with matching
SHA256, then any unbound `best.pt` duplicate may be removed. `/mnt/uav` remains
read-only and historical STOP/INVALID evidence is never deleted.

## Review discipline

Each implementation task follows RED -> minimal GREEN -> focused regression.
B performs spec-compliance review first; code-quality review follows only after
spec approval. Server full-suite verification is required before any scientific
run. No task may alter thresholds, seeds, endpoints, or the state machine in
response to observed val metrics.
