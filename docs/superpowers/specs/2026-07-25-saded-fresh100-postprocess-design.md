# SADED Fresh-100 Single-Endpoint Postprocess Design

## Objective

After the fresh seed-0 stock endpoint reaches epoch 100 and passes every
training canary, produce one paper-facing comparison from that checkpoint:

`Arm A (full view) -> SADED-SM (same checkpoint, full + four tiles)`.

## Frozen execution boundary

- One checkpoint supplies every prediction. There is no second model and no
  checkpoint selection.
- Cache generation executes exactly one full view and four fixed local views
  for each of the 548 ordered dev-val images.
- Cache generation and routing are GT-free processes. Dataset annotations and
  `src.sbr_metrics` may be imported only by the sealed evaluator after cache
  and route checksum closures are verified.
- The single-cache router reuses `route_paired_caches(cache, cache)` only as an
  implementation bridge. It requires byte-identical control/treatment routed
  outputs and then emits only `A` and `route_control`.
- The route algorithm, scale threshold, matching, fragment guard, score
  formula, Top-300 rule, and five formal gates are unchanged.
- The output is one unified routed prediction set; metrics are never selected
  per scale after evaluation.

## Authority

A postprocess manifest is frozen before the fresh endpoint is evaluated. It
binds the postprocess source commit, training protocol and source commit,
expected summary/checkpoint paths, image-list bytes, dataset YAML bytes,
dataset authority, fixed route contract, output paths, and formal thresholds.

The completed endpoint validator independently replays the fresh training
summary and checkpoint checks. An endpoint anchor then freezes the training
summary and checkpoint hashes before GPU cache inference starts.

## Evidence stages

1. `cache`: atomic five-view prediction closure plus external anchor.
2. `route`: atomic GT-free `A`/`route_control` closure plus external anchor.
3. `evaluation`: independent GT-aware metrics closure.
4. `adjudication`: standalone recomputation of exact deltas and five gates,
   checksum closure, and an external root anchor.

Any source, schema, path, checksum, image-order, checkpoint, or invariant
failure is `INVALID`. A valid five-gate failure is
`SADED_SINGLE_SEED_STOP`. Test-dev remains unopened.
