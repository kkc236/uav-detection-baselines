# SADED-SM Final Review Notes

Date: 2026-07-25

Formal external anchor SHA-256:
`07f3f2f07a4e34f3615e174111c5490e9e20ac1c34a908831f7845d6241c34cd`

## B: integrity and causal review

B independently connected to the server and verified the first formal closure,
then identified two hardening opportunities: a theoretical input
time-of-check/time-of-use window and acceptance of extra subdirectories in a
closure. Neither changed the already independently recomputed five-gate result.

Root implemented both recommendations before the final run:

- JSON and checkpoint decisions now consume the authenticated bytes;
- all bindings and closure hashes are recomputed after input use;
- `input_snapshot_unchanged` is a required invariant;
- closure directories must contain only the exact required regular files.

The hardened source was committed as
`1891d6a33d63e93dee5e8c7ab6d73a6a5cc2d3b8`, its full server test suite
reported `831 passed`, and the hardened adjudication returned
`SADED_SINGLE_SEED_GO`.

B's accepted claim boundary is a single seed-0 development-val result only.
It is not an independent test confirmation, a multi-seed stability result, or
a T-ASCV success.

## C: planning and paper-positioning review

C rejected the short-lived full-Arm-A-prefix CTAF idea before execution because
the mature Arm-A output had only 578 aggregate free slots; the recall gate
would have required roughly 543 additional true positives and therefore an
unrealistic approximately 94% slot precision.

C selected the earlier pre-T-ASCV SADED route-control as the highest-success,
shortest valid path because it already had sealed positive evidence and
protects only predicted non-tiny detections, leaving adequate tiny capacity.
C also concluded that retraining the identical baseline was unnecessary: the
final router is training-free and the bound official checkpoint is already the
fixed seed-0 epoch-100 endpoint.

C's required paper wording is:

- call the method a single-model scale-aware multi-view router;
- say one full plus four local forwards, not compute-free;
- retain the custom metric names `AP-tiny-SBR` and `AP-large-SBR`;
- label the result `seed-0 VisDrone development validation`;
- do not claim test-set confirmation or cross-seed stability.

## Final review decision

Both reviews support accepting the exact five-gate result as the Innovation 1
seed-0 development-validation main experiment under the stated limitations.
