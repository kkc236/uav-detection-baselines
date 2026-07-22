# Innovation One Evidence Pipeline Design

## Objective

Determine whether training-only P2 supervision can create a reproducible and
accumulating advantage for stock-query RT-DETR without adding any inference
query, parameter, or latency. Only a mechanism that passes numerical,
multi-seed, attribution, accumulation, and stock-export gates may receive a
fresh 100-epoch full-data run.

## Current Evidence and Failure Classification

E0b proved that contribution-separated P2 gradients can be restricted to
`model.0/1` plus auxiliary-private parameters while all 300 decoder queries
remain stock queries. The first E1 attempt at AMP scale 256 is not valid
effectiveness evidence: TSGR seed 1 failed closed at optimizer attempt 25 when
the scaled pure-stock gradient became non-finite and GradScaler changed from
256 to 128. The auxiliary gradient remained finite, query counts remained
`P2=0, stock=300`, and the routed ratio was inside the E0b envelope.

Three counterfactual checks isolate the numerical cause:

1. The same seed-1 control arm crossed attempt 63 at scale 256 without a skip.
2. The same seed-1 TSGR arm crossed the failing batch at fixed scale 128.
3. A fresh fixed-scale-128 TSGR run completed exactly 100 optimizer attempts
   with zero skip/non-finite/protocol violations, constant `128 -> 128` scale,
   constant `P2=0, stock=300`, routed ratio 0.121%--19.69%, and maximum
   normalized update ratio 4.70.

Therefore scale 256 is a toxic numerical setting for this TSGR trajectory. It
is not evidence that the P2 supervision is ineffective, and it does not justify
post-hoc changes to `lambda_p2` or `eta`.

## Frozen AMP128-v2 Numerical Contract

Both control and TSGR use fixed AMP scale 128 with a growth interval of
`2**31 - 1`. Any skip, non-finite value, scale change, or query isolation
violation invalidates the run and its paired arm. Dynamic backoff is forbidden
because it would change optimizer-update count, EMA evolution, and numerical
paths differently between arms.

Before E1, the 128-version E0 audit must cover A0, H0, and H1 for 100 optimizer
attempts. It must prove:

- zero skips and constant scale 128;
- finite nonzero routed and auxiliary-private H1 gradients;
- H0 has no routed stock gradient;
- H1 P2-only stock gradient is limited to `model.0/1`;
- stock, route, and private contributions are clipped independently;
- all query counts remain `P2=0, stock=300`;
- the routed ratio remains finite and the update monitor does not abort.

The experiment signature, initial-state metadata, runtime settings, evidence
manifest, and tests must all state scale 128. A mixed 256/128 comparison is
invalid.

## E1 Effectiveness Gate

Create new immutable protocols and new run directories, then rerun all six
10-epoch arms from their original seed-specific initial states:

```text
seed 0: control, TSGR
seed 1: TSGR, control
seed 2: control, TSGR
```

The old AMP256 seed-0 pair is marked `SUPERSEDED_AMP256`; the failed seed-1
TSGR is marked `INVALID_OVERFLOW_ATTEMPT25`. Their manifests, results, logs,
diagnostics, and failure JSONL remain forensic evidence but cannot enter any
AMP128 statistic.

E1 is valid only when all six runs pass artifact, pairing, optimizer, query,
checkpoint, and exact diagnostic checks. The frozen comparison then requires:

- final mAP50-95 wins on at least two of three seeds and positive mean delta;
- epoch 8--10 mean wins on at least two seeds and positive mean delta;
- no TSGR tail below 80% of its paired control;
- stock Top-300 tiny coverage wins on at least two seeds and positive mean delta;
- normalized best stock-candidate rank improves on at least two seeds and has
  negative mean delta.

No threshold, `eta`, `lambda_p2`, tiny definition, or diagnostic may change
after AMP128 E1 results are visible.

## Conditional Redesign

### If E1 Is Ineffective

Do not launch 100 epochs. B independently audits causal and pairing evidence;
C independently analyzes metric/query trajectories and proposes the smallest
attribution matrix; the primary agent reproduces both conclusions.

The attribution matrix compares only pre-registered single deltas from the
same initialization. It tests which previously observed uplift source is real:

1. stock-only control;
2. auxiliary-private P2 learning with H0 (`eta=0`);
3. routed shallow contribution with H1 (`eta=0.1`);
4. the legacy combined/global clipping path as a negative control only;
5. any query injection, gamma, EBC, QG, or quality path as a separately labeled
   historical negative control, never silently reintroduced.

The key point is the earliest reproducible divergence that is paired, finite,
and present in at least two seeds. A new version may contain only that point.
If no such point survives, innovation one is rejected rather than sent to a
100-epoch run.

### If E1 Is Effective

Treat the passing causal point as the sole positive mechanism. Build a clean
formal path that includes stock RT-DETR, the training-only P2 objective,
contribution-separated `model.0/1` routing, fixed AMP128, and evidence hooks.
Reject or remove from that path all toxic logic: inference query injection,
EBC competition, gamma fusion, QG/quality weighting, legacy global clipping,
dynamic AMP backoff, and permissive legacy defaults. Historical implementations
may remain only behind clearly named legacy modules/tests; the new formal entry
must fail if any toxic switch is active.

## Accumulating-Advantage Gate

After either conditional redesign produces a candidate, run a fresh paired
30-epoch accumulation experiment on the same hashed 10% subset. This run has an
explicit 30-epoch schedule and is not resumed into the 100-epoch result.

The candidate passes only if:

- epoch 21--30 mAP50-95 mean is above control;
- at least seven of those ten epochs are above control;
- epoch 30 is above control;
- cumulative signed mAP50-95 advantage from epoch 11 through 30 is positive;
- AP-tiny and Recall-tiny at epoch 30 are above control;
- stock coverage improves and normalized rank decreases at raw epochs 19, 24,
  and 29 on average;
- the mechanism, AMP, query, gradient-boundary, update-ratio, and stock-export
  gates remain valid.

Failure returns to attribution once. A second candidate failure rejects the
innovation rather than starting an open-ended parameter search.

## Final 100-Epoch Operation

Only after numerical E0, three-seed E1, conditional cleanup, 30-epoch
accumulation, and stock-export equivalence all pass may a fresh full-data
100-epoch seed-0 pair start. It uses the exact frozen candidate, fixed AMP128,
original seed-0 state, full VisDrone data, and no resume from E1/E2.

The run keeps a rolling window of three resumable checkpoints and permanently
archives the epoch-30 milestone plus raw epochs 97/98/99 (completed epochs
98/99/100), best, last, results, manifests, and diagnostics. Any skip or
protocol drift invalidates the pair. Seed 1/2 full runs begin only after the
seed-0 100-epoch gate passes.

## Collaboration and Decision Authority

- Primary agent: implementation, server execution, evidence collection, and
  final synthesis.
- B: adversarial audit of causal claims, protocol drift, numerical validity,
  and failure handling; no production edits.
- C: independent experiment tree, accumulation analysis, and final feasibility
  design; no production edits.

B and C receive the same frozen artifacts independently. Disagreement blocks
promotion until the primary agent resolves it with a pre-registered
counterfactual. No single arm or single seed can authorize 100 epochs.
