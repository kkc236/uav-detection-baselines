# EBC-QP A1 D2 Isolation Design

## Objective

Determine whether the P2 branch and fixed-budget query injection are useful before
adding another quality mechanism. Preserve the existing stock A0 and fusion-gamma
A2 D2 results as immutable evidence, then run a matched A1 arm that differs from
A2 only by `lambda_ebc=0.0`.

## Frozen Evidence

The existing A0 and fusion-gamma A2 checkpoints, exact validation outputs,
diagnostics, logs, initial state, protocol manifest, and D2 gate are frozen by
path, byte size, and SHA256. The freeze manifest is immutable and is published
with the resumable checkpoints. Existing files are never overwritten.

## A1 Contract

A1 uses the fusion-gamma A2 configuration with exactly one difference:

```text
lambda_ebc: 0.05 -> 0.0
```

The P2 adapter, P2 box head, learnable fusion gamma, P2 auxiliary loss, epoch-4
query injection, global query competition, 300-query budget, initialization,
data subset, sample order, augmentation order, optimizer, scheduler, batch,
workers, AMP, and validation code remain unchanged.

## Preflight Tests

A single-batch A1 test must prove that P2 adapter, P2 box head, and fusion gamma
receive finite nonzero gradients while the total objective excludes EBC. A paired
A1/A2 forward test must prove identical stock/P2 candidate identities, final
query sources, source indices, and fixed query count before either arm trains.

## D2 Run

A1 starts from the exact fusion-gamma initial-state artifact and uses the exact
hashed D2 subset and protocol manifest. It runs for 10 epochs with seed 0. The
checkpoint is revalidated with the same full-validation preprocessing and exact
metric implementation used for A0 and A2. Read-only mechanism diagnostics are
collected from the final EMA checkpoint.

## Tri-Arm Interpretation

The comparison isolates two effects:

- A1 versus A0: P2 branch plus fixed-budget query injection.
- A2 versus A1: EBC loss.
- A2 versus A0: complete fusion-gamma EBC-QP method.

Use the existing frozen D2 mechanism and metric definitions without relaxing a
threshold after seeing A1. The branch decision is:

1. If A1 has positive net replacement (`N_gain > N_loss`, `V_replace > 0`) and
   does not fail the metric trajectory gate against A0, P2 is effective and a
   separate QG-P2 design may begin.
2. If A1 shows a metric signal but net replacement remains non-positive or the
   injection contribution is otherwise mixed, run A1-no-injection from the same
   initialization before designing QG-P2.
3. If A1 fails both metric and mechanism evidence, stop this P2 formulation and
   do not spend a 100-epoch run on it.

Any QG-P2 or no-injection implementation is a new frozen version and requires a
new strict A0/method D2 pair. No 100-epoch run starts until one version passes
both the metric and mechanism gates.
