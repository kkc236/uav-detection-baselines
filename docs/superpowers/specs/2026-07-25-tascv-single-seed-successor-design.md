# T-ASCV Single-Seed Successor Design

## 1. Decision and scope

This design is frozen before the authoritative `SCREEN_10` seed-0 gate is
opened. The user has replaced the earlier three-seed execution objective with
one development-to-formal seed-0 path:

```text
sealed SCREEN_10 seed0 decision
  -> continue only on TASCV_SCREEN_SEED0_GO
  -> fresh matched FORMAL_100 seed0 control and treatment
  -> two endpoint caches
  -> one GT-free route
  -> one sealed development-val evaluation
  -> independent FORMAL_SEED0 adjudication
  -> terminal single-seed result
```

Screen or formal seeds 1 and 2, three-seed gates, ablation runs, confirmation
predictions, and test-dev adjudication are out of scope. Test-dev remains
sealed. The result may be reported only as a single-seed controlled
experiment; it must not be described as a three-seed mean or as evidence of
multi-seed stability.

## 2. Parent authority

The parent development authority remains immutable:

- source commit:
  `dbf84670a89c74eb287978d5d70bb557625ef630`;
- protocol:
  `/home/ubuntu/tascv-protocols/final-tascv-dbf84670/protocol_manifest.json`;
- protocol SHA-256:
  `13D0E3EF66BFA2D35BB6037640888F7AC97993F2E43C090C6EC261A9701C25E3`;
- evidence root:
  `/home/ubuntu/tascv-evidence/final-tascv-dbf84670/screen-seed0`.

The successor may be created only after the complete evaluation and
independent adjudication anchors exist and the sealed decision is exactly
`TASCV_SCREEN_SEED0_GO`. `STOP` or `INVALID` is terminal for this route.
Concrete metrics and deltas are not inputs to successor construction.

The successor manifest records `observed_information=decision_only` and binds
the parent protocol, source commit and bundle, evaluation anchor,
adjudication anchor, gate file, and their SHA-256 digests.

## 3. Cross-commit scientific-core bridge

The successor is a new protocol and must not masquerade as a continuation of
the parent manifest. The new commit may change only protocol and authorization
glue needed to support the single-seed state machine.

Before the successor is sealed, it must prove byte identity between parent and
successor for all scientific-core files, including:

- model and T-ASCV loss implementation;
- training settings and stage runtime;
- endpoint cache and raw-view reconstruction;
- router and fusion implementation;
- evaluator, metric definitions, thresholds, and formal seed-0 adjudicator.

The exact allowlist is stored in the successor manifest. Any scientific-core
hash change invalidates the bridge and requires a fresh protocol beginning at
preflight.

The parent gate must be replayed in a clean checkout of the bound parent
commit, not by importing successor adjudication code. Replay output and exit
status are checksum-bound into the successor authorization.

## 4. Successor state machine

The only permitted states are:

```text
PARENT_SCREEN_SEED0_GO
  -> SINGLE_SEED_SUCCESSOR_AUTHORIZED
  -> SINGLE_SEED_RUNTIME_PREFLIGHT
  -> FORMAL_100_PAIRED_SEED0
  -> FORMAL_SEED0_EVALUATED
  -> FORMAL_SEED0_ADJUDICATED
  -> TERMINAL
```

Seeds 1 and 2 are rejected by protocol validation. A three-seed decision is
neither generated nor accepted. There is no force, skip-gate, synthetic-gate,
or repeated-seed compatibility path.

## 5. Formal training contract

Both formal arms must:

- use seed 0;
- start fresh from the same sealed seed-0 common initial state;
- use the same full VisDrone train split and frozen training configuration;
- run exactly 100 epochs and 80,900 successful batches;
- use fixed batch 8, workers 8, deterministic mode, AMP scale 128, and MuSGD;
- preserve the stock 300-query inference contract;
- write to new fixed successor endpoints under `/home/ubuntu`;
- freeze epoch-100 `last.pt`; `best.pt` is not selected.

The 10-epoch checkpoints are never resumed, initialized, or rebound as formal
checkpoints.

## 6. Evidence closure

After both fresh formal endpoints close:

1. validate and rehash both endpoint summaries and `last.pt` files;
2. generate the control cache, then the treatment cache, serially on GPU 0;
3. seal both cache anchors before routing;
4. create one GT-free paired route and seal its anchor;
5. launch the development-val evaluator only after route closure;
6. run the independent existing formal seed-0 adjudicator;
7. accept only its sealed `TASCV_FORMAL_SEED0_GO` or `TASCV_STOP`.

Intermediate metrics, deltas, partial outputs, and test-dev are not read.

## 7. Failure handling

- Parent `STOP` or `INVALID`: do not create or run the successor.
- Source, checksum, environment, or authority drift: `INVALID`, fail closed.
- Existing or partial fixed training target: `INVALID`; never delete and
  pretend it is fresh.
- Operational failure before an output is created and with staging cleaned:
  retry only the identical command after correcting the operational cause.
- Scientific formal `STOP`: terminal; do not tune thresholds or launch an
  ablation automatically.
- Any scientific-core code change: abandon the bridge and restart from a new
  preflight protocol.

## 8. Reporting boundary

The terminal artifact is the sealed formal seed-0 matched comparison and its
five-gate/attribution analysis. The paper must disclose:

- `n=1`, seed 0;
- the decision-only post-screen protocol amendment;
- fresh matched 100-epoch training;
- absence of multi-seed variance evidence.

No test-dev claim, multi-seed claim, or ablation claim is authorized by this
design.
