# VisDrone LRS-FDR / BPDD / FIA Minimal Rebuild Design

## Goal

Reconstruct a repository-owned, directly runnable launcher for the three pending
VisDrone experiment arms from the already published LRS-FDR, BPDD, and FIA
implementations. The rebuild targets ordinary C-conference reproducibility: the
same checked-in code, dataset split, training settings, and evaluator must be
recoverable on another server, but historical checkpoint bytes and byte-exact
initialization are not required.

## Source of Truth

The implementation branch starts from `origin/codex/lrs-fgl` at
`9f0bfb948a5434f9f0ed902b97f63bcd2f120745`. LRS-FDR remains the model and
training base. Only the audited BPDD loss/runtime behavior and P3-only FIA graph
are ported from their published branches. Historical result files and frozen
evidence are not modified.

## Chosen Architecture

Use a thin unified launcher over three explicit model configurations and the
smallest required Trainer specializations. Do not copy an old FDR+BPDD+FIA
launcher and rename it, because its preliminary-box and supervision defaults do
not represent LRS-FDR.

The public interface is:

```text
python scripts/train_visdrone_lrs_system.py \
  --arm {g,h,i} \
  --dataset-root PATH \
  --initial-state PATH \
  --output-root PATH \
  [--name NAME] \
  [--dry-run]
```

The arm mapping is fixed:

| Arm | Paper identity | LRS-FDR | BPDD | FIA |
|---|---|:---:|:---:|:---:|
| `g` | LRS-FDR + BPDD | yes | yes | no |
| `h` | LRS-FDR + FIA | yes | no | yes |
| `i` | LRS-FDR + BPDD + FIA | yes | yes | yes |

Arm letters are aliases only. They must not change any training hyperparameter
other than selecting the declared graph and its required Trainer.

## Configuration Contract

Create three explicit YAML files:

- `configs/rtdetr-l-lrs-fdr-bpdd.yaml`
- `configs/rtdetr-l-lrs-fdr-fia.yaml`
- `configs/rtdetr-l-lrs-fdr-bpdd-fia.yaml`

All three inherit the semantic settings of `rtdetr-l-lrs-fdr.yaml`:

- `preliminary_box: false`
- `distribution_feedback: false`
- `supervise_pre_boxes: false`
- `supervise_dn_fdr: false`
- `edge_adaptive_fgl: false`
- `reliability_shrinkage_alpha: 0.25`
- 300 queries and six decoder layers

BPDD settings remain `weight=0.5`, `temperature=0.5`, `margin=0.02`, and
`include_dn=false`. FIA remains P3-only and retains its zero-initialized
residual scale and private random-number behavior.

## Training Contract

The launcher reuses the frozen settings from `scripts/train_lrs_fdr.py` and does
not introduce separate defaults. All arms use seed 0, 100 epochs, `imgsz=640`,
and the same data preparation, optimizer, scheduler, augmentation, AMP, EMA,
and best-checkpoint selection behavior. A rebuilt initial-state file is allowed;
the launcher records its SHA256 for traceability but does not require the
historical `51AAB2EB...` bytes.

Before training, the launcher writes an authority JSON containing the arm,
resolved method identity, source commit, configuration hash, initial-state
hash, dataset signature, and fully resolved settings. An existing authority
file with different content is rejected. `--dry-run` performs all validation
and emits this record without constructing a long-running trainer.

## Component Boundaries

1. The launcher parses arguments, resolves the arm, prepares data, validates
   the initial state, records authority, and dispatches the Trainer.
2. Explicit YAML files declare graph topology and loss switches; the launcher
   does not mutate YAML dictionaries at runtime.
3. BPDD integration owns only its training-time loss and statistics.
4. FIA integration owns only the P3 feature path and its residual parameters.
5. LRS-FDR decoder and FGL behavior remain owned by the existing LRS source.

This separation keeps the three arms interpretable as module ablations.

## Error Handling

The launcher fails before training when an arm is unknown, a required path is
missing, the dataset cannot be prepared, an initial-state file cannot be
loaded, a configuration violates the frozen LRS contract, or an existing run
directory carries a conflicting authority record. It must print the resolved
arm and configuration before any GPU work.

## Test Strategy

Implementation follows test-first development. Tests must first fail for the
missing feature and then cover:

- exact `g/h/i` mapping and rejection of other values;
- preservation of frozen LRS-FDR switches in all three YAML files;
- correct isolation of BPDD and FIA in each arm;
- preservation of BPDD constants and P3-only FIA topology;
- resolved settings shared across arms except model/name identity;
- deterministic authority generation and conflict rejection;
- successful `--dry-run` without starting training;
- CPU-level model/config smoke construction when dependencies permit.

Full Formal100 training is outside the local implementation test suite.

## Deliverables and Acceptance

The rebuild is accepted when the new launcher, configurations, required model
integration, and tests are committed on
`codex/lrs-system-visdrone-rebuild`; the focused tests and relevant existing
LRS/BPDD/FIA regression tests pass; and the branch is pushed to GitHub with
copyable dry-run and training commands. No claim of numerical equivalence to a
server-only launcher is made without obtaining and comparing that launcher.

## Non-Goals

- Recovering historical LRS-FDR checkpoint bytes.
- Enforcing byte-identical initialization across historical and new servers.
- Uploading VisDrone images or annotations to GitHub.
- Changing BPDD, FIA, or LRS-FDR algorithms to seek additional accuracy.
- Reinterpreting unfinished G/H/I training results.
