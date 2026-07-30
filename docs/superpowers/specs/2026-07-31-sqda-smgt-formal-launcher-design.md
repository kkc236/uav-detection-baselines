# SQDA-SMGT Formal Launcher Design

## Objective

Provide a resumable, independently initialized 100-epoch SMGT formal-training
entrypoint without ever allowing a bounded-feasibility G2 result to promote to
formal training. The entrypoint changes no detector or SMGT network operation:
it only encodes the already fixed protocol and its advancement condition.

## Chosen approach

The existing `g1`/`g2` training CLI gains a `formal` stage with its own fresh
run namespace and a fixed 100-epoch setting. The existing server launcher
accepts `formal` only when the completed G2 inventory contains a non-empty
`selected_checkpoint`; that field is written only by the strict all-metrics
decision, whereas `g2_eligible_checkpoint` is deliberately insufficient.

The formal run always initializes from the retained inherited adapter rather
than continuing an arbitrarily selected 10-epoch checkpoint. This preserves
the independent formal-trial interpretation. It retains epoch-level saves and
the existing resume discovery, so a server interruption resumes the same run
only after manifest and checkpoint validation.

## Alternatives rejected

1. Extend the G2 run to 100 epochs. This conflates screening and formal
   evidence and makes the formal initial state dependent on a selected G2
   checkpoint.
2. Allow `g2_eligible_checkpoint` to start formal training. This would turn
   the user-approved 0.1-point exploratory tolerance into a formal-pass claim.
3. Manually assemble a shell command after G2. That omits tested preconditions
   and makes resume/synchronization behavior error-prone.

## Safety and observability

The launcher must reject a missing or non-strict G2 inventory before it starts
a process. It uses a unique namespace, checks for an existing formal process,
records `gate: formal` and `target_epochs: 100` in the manifest/status, and
publishes a distinct live synchronization tag. The trainer remains geometry-
trust-only with the identical frozen tensors, data, AMP, image size, batch,
seed, detection and augmentation settings used by G1/G2.

## Acceptance criteria

- `formal` resolves only to `sqda-geometry-smgt-formal-seed0-100ep`.
- Its CLI settings fix `epochs=100`; no epoch override is exposed.
- The server launcher requires G2 `selected_checkpoint`, never just G2
  feasibility eligibility.
- The formal namespace uses the existing atomic manifest and checkpoint resume
  validation and a distinct GitHub synchronization status/tag.
- Focused and full tests pass before deployment. The launcher is not executed
  until a fresh G2 inventory proves strict passage.
