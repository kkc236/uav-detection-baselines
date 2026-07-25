# SADED Fresh-100 Evidence Publishing Design

## Objective

Preserve the current state of `final-saded-fresh100-c5c35374` on GitHub now,
allow the active training processes to continue unchanged, and publish a
terminal evidence bundle automatically when the run ends.

## Frozen scope

- Do not stop, restart, signal, or modify PID `417400` or driver PID `417396`.
- Do not read or publish intermediate validation metrics.
- Do not add rolling resume checkpoints in this deployment.
- Publish the current progress as an explicitly non-terminal snapshot.
- Publish a terminal bundle for both success and failure.
- Never label a failed or incomplete run as a successful endpoint.

## Architecture

The existing Windows watcher remains separate from the scientific training
process. It polls only the remote `status` and `exit_code` files.

For `TRAIN_COMPLETE` with exit code `0`, it downloads and validates the final
training summary, configuration, protocol, log, results table, and
`weights/last.pt`. Lightweight evidence is committed to the dedicated result
branch and the checkpoint is attached to a GitHub Release.

For `TRAIN_INVALID` or a non-zero exit code, it downloads only the available
forensic evidence, writes an `INVALID` manifest, commits the lightweight
failure record, and creates a GitHub prerelease. No failed checkpoint is
presented as a successful endpoint.

## Current snapshot

The current snapshot contains timestamped progress, remote status, process
health, GPU/memory/disk observations, source commit, run identifier, protocol
path, and anomaly counts. It contains no model-selection or intermediate
validation metrics.

## Idempotency and integrity

- Result branch: `final-saded-fresh100-results`
- Success tag: `saded-fresh100-seed0-c5c35374`
- Failure tag prefix: `saded-fresh100-seed0-c5c35374-invalid`
- Every downloaded artifact receives a SHA256 checksum.
- Existing matching releases are reused; conflicting terminal state is an
  error.
- A lock prevents concurrent watcher instances.
- GitHub credentials remain on the Windows host and are not copied to the
  training server.

## Acceptance criteria

1. A current non-terminal snapshot is visible on the result branch.
2. The original training PIDs remain alive and the remote status stays
   `RUNNING`.
3. The watcher is alive and reports `WATCHING`.
4. A simulated success produces a success publication plan.
5. A simulated failure produces an `INVALID` prerelease plan and cannot
   produce a success plan.

