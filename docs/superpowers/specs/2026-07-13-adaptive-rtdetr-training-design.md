# Adaptive RT-DETR Training Design

## Goal

Complete the scratch RT-DETR-L VisDrone baseline on one RTX 5090 as quickly as practical without allowing a CUDA OOM or a disconnected terminal to stop the overall job.

## Starting Point

Resume from the original batch-24 scratch checkpoint after epoch 3. This checkpoint preserves the only uninterrupted optimizer and scheduler history. Later batch-20 and batch-18 runs loaded model weights without true resume and therefore are not used as the scientific baseline starting point.

## Adaptive Batch Policy

- Starting batch: 16.
- Available levels: 10, 12, 14, 16, 18, and 20.
- On OOM: resume from the last completed epoch one level lower.
- After the cooldown and three qualifying stable epochs: promote one level higher.
- Batch 20 is reached only after batch 18 proves stable; it is never the untested starting point.
- A stable epoch has no OOM and peak allocated CUDA memory below the batch-specific promotion threshold.
- Promotion thresholds are 22, 24, 26, 28, and 27.5 GiB at batches 10, 12, 14, 16, and 18 respectively.
- Batches 18 and 20 proactively drop one level after an epoch whose peak allocated memory reaches 29 GiB.
- Cooldown after repeated OOM events grows through 5, 10, and 20 completed epochs. It remains capped at 20 so recovery is still attempted automatically.

The child trainer may exit on an unrecoverable batch, but the supervisor remains alive. It restarts with `resume=True`, so model weights, EMA, optimizer, scaler, scheduler position, and epoch count remain continuous. At most the unfinished current epoch is replayed.

## Runtime

Training keeps image size 640, AMP enabled, RAM dataset caching enabled, 12 data-loader workers, fixed seed 0, and deterministic training enabled. `torch.compile` remains disabled because its compilation and memory behavior add risk to a baseline recovery job.

The supervisor owns a state file and a lock file. The state records current batch, cooldown, OOM count, last completed epoch, last peak memory, and checkpoint path. A heartbeat and a concise status file make the detached job observable.

## Completion And Preservation

The job is complete only when epoch 100 exists, final validation succeeds, and expected artifacts are readable. Metrics, arguments, plots, logs, and checksums are committed to GitHub. `best.pt` and a compact result archive are uploaded as GitHub Release assets because normal Git files exceed the 100 MB per-file limit.

Shutdown is allowed only after the Git push and Release uploads are verified remotely. Upload failures retry with backoff and leave the server running rather than discarding the only complete artifacts.

## Failure Handling

- CUDA OOM: lower batch, clear CUDA state by ending the child, and true-resume automatically.
- Planned promotion or proactive demotion: exit only after `last.pt` has been saved at an epoch boundary.
- Network/upload failure: retry without stopping the server.
- Unexpected trainer failure: retry the same checkpoint twice, then leave the server running with an explicit failure state.
- Duplicate launch: rejected by a process lock.

## Verification

Unit tests cover two-step demotion and promotion, cooldown growth, proactive peak-memory demotion, emergency batch 10, completion detection, and state persistence. A dry-run supervisor test uses a fake child process before the real GPU process is started.
