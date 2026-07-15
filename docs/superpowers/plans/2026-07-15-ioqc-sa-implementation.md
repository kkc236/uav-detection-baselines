# IOQC-SA Standalone Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a standalone, numerically stable IOQC-SA training module for stock RT-DETR-L and a GPU-adaptive, resumable Linux server workflow that publishes protected checkpoints to the existing GitHub repository.

**Architecture:** A forward pre-hook on only the final decoder cross-attention recomputes P3 sampling statistics in FP32 without changing baseline predictions. A pure IOQC-SA loss module performs dense-target selection, Hungarian ownership, Top-1 duplicate selection, margin competition, and normalized alignment. A custom RT-DETR model/trainer adds the training-only loss, while a separate supervisor handles GPU discovery, batch transitions, AMP fallback, checkpoint validation, and remote publication.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, SciPy through Ultralytics HungarianMatcher, Bash, GitHub REST API, pytest.

---

## File Structure

- Create `src/ioqc_sa_probe.py`: final-layer P3 sampling observer and FP32 moments.
- Create `src/ioqc_sa_loss.py`: pure ownership, duplicate selection, and stable auxiliary losses.
- Create `src/rtdetr_ioqc_sa.py`: stock RT-DETR-L integration and trainer.
- Create `src/gpu_adaptive_batch.py`: GPU-derived batch ladders and persistent transition state.
- Create `scripts/train_rtdetr_ioqc_sa.py`: scratch/resume training CLI and diagnostics callbacks.
- Create `scripts/supervise_ioqc_sa.py`: child-process recovery state machine.
- Create `scripts/setup_ioqc_sa_server.sh`: clean Linux server setup.
- Create `scripts/run_ioqc_sa_server.sh`: process lock, watcher, supervisor, and final publication.
- Create `docs/IOQC_SA_SERVER_GUIDE.md`: clean-server and migration instructions.
- Modify `scripts/sync_btdse_checkpoint.py`: include generic IOQC-SA lightweight artifacts without changing BTD-SE defaults.
- Modify `README.md`: add isolated IOQC-SA commands.
- Add focused tests under `tests/` for every new unit.

### Task 1: FP32 P3 Sampling Probe

**Files:**
- Create: `tests/test_ioqc_sa_probe.py`
- Create: `src/ioqc_sa_probe.py`

- [ ] **Step 1: Write failing probe tests**

Define a tiny fake deformable-attention module exposing `n_levels`, `n_points`, `n_heads`, `sampling_offsets`, and `attention_weights`. Assert the desired API:

```python
probe = P3SamplingProbe(cross_attention)
probe.capture(query_half, reference_boxes_half, value, [(8, 8), (4, 4), (2, 2)])
stats = probe.last_statistics
assert stats.center.dtype == torch.float32
assert stats.extent.dtype == torch.float32
assert stats.p3_mass.dtype == torch.float32
assert torch.isfinite(stats.center).all()
```

Also test exact weighted moments, zero-mass invalidation, gradient propagation to query and projection weights, hook registration on the final decoder layer only, and no capture in evaluation mode.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_ioqc_sa_probe.py -q`

Expected: collection fails because `src.ioqc_sa_probe` does not exist.

- [ ] **Step 3: Implement minimal FP32 probe**

Implement:

```python
@dataclass
class P3SamplingStatistics:
    center: torch.Tensor
    extent: torch.Tensor
    p3_mass: torch.Tensor
    valid: torch.Tensor
    p3_shape: tuple[int, int]

class P3SamplingProbe:
    def attach(self, decoder: nn.Module) -> None: ...
    def remove(self) -> None: ...
    def capture(self, query, reference_boxes, value, value_shapes, value_mask=None) -> None: ...
```

Use `torch.autocast(device_type=query.device.type, enabled=False)`, cast query/reference boxes to FP32, reproduce Ultralytics 8.4.90 offset/location equations, select the first level's points, retain original P3 mass, and compute FP32 center/extent with `1e-6` guards.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_ioqc_sa_probe.py -q`

Expected: all probe tests pass.

### Task 2: Stable IOQC-SA Loss

**Files:**
- Create: `tests/test_ioqc_sa_loss.py`
- Create: `src/ioqc_sa_loss.py`

- [ ] **Step 1: Write failing pure-loss tests**

Use synthetic owners, duplicates, predictions, and ground truths to cover:

```python
result = compute_ioqc_sa_loss(
    pred_boxes=boxes,
    pred_logits=logits,
    statistics=stats,
    targets=targets,
    match_indices=matches,
    density_threshold=1.0,
    duplicate_threshold=0.10,
)
assert result.competition.dtype == torch.float32
assert result.alignment.dtype == torch.float32
```

Separate tests must prove exact zero for empty/no-dense/no-duplicate cases, maximum competition for identical sampling, zero competition at one target-scale margin, owner stop-gradient, Top-1 uniqueness, tiny-box P3 floor, FP16 input stability, and deliberate rejection of non-finite aggregate values.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_ioqc_sa_loss.py -q`

Expected: collection fails because `src.ioqc_sa_loss` does not exist.

- [ ] **Step 3: Implement the pure loss**

Create dataclasses `IOQCSATargets` and `IOQCSALossResult`. Implement pure helpers for pairwise xywh IoU, dense-target masks, target-scale floors, owner maps, duplicate quality, and ramp weight. The public loss must run under disabled autocast, mask invalid sampling rows, use normalized Smooth-L1 alignment, and return graph-connected zeros.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_ioqc_sa_loss.py -q`

Expected: all loss tests pass with finite gradients.

### Task 3: Stock RT-DETR-L Integration

**Files:**
- Create: `tests/test_rtdetr_ioqc_sa_integration.py`
- Create: `tests/test_ioqc_sa_training_cli.py`
- Create: `src/rtdetr_ioqc_sa.py`
- Create: `scripts/train_rtdetr_ioqc_sa.py`

- [ ] **Step 1: Write failing integration and CLI tests**

Assert that `IOQCSADetectionModel` builds from `rtdetr-l.yaml`, contains no `BTDSE`, attaches one probe to the final decoder layer, removes the denoising prefix, exposes loss names `giou_loss`, `cls_loss`, `l1_loss`, `ioqc_comp_loss`, and `ioqc_align_loss`, and returns stock inference structure when auxiliary capture is disabled.

Parser tests must assert baseline defaults: 100 epochs, image size 640, scratch initialization, deterministic seed 0, `save_period=1`, `nbs=64`, and separate IOQC weights not forwarded into Ultralytics overrides.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_rtdetr_ioqc_sa_integration.py tests\test_ioqc_sa_training_cli.py -q`

Expected: imports fail because integration files do not exist.

- [ ] **Step 3: Implement model, trainer, and CLI**

Mirror the pinned `RTDETRDetectionModel.loss` data preparation so predictions are computed once. Preserve the original criterion and detection loss, recompute only the final ordinary-query Hungarian indices in FP32, add scheduled IOQC losses, and raise `FloatingPointError("NONFINITE_LOSS ...")` for non-finite detection, auxiliary, or total loss.

The CLI must expose `--amp true|false`, `--batch`, `--resume`, `--state`, four method hyperparameters, and callbacks that update epoch progress, reset peak memory, write atomic diagnostics, and request planned restart only after `on_model_save` confirms the completed checkpoint.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_rtdetr_ioqc_sa_integration.py tests\test_ioqc_sa_training_cli.py -q`

Expected: focused tests pass.

### Task 4: GPU-Adaptive State And Supervisor

**Files:**
- Create: `tests/test_gpu_adaptive_batch.py`
- Create: `tests/test_ioqc_sa_supervisor.py`
- Create: `src/gpu_adaptive_batch.py`
- Create: `scripts/supervise_ioqc_sa.py`

- [ ] **Step 1: Write failing batch-state tests**

Assert exact ladders for 24, 32, 48, and 80 GiB; proportional startup reduction when free memory is below 85%; promotion after three sub-82% epochs; demotion at 94%; OOM demotion/cooldown; numeric failure disabling AMP and demoting; and atomic JSON round-trip.

- [ ] **Step 2: Write failing supervisor tests**

Assert classification of planned restart, OOM markers, `NONFINITE_LOSS`, normal completion, and unexplained failure. Verify generated child commands carry the current batch, AMP mode, run path, state path, and resume checkpoint. Test stale lock recovery and newest-valid-checkpoint fallback using temporary checkpoints.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_gpu_adaptive_batch.py tests\test_ioqc_sa_supervisor.py -q`

Expected: imports fail because adaptive IOQC files do not exist.

- [ ] **Step 4: Implement state and supervisor**

Implement `GPUProfile`, `AdaptiveTrainingState`, `batch_policy_for_vram`, `record_epoch`, `record_oom`, `record_numeric_failure`, atomic save/load, process classification, checked checkpoint selection, capped restart delay, status JSON, disk guard, and PID-aware lock ownership.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_gpu_adaptive_batch.py tests\test_ioqc_sa_supervisor.py -q`

Expected: all state and supervisor tests pass.

### Task 5: Linux Server Protection And Publication

**Files:**
- Create: `tests/test_ioqc_sa_server_scripts.py`
- Create: `scripts/setup_ioqc_sa_server.sh`
- Create: `scripts/run_ioqc_sa_server.sh`
- Modify: `scripts/sync_btdse_checkpoint.py`
- Modify: `tests/test_btdse_sync_cli.py`

- [ ] **Step 1: Write failing script-contract tests**

Read the shell files as text and assert they require persistent storage, validate CUDA/GPU/disk/token permissions, use a process lock, start the checkpoint watcher, invoke the Python supervisor, use tag `ioqc-sa-rtdetr-l-live`, retain three assets, perform final one-shot verification, and default automatic shutdown to off.

Extend artifact tests to include `ioqc_sa_diagnostics.jsonl`, `batch_history.jsonl`, and `adaptive_state.json` while preserving existing BTD-SE artifacts.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_ioqc_sa_server_scripts.py tests\test_btdse_sync_cli.py -q`

Expected: the new shell scripts are missing or IOQC artifacts are absent.

- [ ] **Step 3: Implement setup and run scripts**

The setup script must create a persistent venv/dataset/run/log/secret layout, install pinned PyTorch and Ultralytics versions, configure the Ultralytics dataset directory, prepare VisDrone, and print detected GPU information without hard-coding a model name.

The run script must use `flock`, launch the independent sync watcher, invoke the supervisor, stop the watcher on exit, force a final verified one-shot publication, and optionally call `shutdown -h now` only when `AUTO_SHUTDOWN=1` and publication succeeded.

- [ ] **Step 4: Run focused tests and Bash syntax checks**

Run: `C:\uav_env\Scripts\python.exe -m pytest tests\test_ioqc_sa_server_scripts.py tests\test_btdse_sync_cli.py -q`

Run: `bash -n scripts/setup_ioqc_sa_server.sh scripts/run_ioqc_sa_server.sh`

Expected: tests and syntax checks pass.

### Task 6: Documentation, Full Verification, And GitHub Publication

**Files:**
- Create: `docs/IOQC_SA_SERVER_GUIDE.md`
- Modify: `README.md`
- Modify: any implementation file required by verification findings

- [ ] **Step 1: Write the clean-server guide**

Document persistent disk choice, clone of branch `codex/ioqc-sa`, environment setup, secure token entry, one-command launch, status/log/GPU monitoring, manual stop, automatic resume, AMP fallback, batch history, GitHub Release verification, migration to another server, and result/checkpoint download.

- [ ] **Step 2: Run all automated verification**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest -q
C:\uav_env\Scripts\python.exe -m compileall src scripts
bash -n scripts/setup_ioqc_sa_server.sh scripts/run_ioqc_sa_server.sh
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 3: Run real local smoke verification**

Run a batch-one, reduced-data training forward/backward on the local RTX 4070 with IOQC-SA active through an explicit test progress override. Confirm finite detection loss, competition loss, alignment loss, total loss, and all present gradients. If memory prevents full RT-DETR-L smoke, run the same integration at image size 320 and report that limitation explicitly.

- [ ] **Step 4: Commit implementation and documentation**

Stage only source, scripts, tests, specs, plans, and documentation. Exclude data, weights, runs, logs, secrets, caches, and rendered PDF artifacts. Commit with a message describing standalone IOQC-SA and resilient server training.

- [ ] **Step 5: Push and verify GitHub**

Push `codex/ioqc-sa` to `origin`, verify the remote SHA equals local `HEAD`, and use the exact branch in the operator guide.
