# FDR + FrequencyCM Formal100 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one identity-initialized, AMP-safe FrequencyCM P5 preprocessor to the formal FDR detector and run a fresh strict full-data seed-0 100-epoch experiment with recoverable per-epoch publication.

**Architecture:** Add FrequencyCM as a standalone YAML graph layer after final fused P5, moving the decoder from index 28 to index 29 and routing `[21, 24, 28]` into the unchanged `FDRRTDETRDecoder`. Reuse the frozen FDR initial state through one declared decoder-key alias, initialize only the YAML layer's private tensors with a private seed, and extend the existing formal trainer/protocol without changing frozen hyperparameters.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, pytest, CUDA AMP, Git/GitHub Releases.

---

### Task 1: Implement the isolated FrequencyCM unit with TDD

**Files:**
- Create: `tests/test_frequency_cm.py`
- Create: `src/frequency_cm.py`

- [ ] **Step 1: Write failing tests**

Test shape preservation, exact identity at zero `gamma/beta`, RNG isolation, finite FP32 FFT reconstruction for a 20 x 20 feature, first-step gate gradients, and branch gradients after nonzero gates.

```python
def test_frequency_cm_is_exact_identity_at_initialization():
    module = FrequencyCM(256, private_seed=20_000).eval()
    value = torch.randn(2, 256, 20, 20)
    torch.testing.assert_close(module(value), value, rtol=0, atol=0)
```

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_frequency_cm.py`. Expected: collection failure because `src.frequency_cm` does not exist.

- [ ] **Step 3: Implement the minimum module**

Implement private-seed construction under `torch.random.fork_rng`, two zero residual scales, full-spectrum magnitude processing, explicit FP32 FFT/phase reconstruction under disabled autocast, and same-channel output.

```python
class FrequencyCM(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        frequency = self.frequency(self.norm1(value))
        low = value + frequency * self.gamma
        high = self.spatial(self.norm2(low))
        return low + high * self.beta
```

- [ ] **Step 4: Verify GREEN**

Run `pytest -q tests/test_frequency_cm.py`. Expected: all tests pass.

- [ ] **Step 5: Commit**

Commit `test/feat: add AMP-safe FrequencyCM unit`.

### Task 2: Integrate FrequencyCM as a removable YAML layer

**Files:**
- Create: `tests/test_rtdetr_fdr_frequencycm.py`
- Create: `src/rtdetr_fdr_frequencycm.py`
- Create: `configs/rtdetr-l-fdr-frequencycm.yaml`
- Modify: `src/rtdetr_fdr.py`

- [ ] **Step 1: Write failing integration tests**

Require exactly one FrequencyCM at model index 28, the unchanged `FDRRTDETRDecoder` at index 29, decoder inputs `[21, 24, 28]`, exact initialized shared tensors after the sole `model.28.* -> model.29.*` alias, only `model.28.*` FrequencyCM keys as new private state, and finite inference/training contracts.

```python
def test_frequencycm_model_preserves_fdr_shared_state_keys(fdr_artifact):
    model = FDRFrequencyCMDetectionModel(nc=10, verbose=False)
    report = load_fdr_frequencycm_initial_state(model, fdr_artifact)
    assert report["shared_mismatch_count"] == 0
    assert all(name.startswith("model.28.") for name in report["private_keys"])
```

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_rtdetr_fdr_frequencycm.py`. Expected: import failure for the integration module.

- [ ] **Step 3: Register the standalone parser module safely**

Expose `FrequencyCM` to the Ultralytics YAML parser while holding the existing re-entrant parser lock. Its same-channel contract lets the parser retain the input channel metadata without modifying the installed Ultralytics package.

- [ ] **Step 4: Implement the model/trainer integration and authority alias**

The model uses the unchanged FDR decoder and loss. The initial-state loader remaps only the old decoder prefix from model index 28 to 29, verifies every remapped tensor byte-for-byte, and leaves the FrequencyCM layer at its deterministic identity initialization. The trainer partitions gradient evidence into common, FDR-private, and FrequencyCM-private disjoint groups.

```python
def remap_fdr_decoder_key(name: str) -> str:
    return "model.29." + name.removeprefix("model.28.") if name.startswith("model.28.") else name
```

- [ ] **Step 5: Add an isolated YAML**

Copy the formal FDR graph, insert `[-1, 1, FrequencyCM, [256, 20000]]` as index 28, move the decoder to index 29, and route `[21, 24, 28]`. Leave `configs/rtdetr-l-fdr.yaml` byte-unchanged.

- [ ] **Step 6: Verify GREEN and regression safety**

Run the new tests plus `tests/test_rtdetr_fdr.py`, `tests/test_fdr_yaml_configs.py`, and all FDR head/loss tests.

- [ ] **Step 7: Commit**

Commit `feat: integrate FrequencyCM into FDR P5 path`.

### Task 3: Extend immutable protocol and formal training CLI

**Files:**
- Modify: `tests/test_fdr_protocol.py`
- Modify: `tests/test_train_rtdetr_fdr_cli.py`
- Modify: `src/fdr_protocol.py`
- Modify: `scripts/prepare_fdr_protocol.py`
- Modify: `scripts/train_rtdetr_fdr.py`

- [ ] **Step 1: Write failing protocol/CLI tests**

Require `fdr_frequencycm` as the only new variant, a source-bound formal run identity, the FrequencyCM YAML, 100 epochs, unchanged `FROZEN_SETTINGS`, the FrequencyCM trainer type, and separate FrequencyCM gradient evidence.

- [ ] **Step 2: Verify RED**

Run the two focused test files. Expected: rejection of the unknown variant.

- [ ] **Step 3: Add the variant minimally**

Extend `build_run_identity`, protocol-manifest creation, CLI choices, settings selection, trainer creation, FDR-loss reporting, and evidence fields. Do not alter `FDR_PROTOCOL`, its SHA, or existing control/FDR identities.

- [ ] **Step 4: Verify GREEN**

Run the focused tests and confirm old manifest tests remain green.

- [ ] **Step 5: Commit**

Commit `feat: add strict FDR FrequencyCM formal variant`.

### Task 4: Add verified publication rotation

**Files:**
- Create: `tests/test_frequencycm_publication_rotation.py`
- Create: `scripts/rotate_published_frequencycm_checkpoint.py`

- [ ] **Step 1: Write failing safety tests**

Require fail-closed behavior unless the queue entry, local file size/SHA256, and remote release manifest all match. Prohibit deletion of `last.pt`, `best.pt`, files outside the immutable run's `weights` directory, or unpublished numbered checkpoints.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_frequencycm_publication_rotation.py`. Expected: script import failure.

- [ ] **Step 3: Implement verified rotation**

Delete only the exact numbered checkpoint after positive remote verification; append an immutable rotation record containing its SHA256 and remote asset ID.

- [ ] **Step 4: Verify GREEN**

Run the safety tests.

- [ ] **Step 5: Commit**

Commit `feat: rotate only verified FrequencyCM checkpoints`.

### Task 5: Complete local verification and build immutable deployment bundle

**Files:**
- Modify only if a failing regression identifies a root cause.

- [ ] **Step 1: Run focused tests**

Run all new FrequencyCM, FDR integration, protocol, CLI, checkpoint, and resume tests.

- [ ] **Step 2: Run the full repository suite**

Run `pytest -q`. Expected: all existing and new tests pass, with only previously documented skips.

- [ ] **Step 3: Verify source cleanliness and identity**

Run `git diff --check`, `git status --short`, record HEAD and tracked-tree SHA256, and create a Git bundle from the committed branch.

### Task 6: Deploy and execute the RTX 4090 preflight

**Files:**
- Server-only immutable source, protocol, evidence, log, and run directories under `/data/uav`.

- [ ] **Step 1: Verify SSH host key before every mutation**

Require exact ED25519 fingerprint `SHA256:FPVBIMs2LoVe0RenG9xDN5KvN99tgIcdPP9rY8Ym+u8`.

- [ ] **Step 2: Reclaim generated-cache space**

Resolve and remove only frozen IBER/ITBER/P2-oracle cache directories below `/data/uav/cache`; report removed targets and reclaimed space.

- [ ] **Step 3: Deploy to a new immutable source directory**

Transfer the Git bundle, check out exact HEAD, require a clean tree, and generate a new source-bound protocol manifest using the existing validated formal initial state.

- [ ] **Step 4: Run CUDA preflight**

Execute batch-8, 640-pixel, real-VisDrone forward/loss/AMP128 backward/MuSGD/EMA/save/resume checks. Reject NaN/Inf, AMP scale drift, missing optimizer parameters, shared-state mismatch, or wrong hyperparameters.

- [ ] **Step 5: Commit/publish preflight evidence**

Upload lightweight evidence and retain it locally before training.

### Task 7: Launch and supervise fresh formal100

**Files:**
- Server run directory, append-only publication queue, GitHub Release assets, and final evaluation package.

- [ ] **Step 1: Launch one immutable formal run**

Start `fdr_frequencycm`, formal, seed0, fresh initial state, 100 epochs, fixed batch 8, fixed AMP scale 128, and per-epoch saving.

- [ ] **Step 2: Start publication/rotation supervision**

For each completed epoch, publish checkpoint plus SHA manifest, verify the remote asset, rotate only its numbered local checkpoint, and preserve `last.pt`/`best.pt`.

- [ ] **Step 3: Monitor engineering health**

Check PID, GPU utilization, disk, queue lag, finite losses/gradients, validation metrics, and checkpoint/resume integrity. Engineering failures follow systematic debugging and TDD; scientific metrics are never hidden or threshold-adjusted.

- [ ] **Step 4: Complete epoch-100 evaluation**

Run the frozen independent evaluator against strict stock, strict FDR, and FDR+FrequencyCM; publish all paper metrics, complexity/latency/VRAM, SHA256 values, source authority, and final checkpoints.
