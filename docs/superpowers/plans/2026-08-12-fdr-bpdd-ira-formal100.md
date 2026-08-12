# FDR + BPDD + IRA Formal100 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a YAML-pluggable IRA layer for the mature RT-DETR-L FDR+BPDD detector, prove its isolated contracts, then run one fresh seed-0 full-data 100-epoch arm with per-epoch validation and publication.

**Architecture:** A repository-owned IRA module is inserted only after the 256-channel P3 RepC3 output. The existing FDR decoder and training-only BPDD criterion are reused without mathematical changes. A dedicated model/trainer, protocol manifest, preflight, and launch path keep the new arm isolated from the completed FDR and BPDD authorities.

**Tech Stack:** Python 3.10.12, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.90, PyYAML, pytest, RTX 4090, Git/GitHub Release evidence publication.

---

## File structure

- Create `src/ira.py`: IRA body, zero-initialized outer residual gate, private parameter helpers.
- Create `configs/rtdetr-l-fdr-bpdd-ira.yaml`: declarative P3 insertion plus unchanged FDR/BPDD options.
- Create `src/rtdetr_fdr_bpdd_ira.py`: parser registration, combined model, trainer, strict initial-state loading, three gradient groups.
- Create `src/bpdd_ira_protocol.py`: immutable source/run identity and combined-arm protocol.
- Create `scripts/prepare_bpdd_ira_protocol.py`: create-only manifest generation.
- Create `scripts/train_rtdetr_bpdd_ira.py`: formal-only launch, per-epoch evidence and publication queue.
- Create `scripts/run_bpdd_ira_preflight.py`: authority and real batch-8 CUDA preflight.
- Create focused tests under `tests/test_ira_*.py` and `tests/test_bpdd_ira_*.py`.
- Reuse `scripts/sync_experiment_checkpoint.py` for non-blocking checkpoint publication.

### Task 1: IRA mathematical and identity contract

**Files:**
- Create: `tests/test_ira_module.py`
- Create: `src/ira.py`

- [ ] **Step 1: Write failing identity, shape, gradient, and seed tests**

```python
def test_ira_starts_as_exact_identity():
    module = IRA(256)
    x = torch.randn(2, 256, 16, 16)
    torch.testing.assert_close(module(x), x, rtol=0, atol=0)

def test_ira_private_path_receives_gradients_when_gate_is_open():
    module = IRA(256)
    module.residual_scale.data.fill_(0.1)
    module(torch.randn(2, 256, 16, 16)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/test_ira_module.py -q`

Expected: collection fails because `src.ira` does not exist.

- [ ] **Step 3: Implement the minimal module**

```python
class IRA(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            IRABaseBlock(channels),
            IRABaseBlock(channels),
            IRAAttention(channels),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.residual_scale * (self.refine(x) - x)
```

Implement `IRABaseBlock` with 1x1, depth-wise 3x3, 1x1 and two internal residuals; implement `IRAAttention` with spatial mean/max attention and channel mean attention. Validate `channels > 0` and preserve `[B,C,H,W]` exactly.

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `python -m pytest tests/test_ira_module.py -q`

Expected: all IRA unit tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ira.py tests/test_ira_module.py
git commit -m "feat: add identity-safe IRA feature module"
```

### Task 2: Declarative graph and combined model

**Files:**
- Create: `tests/test_bpdd_ira_yaml_config.py`
- Create: `tests/test_bpdd_ira_integration.py`
- Create: `configs/rtdetr-l-fdr-bpdd-ira.yaml`
- Create: `src/rtdetr_fdr_bpdd_ira.py`
- Modify: `src/rtdetr_fdr.py`

- [ ] **Step 1: Write failing graph-isolation tests**

```python
def test_combined_yaml_changes_only_p3_and_adds_no_new_loss():
    bpdd = load_yaml("configs/rtdetr-l-fdr-bpdd.yaml")
    combined = load_yaml("configs/rtdetr-l-fdr-bpdd-ira.yaml")
    assert combined["fdr_loss"] == bpdd["fdr_loss"]
    assert combined["bpdd_loss"] == bpdd["bpdd_loss"]
    assert sum(row[2] == "IRA" for row in combined["head"]) == 1
    assert combined["head"][-1][0][0] == 22

def test_zero_gate_combined_eval_matches_bpdd_exactly():
    bpdd, combined = paired_models()
    load_shared_state_exact(combined, bpdd.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(combined(image)[0], bpdd(image)[0], rtol=0, atol=0)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/test_bpdd_ira_yaml_config.py tests/test_bpdd_ira_integration.py -q`

Expected: missing YAML/model failures.

- [ ] **Step 3: Add the YAML and combined integration**

Insert after P3 RepC3:

```yaml
  - [-1, 1, IRA, [256]]
```

Shift the subsequent P4/P5 indices and feed `[22, 25, 28]` to the unchanged `FDRRTDETRDecoder`. Register `IRA` beside `FDRRTDETRDecoder` in the locked parser section. Implement `FDRBPDDIRADetectionModel(FDRBPDDDetectionModel)` and `FDRBPDDIRATrainer(FDRBPDDTrainer)`. Load the frozen FDR initial state into common/FDR tensors with exact equality; initialize only names under the single IRA layer with deterministic seed `20000 + experiment_seed`. Partition gradients into `gradient_norm`, `fdr_gradient_norm`, and `ira_gradient_norm`, each clipped independently at 10.

- [ ] **Step 4: Run focused and legacy tests**

Run: `python -m pytest tests/test_ira_module.py tests/test_bpdd_ira_yaml_config.py tests/test_bpdd_ira_integration.py tests/test_bpdd_fdr_integration.py tests/test_bpdd_yaml_config.py -q`

Expected: all tests pass and legacy FDR/BPDD state contracts remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add configs/rtdetr-l-fdr-bpdd-ira.yaml src/rtdetr_fdr.py src/rtdetr_fdr_bpdd_ira.py tests/test_bpdd_ira_yaml_config.py tests/test_bpdd_ira_integration.py
git commit -m "feat: integrate IRA with FDR BPDD graph"
```

### Task 3: Immutable protocol and formal trainer

**Files:**
- Create: `tests/test_bpdd_ira_protocol.py`
- Create: `tests/test_train_rtdetr_bpdd_ira_cli.py`
- Create: `src/bpdd_ira_protocol.py`
- Create: `scripts/prepare_bpdd_ira_protocol.py`
- Create: `scripts/train_rtdetr_bpdd_ira.py`

- [ ] **Step 1: Write failing protocol and CLI tests**

```python
def test_formal_settings_are_frozen():
    settings = build_settings(args, data_yaml)
    assert settings["epochs"] == 100
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["pretrained"] is False
    assert settings["optimizer"] == "MuSGD"
    assert settings["nms"] is False
    assert settings["save_period"] == 1

def test_protocol_has_one_combined_formal_identity():
    identity = build_run_identity(source, stage="formal", variant="fdr_bpdd_ira", seed=0)
    assert identity["variant"] == "fdr_bpdd_ira"
    assert identity["initial_state_sha256"] == FDR_INITIAL_STATE_SHA256
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_bpdd_ira_protocol.py tests/test_train_rtdetr_bpdd_ira_cli.py -q`

Expected: imports fail because dedicated protocol/CLI files do not exist.

- [ ] **Step 3: Implement protocol, manifest, and trainer CLI**

Reuse the existing frozen `FROZEN_SETTINGS`, category mapping, dataset signature, create-only JSONL functions, and fixed AMP trainer. Restrict the CLI to `--stage formal` and `variant=fdr_bpdd_ira`. Add these evidence fields:

```python
EVIDENCE_FIELDS = (*BPDD_EVIDENCE_FIELDS, "ira_gradient_norm", "ira_residual_scale")
```

The epoch callback must append exactly one record for each completed epoch, include checkpoint and EMA SHA256, and append a publication record keyed by `(run_id, completed_epoch)` without mutation.

- [ ] **Step 4: Run focused CLI/protocol tests**

Run: `python -m pytest tests/test_bpdd_ira_protocol.py tests/test_train_rtdetr_bpdd_ira_cli.py tests/test_bpdd_protocol.py tests/test_train_rtdetr_bpdd_cli.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bpdd_ira_protocol.py scripts/prepare_bpdd_ira_protocol.py scripts/train_rtdetr_bpdd_ira.py tests/test_bpdd_ira_protocol.py tests/test_train_rtdetr_bpdd_ira_cli.py
git commit -m "feat: add immutable BPDD IRA formal protocol"
```

### Task 4: Real CUDA preflight and regression closure

**Files:**
- Create: `tests/test_bpdd_ira_preflight.py`
- Create: `scripts/run_bpdd_ira_preflight.py`

- [ ] **Step 1: Write failing fail-closed preflight tests**

```python
def test_preflight_requires_all_gates():
    decision = run_preflight(context, gate_runners=passing_runners())
    assert decision["gate_states"] == {f"I{i}": "passed" for i in range(5)}
    assert decision["formal_eligible"] is True

def test_preflight_stops_after_shared_state_failure():
    runners = passing_runners(); runners["I1"] = failed_runner
    decision = run_preflight(context, gate_runners=runners)
    assert decision["gate_states"]["I2"] == "blocked"
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_bpdd_ira_preflight.py -q`

Expected: missing preflight implementation.

- [ ] **Step 3: Implement five gates**

Implement create-only reports for:

- `I0`: source, environment, GPU, dataset, YAML, and initial-state authority;
- `I1`: exact shared/FDR initial-state equality and IRA-only private keys;
- `I2`: real train batch8 CUDA forward/backward, finite losses/BPDD activity/three gradient groups, AMP128 and one MuSGD step;
- `I3`: 300-query eval output and zero-gate parity against FDR+BPDD before the disposable step;
- `I4`: combined checkpoint round trip and resume authority.

- [ ] **Step 4: Run the complete local suite**

Run: `python -m pytest tests/test_ira_module.py tests/test_bpdd_ira_*.py tests/test_bpdd_*.py tests/test_fdr_*.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_bpdd_ira_preflight.py tests/test_bpdd_ira_preflight.py
git commit -m "test: add BPDD IRA formal preflight"
```

### Task 5: Deploy and start Formal100

**Files:**
- Create on server: `/data/uav/protocols/bpdd-ira-<source>/protocol.json`
- Create on server: `/data/uav/preflight/bpdd-ira-<source>/decision.json`
- Create on server: `/data/uav/runs/bpdd-ira-formal-<source>/formal-seed0-fdr_bpdd_ira-v1/`
- Create on server: `/data/uav/publication/bpdd-ira-formal-<source>/queue.jsonl`

- [ ] **Step 1: Verify clean source and build immutable bundle**

Run: `git status --short`, `git rev-parse HEAD`, source-tree SHA256 calculation, and `git bundle create ... HEAD`.

Expected: only known user-owned untracked files remain; committed source has a stable commit and tree hash.

- [ ] **Step 2: Verify SSH host identity and deploy**

Require ED25519 fingerprint `SHA256:FPVBIMs2LoVe0RenG9xDN5KvN99tgIcdPP9rY8Ym+u8`. Transfer the bundle, clone to a new `/data/uav/source/uav-detection-baselines-<commit>` directory, and never modify an older source directory.

- [ ] **Step 3: Run server tests and real CUDA preflight**

Run the focused pytest suite in `/data/uav/venvs/iber-be-v1`, create the immutable protocol manifest, then run all `I0-I4` gates.

Expected: tests pass; preflight reports `formal_eligible=true`.

- [ ] **Step 4: Launch training and publication independently**

Start the trainer hidden in the background with PID/log files and start the existing publication synchronizer in a separate process. The trainer command must include the frozen protocol manifest, initial state, dataset root, output root, and publication queue. The synchronizer must target release tag `bpdd-ira-formal-<source>-live` and may retry without blocking training.

- [ ] **Step 5: Verify real progress**

Require a live PID, non-idle GPU, finite first batches, generated run identity, and—after epoch 1—one create-only checkpoint, one metrics row, and one publication queue row. Record these paths in the pipeline state file.

### Task 6: Complete, independently validate, and publish

**Files:**
- Create: `scripts/evaluate_bpdd_ira_formal.py`
- Create: `tests/test_bpdd_ira_formal_evaluation.py`
- Create on server: `/data/uav/reports/bpdd-ira-formal-<source>/final-report.json`

- [ ] **Step 1: TDD the final evaluator before epoch 100**

Test that it requires exactly 100 immutable epoch records, loads epoch-100 EMA into the combined graph, rejects test YAMLs, evaluates the fixed 548-image val split, emits all global/scale/class metrics, and marks comparisons as `preliminary_cross_run`.

- [ ] **Step 2: Monitor and repair without changing science**

Check PID/GPU, finite loss/AMP/gradients, disk, checkpoint integrity, publication lag, and resume identity. Engineering failures receive a regression test and resume from the verified `last.pt`; scientific settings never change.

- [ ] **Step 3: Run independent epoch-100 validation and efficiency audit**

Report Precision, Recall, F1, AP50, AP75, mAP50-95, size diagnostics, ten-class AP/AP50/AP75, parameters, GFLOPs, FP16 median/P95 latency, throughput, peak memory, and checkpoint/EMA SHA256.

- [ ] **Step 4: Upload and verify evidence**

Require 100/100 checkpoint+manifest publication entries and upload the final report/logs. Verify remote asset sizes and SHA256 where the API exposes them.

- [ ] **Step 5: Commit final evidence metadata**

```bash
git add docs evidence research
git commit -m "evidence: publish FDR BPDD IRA formal100 result"
```

Do not mark completion until the 100 epochs, independent validation, efficiency report, and remote publication are all verified.
