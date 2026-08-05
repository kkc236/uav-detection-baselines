# FDR YAML Declarative Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the validated FDR-only RT-DETR-L architecture directly constructible from YAML while preserving the exact formal checkpoint state dictionary and numerical behavior.

**Architecture:** Add a repository-owned `FDRRTDETRDecoder` head declared as the final YAML layer. Its constructor builds the stock Ultralytics head first and then installs the same pre-box, six distribution heads, Integral decoder, and cumulative refinement objects used by the validated implementation, preserving all state keys. Keep FGL and pre-box supervision as YAML-controlled training components and provide standalone single-factor ablation YAML files.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, Ultralytics 8.4.90, YAML, pytest.

---

## File map

- Create `configs/rtdetr-l-fdr.yaml`: full standalone model architecture and FDR loss settings.
- Create `configs/rtdetr-l-fdr-no-fgl.yaml`: FGL-only ablation.
- Create `configs/rtdetr-l-fdr-no-prebox-loss.yaml`: preliminary-loss-only ablation.
- Create `configs/rtdetr-l-fdr-no-cumulative.yaml`: cumulative-refinement-only ablation.
- Create `configs/rtdetr-l-fdr-no-prebox.yaml`: complete preliminary-box functional-unit ablation.
- Modify `src/fdr_head.py`: YAML-instantiable head and ablation-aware decoder strategy.
- Modify `src/rtdetr_fdr.py`: module registration, declarative construction, YAML loss settings, strict compatibility loader.
- Modify `tests/test_fdr_head.py`: functional-unit behavior and default parity tests.
- Modify `tests/test_rtdetr_fdr.py`: YAML parsing, old/new state parity, output parity, checkpoint loading, and ablation isolation.
- Modify `tests/test_train_rtdetr_fdr_cli.py`: trainer default model path contract.
- Create `downloads/fdr-yaml/README.md`: download/package explanation and hashes.

### Task 1: Freeze the executable YAML contract in tests

**Files:**
- Modify: `tests/test_rtdetr_fdr.py`
- Modify: `tests/test_train_rtdetr_fdr_cli.py`

- [ ] **Step 1: Write failing YAML declaration tests**

Add tests that load the full YAML and assert the final layer names the custom head, its three input channels are explicit, and loss settings are frozen:

```python
from pathlib import Path
import yaml

FDR_CFG = Path("configs/rtdetr-l-fdr.yaml")


def test_full_fdr_yaml_declares_the_network_module_and_frozen_loss():
    cfg = yaml.safe_load(FDR_CFG.read_text(encoding="utf-8"))
    assert cfg["head"][-1][2] == "FDRRTDETRDecoder"
    assert cfg["head"][-1][3][1] == [256, 256, 256]
    options = cfg["head"][-1][3][2]
    assert options == {
        "hidden_dim": 256,
        "num_queries": 300,
        "num_decoder_layers": 6,
        "reg_max": 32,
        "reg_scale": 4.0,
        "up": 0.5,
        "cumulative": True,
        "preliminary_box": True,
        "private_seed": 10000,
    }
    assert cfg["fdr_loss"] == {
        "fgl_weight": 0.15,
        "supervise_pre_boxes": True,
    }
```

Add a trainer test requiring `FDRTrainer.get_model()` to default to `configs/rtdetr-l-fdr.yaml`, while `FDRControlTrainer` continues to use `rtdetr-l.yaml`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest tests/test_rtdetr_fdr.py::test_full_fdr_yaml_declares_the_network_module_and_frozen_loss tests/test_train_rtdetr_fdr_cli.py -q
```

Expected: FAIL because `configs/rtdetr-l-fdr.yaml` and declarative trainer default do not exist.

- [ ] **Step 3: Commit the failing contract tests**

```powershell
git add tests/test_rtdetr_fdr.py tests/test_train_rtdetr_fdr_cli.py
git commit -m "test: freeze declarative FDR YAML contract"
```

### Task 2: Add the full standalone FDR model YAML

**Files:**
- Create: `configs/rtdetr-l-fdr.yaml`

- [ ] **Step 1: Copy the stock RT-DETR-L graph and replace only the final layer**

Use the exact Ultralytics RT-DETR-L backbone/head graph. Keep layers 0-27 unchanged and define the final head as:

```yaml
  - [[21, 24, 27], 1, FDRRTDETRDecoder,
     [nc, [256, 256, 256],
      {hidden_dim: 256, num_queries: 300, num_decoder_layers: 6,
       reg_max: 32, reg_scale: 4.0, up: 0.5,
       cumulative: true, preliminary_box: true, private_seed: 10000}]]
```

Add:

```yaml
fdr_loss:
  fgl_weight: 0.15
  supervise_pre_boxes: true
```

- [ ] **Step 2: Run the static YAML test**

Run:

```powershell
pytest tests/test_rtdetr_fdr.py::test_full_fdr_yaml_declares_the_network_module_and_frozen_loss -q
```

Expected: PASS for the static contract; model construction tests remain absent/failing.

- [ ] **Step 3: Commit**

```powershell
git add configs/rtdetr-l-fdr.yaml
git commit -m "config: declare full FDR RT-DETR-L architecture"
```

### Task 3: Implement the YAML-instantiable FDR head

**Files:**
- Modify: `tests/test_fdr_head.py`
- Modify: `tests/test_rtdetr_fdr.py`
- Modify: `src/fdr_head.py`

- [ ] **Step 1: Write failing head construction and state-parity tests**

Add a legacy helper that performs the existing post-construction replacement and compare it with the desired YAML model:

```python
def _legacy_injected_fdr(seed=0, private_seed=10_000):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RTDETRDetectionModel("rtdetr-l.yaml", nc=10, verbose=False)
    head = model.model[-1]
    pre = head.dec_bbox_head[0]
    heads = build_distribution_heads(head.hidden_dim, head.num_decoder_layers, private_seed=private_seed)
    head.decoder = FDRDeformableTransformerDecoder.from_stock(head.decoder, pre_bbox_head=pre)
    head.dec_bbox_head = heads
    head.decoder.reg_max = REG_MAX
    head.decoder.final_layers = [module.layers[-1] for module in heads]
    return model


def test_yaml_head_matches_legacy_state_keys_shapes_and_initial_values():
    legacy = _legacy_injected_fdr()
    declarative = FDRRTDETRDetectionModel(FDR_CFG, nc=10, verbose=False)
    assert legacy.state_dict().keys() == declarative.state_dict().keys()
    for key, expected in legacy.state_dict().items():
        torch.testing.assert_close(declarative.state_dict()[key], expected, rtol=0, atol=0)
```

Add a test asserting `type(model.model[-1]).__name__ == "FDRRTDETRDecoder"` and that public RNG advancement equals stock construction.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
pytest tests/test_rtdetr_fdr.py -k "yaml_head or declarative" -q
```

Expected: FAIL because `FDRRTDETRDecoder` is not registered or implemented.

- [ ] **Step 3: Implement `FDRRTDETRDecoder` minimally**

In `src/fdr_head.py`, subclass the stock head and preserve constructor order and object paths:

```python
class FDRRTDETRDecoder(RTDETRDecoder):
    def __init__(self, nc=80, ch=(256, 256, 256), options=None):
        options = dict(options or {})
        hidden_dim = int(options.get("hidden_dim", 256))
        num_queries = int(options.get("num_queries", 300))
        num_layers = int(options.get("num_decoder_layers", 6))
        private_seed = int(options.get("private_seed", 10_000))
        super().__init__(nc=nc, ch=tuple(ch), hd=hidden_dim, nq=num_queries, ndl=num_layers)
        stock_pre_bbox_head = self.dec_bbox_head[0]
        distribution_heads = build_distribution_heads(
            hidden_dim, num_layers, private_seed=private_seed
        )
        self.decoder = FDRDeformableTransformerDecoder.from_stock(
            self.decoder,
            pre_bbox_head=stock_pre_bbox_head,
            reg_max=int(options.get("reg_max", REG_MAX)),
            reg_scale=float(options.get("reg_scale", REG_SCALE)),
            up=float(options.get("up", UP)),
            cumulative=bool(options.get("cumulative", True)),
            preliminary_box=bool(options.get("preliminary_box", True)),
        )
        self.dec_bbox_head = distribution_heads
        self.decoder.reg_max = int(options.get("reg_max", REG_MAX))
        self.decoder.final_layers = [module.layers[-1] for module in distribution_heads]
        self.fdr_options = options
```

Do not add wrapper module levels around `pre_bbox_head`, `dec_bbox_head`, or `integral`; that would change formal checkpoint keys.

- [ ] **Step 4: Register the module without editing Ultralytics**

Add to `src/rtdetr_fdr.py`:

```python
from ultralytics.nn import tasks as ultralytics_tasks
from src.fdr_head import FDRRTDETRDecoder


def register_fdr_module() -> None:
    ultralytics_tasks.FDRRTDETRDecoder = FDRRTDETRDecoder
```

Call `register_fdr_module()` immediately before `super().__init__()` in `FDRRTDETRDetectionModel`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
pytest tests/test_fdr_head.py tests/test_rtdetr_fdr.py -k "yaml_head or declarative or public_rng" -q
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/fdr_head.py src/rtdetr_fdr.py tests/test_fdr_head.py tests/test_rtdetr_fdr.py
git commit -m "feat: construct FDR decoder directly from YAML"
```

### Task 4: Remove post-construction injection and read loss settings from YAML

**Files:**
- Modify: `tests/test_rtdetr_fdr.py`
- Modify: `src/rtdetr_fdr.py`

- [ ] **Step 1: Write failing no-injection and loss-configuration tests**

```python
def test_model_uses_yaml_head_without_post_construction_replacement(monkeypatch):
    monkeypatch.setattr(
        "src.rtdetr_fdr.build_distribution_heads",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy injection called")),
    )
    model = FDRRTDETRDetectionModel(FDR_CFG, nc=10, verbose=False)
    assert isinstance(model.model[-1], FDRRTDETRDecoder)


def test_criterion_reads_frozen_yaml_loss_options():
    model = FDRRTDETRDetectionModel(FDR_CFG, nc=10, verbose=False)
    criterion = model.init_criterion()
    assert criterion.fgl_weight == 0.15
    assert criterion.supervise_pre_boxes is True
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
pytest tests/test_rtdetr_fdr.py -k "without_post or criterion_reads" -q
```

Expected: FAIL because the old constructor still replaces the head and the loss is hard-coded.

- [ ] **Step 3: Simplify the model constructor**

Replace the injection block with validation only:

```python
register_fdr_module()
super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
head = self.model[-1]
if not isinstance(head, FDRRTDETRDecoder):
    raise TypeError("FDR model YAML must end with FDRRTDETRDecoder")
self.private_seed = int(head.fdr_options["private_seed"])
self.fdr_loss_options = dict(self.yaml.get("fdr_loss", {}))
```

Build the criterion from the YAML:

```python
return FDRDetectionLoss(
    nc=self.nc,
    use_vfl=True,
    fgl_weight=float(self.fdr_loss_options.get("fgl_weight", 0.15)),
    supervise_pre_boxes=bool(self.fdr_loss_options.get("supervise_pre_boxes", True)),
)
```

- [ ] **Step 4: Run and verify GREEN**

Run:

```powershell
pytest tests/test_rtdetr_fdr.py -k "without_post or criterion_reads" -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/rtdetr_fdr.py tests/test_rtdetr_fdr.py
git commit -m "refactor: remove FDR post-build head injection"
```

### Task 5: Add isolated ablation behavior and YAML files

**Files:**
- Modify: `tests/test_fdr_head.py`
- Modify: `tests/test_rtdetr_fdr.py`
- Modify: `src/fdr_head.py`
- Create: `configs/rtdetr-l-fdr-no-fgl.yaml`
- Create: `configs/rtdetr-l-fdr-no-prebox-loss.yaml`
- Create: `configs/rtdetr-l-fdr-no-cumulative.yaml`
- Create: `configs/rtdetr-l-fdr-no-prebox.yaml`

- [ ] **Step 1: Write failing strategy tests**

Test that disabling cumulative refinement uses only the current layer logits, and disabling preliminary-box routing uses the incoming reference while retaining state keys:

```python
def test_no_cumulative_uses_each_layer_distribution_without_history():
    deltas = torch.stack([torch.ones(FDR_OUTPUT_DIM), torch.full((FDR_OUTPUT_DIM,), 2.0)])
    actual = combine_distribution_logits(deltas, cumulative=False)
    torch.testing.assert_close(actual, deltas)


def test_no_prebox_preserves_state_contract_but_disables_prebox_routing():
    full = FDRRTDETRDetectionModel("configs/rtdetr-l-fdr.yaml", nc=10, verbose=False)
    ablated = FDRRTDETRDetectionModel("configs/rtdetr-l-fdr-no-prebox.yaml", nc=10, verbose=False)
    assert full.state_dict().keys() == ablated.state_dict().keys()
    assert ablated.fdr.preliminary_box is False
```

Add a YAML diff test that recursively compares each ablation with the full YAML and permits only its named field change. For `no-prebox`, permit `preliminary_box: false`; the criterion must automatically suppress pre-box loss for that functional-unit ablation.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
pytest tests/test_fdr_head.py tests/test_rtdetr_fdr.py -k "cumulative or prebox or ablation" -q
```

Expected: FAIL because switches and ablation YAML files do not exist.

- [ ] **Step 3: Add strategy switches without changing default behavior**

Extend `FDRDeformableTransformerDecoder` with plain, non-parameter attributes:

```python
self.cumulative = bool(cumulative)
self.preliminary_box = bool(preliminary_box)
```

Inside the loop use:

```python
delta = bbox_head[index](output + output_detach)
cumulative_corners = delta + cumulative_corners if self.cumulative else delta
if index == 0:
    preliminary = torch.sigmoid(self.pre_bbox_head(output) + inverse_sigmoid(reference))
    initial_reference = preliminary.detach() if self.preliminary_box else reference.detach()
```

Always retain `pre_bbox_head`, its key paths, and its computed evidence so an existing formal checkpoint loads into every ablation model.

- [ ] **Step 4: Create the four ablation YAML files**

Each file is a standalone copy of the full graph. Change exactly:

```yaml
# no-fgl
fdr_loss: {fgl_weight: 0.0, supervise_pre_boxes: true}

# no-prebox-loss
fdr_loss: {fgl_weight: 0.15, supervise_pre_boxes: false}

# no-cumulative
cumulative: false

# no-prebox
preliminary_box: false
```

When `preliminary_box` is false, `init_criterion()` sets effective `supervise_pre_boxes=False` even though the stored YAML loss block remains unchanged, because the complete preliminary-box functional unit is disabled.

- [ ] **Step 5: Run and verify GREEN**

Run:

```powershell
pytest tests/test_fdr_head.py tests/test_rtdetr_fdr.py -k "cumulative or prebox or ablation" -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/fdr_head.py src/rtdetr_fdr.py tests/test_fdr_head.py tests/test_rtdetr_fdr.py configs/rtdetr-l-fdr*.yaml
git commit -m "feat: add isolated FDR ablation configurations"
```

### Task 6: Prove old/new numerical parity and formal checkpoint compatibility

**Files:**
- Modify: `tests/test_rtdetr_fdr.py`
- Create: `scripts/verify_fdr_yaml_checkpoint.py`

- [ ] **Step 1: Write failing strict-checkpoint and output-parity tests**

The output test loads the legacy state into the declarative model and compares both training and evaluation contracts:

```python
def test_declarative_full_model_is_numerically_equal_to_legacy_injection():
    legacy = _legacy_injected_fdr()
    declarative = FDRRTDETRDetectionModel(FDR_CFG, nc=10, verbose=False)
    declarative.load_state_dict(legacy.state_dict(), strict=True)
    image = torch.zeros(1, 3, 128, 128)
    legacy.eval()
    declarative.eval()
    with torch.no_grad():
        expected = legacy(image)
        actual = declarative(image)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for left, right in zip(actual[1][:-1], expected[1][:-1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
```

The verification script must extract `artifact["model"].state_dict()` from the formal checkpoint, build the YAML model, call `load_state_dict(..., strict=True)`, run one deterministic inference, and emit JSON containing checkpoint SHA256, missing/unexpected key counts, tensor count, and output finiteness.

- [ ] **Step 2: Run parity test and verify RED if any constructor detail differs**

Run:

```powershell
pytest tests/test_rtdetr_fdr.py -k "numerically_equal or formal_checkpoint" -q
```

Expected before final compatibility adjustments: FAIL on any differing key, shape, or output.

- [ ] **Step 3: Make only compatibility corrections**

Correct constructor argument defaults, registration order, buffer dtypes, or `private_seed` flow. Do not rename state paths and do not add compatibility aliases that duplicate parameters.

- [ ] **Step 4: Run local parity tests**

Run:

```powershell
pytest tests/test_rtdetr_fdr.py -k "numerically_equal or state_keys or formal_checkpoint" -q
```

Expected: PASS with zero missing and zero unexpected keys.

- [ ] **Step 5: Verify the actual epoch-100 artifact**

Run:

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
gh release download fdr-formal-d97e1eb7-live --repo kkc236/uav-detection-baselines --pattern 'fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt' --dir artifacts
python scripts/verify_fdr_yaml_checkpoint.py --cfg configs/rtdetr-l-fdr.yaml --checkpoint artifacts/fdr-formal-seed0-fdr-d97e1eb7-epoch-0100.pt --output artifacts/fdr-yaml-checkpoint-compatibility.json
```

Expected JSON fields:

```json
{
  "strict_load": true,
  "missing_keys": 0,
  "unexpected_keys": 0,
  "finite_output": true
}
```

- [ ] **Step 6: Commit**

```powershell
git add scripts/verify_fdr_yaml_checkpoint.py tests/test_rtdetr_fdr.py artifacts/fdr-yaml-checkpoint-compatibility.json
git commit -m "test: prove formal FDR checkpoint YAML compatibility"
```

### Task 7: Switch the FDR trainer default and run the full regression suite

**Files:**
- Modify: `src/rtdetr_fdr.py`
- Modify: `scripts/train_rtdetr_fdr.py`
- Modify: `tests/test_train_rtdetr_fdr_cli.py`

- [ ] **Step 1: Write the failing trainer-default test**

Require an FDR run with no explicit model cfg to receive `configs/rtdetr-l-fdr.yaml`, and require the control arm to remain stock.

- [ ] **Step 2: Run and verify RED**

```powershell
pytest tests/test_train_rtdetr_fdr_cli.py -q
```

Expected: FAIL on the old `rtdetr-l.yaml` FDR default.

- [ ] **Step 3: Change only the FDR default**

Use:

```python
FDR_MODEL_CFG = Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr.yaml"
model = FDRRTDETRDetectionModel(cfg or FDR_MODEL_CFG, ...)
```

Do not change the control trainer default.

- [ ] **Step 4: Run the complete FDR test suite**

```powershell
pytest tests/test_fdr_authority.py tests/test_fdr_math.py tests/test_fdr_head.py tests/test_fdr_loss.py tests/test_fdr_protocol.py tests/test_fdr_runtime_preflight.py tests/test_fdr_preflight.py tests/test_rtdetr_fdr.py tests/test_train_rtdetr_fdr_cli.py -q
```

Expected: all tests PASS with zero failures.

- [ ] **Step 5: Run an executable model smoke test**

```powershell
python -c "from src.rtdetr_fdr import FDRRTDETRDetectionModel; m=FDRRTDETRDetectionModel('configs/rtdetr-l-fdr.yaml',nc=10,verbose=False); print(type(m.model[-1]).__name__)"
```

Expected: `FDRRTDETRDecoder`.

- [ ] **Step 6: Commit**

```powershell
git add src/rtdetr_fdr.py scripts/train_rtdetr_fdr.py tests/test_train_rtdetr_fdr_cli.py
git commit -m "refactor: make declarative YAML the FDR training entrypoint"
```

### Task 8: Package downloadable YAML and documentation

**Files:**
- Create: `downloads/fdr-yaml/README.md`
- Copy release files into: `downloads/fdr-yaml/`

- [ ] **Step 1: Write the download README**

Document each YAML, functional-unit boundaries, full-run parameters, checkpoint compatibility result, and the fact that `src/fdr_head.py`, `src/fdr_math.py`, `src/fdr_loss.py`, and `src/rtdetr_fdr.py` are required runtime code.

- [ ] **Step 2: Copy the five verified YAML files into the download folder**

Use the repository versions as the only source. Do not maintain divergent hand-edited copies.

- [ ] **Step 3: Generate SHA256 hashes**

Run:

```powershell
Get-FileHash -Algorithm SHA256 downloads/fdr-yaml/*.yaml
```

Expected: five unique hash rows.

- [ ] **Step 4: Verify the final worktree**

Run:

```powershell
git diff --check
git status --short
pytest tests/test_fdr_authority.py tests/test_fdr_math.py tests/test_fdr_head.py tests/test_fdr_loss.py tests/test_fdr_protocol.py tests/test_fdr_runtime_preflight.py tests/test_fdr_preflight.py tests/test_rtdetr_fdr.py tests/test_train_rtdetr_fdr_cli.py -q
```

Expected: no whitespace errors and zero test failures.

- [ ] **Step 5: Commit the delivery package**

```powershell
git add downloads/fdr-yaml
git commit -m "docs: package executable FDR YAML configurations"
```
