# OAR All-Pair Amendment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce and freeze the sparse D0 failure, then test an all-3,000-pair objective-aligned ranker and, only when authorized, a 300-Query class-conditional set ranker.

**Architecture:** OAR-R2 applies a zero-initialized 276-to-64-to-1 MLP independently to every Query-by-class pair and directly optimizes final adjusted-logit ordering. OAR-QS2 replaces independent pair processing with one 300-token Query transformer whose output has ten class residuals per Query; no sparse pool or FDR is used.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, pytest, CUDA 12.1, RTX 4090, immutable JSON/SHA-256 artifacts.

---

## Supersession map

- Task 1 of `2026-08-04-objective-aligned-reranker-offline.md` remains required to
  reproduce D0 and has already been implemented.
- This plan replaces Tasks 2 through 4 and the sparse-pool portions of Tasks 6 through 9
  in that plan.
- Cache safety, authority binding, deterministic checkpoints, internal Gate, one-shot
  official validation, server identity, and publication requirements from Tasks 5
  through 9 remain mandatory and are restated where they affect the all-pair design.

### Task 1: Publish the frozen sparse-D0 failure

**Files:**
- Modify: `scripts/run_rtdetr_oar.py`
- Test: `tests/test_run_rtdetr_oar.py`

- [ ] **Step 1: Write a failing report test using the observed metrics**

```python
def test_sparse_d0_fails_without_extending_the_grid():
    report = select_candidate_k(
        stock_map=0.28628865801344866,
        full_map=0.409733588907,
        restricted_map={
            20: 0.304549967436,
            40: 0.335016751522,
            60: 0.358000690130,
            100: 0.385568106152,
        },
    )
    assert report["status"] == "scientific_failed"
    assert report["selected_k"] is None
```

- [ ] **Step 2: Run the test and verify runner-report failure**

Run: `python -m pytest tests/test_run_rtdetr_oar.py -k sparse_d0 -q`

Expected: FAIL because the runner report is absent.

- [ ] **Step 3: Add D0 decomposition output without a training transition**

The runner evaluates stock, presence, query-IoU, same-class, and K 20/40/60/100 from
the verified 129-image cache. It writes canonical `d0-oracle-decomposition.json`,
`d0-k-coverage.json`, and `sparse-d0-decision.json`. The sparse decision is terminal for
the parent design but explicitly hands authority to this amendment; it cannot select a
new K.

- [ ] **Step 4: Run and commit**

Run: `python -m pytest tests/test_run_rtdetr_oar.py -k sparse_d0 -q`

Expected: PASS with exact observed metrics accepted within `1e-12` report serialization
tolerance and exact decimal Gate failure.

```bash
git add scripts/run_rtdetr_oar.py tests/test_run_rtdetr_oar.py
git commit -m "results: freeze sparse OAR D0 failure"
```

### Task 2: Implement all-pair OAR-R2

**Files:**
- Modify: `src/rtdetr_oar.py`
- Modify: `tests/test_rtdetr_oar.py`

- [ ] **Step 1: Write failing all-pair identity and gradient tests**

```python
def test_oar_r2_adjusts_all_pairs_and_starts_as_exact_stock():
    model = OARRanker()
    features = torch.randn(2, 300, 10, 276)
    logits = torch.randn(2, 300, 10)
    adjusted, residual = apply_oar_r2(model, features, logits)
    assert adjusted.shape == residual.shape == (2, 300, 10)
    assert torch.equal(adjusted, logits.sigmoid())
    assert torch.count_nonzero(residual) == 0


def test_oar_r2_detaches_every_evidence_input():
    features = torch.randn(1, 300, 10, 276, requires_grad=True)
    logits = torch.randn(1, 300, 10, requires_grad=True)
    model = OARRanker()
    adjusted, _ = apply_oar_r2(model, features, logits)
    adjusted.sum().backward()
    assert features.grad is None
    assert logits.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())
```

- [ ] **Step 2: Run and confirm missing-symbol failure**

Run: `python -m pytest tests/test_rtdetr_oar.py -k r2 -q`

Expected: FAIL because all-pair application is absent.

- [ ] **Step 3: Implement exact all-pair application**

```python
class OARRanker(nn.Module):
    def __init__(self, feature_dim=276, width=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, width), nn.SiLU(), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        raw = self.network(features.detach()).squeeze(-1)
        return 2.0 * torch.tanh(raw / 2.0)


def apply_oar_r2(model, features, stock_logits):
    residual = model(features)
    adjusted_logits = stock_logits.detach() + residual
    return adjusted_logits.sigmoid(), residual
```

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tests/test_rtdetr_oar.py -k 'r2 or residual' -q`

Expected: PASS with exactly 17,793 parameters, residual range `[-2,2]`, full 3,000-pair
coverage, and no detector-evidence gradients.

```bash
git add src/rtdetr_oar.py tests/test_rtdetr_oar.py
git commit -m "feat: add all-pair OAR ranker"
```

### Task 3: Implement all-pair Top-300 boundary RankNet

**Files:**
- Modify: `src/rtdetr_oar.py`
- Modify: `tests/test_rtdetr_oar.py`

- [ ] **Step 1: Write failing deterministic pair tests**

Use 3,000-element teacher and stock utilities with a known rank rotation. Assert the
pair list contains at most 2,647 unique oriented pairs, includes candidates outside the
old Top-100-per-class pool, excludes exact teacher ties, and is byte-identical across
repeated construction.

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_rtdetr_oar.py -k 'boundary_pairs or rank_loss' -q`

Expected: FAIL because all-pair construction is absent.

- [ ] **Step 3: Implement the exact all-pair loss**

```python
def teacher_utility(stock_logits, quality):
    return stock_logits.detach().sigmoid() * quality.detach().square()


def boundary_rank_loss(adjusted_logits, pairs):
    flat = adjusted_logits.flatten()
    difference = flat[pairs.preferred] - flat[pairs.other]
    element = F.softplus(-difference) * pairs.weight
    return element.sum() / pairs.weight.sum().clamp_min(torch.finfo(element.dtype).eps)
```

Pair construction uses the parent design's exact 2,048 symmetric-difference pairs,
299 teacher-adjacent pairs, and 300 teacher-rank-offset pairs, but has no pool-mask
filter. Stable rank order, tie removal, orientation, deduplication, and teacher-gap
weights remain unchanged.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tests/test_rtdetr_oar.py -q`

Expected: PASS for all D0, model, pair, tie, cap, loss-ordering, and gradient tests.

```bash
git add src/rtdetr_oar.py tests/test_rtdetr_oar.py
git commit -m "feat: train OAR on all-pair ranking"
```

### Task 4: Implement conditional OAR-QS2

**Files:**
- Modify: `src/rtdetr_oar.py`
- Modify: `tests/test_rtdetr_oar.py`

- [ ] **Step 1: Write failing 300-token/10-class tests**

```python
def test_qs2_uses_300_tokens_and_returns_ten_class_residuals():
    model = OARQuerySetRanker()
    features = torch.randn(2, 300, 275)
    residual = model(features)
    assert residual.shape == (2, 300, 10)
    assert torch.count_nonzero(residual) == 0


def test_qs2_never_attends_across_images():
    model = OARQuerySetRanker().eval()
    x = torch.randn(2, 300, 275)
    with torch.inference_mode():
        first = model.encode_before_output(x)
        changed = x.clone(); changed[1] += 100
        second = model.encode_before_output(changed)
    assert torch.equal(first[0], second[0])
```

- [ ] **Step 2: Run and confirm missing-model failure**

Run: `python -m pytest tests/test_rtdetr_oar.py -k qs2 -q`

Expected: FAIL because `OARQuerySetRanker` is absent.

- [ ] **Step 3: Implement the exact Query-set architecture**

```python
class OARQuerySetRanker(nn.Module):
    def __init__(self, feature_dim=275, width=64, heads=4, classes=10):
        super().__init__()
        self.input = nn.Linear(feature_dim, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=128,
            activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Linear(width, classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode_before_output(self, features):
        return self.encoder(self.input(features.detach()))

    def forward(self, features):
        raw = self.output(self.encode_before_output(features))
        return 2.0 * torch.tanh(raw / 2.0)
```

- [ ] **Step 4: Add exact Query-token features**

Create `oar_query_features(boxes, logits, hidden)` returning `[B,300,275]`: detached
hidden 256, detached normalized geometry 8, all ten detached pre-sigmoid class logits,
and mean Bernoulli entropy 1. Validate shapes, float32 conversion, devices, and finiteness.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_rtdetr_oar.py -q`

Expected: PASS for token/output shapes, exact stock initialization, no cross-image
attention, detached inputs, deterministic eval, and all-pair loss compatibility.

```bash
git add src/rtdetr_oar.py tests/test_rtdetr_oar.py
git commit -m "feat: add class-conditional OAR query set ranker"
```

### Task 5: Build the amended offline runner and immutable decisions

**Files:**
- Create or modify: `scripts/run_rtdetr_oar.py`
- Create or modify: `tests/test_run_rtdetr_oar.py`
- Modify: `src/oar_protocol.py`
- Modify: `tests/test_oar_protocol.py`

- [ ] **Step 1: Write failing all-pair state-machine tests**

The legal sequence is:

```text
authority -> verified 518/129 cache -> sparse D0 frozen failure
-> OAR-R2 epochs 1..20 -> R2 selection/decision
-> optional OAR-QS2 epochs 1..20 -> QS2 selection/decision
-> first internal passer official validation once
```

Assert QS2 is authorized only when R2 has positive mAP and AP75 deltas but mAP below
`+0.0050`. Assert a passing R2 skips QS2. Assert non-positive R2 stops the branch.

- [ ] **Step 2: Run and confirm state-machine failure**

Run: `python -m pytest tests/test_oar_protocol.py tests/test_run_rtdetr_oar.py -q`

Expected: FAIL because the amended states are absent.

- [ ] **Step 3: Implement cache authority and historical-cache compatibility**

Verify the existing seven-field hidden-aware records and every external manifest/shard
SHA-256. Never modify the historical cache. Copy validated records into a new OAR
authority only when operational separation requires it; otherwise use a read-only loader
whose authority includes the historical manifest hash and this amendment commit.

- [ ] **Step 4: Implement deterministic R2/QS2 training**

Use seed0, float32, AMP off, batch 8, exactly 20 epochs, pinned MuSGD, permutation seeded
by `seed+epoch`, complete all epochs, create-only checkpoints and sidecars, contiguous
verified resume, and lexicographic `(map,ap75,ap50,-epoch)` selection. Build all 3,000
teacher utilities and adjusted logits for each image; there is no K or pool mask.

- [ ] **Step 5: Implement exact internal and official Gates**

```python
internal_pass = map_delta >= Decimal("0.0050") and ap75_delta > Decimal("0")
qs2_attributed = qs2_map > r2_map and qs2_ap75 > r2_ap75
official_pass = official_map_delta > Decimal("0") and official_ap75_delta > Decimal("0")
```

Official records remain physically inaccessible until the first passing immutable
internal decision. C0 and candidate use the same official tensors and C0 reproduces the
stock authority.

- [ ] **Step 6: Run regression suites and commit**

Run:

```bash
python -m pytest tests/test_rtdetr_oar.py tests/test_oar_protocol.py tests/test_run_rtdetr_oar.py tests/test_rtdetr_quality_oracle.py tests/test_rtdetr_quality_probe.py -q
```

Expected: PASS with historical oracle/probe behavior unchanged.

```bash
git add src/oar_protocol.py scripts/run_rtdetr_oar.py tests/test_oar_protocol.py tests/test_run_rtdetr_oar.py
git commit -m "experiment: add amended OAR offline gate"
```

### Task 6: Deploy and execute the amended Gate

**Files:**
- Runtime artifacts under a new immutable `/data/uav` source/run root.

- [ ] **Step 1: Verify host and environment**

Use `36.103.199.62:22`, user `ubuntu`, with frozen ED25519 fingerprint
`SHA256:FPVBIMs2LoVe0RenG9xDN5KvN99tgIcdPP9rY8Ym+u8`. Use the verified
`/data/uav/venvs/iber-be-v1` runtime and recheck Python 3.10.12, torch 2.5.1+cu121,
torchvision 0.20.1+cu121, Ultralytics 8.4.90, CUDA 12.1, RTX4090, and driver 550.142.

- [ ] **Step 2: Deploy immutable source and run CUDA preflight**

Transfer a commit-complete bundle because direct GitHub access is not assumed. Verify
commit and tracked-tree identity, run focused tests, validate the 210MB historical cache,
prove stock reconstruction, and prove zero-initialized OAR output.

- [ ] **Step 3: Reproduce and freeze D0**

The canonical runner must reproduce the observed D0 values before R2 training. Any metric
drift is engineering-invalid. Publish sparse scientific failure plus the all-pair
amendment authority.

- [ ] **Step 4: Run or resume OAR-R2**

Complete all 20 epochs and publish every checkpoint sidecar and metric report. If R2
passes the internal Gate, skip QS2 and run official once. If both deltas are positive but
mAP gain is below `+0.0050`, run QS2. Otherwise freeze scientific failure.

- [ ] **Step 5: Run QS2 only when authorized**

Complete 20 epochs, require the unchanged C0 Gate and strict improvement over R2, then
run official once only if it passes.

- [ ] **Step 6: Handoff an official-positive arm**

Only an official-positive immutable decision authorizes the last-decoder integration,
fixed-subset seed0 paired 30-epoch screen, unchanged Gate2, fresh full-data seed0
100-epoch training, per-epoch publication, independent evaluation, and overhead audit.
