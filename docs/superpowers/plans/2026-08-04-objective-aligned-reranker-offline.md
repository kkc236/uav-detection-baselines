# Objective-Aligned Reranker Offline Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the frozen D0/OAR-R/optional-OAR-S offline protocol, ending with at most one official-validation decision and immutable evidence for whether detector integration is authorized.

**Architecture:** Reuse detached mature-baseline boxes, logits, final decoder hidden states, and same-class IoU targets. D0 decomposes the oracle and selects a sparse per-class candidate pool. OAR-R learns a zero-initialized bounded residual on stock class logits using deterministic Top-300 boundary RankNet supervision; OAR-S adds one class-wise set-attention layer only when OAR-R satisfies its frozen authorization condition.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, Ultralytics 8.4.90, NumPy, pytest, CUDA 12.1, NVIDIA RTX 4090, canonical JSON/SHA-256 artifacts.

---

## File map

- Create `src/rtdetr_oar.py`: pure oracle, sparse-pool, residual-ranker, pair-construction,
  RankNet, and decision math.
- Create `src/oar_protocol.py`: immutable constants, authority schema, stage names, and
  prefix-overlap audit.
- Create `scripts/run_rtdetr_oar.py`: cache verification/extraction, D0 evaluation,
  deterministic training, checkpoint resume, internal selection, and official lockout.
- Create `tests/test_rtdetr_oar.py`: pure math/model/loss/Gate tests.
- Create `tests/test_oar_protocol.py`: authority, overlap, and state-machine tests.
- Create `tests/test_run_rtdetr_oar.py`: CLI, immutable artifacts, cache reuse, resume,
  and validation-release tests.
- Modify `src/__init__.py`: no eager imports; add no behavior unless export metadata is
  already maintained there.
- Do not modify detector, FDR, IBER, or historical quality-oracle result code in this
  plan.

### Task 1: Freeze protocol constants and D0 oracle math

**Files:**
- Create: `src/oar_protocol.py`
- Create: `src/rtdetr_oar.py`
- Test: `tests/test_oar_protocol.py`
- Test: `tests/test_rtdetr_oar.py`

- [ ] **Step 1: Write failing constant and oracle tests**

```python
def test_oar_constants_are_frozen():
    assert OAR_K_GRID == (20, 40, 60, 100)
    assert OAR_GAIN_RECOVERY == Decimal("0.90")
    assert OAR_MAP_GATE == Decimal("0.0050")
    assert OAR_EPOCHS == 20
    assert OAR_PAIR_CAP == 2647


def test_oracle_families_are_class_conditional():
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]])
    logits = torch.zeros(2, 3)
    targets = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    classes = torch.tensor([1])
    scores = oracle_score_families(boxes, logits, targets, classes, num_classes=3)
    assert torch.equal(scores["stock"], torch.full((2, 3), 0.5))
    assert torch.equal(scores["presence"][:, 0], torch.zeros(2))
    assert scores["same_class"][0, 1] == 0.5
    assert scores["same_class"][0, 0] == 0
```

- [ ] **Step 2: Run the focused tests and confirm collection failure**

Run:

```bash
python -m pytest tests/test_oar_protocol.py tests/test_rtdetr_oar.py -q
```

Expected: FAIL because `src.oar_protocol` and `src.rtdetr_oar` do not exist.

- [ ] **Step 3: Add exact protocol constants**

```python
OAR_K_GRID = (20, 40, 60, 100)
OAR_GAIN_RECOVERY = Decimal("0.90")
OAR_MAP_GATE = Decimal("0.0050")
OAR_EPOCHS = 20
OAR_PAIR_CAP = 2048 + 299 + 300
OAR_NUM_CLASSES = 10
OAR_NUM_QUERIES = 300
OAR_MAX_DET = 300
OAR_HIDDEN_DIM = 256
OAR_FEATURE_DIM = 276
```

- [ ] **Step 4: Implement D0 score families and pool restriction**

```python
def topk_per_class_mask(probabilities: Tensor, k: int) -> Tensor:
    if probabilities.ndim != 2 or k not in OAR_K_GRID:
        raise ValueError("invalid OAR pool")
    index = probabilities.topk(k, dim=0).indices
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    mask.scatter_(0, index, True)
    return mask


def oracle_score_families(boxes, logits, target_boxes, target_classes, *, num_classes):
    p = logits.detach().float().sigmoid()
    same = same_class_iou_quality(
        boxes.detach().float(), target_boxes.detach().float(),
        target_classes.detach().long(), num_classes,
    )
    query = same.amax(dim=1, keepdim=True)
    present = torch.zeros(num_classes, dtype=p.dtype, device=p.device)
    if target_classes.numel():
        present[target_classes.long().unique()] = 1
    return {
        "stock": p,
        "presence": p * present,
        "query_iou": p * query.square(),
        "same_class": p * same.square(),
    }


def restrict_oracle(stock: Tensor, oracle: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask, oracle, stock)
```

- [ ] **Step 5: Implement smallest-K selection using exact decimal gains**

```python
def select_candidate_k(*, stock_map, full_map, restricted_map):
    total = Decimal(str(full_map)) - Decimal(str(stock_map))
    if total <= 0:
        return {"status": "scientific_failed", "selected_k": None}
    for k in OAR_K_GRID:
        recovered = (Decimal(str(restricted_map[k])) - Decimal(str(stock_map))) / total
        if recovered >= OAR_GAIN_RECOVERY:
            return {"status": "passed", "selected_k": k, "recovered": str(recovered)}
    return {"status": "scientific_failed", "selected_k": None}
```

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_oar_protocol.py tests/test_rtdetr_oar.py -q`

Expected: PASS for constants, D0 math, empty targets, exact masks, outside-pool identity,
and 90% boundary behavior.

- [ ] **Step 7: Commit**

```bash
git add src/oar_protocol.py src/rtdetr_oar.py tests/test_oar_protocol.py tests/test_rtdetr_oar.py
git commit -m "experiment: add OAR oracle decomposition"
```

### Task 2: Implement stock-preserving OAR-R residual head

**Files:**
- Modify: `src/rtdetr_oar.py`
- Modify: `tests/test_rtdetr_oar.py`

- [ ] **Step 1: Write failing zero-identity and isolation tests**

```python
def test_oar_r_is_exact_stock_at_initialization():
    model = OARRanker()
    features = torch.randn(2, 300, 10, 276)
    logits = torch.randn(2, 300, 10)
    mask = torch.rand(2, 300, 10) > 0.5
    adjusted, residual = apply_oar(model, features, logits, mask)
    assert torch.equal(adjusted, logits.sigmoid())
    assert torch.equal(residual, torch.zeros_like(logits))


def test_oar_backward_never_reaches_detector_evidence():
    features = torch.randn(1, 300, 10, 276, requires_grad=True)
    logits = torch.randn(1, 300, 10, requires_grad=True)
    model = OARRanker()
    adjusted, _ = apply_oar(model, features, logits, torch.ones_like(logits, dtype=torch.bool))
    adjusted.sum().backward()
    assert features.grad is None
    assert logits.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())
```

- [ ] **Step 2: Run tests and verify missing-symbol failure**

Run: `python -m pytest tests/test_rtdetr_oar.py -k 'oar_r or backward' -q`

Expected: FAIL because `OARRanker` and `apply_oar` are absent.

- [ ] **Step 3: Implement the exact residual head**

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


def apply_oar(model, features, stock_logits, pool_mask):
    residual = torch.where(pool_mask, model(features), torch.zeros_like(stock_logits))
    adjusted = (stock_logits.detach() + residual).sigmoid()
    return adjusted, residual
```

- [ ] **Step 4: Run model tests**

Run: `python -m pytest tests/test_rtdetr_oar.py -k 'oar_r or backward or residual' -q`

Expected: PASS, including residual bounds `[-2,2]`, exact zeros, detached inputs, shape
rejection, and private-gradient assertions.

- [ ] **Step 5: Commit**

```bash
git add src/rtdetr_oar.py tests/test_rtdetr_oar.py
git commit -m "feat: add stock-preserving OAR ranker"
```

### Task 3: Implement deterministic Top-300 boundary supervision

**Files:**
- Modify: `src/rtdetr_oar.py`
- Modify: `tests/test_rtdetr_oar.py`

- [ ] **Step 1: Write failing pair-order and RankNet tests**

```python
def test_boundary_pairs_are_deterministic_unique_and_capped():
    teacher = torch.linspace(1, 0, 3000)
    stock = teacher.roll(37)
    pool = torch.ones(3000, dtype=torch.bool)
    first = build_boundary_pairs(teacher, stock, pool)
    second = build_boundary_pairs(teacher, stock, pool)
    assert torch.equal(first.preferred, second.preferred)
    assert torch.equal(first.other, second.other)
    assert first.preferred.numel() <= 2647
    assert len(set(zip(first.preferred.tolist(), first.other.tolist()))) == first.preferred.numel()
    assert torch.all(teacher[first.preferred] > teacher[first.other])


def test_ranknet_prefers_teacher_order():
    pairs = RankPairs(torch.tensor([0]), torch.tensor([1]), torch.tensor([1.0]))
    aligned = boundary_rank_loss(torch.tensor([2.0, -2.0]), pairs)
    reversed_loss = boundary_rank_loss(torch.tensor([-2.0, 2.0]), pairs)
    assert aligned < reversed_loss
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python -m pytest tests/test_rtdetr_oar.py -k 'boundary or ranknet' -q`

Expected: FAIL because pair construction is missing.

- [ ] **Step 3: Implement the immutable pair container and loss**

```python
@dataclass(frozen=True)
class RankPairs:
    preferred: Tensor
    other: Tensor
    weight: Tensor


def boundary_rank_loss(adjusted_logits: Tensor, pairs: RankPairs) -> Tensor:
    difference = adjusted_logits[pairs.preferred] - adjusted_logits[pairs.other]
    element = F.softplus(-difference) * pairs.weight
    return element.sum() / pairs.weight.sum().clamp_min(torch.finfo(element.dtype).eps)
```

- [ ] **Step 4: Implement pair construction exactly from the design**

Use stable flattened indices, descending stable teacher/stock ranks, 2,048 ordered
symmetric-difference Cartesian pairs, 299 teacher-adjacent pairs, and 300 rank-offset
pairs. Remove teacher ties, remove pairs where neither endpoint is pooled, orient every
pair from greater to smaller teacher utility, and deduplicate while preserving first
occurrence. Set `weight = abs(teacher[preferred] - teacher[other]).detach()` and reject
zero/non-finite weights.

- [ ] **Step 5: Run all pure OAR tests**

Run: `python -m pytest tests/test_rtdetr_oar.py -q`

Expected: PASS with exact cap, tie handling, pool handling, duplicate removal, finite
loss, aligned-loss ordering, and gradients only through adjusted residual logits.

- [ ] **Step 6: Commit**

```bash
git add src/rtdetr_oar.py tests/test_rtdetr_oar.py
git commit -m "feat: align OAR loss with Top-300 ordering"
```

### Task 4: Implement optional OAR-S class-wise set interaction

**Files:**
- Modify: `src/rtdetr_oar.py`
- Modify: `tests/test_rtdetr_oar.py`

- [ ] **Step 1: Write failing class-isolation and zero-output tests**

```python
def test_oar_s_is_zero_initialized_and_class_isolated():
    model = OARSetRanker(k=20)
    tokens = torch.randn(2, 10, 20, 276)
    residual = model(tokens)
    assert torch.equal(residual, torch.zeros(2, 10, 20))
    changed = tokens.clone()
    changed[:, 3] += 100
    before = model.encode_before_output(tokens)
    after = model.encode_before_output(changed)
    assert torch.equal(before[:, :3], after[:, :3])
    assert torch.equal(before[:, 4:], after[:, 4:])
```

- [ ] **Step 2: Run and confirm missing-model failure**

Run: `python -m pytest tests/test_rtdetr_oar.py -k 'oar_s or class_isolated' -q`

Expected: FAIL because `OARSetRanker` is absent.

- [ ] **Step 3: Implement one shared class-wise encoder**

```python
class OARSetRanker(nn.Module):
    def __init__(self, k, feature_dim=276, width=64, heads=4, classes=10):
        super().__init__()
        self.k = k
        self.input = nn.Linear(feature_dim, width)
        self.class_embedding = nn.Embedding(classes, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 2,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens):
        batch, classes, k, _ = tokens.shape
        class_id = torch.arange(classes, device=tokens.device).view(1, classes, 1)
        x = self.input(tokens.detach()) + self.class_embedding(class_id)
        x = self.encoder(x.reshape(batch * classes, k, -1))
        raw = self.output(x).reshape(batch, classes, k)
        return 2.0 * torch.tanh(raw / 2.0)
```

- [ ] **Step 4: Run pure tests**

Run: `python -m pytest tests/test_rtdetr_oar.py -q`

Expected: PASS for class isolation, shared weights, exact zero output, deterministic
forward, candidate scatter/gather identity, and no cross-image attention.

- [ ] **Step 5: Commit**

```bash
git add src/rtdetr_oar.py tests/test_rtdetr_oar.py
git commit -m "feat: add gated OAR set interaction"
```

### Task 5: Build immutable authority, cache, and prefix audit

**Files:**
- Modify: `src/oar_protocol.py`
- Create: `scripts/run_rtdetr_oar.py`
- Modify: `tests/test_oar_protocol.py`
- Create: `tests/test_run_rtdetr_oar.py`

- [ ] **Step 1: Write failing authority and cache-compatibility tests**

Test exact baseline/dataset/subset/split hashes, source commit, schema hash, selected K,
oracle decision digest, cache manifest digest, and environment digest. Build a temporary
two-shard cache using the existing seven-field record schema and assert that corruption,
extra files, reordered IDs, alternate dtype, symlink/reparse roots, or a mismatched
external digest is rejected before `torch.load` returns data.

- [ ] **Step 2: Run tests and confirm runner/module failure**

Run:

```bash
python -m pytest tests/test_oar_protocol.py tests/test_run_rtdetr_oar.py -q
```

Expected: FAIL because OAR authority and runner do not exist.

- [ ] **Step 3: Implement canonical authority and prefix audit**

```python
def prefix_overlap(train_ids, dev_ids):
    def group(image_id, mode):
        stem = PurePosixPath(image_id).stem
        return stem.split("_", 1)[0] if mode == "first_field" else stem.split("_d_", 1)[0]
    return {
        mode: {
            "train_groups": len({group(x, mode) for x in train_ids}),
            "dev_groups": len({group(x, mode) for x in dev_ids}),
            "overlap_groups": len(
                {group(x, mode) for x in train_ids} & {group(x, mode) for x in dev_ids}
            ),
            "dev_images_in_overlap": sum(
                group(x, mode) in {group(y, mode) for y in train_ids} for x in dev_ids
            ),
        }
        for mode in ("first_field", "before_d")
    }
```

- [ ] **Step 4: Reuse the existing hidden-aware record contract**

Import only pure evidence helpers from `scripts.run_rtdetr_quality_probe` for hook-neutral
extraction and record validation. OAR owns a new create-only cache/report root and new
authority; it must never overwrite or mutate the historical quality-probe cache. When a
historical cache is supplied, verify all old manifest and shard hashes before copying
validated CPU records into the new OAR authority. When absent, execute exactly one
frozen-detector extraction for 518/129 and write OAR shards of 32 records.

- [ ] **Step 5: Run authority/cache tests**

Run: `python -m pytest tests/test_oar_protocol.py tests/test_run_rtdetr_oar.py -q`

Expected: PASS for canonical JSON, exact hashes, safe loading, create-only writes,
partial-shard resume, hook neutrality, detector-state identity, and prefix report values.

- [ ] **Step 6: Commit**

```bash
git add src/oar_protocol.py scripts/run_rtdetr_oar.py tests/test_oar_protocol.py tests/test_run_rtdetr_oar.py
git commit -m "experiment: add immutable OAR evidence protocol"
```

### Task 6: Implement D0 reports, deterministic training, and internal decisions

**Files:**
- Modify: `scripts/run_rtdetr_oar.py`
- Modify: `tests/test_run_rtdetr_oar.py`

- [ ] **Step 1: Write failing D0/state-machine/training tests**

Tests must prove the legal stage order:

```text
authority -> cache -> d0 -> oar-r epochs 1..20 -> oar-r selection/decision
          -> optional oar-s epochs 1..20 -> oar-s selection/decision
          -> optional official
```

Assert that OAR-S is rejected unless OAR-R has positive mAP and AP75 but less than
`+0.0050` mAP, and that official records cannot be enumerated before a passing immutable
internal decision.

- [ ] **Step 2: Run tests and verify state-machine failure**

Run: `python -m pytest tests/test_run_rtdetr_oar.py -k 'd0 or train or decision or official' -q`

Expected: FAIL because stages are not implemented.

- [ ] **Step 3: Implement D0 evaluation and immutable report**

For every dev record, evaluate C0, presence, query-IoU, same-class, and each restricted K
with `flattened_topk` and `compute_detection_metrics`. Publish
`d0-oracle-decomposition.json`, `d0-k-coverage.json`, and `d0-decision.json` create-only.
Bind selected K into all subsequent checkpoint authorities.

- [ ] **Step 4: Implement OAR-R training and resume**

Use deterministic epoch permutation `torch.randperm(518)` from a CPU generator seeded
with `seed + epoch`, batch 8, float32, AMP off, 20 epochs, and the existing pinned MuSGD
construction. For each batch, build `[B,300,10,276]` detached features, selected pool
masks, teacher utilities, deterministic pairs, and mean per-image RankNet loss. Save one
create-only `.pt` checkpoint and canonical hash sidecar per epoch. Resume only the
highest contiguous verified checkpoint.

- [ ] **Step 5: Implement selection and exact decisions**

Select checkpoints by `(map, ap75, ap50, -epoch)`. Use:

```python
passed = (
    Decimal(str(candidate["map"])) - Decimal(str(c0["map"])) >= Decimal("0.0050")
    and Decimal(str(candidate["ap75"])) - Decimal(str(c0["ap75"])) > 0
)
```

Publish the selected checkpoint bytes/SHA-256, all epoch metrics, deltas, thresholds,
prefix audit, D0 authority, cache authority, and terminal or continuation state.

- [ ] **Step 6: Implement optional OAR-S under the frozen condition**

Gather top-K-per-class features, train the class-wise set model with the same pair loss,
schedule, sample order, and checkpoint rules, then scatter residuals back into the full
`[300,10]` score tensor. Its passing decision also requires strict mAP and AP75 gains
over selected OAR-R.

- [ ] **Step 7: Run complete local tests**

Run:

```bash
python -m pytest tests/test_rtdetr_oar.py tests/test_oar_protocol.py tests/test_run_rtdetr_oar.py -q
```

Expected: PASS, including interrupted resume and every exact Gate boundary.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_rtdetr_oar.py tests/test_run_rtdetr_oar.py
git commit -m "experiment: add resumable OAR offline gate"
```

### Task 7: Implement one-shot official validation and publication

**Files:**
- Modify: `scripts/run_rtdetr_oar.py`
- Modify: `tests/test_run_rtdetr_oar.py`

- [ ] **Step 1: Write failing release-lock tests**

Assert no `images/val` enumeration, loader construction, cache read, or identity hashing
before a passing internal decision. After passage, assert exactly one shared C0/candidate
detector evidence pass, exact stock-authority reconstruction, strict positive official
mAP/AP75 decision, and immutable terminal failure.

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_run_rtdetr_oar.py -k 'official or validation_lock' -q`

Expected: FAIL because official release is absent.

- [ ] **Step 3: Implement official release and reports**

Create official cache intent only after validating the passing decision and every prior
hash. Evaluate C0 and the frozen selected arm from the same records. Publish
`official-oar-report.json`, `oar-decision.json`, `environment.json`, and
`sha256-inventory.json`. Never expose a CLI flag that can alter K, architecture, epochs,
loss, split, thresholds, or checkpoint selection.

- [ ] **Step 4: Run the full OAR suite and existing quality tests**

Run:

```bash
python -m pytest tests/test_rtdetr_oar.py tests/test_oar_protocol.py tests/test_run_rtdetr_oar.py tests/test_rtdetr_quality_oracle.py tests/test_rtdetr_quality_probe.py -q
```

Expected: PASS with no regression to historical oracle/probe math.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_rtdetr_oar.py tests/test_run_rtdetr_oar.py
git commit -m "experiment: gate OAR official validation"
```

### Task 8: Deploy immutable source and execute D0 on the RTX 4090

**Files:**
- Runtime artifacts only under a new server run root.

- [ ] **Step 1: Verify server identity before login**

Target: `36.103.199.62:22`, user `ubuntu`.

Frozen ED25519 fingerprint:

```text
SHA256:FPVBIMs2LoVe0RenG9xDN5KvN99tgIcdPP9rY8Ym+u8
```

Run a direct strict SSH handshake using the task-specific known-host entry. Abort on any
different fingerprint; never print credentials.

- [ ] **Step 2: Audit hardware and runtime**

Run read-only checks for Ubuntu, Python, disk, RAM, `nvidia-smi`, driver, GPU name and
memory, CUDA availability, existing `/data/uav` dataset/checkpoint/cache paths, and
GitHub reachability. Record the output without secrets.

Expected GPU: NVIDIA GeForce RTX 4090, 24 GB. Expected driver/runtime authority remains
550.142 / PyTorch 2.5.1+cu121 / CUDA 12.1 / Ultralytics 8.4.90.

- [ ] **Step 3: Provision the pinned environment and evidence**

Prefer the previously validated image/runtime. Otherwise create a Python 3.10 virtual
environment and install exact pinned wheels. Verify or transfer the dataset, 647 subset,
mature baseline checkpoint, oracle decision, and optional old cache by SHA-256 before
use. Do not download a replacement dataset with a different hash.

- [ ] **Step 4: Deploy a new immutable source root**

Create a source root named by the exact commit, verify clean tracked files and source
SHA-256, then create a run root named `oar-offline-<short-commit>-seed0`. Do not reuse an
older scientific run root.

- [ ] **Step 5: Run focused CUDA preflight**

Run the pure suite, runner contract suite, one production-shape hidden capture, exact
stock reconstruction, zero-output OAR canary, and one-batch backward. Expected: all pass,
GPU utilization visible, detector state unchanged, no detector gradients.

- [ ] **Step 6: Execute D0 and publish evidence**

Run `scripts/run_rtdetr_oar.py` with operational path/device arguments only. Verify D0
reports are canonical and immutable. If D0 cannot find a K recovering 90% of oracle
gain, publish `scientific_failed` and do not train OAR-R.

### Task 9: Execute OAR-R, optional OAR-S, and the one-shot decision

**Files:**
- Runtime artifacts and result branch only.

- [ ] **Step 1: Run or resume all 20 OAR-R epochs**

Keep the GPU occupied only with this authorized branch. After every epoch, verify the
checkpoint/sidecar pair, update canonical latest state, and publish metrics and hashes.
Never overwrite an epoch.

- [ ] **Step 2: Freeze OAR-R selection and decision**

If OAR-R passes `+0.0050` mAP and positive AP75, skip OAR-S and unlock official once. If
both deltas are positive but mAP is below `+0.0050`, authorize OAR-S. Otherwise publish
scientific failure and stop the OAR route.

- [ ] **Step 3: Run OAR-S only when authorized**

Complete all 20 deterministic epochs and freeze selection. It must pass the C0 Gate and
strictly improve mAP and AP75 over OAR-R. Failure is terminal for set interaction.

- [ ] **Step 4: Run official validation once for the first internal passer**

Verify C0 stock authority, evaluate the frozen candidate, and publish the strict
official decision. No official-positive claim is allowed without exact metrics and
artifact hashes.

- [ ] **Step 5: Handoff only a passing candidate to detector integration**

If official-positive, write the next design/implementation plan for the isolated
last-decoder OAR head, paired 30-epoch screen, per-epoch publication, Gate2, and fresh
full-data 100 epochs. If scientific-failed, freeze this plan's evidence and select the
next minimal single-variable branch; do not implement detector integration.
