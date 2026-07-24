# SADED R0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and seal a deterministic, GT-free SADED route-control replay from the authoritative Arm-A/Arm-C raw cache, then perform one isolated val safety adjudication before any new training.

**Architecture:** A pure `src.saded` module owns scale weights, cross-expert matching, conflict resolution, protected-prefix selection, and invariants. A prediction-only CLI authenticates the existing G0 closure, produces a read-only route closure without importing annotations, and a separate evaluator verifies the closure before reading GT. The R0 control uses the same stock checkpoint as both global and local experts.

**Tech Stack:** Python 3.10, NumPy, existing `Detection`/SBR fusion primitives, pytest, existing JSONL-gzip/checksum/anchor utilities.

---

## File structure

- Create `src/saded.py`: pure SADED constants, candidate types, matching, fusion, protected Top-300, capacity report, and invariants.
- Create `scripts/route_saded.py`: GT-free authenticated replay and sealed route closure.
- Create `scripts/evaluate_saded.py`: closure verification, delayed GT import, metrics, deltas, and R0 safety gate.
- Create `tests/test_saded.py`: unit and invariant tests for the pure router.
- Create `tests/test_saded_cli.py`: input authentication, GT isolation, closure, replay, and mutation tests.
- Modify `README.md`: link the frozen SADED design and R0 commands after verification.

### Task 1: Freeze constants and public types

**Files:**
- Create: `src/saded.py`
- Test: `tests/test_saded.py`

- [ ] **Step 1: Write the failing constant and signature tests**

```python
import inspect
import math

from src import saded


def test_saded_constants_are_frozen():
    assert saded.CONF_THRESHOLD == 0.001
    assert saded.MAX_DET == 300
    assert saded.TINY_EFFECTIVE_SIZE == 16.0
    assert saded.LARGE_EFFECTIVE_SIZE == 96.0
    assert saded.MATCH_IOU == 0.5
    assert saded.FRAGMENT_IOS == 0.5
    assert saded.ROUTER_K == math.log(9.0) / 8.0


def test_router_public_api_has_no_ground_truth_inputs():
    forbidden = {"gt", "target", "label", "annotation"}
    names = set(inspect.signature(saded.route_saded_image).parameters)
    assert not any(any(token in name.lower() for token in forbidden) for name in names)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest -q tests/test_saded.py
```

Expected: collection fails because `src.saded` does not exist.

- [ ] **Step 3: Add the frozen constants and immutable types**

Implement:

```python
CONF_THRESHOLD = 0.001
MAX_DET = 300
TINY_EFFECTIVE_SIZE = 16.0
LARGE_EFFECTIVE_SIZE = 96.0
MATCH_IOU = 0.5
FRAGMENT_IOS = 0.5
ROUTER_K = math.log(9.0) / 8.0

@dataclass(frozen=True)
class ExpertCandidate:
    detection: Detection
    image_id: str
    original_index: int

@dataclass(frozen=True)
class SADEDImageResult:
    predictions: tuple[Detection, ...]
    protected_baseline: tuple[Detection, ...]
    selected_matches: tuple[tuple[int, int], ...]
    coverage: Mapping[str, int]
    invariants: Mapping[str, bool]
```

Add validators that reject non-finite scores, degenerate coordinates, missing global coordinates, negative query/source/index values, or inconsistent image IDs.

- [ ] **Step 4: Run the focused tests**

Run: `pytest -q tests/test_saded.py`

Expected: the two initial tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/saded.py tests/test_saded.py
git commit -m "feat: freeze SADED router contract"
```

### Task 2: Implement analytic weights and deterministic matching

**Files:**
- Modify: `src/saded.py`
- Modify: `tests/test_saded.py`

- [ ] **Step 1: Write failing tests for the analytic router**

Add tests asserting:

```python
assert local_weight(8.0) == pytest.approx(0.9)
assert local_weight(16.0) == pytest.approx(0.5)
assert local_weight(24.0) == pytest.approx(0.1)
```

Create same-class A/C candidates with IoU values `0.8`, `0.7`, and `0.6`; assert `match_cross_expert` returns a one-to-one greedy matching ordered by descending IoU, then baseline index, source, query, and original index. Add a different-class IoU-1.0 pair and an exact-IoU-0.5 pair and assert neither is eligible.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_saded.py -k "weight or match"`

Expected: missing functions.

- [ ] **Step 3: Implement the minimal functions**

Implement:

```python
def local_weight(effective_size_px: float) -> float:
    value = float(effective_size_px)
    if not math.isfinite(value) or value < 0:
        raise ValueError("effective size must be finite and nonnegative")
    return 1.0 / (1.0 + math.exp(-ROUTER_K * (TINY_EFFECTIVE_SIZE - value)))
```

Implement strict same-class IoU `> MATCH_IOU`, stable pair sorting, and greedy one-to-one selection. Reuse the existing box IoU primitive rather than duplicating geometry.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_saded.py -k "weight or match"`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/saded.py tests/test_saded.py
git commit -m "feat: add deterministic SADED expert matching"
```

### Task 3: Implement conflict resolution and protected Top-300

**Files:**
- Modify: `src/saded.py`
- Modify: `tests/test_saded.py`

- [ ] **Step 1: Write failing routing tests**

Cover these exact cases:

1. unmatched baseline predictions are retained;
2. matched baseline effective size `>16` is byte-for-byte unchanged and the local member is absent;
3. matched baseline effective size `<=16` uses the local box/class/provenance and the analytic blended score;
4. unmatched local effective size `>16` is rejected;
5. unmatched local with incomplete provenance is rejected;
6. unmatched tiny local with IoS `>=0.5` against protected baseline non-tiny is rejected;
7. protected baseline non-tiny detections retain relative order and are never truncated;
8. remaining candidates fill only the unused portion of `max_det=300`;
9. exact score ties resolve by source, query, and original index;
10. the function calls frozen `effective_size(box, width, height)` and routes the same physical box consistently across image resolutions.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_saded.py -k "route or protected or fragment"`

Expected: missing routing implementation.

- [ ] **Step 3: Implement `route_saded_image`**

Use this exact signature:

```python
def route_saded_image(
    *,
    image_id: str,
    width: int,
    height: int,
    baseline: Sequence[ExpertCandidate],
    local_fused: Sequence[ExpertCandidate],
) -> SADEDImageResult:
```

Follow Section 5 of the frozen design exactly. Use `dataclasses.replace` to create a tiny matched fused prediction without mutating either input. Validate `len(protected_baseline) <= MAX_DET`, construct capacity counts, and return explicit invariants:

```python
{
    "protected_identity_exact": bool,
    "protected_relative_order_exact": bool,
    "no_local_non_tiny_leak": bool,
    "all_local_provenance_complete": bool,
    "max_det_respected": bool,
    "deterministic_tie_break": True,
    "passed": bool,
}
```

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_saded.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/saded.py tests/test_saded.py
git commit -m "feat: protect baseline predictions in SADED"
```

### Task 4: Build the GT-free route closure

**Files:**
- Create: `scripts/route_saded.py`
- Create: `tests/test_saded_cli.py`

- [ ] **Step 1: Write failing CLI isolation tests**

Assert the router script:

- accepts only `--input-manifest` and `--output`;
- rejects an output path that exists;
- authenticates all G0/SP-PPAF input hashes before routing;
- never imports `src.sbr_artifacts.load_dataset` or `src.sbr_metrics`;
- never opens label/annotation paths;
- fails closed on raw-cache mutation, missing provenance, source drift, wrong image order, duplicate output, or a dirty tracked worktree.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_saded_cli.py`

Expected: script/module missing.

- [ ] **Step 3: Implement `scripts/route_saded.py`**

Reuse the established authenticated input closure from `scripts/route_sbr_ppaf.py`. Reconstruct Arm A and Arm C exactly and require equality with their sealed predictions before calling `route_saded_image`. Write a staging directory containing exactly:

```text
route_manifest.json
predictions.jsonl.gz
capacity.json
route_invariants.json
checksums.sha256
```

Write `route_anchor.json` outside the closure. Include source commit/tree, input hashes, constants, image count, prediction hash, capacity hash, invariant hash, and checksum-root hash. Rename staging atomically and make the closure read-only only after verification.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_saded_cli.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/route_saded.py tests/test_saded_cli.py
git commit -m "feat: seal GT-free SADED routes"
```

### Task 5: Add isolated R0 evaluation and safety gate

**Files:**
- Create: `scripts/evaluate_saded.py`
- Modify: `tests/test_saded_cli.py`

- [ ] **Step 1: Write failing evaluation tests**

Require the evaluator to:

- verify route closure and external anchor before importing GT-aware modules;
- reproduce the sealed Arm-A metrics exactly;
- output Arm A and route-control metrics plus deltas;
- report protected-count and remaining-slot min/median/max;
- emit `R0_GO` only when every route invariant passes, AP75 delta is at least `-0.002`, AP-large delta is at least `-0.005`, and aggregate remaining tiny slots are positive;
- emit `R0_STOP` for a valid safety-gate failure and `INVALID` for evidence/software failure;
- reject changed source state or a route mutation before/during evaluation.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_saded_cli.py -k "evaluate or r0"`

Expected: evaluator missing.

- [ ] **Step 3: Implement the evaluator**

Delay:

```python
from src.sbr_artifacts import load_dataset
from src.sbr_metrics import evaluate_dataset
```

until all route hashes, schemas, source state, and input bindings pass. Produce:

```text
evaluation_manifest.json
metrics.json
deltas.json
capacity.json
evaluation_invariants.json
r0_gate.json
checksums.sha256
```

and an external evaluation anchor. Never select a threshold or score mapping from the metrics.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_saded.py tests/test_saded_cli.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_saded.py tests/test_saded_cli.py
git commit -m "feat: adjudicate SADED R0 safety"
```

### Task 6: Verify locally and run authoritative R0

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run focused and full tests**

Run:

```bash
pytest -q tests/test_saded.py tests/test_saded_cli.py
pytest -q
```

Expected: all tests pass; no regressions.

- [ ] **Step 2: Update documentation**

Link:

```markdown
- [SADED design](docs/superpowers/specs/2026-07-25-saded-design.md)
- [SADED R0 plan](docs/superpowers/plans/2026-07-25-saded-r0-implementation.md)
```

Document the two commands without secrets.

- [ ] **Step 3: Commit and push the clean source**

```bash
git add README.md docs/superpowers/specs/2026-07-25-saded-design.md docs/superpowers/plans/2026-07-25-saded-r0-implementation.md
git commit -m "docs: freeze SADED R0 protocol"
git push origin codex/sbr-rtdetr-g0
```

- [ ] **Step 4: Re-run focused tests on the RTX 4090 server**

Clone/bundle the exact commit to a new `/home/ubuntu/saded-*` path. Run:

```bash
/mnt/uav/venv/bin/python -m pytest -q tests/test_saded.py tests/test_saded_cli.py
```

Expected: all focused tests pass.

- [ ] **Step 5: Run prediction-only R0 and seal it**

Use the authoritative existing input manifest and write all new route/evidence paths under `/home/ubuntu/saded-*`. Do not read `test-dev`. Record PID, source commit, `df -B1`, input hashes, route checksum root, and external anchor.

- [ ] **Step 6: Run the isolated R0 evaluator once**

Verify the route snapshot, launch the evaluator once, seal its checksum root and external anchor, and request independent B/C review.

Expected terminal decision: `R0_GO` or `R0_STOP`; `INVALID` is debugged with TDD and a new evidence directory.

