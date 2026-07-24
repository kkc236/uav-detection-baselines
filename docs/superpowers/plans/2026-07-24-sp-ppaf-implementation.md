# SP-PPAF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and run one deterministic zero-inference SP-PPAF replay that produces A, All-A, P1, P2, and P3 outputs and adjudicates the original five gates.

**Architecture:** A pure `src/sbr_ppaf.py` module consumes sealed Arm A final detections plus an existing Arm C pre-cap reconstruction and returns the five frozen arms without accepting ground truth. A routing CLI creates and checksum-seals prediction-only evidence without importing the dataset loader or evaluator. A separate evaluation CLI verifies that closure, then loads labels and computes the frozen metrics and decision. B/C perform read-only integrity and decision review; no standalone paper-grade adjudicator is required for this internal feasibility replay.

**Tech Stack:** Python 3.10+, dataclasses, NumPy float64, existing `src.sbr_fusion`, `src.sbr_v2_audit`, `src.sbr_metrics`, pytest, gzip JSONL, SHA256 evidence utilities.

---

## File Structure

- Create `src/sbr_ppaf.py`: frozen constants, score mapping, candidate filtering, P1/P2/P3/All-A construction, coverage, invariants, and five-gate decision.
- Create `tests/test_sbr_ppaf.py`: pure RED/GREEN tests and evaluator property tests.
- Create `scripts/route_sbr_ppaf.py`: prediction-only cache replay and checksum-sealed route closure.
- Create `scripts/evaluate_sbr_ppaf.py`: checksum-verified, GT-aware evaluation closure and frozen decision.
- Create `tests/test_sbr_ppaf_cli.py`: synthetic route/evaluation process-boundary and tamper tests.
- Modify `README.md`: link the frozen SP-PPAF design after implementation is complete.

### Task 1: Pure frozen score band

**Files:**
- Create: `src/sbr_ppaf.py`
- Create: `tests/test_sbr_ppaf.py`

- [ ] **Step 1: Write the failing score-band tests**

```python
import inspect
import math

import pytest

from src.sbr_ppaf import (
    A_FLOOR,
    C_CEILING,
    CONF_THRESHOLD,
    map_tail_score,
)


def test_tail_score_map_is_strictly_inside_frozen_band_and_monotone():
    values = [CONF_THRESHOLD, 0.1, 0.5, 1.0]
    mapped = [map_tail_score(value) for value in values]
    assert all(CONF_THRESHOLD < value < C_CEILING < A_FLOOR for value in mapped)
    assert mapped == sorted(mapped)
    assert len(set(mapped)) == len(mapped)


@pytest.mark.parametrize("value", [True, -1.0, math.nan, math.inf, 1.1])
def test_tail_score_map_rejects_invalid_scores(value):
    with pytest.raises(ValueError):
        map_tail_score(value)


def test_router_public_api_contains_no_ground_truth_inputs():
    from src.sbr_ppaf import build_ppaf_arms

    names = set(inspect.signature(build_ppaf_arms).parameters)
    assert not names & {"gt", "gt_boxes", "gt_classes", "ignore_boxes", "matches"}
```

- [ ] **Step 2: Run the score-band tests and verify RED**

Run:

```powershell
python -m pytest tests/test_sbr_ppaf.py -q
```

Expected: collection fails because `src.sbr_ppaf` does not exist.

- [ ] **Step 3: Implement the frozen constants and mapping**

Create `src/sbr_ppaf.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from numbers import Real
from typing import Any

from src.sbr_fusion import Detection, intersection_over_smaller
from src.sbr_v2_audit import (
    AuditRawDetection,
    CClusterReconstruction,
    effective_size,
)

CONF_THRESHOLD = 0.001
MAX_DET = 300
LARGE_EFFECTIVE_SIZE = 96.0
FRAGMENT_IOS = 0.5
A_FLOOR = 0.01706760562956333
C_CEILING = 0.008533802814781666
SCORE_LOW = math.nextafter(CONF_THRESHOLD, math.inf)
SCORE_HIGH = math.nextafter(C_CEILING, -math.inf)


def map_tail_score(score: object) -> float:
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("tail score must be a finite real in [conf, 1]")
    value = float(score)
    if not math.isfinite(value) or not CONF_THRESHOLD <= value <= 1.0:
        raise ValueError("tail score must be a finite real in [conf, 1]")
    mapped = SCORE_LOW + (SCORE_HIGH - SCORE_LOW) * (
        (value - CONF_THRESHOLD) / (1.0 - CONF_THRESHOLD)
    )
    if not CONF_THRESHOLD < mapped < C_CEILING:
        raise ValueError("mapped tail score escaped the frozen band")
    return mapped
```

- [ ] **Step 4: Run the score-band tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_sbr_ppaf.py -q
```

Expected: score-band tests pass; the API-signature test may still fail until the next task adds `build_ppaf_arms`.

- [ ] **Step 5: Commit the score-band primitive**

```powershell
git add src/sbr_ppaf.py tests/test_sbr_ppaf.py
git commit -m "feat: add frozen SP-PPAF score band"
```

### Task 2: Pure P1/P2/P3 and All-A routing

**Files:**
- Modify: `src/sbr_ppaf.py`
- Modify: `tests/test_sbr_ppaf.py`

- [ ] **Step 1: Add failing synthetic routing tests**

Append tests that construct `AuditRawDetection.synthetic` records and use
`reconstruct_c_clusters`:

```python
from src.sbr_ppaf import build_ppaf_arms
from src.sbr_v2_audit import AuditRawDetection, reconstruct_c_clusters


def raw(arm, *, source, query, score, box, index, cls=0):
    return AuditRawDetection.synthetic(
        "i.jpg", arm, source=source, query=query, score=score, box=box,
        width=640, height=640, original_index=index, cls=cls,
    )


def test_primary_arms_protect_only_a_large_and_fill_from_tile_nonlarge():
    a_large = raw("A", source=0, query=1, score=.8, box=(0, 0, 120, 120), index=1).to_detection()
    a_small = raw("A", source=0, query=2, score=.7, box=(200, 0, 220, 20), index=2).to_detection()
    c_full = raw("C", source=0, query=1, score=.8, box=(0, 0, 120, 120), index=11)
    c_tile = raw("C", source=1, query=3, score=.6, box=(300, 0, 320, 20), index=12)
    reconstruction = reconstruct_c_clusters((c_full, c_tile))

    result = build_ppaf_arms(
        image_id="i.jpg", width=640, height=640,
        a_final=(a_large, a_small),
        c_reconstruction=reconstruction, c_raw=(c_full, c_tile),
    )

    assert result.arms["P1"][0] == a_large
    assert a_small not in result.arms["P1"]
    assert result.arms["P3"][0] == a_large
    assert result.arms["All-A"][:2] == (a_large, a_small)
    assert all(len(predictions) <= 300 for predictions in result.arms.values())


def test_p2_removes_cluster_with_exact_selected_full_provenance():
    a_large = raw("A", source=0, query=4, score=.8, box=(0, 0, 120, 120), index=1).to_detection()
    c_full = raw("C", source=0, query=4, score=.8, box=(0, 0, 120, 120), index=10)
    c_local = raw("C", source=1, query=4, score=.9, box=(10, 10, 80, 80), index=11)
    reconstruction = reconstruct_c_clusters((c_full, c_local))

    result = build_ppaf_arms(
        image_id="i.jpg", width=640, height=640, a_final=(a_large,),
        c_reconstruction=reconstruction, c_raw=(c_full, c_local),
    )

    assert len(result.arms["P1"]) == 2
    assert result.arms["P2"] == (a_large,)
    assert result.coverage["P2"]["provenance_rejected"] == 1


def test_p3_removes_only_same_class_tile_only_ios_fragment_at_half():
    a_large = raw("A", source=0, query=1, score=.8, box=(0, 0, 120, 120), index=1, cls=2).to_detection()
    fragment = raw("C", source=1, query=2, score=.7, box=(0, 0, 60, 60), index=10, cls=2)
    other_class = raw("C", source=2, query=3, score=.6, box=(0, 0, 60, 60), index=11, cls=3)
    reconstruction = reconstruct_c_clusters((fragment, other_class))

    result = build_ppaf_arms(
        image_id="i.jpg", width=640, height=640, a_final=(a_large,),
        c_reconstruction=reconstruction, c_raw=(fragment, other_class),
    )

    assert len(result.arms["P2"]) == 3
    assert len(result.arms["P3"]) == 2
    assert result.arms["P3"][1].class_id == 3
    assert result.coverage["P3"]["fragment_rejected"] == 1
```

Add boundary, invalid-provenance, stable-tie, no-capacity, and 301-candidate
tests. Assert that exact size 96 is non-large and exact IoS 0.5 is rejected by
P3.

- [ ] **Step 2: Run the routing tests and verify RED**

Run:

```powershell
python -m pytest tests/test_sbr_ppaf.py -q
```

Expected: failures because `build_ppaf_arms` and result dataclasses are missing.

- [ ] **Step 3: Implement result types and the frozen router**

Add to `src/sbr_ppaf.py`:

```python
ARM_NAMES = ("A", "All-A", "P1", "P2", "P3")


@dataclass(frozen=True)
class PPAFImageResult:
    arms: Mapping[str, tuple[Detection, ...]]
    coverage: Mapping[str, Mapping[str, int]]
    invariants: Mapping[str, bool]


@dataclass(frozen=True)
class _TailCandidate:
    prediction: Detection
    member_identities: frozenset[tuple[str, int, int, int]]
    tile_only: bool


def _identity(image_id: str, detection: Detection) -> tuple[str, int, int, int]:
    return (
        image_id, int(detection.class_id),
        int(detection.source_order), int(detection.query_index),
    )


def _mapped_candidate(prediction: Detection) -> Detection:
    box = prediction.global_xyxy
    if box is None:
        raise ValueError("Arm-C candidate is missing sealed global_xyxy")
    return replace(
        prediction,
        box=tuple(float(value) for value in box),
        score=map_tail_score(prediction.score),
    )


def _fill(
    prefix: tuple[Detection, ...],
    candidates: tuple[_TailCandidate, ...],
) -> tuple[Detection, ...]:
    remaining = MAX_DET - len(prefix)
    if remaining < 0:
        raise ValueError("prefix exceeds max_det")
    return prefix + tuple(item.prediction for item in candidates[:remaining])


def _coverage(
    *,
    prefix: int,
    raw_candidates: int,
    eligible: int,
    provenance_rejected: int,
    fragment_rejected: int,
    output: int,
) -> dict[str, int]:
    return {
        "prefix": prefix,
        "remaining": MAX_DET - prefix,
        "raw_candidates": raw_candidates,
        "source_scale_eligible": eligible,
        "provenance_rejected": provenance_rejected,
        "fragment_rejected": fragment_rejected,
        "appended": output - prefix,
        "output": output,
    }


def build_ppaf_arms(
    *,
    image_id: str,
    width: int,
    height: int,
    a_final: Sequence[Detection],
    c_reconstruction: CClusterReconstruction,
    c_raw: Sequence[AuditRawDetection],
) -> PPAFImageResult:
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a nonempty string")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    a = tuple(a_final)
    raw = tuple(c_raw)
    if len(a) > MAX_DET:
        raise ValueError("sealed Arm A exceeds max_det")
    for prediction in a:
        if not isinstance(prediction, Detection):
            raise ValueError("Arm A values must be Detection instances")
        if prediction.source_order != 0:
            raise ValueError("Arm A final detections must be full-view")
        if (
            not math.isfinite(float(prediction.score))
            or not CONF_THRESHOLD <= float(prediction.score) <= 1.0
            or prediction.global_xyxy is None
        ):
            raise ValueError("invalid sealed Arm A detection")
    for item in raw:
        if (
            not isinstance(item, AuditRawDetection)
            or item.arm != "C"
            or item.image_id != image_id
            or item.width != width
            or item.height != height
        ):
            raise ValueError("invalid Arm C raw provenance")
    rebuilt = reconstruct_c_clusters(raw)
    if rebuilt != c_reconstruction:
        raise ValueError("Arm C reconstruction disagrees with raw cache")
    if len(c_reconstruction.pre_cap_predictions) != len(
        c_reconstruction.cluster_members
    ):
        raise ValueError("Arm C candidate/member alignment is invalid")

    raw_by_index = {item.original_index: item for item in raw}
    if len(raw_by_index) != len(raw):
        raise ValueError("Arm C raw indices must be unique")
    a_large = tuple(
        prediction
        for prediction in a
        if effective_size(
            prediction.global_xyxy, width=width, height=height
        )
        > LARGE_EFFECTIVE_SIZE
    )
    a_large_ids = frozenset(_identity(image_id, item) for item in a_large)
    all_a_ids = frozenset(_identity(image_id, item) for item in a)

    eligible_items: list[_TailCandidate] = []
    for prediction, indices in zip(
        c_reconstruction.pre_cap_predictions,
        c_reconstruction.cluster_members,
    ):
        try:
            members = tuple(raw_by_index[index] for index in indices)
        except KeyError as exc:
            raise ValueError("cluster references missing raw provenance") from exc
        if not members or not any(item.source_order > 0 for item in members):
            continue
        if prediction.global_xyxy is None:
            raise ValueError("Arm C seed is missing global_xyxy")
        if (
            effective_size(
                prediction.global_xyxy, width=width, height=height
            )
            > LARGE_EFFECTIVE_SIZE
        ):
            continue
        eligible_items.append(
            _TailCandidate(
                prediction=_mapped_candidate(prediction),
                member_identities=frozenset(
                    item.identity_key for item in members
                ),
                tile_only=all(item.source_order > 0 for item in members),
            )
        )
    eligible = tuple(eligible_items)

    p2_tail = tuple(
        item
        for item in eligible
        if item.member_identities.isdisjoint(a_large_ids)
    )
    p3_tail = tuple(
        item
        for item in p2_tail
        if not (
            item.tile_only
            and any(
                item.prediction.class_id == anchor.class_id
                and intersection_over_smaller(
                    item.prediction.box, anchor.global_xyxy
                )
                >= FRAGMENT_IOS
                for anchor in a_large
            )
        )
    )
    all_a_p2_tail = tuple(
        item for item in eligible if item.member_identities.isdisjoint(all_a_ids)
    )
    all_a_p3_tail = tuple(
        item
        for item in all_a_p2_tail
        if not (
            item.tile_only
            and any(
                item.prediction.class_id == anchor.class_id
                and intersection_over_smaller(
                    item.prediction.box, anchor.global_xyxy
                )
                >= FRAGMENT_IOS
                for anchor in a_large
            )
        )
    )

    arms = {
        "A": a,
        "All-A": _fill(a, all_a_p3_tail),
        "P1": _fill(a_large, eligible),
        "P2": _fill(a_large, p2_tail),
        "P3": _fill(a_large, p3_tail),
    }
    coverage = {
        "A": _coverage(
            prefix=len(a), raw_candidates=0, eligible=0,
            provenance_rejected=0, fragment_rejected=0, output=len(a),
        ),
        "All-A": _coverage(
            prefix=len(a),
            raw_candidates=len(c_reconstruction.pre_cap_predictions),
            eligible=len(eligible),
            provenance_rejected=len(eligible) - len(all_a_p2_tail),
            fragment_rejected=len(all_a_p2_tail) - len(all_a_p3_tail),
            output=len(arms["All-A"]),
        ),
        "P1": _coverage(
            prefix=len(a_large),
            raw_candidates=len(c_reconstruction.pre_cap_predictions),
            eligible=len(eligible), provenance_rejected=0,
            fragment_rejected=0, output=len(arms["P1"]),
        ),
        "P2": _coverage(
            prefix=len(a_large),
            raw_candidates=len(c_reconstruction.pre_cap_predictions),
            eligible=len(eligible),
            provenance_rejected=len(eligible) - len(p2_tail),
            fragment_rejected=0, output=len(arms["P2"]),
        ),
        "P3": _coverage(
            prefix=len(a_large),
            raw_candidates=len(c_reconstruction.pre_cap_predictions),
            eligible=len(eligible),
            provenance_rejected=len(eligible) - len(p2_tail),
            fragment_rejected=len(p2_tail) - len(p3_tail),
            output=len(arms["P3"]),
        ),
    }
    invariants = {
        "a_prefix_identity": arms["P3"][: len(a_large)] == a_large,
        "all_a_identity": arms["All-A"][: len(a)] == a,
        "score_band": all(
            CONF_THRESHOLD < item.prediction.score < C_CEILING
            for item in eligible
        ),
        "max_det": all(len(value) <= MAX_DET for value in arms.values()),
        "passed": True,
    }
    invariants["passed"] = all(
        value for key, value in invariants.items() if key != "passed"
    )
    return PPAFImageResult(
        arms=arms, coverage=coverage, invariants=invariants
    )
```

Also import `reconstruct_c_clusters` from `src.sbr_v2_audit`. Keep the
implementation literal: do not introduce configurable thresholds or quotas.

- [ ] **Step 4: Run pure tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_sbr_ppaf.py -q
```

Expected: all pure routing tests pass.

- [ ] **Step 5: Add evaluator property tests**

Use `evaluate_dataset` to prove:

- All-A with zero fillers exactly reproduces A;
- appending lower-band TP, FP, neutral ignore, and out-of-bin detections cannot
  reduce All-A AP;
- P3 uses one unified prediction set for overall and size-bin metrics;
- scores below/at conf and equal-score source/query ties behave as expected.

Run:

```powershell
python -m pytest tests/test_sbr_ppaf.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the pure router**

```powershell
git add src/sbr_ppaf.py tests/test_sbr_ppaf.py
git commit -m "feat: add deterministic SP-PPAF router"
```

### Task 3: Gate, coverage aggregation, and invariants

**Files:**
- Modify: `src/sbr_ppaf.py`
- Modify: `tests/test_sbr_ppaf.py`

- [ ] **Step 1: Write failing gate and invariant tests**

```python
from src.sbr_ppaf import decide_ppaf, metric_deltas, verify_dataset_invariants


def five(value):
    return {
        "AP-tiny-SBR": value,
        "mAP50-95": value,
        "tiny_recall": value,
        "AP75": value,
        "AP-large-SBR": value,
    }


def test_decision_prefers_p3_then_fallback_and_otherwise_stops():
    a = five(0.5)
    p3_pass = {**a, "AP-tiny-SBR": .51, "mAP50-95": .503,
               "tiny_recall": .52, "AP75": .498, "AP-large-SBR": .495}
    fallback_pass = dict(p3_pass)
    failed = dict(a)

    assert decide_ppaf(a, p3_pass, failed, invariants_passed=True)["status"] == "SP_PPAF_PASS"
    assert decide_ppaf(a, failed, fallback_pass, invariants_passed=True)["status"] == "SP_PPAF_FALLBACK_PASS"
    assert decide_ppaf(a, failed, failed, invariants_passed=True)["status"] == "SP_PPAF_STOP"
    assert decide_ppaf(a, p3_pass, fallback_pass, invariants_passed=False)["status"] == "SP_PPAF_INVALID"
```

Add tests that tamper with one A box, A score, C mapped band, output count,
cluster provenance, or coverage sum and require `passed=False`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_sbr_ppaf.py -q
```

Expected: missing gate and invariant functions.

- [ ] **Step 3: Implement exact deltas, gates, and invariant aggregation**

Add frozen thresholds:

```python
GATE_THRESHOLDS = {
    "AP-tiny-SBR": 0.010,
    "mAP50-95": 0.003,
    "tiny_recall": 0.020,
    "AP75": -0.002,
    "AP-large-SBR": -0.005,
}
```

Implement:

```python
def metric_deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    deltas = {}
    for key in GATE_THRESHOLDS:
        left = candidate.get(key)
        right = baseline.get(key)
        if (
            isinstance(left, bool)
            or not isinstance(left, Real)
            or isinstance(right, bool)
            or not isinstance(right, Real)
            or not math.isfinite(float(left))
            or not math.isfinite(float(right))
        ):
            raise ValueError(f"invalid metric: {key}")
        deltas[key] = float(left) - float(right)
    return deltas


def decide_ppaf(
    a_metrics: Mapping[str, Any],
    p3_metrics: Mapping[str, Any],
    fallback_metrics: Mapping[str, Any],
    *,
    invariants_passed: bool,
) -> dict[str, Any]:
    p3_delta = metric_deltas(p3_metrics, a_metrics)
    fallback_delta = metric_deltas(fallback_metrics, a_metrics)
    p3_gates = {
        key: p3_delta[key] >= threshold
        for key, threshold in GATE_THRESHOLDS.items()
    }
    fallback_gates = {
        key: fallback_delta[key] >= threshold
        for key, threshold in GATE_THRESHOLDS.items()
    }
    if invariants_passed is not True:
        status = "SP_PPAF_INVALID"
    elif all(p3_gates.values()):
        status = "SP_PPAF_PASS"
    elif all(fallback_gates.values()):
        status = "SP_PPAF_FALLBACK_PASS"
    else:
        status = "SP_PPAF_STOP"
    return {
        "status": status,
        "p3_delta": p3_delta,
        "fallback_delta": fallback_delta,
        "p3_gates": p3_gates,
        "fallback_gates": fallback_gates,
        "invariants_passed": invariants_passed is True,
    }


def verify_dataset_invariants(
    per_image_results: Sequence[PPAFImageResult],
    *,
    expected_image_count: int,
) -> dict[str, Any]:
    rows = tuple(per_image_results)
    if (
        isinstance(expected_image_count, bool)
        or not isinstance(expected_image_count, int)
        or expected_image_count <= 0
    ):
        raise ValueError("expected_image_count must be positive")
    image_count_equal = len(rows) == expected_image_count
    per_image_passed = all(
        isinstance(row, PPAFImageResult)
        and row.invariants.get("passed") is True
        for row in rows
    )
    coverage_consistent = all(
        all(
            arm in row.arms
            and arm in row.coverage
            and row.coverage[arm]["output"] == len(row.arms[arm])
            and row.coverage[arm]["output"] <= MAX_DET
            and row.coverage[arm]["appended"]
            == row.coverage[arm]["output"] - row.coverage[arm]["prefix"]
            for arm in ARM_NAMES
        )
        for row in rows
    )
    return {
        "image_count": len(rows),
        "expected_image_count": expected_image_count,
        "image_count_equal": image_count_equal,
        "per_image_passed": per_image_passed,
        "coverage_consistent": coverage_consistent,
        "passed": image_count_equal
        and per_image_passed
        and coverage_consistent,
    }
```

Require finite numbers, exact gate comparisons with no rounding, P3-first
decision order, and invalid status whenever invariants fail.

- [ ] **Step 4: Run tests and verify GREEN**

```powershell
python -m pytest tests/test_sbr_ppaf.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the gate**

```powershell
git add src/sbr_ppaf.py tests/test_sbr_ppaf.py
git commit -m "feat: add SP-PPAF invariants and gate"
```

### Task 4: Prediction-only routing CLI

**Files:**
- Create: `scripts/route_sbr_ppaf.py`
- Create: `tests/test_sbr_ppaf_cli.py`

- [ ] **Step 1: Write failing route-boundary tests**

Create synthetic V2-style cache evidence and test:

- direct `python scripts/route_sbr_ppaf.py --help`;
- the route script source/import graph contains neither the dataset loader nor
  `src.sbr_metrics`;
- a sentinel dataset loader cannot be called during routing;
- input/output overlap, existing output, dirty source, malformed provenance,
  and input checksum mismatch fail closed;
- the real Arm A minimum must equal `A_FLOOR` exactly;
- all distinct eligible real-cache C scores map strictly with no float64
  collision, while equal-score ties retain frozen order;
- exact route artifact names and checksum verification;
- output coverage, cluster identities, pre-cap ranks, mapped ordering, and
  P2/P3 set differences agree with the route rows.

- [ ] **Step 2: Run CLI tests and verify RED**

```powershell
python -m pytest tests/test_sbr_ppaf_cli.py -q
```

Expected: route script missing.

- [ ] **Step 3: Implement the routing closure**

`scripts/route_sbr_ppaf.py` must:

1. Insert the repository root into `sys.path`.
2. Validate only prediction/cache manifests, hashes, image identifiers, image
   dimensions, frozen Arm A rows, and Arm C raw provenance.
3. Never import or call the dataset loader, annotation parser, evaluator, or
   metric helpers.
4. Reconstruct Arm C pre-cap clusters and build A, All-A, P1, P2, and P3 once.
5. Verify the real-cache `A_FLOOR`, complete cluster identity coverage,
   pre-cap rank continuity, exact score-map ordering/collision properties,
   prefix identity, P2 exact-provenance difference, P3 exact fragment
   difference, and all coverage arithmetic.
6. Atomically create `output/route` containing only:

```text
route_manifest.json
predictions.jsonl.gz
coverage.json
route_invariants.json
checksums.sha256
```

7. Write checksums last, verify them, atomically rename the staging directory,
   and print `SP_PPAF_ROUTE_SEALED`. Do not mark files read-only.
8. Fail closed with `SP_PPAF_ROUTE_INVALID: <reason>` and exit 2.

- [ ] **Step 4: Run tests and verify GREEN**

```powershell
python -m pytest tests/test_sbr_ppaf_cli.py -q
```

- [ ] **Step 5: Commit the route process**

```powershell
git add scripts/route_sbr_ppaf.py tests/test_sbr_ppaf_cli.py
git commit -m "feat: seal prediction-only SP-PPAF routes"
```

### Task 5: Separate GT-aware evaluation CLI

**Files:**
- Create: `scripts/evaluate_sbr_ppaf.py`
- Modify: `tests/test_sbr_ppaf_cli.py`

- [ ] **Step 1: Write failing evaluation tests**

Test that evaluation:

- refuses a missing or tampered route checksum;
- loads annotations only after every route artifact and checksum is verified;
- reproduces the sealed Arm A and Arm C baselines exactly;
- evaluates the same sealed row set for every metric;
- uses exact, unrounded deltas;
- applies only `P3 -> All-A -> STOP`;
- creates a new evaluation closure and never changes the route closure;
- fails on existing evaluation output or source mutation.

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_sbr_ppaf_cli.py -q
```

Expected: evaluation script missing.

- [ ] **Step 3: Implement evaluation**

`scripts/evaluate_sbr_ppaf.py` must verify the route closure before importing
or invoking annotation/evaluator code. It then loads the original dataset,
evaluates A, C, All-A, P1, P2, and P3, verifies A/C reproduction, computes the
frozen gates, and atomically creates `output/evaluation` containing:

```text
evaluation_manifest.json
metrics.json
deltas.json
evaluation_invariants.json
primary_gate.json
checksums.sha256
```

The decision is `SP_PPAF_PASS`, `SP_PPAF_FALLBACK_PASS`, `SP_PPAF_STOP`, or
`SP_PPAF_INVALID`. P1/P2 are reported only as mechanism ablations and can never
be selected.

- [ ] **Step 4: Run tests and verify GREEN**

```powershell
python -m pytest tests/test_sbr_ppaf_cli.py -q
```

- [ ] **Step 5: Commit evaluation**

```powershell
git add scripts/evaluate_sbr_ppaf.py tests/test_sbr_ppaf_cli.py
git commit -m "feat: evaluate sealed SP-PPAF routes"
```

### Task 6: S0 and full local regression

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run focused S0**

```powershell
python -m pytest tests/test_sbr_ppaf.py tests/test_sbr_ppaf_cli.py -q
```

- [ ] **Step 2: Run existing SBR regression**

```powershell
python -m pytest tests/test_sbr_fusion.py tests/test_sbr_metrics.py tests/test_sbr_v2_audit.py tests/test_sbr_score_oracle.py -q
```

- [ ] **Step 3: Run the complete suite**

```powershell
python -m pytest -q
```

- [ ] **Step 4: Link the design in `README.md`**

- [ ] **Step 5: Run `git diff --check`, inspect status/staged stats, and confirm
no credentials, datasets, caches, weights, logs, or server evidence are staged**

- [ ] **Step 6: Commit the documentation**

### Task 7: Server sync and one 548-image replay

**Files:**
- No repository source edits after launch.
- Create one new server output root with separate route/evaluation children.

- [ ] **Step 1: Push the clean branch and fast-forward the clean server checkout**

- [ ] **Step 2: Run focused and full tests on the server**

```bash
python -m pytest tests/test_sbr_ppaf.py tests/test_sbr_ppaf_cli.py -q
python -m pytest -q
```

- [ ] **Step 3: Freeze a never-existing output**

```text
/mnt/uav/evidence/sbr-sp-ppaf-<commit8>-<UTC>
```

Preserve all previous STOP and INVALID evidence.

- [ ] **Step 4: Route once without GT**

```bash
python scripts/route_sbr_ppaf.py \
  --input-manifest /mnt/uav/protocols/sbr-v2-audit-b6a10f16-20260723T204530Z/input_manifest.json \
  --output /mnt/uav/evidence/sbr-sp-ppaf-<commit8>-<UTC>
```

Do not inspect partial outputs. On exit 0, verify
`route/checksums.sha256`, exact artifacts, 548-image coverage, source state,
`A_FLOOR`, score-collision checks, and route invariants.

- [ ] **Step 5: Evaluate the sealed route once**

```bash
python scripts/evaluate_sbr_ppaf.py \
  --input-manifest /mnt/uav/protocols/sbr-v2-audit-b6a10f16-20260723T204530Z/input_manifest.json \
  --route /mnt/uav/evidence/sbr-sp-ppaf-<commit8>-<UTC>/route \
  --output /mnt/uav/evidence/sbr-sp-ppaf-<commit8>-<UTC>/evaluation
```

Verify `evaluation/checksums.sha256`, exact artifacts, baseline reproduction,
and all invariants before reading `primary_gate.json`.

- [ ] **Step 6: Apply the frozen decision**

- P3 passes all five gates: freeze P3.
- Otherwise, if All-A passes all five gates: freeze All-A.
- Otherwise: STOP post-processing repair and enter the predeclared
  training-time asymmetric cross-view consistency route.
- Integrity failure: reproduce with a test and fix software only; do not change
  scientific constants.

### Task 8: B/C review and handoff

- [ ] **Step 1: Ask B to verify the GT boundary, constants, provenance,
checksums, invariants, and unmodified state machine**

- [ ] **Step 2: Ask C to verify the complete five-metric decision and next route**

- [ ] **Step 3: Append the sealed paths, checksums, metrics, decision, and exact
next route to the private handoff without credentials**
