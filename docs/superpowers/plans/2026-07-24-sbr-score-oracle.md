# SBR Score-Only Causal Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, independently adjudicate, and run the one frozen validation-only score oracle that decides whether causal cross-view score calibration is eligible for development.

**Architecture:** A focused core module identifies stock-C mixed-cluster aggressor groups, overlays the one frozen float64 score cap, rebuilds the applicable fusion pipeline, and labels groups with exact per-threshold TP profiles. A primary CLI reuses the already-verified immutable G0 input chain and writes canonical evidence; a standalone adjudicator uses only the standard library and NumPy to replay the raw evidence, recompute the joint oracle, and independently decide the original five gates.

**Tech Stack:** Python 3.10, NumPy, pytest, existing SBR geometry/fusion/metric primitives, gzip JSONL, SHA-256, Windows PowerShell, Ubuntu SSH, RTX 4090 server for the authoritative run although no new GPU inference is required.

---

## File map

- Create `src/sbr_score_oracle.py`: frozen oracle types, eligibility, score overlay, fusion replay, TP profiles, group labels, joint evaluation, invariants, and five-gate decision.
- Create `scripts/prepare_sbr_score_oracle_protocol.py`: hash-only protocol wrapper that binds the trusted V2 input manifest, approved spec, exact implementation commit/tree, and frozen rule schema without opening raw evidence or metrics.
- Create `scripts/run_sbr_score_oracle.py`: fail-closed input validation, immutable evidence streaming, baseline reproduction, canonical artifact writing, and primary CLI.
- Create `scripts/adjudicate_sbr_score_oracle.py`: standalone replay and independent evidence/gate adjudication; imports no `src` or other `scripts` modules.
- Create `tests/test_sbr_score_oracle.py`: unit and adversarial tests for the scientific rule.
- Create `tests/test_sbr_score_oracle_cli.py`: protocol-wrapper, input-chain, output-schema, parser, deterministic-worker, and primary evidence tests.
- Create `tests/test_sbr_score_oracle_adjudicator.py`: independent-replay, tamper, import-isolation, and gate-boundary tests.
- Create `docs/SBR_SCORE_ORACLE_SERVER_GUIDE.md`: exact smoke, authoritative run, adjudication, checksum, and evidence-copy commands.
- Reuse without scientific modification: `src/sbr_v2_audit.py`, `src/sbr_fusion.py`, `src/sbr_metrics.py`, `src/sbr_artifacts.py`, and the existing `sbr-v2-audit-input/v1` manifest that already seals the G0 evidence, checkpoint, validation list, labels, and dataset signature. The new wrapper references this existing manifest; it does not replace or rewrite it.

## Frozen names and constants

```python
ORACLE_SCHEMA_VERSION = "sbr-score-oracle-evidence/v1"
INPUT_SCHEMA_VERSION = "sbr-v2-audit-input/v1"
CONF = 0.001
MAX_DET = 300
IOS = 0.5
THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
GATES = {
    "AP-tiny-SBR": 0.010,
    "mAP50-95": 0.003,
    "tiny_recall": 0.020,
    "AP75": -0.002,
    "AP-large-SBR": -0.005,
}
PRIMARY_REQUIRED_ARTIFACTS = (
    "oracle_manifest.json",
    "unit_events.jsonl.gz",
    "score_patches.jsonl.gz",
    "coverage.json",
    "oracle_metrics.json",
    "invariants.json",
    "primary_gate.json",
    "runtime.json",
    "checksums.sha256",
)
FINAL_REQUIRED_ARTIFACTS = (
    "primary/checksums.sha256",
    "independent_adjudication.json",
    "final_status.json",
    "checksums.sha256",
)
```

### Task 1: Freeze aggressor-group eligibility

**Files:**
- Create: `src/sbr_score_oracle.py`
- Create: `tests/test_sbr_score_oracle.py`

- [ ] **Step 1: Write failing tests for group boundaries and stable identity**

Add these fixtures and tests:

```python
import math
from dataclasses import replace

from src.sbr_score_oracle import (
    AggressorGroup,
    find_aggressor_groups,
)
from src.sbr_v2_audit import AuditRawDetection, reconstruct_c_clusters


def raw(source, score, box, *, query=0, original=0, cls=0, arm="C"):
    return AuditRawDetection.synthetic(
        "images/i.jpg", arm, source=source, score=score, box=box,
        query=query, original_index=original, cls=cls,
        width=640, height=640,
    )


def test_group_requires_mixed_cluster_and_strict_tile_advantage():
    full = raw(0, 0.80, (0, 0, 100, 100), original=10)
    tied = raw(1, 0.80, (0, 0, 90, 90), original=11)
    low = raw(2, 0.79, (0, 0, 80, 80), original=12)
    assert find_aggressor_groups((full, tied, low)) == ()


def test_group_caps_every_tile_strictly_above_best_full():
    full_low = raw(0, 0.70, (0, 0, 100, 100), query=2, original=10)
    full = raw(0, 0.80, (0, 0, 100, 100), query=1, original=11)
    tile_a = raw(1, 0.95, (0, 0, 90, 90), original=20)
    tile_b = raw(2, 0.90, (0, 0, 80, 80), original=21)
    group = find_aggressor_groups((full_low, full, tile_a, tile_b))[0]
    assert group.full_anchor_index == 11
    assert group.aggressor_indices == (20, 21)
    assert group.anchor_score == 0.80
    assert group.unit_id.startswith("images/i.jpg:")


def test_group_is_gt_free_and_class_aware_by_stock_cluster():
    full = raw(0, 0.80, (0, 0, 100, 100), original=1, cls=0)
    other_class = raw(1, 0.99, (0, 0, 100, 100), original=2, cls=1)
    far_tile = raw(1, 0.99, (300, 300, 400, 400), original=3, cls=0)
    assert find_aggressor_groups((full, other_class, far_tile)) == ()


def test_stock_probe_is_strict_at_ios_half_and_nontransitive():
    # A overlaps B at IoS > .5, B overlaps C at IoS > .5, A/C do not.
    # Seed-only clustering must form (A,B) and (C), never one transitive group.
    records = (
        raw(1, .90, (0, 0, 100, 100), original=1),
        raw(1, .80, (40, 0, 140, 100), original=2),
        raw(1, .70, (80, 0, 180, 100), original=3),
    )
    reconstruction = reconstruct_c_clusters(records)
    assert reconstruction.cluster_members == ((1, 2), (3,))
    exact_half = (
        raw(1, .90, (0, 0, 100, 100), original=4),
        raw(1, .80, (50, 0, 150, 100), original=5),
    )
    assert reconstruct_c_clusters(exact_half).cluster_members == ((4,), (5,))


def test_stock_order_uses_score_source_query_original_index():
    records = (
        raw(1, .80, (0, 0, 10, 10), query=2, original=8),
        raw(1, .80, (20, 0, 30, 10), query=2, original=7),
    )
    assert reconstruct_c_clusters(records).cluster_members[0][0] == 7
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_score_oracle.py -q
```

Expected: collection fails because `src.sbr_score_oracle` does not exist.

- [ ] **Step 3: Add immutable types and eligibility**

Create `src/sbr_score_oracle.py` with these public types and functions:

```python
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .sbr_v2_audit import (
    AuditRawDetection,
    CClusterReconstruction,
    reconstruct_c_clusters,
)

CONF = 0.001
MAX_DET = 300
IOS = 0.5
THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
SIZE_BINS = ("tiny", "small", "medium", "large")
GATES = {
    "AP-tiny-SBR": 0.010,
    "mAP50-95": 0.003,
    "tiny_recall": 0.020,
    "AP75": -0.002,
    "AP-large-SBR": -0.005,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class AggressorGroup:
    image_id: str
    unit_id: str
    stock_cluster_position: int
    stock_member_indices: tuple[int, ...]
    full_anchor_index: int
    aggressor_indices: tuple[int, ...]
    anchor_score: float


def _rank(record: AuditRawDetection) -> tuple[float, int, int, int]:
    return (
        -float(record.score), int(record.source_order),
        int(record.query_index), int(record.original_index),
    )


def find_aggressor_groups(
    retained_raw: Iterable[AuditRawDetection],
) -> tuple[AggressorGroup, ...]:
    raw = tuple(retained_raw)
    stock = reconstruct_c_clusters(raw)
    by_index = {record.original_index: record for record in raw}
    groups: list[AggressorGroup] = []
    for position, member_indices in enumerate(stock.cluster_members):
        members = tuple(by_index[index] for index in member_indices)
        full = tuple(sorted(
            (member for member in members if member.source_order == 0),
            key=_rank,
        ))
        if not full or not any(member.source_order > 0 for member in members):
            continue
        anchor = full[0]
        aggressors = tuple(sorted(
            (
                member for member in members
                if member.source_order > 0
                and float(member.score) > float(anchor.score)
            ),
            key=_rank,
        ))
        if not aggressors:
            continue
        payload = {
            "image_id": anchor.image_id,
            "members": list(member_indices),
            "anchor": anchor.original_index,
            "aggressors": [member.original_index for member in aggressors],
        }
        unit_id = (
            f"{anchor.image_id}:"
            f"{hashlib.sha256(_canonical(payload)).hexdigest()[:24]}"
        )
        groups.append(AggressorGroup(
            image_id=anchor.image_id,
            unit_id=unit_id,
            stock_cluster_position=position,
            stock_member_indices=tuple(member_indices),
            full_anchor_index=anchor.original_index,
            aggressor_indices=tuple(
                member.original_index for member in aggressors
            ),
            anchor_score=float(anchor.score),
        ))
    return tuple(groups)
```

- [ ] **Step 4: Run Task 1 tests and the existing cluster tests**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_score_oracle.py tests/test_sbr_v2_audit.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/sbr_score_oracle.py tests/test_sbr_score_oracle.py
git commit -m "Add frozen score-oracle group eligibility"
```

### Task 2: Overlay scores and replay the applicable fusion pipeline

**Files:**
- Modify: `src/sbr_score_oracle.py`
- Modify: `tests/test_sbr_score_oracle.py`

- [ ] **Step 1: Write failing overlay and replay tests**

Append:

```python
from src.sbr_score_oracle import apply_group_overlay, replay_overlay


def test_overlay_uses_exact_float64_predecessor_and_full_bypass():
    full = raw(0, 0.80, (0, 0, 100, 100), original=10)
    tile = raw(1, 0.90, (10, 0, 110, 100), original=20)
    group = find_aggressor_groups((full, tile))[0]
    overlaid, patches = apply_group_overlay((full, tile), (group,))
    mapped = {record.original_index: record for record in overlaid}
    assert mapped[10] == full
    assert mapped[20].score == math.nextafter(0.80, -math.inf)
    assert patches[0].old_score == 0.90
    assert patches[0].new_score == math.nextafter(0.80, -math.inf)
    assert replace(mapped[20], score=tile.score) == tile


def test_post_overlay_conf_filter_precedes_reclustering():
    full = raw(0, 0.001, (0, 0, 100, 100), original=10)
    tile = raw(1, 0.002, (10, 0, 110, 100), original=20)
    group = find_aggressor_groups((full, tile))[0]
    replay = replay_overlay((full, tile), (group,))
    assert tuple(record.original_index for record in replay.active_raw) == (10,)
    assert replay.reconstruction.standard_predictions[0].source_order == 0


def test_noop_replay_matches_stock_reconstruction():
    records = (
        raw(0, 0.8, (0, 0, 100, 100), original=1),
        raw(1, 0.7, (0, 0, 90, 90), original=2),
    )
    replay = replay_overlay(records, ())
    assert replay.reconstruction == reconstruct_c_clusters(records)
    assert replay.patches == ()
```

- [ ] **Step 2: Run the three tests and verify RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_score_oracle.py -q
```

Expected: import errors for `apply_group_overlay` and `replay_overlay`.

- [ ] **Step 3: Add exact overlay and replay**

Append to `src/sbr_score_oracle.py`:

```python
@dataclass(frozen=True)
class ScorePatch:
    image_id: str
    unit_id: str
    original_index: int
    full_anchor_index: int
    old_score: float
    new_score: float


@dataclass(frozen=True)
class OverlayReplay:
    retained_raw: tuple[AuditRawDetection, ...]
    active_raw: tuple[AuditRawDetection, ...]
    patches: tuple[ScorePatch, ...]
    reconstruction: CClusterReconstruction


def apply_group_overlay(
    retained_raw: Iterable[AuditRawDetection],
    groups: Iterable[AggressorGroup],
) -> tuple[tuple[AuditRawDetection, ...], tuple[ScorePatch, ...]]:
    raw = tuple(retained_raw)
    by_index = {record.original_index: record for record in raw}
    if len(by_index) != len(raw):
        raise ValueError("retained raw identities must be unique")
    replacements: dict[int, tuple[float, AggressorGroup]] = {}
    for group in groups:
        new_score = math.nextafter(float(group.anchor_score), -math.inf)
        if not math.isfinite(new_score) or new_score >= group.anchor_score:
            raise ValueError("invalid frozen predecessor score")
        anchor = by_index.get(group.full_anchor_index)
        if anchor is None or anchor.source_order != 0:
            raise ValueError("group full anchor is missing or not full-view")
        if float(anchor.score) != float(group.anchor_score):
            raise ValueError("group anchor score disagrees with retained raw")
        for original_index in group.aggressor_indices:
            record = by_index.get(original_index)
            if (
                record is None or record.source_order == 0
                or float(record.score) <= float(group.anchor_score)
                or original_index in replacements
            ):
                raise ValueError("invalid or duplicated aggressor")
            replacements[original_index] = (new_score, group)
    overlaid: list[AuditRawDetection] = []
    patches: list[ScorePatch] = []
    for record in raw:
        change = replacements.get(record.original_index)
        if change is None:
            overlaid.append(record)
            continue
        new_score, group = change
        overlaid.append(replace(record, score=new_score))
        patches.append(ScorePatch(
            image_id=record.image_id,
            unit_id=group.unit_id,
            original_index=record.original_index,
            full_anchor_index=group.full_anchor_index,
            old_score=float(record.score),
            new_score=float(new_score),
        ))
    return tuple(overlaid), tuple(patches)


def replay_overlay(
    retained_raw: Iterable[AuditRawDetection],
    groups: Iterable[AggressorGroup],
) -> OverlayReplay:
    raw = tuple(retained_raw)
    overlaid, patches = apply_group_overlay(raw, groups)
    active = tuple(record for record in overlaid if record.score >= CONF)
    reconstruction = reconstruct_c_clusters(active)
    return OverlayReplay(raw, active, patches, reconstruction)
```

- [ ] **Step 4: Run core tests**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_score_oracle.py tests/test_sbr_fusion.py tests/test_sbr_v2_audit.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/sbr_score_oracle.py tests/test_sbr_score_oracle.py
git commit -m "Replay frozen score overlays through SBR fusion"
```

### Task 3: Label groups and build the unique joint oracle

**Files:**
- Modify: `src/sbr_score_oracle.py`
- Modify: `tests/test_sbr_score_oracle.py`

- [ ] **Step 1: Write failing TP-profile and interaction tests**

Add synthetic fixtures with one large target, one tiny target, and two eligible
groups. Assert the exact fixed rule:

```python
from src.sbr_score_oracle import (
    GATES,
    OracleImage,
    evaluate_oracle_image,
    gate_oracle_metrics,
)


def oracle_image(c_raw, gt_boxes):
    a_raw = tuple(
        raw(
            0, .99 - index * .01, box,
            original=1000 + index, arm="A",
        )
        for index, box in enumerate(gt_boxes)
    )
    return OracleImage(
        image_id="images/i.jpg",
        width=640,
        height=640,
        gt_boxes=tuple(gt_boxes),
        gt_classes=tuple(0 for _ in gt_boxes),
        ignore_boxes=(),
        a_raw=a_raw,
        c_raw=tuple(c_raw),
    )


def safe_large_recovery_image():
    full = raw(0, .80, (0, 0, 200, 200), original=10)
    local = raw(1, .90, (50, 0, 250, 200), original=11)
    return oracle_image((full, local), ((0, 0, 200, 200),))


def tiny_tradeoff_image():
    full = raw(0, .80, (0, 0, 200, 200), original=10)
    local = raw(1, .90, (10, 10, 20, 20), original=11)
    return oracle_image(
        (full, local),
        ((0, 0, 200, 200), (10, 10, 20, 20)),
    )


def interacting_groups_image():
    full_a = raw(0, .80, (0, 0, 100, 100), original=10)
    local_a = raw(1, .95, (0, 0, 50, 100), original=11)
    full_b = raw(0, .79, (40, 0, 140, 100), original=20)
    local_b = raw(2, .94, (90, 0, 140, 100), original=21)
    return oracle_image(
        (full_a, local_a, full_b, local_b),
        ((0, 0, 100, 100), (40, 0, 140, 100)),
    )


def test_group_selected_only_for_large_gain_without_any_all_tiny_large_loss():
    image = safe_large_recovery_image()
    result = evaluate_oracle_image(image)
    assert len(result.groups) == 1
    assert result.events[0].selected is True
    assert sum(
        row["large"] for row in result.events[0].tp_delta.values()
    ) > 0
    assert all(
        row["all"] >= 0 and row["tiny"] >= 0 and row["large"] >= 0
        for row in result.events[0].tp_delta.values()
    )


def test_any_threshold_tiny_loss_rejects_group():
    image = tiny_tradeoff_image()
    event = evaluate_oracle_image(image).events[0]
    assert event.selected is False
    assert event.reason == "TP_SAFETY_FAIL"


def test_joint_pass_applies_all_independently_selected_groups_once():
    image = interacting_groups_image()
    result = evaluate_oracle_image(image)
    assert [event.selected for event in result.events] == [True, True]
    assert {patch.unit_id for patch in result.joint.patches} == {
        event.unit_id for event in result.events
    }
    assert result.selection_rounds == 1
    independent_gain = sum(
        row["large"]
        for event in result.events
        for row in event.tp_delta.values()
    )
    joint_gain = sum(
        result.joint_profile[key]["large"]["tp"]
        - result.stock_profile[key]["large"]["tp"]
        for key in result.joint_profile
    )
    assert 0 < joint_gain < independent_gain


def test_gate_is_joint_minus_a_and_inclusive():
    a = {
        "AP-tiny-SBR": 0.0, "mAP50-95": 0.0,
        "tiny_recall": 0.0, "AP75": .002, "AP-large-SBR": .005,
    }
    oracle = {
        "AP-tiny-SBR": .010, "mAP50-95": .003,
        "tiny_recall": .020, "AP75": 0.0, "AP-large-SBR": 0.0,
    }
    decision = gate_oracle_metrics(a, oracle, selected_count=1)
    assert decision.status == "SBR_SCORE_ORACLE_GO"
    assert all(decision.gates.values())
    for name, threshold in GATES.items():
        below_a = dict(a)
        below_oracle = dict(oracle)
        if threshold >= 0:
            below_oracle[name] = math.nextafter(
                below_oracle[name], -math.inf,
            )
        else:
            below_a[name] = math.nextafter(below_a[name], math.inf)
        failed = gate_oracle_metrics(
            below_a, below_oracle, selected_count=1,
        )
        assert failed.status == "SBR_SCORE_ORACLE_STOP"
        assert failed.gates[name] is False
```

- [ ] **Step 2: Run Task 3 tests and verify RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_score_oracle.py -q
```

Expected: imports or assertions fail because image evaluation does not exist.

- [ ] **Step 3: Add exact metric projection and TP profiles**

Add these production interfaces:

```python
@dataclass(frozen=True)
class OracleImage:
    image_id: str
    width: int
    height: int
    gt_boxes: tuple[tuple[float, float, float, float], ...]
    gt_classes: tuple[int, ...]
    ignore_boxes: tuple[tuple[float, float, float, float], ...]
    a_raw: tuple[AuditRawDetection, ...]
    c_raw: tuple[AuditRawDetection, ...]


@dataclass(frozen=True)
class GroupEvent:
    unit_id: str
    selected: bool
    reason: str
    tp_delta: Mapping[str, Mapping[str, int]]
    fp_delta: Mapping[str, Mapping[str, int]]
    group: AggressorGroup
    patches: tuple[ScorePatch, ...]


@dataclass(frozen=True)
class OracleImageResult:
    image_id: str
    groups: tuple[AggressorGroup, ...]
    events: tuple[GroupEvent, ...]
    stock: OverlayReplay
    joint: OverlayReplay
    selection_rounds: int
    stock_profile: Mapping[str, Mapping[str, Mapping[str, int]]]
    joint_profile: Mapping[str, Mapping[str, Mapping[str, int]]]
```

Implement `_metric_projection(image, reconstruction)` so every prediction box
is `prediction.global_xyxy`, never `prediction.box`. Use the existing frozen
metric internals for exact ignore handling, stable ordering, final-300,
effective bins, and one-to-one matching:

```python
from .sbr_metrics import (
    _evaluate_threshold,
    _in_bin,
    _ioa_prediction_ignore,
    _prepare_predictions,
    _sqrt_effective_area,
    _validate,
    box_iou,
)


def tp_fp_profile(image: OracleImage, replay: OverlayReplay):
    predictions = replay.reconstruction.standard_predictions
    boxes = [
        tuple(float(v) for v in prediction.global_xyxy)
        for prediction in predictions
    ]
    scores = [float(prediction.score) for prediction in predictions]
    classes = [int(prediction.class_id) for prediction in predictions]
    sources = [int(prediction.source_order) for prediction in predictions]
    queries = [int(prediction.query_index) for prediction in predictions]
    pb, ps, pc, gb, gc, ign, src, qry = _validate(
        boxes, scores, classes, image.gt_boxes, image.gt_classes,
        image.ignore_boxes, sources, queries, CONF,
    )
    pb, ps, pc, src, qry, _ = _prepare_predictions(
        pb, ps, pc, src, qry, CONF, MAX_DET,
    )
    neutral = _ioa_prediction_ignore(pb, ign)
    iou = box_iou(pb, gb)
    gain = min(640.0 / image.width, 640.0 / image.height, 1.0)
    radius = _sqrt_effective_area(gb, gain)
    profile = {}
    for threshold in THRESHOLDS:
        masks = {"all": gc == gc}
        masks.update({
            name: _in_bin(radius, name) for name in SIZE_BINS
        })
        profile[f"{threshold:.2f}"] = {}
        for name, selected in masks.items():
            counts, _ = _evaluate_threshold(
                pb, ps, pc, neutral, gb, gc, selected, iou, threshold,
            )
            profile[f"{threshold:.2f}"][name] = counts
    return profile
```

Implement `evaluate_oracle_image` with one stock profile, one independent
single-group replay per eligible group, a frozen safe-beneficial label, and
exactly one joint replay:

```python
def _count_delta(before, after, field):
    return {
        threshold: {
            name: int(
                after[threshold][name][field]
                - before[threshold][name][field]
            )
            for name in ("all", "tiny", "small", "medium", "large")
        }
        for threshold in (f"{value:.2f}" for value in THRESHOLDS)
    }


def _selected(delta):
    protected = ("all", "tiny", "large")
    safe = all(
        delta[f"{threshold:.2f}"][name] >= 0
        for threshold in THRESHOLDS for name in protected
    )
    large_gain = sum(
        delta[f"{threshold:.2f}"]["large"] for threshold in THRESHOLDS
    ) > 0
    return safe and large_gain, (
        "SAFE_LARGE_GAIN" if safe and large_gain
        else "TP_SAFETY_FAIL" if not safe else "NO_LARGE_GAIN"
    )


def evaluate_oracle_image(image: OracleImage) -> OracleImageResult:
    groups = find_aggressor_groups(image.c_raw)
    stock = replay_overlay(image.c_raw, ())
    stock_profile = tp_fp_profile(image, stock)
    events: list[GroupEvent] = []
    selected: list[AggressorGroup] = []
    for group in groups:
        single = replay_overlay(image.c_raw, (group,))
        profile = tp_fp_profile(image, single)
        tp_delta = _count_delta(stock_profile, profile, "tp")
        fp_delta = _count_delta(stock_profile, profile, "fp")
        take, reason = _selected(tp_delta)
        if take:
            selected.append(group)
        events.append(GroupEvent(
            unit_id=group.unit_id,
            selected=take,
            reason=reason,
            tp_delta=tp_delta,
            fp_delta=fp_delta,
            group=group,
            patches=single.patches,
        ))
    joint = replay_overlay(image.c_raw, tuple(selected))
    return OracleImageResult(
        image_id=image.image_id,
        groups=groups,
        events=tuple(events),
        stock=stock,
        joint=joint,
        selection_rounds=1,
        stock_profile=stock_profile,
        joint_profile=tp_fp_profile(image, joint),
    )
```

Do not add iteration, subset search, candidate-level alternatives, or a second
demotion value.

- [ ] **Step 4: Add and test the five-gate decision**

Add:

```python
@dataclass(frozen=True)
class OracleGate:
    status: str
    deltas: Mapping[str, float]
    gates: Mapping[str, bool]


def gate_oracle_metrics(a_metrics, oracle_metrics, *, selected_count):
    deltas = {
        name: float(oracle_metrics[name]) - float(a_metrics[name])
        for name in GATES
    }
    if not all(math.isfinite(value) for value in deltas.values()):
        raise ValueError("oracle gate metrics must be finite")
    gates = {
        name: deltas[name] >= threshold
        for name, threshold in GATES.items()
    }
    status = (
        "SBR_SCORE_ORACLE_GO"
        if selected_count > 0 and all(gates.values())
        else "SBR_SCORE_ORACLE_STOP"
    )
    return OracleGate(status, deltas, gates)
```

The comparison is the bare full-precision `delta >= threshold`. Do not add an
epsilon, rounding, `Decimal`, or `isclose` tolerance in either primary or
independent code.

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_score_oracle.py tests/test_sbr_metrics.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/sbr_score_oracle.py tests/test_sbr_score_oracle.py
git commit -m "Add frozen group labels and joint score oracle"
```

### Task 4: Seal the protocol wrapper, stream inputs, and write primary evidence

**Files:**
- Create: `scripts/prepare_sbr_score_oracle_protocol.py`
- Create: `scripts/run_sbr_score_oracle.py`
- Create: `tests/test_sbr_score_oracle_cli.py`
- Modify: `src/sbr_score_oracle.py`

- [ ] **Step 1: Write failing parser and input-boundary tests**

Use the existing portable sealed-evidence fixture pattern from
`tests/test_sbr_v2_audit_cli.py`. Add:

```python
def test_parser_exposes_only_operational_paths():
    from scripts.run_sbr_score_oracle import build_parser
    args = build_parser().parse_args([
        "--input-manifest", "input.json",
        "--spec", "design.md",
        "--output", "evidence",
    ])
    assert vars(args) == {
        "input_manifest": Path("input.json"),
        "spec": Path("design.md"),
        "output": Path("evidence"),
        "workers": 0,
    }


@pytest.mark.parametrize("name", [
    "--conf", "--max-det", "--ios", "--demotion",
    "--size-threshold", "--subset", "--split",
])
def test_parser_rejects_scientific_overrides(name):
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--input-manifest", "i", "--spec", "s", "--output", "o",
            name, "1",
        ])


def test_input_requires_exact_val_548_and_rejects_test_artifacts(
    sealed_input_manifest,
):
    payload = json.loads(sealed_input_manifest.read_text())
    payload["dataset"]["split"] = "test-dev"
    sealed_input_manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="val"):
        validate_oracle_input(sealed_input_manifest, Path("design.md"))


def test_protocol_wrapper_binds_upstream_spec_commit_tree_and_rule(
    tmp_path, sealed_input_manifest, approved_spec,
):
    from scripts.prepare_sbr_score_oracle_protocol import prepare_protocol
    wrapper = prepare_protocol(
        upstream=sealed_input_manifest,
        spec=approved_spec,
        commit="a" * 40,
        tree="b" * 40,
    )
    assert wrapper["schema_version"] == "sbr-score-oracle-input/v1"
    assert wrapper["upstream_input"]["sha256"] == sha256_file(
        sealed_input_manifest
    )
    assert wrapper["approved_spec"]["sha256"] == sha256_file(approved_spec)
    assert wrapper["expected_source"] == {
        "commit": "a" * 40, "tree": "b" * 40,
    }
    assert wrapper["frozen_rule"]["conf"] == 0.001
    assert wrapper["forbidden_inputs"] == ["test-dev", "external-dataset"]
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_score_oracle_cli.py -q
```

Expected: collection fails because the runner does not exist.

- [ ] **Step 3: Add primary schema, validator, and streaming run**

`scripts/prepare_sbr_score_oracle_protocol.py` accepts only
`--upstream-input`, `--spec`, `--repo`, and `--output`. It reads and hashes the
upstream manifest and spec plus clean Git commit/tree metadata. It must not
open any URI named inside the upstream manifest and must not compute a metric.
It atomically writes:

```python
{
    "schema_version": "sbr-score-oracle-input/v1",
    "upstream_input": {"uri": str, "sha256": str},
    "approved_spec": {"uri": str, "sha256": str},
    "expected_source": {"commit": str, "tree": str},
    "frozen_rule": {
        "conf": 0.001, "max_det": 300, "ios": 0.5,
        "thresholds": list(THRESHOLDS), "gates": GATES,
        "group_rule": "mixed-cluster-all-local-strictly-above-best-full",
        "demotion": "float64-nextafter-anchor-toward-negative-infinity",
        "selection": "all-threshold-all-tiny-large-nondecrease-and-large-sum-positive",
    },
    "forbidden_inputs": ["test-dev", "external-dataset"],
}
```

`scripts/run_sbr_score_oracle.py` must:

1. expose only `--input-manifest`, `--spec`, `--output`, and operational
   `--workers`;
2. require `sbr-score-oracle-input/v1`, verify its spec hash, clean exact
   commit/tree and frozen-rule values, then call the already-tested G0 chain
   validator on its referenced `sbr-v2-audit-input/v1`;
3. additionally require upstream split `val`, exactly 548 manifest images, and
   an output path outside every input;
4. load only Arm A and Arm C retained rows;
5. reproduce sealed A and C prediction digests and complete metrics before
   evaluating groups;
6. process images in manifest order and accumulate A/C/O metric rows;
7. call `evaluate_dataset` once per arm after all images;
8. write canonical gzip JSONL with `mtime=0` and JSON with `allow_nan=False`;
9. write immutable primary artifacts under `OUTPUT_ROOT/primary`, seal
   `OUTPUT_ROOT/primary/checksums.sha256`, and never modify that subtree during
   independent adjudication;
10. use per-image atomic staging shards keyed by
   `sha256(wrapper + commit + rule schema + primary script)` so an interrupted
   infrastructure run may resume only with the identical run identity;
11. make `workers=1` and `workers=8` yield the same deterministic primary
   hashes except `runtime.json`, host, and peak-memory fields;
12. catch every exception in `main`, write no partial final status, print
   `SBR_SCORE_ORACLE_INVALID: {reason}` to stderr, and return `2`.

Each gzip JSON shard has exactly:

```python
{
    "schema_version": "sbr-score-oracle-shard/v1",
    "run_identity": str,
    "image_order": int,
    "image_id": str,
    "input_image_hash": str,
    "payload": dict,
    "payload_hash": str,
}
```

`input_image_hash` covers the canonical A/C retained raw records, GT boxes and
classes, ignore boxes, width, height, and manifest image ID. `payload_hash`
covers canonical `payload`. Staging is a sibling named
`.OUTPUT_NAME.oracle-staging`, containing `run_identity.json` and
`shards/000000.json.gz` through `shards/000547.json.gz`.

On resume, the runner first requires the final output path to be absent. An
existing staging directory is accepted only when `run_identity.json` matches
the newly recomputed identity byte-for-byte. Every existing shard must have a
unique order in `0..547`, the exact manifest image ID and input hash at that
order, a valid payload hash, and no unknown fields. A duplicate, out-of-range,
unknown, corrupt, or mismatched shard is `INVALID`; it is never silently
recomputed. Missing valid orders are submitted through `executor.map` in
manifest order. Final merge requires all 548 unique continuous orders and
sorts only by `image_order`.

The authoritative implementation uses the existing reference
`reconstruct_c_clusters` for every stock, single-group, and joint replay.
Precomputed adjacency or another clustering implementation is not introduced
after metrics become visible. `workers` changes only per-image scheduling and
never scientific code.

After the primary checksum file is sealed on Linux, the runner changes every
file under `primary/` to mode `0444` and every directory under `primary/` to
`0555`; the output root remains writable for independent artifacts. A
platform-specific integration test on Linux must prove ordinary writes to a
primary file fail after sealing. Windows unit fixtures assert the intended
mode bits but do not claim Windows ACL enforcement.

Use this output schema:

```python
ORACLE_SCHEMA = {
    "schema_version": "sbr-score-oracle-evidence/v1",
    "required_artifacts": [
        "oracle_manifest.json",
        "unit_events.jsonl.gz",
        "score_patches.jsonl.gz",
        "coverage.json",
        "oracle_metrics.json",
        "invariants.json",
        "primary_gate.json",
        "runtime.json",
        "checksums.sha256",
    ],
    "unit_id_fields": [
        "image_id", "stock_member_indices",
        "full_anchor_index", "aggressor_indices",
    ],
    "primary_gate_inputs": [
        "joint_minus_a.AP-tiny-SBR",
        "joint_minus_a.mAP50-95",
        "joint_minus_a.tiny_recall",
        "joint_minus_a.AP75",
        "joint_minus_a.AP-large-SBR",
        "invariants.passed",
    ],
    "authoritative_gate_inputs": [
        "primary_gate.status",
        "independent_adjudication.primary_gate_agrees",
        "independent_adjudication.joint_metrics_agree",
        "independent_adjudication.unit_labels_agree",
    ],
}
```

The schema is stored in `OUTPUT_ROOT/primary/oracle_manifest.json`, and every
listed relative artifact resolves inside `OUTPUT_ROOT/primary`.

The primary gate records `independent_adjudication: "PENDING"` and a
provisional `SBR_SCORE_ORACLE_GO` or `SBR_SCORE_ORACLE_STOP`. The authoritative
decision is not claimed until Task 5.

- [ ] **Step 4: Add explicit invariant and coverage tests**

Tests must tamper, reseal, and verify rejection of:

- one full score change;
- one box/class/query/source change;
- a patch not equal to `nextafter(anchor_score, -inf)`;
- a patch outside an eligible stock group;
- a no-op digest mismatch;
- a non-finite unit delta;
- 547 or 549 processed images;
- an input/output path overlap;
- a dirty source tree at authoritative-run start.

The no-op test compares the complete sealed A/C metrics plus a canonical
per-image prediction digest over box, score, class, source, query, and original
index. A match of only the five headline numbers is insufficient.

Tests must also assert coverage includes:

```python
{
    "eligible_units": int,
    "selected_units": int,
    "eligible_members": int,
    "patched_members": int,
    "affected_images": int,
    "large_positive_affected_images": int,
    "by_class": dict,
    "by_source": dict,
    "by_sequence_token": dict,
}
```

Coverage never enters `primary_gate.json`.

Add an integration fixture that runs the same sealed fake input with
`--workers 1` and `--workers 8` into two new directories and asserts equal
deterministic artifact hashes after excluding the explicitly nondeterministic
runtime fields.

- [ ] **Step 5: Run primary CLI tests and regression tests**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_score_oracle.py tests/test_sbr_score_oracle_cli.py tests/test_sbr_v2_audit.py tests/test_sbr_v2_audit_cli.py tests/test_sbr_metrics.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 4**

```powershell
git add src/sbr_score_oracle.py scripts/prepare_sbr_score_oracle_protocol.py scripts/run_sbr_score_oracle.py tests/test_sbr_score_oracle_cli.py
git commit -m "Write fail-closed score-oracle evidence"
```

### Task 5: Independently replay and adjudicate the oracle

**Files:**
- Create: `scripts/adjudicate_sbr_score_oracle.py`
- Create: `tests/test_sbr_score_oracle_adjudicator.py`

- [ ] **Step 1: Write failing import-isolation and tamper tests**

Add:

```python
import ast
import gzip
import hashlib
import json
from pathlib import Path


def imported_top_level_modules(tree):
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    return imported


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_adjudicator_imports_only_stdlib_and_numpy():
    path = Path(__file__).parents[1] / "scripts" / "adjudicate_sbr_score_oracle.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = imported_top_level_modules(tree)
    assert imported <= {
        "__future__", "argparse", "collections", "dataclasses", "gzip",
        "hashlib", "json", "math", "os", "pathlib", "platform",
        "subprocess", "sys", "tempfile", "typing", "urllib", "numpy",
    }
    assert not {"src", "scripts"} & imported


def test_adjudicator_replays_joint_and_agrees(primary_oracle_fixture):
    from scripts.adjudicate_sbr_score_oracle import adjudicate_evidence
    root = primary_oracle_fixture
    anchor = sha256_file(root / "primary" / "checksums.sha256")
    report = adjudicate_evidence(root, anchor)
    assert report["decision"] == "PASS"
    assert report["primary_gate_agrees"] is True
    assert report["joint_metrics_agree"] is True
    assert report["unit_labels_agree"] is True


def test_resealed_metric_tampering_still_fails(primary_oracle_fixture):
    from scripts.adjudicate_sbr_score_oracle import adjudicate_evidence
    root = primary_oracle_fixture
    path = root / "primary" / "oracle_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["joint"]["AP-large-SBR"] += 0.1
    path.write_text(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    reseal_primary_checksums(root / "primary")
    report = adjudicate_evidence(
        root, sha256_file(root / "primary" / "checksums.sha256"),
    )
    assert report["decision"] == "FAIL"
    assert "metric" in report["error"].lower()
```

Define `reseal_primary_checksums` in the test file as:

```python
def reseal_primary_checksums(primary):
    paths = sorted(
        path for path in Path(primary).iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    text = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in paths
    )
    (Path(primary) / "checksums.sha256").write_text(
        text, encoding="utf-8",
    )
```

Add equivalent explicit tests for unit-label flips, score increases, forged
anchor/group membership, joint-patch omission/addition, gate-threshold edits,
spec/source hash edits, and a test-dev URI. Each mutation must reseal primary
checksums so the failure comes from independent replay rather than the first
checksum comparison.

- [ ] **Step 2: Run adjudicator tests and verify RED**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_score_oracle_adjudicator.py -q
```

Expected: collection fails because the adjudicator does not exist.

- [ ] **Step 3: Build the standalone replay**

Write `scripts/adjudicate_sbr_score_oracle.py` without importing project code.
It must independently implement:

- safe URI resolution and checksum-chain verification;
- canonical JSON and gzip JSONL parsing with finite-number validation;
- raw-record parsing and unique identity validation;
- strict class-aware non-transitive Greedy IoS `> 0.5`;
- standard score-weighted fusion with final metric coordinates taken from the
  recomputed seed's `global_xyxy`;
- post-overlay `score >= 0.001` filtering;
- stable order `(-score, source_order, query_index, original_index)`;
- final `max_det=300`;
- ignore-region neutralization;
- effective-size bins and the ten IoU thresholds;
- per-unit TP deltas and the fixed safe-beneficial label;
- one simultaneous joint overlay;
- pooled class AP, AP75, mAP50-95, AP-tiny, AP-large, and tiny recall;
- exact joint-minus-A gates.

Use these top-level entry points:

```python
def replay_primary_evidence(root: Path) -> dict[str, object]:
    """Return independently recomputed unit labels, patches, metrics, and gate."""


def adjudicate_evidence(
    root: Path, primary_checksums_sha256: str,
) -> dict[str, object]:
    """Fail closed, write independent_adjudication.json, and reseal checksums."""
```

All expected cluster memberships, group labels, TP/AP values, and gate values
in adjudicator tests are hand-computed constants from static fixture bytes;
tests must not call or monkeypatch any primary scientific helper to generate
expected values.

The adjudicator captures its own clean commit, tree, and script SHA before and
after replay. Any self-change, source dirtiness, input-chain mismatch, event
count mismatch, metric mismatch, gate mismatch, or primary checksum mismatch
returns `decision: "FAIL"` and authoritative
`SBR_SCORE_ORACLE_INVALID`.

Before reading scientific content, it walks the entire `primary/` tree and
requires every mode to have `(st_mode & 0o222) == 0`. It records a sorted
pre-adjudication snapshot of relative path, type, mode, size, and SHA-256.
Immediately before writing independent files it takes the same snapshot and
requires exact equality. Any writable primary node or snapshot difference is
`INVALID`. Tamper tests explicitly restore write permission only on their
disposable fixture copies before mutation and never weaken the authoritative
path.

The adjudicator's only permitted subprocess calls are these three Git
provenance commands, executed with `shell=False` and an exact argv allowlist:

```python
(
    ("git", "-C", repo_root, "rev-parse", "HEAD"),
    ("git", "-C", repo_root, "rev-parse", "HEAD^{tree}"),
    (
        "git", "-C", repo_root, "status", "--porcelain=v1",
        "--untracked-files=all",
    ),
)
```

It must reject any attempt to start Python, a project script, the primary CLI,
or a shell. A runtime test monkeypatches `subprocess.run`, records every argv,
and asserts each call is an exact member of this allowlist. Scientific replay,
matching, AP, and gates execute inside the standalone adjudicator file.

On agreement:

- a provisional primary GO becomes authoritative
  `SBR_SCORE_ORACLE_GO`;
- a provisional primary STOP becomes authoritative
  `SBR_SCORE_ORACLE_STOP`;
- `decision` is `PASS` in both cases because STOP is a valid scientific
  outcome.

It writes only `OUTPUT_ROOT/independent_adjudication.json`,
`OUTPUT_ROOT/final_status.json`, and the root
`OUTPUT_ROOT/checksums.sha256`. It opens `OUTPUT_ROOT/primary` read-only and
records the unchanged primary checksum anchor.

- [ ] **Step 4: Add exact gate-boundary and corruption coverage**

Test all five inclusive thresholds using full-precision fractions. Also test:

- zero eligible and zero selected groups independently reproduce STOP;
- a checksum path escape is rejected;
- a gzip duplicate unit ID is rejected;
- one raw identity collision is rejected;
- weighted `.box` and seed `.global_xyxy` intentionally disagree, and the
  adjudicator matches the seed-coordinate primary result;
- a joint interaction can fail a gate even though both unit labels remain
  independently selected;
- the adjudicator does not regenerate checksums after a failed verification.
- any writable file or directory under `primary/` is rejected before replay;
- the primary path/type/mode/size/SHA-256 snapshot is identical before and
  after replay;
- on Linux, a sealed `0444` primary file cannot be opened for ordinary write;
- every recorded `subprocess.run` argv is one of the three exact Git
  provenance commands and no Python or shell process is launched.

- [ ] **Step 5: Run adjudicator and complete oracle test set**

Run:

```powershell
C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_score_oracle_adjudicator.py tests/test_sbr_score_oracle_cli.py tests/test_sbr_score_oracle.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 5**

```powershell
git add scripts/adjudicate_sbr_score_oracle.py tests/test_sbr_score_oracle_adjudicator.py
git commit -m "Independently adjudicate the score oracle"
```

### Task 6: Document and verify the authoritative execution

**Files:**
- Create: `docs/SBR_SCORE_ORACLE_SERVER_GUIDE.md`
- Modify: `README.md`

- [ ] **Step 1: Write the exact server guide**

The guide fixes:

```text
Repository: /mnt/uav/repo-sbr-rtdetr-g0
Python: /mnt/uav/venv/bin/python
Immutable G0 evidence: /mnt/uav/evidence/sbr-g0a-51ee6c44
Approved design: docs/superpowers/specs/2026-07-24-sbr-score-oracle-design.md
Trusted V2 audit manifest: /mnt/uav/evidence/sbr-v2-audit-b6a10f16-20260723T204530Z/audit_manifest.json
Upstream input: resolved deterministically from audit_manifest.json[input_manifest][uri]
Protocol input: a new sbr-score-oracle-input/v1 wrapper that hashes the trusted input, approved spec, exact commit/tree, and frozen rule
Output: /mnt/uav/evidence/sbr-score-oracle-${COMMIT8}-${UTC}
```

Document these exact commands with shell variables that do not use `HOME`:

```bash
set -euo pipefail
REPO=/mnt/uav/repo-sbr-rtdetr-g0
PYTHON=/mnt/uav/venv/bin/python
TRUSTED_AUDIT=/mnt/uav/evidence/sbr-v2-audit-b6a10f16-20260723T204530Z/audit_manifest.json
cd "$REPO"
COMMIT="$(git rev-parse HEAD)"
COMMIT8="${COMMIT:0:8}"
UTC="$(date -u +%Y%m%dT%H%M%SZ)"
PROTOCOL="/mnt/uav/manifests/sbr-score-oracle-${COMMIT8}.json"
OUTPUT="/mnt/uav/evidence/sbr-score-oracle-${COMMIT8}-${UTC}"
UPSTREAM_INPUT="$("$PYTHON" -c 'import json, pathlib, sys, urllib.parse; p=pathlib.Path(sys.argv[1]); u=json.loads(p.read_text(encoding="utf-8"))["input_manifest"]["uri"]; q=urllib.parse.urlparse(u); print(pathlib.Path(urllib.parse.unquote(q.path)) if q.scheme=="file" else pathlib.Path(u))' "$TRUSTED_AUDIT")"

/mnt/uav/venv/bin/python -m pytest \
  tests/test_sbr_score_oracle.py \
  tests/test_sbr_score_oracle_cli.py \
  tests/test_sbr_score_oracle_adjudicator.py -q

"$PYTHON" scripts/prepare_sbr_score_oracle_protocol.py \
  --upstream-input "$UPSTREAM_INPUT" \
  --spec docs/superpowers/specs/2026-07-24-sbr-score-oracle-design.md \
  --repo "$REPO" \
  --output "$PROTOCOL"

"$PYTHON" scripts/run_sbr_score_oracle.py \
  --input-manifest "$PROTOCOL" \
  --spec docs/superpowers/specs/2026-07-24-sbr-score-oracle-design.md \
  --output "$OUTPUT" \
  --workers 8

test -z "$(find "$OUTPUT/primary" -perm /222 -print)"
PRIMARY_ANCHOR="$(sha256sum "$OUTPUT/primary/checksums.sha256" | cut -d' ' -f1)"
"$PYTHON" scripts/adjudicate_sbr_score_oracle.py \
  --evidence "$OUTPUT" \
  --primary-checksums-sha256 "$PRIMARY_ANCHOR"
(cd "$OUTPUT" && sha256sum -c checksums.sha256)
printf 'AUTHORITATIVE_OUTPUT=%s\n' "$OUTPUT"
```

The guide must explicitly forbid `test-dev`, external datasets, alternate
thresholds, retries with changed rules, overwriting old output, or deleting
the failed V2 evidence.

- [ ] **Step 2: Run formatting, complete tests, and source checks**

Run:

```powershell
git diff --check
C:\uav_env\Scripts\python.exe -m pytest -q
git status --short
```

Expected: no whitespace errors, the complete suite passes, and only intended
documentation changes remain before commit.

- [ ] **Step 3: Commit Task 6**

```powershell
git add docs/SBR_SCORE_ORACLE_SERVER_GUIDE.md README.md
git commit -m "Document authoritative score-oracle execution"
```

### Task 7: B/C review checkpoint and clean-source freeze

**Files:**
- No scientific code changes unless a reviewer identifies a spec violation.

- [ ] **Step 1: Have B perform a read-only compliance review**

B checks every approved design section against the implementation and reports
either `APPROVE` or exact blockers. Required checks include intervention
boundary, group rule, full bypass, complete replay, original five gates,
test-dev exclusion, evidence completeness, and adjudicator independence.

- [ ] **Step 2: Have C perform a read-only critical-path review**

C verifies the run uses the existing trusted input manifest, does not schedule
new inference, does not add a data audit, and has an unambiguous GO/STOP next
step. C reports either `APPROVE` or exact blockers.

- [ ] **Step 3: Resolve only proven implementation defects**

For each blocker, first add a failing regression test that demonstrates
departure from the approved spec, run it to verify RED, make the smallest
correction, rerun the focused and complete suites, and commit with a message
that names the invariant. Do not change scientific constants or selection
semantics.

- [ ] **Step 4: Freeze a clean commit**

Run:

```powershell
git status --short
git rev-parse HEAD
git rev-parse HEAD^{tree}
C:\uav_env\Scripts\python.exe -m pytest -q
```

Expected: empty status, stable commit/tree IDs, and the complete suite passes.

### Task 8: Deploy, run the unique oracle, adjudicate, and sync evidence

**Files:**
- Create outside Git: one new server evidence directory.
- Create outside Git: a local byte-preserving copy whose directory name is
  exactly the basename of the server output followed by `/evidence`.

- [ ] **Step 1: Verify the server target before mutation**

Using strict SSH host-key checking, read:

```bash
cd /mnt/uav/repo-sbr-rtdetr-g0
git status --short
git rev-parse HEAD
git rev-parse HEAD^{tree}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Expected: the known isolated worktree, no unrelated running experiment that
would be disturbed, and the single RTX 4090. Do not expose credentials in
logs or artifacts.

- [ ] **Step 2: Transfer the exact clean commit**

Use exactly one Git-bundle path. From local PowerShell:

```powershell
$repo = 'C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\tmp\worktrees\sbr-rtdetr-g0'
$branch = 'codex/sbr-rtdetr-g0'
$bundle = 'C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\tmp\sbr-score-oracle-authoritative.bundle'
$commitFile = "$bundle.commit"
Set-Location -LiteralPath $repo
$expectedCommit = (git rev-parse HEAD).Trim()
$expectedTree = (git rev-parse 'HEAD^{tree}').Trim()
git bundle create $bundle $branch
[System.IO.File]::WriteAllText(
    $commitFile, $expectedCommit,
    [System.Text.UTF8Encoding]::new($false)
)
scp -P 22 $bundle $commitFile ubuntu@36.103.177.186:/mnt/uav/incoming/
```

On the server, use only fast-forward:

```bash
set -euo pipefail
REPO=/mnt/uav/repo-sbr-rtdetr-g0
BUNDLE=/mnt/uav/incoming/sbr-score-oracle-authoritative.bundle
COMMIT_FILE=/mnt/uav/incoming/sbr-score-oracle-authoritative.bundle.commit
BRANCH=codex/sbr-rtdetr-g0
EXPECTED_COMMIT="$(cat "$COMMIT_FILE")"
cd "$REPO"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git bundle verify "$BUNDLE"
git fetch "$BUNDLE" "$BRANCH"
git merge --ff-only FETCH_HEAD
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git status --porcelain=v1 --untracked-files=all
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
```

The final status output must be empty before the server tests. Do not fetch
from GitHub and do not use a second transfer path.

- [ ] **Step 3: Run focused and full server tests**

Run:

```bash
/mnt/uav/venv/bin/python -m pytest \
  tests/test_sbr_score_oracle.py \
  tests/test_sbr_score_oracle_cli.py \
  tests/test_sbr_score_oracle_adjudicator.py -q
/mnt/uav/venv/bin/python -m pytest -q
```

Expected: both commands pass before any real effect metric is produced.

- [ ] **Step 4: Resolve and hash the trusted input manifest**

Locate the exact upstream manifest used by
`sbr-v2-audit-b6a10f16-20260723T204530Z`, verify its own SHA-256 and all
referenced G0/checkpoint/dataset files. Run the hash-only protocol preparation
CLI to bind that manifest, the approved spec, and the exact clean commit/tree.
Verify the wrapper contains no test-dev or external-dataset input. Do not
construct a new data split.

- [ ] **Step 5: Run the primary oracle exactly once**

Create a fresh timestamped output name from the frozen commit and UTC time,
then run the documented primary command. Capture stdout, stderr, elapsed time,
peak RSS, and exit code without modifying the output afterward.

Expected: exit `0` and provisional `SBR_SCORE_ORACLE_GO` or
`SBR_SCORE_ORACLE_STOP`; exit `2` is software-invalid and must be diagnosed
without changing the frozen scientific rule.

- [ ] **Step 6: Independently adjudicate**

Hash `primary/checksums.sha256`, pass that exact anchor to the standalone
adjudicator, and require:

```text
decision = PASS
checksums_verified = true
primary_gate_agrees = true
joint_metrics_agree = true
unit_labels_agree = true
```

The authoritative outcome is either
`SBR_SCORE_ORACLE_GO` or
`SBR_SCORE_ORACLE_STOP`; `independent_adjudication.decision` is `PASS` for
either valid matching outcome.

- [ ] **Step 7: Copy evidence byte-for-byte and verify both sides**

Let `$serverOutput` be the exact `OUTPUT` printed by the authoritative guide
command, let `$runName = Split-Path -Leaf $serverOutput`, and copy with:

```powershell
$artifactRoot = 'C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\artifacts'
$localEvidence = Join-Path (Join-Path $artifactRoot $runName) 'evidence'
New-Item -ItemType Directory -Path $localEvidence -Force | Out-Null
scp -P 22 -r "ubuntu@36.103.177.186:${serverOutput}/." $localEvidence
```

Recompute every `checksums.sha256` entry locally, compare the primary anchor,
and compare the server/local file-name, size, and SHA-256 manifests.

- [ ] **Step 8: Apply the predeclared result branch**

- On `SBR_SCORE_ORACLE_GO`: stop all oracle work and start a new brainstorming/spec
  cycle for the GT-free CCCH head. The oracle metrics remain development-only.
- On `SBR_SCORE_ORACLE_STOP`: permanently close score calibration and start a new
  brainstorming/spec cycle for the training-time cross-view consistency
  fallback. Do not start training from C's high-level sketch until its own
  loss, matching, data, seed, and screen specification is approved.
- On INVALID: retain the failed artifact, use systematic debugging to prove the
  software defect, add a regression test, correct only the defect, and rerun
  the same frozen rule into a new output directory.
