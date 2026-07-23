# SBR-V2 Causal Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic, streaming, independently adjudicated audit that decides whether the sole SBR-V2 Large-View Guard hypothesis is eligible for implementation.

**Architecture:** A focused `src/sbr_v2_audit.py` module streams the immutable G0-A raw cache image by image, reconstructs Arm-A and Arm-C clusters, reproduces frozen large-target matching, attributes AP75 A-TP-to-C-FN events, and computes the exact Large-View Guard recoverable upper bound. A strict primary CLI writes checksummed evidence; a separate adjudicator process reads only canonical artifacts and independently recomputes gates without importing the primary module.

**Tech Stack:** Python 3.10, NumPy, gzip/JSON, existing `src.sbr_fusion`, `src.sbr_metrics`, `src.sbr_artifacts`, pytest.

---

## File Structure

- Create `src/sbr_v2_audit.py`: streaming cache parser, cluster reconstruction, deterministic matching, attribution, and V2 upper-bound calculation.
- Create `scripts/audit_sbr_v2.py`: strict operational CLI and atomic evidence writer.
- Create `scripts/adjudicate_sbr_v2_audit.py`: independent artifact/checksum/gate adjudicator; it must not import `src.sbr_v2_audit` or `src.sbr_metrics`.
- Create `tests/test_sbr_v2_audit.py`: synthetic TDD coverage for mapping, matching, attribution, and guard invariants.
- Create `tests/test_sbr_v2_audit_cli.py`: artifact, schema, checksum, and fail-closed CLI tests.
- Create `tests/test_sbr_v2_audit_adjudicator.py`: independent summary/gate/checksum tests.

## Task 1: Reconstruct Immutable A/C Inputs and Clusters

**Files:**
- Create: `src/sbr_v2_audit.py`
- Create: `tests/test_sbr_v2_audit.py`

- [ ] **Step 1: Write failing tests for immutable mapping and streaming groups**

Add tests with this public API:

```python
from src.sbr_v2_audit import (
    AuditRawDetection,
    group_relevant_raw_rows,
    map_full_a_to_c,
    reconstruct_c_clusters,
)


def test_group_relevant_raw_rows_keeps_only_a_and_c_in_manifest_order():
    rows = [
        {"image_id": "a.jpg", "arm": "A", "source_order": 0, "query_index": 0},
        {"image_id": "a.jpg", "arm": "B", "source_order": 1, "query_index": 0},
        {"image_id": "a.jpg", "arm": "C", "source_order": 0, "query_index": 0},
        {"image_id": "b.jpg", "arm": "A", "source_order": 0, "query_index": 0},
    ]
    groups = list(group_relevant_raw_rows(rows, ["a.jpg", "b.jpg"]))
    assert [g.image_id for g in groups] == ["a.jpg", "b.jpg"]
    assert [r["arm"] for r in groups[0].rows] == ["A", "C"]


def test_a_full_detection_maps_to_exact_c_key_and_bytes():
    a = AuditRawDetection.synthetic("i.jpg", "A", source=0, query=7, score=.8)
    c = AuditRawDetection.synthetic("i.jpg", "C", source=0, query=7, score=.8)
    assert map_full_a_to_c([a], [c]) == {a.identity_key: 0}


def test_mapping_rejects_collision_or_coordinate_difference():
    a = AuditRawDetection.synthetic("i.jpg", "A", source=0, query=7, score=.8)
    c = AuditRawDetection.synthetic("i.jpg", "C", source=0, query=7, score=.8)
    bad = AuditRawDetection.synthetic("i.jpg", "C", source=0, query=7, score=.8, box=(1, 1, 3, 3))
    with pytest.raises(ValueError):
        map_full_a_to_c([a], [c, bad])


def test_reconstructed_clusters_match_frozen_strict_ios_and_raw_indices():
    full = AuditRawDetection.synthetic("i.jpg", "C", source=0, query=0, score=.9, box=(0, 0, 10, 10))
    tile = AuditRawDetection.synthetic("i.jpg", "C", source=1, query=1, score=.8, box=(1, 1, 9, 9))
    result = reconstruct_c_clusters([full, tile])
    assert result.cluster_members == ((0, 1),)
    assert result.standard_predictions[0].box == pytest.approx((8 / 17, 8 / 17, 162 / 17, 162 / 17))
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
& C:\uav_env\Scripts\python.exe -m pytest tests/test_sbr_v2_audit.py -q
```

Expected: collection failure because `src.sbr_v2_audit` does not exist.

- [ ] **Step 3: Implement the minimal immutable data layer**

Implement:

```python
@dataclass(frozen=True)
class AuditRawDetection:
    image_id: str
    arm: str
    width: int
    height: int
    source_order: int
    query_index: int
    class_id: int
    score: float
    network_xyxy: tuple[float, float, float, float]
    view_xyxy: tuple[float, float, float, float]
    global_xyxy: tuple[float, float, float, float]
    tile_bounds: tuple[int, int, int, int] | None
    original_index: int

    @property
    def identity_key(self) -> tuple[str, int, int, int]:
        return (self.image_id, self.class_id, self.source_order, self.query_index)

    def to_detection(self) -> Detection:
        return Detection(
            box=self.global_xyxy,
            score=self.score,
            class_id=self.class_id,
            source_order=self.source_order,
            query_index=self.query_index,
            view_xyxy=self.view_xyxy,
            global_xyxy=self.global_xyxy,
            network_xyxy=self.network_xyxy,
            tile_local_box=self.view_xyxy if self.tile_bounds is not None else None,
            global_box=self.global_xyxy,
            tile_bounds=self.tile_bounds,
            tile_index=self.source_order - 1 if self.tile_bounds is not None else None,
        )
```

`group_relevant_raw_rows` must stream one image at a time, require exact
manifest order, retain only Arms A/C, and reject an unknown or repeated image
group. `map_full_a_to_c` must use the exact identity key and canonical JSON
bytes for score and all three coordinate frames. `reconstruct_c_clusters`
must call `greedy_ios_clusters(..., ios_threshold=0.5)`, record raw indices,
score-weight coordinates in float64, retain pre-cap ordering with
`(-score, source_order, query_index, original_cluster_index)`, and expose both
pre-cap and top-300 predictions.

- [ ] **Step 4: Run Task 1 tests**

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/sbr_v2_audit.py tests/test_sbr_v2_audit.py
git commit -m "Add deterministic SBR-V2 cache reconstruction"
```

## Task 2: Implement Frozen Large Matching and Failure Attribution

**Files:**
- Modify: `src/sbr_v2_audit.py`
- Modify: `tests/test_sbr_v2_audit.py`

- [ ] **Step 1: Write failing matching and attribution tests**

Add:

```python
from src.sbr_v2_audit import (
    AttributionCategory,
    audit_image_at_threshold,
    effective_size,
    match_large_targets,
)


def test_effective_size_uses_arm_a_640_gain():
    assert effective_size((0, 0, 200, 200), width=1000, height=500) == pytest.approx(128.0)


def test_large_matching_is_class_aware_and_ignore_neutral():
    result = match_large_targets(
        predictions=[
            pred((0, 0, 120, 120), .9, cls=0),
            pred((200, 200, 260, 260), .8, cls=1),
        ],
        gt_boxes=[(0, 0, 120, 120)],
        gt_classes=[0],
        ignore_boxes=[(200, 200, 260, 260)],
        width=640,
        height=640,
        iou_threshold=.75,
    )
    assert result.gt_to_prediction == {0: 0}
    assert result.neutral_prediction_indices == (1,)


def test_mixed_localization_requires_full_counterfactual_to_recover_target():
    fixture = localization_loss_fixture()
    event = audit_image_at_threshold(fixture, iou_threshold=.75).events[0]
    assert event.category is AttributionCategory.MIXED_CLUSTER_LOCALIZATION
    assert event.counterfactual_recovers is True


def test_category_precedence_is_localization_then_truncation_then_competition():
    assert audit_image_at_threshold(truncation_fixture(), .75).events[0].category is AttributionCategory.FINAL_300_TRUNCATION
    assert audit_image_at_threshold(competition_fixture(), .75).events[0].category is AttributionCategory.MATCHING_COMPETITION
```

- [ ] **Step 2: Run the new tests and verify RED**

Expected: imports or assertions fail because matching/attribution is absent.

- [ ] **Step 3: Implement the exact matcher**

Implement a float64 IoU/IoA matcher that:

- filters scores at `0.001`;
- orders by `(-score, source_order, query_index, original_index)`;
- caps at 300;
- neutralizes predictions with ignore IoA over prediction area `>=0.5`;
- treats predictions matching non-large GT at the current IoU threshold as
  neutral for the large-bin audit;
- uses class-aware one-to-one matching with highest IoU then lowest GT index;
- returns prediction-to-GT and GT-to-prediction indices.

Implement `audit_image_at_threshold` so only unique large GTs that are TP in A
and FN in C enter the denominator. Apply the exact category precedence from
the design. The mixed counterfactual replaces one cluster's coordinates,
leaves all other fields fixed, and reruns the complete matcher.

- [ ] **Step 4: Run Task 2 tests**

Expected: all matching and category tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/sbr_v2_audit.py tests/test_sbr_v2_audit.py
git commit -m "Add large-loss attribution audit"
```

## Task 3: Compute the Large-View Guard Upper Bound and Invariants

**Files:**
- Modify: `src/sbr_v2_audit.py`
- Modify: `tests/test_sbr_v2_audit.py`

- [ ] **Step 1: Write failing guard and invariant tests**

Add:

```python
from src.sbr_v2_audit import apply_large_view_guard, verify_guard_invariants


def test_guard_anchors_only_mixed_cluster_with_predicted_size_over_96():
    fixture = mixed_large_cluster_fixture()
    guarded = apply_large_view_guard(fixture.clusters, width=1000, height=500)
    assert guarded[0].box == fixture.full_member.box
    assert guarded[0].score == fixture.standard[0].score
    assert guarded[0].class_id == fixture.standard[0].class_id


def test_guard_does_not_change_small_mixed_or_local_only_clusters():
    fixture = nonqualifying_clusters_fixture()
    guarded = apply_large_view_guard(fixture.clusters, width=640, height=640)
    assert [p.box for p in guarded] == pytest.approx([p.box for p in fixture.standard])


def test_guard_preserves_cluster_selection_scores_classes_and_singletons():
    report = verify_guard_invariants(invariant_fixture())
    assert report == {
        "raw_hash_equal": True,
        "cluster_hash_equal": True,
        "cluster_count_equal": True,
        "scores_equal": True,
        "classes_equal": True,
        "selected_cluster_ids_equal": True,
        "singleton_preservation": 1.0,
        "passed": True,
    }
```

- [ ] **Step 2: Run the tests and verify RED**

Expected: guard functions are missing.

- [ ] **Step 3: Implement the minimal guard**

For every mixed cluster, choose the full member by
`(-score, source_order, query_index, original_raw_index)`. If its predicted
effective size is strictly greater than 96, copy only its coordinates into
the cluster output. Keep the standard cluster score, class, seed ordering,
cluster ID, and final top-300 selection. Do not implement large-priority
ranking or any SP-BRF change.

Add `evaluate_guard_upper_bound` that feeds A, C, and V2 rows to the existing
`evaluate_dataset`, calculates the five V2-A deltas, and returns:

```python
{
    "mechanism_share_ap75": float,
    "mechanism_gate": "PASS" | "FAIL",
    "a_metrics": {...},
    "c_metrics": {...},
    "v2_metrics": {...},
    "v2_minus_a": {...},
    "v2_minus_c": {...},
    "recoverable_upper_bound_gate": "PASS" | "FAIL",
    "invariants": {...},
}
```

The upper-bound gate requires `V2 AP-large >= A AP-large - 0.005`; it is
independent of the 60% mechanism gate.

- [ ] **Step 4: Run Task 3 tests**

Expected: all guard/invariant tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/sbr_v2_audit.py tests/test_sbr_v2_audit.py
git commit -m "Add SBR-V2 recoverable upper-bound audit"
```

## Task 4: Add the Streaming Primary Audit CLI and Evidence Contract

**Files:**
- Create: `scripts/audit_sbr_v2.py`
- Create: `tests/test_sbr_v2_audit_cli.py`
- Modify: `src/sbr_v2_audit.py`

- [ ] **Step 1: Write failing CLI tests**

Add tests asserting:

```python
def test_cli_exposes_only_operational_arguments():
    parser = build_parser()
    options = {action.dest for action in parser._actions}
    assert {"input_manifest", "output", "workers"} <= options
    assert not {"ios", "conf", "max_det", "large_threshold", "mechanism_share"} & options


def test_manifest_rejects_hash_mismatch_or_original_evidence_output(tmp_path):
    with pytest.raises(ValueError):
        validate_input_manifest(bad_hash_manifest(tmp_path))
    with pytest.raises(ValueError):
        validate_input_manifest(output_inside_original_evidence_manifest(tmp_path))


def test_primary_output_is_atomic_versioned_and_checksummed(tmp_path):
    run_synthetic_audit(tmp_path)
    required = {
        "audit_manifest.json",
        "attribution_events.jsonl.gz",
        "attribution_summary.json",
        "upper_bound_metrics.json",
        "invariants.json",
        "primary_gate.json",
        "checksums.sha256",
    }
    assert required <= {p.name for p in tmp_path.iterdir()}
```

- [ ] **Step 2: Run CLI tests and verify RED**

Expected: missing CLI module/functions.

- [ ] **Step 3: Implement strict streaming CLI**

The parser accepts only:

- `--input-manifest`
- `--output`
- `--workers` (frozen operational default 0)

The input manifest contains portable artifact URIs, expected content hashes,
dataset YAML/root, image list, schema version, source commit/tree hash, and
protocol hash. Validate every checksum before opening raw evidence. Require a
clean output directory outside the original evidence directory.

Stream `raw_views.jsonl.gz` once, process each image, and store only bounded
per-image state plus global metric rows and event records. Write canonical
JSON/JSONL-GZip atomically. Record schema JSON and schema hash, environment,
runtime, peak RSS, input hashes, primary script hash, and exact Git
provenance. `primary_gate.json` is `SBR_V2_AUDIT_ELIGIBLE` only when:

- AP75 unique-GT mechanism share is at least 0.60;
- upper-bound AP-large reaches `A AP-large - 0.005`;
- all invariants pass.

Otherwise it is `SBR_V2_AUDIT_STOP`.

- [ ] **Step 4: Run CLI tests**

Expected: all CLI tests pass.

- [ ] **Step 5: Commit Task 4**

```powershell
git add scripts/audit_sbr_v2.py src/sbr_v2_audit.py tests/test_sbr_v2_audit_cli.py
git commit -m "Add audited SBR-V2 causal analysis CLI"
```

## Task 5: Add Independent Adjudication

**Files:**
- Create: `scripts/adjudicate_sbr_v2_audit.py`
- Create: `tests/test_sbr_v2_audit_adjudicator.py`

- [ ] **Step 1: Write failing adjudicator tests**

Add:

```python
def test_adjudicator_recomputes_eligible_gate_without_primary_import(tmp_path):
    evidence = eligible_primary_fixture(tmp_path)
    report = adjudicate(evidence)
    assert report["decision"] == "PASS"
    assert report["status"] == "SBR_V2_AUDIT_INDEPENDENT_PASS"


def test_adjudicator_fails_on_event_summary_or_checksum_tampering(tmp_path):
    evidence = eligible_primary_fixture(tmp_path)
    tamper_event_count(evidence)
    report = adjudicate(evidence)
    assert report["decision"] == "FAIL"
```

- [ ] **Step 2: Run the tests and verify RED**

Expected: adjudicator module is missing.

- [ ] **Step 3: Implement the independent process**

The script may import only Python standard library plus NumPy. It must not
import `src.sbr_v2_audit`, `src.sbr_metrics`, or the primary CLI. It verifies:

- all checksums and safe relative paths;
- schema version and schema hash;
- event JSON finiteness and unique event IDs;
- AP75 denominator/category counts against the summary;
- mechanism share against the frozen 0.60 threshold;
- upper-bound AP-large against recorded A AP-large minus 0.005;
- all invariant booleans;
- primary gate agreement.

It records its own script hash, Git commit/tree hash, environment, input
manifest hash, and output hash, writes
`independent_adjudication.json`, and regenerates checksums.

- [ ] **Step 4: Run adjudicator tests**

Expected: all adjudicator tests pass.

- [ ] **Step 5: Commit Task 5**

```powershell
git add scripts/adjudicate_sbr_v2_audit.py tests/test_sbr_v2_audit_adjudicator.py
git commit -m "Add independent SBR-V2 audit adjudicator"
```

## Task 6: Local Verification, Server Audit, and Stop/Continue Decision

**Files:**
- Modify only if verification exposes a tested defect.

- [ ] **Step 1: Run the complete local SBR suite**

```powershell
& C:\uav_env\Scripts\python.exe -m pytest `
  tests/test_sbr_geometry.py tests/test_sbr_fusion.py `
  tests/test_sbr_metrics.py tests/test_sbr_g0.py `
  tests/test_sbr_artifacts.py tests/test_sbr_cli.py `
  tests/test_sbr_adjudicator.py tests/test_sbr_v2_audit.py `
  tests/test_sbr_v2_audit_cli.py `
  tests/test_sbr_v2_audit_adjudicator.py -q
git diff --check
git status --short
```

Expected: zero failures, clean diff check, clean worktree after commits.

- [ ] **Step 2: Push the exact branch and update the isolated server worktree**

Push `codex/sbr-rtdetr-g0`. Verify the server worktree commit equals local
HEAD and contains no tracked or untracked changes. Use the previously verified
SSH host key; never print credentials or tokens.

- [ ] **Step 3: Freeze the portable input manifest**

Create a manifest outside the source worktree that resolves the existing
G0-A evidence, exact dataset YAML/root, image list, and all recorded hashes.
Write and record its SHA-256 before the audit.

- [ ] **Step 4: Run the primary audit once**

Run `scripts/audit_sbr_v2.py` against the frozen manifest. Do not alter the
output after metrics become visible.

- [ ] **Step 5: Run independent adjudication**

Run `scripts/adjudicate_sbr_v2_audit.py` in a separate process and verify
checksums again.

- [ ] **Step 6: Apply the stop rule**

Continue to a separate V2 implementation/fold plan only when both artifacts
agree on `SBR_V2_AUDIT_ELIGIBLE`. If either reports stop/fail, archive the
negative audit and do not implement V2 or run train folds.

- [ ] **Step 7: Commit only server-guide or evidence-manifest documentation**

Do not commit raw evidence. If documentation changes are required:

```powershell
git add docs/SBR_RTDETR_SERVER_GUIDE.md
git commit -m "Document SBR-V2 causal audit execution"
```
