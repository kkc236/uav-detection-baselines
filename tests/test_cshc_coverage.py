def test_coverage_counts_only_new_c2_candidates_against_supplied_stock_misses():
    from scripts.audit_cshc_coverage import summarize_new_candidate_coverage

    result = summarize_new_candidate_coverage(
        missed=[{"image_id": "a", "class_id": 1, "box": [0.4, 0.4, 0.6, 0.6]}],
        candidates=[{"image_id": "a", "class_id": 1, "box": [0.41, 0.41, 0.59, 0.59]}],
        iou_threshold=0.5,
    )

    assert result == {"missed_tiny": 1, "covered_by_new_candidates": 1, "coverage": 1.0}
