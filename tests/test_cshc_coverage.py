def test_coverage_counts_only_new_c2_candidates_against_supplied_stock_misses():
    from scripts.audit_cshc_coverage import summarize_new_candidate_coverage

    result = summarize_new_candidate_coverage(
        missed=[{"image_id": "a", "class_id": 1, "box": [0.4, 0.4, 0.6, 0.6]}],
        candidates=[{"image_id": "a", "class_id": 1, "box": [0.41, 0.41, 0.59, 0.59]}],
        iou_threshold=0.5,
    )

    assert result == {"missed_tiny": 1, "covered_by_new_candidates": 1, "coverage": 1.0}


def test_export_records_use_only_frozen_misses_and_pre_top300_c2_candidates():
    import torch

    from src.cshc import SparseCandidates
    from src.cshc_coverage import c2_candidate_records, frozen_miss_records

    ledger = [
        {
            "image_id": "a.jpg",
            "tiny_targets": [
                {"gt_index": 0, "gt_class": 1, "status": "covered"},
                {"gt_index": 1, "gt_class": 2, "status": "no_boundary_positive"},
            ],
        }
    ]
    misses = frozen_miss_records(
        image_files=["/dataset/images/train/a.jpg"],
        batch_idx=torch.tensor([0, 0]),
        classes=torch.tensor([[1], [2]]),
        boxes_xywh=torch.tensor([[0.20, 0.20, 0.10, 0.10], [0.75, 0.25, 0.20, 0.20]]),
        ledger=ledger,
    )
    candidates = SparseCandidates(
        tokens=torch.zeros(1, 2, 4),
        anchor_logits=torch.logit(torch.tensor([[[0.20, 0.20, 0.10, 0.10], [0.75, 0.25, 0.20, 0.20]]])),
        class_logits=torch.tensor([[[3.0, 0.0, -1.0], [-1.0, 0.0, 3.0]]]),
        objectness_logits=torch.zeros(1, 1, 2, 2),
        indices=torch.tensor([[0, 1]]),
    )
    proposal_records = c2_candidate_records(["/dataset/images/train/a.jpg"], candidates)

    assert misses == [{"image_id": "a.jpg", "class_id": 2, "box": [0.65, 0.15, 0.85, 0.35]}]
    assert proposal_records == [
        {"image_id": "a.jpg", "class_id": 0, "box": [0.15, 0.15, 0.25, 0.25]},
        {"image_id": "a.jpg", "class_id": 2, "box": [0.65, 0.15, 0.85, 0.35]},
    ]


def test_validator_decoder_resolution_accepts_a_single_framework_wrapper():
    from src.cshc_coverage import resolve_cshc_decoder

    class InnerModel:
        model = ["not-the-decoder", "cshc-decoder"]

    class Wrapper:
        model = InnerModel()

    assert resolve_cshc_decoder(Wrapper(), decoder_type=str) == "cshc-decoder"
