from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from scripts import export_transient_dcf_fdr as export_cli
from src.fdr_head import (
    DistributionConditionedFeedback,
    FDRDeformableTransformerDecoder,
)
from src.rtdetr_fdr import FDRTrainer
from src.transient_dcf import find_distribution_feedback_decoder
from src.transient_dcf_export import (
    assert_exact_output_structure,
    detach_distribution_feedback,
    load_eligible_schedule_evidence,
    require_zero_feedback_scale,
    strip_distribution_feedback_state,
)


class _GradientPartitionFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        head = nn.Module()
        head.decoder = nn.Module()
        head.decoder.distribution_feedback = nn.Module()
        head.decoder.distribution_feedback.output = nn.Linear(2, 2)
        self.model = nn.ModuleDict({"28": head})

    @property
    def dcf(self) -> nn.Linear:
        return self.model["28"].decoder.distribution_feedback.output


class _Layer(nn.Module):
    pass


class _StockDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(6)])
        self.hidden_dim = 16
        self.num_layers = 6
        self.eval_idx = 5


def _model_with_feedback_decoder() -> nn.Module:
    model = nn.Module()
    model.decoder = FDRDeformableTransformerDecoder.from_stock(
        _StockDecoder(),
        pre_bbox_head=nn.Linear(16, 4),
        distribution_feedback=DistributionConditionedFeedback(
            16, private_seed=10_001
        ),
    )
    return model


def test_dcf_parameters_are_private_gradient_evidence() -> None:
    trainer = object.__new__(FDRTrainer)
    trainer.model = _GradientPartitionFixture()

    groups = trainer.gradient_parameter_groups()

    assert trainer.model.dcf.weight in groups["fdr_gradient_norm"]
    assert trainer.model.backbone.weight in groups["gradient_norm"]


def test_strip_feedback_keys_removes_only_declared_adapter_state() -> None:
    source = {
        "model.28.backbone.weight": torch.ones(1),
        "model.28.decoder.distribution_feedback.output.weight": torch.ones(1),
        "model.28.decoder.distribution_feedback.output.bias": torch.zeros(1),
    }

    clean, removed = strip_distribution_feedback_state(source)

    assert set(clean) == {"model.28.backbone.weight"}
    assert set(removed) == {
        "model.28.decoder.distribution_feedback.output.weight",
        "model.28.decoder.distribution_feedback.output.bias",
    }


def test_strip_feedback_keys_requires_declared_adapter_state() -> None:
    with pytest.raises(ValueError, match="no declared DCF state"):
        strip_distribution_feedback_state({"backbone.weight": torch.ones(1)})


def test_export_rejects_nonzero_feedback_scale() -> None:
    model = _model_with_feedback_decoder()
    find_distribution_feedback_decoder(model).set_distribution_feedback_scale(0.1)
    with pytest.raises(ValueError, match="scale zero"):
        require_zero_feedback_scale(model)


def test_detach_feedback_produces_clean_state_without_mutating_source() -> None:
    source = _model_with_feedback_decoder()
    find_distribution_feedback_decoder(source).set_distribution_feedback_scale(0.0)
    exported = deepcopy(source)

    removed = detach_distribution_feedback(exported)

    assert removed > 0
    assert find_distribution_feedback_decoder(source).distribution_feedback is not None
    assert all(".distribution_feedback." not in name for name in exported.state_dict())


def test_export_requires_tail_eligible_schedule_evidence(tmp_path) -> None:
    evidence = tmp_path / "schedule.jsonl"
    evidence.write_text(
        "\n".join(
            json.dumps(
                {
                    "paper_epoch": epoch,
                    "checkpoint_eligible": epoch >= 75,
                    "scale": 0.0 if epoch >= 75 else 0.1,
                }
            )
            for epoch in (74, 75)
        )
        + "\n",
        encoding="utf-8",
    )

    row = load_eligible_schedule_evidence(evidence, 75)

    assert row["paper_epoch"] == 75
    with pytest.raises(ValueError, match="eligible range"):
        load_eligible_schedule_evidence(evidence, 74)
    with pytest.raises(ValueError, match="no schedule evidence"):
        load_eligible_schedule_evidence(evidence, 76)


def test_exact_output_structure_requires_bit_exact_tensors() -> None:
    assert_exact_output_structure(
        {"boxes": (torch.ones(2), [torch.zeros(1)])},
        {"boxes": (torch.ones(2), [torch.zeros(1)])},
    )
    with pytest.raises(ValueError, match="not bit-exact"):
        assert_exact_output_structure(torch.zeros(1), torch.ones(1))


def test_export_cli_requires_checkpoint_evidence_and_selected_epoch() -> None:
    parser = export_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--checkpoint",
            "best.pt",
            "--schedule-evidence",
            "schedule.jsonl",
            "--paper-epoch",
            "75",
            "--output",
            "clean.pt",
        ]
    )
    assert args.paper_epoch == 75
    assert args.verify_size > 0
