from __future__ import annotations

import pytest

from scripts.train_rtdetr_ra_glgm import _BoundOptimizerEvidenceMixin


class _Recorder:
    def _record_optimizer_evidence(self, record: dict) -> None:
        self.record = record


class _Subject(_BoundOptimizerEvidenceMixin, _Recorder):
    pass


def test_optimizer_attempt_is_bound_to_run_stage_variant_and_epoch() -> None:
    subject = _Subject()
    subject.epoch = 1
    subject.optimizer_evidence_context = {
        "run_id": "method-smoke",
        "variant": "ra_glgm",
        "stage": "smoke",
        "recovery_generation": 2,
    }

    subject._record_optimizer_evidence(
        {
            "amp_scale_before": 128.0,
            "amp_scale_after": 128.0,
            "gradient_norm": 1.0,
        }
    )

    assert subject.record["run_id"] == "method-smoke"
    assert subject.record["variant"] == "ra_glgm"
    assert subject.record["stage"] == "smoke"
    assert subject.record["recovery_generation"] == 2
    assert subject.record["completed_epoch"] == 2


def test_optimizer_attempt_refuses_missing_authority() -> None:
    subject = _Subject()
    subject.epoch = 0
    with pytest.raises(RuntimeError, match="authority is missing"):
        subject._record_optimizer_evidence({})
