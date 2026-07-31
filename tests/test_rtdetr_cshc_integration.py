from pathlib import Path

import torch

from src.cshc import CSHCRTDDETRDecoder
from src.rtdetr_cshc import CSHCDetectionModel


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-cshc.yaml"


def test_decoder_keeps_three_stock_memory_levels_but_selects_from_extra_c2_candidates():
    decoder = CSHCRTDDETRDecoder(nc=10, ch=[64, 256, 256, 256], nq=300, candidates=512)
    decoder.train()
    raw = decoder(
        [
            torch.randn(2, 64, 160, 160),
            torch.randn(2, 256, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 256, 20, 20),
        ],
        batch={"cls": torch.empty(0, 1, dtype=torch.long), "bboxes": torch.empty(0, 4), "batch_idx": torch.empty(0, dtype=torch.long)},
    )

    assert raw[0].shape[-2:] == (300, 4)
    assert raw[1].shape[-2:] == (300, 10)
    assert decoder.last_candidates is not None
    assert decoder.last_candidates.objectness_logits.shape[-2:] == (160, 160)


def test_yaml_builds_registered_cshc_decoder_and_preserves_300_queries():
    model = CSHCDetectionModel(CONFIG, ch=3, nc=10, verbose=False).eval()

    assert model.model[-1].num_queries == 300
    with torch.no_grad():
        prediction = model.predict(torch.rand(1, 3, 160, 160))
    assert prediction[0].shape == (1, 300, 6)


def test_model_loss_adds_finite_candidate_map_term():
    model = CSHCDetectionModel(CONFIG, ch=3, nc=10, verbose=False).train()
    batch = {
        "img": torch.rand(1, 3, 160, 160),
        "cls": torch.tensor([[0]]),
        "bboxes": torch.tensor([[0.25, 0.75, 0.02, 0.02]]),
        "batch_idx": torch.tensor([0]),
    }

    loss, items = model.loss(batch)

    assert torch.isfinite(loss)
    assert items.shape[-1] == 4
