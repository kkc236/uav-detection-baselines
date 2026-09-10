from __future__ import annotations

import torch

from src.fdr_head import FDRRTDETRDecoder


def _head() -> FDRRTDETRDecoder:
    return FDRRTDETRDecoder(
        nc=3,
        ch=(8, 12, 16, 20),
        declared_ch_or_options=(8, 12, 16, 20),
        options={
            "hidden_dim": 32,
            "num_queries": 5,
            "num_decoder_layers": 6,
            "reg_max": 32,
            "reg_scale": 4.0,
            "up": 0.5,
            "local_expert": True,
            "local_expert_preserve_base": True,
            "local_expert_hidden": 32,
            "local_expert_heads": 4,
            "local_expert_ff": 64,
            "local_expert_chunk_size": 2,
        },
    )


def _features() -> list[torch.Tensor]:
    return [
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 12, 8, 8),
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 20, 2, 2),
    ]


def _batch() -> dict[str, torch.Tensor]:
    return {
        "cls": torch.tensor([1]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "batch_idx": torch.tensor([0]),
        "gt_groups": [1],
    }


def test_four_input_head_preserves_six_base_layers_and_excludes_dn_from_expert() -> None:
    head = _head().train()
    observed: list[list[tuple[int, ...]]] = []
    handle = head.decoder.local_expert.register_forward_pre_hook(
        lambda _module, args: observed.append([tuple(value.shape) for value in args[4]])
    )
    dec_boxes, dec_scores, _enc_boxes, _enc_scores, dn_meta = head(_features(), _batch())
    handle.remove()

    assert dn_meta is not None and dn_meta["dn_num_split"][1] == 5
    assert dec_boxes.shape[0] == dec_scores.shape[0] == 6
    assert observed == [[(1, 8, 16, 16), (1, 12, 8, 8), (1, 16, 4, 4)]]
    expert = head.decoder.last_expert_prediction
    assert expert is not None
    assert expert.corners.shape == (1, 5, 132)
    assert expert.classes.shape == (1, 5, 3)
    assert expert.boxes.shape == (1, 5, 4)
    torch.testing.assert_close(
        expert.corners, head.decoder.last_corner_logits[-1, :, -5:], rtol=0, atol=0
    )
    torch.testing.assert_close(expert.classes, dec_scores[-1, :, -5:], rtol=0, atol=0)
    torch.testing.assert_close(expert.boxes, dec_boxes[-1, :, -5:], rtol=0, atol=0)


def test_eval_returns_cached_expert_prediction() -> None:
    head = _head().eval()
    with torch.no_grad():
        head.decoder.local_expert.box_out.bias[0] = 0.2
        head.decoder.local_expert.class_out.bias[0] = 0.3
        _postprocessed, raw = head(_features())
    expert = head.decoder.last_expert_prediction
    assert expert is not None
    dec_boxes, dec_scores = raw[:2]
    torch.testing.assert_close(dec_boxes[0], expert.boxes)
    torch.testing.assert_close(dec_scores[0], expert.classes)
    assert not torch.equal(expert.corners, head.decoder.last_corner_logits[-1])


def test_expert_cache_is_cleared_before_a_failed_forward() -> None:
    head = _head().eval()
    with torch.no_grad():
        head(_features())
    assert head.decoder.last_expert_prediction is not None
    try:
        head(_features()[:3])
    except ValueError:
        pass
    else:
        raise AssertionError("four-input contract was not enforced")
    assert head.decoder.last_expert_prediction is None
