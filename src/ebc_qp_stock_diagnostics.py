from __future__ import annotations

from dataclasses import dataclass, field
import statistics

import torch


@dataclass
class StockTinyQueryAccumulator:
    query_budget: int = 300
    tiny_radius: float = 16.0
    tiny_gt: int = 0
    covered_tiny_gt: int = 0
    best_ranks: list[int] = field(default_factory=list)
    normalized_best_ranks: list[float] = field(default_factory=list)
    candidate_counts: list[int] = field(default_factory=list)

    def update(self, scores: torch.Tensor, centers: torch.Tensor, batch: dict) -> None:
        if scores.ndim != 2 or centers.shape != (*scores.shape, 2):
            raise ValueError("stock scores/centers must have shapes [B,N] and [B,N,2]")
        if scores.shape[0] != int(batch["img"].shape[0]):
            raise ValueError("stock diagnostic batch size mismatch")

        image_height, image_width = batch["img"].shape[-2:]
        batch_indices = batch["batch_idx"].reshape(-1).long().to(scores.device)
        boxes = batch["bboxes"].reshape(-1, 4).float().to(scores.device)
        candidate_count = int(scores.shape[1])
        stable_index = torch.arange(candidate_count, device=scores.device)

        for image_index in range(scores.shape[0]):
            image_boxes = boxes[batch_indices == image_index]
            radius = (
                (image_boxes[:, 2] * image_width) * (image_boxes[:, 3] * image_height)
            ).clamp_min(0).sqrt()
            tiny_boxes = image_boxes[radius <= self.tiny_radius]
            if not len(tiny_boxes):
                continue

            order = torch.argsort(scores[image_index].float(), descending=True, stable=True)
            ranks = torch.empty_like(stable_index)
            ranks[order] = stable_index + 1
            image_centers = centers[image_index].float()
            box_min = tiny_boxes[:, :2] - tiny_boxes[:, 2:] / 2
            box_max = tiny_boxes[:, :2] + tiny_boxes[:, 2:] / 2
            inside = (
                (image_centers[None] >= box_min[:, None]).all(-1)
                & (image_centers[None] <= box_max[:, None]).all(-1)
            )
            missing_rank = candidate_count + 1
            for row in inside:
                best_rank = int(ranks[row].min().item()) if bool(row.any()) else missing_rank
                self.tiny_gt += 1
                self.covered_tiny_gt += int(best_rank <= self.query_budget)
                self.best_ranks.append(best_rank)
                self.normalized_best_ranks.append(best_rank / candidate_count)
                self.candidate_counts.append(candidate_count)

    def compute(self) -> dict[str, float | int | list[int]]:
        if not self.best_ranks:
            return {
                "tiny_gt": 0,
                "stock_top300_coverage": 0.0,
                "best_rank_mean": 0.0,
                "best_rank_median": 0.0,
                "normalized_best_rank_mean": 0.0,
                "normalized_best_rank_median": 0.0,
                "candidate_count_values": [],
            }
        return {
            "tiny_gt": self.tiny_gt,
            "stock_top300_coverage": self.covered_tiny_gt / self.tiny_gt,
            "best_rank_mean": statistics.mean(self.best_ranks),
            "best_rank_median": statistics.median(self.best_ranks),
            "normalized_best_rank_mean": statistics.mean(self.normalized_best_ranks),
            "normalized_best_rank_median": statistics.median(self.normalized_best_ranks),
            "candidate_count_values": sorted(set(self.candidate_counts)),
        }


class StockQueryProbe:
    def __init__(self, decoder) -> None:
        if not hasattr(decoder, "enc_score_head"):
            raise TypeError("decoder has no stock encoder score head")
        self.decoder = decoder
        self.pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self.handle = decoder.enc_score_head.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output: torch.Tensor) -> None:
        anchors = getattr(self.decoder, "anchors", None)
        if anchors is None or anchors.shape[1] != output.shape[1]:
            raise RuntimeError("stock anchors are unavailable during query diagnostics")
        centers = anchors.sigmoid()[..., :2].expand(output.shape[0], -1, -1)
        self.pending = (output.detach().float().max(-1).values, centers.detach().float())

    def consume(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pending is None:
            raise RuntimeError("stock query probe did not observe an encoder score forward")
        value = self.pending
        self.pending = None
        return value

    def close(self) -> None:
        self.handle.remove()
