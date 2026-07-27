"""Detection-supervised local tiny-expert query adapter."""

from __future__ import annotations

import torch
from torch import nn

from src.gcte_types import QueryEvidence


class DetectionSupervisedTinyExpert(nn.Module):
    """Adapt local decoder evidence without perturbing its initial prediction."""

    def __init__(
        self,
        *,
        query_dim: int = 256,
        num_classes: int = 10,
        adapter_ratio: float = 0.5,
        residual_cap: float = 0.2,
    ) -> None:
        super().__init__()
        if query_dim <= 0 or num_classes <= 0:
            raise ValueError("query_dim and num_classes must be positive")
        if not 0.0 < adapter_ratio <= 1.0:
            raise ValueError("adapter_ratio must be in (0,1]")
        if not 0.0 < residual_cap <= 1.0:
            raise ValueError("residual_cap must be in (0,1]")
        hidden_dim = max(1, int(round(query_dim * adapter_ratio)))
        self.query_dim = int(query_dim)
        self.num_classes = int(num_classes)
        self.residual_cap = float(residual_cap)
        self.norm = nn.LayerNorm(query_dim)
        self.adapter = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, query_dim),
        )
        self.class_head = nn.Linear(query_dim, num_classes)
        self.box_head = nn.Linear(query_dim, 4)
        self.quality_head = nn.Linear(query_dim, 1)
        self.reset_identity_parameters()

    def reset_identity_parameters(self) -> None:
        """Zero residual-producing layers so the initial mapping is exact."""

        for layer in (
            self.adapter[-1],
            self.class_head,
            self.box_head,
            self.quality_head,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, evidence: QueryEvidence) -> QueryEvidence:
        if evidence.query_dim != self.query_dim:
            raise ValueError(
                f"query_dim mismatch: expected {self.query_dim}, "
                f"received {evidence.query_dim}"
            )
        if evidence.num_classes != self.num_classes:
            raise ValueError(
                f"num_classes mismatch: expected {self.num_classes}, "
                f"received {evidence.num_classes}"
            )
        normalized = self.norm(evidence.queries)
        adapted = evidence.queries + self.residual_cap * torch.tanh(
            self.adapter(normalized)
        )
        class_delta = self.residual_cap * torch.tanh(self.class_head(adapted))
        box_delta = self.residual_cap * torch.tanh(self.box_head(adapted))
        quality_delta = self.residual_cap * torch.tanh(
            self.quality_head(adapted)
        )
        return QueryEvidence(
            queries=adapted,
            logits=evidence.logits + class_delta,
            boxes=(evidence.boxes + box_delta).clamp(0.0, 1.0),
            quality=evidence.quality + quality_delta,
        )
