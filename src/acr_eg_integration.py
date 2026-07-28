"""YAML-configured ACR-EG integration around a mature RT-DETR detector.

The detector remains the shared evidence producer.  ACR-EG is registered as
an ordinary child module and is invoked by this wrapper's forward path, so its
parameters are present in ``state_dict()``, optimizer parameter groups, and
checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn

from src.gcqf import GCQF, GCQFOutput
from src.gcte_types import QueryEvidence, ViewGeometry


@dataclass(frozen=True)
class ACREGConfig:
    enabled: bool = True
    forward_integration: bool = True
    query_dim: int = 256
    num_classes: int = 10
    num_heads: int = 8
    num_views: int = 4
    residual_eta: float = 0.2
    residual_enabled: bool = True
    acr_eg_off: bool = False
    gcte_off: bool = False

    def __post_init__(self) -> None:
        if self.query_dim <= 0 or self.num_classes <= 0:
            raise ValueError("ACR-EG dimensions must be positive")
        if self.num_heads <= 0 or self.query_dim % self.num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        if self.num_views != 4:
            raise ValueError("ACR-EG freezes four local views")
        if not 0.0 < self.residual_eta <= 1.0:
            raise ValueError("residual_eta must be in (0,1]")
        if not self.forward_integration:
            raise ValueError("ACR-EG must be enabled in the model forward")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ACREGConfig":
        fields = {
            "enabled": bool(value.get("enabled", True)),
            "forward_integration": bool(value.get("forward_integration", True)),
            "query_dim": int(value.get("query_dim", 256)),
            "num_classes": int(value.get("num_classes", 10)),
            "num_heads": int(value.get("num_heads", 8)),
            "num_views": int(value.get("num_views", 4)),
            "residual_eta": float(value.get("residual_eta", 0.2)),
            "residual_enabled": bool(value.get("residual_enabled", True)),
            "acr_eg_off": bool(value.get("acr_eg_off", False)),
            "gcte_off": bool(value.get("gcte_off", False)),
        }
        return cls(**fields)


def load_acr_eg_config(path: str | Path) -> ACREGConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ACR-EG YAML root must be a mapping")
    block = payload.get("gcte")
    if not isinstance(block, Mapping):
        raise ValueError("ACR-EG YAML must contain a gcte mapping")
    config = ACREGConfig.from_mapping(block)
    if not config.enabled and not config.gcte_off:
        raise ValueError("disabled ACR-EG must set gcte_off")
    return config


@dataclass(frozen=True)
class ACREGForwardOutput:
    global_evidence: QueryEvidence
    module_output: GCQFOutput | None


class ACREGIntegratedRTDETR(nn.Module):
    """Register a mature detector and invoke ACR-EG in one forward path."""

    def __init__(self, detector: nn.Module, config: ACREGConfig) -> None:
        super().__init__()
        self.detector = detector
        self.config = config
        self.acr_eg = GCQF(
            query_dim=config.query_dim,
            num_classes=config.num_classes,
            num_heads=config.num_heads,
            num_views=config.num_views,
            residual_eta=config.residual_eta,
        )

    @property
    def acr_eg_enabled(self) -> bool:
        return bool(
            self.config.enabled
            and not self.config.gcte_off
            and self.config.forward_integration
        )

    def detector_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call the shared mature RT-DETR detector without bypassing the wrapper."""

        return self.detector(*args, **kwargs)

    def forward(
        self,
        *,
        global_evidence: QueryEvidence,
        local_evidence: QueryEvidence,
        geometry: ViewGeometry,
        anchor_mask: torch.Tensor,
        residual_enabled: bool | None = None,
    ) -> ACREGForwardOutput:
        if not self.acr_eg_enabled:
            return ACREGForwardOutput(
                global_evidence=global_evidence,
                module_output=None,
            )
        effective_residual = (
            self.config.residual_enabled
            if residual_enabled is None
            else bool(residual_enabled)
        )
        if self.config.acr_eg_off:
            effective_residual = False
        output = self.acr_eg(
            global_evidence,
            local_evidence,
            geometry,
            anchor_mask=anchor_mask,
            residual_enabled=effective_residual,
        )
        return ACREGForwardOutput(
            global_evidence=global_evidence,
            module_output=output,
        )


def _require_sha256(name: str, value: str) -> str:
    normalized = str(value).upper()
    if len(normalized) != 64 or any(
        character not in "0123456789ABCDEF" for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return normalized


def build_integrated_artifact(
    wrapper: ACREGIntegratedRTDETR,
    *,
    baseline_sha256: str,
    module_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    commit = str(source_commit).lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("source_commit must be an exact Git SHA")
    state = {
        name: value.detach().cpu().clone()
        for name, value in wrapper.state_dict().items()
    }
    if not any(name.startswith("detector.") for name in state):
        raise ValueError("integrated checkpoint has no detector state")
    if not any(name.startswith("acr_eg.") for name in state):
        raise ValueError("integrated checkpoint has no ACR-EG state")
    return {
        "schema_version": "gcte-acr-eg-integrated/v1",
        "source_commit": commit,
        "baseline_sha256": _require_sha256(
            "baseline_sha256",
            baseline_sha256,
        ),
        "module_sha256": _require_sha256(
            "module_sha256",
            module_sha256,
        ),
        "config": asdict(wrapper.config),
        "wrapper_state": state,
    }


__all__ = [
    "ACREGConfig",
    "ACREGForwardOutput",
    "ACREGIntegratedRTDETR",
    "build_integrated_artifact",
    "load_acr_eg_config",
]
