from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping


ULTRALYTICS_VERSION = "8.4.90"

SOURCE_SHA256 = {
    "head.py": "5701116D86881827AC9E1E7462DFAA44C33937BD68E23324763459685729E06F",
    "tasks.py": "B00935C1851BB9CEA240985704C12E654E68B369F6C59DE20E45FA295CB79B92",
    "rtdetr-l.yaml": "85716F626769CB5DDF00D59FCF6CAFB5814AAD196328100BDC7C93306F650E83",
}


@dataclass(frozen=True)
class EBCQPConfig:
    query_budget: int = 300
    p2_candidates: int = 50
    warmup_epochs: int = 3
    tiny_radius: float = 16.0
    p2_anchor_size: float = 0.025
    lambda_p2: float = 0.25
    lambda_ebc: float = 0.05
    quality_weighted_ebc: bool = False
    learnable_fusion_gamma: bool = False
    query_injection_enabled: bool = True
    quality_gated_p2: bool = False
    p2_c2_grad_scale: float = 0.0
    contribution_separated_aux_gradients: bool = False
    lambda_quality: float = 0.25
    local_radius: int = 1
    update_ratio_limit: float = 10.0
    update_ratio_patience: int = 20
    update_monitor_steps: int = 200
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.p2_c2_grad_scale <= 1.0:
            raise ValueError("p2_c2_grad_scale must be in [0, 1]")
        if not self.contribution_separated_aux_gradients:
            return
        if self.query_injection_enabled:
            raise ValueError("contribution-separated gradients require query injection to be disabled")
        if self.lambda_ebc != 0.0 or self.lambda_quality != 0.0:
            raise ValueError("contribution-separated gradients require zero EBC and quality loss weights")
        if self.quality_gated_p2 or self.quality_weighted_ebc or self.learnable_fusion_gamma:
            raise ValueError("contribution-separated gradients require the minimal P2-only branch")

    def as_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def assert_ultralytics_source_lock(paths: Mapping[str, Path]) -> None:
    mismatches = []
    for name, path in paths.items():
        expected = SOURCE_SHA256[name]
        if file_sha256(path) != expected:
            mismatches.append(name)

    if mismatches:
        joined = ", ".join(sorted(mismatches))
        raise RuntimeError(f"Ultralytics source lock mismatch: {joined}")
