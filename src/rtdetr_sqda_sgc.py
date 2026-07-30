from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from ultralytics.nn.tasks import RTDETRDetectionModel

from src.sqda_sgc import SQDASGCAdapter


BASELINE_SHA256 = "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def adapt_decoder_inputs(
    adapter: nn.Module,
    args: tuple,
    kwargs: dict,
    raw_c2: Tensor | None,
    *,
    query_count: int,
    identity_override: bool,
) -> tuple[tuple, dict, dict[str, Tensor], Tensor]:
    """Apply SQDA-SGC to native object-query embeddings while preserving every other decoder input."""
    if raw_c2 is None:
        raise RuntimeError("SQDA-SGC decoder hook reached without a C2 tensor from the same prediction")
    if len(args) < 7:
        raise RuntimeError(f"unexpected RT-DETR decoder signature with {len(args)} positional inputs")
    embed, refer_bbox = args[:2]
    if not isinstance(embed, Tensor) or not isinstance(refer_bbox, Tensor):
        raise RuntimeError("RT-DETR decoder query and reference inputs must be tensors")
    if embed.ndim != 3 or refer_bbox.ndim != 3:
        raise RuntimeError("RT-DETR decoder query and reference inputs must be rank-three tensors")
    if embed.shape[1] < query_count or refer_bbox.shape[1] < query_count:
        raise RuntimeError(
            f"RT-DETR decoder must contain at least {query_count} native object queries"
        )
    if embed.shape[:2] != refer_bbox.shape[:2]:
        raise RuntimeError("RT-DETR decoder query and reference counts do not match")

    native_queries = embed[:, -query_count:, :]
    reference_boxes = refer_bbox[:, -query_count:, :].sigmoid().detach()
    enhanced_queries, diagnostics = adapter(
        native_queries,
        reference_boxes,
        raw_c2,
        identity_override=identity_override,
    )
    if enhanced_queries.shape != native_queries.shape:
        raise RuntimeError(
            f"SQDA-SGC changed native query shape from {native_queries.shape} to {enhanced_queries.shape}"
        )

    if enhanced_queries is native_queries:
        enhanced_embed = embed
    elif embed.shape[1] == query_count:
        enhanced_embed = enhanced_queries
    else:
        enhanced_embed = torch.cat((embed[:, :-query_count, :], enhanced_queries), dim=1)
    new_args = (enhanced_embed, *args[1:])
    return new_args, kwargs, diagnostics, reference_boxes


class SQDASGCDetectionModel(RTDETRDetectionModel):
    """Stock RT-DETR model with a transient, post-Top-300 SQDA-SGC adapter hook."""

    def __init__(
        self,
        cfg: str | Path | dict = "rtdetr-l.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        query_count = int(getattr(head, "num_queries", -1))
        hidden_dim = int(getattr(head, "hidden_dim", -1))
        if query_count != 300 or hidden_dim != 256:
            raise ValueError(
                "SQDA-SGC requires a stock RT-DETR head with exactly 300 queries and hidden_dim=256"
            )
        self.sqda_sgc = SQDASGCAdapter(query_count=query_count, hidden_dim=hidden_dim)
        self.identity_override = False
        self.last_sqda_diagnostics: dict[str, Tensor] | None = None
        self.last_sqda_reference_boxes: Tensor | None = None
        self._sqda_prediction_active = False

    def predict(
        self,
        x,
        profile: bool = False,
        visualize: bool = False,
        batch: dict | None = None,
        augment: bool = False,
        embed: list[int] | None = None,
    ):
        """Run the stock predict path with two forward-local hooks that are always removed."""
        if getattr(self, "_sqda_prediction_active", False):
            raise RuntimeError("nested or concurrent SQDA-SGC prediction on one model is unsupported")
        self._sqda_prediction_active = True
        transient: dict[str, Any] = {"c2": None, "decoder_calls": 0}
        head = self.model[-1]

        def capture_c2(_module: nn.Module, _inputs: tuple, output: Tensor) -> None:
            if not isinstance(output, Tensor) or output.ndim != 4:
                raise RuntimeError("stock RT-DETR layer 1 did not produce a BCHW C2 tensor")
            if output.shape[1] != 128:
                raise RuntimeError(
                    f"stock RT-DETR layer 1 C2 must have 128 channels, got {output.shape[1]}"
                )
            transient["c2"] = output

        def enhance_queries(
            _module: nn.Module,
            hook_args: tuple,
            hook_kwargs: dict,
        ) -> tuple[tuple, dict]:
            if transient["decoder_calls"]:
                raise RuntimeError("stock RT-DETR deformable decoder was invoked more than once")
            transient["decoder_calls"] += 1
            new_args, new_kwargs, diagnostics, reference_boxes = adapt_decoder_inputs(
                self.sqda_sgc,
                hook_args,
                hook_kwargs,
                transient["c2"],
                query_count=int(head.num_queries),
                identity_override=bool(self.identity_override),
            )
            self.last_sqda_diagnostics = diagnostics
            self.last_sqda_reference_boxes = reference_boxes
            transient["c2"] = None
            return new_args, new_kwargs

        c2_handle = self.model[1].register_forward_hook(capture_c2)
        decoder_handle = head.decoder.register_forward_pre_hook(
            enhance_queries,
            with_kwargs=True,
        )
        try:
            result = super().predict(
                x,
                profile=profile,
                visualize=visualize,
                batch=batch,
                augment=augment,
                embed=embed,
            )
            if embed is None and transient["decoder_calls"] != 1:
                raise RuntimeError("stock RT-DETR prediction did not invoke its deformable decoder exactly once")
            return result
        finally:
            c2_handle.remove()
            decoder_handle.remove()
            transient["c2"] = None
            self._sqda_prediction_active = False


def _torch_load_full_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_mature_baseline(
    target: nn.Module,
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly copy the stock sequential model from an immutable mature checkpoint."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_sha = sha256_file(checkpoint_path)
    if expected_sha256 is not None and actual_sha != expected_sha256.upper():
        raise ValueError(
            f"baseline SHA256 mismatch: expected {expected_sha256.upper()}, got {actual_sha}"
        )

    payload = _torch_load_full_checkpoint(checkpoint_path)
    if not isinstance(payload, dict):
        raise TypeError("mature baseline checkpoint must be a dictionary")
    source_key = next(
        (
            key
            for key in ("ema", "model")
            if isinstance(payload.get(key), nn.Module)
        ),
        None,
    )
    if source_key is None:
        raise TypeError("mature baseline checkpoint must contain an nn.Module under 'ema' or 'model'")
    source = payload[source_key].float()
    source_stock = getattr(source, "model", None)
    target_stock = getattr(target, "model", None)
    if not isinstance(source_stock, nn.Module) or not isinstance(target_stock, nn.Module):
        raise TypeError("source and target must expose the stock detector under '.model'")

    source_state = source_stock.state_dict()
    target_state = target_stock.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    mismatched = sorted(
        key
        for key in set(source_state).intersection(target_state)
        if source_state[key].shape != target_state[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "incompatible stock state: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, shape_mismatch={mismatched[:5]}"
        )
    target_stock.load_state_dict(source_state, strict=True)
    if hasattr(source, "names"):
        target.names = source.names
    return {
        "path": str(checkpoint_path),
        "sha256": actual_sha,
        "source_key": source_key,
        "stock_tensors": len(source_state),
    }
