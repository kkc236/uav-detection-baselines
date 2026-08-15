"""Frozen Ultralytics RT-DETR evidence adapter for IBER-BE."""

from __future__ import annotations

from functools import partial
from numbers import Integral
from threading import Lock, RLock, local
from typing import Any
from weakref import WeakKeyDictionary, finalize, ref

import torch
from torch import nn
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.modules.transformer import DeformableTransformerDecoder
from ultralytics.nn.modules.utils import inverse_sigmoid

from src.iber_head import IBEROutput, IBERRefiner
from src.itber_geometry import cxcywh_to_xyxy
from src.itber_loss import ITBERLosses, itber_private_loss


_ADAPTER_OWNERS = WeakKeyDictionary()
_ADAPTER_OWNERS_LOCK = RLock()


class _AdapterCleanupState:
    """Non-module lifecycle state; checkpoints intentionally remain state_dict-only."""

    def __init__(
        self,
        detector: nn.Module,
        head: nn.Module,
        detector_requires_grad: tuple[tuple[nn.Parameter, bool], ...],
        detector_training: tuple[tuple[nn.Module, bool], ...],
    ) -> None:
        self.lock = RLock()
        self.detector = detector
        self.head = head
        self.detector_requires_grad = detector_requires_grad
        self.detector_training = detector_training
        self.owner_reference: Any | None = None
        self.head_hook: Any | None = None
        self.original_decoder: nn.Module | None = None
        self.installed_decoder: IBERRecordingDecoder | None = None
        self.replaced_decoder = False
        self.closed = False


def _cleanup_adapter_state(state: _AdapterCleanupState) -> None:
    """Best-effort all cleanup phases without retaining the adapter."""
    with _ADAPTER_OWNERS_LOCK:
        with state.lock:
            if state.closed:
                return
            state.closed = True

            if state.head_hook is not None:
                try:
                    state.head_hook.remove()
                except Exception:
                    pass
                state.head_hook = None

            try:
                if (
                    state.replaced_decoder
                    and state.installed_decoder is not None
                    and state.head.decoder is state.installed_decoder
                ):
                    state.head.decoder = state.original_decoder
            except Exception:
                pass

            for parameter, requires_grad in state.detector_requires_grad:
                try:
                    parameter.requires_grad_(requires_grad)
                except Exception:
                    pass
            for module, training in state.detector_training:
                try:
                    module.training = training
                except Exception:
                    pass

            if _ADAPTER_OWNERS.get(state.detector) is state:
                _ADAPTER_OWNERS.pop(state.detector, None)


def _finalize_adapter_cleanup(state: _AdapterCleanupState) -> None:
    _cleanup_adapter_state(state)


def _capture_head_input_weak(
    owner_reference: Any,
    module: nn.Module,
    inputs: tuple[Any, ...],
) -> None:
    owner = owner_reference()
    if owner is None or owner._closed:
        return
    owner._capture_head_input(module, inputs)


def _require_normal_query_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value != 300:
        raise ValueError(f"{name} must be an integral value exactly 300")
    return int(value)


def _active_adapter(detector: nn.Module) -> "FrozenIBERAdapter | None":
    """Return the live owner, discarding stale weak ownership records."""
    cleanup_state = _ADAPTER_OWNERS.get(detector)
    if cleanup_state is None:
        return None
    owner_reference = cleanup_state.owner_reference
    owner = owner_reference() if owner_reference is not None else None
    if owner is None or owner._closed:
        _cleanup_adapter_state(cleanup_state)
        return None
    return owner


class IBERRecordingDecoder(nn.Module):
    """Reproduce the stock decoder and detach only its final evidence."""

    def __init__(
        self,
        layers: nn.ModuleList,
        hidden_dim: int,
        num_layers: int,
        eval_idx: int,
        *,
        normal_query_count: int = 300,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.eval_idx = int(eval_idx)
        self.normal_query_count = _require_normal_query_count(
            normal_query_count,
            name="normal_query_count",
        )
        self.last_hidden: torch.Tensor | None = None
        self.last_stock_scores: torch.Tensor | None = None
        self.last_stock_boxes: torch.Tensor | None = None

    @classmethod
    def from_stock(
        cls,
        stock: nn.Module,
        *,
        normal_query_count: int = 300,
    ) -> "IBERRecordingDecoder":
        """Reuse every stock decoder layer without changing its state keys."""
        wrapped = cls(
            layers=stock.layers,
            hidden_dim=stock.hidden_dim,
            num_layers=stock.num_layers,
            eval_idx=stock.eval_idx,
            normal_query_count=normal_query_count,
        )
        wrapped.training = stock.training
        return wrapped

    def _record(
        self,
        hidden: torch.Tensor,
        score: torch.Tensor,
        stock_box: torch.Tensor,
    ) -> None:
        evidence = {
            "hidden": hidden,
            "score": score,
            "stock_box": stock_box,
        }
        if any(value.ndim < 2 for value in evidence.values()):
            raise ValueError("stock evidence must include batch and query dimensions")
        batch_queries = hidden.shape[:2]
        if any(value.shape[:2] != batch_queries for value in evidence.values()):
            raise ValueError("stock evidence batch and query dimensions must agree")
        query_count = hidden.shape[1]
        if query_count < self.normal_query_count:
            raise ValueError(
                "stock evidence has fewer queries than the configured normal query "
                f"count: {query_count} < {self.normal_query_count}"
            )
        selection = slice(
            query_count - self.normal_query_count,
            query_count,
        )
        self.last_hidden = hidden[:, selection].detach()
        self.last_stock_scores = score[:, selection].detach()
        self.last_stock_boxes = stock_box[:, selection].detach()

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the stock equations while recording only the final state."""
        self.last_hidden = None
        self.last_stock_scores = None
        self.last_stock_boxes = None
        output = embed
        decoded_boxes: list[torch.Tensor] = []
        decoded_scores: list[torch.Tensor] = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()

        for index, layer in enumerate(self.layers):
            output = layer(
                output,
                refer_bbox,
                feats,
                shapes,
                padding_mask,
                attn_mask,
                pos_mlp(refer_bbox),
            )
            bbox = bbox_head[index](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                score = score_head[index](output)
                stock_box = (
                    refined_bbox
                    if index == 0
                    else torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox))
                )
                decoded_scores.append(score)
                decoded_boxes.append(stock_box)
                if index == self.num_layers - 1:
                    self._record(output, score, stock_box)
            elif index == self.eval_idx:
                score = score_head[index](output)
                decoded_scores.append(score)
                decoded_boxes.append(refined_bbox)
                self._record(output, score, refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(decoded_boxes), torch.stack(decoded_scores)


class FrozenIBERAdapter(nn.Module):
    """Keep RT-DETR immutable and train only the private IBER refiner."""

    def __init__(
        self,
        detector: nn.Module,
        refiner: IBERRefiner,
        criterion: RTDETRDetectionLoss,
        *,
        rho: float,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.refiner = refiner
        self.criterion = criterion
        self.rho = float(rho)
        self.output_mode = "refined"
        self.last_match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.last_output: IBEROutput | None = None
        self._last_f3: torch.Tensor | None = None
        self._head_hook: Any | None = None
        self._closed = False
        self._owns_detector = False
        self._original_decoder: nn.Module | None = None
        self._installed_decoder: IBERRecordingDecoder | None = None
        self._replaced_decoder = False
        self._evidence_lock = Lock()
        self._evidence_local = local()
        self._detector_requires_grad = tuple(
            (parameter, parameter.requires_grad)
            for parameter in detector.parameters()
        )
        self._detector_training = tuple(
            (module, module.training) for module in detector.modules()
        )

        head = detector.model[-1]
        if type(head.decoder) not in {
            DeformableTransformerDecoder,
            IBERRecordingDecoder,
        }:
            raise TypeError(
                "FrozenIBERAdapter accepts only the pinned stock decoder or "
                "IBERRecordingDecoder"
            )

        cleanup_state = _AdapterCleanupState(
            detector,
            head,
            self._detector_requires_grad,
            self._detector_training,
        )
        self._cleanup_state = cleanup_state
        owner_reference = ref(self)
        cleanup_state.owner_reference = owner_reference
        self._cleanup_finalizer = finalize(
            self,
            _finalize_adapter_cleanup,
            cleanup_state,
        )

        with _ADAPTER_OWNERS_LOCK:
            if _active_adapter(detector) is not None:
                raise RuntimeError("RT-DETR detector already has an active owner")
            _ADAPTER_OWNERS[detector] = cleanup_state
            self._owns_detector = True
            try:
                self._head_hook = head.register_forward_pre_hook(
                    partial(_capture_head_input_weak, owner_reference)
                )
                cleanup_state.head_hook = self._head_hook
                self.detector.requires_grad_(False)
                self.detector.eval()
            except BaseException:
                self._rollback_takeover()
                raise

    @classmethod
    def from_detector(
        cls,
        detector: nn.Module,
        *,
        private_seed: int,
        probe: str = "b3",
        image_size: int = 640,
        rho: float = 0.05,
        normal_query_count: int = 300,
    ) -> "FrozenIBERAdapter":
        with _ADAPTER_OWNERS_LOCK:
            if _active_adapter(detector) is not None:
                raise RuntimeError("RT-DETR detector already has an active owner")

            head = detector.model[-1]
            head_query_count = _require_normal_query_count(
                head.num_queries,
                name="RT-DETR head.num_queries",
            )
            requested_query_count = _require_normal_query_count(
                normal_query_count,
                name="normal_query_count",
            )
            if requested_query_count != head_query_count:
                raise ValueError(
                    "normal_query_count must equal RT-DETR head.num_queries"
                )

            original_decoder = head.decoder
            if type(original_decoder) is IBERRecordingDecoder:
                wrapped_query_count = _require_normal_query_count(
                    original_decoder.normal_query_count,
                    name="existing wrapped decoder normal_query_count",
                )
                if wrapped_query_count != head_query_count:
                    raise ValueError(
                        "wrapped decoder and RT-DETR head query counts differ"
                    )
                candidate_decoder = original_decoder
                replace_decoder = False
            elif type(original_decoder) is DeformableTransformerDecoder:
                candidate_decoder = IBERRecordingDecoder.from_stock(
                    original_decoder,
                    normal_query_count=requested_query_count,
                )
                replace_decoder = True
            else:
                raise TypeError(
                    "FrozenIBERAdapter refuses a foreign RT-DETR decoder wrapper"
                )

            first_projection = head.input_proj[0][0]
            f3_channels = int(first_projection.in_channels)
            hidden_dim = int(candidate_decoder.hidden_dim)
            device_parameter = next(detector.parameters())
            refiner = IBERRefiner(
                hidden_dim=hidden_dim,
                f3_channels=f3_channels,
                private_seed=private_seed,
                probe=probe,
                image_size=image_size,
                rho=rho,
            ).to(device=device_parameter.device, dtype=device_parameter.dtype)
            nc = int(getattr(detector, "nc", detector.yaml["nc"]))
            criterion = RTDETRDetectionLoss(nc=nc, use_vfl=True)

            adapter: FrozenIBERAdapter | None = None
            try:
                adapter = cls(detector, refiner, criterion, rho=rho)
                object.__setattr__(adapter, "_original_decoder", original_decoder)
                object.__setattr__(adapter, "_installed_decoder", candidate_decoder)
                adapter._replaced_decoder = replace_decoder
                adapter._cleanup_state.original_decoder = original_decoder
                adapter._cleanup_state.installed_decoder = candidate_decoder
                adapter._cleanup_state.replaced_decoder = replace_decoder
                if replace_decoder:
                    candidate_decoder.eval()
                    head.decoder = candidate_decoder
                return adapter
            except BaseException:
                if adapter is not None:
                    adapter.close()
                raise

    def _finish_cleanup(self) -> None:
        self._closed = True
        try:
            _cleanup_adapter_state(self._cleanup_state)
        finally:
            self._owns_detector = False
            self._head_hook = None
            try:
                self._cleanup_finalizer.detach()
            except Exception:
                pass

    def _rollback_takeover(self) -> None:
        self._finish_cleanup()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("FrozenIBERAdapter is closed")

    def close(self) -> None:
        """Release the detector and restore the state owned by this adapter."""
        if self._closed:
            return
        if getattr(self._evidence_local, "active", False):
            raise RuntimeError("cannot close during recursive evidence capture")
        with self._evidence_lock:
            if self._closed:
                return
            self._finish_cleanup()

    def __enter__(self) -> "FrozenIBERAdapter":
        self._require_open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _capture_head_input(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], (list, tuple)) or not inputs[0]:
            raise RuntimeError("RT-DETR head did not receive a feature pyramid")
        self._last_f3 = inputs[0][0].detach()

    def train(self, mode: bool = True) -> "FrozenIBERAdapter":
        """Toggle the private module while permanently locking detector eval."""
        self._require_open()
        super().train(mode)
        self.detector.requires_grad_(False)
        self.detector.eval()
        return self

    def set_output_mode(self, mode: str) -> None:
        if mode not in {"stock", "refined", "boundary_off"}:
            raise ValueError("IBER output mode must be stock, refined, or boundary_off")
        self.output_mode = mode

    def selected_boxes(self, output: IBEROutput) -> torch.Tensor:
        return output.select_boxes(self.output_mode)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return stock-score Top-300 detections with the selected boxes."""
        self._require_open()
        output = self.forward_evidence(image)
        return self.detector.model[-1].postprocess(
            self.selected_boxes(output),
            output.stock_scores.sigmoid(),
        )

    def forward_evidence(self, image: torch.Tensor) -> IBEROutput:
        """Run one immutable detector forward and invoke the private head."""
        self._require_open()
        if getattr(self._evidence_local, "active", False):
            raise RuntimeError("recursive IBER evidence capture is not allowed")
        with self._evidence_lock:
            self._require_open()
            self._evidence_local.active = True
            try:
                self.detector.eval()
                self._last_f3 = None
                detached_image = image.detach()
                with torch.no_grad():
                    self.detector.predict(detached_image)
                decoder = self.detector.model[-1].decoder
                last_hidden = getattr(decoder, "last_hidden", None)
                last_stock_scores = getattr(decoder, "last_stock_scores", None)
                last_stock_boxes = getattr(decoder, "last_stock_boxes", None)
                if (
                    last_hidden is None
                    or last_stock_scores is None
                    or last_stock_boxes is None
                    or self._last_f3 is None
                ):
                    raise RuntimeError("frozen RT-DETR evidence capture is incomplete")
                return self.refiner(
                    last_hidden,
                    last_stock_boxes,
                    last_stock_scores,
                    self._last_f3,
                    detached_image,
                )
            finally:
                self._evidence_local.active = False

    def training_step(self, batch: dict[str, torch.Tensor]) -> ITBERLosses:
        """Match stock outputs once and compute only the private objective."""
        self._require_open()
        image = batch["img"]
        output = self.forward_evidence(image)
        self.last_output = output
        target_boxes = batch["bboxes"].detach().to(
            device=image.device, dtype=output.stock_boxes.dtype
        )
        target_classes = batch["cls"].detach().to(
            device=image.device, dtype=torch.long
        ).view(-1)
        batch_index = batch["batch_idx"].detach().to(
            device=image.device, dtype=torch.long
        ).view(-1)
        groups = [
            int((batch_index == index).sum()) for index in range(image.shape[0])
        ]
        self.last_match_indices = self.criterion.matcher(
            output.stock_boxes.detach(),
            output.stock_scores.detach(),
            target_boxes,
            target_classes,
            groups,
        )
        return itber_private_loss(
            output,
            target_edges=cxcywh_to_xyxy(target_boxes),
            match_indices=self.last_match_indices,
            rho=self.rho,
        )
