from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from src.sqda_sgc import SQDASGCAdapter


BASELINE_SHA256 = "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
MATCHED_AMP_SCALE = 128.0
MATCHED_AMP_GROWTH_INTERVAL = 2**31 - 1


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
    residual_mode: str | None = None,
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
    adapter_c2 = raw_c2.to(dtype=native_queries.dtype)
    enhanced_queries, diagnostics = adapter(
        native_queries,
        reference_boxes,
        adapter_c2,
        identity_override=identity_override,
        residual_mode=residual_mode,
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
        self.nc = int(self.yaml["nc"])
        head = self.model[-1]
        query_count = int(getattr(head, "num_queries", -1))
        hidden_dim = int(getattr(head, "hidden_dim", -1))
        if query_count != 300 or hidden_dim != 256:
            raise ValueError(
                "SQDA-SGC requires a stock RT-DETR head with exactly 300 queries and hidden_dim=256"
            )
        self.sqda_sgc = SQDASGCAdapter(query_count=query_count, hidden_dim=hidden_dim)
        self.identity_override = False
        self.residual_mode: str | None = None
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
                residual_mode=getattr(self, "residual_mode", None),
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


def load_inherited_sqda_adapter(
    target: nn.Module,
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly load an old G2 adapter while permitting only the new gate keys."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_sha = sha256_file(checkpoint_path)
    if expected_sha256 is not None and actual_sha != expected_sha256.upper():
        raise ValueError(
            f"adapter SHA256 mismatch: expected {expected_sha256.upper()}, got {actual_sha}"
        )
    target_adapter = getattr(target, "sqda_sgc", None)
    if not isinstance(target_adapter, SQDASGCAdapter):
        raise TypeError("target must expose an SQDASGCAdapter under '.sqda_sgc'")

    payload = _torch_load_full_checkpoint(checkpoint_path)
    if not isinstance(payload, dict):
        raise TypeError("adapter checkpoint must be a dictionary")
    source_key = next(
        (
            key
            for key in ("ema", "model")
            if isinstance(payload.get(key), nn.Module)
        ),
        None,
    )
    if source_key is None:
        raise TypeError("adapter checkpoint must contain an nn.Module under 'ema' or 'model'")
    source_adapter = getattr(payload[source_key], "sqda_sgc", None)
    if not isinstance(source_adapter, nn.Module):
        raise TypeError("adapter checkpoint source does not expose '.sqda_sgc'")

    source_state = source_adapter.state_dict()
    target_state = target_adapter.state_dict()
    permitted_missing = {
        key for key in target_state if key.startswith("geometry_trust.")
    }
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    mismatched = sorted(
        key
        for key in set(source_state).intersection(target_state)
        if source_state[key].shape != target_state[key].shape
    )
    if set(missing) != permitted_missing or unexpected or mismatched:
        raise RuntimeError(
            "incompatible inherited adapter state: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={mismatched[:5]}"
        )
    result = target_adapter.load_state_dict(source_state, strict=False)
    if set(result.missing_keys) != permitted_missing or result.unexpected_keys:
        raise RuntimeError(
            "strict inherited adapter load failed: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    loaded_state = target_adapter.state_dict()
    unequal = [
        key
        for key, value in source_state.items()
        if not torch.equal(loaded_state[key], value)
    ]
    if unequal:
        raise RuntimeError(f"inherited adapter tensors were not copied exactly: {unequal[:5]}")
    return {
        "path": str(checkpoint_path),
        "sha256": actual_sha,
        "source_key": source_key,
        "inherited_tensors": len(source_state),
        "new_geometry_trust_tensors": len(permitted_missing),
    }


def load_trained_geometry_adapter(
    target: nn.Module,
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly load a completed geometry-gate adapter with every current key present."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_sha = sha256_file(checkpoint_path)
    if expected_sha256 is not None and actual_sha != expected_sha256.upper():
        raise ValueError(
            f"trained adapter SHA256 mismatch: expected {expected_sha256.upper()}, got {actual_sha}"
        )
    target_adapter = getattr(target, "sqda_sgc", None)
    if not isinstance(target_adapter, SQDASGCAdapter):
        raise TypeError("target must expose an SQDASGCAdapter under '.sqda_sgc'")
    payload = _torch_load_full_checkpoint(checkpoint_path)
    if not isinstance(payload, dict):
        raise TypeError("trained adapter checkpoint must be a dictionary")
    source_key = next(
        (
            key
            for key in ("ema", "model")
            if isinstance(payload.get(key), nn.Module)
        ),
        None,
    )
    if source_key is None:
        raise TypeError("trained adapter checkpoint must contain an nn.Module under 'ema' or 'model'")
    source_adapter = getattr(payload[source_key], "sqda_sgc", None)
    if not isinstance(source_adapter, nn.Module):
        raise TypeError("trained adapter checkpoint source does not expose '.sqda_sgc'")
    source_state = source_adapter.state_dict()
    target_state = target_adapter.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    mismatched = sorted(
        key
        for key in set(source_state).intersection(target_state)
        if source_state[key].shape != target_state[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "incompatible trained geometry adapter state: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={mismatched[:5]}"
        )
    target_adapter.load_state_dict(source_state, strict=True)
    loaded_state = target_adapter.state_dict()
    unequal = [
        key
        for key, value in source_state.items()
        if not torch.equal(loaded_state[key], value)
    ]
    if unequal:
        raise RuntimeError(f"trained geometry adapter tensors were not copied exactly: {unequal[:5]}")
    return {
        "path": str(checkpoint_path),
        "sha256": actual_sha,
        "source_key": source_key,
        "adapter_tensors": len(source_state),
    }


def freeze_stock_model(model: nn.Module) -> None:
    """Freeze every stock detector parameter while leaving SQDA-SGC trainable."""
    stock = getattr(model, "model", None)
    adapter = getattr(model, "sqda_sgc", None)
    if not isinstance(stock, nn.Module) or not isinstance(adapter, SQDASGCAdapter):
        raise TypeError("SQDA-SGC training requires '.model' stock layers and a '.sqda_sgc' adapter")
    for parameter in stock.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    for module in stock.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def freeze_inherited_sqda(model: nn.Module) -> None:
    """Freeze stock plus inherited SQDA tensors, leaving only geometry_trust trainable."""
    stock = getattr(model, "model", None)
    adapter = getattr(model, "sqda_sgc", None)
    if not isinstance(stock, nn.Module) or not isinstance(adapter, SQDASGCAdapter):
        raise TypeError("geometry-gate training requires stock layers and an SQDASGCAdapter")
    for parameter in stock.parameters():
        parameter.requires_grad_(False)
    for name, parameter in adapter.named_parameters():
        parameter.requires_grad_(name.startswith("geometry_trust."))
    for module in stock.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def build_sqda_optimizer(model: nn.Module) -> torch.optim.AdamW:
    """Build the exact module-only optimizer without Ultralytics decay scaling."""
    unwrapped = unwrap_model(model)
    adapter = getattr(unwrapped, "sqda_sgc", None)
    if not isinstance(adapter, SQDASGCAdapter):
        raise TypeError("optimizer target does not expose an SQDASGCAdapter")

    matrix_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for _module_name, module in adapter.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                raise RuntimeError("an SQDA-SGC parameter was unexpectedly frozen")
            if id(parameter) in seen:
                raise RuntimeError("an SQDA-SGC parameter appeared in multiple optimizer groups")
            seen.add(id(parameter))
            target = (
                matrix_parameters
                if isinstance(module, nn.Linear) and parameter_name == "weight"
                else no_decay_parameters
            )
            target.append(parameter)

    expected = {id(parameter) for parameter in adapter.parameters()}
    if seen != expected:
        raise RuntimeError("optimizer grouping did not cover every SQDA-SGC parameter exactly once")
    groups = [
        {
            "params": matrix_parameters,
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 1e-4,
            "param_group": "matrix",
        },
        {
            "params": no_decay_parameters,
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "param_group": "no_decay",
        },
    ]
    return torch.optim.AdamW(groups)


def build_geometry_trust_optimizer(model: nn.Module) -> torch.optim.AdamW:
    """Build the exact AdamW optimizer over only the new geometry-trust MLP."""
    unwrapped = unwrap_model(model)
    freeze_inherited_sqda(unwrapped)
    assert_geometry_trust_contract(unwrapped)
    adapter = getattr(unwrapped, "sqda_sgc", None)
    if not isinstance(adapter, SQDASGCAdapter):
        raise TypeError("optimizer target does not expose an SQDASGCAdapter")

    matrix_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for _module_name, module in adapter.geometry_trust.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                raise RuntimeError("a geometry_trust parameter was unexpectedly frozen")
            if id(parameter) in seen:
                raise RuntimeError("a geometry_trust parameter appeared in multiple optimizer groups")
            seen.add(id(parameter))
            target = (
                matrix_parameters
                if isinstance(module, nn.Linear) and parameter_name == "weight"
                else no_decay_parameters
            )
            target.append(parameter)

    expected = {id(parameter) for parameter in adapter.geometry_trust.parameters()}
    if seen != expected:
        raise RuntimeError("optimizer grouping did not cover every geometry_trust parameter")
    return torch.optim.AdamW(
        [
            {
                "params": matrix_parameters,
                "lr": 1e-4,
                "betas": (0.9, 0.999),
                "weight_decay": 1e-4,
                "param_group": "matrix",
            },
            {
                "params": no_decay_parameters,
                "lr": 1e-4,
                "betas": (0.9, 0.999),
                "weight_decay": 0.0,
                "param_group": "no_decay",
            },
        ]
    )


def assert_training_contract(model: nn.Module) -> None:
    unwrapped = unwrap_model(model)
    stock = getattr(unwrapped, "model", None)
    adapter = getattr(unwrapped, "sqda_sgc", None)
    if not isinstance(stock, nn.Module) or not isinstance(adapter, SQDASGCAdapter):
        raise RuntimeError("model does not satisfy the SQDA-SGC training structure")
    trainable_stock = [name for name, parameter in stock.named_parameters() if parameter.requires_grad]
    frozen_adapter = [name for name, parameter in adapter.named_parameters() if not parameter.requires_grad]
    if trainable_stock or frozen_adapter:
        raise RuntimeError(
            f"freeze contract violated: trainable_stock={trainable_stock[:5]}, "
            f"frozen_adapter={frozen_adapter[:5]}"
        )


def assert_geometry_trust_contract(model: nn.Module) -> None:
    """Reject any geometry-gate run that can modify stock or inherited SQDA tensors."""
    unwrapped = unwrap_model(model)
    stock = getattr(unwrapped, "model", None)
    adapter = getattr(unwrapped, "sqda_sgc", None)
    if not isinstance(stock, nn.Module) or not isinstance(adapter, SQDASGCAdapter):
        raise RuntimeError("model does not satisfy the SQDA geometry-gate structure")
    trainable_stock = [name for name, parameter in stock.named_parameters() if parameter.requires_grad]
    trainable_inherited = [
        name
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad and not name.startswith("geometry_trust.")
    ]
    frozen_gate = [
        name
        for name, parameter in adapter.named_parameters()
        if name.startswith("geometry_trust.") and not parameter.requires_grad
    ]
    if trainable_stock or trainable_inherited or frozen_gate:
        raise RuntimeError(
            f"geometry-trust freeze contract violated: trainable_stock={trainable_stock[:5]}, "
            f"trainable_inherited={trainable_inherited[:5]}, frozen_gate={frozen_gate[:5]}"
        )


class SQDASGCTrainer(RTDETRTrainer):
    """Frozen-stock RT-DETR trainer with an exact SQDA-SGC-only AdamW optimizer."""

    def __init__(
        self,
        *args,
        baseline_checkpoint: str | Path,
        baseline_sha256: str = BASELINE_SHA256,
        manifest_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        self.baseline_checkpoint = Path(baseline_checkpoint).expanduser().resolve()
        self.baseline_sha256 = baseline_sha256.upper()
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.baseline_metadata: dict[str, Any] | None = None
        self.last_module_gradient_norm: float | None = None
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides: dict) -> None:
        requested_epochs = overrides.get("epochs")
        requested_project = overrides.get("project")
        requested_name = overrides.get("name")
        requested_exist_ok = overrides.get("exist_ok")
        super().check_resume(overrides)
        if self.resume:
            # Ultralytics restores the checkpoint's original run identity and
            # epoch budget. Restore explicit values so crash recovery resumes
            # in place, while a passed G2 can continue in a separate formal run.
            if requested_epochs is not None:
                self.args.epochs = requested_epochs
            if requested_project is not None:
                self.args.project = requested_project
            if requested_name is not None:
                self.args.name = requested_name
            if requested_exist_ok is not None:
                self.args.exist_ok = requested_exist_ok

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | nn.Module | None = None,
        verbose: bool = True,
    ) -> SQDASGCDetectionModel:
        if weights is not None and not isinstance(weights, SQDASGCDetectionModel):
            raise RuntimeError(
                "SQDA-SGC resume checkpoint must contain SQDASGCDetectionModel weights"
            )
        model = SQDASGCDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if len(model.model) != 29:
            raise RuntimeError(f"expected 29 stock RT-DETR layers, got {len(model.model)}")
        self.baseline_metadata = load_mature_baseline(
            model,
            self.baseline_checkpoint,
            expected_sha256=self.baseline_sha256,
        )
        if weights is not None:
            source_adapter = getattr(weights, "sqda_sgc", None)
            if source_adapter is None:
                raise RuntimeError("SQDA-SGC resume checkpoint is missing adapter weights")
            model.sqda_sgc.load_state_dict(source_adapter.state_dict(), strict=True)
        freeze_stock_model(model)
        self.args.freeze = list(range(len(model.model)))
        self._update_manifest_with_model(model)
        return model

    def _setup_train(self) -> None:
        # Avoid Ultralytics' dynamic AMP probe, then restore the baseline's exact
        # fixed-scale mixed-precision contract after the stock setup is complete.
        self.args.amp = False
        super()._setup_train()
        if not torch.cuda.is_available() or self.device.type != "cuda":
            raise RuntimeError("SQDA-SGC formal training requires CUDA for the fixed AMP contract")
        self.args.amp = True
        self.amp = True
        if hasattr(self.validator, "args"):
            self.validator.args.amp = True
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
            init_scale=MATCHED_AMP_SCALE,
            growth_interval=MATCHED_AMP_GROWTH_INTERVAL,
        )

    def build_optimizer(
        self,
        model: nn.Module,
        name: str = "AdamW",
        lr: float = 1e-4,
        momentum: float = 0.9,
        decay: float = 1e-4,
        iterations: float = 1e5,
    ) -> torch.optim.AdamW:
        del iterations
        if name != "AdamW" or lr != 1e-4 or momentum != 0.9:
            raise RuntimeError(
                f"optimizer protocol changed: name={name}, lr={lr}, momentum={momentum}"
            )
        assert_training_contract(model)
        return build_sqda_optimizer(model)

    def optimizer_step(self) -> None:
        scale_before = float(self.scaler.get_scale())
        if scale_before != MATCHED_AMP_SCALE:
            raise RuntimeError(
                f"fixed AMP scale drifted before optimizer step: {scale_before}"
            )
        self.scaler.unscale_(self.optimizer)
        adapter = unwrap_model(self.model).sqda_sgc
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            adapter.parameters(),
            max_norm=0.1,
        )
        self.last_module_gradient_norm = float(gradient_norm.detach().cpu())
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("SQDA-SGC gradient norm became non-finite")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        if scale_after != MATCHED_AMP_SCALE:
            raise FloatingPointError(
                f"fixed AMP scale drifted after optimizer step: {scale_after}"
            )
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def _update_manifest_with_model(self, model: SQDASGCDetectionModel) -> None:
        if self.manifest_path is None:
            return
        import json

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["baseline"] = self.baseline_metadata
        manifest["model"] = {
            "stock_parameters": sum(parameter.numel() for parameter in model.model.parameters()),
            "adapter_parameters": sum(parameter.numel() for parameter in model.sqda_sgc.parameters()),
            "trainable_keys": [
                f"sqda_sgc.{name}"
                for name, parameter in model.sqda_sgc.named_parameters()
                if parameter.requires_grad
            ],
            "frozen_stock_keys": [
                f"model.{name}"
                for name, parameter in model.model.named_parameters()
                if not parameter.requires_grad
            ],
        }
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)


class SQDAGeometryTrustTrainer(SQDASGCTrainer):
    """SQDA trainer that freezes the inherited G2 adapter and trains only geometry_trust."""

    def __init__(
        self,
        *args,
        adapter_checkpoint: str | Path,
        adapter_sha256: str | None = None,
        **kwargs,
    ) -> None:
        self.adapter_checkpoint = Path(adapter_checkpoint).expanduser().resolve()
        self.adapter_sha256 = adapter_sha256.upper() if adapter_sha256 else None
        self.inherited_adapter_metadata: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | nn.Module | None = None,
        verbose: bool = True,
    ) -> SQDASGCDetectionModel:
        if weights is not None and not isinstance(weights, SQDASGCDetectionModel):
            raise RuntimeError(
                "geometry-gate resume checkpoint must contain SQDASGCDetectionModel weights"
            )
        model = SQDASGCDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if len(model.model) != 29:
            raise RuntimeError(f"expected 29 stock RT-DETR layers, got {len(model.model)}")
        self.baseline_metadata = load_mature_baseline(
            model,
            self.baseline_checkpoint,
            expected_sha256=self.baseline_sha256,
        )
        self.inherited_adapter_metadata = load_inherited_sqda_adapter(
            model,
            self.adapter_checkpoint,
            expected_sha256=self.adapter_sha256,
        )
        if weights is not None:
            source_adapter = getattr(weights, "sqda_sgc", None)
            if not isinstance(source_adapter, SQDASGCAdapter):
                raise RuntimeError("geometry-gate resume checkpoint is missing adapter weights")
            model.sqda_sgc.load_state_dict(source_adapter.state_dict(), strict=True)
        freeze_inherited_sqda(model)
        self.args.freeze = list(range(len(model.model)))
        self._update_manifest_with_model(model)
        return model

    def build_optimizer(
        self,
        model: nn.Module,
        name: str = "AdamW",
        lr: float = 1e-4,
        momentum: float = 0.9,
        decay: float = 1e-4,
        iterations: float = 1e5,
    ) -> torch.optim.AdamW:
        del iterations, decay
        if name != "AdamW" or lr != 1e-4 or momentum != 0.9:
            raise RuntimeError(
                f"geometry-gate optimizer protocol changed: name={name}, lr={lr}, momentum={momentum}"
            )
        return build_geometry_trust_optimizer(model)

    def optimizer_step(self) -> None:
        scale_before = float(self.scaler.get_scale())
        if scale_before != MATCHED_AMP_SCALE:
            raise RuntimeError(
                f"fixed AMP scale drifted before optimizer step: {scale_before}"
            )
        self.scaler.unscale_(self.optimizer)
        geometry_trust = unwrap_model(self.model).sqda_sgc.geometry_trust
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            geometry_trust.parameters(),
            max_norm=0.1,
        )
        self.last_module_gradient_norm = float(gradient_norm.detach().cpu())
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("geometry_trust gradient norm became non-finite")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        if scale_after != MATCHED_AMP_SCALE:
            raise FloatingPointError(
                f"fixed AMP scale drifted after optimizer step: {scale_after}"
            )
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def _update_manifest_with_model(self, model: SQDASGCDetectionModel) -> None:
        super()._update_manifest_with_model(model)
        if self.manifest_path is None:
            return
        import json

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["inherited_adapter"] = self.inherited_adapter_metadata
        manifest["training_scope"] = "sqda_sgc.geometry_trust only"
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)
