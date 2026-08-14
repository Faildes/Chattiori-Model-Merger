from __future__ import annotations

import copy
import gc
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch

try:
    import safetensors
    import safetensors.torch
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safetensors = None
    safe_open = None

# -----------------------------------------------------------------------------
# Shared quantization helpers
# -----------------------------------------------------------------------------

from Utils import (
    _INT8_METADATA_KEYS,
    _INT8_SCALES_KEY,
    _INT8_SCHEME,
    _INT8_SCHEME_KEY,
    _dequantize_int8_theta,
    _encode_int8_scales,
    _info_dict_without_large_quantization_metadata,
    _normalize_int8_compute_dtype,
    _quantize_int8_theta,
    _quantize_int8_theta_inplace,
    _required_quantization_metadata,
)

# -----------------------------------------------------------------------------
# Optional project-local helpers
# -----------------------------------------------------------------------------

try:
    from Utils import (
        detect_arch as _detect_arch_impl,
        normalize_path as _normalize_path_impl,
        prepare_state_dict_for_save as _prepare_state_dict_for_save_impl,
        read_metadata_from_safetensors as _read_st_metadata_impl,
        upcast_fp8_state_dict as _upcast_fp8_state_dict_impl,
        sha256
    )
except Exception:  # pragma: no cover
    _detect_arch_impl = None
    _normalize_path_impl = None
    _prepare_state_dict_for_save_impl = None
    _read_st_metadata_impl = None
    _upcast_fp8_state_dict_impl = None

# -----------------------------------------------------------------------------
# Model info
# -----------------------------------------------------------------------------

@dataclass
class ModelInfo:
    path: Optional[str] = None
    name: Optional[str] = None

    format: Optional[str] = None        # ckpt / safetensors
    model_type: Optional[str] = None    # checkpoint / lora / vae / text_encoder / merged

    arch: dict[str, bool] = field(default_factory=dict)

    sha256: Optional[str] = None
    legacy_hash: Optional[str] = None
    quantization: Optional[str] = None

    tensor_count: int = 0
    param_count: int = 0
    dtype_summary: dict[str, int] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)
    extra_info: dict[str, Any] = field(default_factory=dict)
    parent_models: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "ModelInfo":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "format": self.format,
            "model_type": self.model_type,
            "arch": dict(self.arch),
            "sha256": self.sha256,
            "legacy_hash": self.legacy_hash,
            "quantization": self.quantization,
            "tensor_count": self.tensor_count,
            "param_count": self.param_count,
            "dtype_summary": dict(self.dtype_summary),
            "metadata": copy.deepcopy(self.metadata),
            "extra_info": copy.deepcopy(self.extra_info),
            "parent_models": copy.deepcopy(self.parent_models),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        return cls(
            path=data.get("path"),
            name=data.get("name"),
            format=data.get("format"),
            model_type=data.get("model_type"),
            arch=dict(data.get("arch") or {}),
            sha256=data.get("sha256"),
            legacy_hash=data.get("legacy_hash"),
            quantization=data.get("quantization"),
            tensor_count=int(data.get("tensor_count") or 0),
            param_count=int(data.get("param_count") or 0),
            dtype_summary=dict(data.get("dtype_summary") or {}),
            metadata=copy.deepcopy(data.get("metadata") or {}),
            extra_info=copy.deepcopy(data.get("extra_info") or {}),
            parent_models=copy.deepcopy(data.get("parent_models") or []),
        )

    def to_safetensors_metadata(self) -> dict[str, str]:
        base = {}
        for k, v in (self.metadata or {}).items():
            if v is None:
                continue
            if isinstance(v, str):
                base[str(k)] = v
            else:
                base[str(k)] = json.dumps(v, ensure_ascii=False)

        base["unified_model_info"] = json.dumps(_info_dict_without_large_quantization_metadata(self), ensure_ascii=False)
        if self.model_type is not None:
            base.setdefault("model_type", str(self.model_type))
        if self.format is not None:
            base.setdefault("format", str(self.format))
        return base


# -----------------------------------------------------------------------------
# Unified model
# -----------------------------------------------------------------------------

@dataclass
class UnifiedModel:
    info: ModelInfo = field(default_factory=ModelInfo)
    theta: dict[str, torch.Tensor] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, *, model_type: str = "checkpoint") -> "UnifiedModel":
        return cls(info=ModelInfo(model_type=model_type), theta={})

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        name: Optional[str] = None,
        device: str = "cpu",
        model_type: Optional[str] = None,
        verify_hash: bool = True,
        cache_path: str | None = None,
        int8_compute_dtype: Any = None,
    ) -> "UnifiedModel":
        model = cls(info=ModelInfo(path=_normalize_path(path)))
        if cache_path is not None:
            model.info.extra_info["cache_path"] = _normalize_path(cache_path)
        return model.load(
            device=device,
            name=name,
            model_type=model_type,
            verify_hash=verify_hash,
            cache_path=cache_path,
            int8_compute_dtype=int8_compute_dtype,
        )

    @classmethod
    def from_theta(
        cls,
        theta: dict[str, torch.Tensor],
        *,
        info: ModelInfo | None = None,
        path: Optional[str] = None,
        name: Optional[str] = None,
        model_type: str = "checkpoint",
        arch: Optional[dict[str, bool]] = None,
        metadata: Optional[dict[str, Any]] = None,
        parent_models: Optional[list[dict[str, Any]]] = None,
        clone_tensors: bool = False,
    ) -> "UnifiedModel":
        if info is None:
            info = ModelInfo(
                path=_normalize_path(path) if path else None,
                name=name,
                model_type=model_type,
                metadata=copy.deepcopy(metadata or {}),
                arch=dict(arch or {}),
                parent_models=copy.deepcopy(parent_models or []),
            )
        else:
            info = info.clone()
            if path is not None:
                info.path = _normalize_path(path)
            if name is not None:
                info.name = name
            if metadata is not None:
                info.metadata = copy.deepcopy(metadata)
            if arch is not None:
                info.arch = dict(arch)
            if parent_models is not None:
                info.parent_models = copy.deepcopy(parent_models)
            if model_type is not None:
                info.model_type = model_type

        body = _clone_theta(theta) if clone_tensors else dict(theta)
        obj = cls(info=info, theta=body)
        obj.refresh_info(refresh_arch=(not bool(info.arch)), refresh_hash=False)
        return obj

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Optional[str]:
        return self.info.path

    @path.setter
    def path(self, value: Optional[str]) -> None:
        self.info.path = _normalize_path(value) if value else None

    @property
    def name(self) -> Optional[str]:
        return self.info.name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        self.info.name = value

    @property
    def format(self) -> Optional[str]:
        return self.info.format

    @property
    def model_type(self) -> Optional[str]:
        return self.info.model_type

    @property
    def arch(self) -> dict[str, bool]:
        return self.info.arch

    @property
    def metadata(self) -> dict[str, Any]:
        return self.info.metadata

    @property
    def sha256(self) -> Optional[str]:
        return self.info.sha256

    @property
    def device(self) -> str:
        for tensor in self.theta.values():
            if isinstance(tensor, torch.Tensor):
                return str(tensor.device)
        return "cpu"

    @property
    def num_tensors(self) -> int:
        return len(self.theta)

    @property
    def has_tensors(self) -> bool:
        """True while this model still owns a non-empty state dict.

        Tensor payloads are released by clearing the state dict instead of
        deleting the ``theta`` attribute.  Keeping the attribute stable avoids
        late AttributeError crashes in common cleanup/remerge paths.
        """
        return isinstance(self.theta, dict) and bool(self.theta)

    def release_tensors(self) -> None:
        """Release tensor references without invalidating the UnifiedModel.

        Do not ``del model.theta``.  A UnifiedModel is passed through several
        shared merge/cleanup branches and those branches are allowed to inspect
        ``theta`` after a payload has been consumed.  Clearing the dict releases
        tensor storage while preserving the object's structural contract.
        """
        theta = getattr(self, "theta", None)
        if isinstance(theta, dict):
            theta.clear()
        self.theta = {}

    @property
    def num_parameters(self) -> int:
        return int(sum(int(v.numel()) for v in self.theta.values() if isinstance(v, torch.Tensor)))

    @property
    def is_checkpoint(self) -> bool:
        return self.info.model_type in {"checkpoint", "merged", "vae", "text_encoder"}

    @property
    def is_lora(self) -> bool:
        return self.info.model_type == "lora"

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def clone(self, *, clone_tensors: bool = True) -> "UnifiedModel":
        return UnifiedModel(
            info=self.info.clone(),
            theta=_clone_theta(self.theta) if clone_tensors else dict(self.theta),
        )

    def to(self, device: str | torch.device, *, dtype: torch.dtype | None = None) -> "UnifiedModel":
        new_theta: dict[str, torch.Tensor] = {}
        for key, value in self.theta.items():
            if not isinstance(value, torch.Tensor):
                continue
            if dtype is None:
                new_theta[key] = value.to(device)
            else:
                new_theta[key] = value.to(device=device, dtype=dtype)
        self.theta = new_theta
        return self

    def cpu(self) -> "UnifiedModel":
        return self.to("cpu")

    def detach(self) -> "UnifiedModel":
        self.theta = {k: v.detach() for k, v in self.theta.items() if isinstance(v, torch.Tensor)}
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.theta

    def load(
        self,
        *,
        name: Optional[str] = None,
        device: str = "cpu",
        model_type: Optional[str] = None,
        verify_hash: bool = True,
        cache_path: str | None = None,
        int8_compute_dtype: Any = None,
    ) -> "UnifiedModel":
        if not self.info.path:
            raise ValueError("ModelInfo.path is not set")

        self.info.path = _normalize_path(self.info.path)
        self.info.name = name or self.info.name or _infer_name_from_path(self.info.path)
        self.info.format = _infer_format_from_path(self.info.path)

        if self.info.format == "ckpt":
            self._load_ckpt(self.info.path, device=device)
        elif self.info.format == "safetensors":
            self._load_safetensors(self.info.path, device=device)
        else:
            raise ValueError(f"Unsupported format: {self.info.format}")

        self.theta = _sanitize_theta(self.theta, device=device)
        source_int8 = any(
            isinstance(v, torch.Tensor) and v.dtype in {torch.int8, torch.uint8}
            for v in self.theta.values()
        )
        if source_int8:
            target_dtype = _normalize_int8_compute_dtype(int8_compute_dtype)
            self.theta, _, used_scales = _dequantize_int8_theta(
                self.theta, self.info.metadata, device=device, dtype=target_dtype
            )
            self.info.extra_info["int8_compute_dtype"] = str(target_dtype).replace("torch.", "")
            self.info.extra_info["source_quantization"] = "int8"
            self.info.extra_info["int8_scale_metadata"] = bool(used_scales)
            if not used_scales:
                self.info.extra_info["int8_warning"] = (
                    "No Chattiori scale metadata was found; raw integer values were cast without dequantization scales."
                )
                print(f"[int8] Scale metadata not found; raw int8 values were cast to {target_dtype} without scales.")
        if _upcast_fp8_state_dict_impl is not None and self.theta:
            self.theta = _upcast_fp8_state_dict_impl(self.theta)
            
        if cache_path is not None:
            self.info.extra_info["cache_path"] = _normalize_path(cache_path)
        else:
            cache_path = self.info.extra_info.get("cache_path")

        self.info.model_type = model_type or self.info.model_type or _infer_model_type_from_theta(self.theta)
        self.refresh_info(refresh_arch=True, refresh_hash=verify_hash, cache_path=cache_path)
        return self

    def save(
        self,
        path: str,
        *,
        name: Optional[str] = None,
        format: Optional[str] = None,
        no_metadata: bool = False,
        save_half: bool = False,
        save_quarter: bool = False,
        save_bhalf: bool = False,
        save_int8: bool = False,
        prune: bool = False,
        args: Any = None,
        cache_path: str | None = None,
    ) -> str:
        if not self.theta:
            raise ValueError("Cannot save an empty model (theta is empty)")

        out_path = _normalize_path(path)
        out_format = format or _infer_format_from_path(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # The merge subprocess exits after saving, so prepare the state dict in
        # place.  This avoids holding a complete cloned model alongside the
        # precision-converted output at the RAM peak.
        prepared = self.prepare_for_save(
            save_half=save_half,
            save_quarter=save_quarter,
            save_bhalf=save_bhalf,
            save_int8=save_int8,
            prune=prune,
            args=args,
            in_place=True,
        )

        print(f"Saving as {name or out_path.split('/')[-1]}...")

        try:
            if out_format == "ckpt":
                self._save_ckpt(out_path, prepared, no_metadata=no_metadata)
            elif out_format == "safetensors":
                self._save_safetensors(out_path, prepared, no_metadata=no_metadata)
            else:
                raise ValueError(f"Unsupported save format: {out_format}")
        finally:
            # Drop saver temporaries promptly.  prepared is self.theta for the
            # in-place path, so do not clear the state dict itself here.
            gc.collect()

        if cache_path is not None:
            self.info.extra_info["cache_path"] = _normalize_path(cache_path)
        else:
            cache_path = self.info.extra_info.get("cache_path")

        self.info.path = out_path
        self.info.name = _infer_name_from_path(out_path)
        self.info.format = out_format
        # self.refresh_info(refresh_arch=False, refresh_hash=True, cache_path=cache_path)
        return out_path

    def prepare_for_save(
        self,
        *,
        save_half: bool = False,
        save_quarter: bool = False,
        save_bhalf: bool = False,
        save_int8: bool = False,
        prune: bool = False,
        args: Any = None,
        in_place: bool = False,
    ) -> dict[str, torch.Tensor]:
        # Container-only copy by default.  The old implementation cloned every
        # tensor, doubling model memory before any save conversion occurred.
        theta = self.theta if in_place else dict(self.theta)

        # Remove stale format metadata before preparing a different precision.
        for key in _INT8_METADATA_KEYS:
            self.info.metadata.pop(key, None)

        if self.is_lora:
            if in_place:
                theta = _cast_theta_inplace(
                    theta, save_half=save_half, save_quarter=save_quarter,
                    save_bhalf=save_bhalf
                )
                theta = _make_theta_cpu_contiguous_inplace(theta)
            else:
                theta = _cast_theta(
                    theta, save_half=save_half, save_quarter=save_quarter,
                    save_bhalf=save_bhalf, save_int8=False
                )
                theta = _make_theta_cpu_contiguous(theta)
        else:
            arch = dict(self.info.arch or {})
            if (not arch) and self.theta:
                arch = _infer_arch(self.theta, model_type=self.info.model_type)[0]

            vae_prefix = _guess_vae_prefix(arch)

            if _prepare_state_dict_for_save_impl is not None:
                local_args = args or SimpleNamespace(
                    save_half=save_half,
                    save_quarter=save_quarter,
                    save_bhalf=save_bhalf,
                    save_int8=save_int8,
                    prune=prune,
                )
                theta = _prepare_state_dict_for_save_impl(
                    theta,
                    args=local_args,
                    arch=arch,
                    vae_prefix=vae_prefix,
                    prune=bool(prune),
                    make_cpu=True,
                    make_contiguous=True,
                )
            else:
                if in_place:
                    theta = _cast_theta_inplace(
                        theta, save_half=save_half, save_quarter=save_quarter,
                        save_bhalf=save_bhalf
                    )
                    theta = _make_theta_cpu_contiguous_inplace(theta)
                else:
                    theta = _cast_theta(
                        theta, save_half=save_half, save_quarter=save_quarter,
                        save_bhalf=save_bhalf, save_int8=False
                    )
                    theta = _make_theta_cpu_contiguous(theta)

        if save_int8:
            if in_place:
                scales = _quantize_int8_theta_inplace(theta)
            else:
                theta, scales = _quantize_int8_theta(theta)
            self.info.metadata[_INT8_SCHEME_KEY] = _INT8_SCHEME
            self.info.metadata[_INT8_SCALES_KEY] = _encode_int8_scales(scales)
            self.info.quantization = "int8"
        else:
            self.info.quantization = _infer_quantization(theta)
        return theta

    def refresh_info(
        self,
        *,
        refresh_arch: bool = True,
        refresh_hash: bool = False,
        cache_path: str | None = None,
    ) -> "UnifiedModel":
        self.info.tensor_count = len(self.theta)
        self.info.param_count = self.num_parameters
        self.info.dtype_summary = _dtype_summary(self.theta)
        self.info.quantization = _infer_quantization(self.theta)

        if self.info.name is None and self.info.path:
            self.info.name = _infer_name_from_path(self.info.path)
        if self.info.model_type is None:
            self.info.model_type = _infer_model_type_from_theta(self.theta)

        if refresh_arch and self.theta:
            arch, maybe_theta = _infer_arch(self.theta, model_type=self.info.model_type)
            self.info.arch = arch
            self.theta = maybe_theta

        if cache_path is not None:
            cache_path = _normalize_path(cache_path)
            self.info.extra_info["cache_path"] = cache_path
        else:
            cache_path = self.info.extra_info.get("cache_path")

        if refresh_hash and self.info.path and os.path.isfile(self.info.path):
            title = _title_for_hash(self.info.path, self.info.model_type, self.info.name)
            self.info.sha256 = sha256(self.info.path, title, cache_path)
            self.info.legacy_hash = self.info.legacy_hash or _extract_legacy_hash(self.info.metadata)

        return self

    def register_parent(self, parent: "UnifiedModel", *, role: Optional[str] = None) -> None:
        self.info.parent_models.append(
            {
                "role": role,
                "name": parent.info.name,
                "path": parent.info.path,
                "format": parent.info.format,
                "model_type": parent.info.model_type,
                "sha256": parent.info.sha256,
                "arch": dict(parent.info.arch),
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.info.name,
            "path": self.info.path,
            "format": self.info.format,
            "model_type": self.info.model_type,
            "arch": dict(self.info.arch),
            "sha256": self.info.sha256,
            "quantization": self.info.quantization,
            "tensor_count": self.info.tensor_count,
            "param_count": self.info.param_count,
            "dtype_summary": dict(self.info.dtype_summary),
        }

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_ckpt(self, path: str, *, device: str) -> None:
        raw = torch.load(path, map_location=device)
        metadata: dict[str, Any] = {}

        if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            theta = raw["state_dict"]
            metadata = {k: v for k, v in raw.items() if k != "state_dict"}
        elif isinstance(raw, dict):
            theta = raw
        else:
            raise TypeError(f"Unsupported ckpt payload type: {type(raw)!r}")

        self.theta = _sanitize_theta(theta, device=device)
        self.info.metadata = metadata
        self.info.legacy_hash = _extract_legacy_hash(metadata)

    def _load_safetensors(self, path: str, *, device: str) -> None:
        if safe_open is None or safetensors is None:
            raise ImportError("safetensors is not installed")

        theta: dict[str, torch.Tensor] = {}
        metadata: dict[str, Any] = {}

        with safe_open(path, framework="pt", device=device) as handle:
            metadata = dict(handle.metadata() or {})
            for key in handle.keys():
                theta[key] = handle.get_tensor(key)

        if not metadata and _read_st_metadata_impl is not None:
            metadata = dict(_read_st_metadata_impl(path) or {})

        self.theta = _sanitize_theta(theta, device=device)
        self.info.metadata = metadata
        self.info.legacy_hash = _extract_legacy_hash(metadata)

    # ------------------------------------------------------------------
    # Savers
    # ------------------------------------------------------------------

    def _save_ckpt(
        self,
        path: str,
        theta: dict[str, torch.Tensor],
        *,
        no_metadata: bool,
    ) -> None:
        payload: dict[str, Any] = {"state_dict": theta}
        required_metadata = _required_quantization_metadata(self.info.metadata)
        if not no_metadata:
            payload["metadata"] = _info_dict_without_large_quantization_metadata(self.info)
            payload.update({k: v for k, v in self.info.metadata.items() if k != "state_dict"})
        elif required_metadata:
            # Scale data is part of the int8 storage format, not descriptive metadata.
            payload.update(required_metadata)
        torch.save(payload, path, _use_new_zipfile_serialization=False)

    def _save_safetensors(
        self,
        path: str,
        theta: dict[str, torch.Tensor],
        *,
        no_metadata: bool,
    ) -> None:
        if safetensors is None:
            raise ImportError("safetensors is not installed")

        if no_metadata:
            metadata = _required_quantization_metadata(self.info.metadata) or None
        else:
            metadata = self.info.to_safetensors_metadata()
        safetensors.torch.save_file(theta, path, metadata=metadata)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_path(path: str | os.PathLike[str] | None) -> str | None:
    if path is None:
        return None
    text = os.fspath(path)
    if _normalize_path_impl is not None:
        return _normalize_path_impl(text)
    return os.path.normpath(os.path.expanduser(text))


def _infer_name_from_path(path: str | None) -> Optional[str]:
    if not path:
        return None
    return Path(path).stem


def _infer_format_from_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".ckpt":
        return "ckpt"
    if ext in {".safetensors", ".sft"}:
        return "safetensors"
    raise ValueError(f"Unsupported file extension: {ext}")


def _sanitize_theta(theta: dict[str, Any], *, device: str = "cpu") -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in theta.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().to(device)
            continue
        try:
            out[key] = torch.as_tensor(value, device=device)
        except Exception:
            continue
    return out


def _clone_theta(theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in theta.items() if isinstance(v, torch.Tensor)}


def _make_theta_cpu_contiguous(theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in theta.items():
        out[key] = value.detach().to("cpu").contiguous()
    return out


def _make_theta_cpu_contiguous_inplace(theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for key in list(theta.keys()):
        value = theta[key]
        theta[key] = value.detach().to("cpu").contiguous()
    return theta


def _cast_theta_inplace(
    theta: dict[str, torch.Tensor],
    *,
    save_half: bool = False,
    save_quarter: bool = False,
    save_bhalf: bool = False,
) -> dict[str, torch.Tensor]:
    quarter_dtype = getattr(torch, "float8_e4m3fn", None)
    use_quarter = bool(save_quarter and quarter_dtype is not None)
    for key in list(theta.keys()):
        value = theta[key]
        if not isinstance(value, torch.Tensor):
            continue
        if not value.is_floating_point():
            theta[key] = value.detach()
        elif use_quarter:
            theta[key] = value.detach().to(dtype=quarter_dtype)
        elif save_bhalf:
            theta[key] = value.detach().to(dtype=torch.bfloat16)
        elif save_half:
            theta[key] = value.detach().to(dtype=torch.float16)
        else:
            theta[key] = value.detach()
    return theta



def _cast_theta(
    theta: dict[str, torch.Tensor],
    *,
    save_half: bool = False,
    save_quarter: bool = False,
    save_bhalf: bool = False,
    save_int8: bool = False,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}

    quarter_dtype = getattr(torch, "float8_e4m3fn", None)
    use_quarter = bool(save_quarter and quarter_dtype is not None)

    for key, value in theta.items():
        if not value.is_floating_point():
            out[key] = value.detach()
            continue

        if use_quarter:
            out[key] = value.detach().to(dtype=quarter_dtype)
        elif save_bhalf:
            out[key] = value.detach().to(dtype=torch.bfloat16)
        elif save_half:
            out[key] = value.detach().to(dtype=torch.float16)
        else:
            out[key] = value.detach()
    return out


def _dtype_summary(theta: dict[str, torch.Tensor]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for value in theta.values():
        if not isinstance(value, torch.Tensor):
            continue
        name = str(value.dtype).replace("torch.", "")
        counter[name] = counter.get(name, 0) + 1
    return counter


def _infer_quantization(theta: dict[str, torch.Tensor]) -> Optional[str]:
    dtypes = {str(v.dtype).replace("torch.", "") for v in theta.values() if isinstance(v, torch.Tensor)}
    if not dtypes:
        return None
    if "int8" in dtypes or "uint8" in dtypes:
        return "int8"
    if any(dt.startswith("float8") for dt in dtypes):
        return "fp8"
    if dtypes == {"float16"}:
        return "f16"
    if dtypes == {"bfloat16"}:
        return "bf16"
    if dtypes == {"float32"}:
        return "f32"
    return ",".join(sorted(dtypes))


def _infer_model_type_from_theta(theta: dict[str, torch.Tensor]) -> str:
    if not theta:
        return "checkpoint"

    keys = list(theta.keys())
    lower_keys = [k.lower() for k in keys]

    if any(
        (
            "lora_" in k
            or ".lora_up." in k
            or ".lora_down." in k
            or ".hada_" in k
            or ".oft_" in k
            or ".dora_" in k
            or k.endswith(".alpha")
        )
        for k in lower_keys
    ):
        return "lora"

    vae_ratio = sum(k.startswith(("first_stage_model.", "vae.", "model.vae.")) for k in keys) / max(len(keys), 1)
    if vae_ratio > 0.95:
        return "vae"

    text_ratio = sum(
        k.startswith((
            "cond_stage_model.",
            "conditioner.",
            "text_encoders.",
            "text_model.",
            "transformer.text_model.",
        ))
        for k in keys
    ) / max(len(keys), 1)
    if text_ratio > 0.95:
        return "text_encoder"

    return "checkpoint"


def _fallback_arch(theta: dict[str, torch.Tensor], *, model_type: Optional[str]) -> dict[str, bool]:
    keys = list(theta.keys())
    arch = {
        "XL": False,
        "FLUX": False,
        "ZI": False,
        "AM": False,
        "AM29": False,
        "K2": False,
    }

    if any("conditioner.embedders" in k or "text_encoders.clip_g" in k for k in keys):
        arch["XL"] = True

    if any(
        (
            "double_blocks." in k
            or "single_blocks." in k
            or "img_in." in k
            or "txt_in." in k
            or "time_in." in k
        )
        for k in keys
    ):
        arch["FLUX"] = True

    if any("cap_embedder" in k for k in keys):
        arch["ZI"] = True

    # Krea 2 SingleStreamDiT detection (official Raw/Turbo and common wrappers).
    def _k2strip(k: str) -> str:
        for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
            if k.startswith(prefix):
                return k[len(prefix):]
        return k
    k2keys = [_k2strip(k) for k in keys]
    if ("txtfusion.projector.weight" in k2keys) and (
        "first.weight" in k2keys or any(re.match(r"^blocks\.\d+\.attn\.w[qkvo]\.weight$", k) for k in k2keys)
    ):
        arch["K2"] = True

    # Anima detection.  Keep the checkpoint-side checks structure-specific, and
    # recognize the kohya-style Anima LoRA families used by Anima trainers.
    if (not arch["K2"]) and any("anima" in k.lower() for k in keys):
        arch["AM"] = True

    if (not arch["K2"]) and any(
        (
            k.startswith(("blocks.", "net.blocks.", "diffusion_model.blocks.", "model.diffusion_model.blocks."))
            and (
                ".self_attn." in k
                or ".cross_attn." in k
                or ".q_proj.weight" in k
                or ".k_proj.weight" in k
                or ".v_proj.weight" in k
                or "adaln_modulation" in k
            )
        )
        or k.startswith("cond_stage_model.qwen3_06b.")
        or k.startswith((
            "model.diffusion_model.final_layer",
            "diffusion_model.final_layer",
            "net.final_layer",
            "final_layer",
            "model.diffusion_model.llm_adapter",
            "diffusion_model.llm_adapter",
            "net.llm_adapter",
            "llm_adapter",
            "model.diffusion_model.t_embedder",
            "diffusion_model.t_embedder",
            "net.t_embedder",
            "t_embedder",
            "model.diffusion_model.x_embedder",
            "diffusion_model.x_embedder",
            "net.x_embedder",
            "x_embedder",
        ))
        for k in keys
    ):
        arch["AM"] = True

    if arch["AM"]:
        anima_idxs = set()
        for k in keys:
            m = re.match(r"^(?:(?:model\.diffusion_model\.|diffusion_model\.|net\.)?)blocks\.(\d+)\.", k)
            if m:
                anima_idxs.add(int(m.group(1)))
        arch["AM29"] = bool(anima_idxs and (max(anima_idxs) + 1 >= 40))

    if model_type == "lora":
        if any("conditioner.embedders" in k or "clip_g" in k for k in keys):
            arch["XL"] = True
        if any("double_blocks" in k or "single_blocks" in k for k in keys):
            arch["FLUX"] = True
        if any(
            (
                "lora_unet_blocks_" in k
                or "lora_te_layers_" in k
                or (
                    k.startswith(("blocks.", "net.blocks.", "diffusion_model.blocks.", "model.diffusion_model.blocks."))
                    and (".lora_A.weight" in k or ".lora_B.weight" in k or ".lora_down.weight" in k or ".lora_up.weight" in k)
                )
                or (
                    k.startswith((
                        "diffusion_model.final_layer.", "model.diffusion_model.final_layer.",
                        "net.final_layer.", "final_layer.",
                        "diffusion_model.llm_adapter.blocks.", "model.diffusion_model.llm_adapter.blocks.",
                        "net.llm_adapter.blocks.", "llm_adapter.blocks.",
                        "diffusion_model.llm_adapter.", "model.diffusion_model.llm_adapter.",
                        "net.llm_adapter.", "llm_adapter.",
                        "diffusion_model.t_embedder.", "model.diffusion_model.t_embedder.",
                        "net.t_embedder.", "t_embedder.",
                        "diffusion_model.x_embedder.", "model.diffusion_model.x_embedder.",
                        "net.x_embedder.", "x_embedder.",
                    ))
                    and (".lora_A.weight" in k or ".lora_B.weight" in k or ".lora_down.weight" in k or ".lora_up.weight" in k)
                )
            )
            for k in keys
        ):
            arch["AM"] = True

    if arch["AM"] and not arch["AM29"]:
        anima_idxs = set()
        for k in keys:
            m = re.search(r"(?:^|[._])blocks[._](\d+)[._]", k)
            if m:
                anima_idxs.add(int(m.group(1)))
        arch["AM29"] = bool(anima_idxs and (max(anima_idxs) + 1 >= 40))

    return arch


def _infer_arch(
    theta: dict[str, torch.Tensor],
    *,
    model_type: Optional[str],
) -> tuple[dict[str, bool], dict[str, torch.Tensor]]:
    if not theta:
        return ({}, theta)

    if _detect_arch_impl is not None and model_type != "lora":
        try:
            arch, maybe_theta = _detect_arch_impl(theta)
            return dict(arch or {}), maybe_theta
        except Exception:
            pass

    return _fallback_arch(theta, model_type=model_type), theta


def _guess_vae_prefix(arch: dict[str, bool]) -> str:
    if arch.get("FLUX", False) or arch.get("ZI", False) or arch.get("K2", False):
        return "vae"
    return "first_stage_model"


def _extract_legacy_hash(metadata: dict[str, Any]) -> Optional[str]:
    for key in (
        "legacy_hash",
        "sd_model_hash",
        "sshs_model_hash",
        "model_hash",
        "hash",
    ):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _title_for_hash(path: str, model_type: str | None, name: str | None) -> str:
    stem = name or _infer_name_from_path(path) or os.path.splitext(os.path.basename(path))[0]
    if model_type == "lora":
        return f"lora/{stem}"
    if model_type == "vae":
        return f"vae/{stem}"
    return f"checkpoint/{stem}"