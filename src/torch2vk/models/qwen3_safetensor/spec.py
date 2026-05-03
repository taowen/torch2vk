"""Qwen3 safetensors model specification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class Qwen3Spec:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool = False
    use_cache: bool = True
    use_sliding_window: bool = False
    sliding_window: int | None = None
    max_window_layers: int = 0

    @property
    def q_proj_out_features(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_out_features(self) -> int:
        return self.num_key_value_heads * self.head_dim


def load_qwen3_spec(model_dir: str | Path) -> Qwen3Spec:
    path = Path(model_dir).expanduser().resolve() / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Qwen3 config is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return qwen3_spec_from_config(cast("Mapping[str, Any]", raw), path=path)


def qwen3_spec_from_config(raw: Mapping[str, Any], *, path: Path | None = None) -> Qwen3Spec:
    model_type = _required_str(raw, "model_type", path=path)
    if model_type != "qwen3":
        raise ValueError(f"Expected model_type='qwen3', got {model_type!r}")
    return Qwen3Spec(
        model_type=model_type,
        vocab_size=_required_int(raw, "vocab_size", path=path),
        hidden_size=_required_int(raw, "hidden_size", path=path),
        intermediate_size=_required_int(raw, "intermediate_size", path=path),
        num_hidden_layers=_required_int(raw, "num_hidden_layers", path=path),
        num_attention_heads=_required_int(raw, "num_attention_heads", path=path),
        num_key_value_heads=_required_int(raw, "num_key_value_heads", path=path),
        head_dim=_required_int(raw, "head_dim", path=path),
        hidden_act=_required_str(raw, "hidden_act", path=path),
        max_position_embeddings=_required_int(raw, "max_position_embeddings", path=path),
        rms_norm_eps=_required_float(raw, "rms_norm_eps", path=path),
        rope_theta=_required_float(raw, "rope_theta", path=path),
        attention_bias=_optional_bool(raw, "attention_bias", default=False, path=path),
        use_cache=_optional_bool(raw, "use_cache", default=True, path=path),
        use_sliding_window=_optional_bool(raw, "use_sliding_window", default=False, path=path),
        sliding_window=_optional_int_or_none(raw, "sliding_window", path=path),
        max_window_layers=_optional_int(raw, "max_window_layers", default=0, path=path),
    )


def _label(key: str, path: Path | None) -> str:
    return key if path is None else f"{path}:{key}"


def _required_int(raw: Mapping[str, Any], key: str, *, path: Path | None) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{_label(key, path)} must be an int, got {value!r}")
    return value


def _optional_int(raw: Mapping[str, Any], key: str, *, default: int, path: Path | None) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int):
        raise TypeError(f"{_label(key, path)} must be an int, got {value!r}")
    return value


def _optional_int_or_none(raw: Mapping[str, Any], key: str, *, path: Path | None) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError(f"{_label(key, path)} must be an int or null, got {value!r}")
    return value


def _required_float(raw: Mapping[str, Any], key: str, *, path: Path | None) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float):
        raise TypeError(f"{_label(key, path)} must be a float, got {value!r}")
    return float(value)


def _required_str(raw: Mapping[str, Any], key: str, *, path: Path | None) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{_label(key, path)} must be a non-empty string, got {value!r}")
    return value


def _optional_bool(raw: Mapping[str, Any], key: str, *, default: bool, path: Path | None) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{_label(key, path)} must be a bool, got {value!r}")
    return value
