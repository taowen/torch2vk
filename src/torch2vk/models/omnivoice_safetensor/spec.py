"""OmniVoice safetensor model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec


@dataclass(frozen=True, slots=True)
class OmniVoiceSpec:
    qwen3: Qwen3Spec
    audio_vocab_size: int
    audio_mask_id: int
    num_audio_codebook: int


def load_omnivoice_spec(model_dir: str | Path) -> OmniVoiceSpec:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"OmniVoice config is missing: {config_path}")

    config_value = cast("object", json.loads(config_path.read_text(encoding="utf-8")))
    if not isinstance(config_value, dict):
        raise TypeError(f"Expected config object in {config_path}")
    config = cast("dict[str, object]", config_value)

    llm_cfg_value = config.get("llm_config")
    if not isinstance(llm_cfg_value, dict):
        raise TypeError(f"{config_path} must contain object field 'llm_config'")
    llm_cfg = cast("dict[str, object]", llm_cfg_value)

    num_attention_heads = _required_int(llm_cfg, "num_attention_heads", path=config_path)
    hidden_size = _required_int(llm_cfg, "hidden_size", path=config_path)
    return OmniVoiceSpec(
        qwen3=Qwen3Spec(
            model_type="qwen3",
            vocab_size=_required_int(llm_cfg, "vocab_size", path=config_path),
            hidden_size=hidden_size,
            intermediate_size=_required_int(llm_cfg, "intermediate_size", path=config_path),
            num_hidden_layers=_required_int(llm_cfg, "num_hidden_layers", path=config_path),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=_required_int(llm_cfg, "num_key_value_heads", path=config_path),
            head_dim=hidden_size // num_attention_heads,
            hidden_act=_required_str(llm_cfg, "hidden_act", path=config_path),
            max_position_embeddings=_required_int(
                llm_cfg,
                "max_position_embeddings",
                path=config_path,
            ),
            rms_norm_eps=_required_float(llm_cfg, "rms_norm_eps", path=config_path),
            rope_theta=_resolve_rope_theta(llm_cfg, config_path=config_path),
            attention_bias=bool(llm_cfg.get("attention_bias", False)),
            use_cache=bool(llm_cfg.get("use_cache", True)),
            use_sliding_window=bool(llm_cfg.get("use_sliding_window", False)),
            sliding_window=None,
            max_window_layers=0,
        ),
        audio_vocab_size=_required_int(config, "audio_vocab_size", path=config_path),
        audio_mask_id=_required_int(config, "audio_mask_id", path=config_path),
        num_audio_codebook=_required_int(config, "num_audio_codebook", path=config_path),
    )


def _required_int(payload: dict[str, object], key: str, *, path: Path) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{path} missing integer field {key!r}: {value!r}")
    return int(value)


def _required_float(payload: dict[str, object], key: str, *, path: Path) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise TypeError(f"{path} missing numeric field {key!r}: {value!r}")
    return float(value)


def _required_str(payload: dict[str, object], key: str, *, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{path} missing string field {key!r}: {value!r}")
    return value


def _resolve_rope_theta(payload: dict[str, object], *, config_path: Path) -> float:
    direct = payload.get("rope_theta")
    if isinstance(direct, int | float):
        return float(direct)
    rope_params = payload.get("rope_parameters")
    if isinstance(rope_params, dict):
        value = cast("dict[str, object]", rope_params).get("rope_theta")
        if isinstance(value, int | float):
            return float(value)
    raise ValueError(f"{config_path} missing rope_theta in llm_config/rope_parameters")
