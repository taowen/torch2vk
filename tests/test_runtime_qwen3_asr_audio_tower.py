from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors import safe_open

from models.hf_cache import load_config_json, resolve_cached_model
from models.qwen3_asr.execution import run_qwen3_asr_audio_tower
from models.qwen3_asr.pytorch.example import REPO_ID
from models.qwen3_asr.tensors.audio_tower import declare_qwen3_asr_audio_tower_tensors
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
from torch2vk.runtime.session import RuntimeSession


def _load_qwen3_asr_audio_encoder_frontend(model_dir: Path) -> Qwen3ASRAudioEncoder:
    config_dict = dict(load_config_json(model_dir)["thinker_config"]["audio_config"])
    config_dict["encoder_layers"] = 0
    config_dict["num_hidden_layers"] = 0
    model = Qwen3ASRAudioEncoder(Qwen3ASRAudioEncoderConfig(**config_dict))
    state_dict = {}
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as storage:
        for key in storage.keys():
            if not key.startswith("thinker.audio_tower."):
                continue
            local_key = key.removeprefix("thinker.audio_tower.")
            if not local_key.startswith("layers."):
                state_dict[local_key] = storage.get_tensor(key)
    unexpected = model.load_state_dict(state_dict, strict=False).unexpected_keys
    assert not unexpected
    return model.eval()


def test_runtime_compares_qwen3_asr_audio_tower_frame(tmp_path) -> None:
    model_dir = resolve_cached_model(REPO_ID)
    input_shape = (1, 1, 128, 8)
    tensors = declare_qwen3_asr_audio_tower_tensors(input_shape=input_shape)
    rng = np.random.default_rng(0)
    padded_feature_np = rng.normal(0.0, 0.1, size=input_shape).astype(np.float32)
    input_features = padded_feature_np[0, 0]
    feature_lens = np.array([input_features.shape[1]], dtype=np.int64)

    with RuntimeSession.open(
        device_index=0,
        artifact_dir=tmp_path / "generated",
        model_dir=model_dir,
    ) as rt:
        rt.register_inputs(
            {
                tensors.input_features: input_features,
                tensors.feature_lens: feature_lens,
                tensors.padded_feature: padded_feature_np,
            }
        )
        run_qwen3_asr_audio_tower(
            rt,
            tensors,
            pytorch_model=_load_qwen3_asr_audio_encoder_frontend(model_dir),
        )
        final_result = next(
            result for result in rt.compare_results if result.tensor is tensors.conv_out_add_position
        )

    assert final_result.passed
