from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperFeatureExtractor

from models.hf_cache import load_config_json, resolve_cached_model
from models.qwen3_asr.execution import run_qwen3_asr_audio_tower
from models.qwen3_asr.pytorch.example import REPO_ID
from models.qwen3_asr.tensors.audio_tower import declare_qwen3_asr_audio_tower_tensors
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
from torch2vk.runtime.session import RuntimeSession

_FIXTURE_WAV = Path(__file__).resolve().parent / "fixtures" / "qwen3_asr_asknot.wav"


def _load_qwen3_asr_audio_encoder(model_dir: Path, *, encoder_layers: int) -> Qwen3ASRAudioEncoder:
    config_dict = dict(load_config_json(model_dir)["thinker_config"]["audio_config"])
    config_dict["encoder_layers"] = encoder_layers
    config_dict["num_hidden_layers"] = encoder_layers
    model = Qwen3ASRAudioEncoder(Qwen3ASRAudioEncoderConfig(**config_dict))
    state_dict = {}
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as storage:
        for key in storage.keys():
            if not key.startswith("thinker.audio_tower."):
                continue
            local_key = key.removeprefix("thinker.audio_tower.")
            if local_key.startswith("layers."):
                layer = int(local_key.split(".", 2)[1])
                if layer < encoder_layers:
                    state_dict[local_key] = storage.get_tensor(key)
            else:
                state_dict[local_key] = storage.get_tensor(key)
    unexpected = model.load_state_dict(state_dict, strict=False).unexpected_keys
    assert not unexpected
    return model.eval()


@pytest.fixture
def qwen3_asr_fixture_wav() -> Path:
    return _FIXTURE_WAV


def _audio_tower_inputs_from_wav(model_dir: Path, wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    if sample_rate != feature_extractor.sampling_rate:
        waveform = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(waveform)).unsqueeze(0),
            orig_freq=sample_rate,
            new_freq=feature_extractor.sampling_rate,
        ).squeeze(0).numpy()
        sample_rate = feature_extractor.sampling_rate
    features = feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="np",
        padding=True,
        truncation=False,
        return_attention_mask=True,
    )
    feature_len = int(features["attention_mask"].sum(axis=-1)[0])
    input_features = np.asarray(features["input_features"][0, :, :feature_len], dtype=np.float32)
    feature_lens = np.array([feature_len], dtype=np.int64)
    return input_features, feature_lens


def test_runtime_compares_qwen3_asr_audio_tower_frame(tmp_path: Path, qwen3_asr_fixture_wav: Path) -> None:
    model_dir = resolve_cached_model(REPO_ID)
    input_features, feature_lens = _audio_tower_inputs_from_wav(model_dir, qwen3_asr_fixture_wav)
    audio_config = load_config_json(model_dir)["thinker_config"]["audio_config"]
    encoder_layers = int(audio_config["encoder_layers"])
    tensors = declare_qwen3_asr_audio_tower_tensors(
        input_features_shape=input_features.shape,
        encoder_layers=encoder_layers,
    )

    with RuntimeSession.open(
        device_index=0,
        artifact_dir=tmp_path / "generated_qwen3_asr_audio_tower",
        model_dir=model_dir,
    ) as rt:
        rt.register_inputs(
            {
                tensors.input_features: input_features,
                tensors.feature_lens: feature_lens,
            }
        )
        run_qwen3_asr_audio_tower(
            rt,
            tensors,
            pytorch_model=_load_qwen3_asr_audio_encoder(
                model_dir,
                encoder_layers=encoder_layers,
            ),
        )
