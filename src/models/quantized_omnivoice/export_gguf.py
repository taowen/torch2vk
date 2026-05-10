"""Export OmniVoice safetensors to Q4_K_M GGUF via llama.cpp."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open

from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.pytorch.example import REPO_ID
from torch2vk.checkpoints.gguf import GGUFTensorType, open_gguf_mmap


REPO_ROOT = Path(__file__).resolve().parents[3]
LLAMA_ROOT = Path.home() / "projects" / "agentorch" / "third_party" / "llama.cpp"
LLAMA_GGUF_PY = LLAMA_ROOT / "gguf-py"
QUANTIZE_GGUF_ARCH = "clip"
DEFAULT_Q4_K_M_GGUF = REPO_ROOT / "dist" / "quantized_omnivoice" / "model.gguf"
Q8_TENSOR_NAMES = (
    "llm.embed_tokens.weight",
    "audio_embeddings.weight",
    "audio_heads.weight",
)

if LLAMA_GGUF_PY.exists():
    sys.path.insert(0, str(LLAMA_GGUF_PY))


def export_omnivoice_q4_k_m_gguf(
    *,
    model_dir: str | Path | None = None,
    output: str | Path = DEFAULT_Q4_K_M_GGUF,
    overwrite: bool = False,
) -> Path:
    q4_path = Path(output).expanduser().resolve()
    if q4_path.exists() and not overwrite and _gguf_matches_quantization(q4_path):
        return q4_path
    _export_q4_k_m_gguf(
        model_dir=resolve_cached_model(REPO_ID) if model_dir is None else Path(model_dir),
        output=q4_path,
        overwrite=overwrite,
    )
    return q4_path


def _export_q4_k_m_gguf(*, model_dir: Path, output: Path, overwrite: bool) -> None:
    if not LLAMA_GGUF_PY.exists():
        raise FileNotFoundError(f"Missing llama.cpp gguf-py package: {LLAMA_GGUF_PY}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    f16_output = output.with_suffix(".f16.tmp.gguf")
    if f16_output.exists():
        f16_output.unlink()
    _export_f16_gguf(model_dir=model_dir, output=f16_output)
    _run_llama_quantize(input_path=f16_output, output_path=output)
    f16_output.unlink()


def _export_f16_gguf(*, model_dir: Path, output: Path) -> None:
    config = _load_config(model_dir)
    gguf = _load_gguf_module()
    writer = gguf.GGUFWriter(path=None, arch=QUANTIZE_GGUF_ARCH)
    _add_omnivoice_metadata(writer=writer, config=config, file_type=gguf.LlamaFileType.MOSTLY_F16)
    for name, tensor in _iter_safetensor_tensors(model_dir):
        data, dtype = _tensor_to_f16_or_f32(tensor)
        writer.add_tensor(name, data, raw_dtype=dtype)
    writer.write_header_to_file(path=output)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def _run_llama_quantize(*, input_path: Path, output_path: Path) -> None:
    quantize = LLAMA_ROOT / "build" / "bin" / "llama-quantize"
    if not quantize.exists():
        raise FileNotFoundError(f"Missing llama.cpp quantizer: {quantize}")
    command = [str(quantize)]
    for name in Q8_TENSOR_NAMES:
        command.extend(["--tensor-type", f"{name}=q8_0"])
    command.extend([str(input_path), str(output_path), "Q4_K_M"])
    subprocess.run(
        command,
        check=True,
    )


def _gguf_matches_quantization(path: Path) -> bool:
    with open_gguf_mmap(path) as gguf:
        for name in Q8_TENSOR_NAMES:
            if gguf.entry(name).ggml_type is not GGUFTensorType.Q8_0:
                return False
    return True


def _iter_safetensor_tensors(model_dir: Path) -> Iterator[tuple[str, torch.Tensor]]:
    safetensor_paths = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_paths:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    for safetensor_path in safetensor_paths:
        with cast(Any, safe_open(safetensor_path, framework="pt", device="cpu")) as checkpoint:
            for name in checkpoint.keys():
                yield cast(str, name), cast(torch.Tensor, checkpoint.get_tensor(name))


def _tensor_to_f16_or_f32(tensor: torch.Tensor) -> tuple[np.ndarray, Any]:
    gguf = _load_gguf_module()
    array = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
    if array.ndim <= 1:
        return np.asarray(array, dtype=np.float32), gguf.GGMLQuantizationType.F32
    return np.ascontiguousarray(array.astype(np.float16)), gguf.GGMLQuantizationType.F16


def _load_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    config = cast(object, json.loads(config_path.read_text(encoding="utf-8")))
    if not isinstance(config, dict):
        raise TypeError(f"Expected config object in {config_path}")
    return cast(dict[str, object], config)


def _add_omnivoice_metadata(*, writer: Any, config: dict[str, object], file_type: Any) -> None:
    gguf = _load_gguf_module()
    writer.add_name("OmniVoice")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_file_type(file_type)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_uint32(f"{QUANTIZE_GGUF_ARCH}.audio_vocab_size", _config_int(config, "audio_vocab_size", 1025))
    writer.add_uint32(f"{QUANTIZE_GGUF_ARCH}.num_audio_codebook", _config_int(config, "num_audio_codebook", 8))
    writer.add_uint32(f"{QUANTIZE_GGUF_ARCH}.audio_mask_id", _config_int(config, "audio_mask_id", 1024))


def _config_int(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int):
        raise TypeError(f"Expected integer config value {key}, got {value!r}")
    return value


def _load_gguf_module() -> Any:
    if not LLAMA_GGUF_PY.exists():
        raise FileNotFoundError(f"Missing llama.cpp gguf-py package: {LLAMA_GGUF_PY}")
    return importlib.import_module("gguf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_Q4_K_M_GGUF)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = export_omnivoice_q4_k_m_gguf(
        model_dir=args.model_dir,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
