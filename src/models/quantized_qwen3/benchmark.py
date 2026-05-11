"""Compare torch2vk Qwen3 Vulkan generation with llama.cpp Vulkan Q4_K_M.

Run from project root:
    uv run python -m models.quantized_qwen3.benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

from models.hf_cache import resolve_cached_model
from models.quantized_qwen3.export_gguf import REPO_ID
from models.quantized_qwen3.input_prep import (
    DEFAULT_PROMPT,
    load_qwen3_tokenizer,
    prepare_qwen3_inputs,
)
from models.quantized_qwen3.run import Qwen3RunResult, main as run_torch2vk_qwen3


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LLAMA_CPP_ROOT = Path.home() / "projects" / "agentorch" / "third_party" / "llama.cpp"
DEFAULT_LLAMA_GGUF_DIR = REPO_ROOT / "dist" / "llama_cpp_qwen3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--llama-cpp-root", type=Path, default=DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--llama-gguf-dir", type=Path, default=DEFAULT_LLAMA_GGUF_DIR)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    model_dir = resolve_cached_model(REPO_ID)
    tokenizer = load_qwen3_tokenizer(model_dir)
    prepared = prepare_qwen3_inputs(tokenizer=tokenizer, prompt=args.prompt)
    torch2vk_result = run_torch2vk_qwen3(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    llama_q4 = ensure_llama_cpp_q4_k_m(
        model_dir=model_dir,
        llama_cpp_root=args.llama_cpp_root,
        output_dir=args.llama_gguf_dir,
    )
    llama_result = run_llama_bench(
        llama_cpp_root=args.llama_cpp_root,
        model=llama_q4,
        prompt_tokens=prepared.prompt_length,
        decode_tokens=max(args.max_new_tokens - 1, 1),
        repetitions=args.repetitions,
    )
    print(json.dumps(_summary(torch2vk_result, llama_result), indent=2, ensure_ascii=False))


def ensure_llama_cpp_q4_k_m(
    *,
    model_dir: Path,
    llama_cpp_root: Path,
    output_dir: Path,
) -> Path:
    convert = llama_cpp_root / "convert_hf_to_gguf.py"
    quantize = llama_cpp_root / "build" / "bin" / "llama-quantize"
    if not convert.is_file():
        raise FileNotFoundError(f"llama.cpp converter not found: {convert}")
    if not quantize.is_file():
        raise FileNotFoundError(f"llama.cpp quantizer not found: {quantize}")

    output_dir.mkdir(parents=True, exist_ok=True)
    f16_path = output_dir / "qwen3-0.6b-f16.gguf"
    q4_path = output_dir / "qwen3-0.6b-q4_k_m.gguf"
    env = os.environ.copy()
    gguf_py = llama_cpp_root / "gguf-py"
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(gguf_py) if old_pythonpath is None else f"{gguf_py}:{old_pythonpath}"

    if not f16_path.is_file():
        _run(
            [
                sys.executable,
                str(convert),
                str(model_dir),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ],
            env=env,
        )
    if not q4_path.is_file():
        _run([str(quantize), str(f16_path), str(q4_path), "Q4_K_M"], env=env)
    return q4_path


def run_llama_bench(
    *,
    llama_cpp_root: Path,
    model: Path,
    prompt_tokens: int,
    decode_tokens: int,
    repetitions: int,
) -> list[dict[str, object]]:
    bench = llama_cpp_root / "build" / "bin" / "llama-bench"
    if not bench.is_file():
        raise FileNotFoundError(f"llama.cpp benchmark binary not found: {bench}")
    completed = _run(
        [
            str(bench),
            "-m",
            str(model),
            "-p",
            str(prompt_tokens),
            "-n",
            str(decode_tokens),
            "-ngl",
            "99",
            "-fa",
            "1",
            "-r",
            str(repetitions),
            "-o",
            "json",
        ],
    )
    return _parse_llama_json(completed.stdout)


def _summary(
    torch2vk_result: Qwen3RunResult,
    llama_result: list[dict[str, object]],
) -> dict[str, object]:
    llama_prefill = _find_llama_case(llama_result, n_prompt=torch2vk_result.prompt_length)
    llama_decode = _find_llama_decode_case(llama_result)
    return {
        "torch2vk": {
            **asdict(torch2vk_result),
            "prefill_tokens_per_second": torch2vk_result.prefill_tokens_per_second,
            "decode_tokens_per_second": (
                1000.0 / torch2vk_result.decode_ms_per_token
                if torch2vk_result.decode_ms_per_token > 0.0
                else 0.0
            ),
        },
        "llama_cpp": {
            "model": llama_decode.get("model_filename"),
            "backend": llama_decode.get("backends"),
            "prompt_tokens_per_second": llama_prefill.get("avg_ts"),
            "decode_tokens_per_second": llama_decode.get("avg_ts"),
        },
    }


def _find_llama_case(
    cases: list[dict[str, object]],
    *,
    n_prompt: int,
) -> dict[str, object]:
    for case in cases:
        if case.get("n_prompt") == n_prompt:
            return case
    raise ValueError(f"llama-bench output did not contain n_prompt={n_prompt}")


def _find_llama_decode_case(cases: list[dict[str, object]]) -> dict[str, object]:
    for case in cases:
        value = case.get("n_gen")
        if isinstance(value, int) and value > 0:
            return case
    raise ValueError("llama-bench output did not contain a decode case")


def _parse_llama_json(output: str) -> list[dict[str, object]]:
    start = output.find("[")
    if start < 0:
        raise ValueError(f"llama-bench did not print a JSON array:\n{output}")
    loaded = json.loads(output[start:])
    if not isinstance(loaded, list):
        raise TypeError(f"Expected llama-bench JSON list, got {type(loaded).__name__}")
    return cast(list[dict[str, object]], loaded)


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command))
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )


if __name__ == "__main__":
    main()
