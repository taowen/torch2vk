"""Export quantized FLUX.2 Klein 9B denoiser Vulkan modules.

Run from project root:
    uv run python -m models.quantized_klein9b.export
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from models.quantized_klein9b.export_gguf import REPO_ID
from models.quantized_klein9b.pytorch_modules import Flux2, Klein9BParams
from models.quantized_klein9b.quantization import klein9b_q4_k_m_config
from torch2vk.export import (
    bind_dispatch_function_to_tensors,
    clear_python_modules,
    clear_shader_package,
    count_python_modules,
    export_submodule,
    generate_dispatch_function_source,
    generate_tensor_class_source,
    rename_shader_variant,
    render_model_dispatch_module,
    write_shader_file,
)
from torch2vk.export.reference_codegen import (
    render_exported_reference_function,
    render_reference_module,
)
from torch2vk.export.tensor_codegen import render_tensor_module
from torch2vk.runtime.shader import ShaderVariant


MODEL_PACKAGE = "models.quantized_klein9b"
DEFAULT_IMAGE_SEQ_LEN = 1024
DEFAULT_TEXT_SEQ_LEN = 512


def _write_model_tensors_module(
    path: Path,
    *,
    image_seq_len: int,
    text_seq_len: int,
) -> None:
    path.write_text(
        f'''"""Generated model-level tensor wiring for FLUX.2 Klein 9B."""

from __future__ import annotations

from dataclasses import dataclass

from models.quantized_klein9b.tensors.flux import FLUX_OUTPUT, FluxTensors, create_flux
from torch2vk.runtime.logical import LogicalTensor


@dataclass(frozen=True, slots=True)
class QuantizedKlein9BTensors:
    flux: FluxTensors


_MODEL_TENSORS: QuantizedKlein9BTensors | None = None


def create_model_tensors(
    *,
    image_seq_len: int = {image_seq_len},
    text_seq_len: int = {text_seq_len},
) -> QuantizedKlein9BTensors:
    global _MODEL_TENSORS
    _MODEL_TENSORS = QuantizedKlein9BTensors(
        flux=create_flux(
            "klein9b.flux",
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
            request_state_outputs=frozenset((FLUX_OUTPUT,)),
        )
    )
    return _MODEL_TENSORS


def model_tensors() -> QuantizedKlein9BTensors:
    if _MODEL_TENSORS is None:
        raise RuntimeError("create_model_tensors() must be called before model_tensors()")
    return _MODEL_TENSORS


def flux_output(tensors: QuantizedKlein9BTensors | None = None) -> LogicalTensor:
    resolved = model_tensors() if tensors is None else tensors
    return getattr(resolved.flux, FLUX_OUTPUT)
''',
        encoding="utf-8",
    )


def _write_init(path: Path) -> None:
    path.write_text('"""Generated package."""\n', encoding="utf-8")


def main() -> int:
    output_dir = Path(__file__).parent
    shaders_dir = output_dir / "shaders"
    tensors_dir = output_dir / "tensors"
    dispatch_dir = output_dir / "dispatch"
    shaders_dir.mkdir(exist_ok=True)
    tensors_dir.mkdir(exist_ok=True)
    dispatch_dir.mkdir(exist_ok=True)
    clear_shader_package(shaders_dir)
    clear_python_modules(tensors_dir)
    clear_python_modules(dispatch_dir)

    print("Exporting FLUX.2 Klein 9B denoiser...")
    params = Klein9BParams()
    with torch.device("meta"):
        model = Flux2(params).eval().float()
        args = (
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, params.in_channels, device="meta"),
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, 4, dtype=torch.long, device="meta"),
            torch.zeros(1, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, params.context_in_dim, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, 4, dtype=torch.long, device="meta"),
            torch.zeros(1, device="meta"),
        )
    prog = export_submodule(model, args=args)
    q4_k_m_config = klein9b_q4_k_m_config()
    shape_exprs = {
        DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
        DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
    }
    reference_functions = [
        render_exported_reference_function(
            prog,
            name="flux",
            reference_source="reference",
            tensors="model_tensors().flux",
            frame_name="klein9b.flux",
            policy="q4_tensor",
        )
    ]

    tensor_ctx = generate_tensor_class_source(
        prog,
        class_name="FluxTensors",
        function_name="create_flux",
        quantization_config=q4_k_m_config,
        shape_exprs=shape_exprs,
    )
    (tensors_dir / "flux.py").write_text(render_tensor_module([tensor_ctx]), encoding="utf-8")
    _write_model_tensors_module(
        tensors_dir / "model.py",
        image_seq_len=DEFAULT_IMAGE_SEQ_LEN,
        text_seq_len=DEFAULT_TEXT_SEQ_LEN,
    )
    _write_init(tensors_dir / "__init__.py")

    func_src, shader_imports, used_variants = generate_dispatch_function_source(
        prog,
        class_name="FluxTensors",
        function_name="run_flux",
        shader_package=f"{MODEL_PACKAGE}.shaders",
        quantization_config=q4_k_m_config,
    )
    seen_shader_variants: dict[str, ShaderVariant] = {}
    rename_map: dict[str, str] = {}
    for variant in used_variants.values():
        existing = seen_shader_variants.get(variant.name)
        if existing is None:
            seen_shader_variants[variant.name] = variant
            write_shader_file(shaders_dir, variant)
            continue
        if existing.contract == variant.contract:
            continue
        renamed = rename_shader_variant(variant, f"flux_{variant.name}")
        rename_map[variant.name] = renamed.name
        seen_shader_variants[renamed.name] = renamed
        write_shader_file(shaders_dir, renamed)

    for old_name in sorted(rename_map, key=len, reverse=True):
        new_name = rename_map[old_name]
        func_src = re.sub(rf"\\b{re.escape(old_name.upper())}\\b", new_name.upper(), func_src)
        if old_name in shader_imports:
            shader_imports[new_name] = new_name.upper()
            del shader_imports[old_name]

    (dispatch_dir / "flux.py").write_text(
        render_model_dispatch_module(
            model_package=MODEL_PACKAGE,
            function_name="run_flux",
            tensor_file="flux",
            tensor_class="FluxTensors",
            tensor_expr="model_tensors().flux",
            shader_imports=shader_imports,
            function_source=bind_dispatch_function_to_tensors(func_src),
            uses_quantized_linear_dispatch="run_quantized_linear(" in func_src,
        ),
        encoding="utf-8",
    )
    _write_init(dispatch_dir / "__init__.py")

    (output_dir / "reference.py").write_text(
        render_reference_module(
            model_package=MODEL_PACKAGE,
            model_imports=["from models.quantized_klein9b.pytorch_modules import Flux2"],
            model_type="Flux2",
            reference_functions=reference_functions,
            loader_fields=[],
            loader_sources=[],
        ),
        encoding="utf-8",
    )

    manifest = {
        "repo_id": REPO_ID,
        "image_seq_len": DEFAULT_IMAGE_SEQ_LEN,
        "text_seq_len": DEFAULT_TEXT_SEQ_LEN,
        "generated_by": "models.quantized_klein9b.export",
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  {len(seen_shader_variants)} shader files written")
    print(f"  tensors/ written ({count_python_modules(tensors_dir)} files)")
    print(f"  dispatch/ written ({count_python_modules(dispatch_dir)} files)")
    print("  reference.py written")
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
