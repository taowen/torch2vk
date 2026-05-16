"""Export quantized FLUX.2 Klein 9B denoiser Vulkan modules.

Run from project root:
    uv run python -m models.quantized_klein9b.export
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from einops import rearrange
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from models.quantized_klein9b.autoencoder import AutoEncoder, AutoEncoderParams
from models.quantized_klein9b.custom_shaders import (
    KLEIN9B_CAPTURE_QWEN3_CTX_F16,
    KLEIN9B_EULER_UPDATE_F16,
)
from models.quantized_klein9b.export_gguf import REPO_ID
from models.quantized_klein9b.model_sources import resolve_model_dirs
from models.quantized_klein9b.pytorch_modules import Flux2, Klein9BParams
from models.quantized_klein9b.quantization import (
    ae_q4_k_m_config,
    klein9b_q4_k_m_config,
    qwen3_text_encoder_q8_config,
)
from torch2vk.export import (
    bind_dispatch_function_to_tensors,
    cast_floating_tensors,
    clear_python_modules,
    clear_shader_package,
    count_python_modules,
    export_submodule,
    generate_dispatch_function_source,
    generate_tensor_class_source,
    module_floating_dtype,
    rename_shader_variant,
    ReferencePolicy,
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
DEFAULT_LATENT_HEIGHT = 32
DEFAULT_LATENT_WIDTH = 32


class AEDecodeModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        ae = AutoEncoder(AutoEncoderParams())
        self.decoder = ae.decoder
        self.bn = ae.bn
        self.bn_eps = ae.bn_eps
        self.ps = ae.ps

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = tokens.reshape(1, DEFAULT_LATENT_HEIGHT, DEFAULT_LATENT_WIDTH, 128)
        z = z.permute(0, 3, 1, 2).contiguous()
        running_var = self.bn.running_var
        running_mean = self.bn.running_mean
        if running_var is None or running_mean is None:
            raise RuntimeError("AutoEncoder BatchNorm running stats are required for decode")
        scale = torch.sqrt(running_var.view(1, -1, 1, 1) + self.bn_eps)
        mean = running_mean.view(1, -1, 1, 1)
        z = z * scale + mean
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return self.decoder(z)


def _write_model_tensors_module(
    path: Path,
    *,
    image_seq_len: int,
    text_seq_len: int,
    num_text_layers: int,
    text_hidden_size: int,
    text_head_dim: int,
) -> None:
    path.write_text(
        f'''"""Generated model-level tensor wiring for FLUX.2 Klein 9B."""

from __future__ import annotations

from dataclasses import dataclass

from models.quantized_klein9b.tensors.ae_decode import (
    AE_DECODE_OUTPUT,
    AeDecodeTensors,
    create_ae_decode,
)
from models.quantized_klein9b.tensors.embed_tokens import EmbedTokensTensors, create_embed_tokens
from models.quantized_klein9b.tensors.flux import FLUX_OUTPUT, FluxTensors, create_flux
from models.quantized_klein9b.tensors.rope import RopeTableTensors, create_rope_table
from models.quantized_klein9b.tensors.text_layer import TextLayerTensors, create_text_layer
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.logical import (
    bind_logical_tensor_names,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSpec,
)


@dataclass(frozen=True, slots=True)
class QuantizedKlein9BTensors:
    input_ids: LogicalTensor
    text_rope: RopeTableTensors
    text_embed: EmbedTokensTensors
    text_layers: tuple[TextLayerTensors, ...]
    ctx: LogicalTensor
    latent_tokens: LogicalTensor
    flux: FluxTensors
    ae_decode: AeDecodeTensors | None


_MODEL_TENSORS: QuantizedKlein9BTensors | None = None


def create_model_tensors(
    *,
    image_seq_len: int = {image_seq_len},
    text_seq_len: int = {text_seq_len},
    num_text_layers: int = {num_text_layers},
    text_hidden_size: int = {text_hidden_size},
    text_head_dim: int = {text_head_dim},
    include_ae_decode: bool = True,
) -> QuantizedKlein9BTensors:
    global _MODEL_TENSORS
    input_ids = _host_input_tensor("int64", (1, text_seq_len))
    text_rope = create_rope_table(
        "klein9b.text.rope",
        batch=1,
        sequence_length=text_seq_len,
        head_dim=text_head_dim,
    )
    text_embed = create_embed_tokens(
        "klein9b.text.embed",
        sequence_length=text_seq_len,
        input=input_ids,
    )
    text_layers_list: list[TextLayerTensors] = []
    text_hidden = text_embed.embedding
    for layer_idx in range(num_text_layers):
        layer_tensors = create_text_layer(
            f"klein9b.text.layer.{{layer_idx}}",
            layer_idx=layer_idx,
            sequence_length=text_seq_len,
            hidden_states=text_hidden,
            position_embeddings_0=text_rope.cos,
            position_embeddings_1=text_rope.sin,
        )
        text_layers_list.append(layer_tensors)
        text_hidden = layer_tensors.add_7
    text_layers = tuple(text_layers_list)
    ctx = _request_state_tensor("float16", (1, text_seq_len, text_hidden_size * 3))
    latent_tokens = _request_state_tensor("float16", (1, image_seq_len, 128))
    flux = create_flux(
        "klein9b.flux",
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        x=latent_tokens,
        ctx=ctx,
        request_state_outputs=frozenset((FLUX_OUTPUT,)),
    )
    ae_decode = (
        create_ae_decode(
            "klein9b.ae_decode",
            tokens=latent_tokens,
            request_state_outputs=frozenset((AE_DECODE_OUTPUT,)),
        )
        if include_ae_decode
        else None
    )
    _MODEL_TENSORS = QuantizedKlein9BTensors(
        input_ids=input_ids,
        text_rope=text_rope,
        text_embed=text_embed,
        text_layers=text_layers,
        ctx=ctx,
        latent_tokens=latent_tokens,
        flux=flux,
        ae_decode=ae_decode,
    )
    bind_logical_tensor_names(_MODEL_TENSORS)
    return _MODEL_TENSORS


def model_tensors() -> QuantizedKlein9BTensors:
    if _MODEL_TENSORS is None:
        raise RuntimeError("create_model_tensors() must be called before model_tensors()")
    return _MODEL_TENSORS


def flux_output(tensors: QuantizedKlein9BTensors | None = None) -> LogicalTensor:
    resolved = model_tensors() if tensors is None else tensors
    return getattr(resolved.flux, FLUX_OUTPUT)


def image_output(tensors: QuantizedKlein9BTensors | None = None) -> LogicalTensor:
    resolved = model_tensors() if tensors is None else tensors
    if resolved.ae_decode is None:
        raise RuntimeError("AE decode tensors were not created")
    return getattr(resolved.ae_decode, AE_DECODE_OUTPUT)


def _host_input_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def _request_state_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )
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

    seen_shader_variants: dict[str, ShaderVariant] = {}
    shader_file_count = 0

    def write_generated_shader(variant: ShaderVariant) -> None:
        nonlocal shader_file_count
        existing = seen_shader_variants.get(variant.name)
        if existing is not None:
            if existing.contract != variant.contract:
                raise ValueError(f"shader name conflict after rename: {variant.name}")
            return
        seen_shader_variants[variant.name] = variant
        write_shader_file(shaders_dir, variant)
        shader_file_count += 1

    for custom_variant in (KLEIN9B_CAPTURE_QWEN3_CTX_F16, KLEIN9B_EULER_UPDATE_F16):
        write_generated_shader(custom_variant)

    reference_functions: list[str] = []

    def export_one(
        *,
        dispatch_name: str,
        tensor_file: str,
        tensor_class: str,
        create_function: str,
        tensor_expr: str,
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, object] | None = None,
        checkpoint: str,
        weight_prefix: str = "",
        quantization_config=None,
        shape_exprs: dict[int, str] | None = None,
        parameters_source: str = "",
        arguments_source: str = "",
        reference_name: str | None = None,
        reference_policy: ReferencePolicy | None = None,
        extra_dispatch_functions: str = "",
    ) -> None:
        module = module.float()
        export_dtype = module_floating_dtype(module)
        if export_dtype is not None:
            args = cast_floating_tensors(args, export_dtype)
            kwargs = cast_floating_tensors(kwargs, export_dtype)
        prog = export_submodule(module, args=args, kwargs=kwargs)
        func_name = dispatch_name.removeprefix("run_")
        if reference_name is not None and reference_policy is not None:
            reference_functions.append(
                render_exported_reference_function(
                    prog,
                    name=func_name,
                    reference_source="reference",
                    tensors=tensor_expr,
                    frame_name=reference_name,
                    policy=reference_policy,
                )
            )

        tensor_ctx = generate_tensor_class_source(
            prog,
            class_name=tensor_class,
            function_name=create_function,
            weight_prefix=weight_prefix,
            checkpoint=checkpoint,
            quantization_config=quantization_config,
            shape_exprs=shape_exprs,
        )
        (tensors_dir / f"{tensor_file}.py").write_text(
            render_tensor_module([tensor_ctx]),
            encoding="utf-8",
        )

        func_src, shader_imports, used_variants = generate_dispatch_function_source(
            prog,
            class_name=tensor_class,
            function_name=dispatch_name,
            shader_package=f"{MODEL_PACKAGE}.shaders",
            weight_prefix=weight_prefix,
            quantization_config=quantization_config,
        )
        rename_map: dict[str, str] = {}
        for variant in used_variants.values():
            existing = seen_shader_variants.get(variant.name)
            if existing is None:
                write_generated_shader(variant)
                continue
            if existing.contract == variant.contract:
                continue
            renamed = rename_shader_variant(variant, f"{func_name}_{variant.name}")
            rename_map[variant.name] = renamed.name
            write_generated_shader(renamed)

        for old_name in sorted(rename_map, key=len, reverse=True):
            new_name = rename_map[old_name]
            func_src = re.sub(rf"\b{re.escape(old_name.upper())}\b", new_name.upper(), func_src)
            if old_name in shader_imports:
                shader_imports[new_name] = new_name.upper()
                del shader_imports[old_name]

        (dispatch_dir / f"{func_name}.py").write_text(
            render_model_dispatch_module(
                model_package=MODEL_PACKAGE,
                function_name=dispatch_name,
                tensor_file=tensor_file,
                tensor_class=tensor_class,
                tensor_expr=tensor_expr,
                shader_imports=shader_imports,
                function_source="\n\n\n".join(
                    part
                    for part in (
                        bind_dispatch_function_to_tensors(func_src),
                        extra_dispatch_functions.strip(),
                    )
                    if part
                ),
                parameters_source=parameters_source,
                arguments_source=arguments_source,
                uses_quantized_linear_dispatch="run_quantized_linear(" in func_src,
            ),
            encoding="utf-8",
        )
        print(f"  {dispatch_name}: {len(used_variants)} shaders")

    print("Exporting FLUX.2 Klein 9B Vulkan modules...")
    flux_params = Klein9BParams()
    model_dirs = resolve_model_dirs()
    text_config = AutoConfig.from_pretrained(model_dirs.text_encoder)
    text_layers = int(text_config.num_hidden_layers)
    text_hidden_size = int(text_config.hidden_size)
    text_head_dim = int(text_config.head_dim)
    qwen3_config = qwen3_text_encoder_q8_config()
    flux_config = klein9b_q4_k_m_config()
    ae_config = ae_q4_k_m_config()

    with torch.device("meta"):
        qwen3_model = Qwen3ForCausalLM(text_config).eval().float()
        flux_model = Flux2(flux_params).eval().float()
        ae_decode = AEDecodeModule().eval().float()

    export_one(
        dispatch_name="run_text_embed",
        tensor_file="embed_tokens",
        tensor_class="EmbedTokensTensors",
        create_function="create_embed_tokens",
        tensor_expr="model_tensors().text_embed",
        module=qwen3_model.model.embed_tokens,
        args=(torch.zeros((1, DEFAULT_TEXT_SEQ_LEN), dtype=torch.long, device="meta"),),
        checkpoint="text_encoder/model.gguf",
        weight_prefix="model.embed_tokens.",
        quantization_config=qwen3_config,
        shape_exprs={DEFAULT_TEXT_SEQ_LEN: "sequence_length"},
    )
    export_one(
        dispatch_name="run_text_layer",
        tensor_file="text_layer",
        tensor_class="TextLayerTensors",
        create_function="create_text_layer",
        tensor_expr="model_tensors().text_layers[layer_idx]",
        module=qwen3_model.model.layers[0],
        args=(torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_hidden_size, device="meta"),),
        kwargs={
            "attention_mask": None,
            "position_embeddings": (
                torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_head_dim, device="meta"),
                torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_head_dim, device="meta"),
            ),
            "past_key_values": None,
        },
        checkpoint="text_encoder/model.gguf",
        weight_prefix="model.layers.0.",
        quantization_config=qwen3_config,
        shape_exprs={DEFAULT_TEXT_SEQ_LEN: "sequence_length"},
        parameters_source=", layer_idx: int",
    )
    export_one(
        dispatch_name="run_flux",
        tensor_file="flux",
        tensor_class="FluxTensors",
        create_function="create_flux",
        tensor_expr="model_tensors().flux",
        module=flux_model,
        args=(
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.in_channels, device="meta"),
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, 4, dtype=torch.long, device="meta"),
            torch.zeros(1, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, flux_params.context_in_dim, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, 4, dtype=torch.long, device="meta"),
            torch.zeros(1, device="meta"),
        ),
        checkpoint="flux/model.gguf",
        quantization_config=flux_config,
        shape_exprs={
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
        reference_name="klein9b.flux",
        reference_policy="q4_tensor",
    )
    export_one(
        dispatch_name="run_ae_decode",
        tensor_file="ae_decode",
        tensor_class="AeDecodeTensors",
        create_function="create_ae_decode",
        tensor_expr="_require_ae_decode_tensors()",
        module=ae_decode,
        args=(torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, 128, device="meta"),),
        checkpoint="ae/model.gguf",
        quantization_config=ae_config,
        extra_dispatch_functions='''def _require_ae_decode_tensors() -> AeDecodeTensors:
    tensors = model_tensors().ae_decode
    if tensors is None:
        raise RuntimeError("AE decode tensors were not created")
    return tensors''',
    )

    (tensors_dir / "rope.py").write_text(
        '''"""Generated RoPE tensor declarations."""

from __future__ import annotations

from torch2vk.runtime.rope_table import RopeTableTensors, declare_rope_table_tensors


def create_rope_table(
    prefix: str,
    *,
    batch: int,
    sequence_length: int,
    head_dim: int,
) -> RopeTableTensors:
    return declare_rope_table_tensors(
        prefix,
        batch=batch,
        sequence_length=sequence_length,
        head_dim=head_dim,
    )
''',
        encoding="utf-8",
    )
    _write_model_tensors_module(
        tensors_dir / "model.py",
        image_seq_len=DEFAULT_IMAGE_SEQ_LEN,
        text_seq_len=DEFAULT_TEXT_SEQ_LEN,
        num_text_layers=text_layers,
        text_hidden_size=text_hidden_size,
        text_head_dim=text_head_dim,
    )
    _write_init(tensors_dir / "__init__.py")
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
        "num_text_layers": text_layers,
        "generated_by": "models.quantized_klein9b.export",
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  {shader_file_count} shader files written")
    print(f"  tensors/ written ({count_python_modules(tensors_dir)} files)")
    print(f"  dispatch/ written ({count_python_modules(dispatch_dir)} files)")
    print("  reference.py written")
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
