"""Export quantized FLUX.2 Klein 9B denoiser Vulkan modules.

Run from project root:
    uv run python -m models.quantized_klein9b.export
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from models.quantized_klein9b.model_sources import resolve_model_dirs
from models.quantized_klein9b.pytorch_modules import (
    AEDecodeModule,
    AEEntryModule,
    DoubleStreamBlock,
    EulerUpdateModule,
    FLUX_DOUBLE_BLOCK_OUTPUT_NAMES,
    FLUX_FINAL_LAYER_OUTPUT_NAMES,
    FLUX_PROLOGUE_OUTPUT_NAMES,
    FLUX_SINGLE_BLOCK_OUTPUT_NAMES,
    Flux2,
    FluxDoubleBlockModule,
    FluxFinalLayerModule,
    FluxJoinModule,
    FluxPeJoinModule,
    FluxPrologueModule,
    FluxSingleBlockModule,
    Klein9BParams,
    SingleStreamBlock,
    TextContextCaptureModule,
)
from models.quantized_klein9b.quantization import (
    ae_q4_k_m_config,
    klein9b_q4_k_m_config,
    qwen3_text_encoder_q8_config,
)
from torch2vk.export import (
    TensorSpecOverride,
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
    render_model_dispatch_module,
    write_shader_file,
)
from torch2vk.export.reference_codegen import (
    render_reference_module,
    render_streaming_compare_function,
)
from torch2vk.export.registry import DEFAULT_REGISTRY
from torch2vk.export.graph import graph_output_names
from torch2vk.export.tensor_codegen import render_tensor_module
from torch2vk.runtime.shader import ShaderVariant


MODEL_PACKAGE = "models.quantized_klein9b"
DEFAULT_IMAGE_SEQ_LEN = 1024
DEFAULT_TEXT_SEQ_LEN = 512
DEFAULT_LATENT_HEIGHT = 32
DEFAULT_LATENT_WIDTH = 32
_TEMPLATE_DIR = Path(__file__).with_name("templates")
_JINJA = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    loader=FileSystemLoader(_TEMPLATE_DIR),
    lstrip_blocks=True,
    trim_blocks=True,
    undefined=StrictUndefined,
)


def _render_template(template_name: str, **context: object) -> str:
    return _JINJA.get_template(template_name).render(**context)


def _write_model_tensors_module(
    path: Path,
    *,
    text_seq_len: int,
    num_text_layers: int,
    text_hidden_size: int,
    text_head_dim: int,
    flux_prologue_outputs: dict[str, str],
    flux_double_block_outputs: dict[str, str],
    flux_single_block_outputs: dict[str, str],
    flux_final_layer_outputs: dict[str, str],
) -> None:
    path.write_text(
        _render_template(
            "model.py.j2",
            model_package=MODEL_PACKAGE,
            default_latent_height=DEFAULT_LATENT_HEIGHT,
            default_latent_width=DEFAULT_LATENT_WIDTH,
            text_seq_len=text_seq_len,
            num_text_layers=num_text_layers,
            text_hidden_size=text_hidden_size,
            text_head_dim=text_head_dim,
            flux_prologue_outputs_repr=repr(flux_prologue_outputs),
            flux_double_block_outputs_repr=repr(flux_double_block_outputs),
            flux_single_block_outputs_repr=repr(flux_single_block_outputs),
            flux_final_layer_outputs_repr=repr(flux_final_layer_outputs),
        ),
        encoding="utf-8",
    )


def _write_init(path: Path) -> None:
    path.write_text('"""Generated package."""\n', encoding="utf-8")


def _shape_symbol_exprs(
    prog: torch.export.ExportedProgram,
    axes_by_placeholder: dict[str, dict[int, str]],
) -> dict[str, str]:
    symbol_exprs: dict[str, str] = {}
    for node in prog.graph_module.graph.nodes:
        axes = axes_by_placeholder.get(node.name)
        if axes is None:
            continue
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is None:
            raise ValueError(f"Placeholder {node.name!r} is missing tensor_meta")
        shape = tensor_meta.shape
        for axis, expression in axes.items():
            dim_source = str(shape[axis])
            if dim_source.lstrip("-").isdigit():
                raise ValueError(
                    f"Placeholder {node.name!r} axis {axis} is static; "
                    "expected a dynamic torch.export dimension"
                )
            symbol_exprs[dim_source] = expression
    return symbol_exprs


def _torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.float32:
        return "float32"
    raise ValueError(f"Unsupported activation dtype for Klein9B export: {dtype}")


def _output_map(stage: str, names: tuple[str, ...], semantic_names: tuple[str, ...]) -> dict[str, str]:
    if len(names) != len(semantic_names):
        raise RuntimeError(
            f"{stage} exported {len(names)} outputs, expected {len(semantic_names)}: {names}"
        )
    return dict(zip(semantic_names, names, strict=True))


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

    def prepare_dispatch_source(
        *,
        func_name: str,
        func_src: str,
        shader_imports: dict[str, str],
        used_variants: dict[str, ShaderVariant],
    ) -> tuple[str, dict[str, str]]:
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
        return func_src, shader_imports

    reference_functions: list[str] = []
    reference_dispatch_imports: list[str] = []

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
        shape_symbol_axes: dict[str, dict[int, str]] | None = None,
        parameters_source: str = "",
        arguments_source: str = "",
        extra_dispatch_functions: str = "",
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
        strict: bool = False,
        is_layered: bool | None = None,
        export_weight_dtype: torch.dtype = torch.float32,
        export_activation_dtype: torch.dtype | None = None,
        export_input_dtype: torch.dtype | None = None,
        tensor_spec_overrides: dict[str, TensorSpecOverride] | None = None,
        semantic_output_names: tuple[str, ...] = (),
        runtime_output_semantics: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        module = module.to(dtype=export_weight_dtype)
        export_dtype = module_floating_dtype(module)
        input_dtype = (
            export_input_dtype
            if export_input_dtype is not None
            else export_activation_dtype
            if export_activation_dtype is not None
            else export_dtype
        )
        registry = (
            DEFAULT_REGISTRY.with_activation_dtype(_torch_dtype_name(export_activation_dtype))
            if export_activation_dtype is not None
            else DEFAULT_REGISTRY
        )
        if input_dtype is not None:
            args = cast_floating_tensors(args, input_dtype)
            kwargs = cast_floating_tensors(kwargs, input_dtype)
        prog = export_submodule(
            module,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )
        output_names = tuple(graph_output_names(prog.graph_module.graph))
        escaped_output_names = output_names
        if runtime_output_semantics:
            output_by_semantic = _output_map(
                dispatch_name,
                output_names,
                semantic_output_names,
            )
            escaped_output_names = tuple(output_by_semantic[name] for name in runtime_output_semantics)
        shape_symbol_exprs = _shape_symbol_exprs(prog, shape_symbol_axes or {})
        func_name = dispatch_name.removeprefix("run_")
        tensor_ctx = generate_tensor_class_source(
            prog,
            class_name=tensor_class,
            function_name=create_function,
            weight_prefix=weight_prefix,
            checkpoint=checkpoint,
            is_layered=is_layered,
            registry=registry,
            quantization_config=quantization_config,
            shape_exprs=shape_exprs,
            shape_symbol_exprs=shape_symbol_exprs,
            tensor_spec_overrides=tensor_spec_overrides,
        )
        tensor_source = render_tensor_module([tensor_ctx])
        (tensors_dir / f"{tensor_file}.py").write_text(
            tensor_source,
            encoding="utf-8",
        )

        func_src, shader_imports, used_variants = generate_dispatch_function_source(
            prog,
            class_name=tensor_class,
            function_name=dispatch_name,
            shader_package=f"{MODEL_PACKAGE}.shaders",
            registry=registry,
            weight_prefix=weight_prefix,
            quantization_config=quantization_config,
            escaped_output_names=escaped_output_names,
        )
        func_src, shader_imports = prepare_dispatch_source(
            func_name=func_name,
            func_src=func_src,
            shader_imports=shader_imports,
            used_variants=used_variants,
        )

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
        return output_names

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
        dispatch_name="run_text_context_capture",
        tensor_file="text_context_capture",
        tensor_class="TextContextCaptureTensors",
        create_function="create_text_context_capture",
        tensor_expr="model_tensors().text_context_capture",
        module=TextContextCaptureModule(),
        args=(
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_hidden_size, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_hidden_size, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, text_hidden_size, device="meta"),
        ),
        checkpoint="text_encoder/model.gguf",
        export_activation_dtype=torch.float32,
        export_input_dtype=torch.float16,
        shape_exprs={DEFAULT_TEXT_SEQ_LEN: "sequence_length"},
        tensor_spec_overrides={
            "layer_9": TensorSpecOverride(
                dtype="float16",
                shape=(1, "sequence_length", text_hidden_size),
            ),
            "layer_18": TensorSpecOverride(
                dtype="float16",
                shape=(1, "sequence_length", text_hidden_size),
            ),
            "layer_27": TensorSpecOverride(
                dtype="float16",
                shape=(1, "sequence_length", text_hidden_size),
            ),
        },
    )
    flux_prologue_output_names = export_one(
        dispatch_name="run_flux_prologue",
        tensor_file="flux_prologue",
        tensor_class="FluxPrologueTensors",
        create_function="create_flux_prologue",
        tensor_expr="model_tensors().flux_prologue",
        module=FluxPrologueModule(flux_model),
        args=(
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.in_channels, device="meta"),
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, 4, dtype=torch.long, device="meta"),
            torch.zeros(1, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, flux_params.context_in_dim, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, 4, dtype=torch.long, device="meta"),
        ),
        checkpoint="flux/model.gguf",
        quantization_config=flux_config,
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
    )
    flux_double_block_output_names = export_one(
        dispatch_name="run_flux_double_block",
        tensor_file="flux_double_block",
        tensor_class="FluxDoubleBlockTensors",
        create_function="create_flux_double_block",
        tensor_expr="model_tensors().flux_double_blocks[layer_idx]",
        module=FluxDoubleBlockModule(cast(DoubleStreamBlock, flux_model.double_blocks[0])),
        args=(
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.hidden_size, device="meta"),
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, flux_params.hidden_size, device="meta"),
            torch.zeros(
                1,
                1,
                DEFAULT_IMAGE_SEQ_LEN,
                flux_params.hidden_size // flux_params.num_heads // 2,
                2,
                2,
                device="meta",
            ),
            torch.zeros(
                1,
                1,
                DEFAULT_TEXT_SEQ_LEN,
                flux_params.hidden_size // flux_params.num_heads // 2,
                2,
                2,
                device="meta",
            ),
            *(
                torch.zeros(1, 1, flux_params.hidden_size, device="meta")
                for _ in range(12)
            ),
        ),
        checkpoint="flux/model.gguf",
        weight_prefix="double_blocks.0.",
        quantization_config=flux_config,
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
        parameters_source=", layer_idx: int",
        is_layered=True,
        semantic_output_names=FLUX_DOUBLE_BLOCK_OUTPUT_NAMES,
        runtime_output_semantics=("img", "txt"),
    )
    export_one(
        dispatch_name="run_flux_join",
        tensor_file="flux_join",
        tensor_class="FluxJoinTensors",
        create_function="create_flux_join",
        tensor_expr="model_tensors().flux_join",
        module=FluxJoinModule(),
        args=(
            torch.zeros(1, DEFAULT_TEXT_SEQ_LEN, flux_params.hidden_size, device="meta"),
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.hidden_size, device="meta"),
        ),
        checkpoint="flux/model.gguf",
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
    )
    export_one(
        dispatch_name="run_flux_pe_join",
        tensor_file="flux_pe_join",
        tensor_class="FluxPeJoinTensors",
        create_function="create_flux_pe_join",
        tensor_expr="model_tensors().flux_pe_join",
        module=FluxPeJoinModule(),
        args=(
            torch.zeros(
                1,
                1,
                DEFAULT_TEXT_SEQ_LEN,
                flux_params.hidden_size // flux_params.num_heads // 2,
                2,
                2,
                device="meta",
            ),
            torch.zeros(
                1,
                1,
                DEFAULT_IMAGE_SEQ_LEN,
                flux_params.hidden_size // flux_params.num_heads // 2,
                2,
                2,
                device="meta",
            ),
        ),
        checkpoint="flux/model.gguf",
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
    )
    flux_single_block_output_names = export_one(
        dispatch_name="run_flux_single_block",
        tensor_file="flux_single_block",
        tensor_class="FluxSingleBlockTensors",
        create_function="create_flux_single_block",
        tensor_expr="model_tensors().flux_single_blocks[layer_idx]",
        module=FluxSingleBlockModule(
            cast(SingleStreamBlock, flux_model.single_blocks[0]),
            text_seq_len=DEFAULT_TEXT_SEQ_LEN,
        ),
        args=(
            torch.zeros(
                1,
                DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN,
                flux_params.hidden_size,
                device="meta",
            ),
            torch.zeros(
                1,
                1,
                DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN,
                flux_params.hidden_size // flux_params.num_heads // 2,
                2,
                2,
                device="meta",
            ),
            *(
                torch.zeros(1, 1, flux_params.hidden_size, device="meta")
                for _ in range(3)
            ),
        ),
        checkpoint="flux/model.gguf",
        weight_prefix="single_blocks.0.",
        quantization_config=flux_config,
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
        parameters_source=", layer_idx: int",
        is_layered=True,
        semantic_output_names=FLUX_SINGLE_BLOCK_OUTPUT_NAMES,
        runtime_output_semantics=("hidden_states",),
    )
    flux_final_layer_output_names = export_one(
        dispatch_name="run_flux_final_layer",
        tensor_file="flux_final_layer",
        tensor_class="FluxFinalLayerTensors",
        create_function="create_flux_final_layer",
        tensor_expr="model_tensors().flux_final_layer",
        module=FluxFinalLayerModule(flux_model, text_seq_len=DEFAULT_TEXT_SEQ_LEN),
        args=(
            torch.zeros(
                1,
                DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN,
                flux_params.hidden_size,
                device="meta",
            ),
            torch.zeros(1, flux_params.hidden_size, device="meta"),
        ),
        checkpoint="flux/model.gguf",
        quantization_config=flux_config,
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_TEXT_SEQ_LEN + DEFAULT_IMAGE_SEQ_LEN: "text_seq_len + image_seq_len",
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
    )
    export_one(
        dispatch_name="run_euler_update",
        tensor_file="euler_update",
        tensor_class="EulerUpdateTensors",
        create_function="create_euler_update",
        tensor_expr="model_tensors().euler_update",
        module=EulerUpdateModule(),
        args=(
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.in_channels, device="meta"),
            torch.zeros(1, DEFAULT_IMAGE_SEQ_LEN, flux_params.in_channels, device="meta"),
            torch.zeros(1, device="meta"),
        ),
        checkpoint="flux/model.gguf",
        export_activation_dtype=torch.float32,
        shape_exprs={DEFAULT_IMAGE_SEQ_LEN: "image_seq_len"},
    )
    flux_prologue_outputs = _output_map(
        "flux_prologue",
        flux_prologue_output_names,
        FLUX_PROLOGUE_OUTPUT_NAMES,
    )
    flux_double_block_outputs = _output_map(
        "flux_double_block",
        flux_double_block_output_names,
        FLUX_DOUBLE_BLOCK_OUTPUT_NAMES,
    )
    flux_single_block_outputs = _output_map(
        "flux_single_block",
        flux_single_block_output_names,
        FLUX_SINGLE_BLOCK_OUTPUT_NAMES,
    )
    flux_final_layer_outputs = _output_map(
        "flux_final_layer",
        flux_final_layer_output_names,
        FLUX_FINAL_LAYER_OUTPUT_NAMES,
    )
    reference_dispatch_imports.extend(
        (
            "from models.quantized_klein9b.dispatch.flux_prologue import run_flux_prologue as _dispatch_flux_prologue",
            "from models.quantized_klein9b.dispatch.flux_double_block import run_flux_double_block as _dispatch_flux_double_block",
            "from models.quantized_klein9b.dispatch.flux_single_block import run_flux_single_block as _dispatch_flux_single_block",
            "from models.quantized_klein9b.dispatch.flux_final_layer import run_flux_final_layer as _dispatch_flux_final_layer",
        )
    )
    reference_functions.extend(
        (
            render_streaming_compare_function(
                name="flux_prologue",
                dispatch_source="_dispatch_flux_prologue",
                tensors="model_tensors().flux_prologue",
                frame_name="klein9b.flux.compare.{step:04d}.prologue",
                policy="q8_tensor",
                input_bindings={
                    "x": "x",
                    "x_ids": "x_ids",
                    "timesteps": "timesteps",
                    "ctx": "ctx",
                    "ctx_ids": "ctx_ids",
                },
                output_bindings=flux_prologue_outputs,
            ),
            render_streaming_compare_function(
                name="flux_double_block",
                dispatch_source="_dispatch_flux_double_block",
                tensors="model_tensors().flux_double_blocks[layer_idx]",
                frame_name="klein9b.flux.compare.{step:04d}.double_block.{layer_idx}",
                policy="q8_tensor",
                input_bindings={
                    "img": "img",
                    "txt": "txt",
                    "pe": "pe",
                    "pe_ctx": "pe_ctx",
                    "img_mod1_shift": "img_mod1_shift",
                    "img_mod1_scale": "img_mod1_scale",
                    "img_mod1_gate": "img_mod1_gate",
                    "img_mod2_shift": "img_mod2_shift",
                    "img_mod2_scale": "img_mod2_scale",
                    "img_mod2_gate": "img_mod2_gate",
                    "txt_mod1_shift": "txt_mod1_shift",
                    "txt_mod1_scale": "txt_mod1_scale",
                    "txt_mod1_gate": "txt_mod1_gate",
                    "txt_mod2_shift": "txt_mod2_shift",
                    "txt_mod2_scale": "txt_mod2_scale",
                    "txt_mod2_gate": "txt_mod2_gate",
                },
                output_bindings={
                    "img": flux_double_block_outputs["img"],
                    "txt": flux_double_block_outputs["txt"],
                },
                dispatch_args=("layer_idx",),
            ),
            render_streaming_compare_function(
                name="flux_single_block",
                dispatch_source="_dispatch_flux_single_block",
                tensors="model_tensors().flux_single_blocks[layer_idx]",
                frame_name="klein9b.flux.compare.{step:04d}.single_block.{layer_idx}",
                policy="q8_tensor",
                input_bindings={
                    "hidden_states": "hidden_states",
                    "pe": "pe",
                    "mod_shift": "mod_shift",
                    "mod_scale": "mod_scale",
                    "mod_gate": "mod_gate",
                },
                output_bindings={
                    "hidden_states": flux_single_block_outputs["hidden_states"],
                },
                dispatch_args=("layer_idx",),
            ),
            render_streaming_compare_function(
                name="flux_final_layer",
                dispatch_source="_dispatch_flux_final_layer",
                tensors="model_tensors().flux_final_layer",
                frame_name="klein9b.flux.compare.{step:04d}.final_layer",
                policy="q8_tensor",
                input_bindings={
                    "hidden_states": "hidden_states",
                    "vec": "vec",
                },
                output_bindings=flux_final_layer_outputs,
            ),
        )
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
        export_activation_dtype=torch.float32,
        shape_exprs={
            DEFAULT_IMAGE_SEQ_LEN: "image_seq_len",
            DEFAULT_TEXT_SEQ_LEN: "text_seq_len",
        },
    )
    export_one(
        dispatch_name="run_ae_entry",
        tensor_file="ae_entry",
        tensor_class="AEEntryTensors",
        create_function="create_ae_entry",
        tensor_expr="_require_ae_entry_tensors()",
        module=AEEntryModule(),
        args=(torch.zeros(1, DEFAULT_LATENT_HEIGHT, DEFAULT_LATENT_WIDTH, 128, device="meta"),),
        checkpoint="ae/model.gguf",
        export_activation_dtype=torch.float16,
        export_input_dtype=torch.float32,
        shape_symbol_axes={"tokens": {1: "latent_height", 2: "latent_width"}},
        tensor_spec_overrides={
            "tokens": TensorSpecOverride(
                dtype="float32",
                shape=(1, "latent_height", "latent_width", 128),
            ),
        },
        dynamic_shapes={
            "tokens": {
                1: torch.export.Dim("latent_height", min=4, max=256),
                2: torch.export.Dim("latent_width", min=4, max=256),
            }
        },
        strict=True,
        extra_dispatch_functions='''def _require_ae_entry_tensors() -> AEEntryTensors:
    tensors = model_tensors().ae_entry
    if tensors is None:
        raise RuntimeError("AE entry tensors were not created")
    return tensors''',
    )
    export_one(
        dispatch_name="run_ae_decode",
        tensor_file="ae_decode",
        tensor_class="AeDecodeTensors",
        create_function="create_ae_decode",
        tensor_expr="_require_ae_decode_tensors()",
        module=ae_decode,
        args=(torch.zeros(1, 128, DEFAULT_LATENT_HEIGHT, DEFAULT_LATENT_WIDTH, device="meta"),),
        checkpoint="ae/model.gguf",
        quantization_config=ae_config,
        export_activation_dtype=torch.float16,
        shape_symbol_axes={"tokens": {2: "latent_height", 3: "latent_width"}},
        dynamic_shapes={
            "tokens": {
                2: torch.export.Dim("latent_height", min=4, max=256),
                3: torch.export.Dim("latent_width", min=4, max=256),
            }
        },
        strict=True,
        extra_dispatch_functions='''def _require_ae_decode_tensors() -> AeDecodeTensors:
    tensors = model_tensors().ae_decode
    if tensors is None:
        raise RuntimeError("AE decode tensors were not created")
    return tensors''',
    )

    (tensors_dir / "rope.py").write_text(_render_template("rope.py.j2"), encoding="utf-8")
    _write_model_tensors_module(
        tensors_dir / "model.py",
        text_seq_len=DEFAULT_TEXT_SEQ_LEN,
        num_text_layers=text_layers,
        text_hidden_size=text_hidden_size,
        text_head_dim=text_head_dim,
        flux_prologue_outputs=flux_prologue_outputs,
        flux_double_block_outputs=flux_double_block_outputs,
        flux_single_block_outputs=flux_single_block_outputs,
        flux_final_layer_outputs=flux_final_layer_outputs,
    )
    _write_init(tensors_dir / "__init__.py")
    _write_init(dispatch_dir / "__init__.py")

    (output_dir / "reference.py").write_text(
        render_reference_module(
            model_package=MODEL_PACKAGE,
            reference_functions=reference_functions,
            dispatch_imports=reference_dispatch_imports,
        ),
        encoding="utf-8",
    )

    print(f"  {shader_file_count} shader files written")
    print(f"  tensors/ written ({count_python_modules(tensors_dir)} files)")
    print(f"  dispatch/ written ({count_python_modules(dispatch_dir)} files)")
    print("  reference.py written")
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
