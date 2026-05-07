"""Generic code-generation helpers for model-to-runtime adapter scaffolds."""

from torch2vk.export.exported_program import (
    export_torch_program,
    iter_fx_call_function_nodes,
    torch_ops_from_exported_program,
)
from torch2vk.export.contract_codegen import (
    ParamsFieldDecl,
    TensorFieldDecl,
    render_shader_contract_variant_body,
)
from torch2vk.export.ir import TensorFieldPattern
from torch2vk.export.logical_tensor_codegen import render_logical_tensor_helpers_file
from torch2vk.export.package_codegen import PythonImportDecl, render_python_init_file
from torch2vk.export.lowering import (
    DEFAULT_LOWERING_REGISTRY,
    OpLoweringRegistry,
    OpShaderBinding,
    resolve_shader_symbol_from_target_inputs,
    resolve_shader_symbol,
)
from torch2vk.export.protocols import (
    ExportedProgramLike,
    ExportGraphArgumentLike,
    ExportGraphInputSpecLike,
    ExportGraphOutputSpecLike,
    ExportGraphSignatureLike,
    ExportOpLike,
    FxGraphLike,
    FxNodeLike,
    FxTensorMetaLike,
)
from torch2vk.export.reflection import (
    TorchModuleReflection,
    instantiate_torch_module_on_meta,
    reflect_torch_module,
)
from torch2vk.export.frame_dispatch_codegen import (
    FrameSpec,
    render_dispatch_body,
    render_frame_module,
)
from torch2vk.export.rope import precompute_qwen3_asr_mrope
from torch2vk.export.tensor_scaffold_codegen import (
    LoweredOpContract,
    TensorDataclassDecl,
    TensorDataclassFieldDecl,
    logical_tensor_dataclass_from_patterns,
    render_parameter_fields_constant,
    render_tensor_dtype_constant,
    render_tensor_dataclass,
    render_tensor_dataclasses,
    tensor_scaffold_fields_from_lowered_ops,
)
from torch2vk.export.writer import (
    ExportCheckError,
    ExportWriteResult,
    RenderedFile,
    TemplateRenderer,
    format_python_source,
    remove_stale_files,
    write_rendered_files,
)

__all__ = [
    "ExportCheckError",
    "ExportWriteResult",
    "DEFAULT_LOWERING_REGISTRY",
    "FrameSpec",
    "render_dispatch_body",
    "render_frame_module",
    "OpLoweringRegistry",
    "ParamsFieldDecl",
    "OpShaderBinding",
    "RenderedFile",
    "TemplateRenderer",
    "TensorFieldPattern",
    "TensorFieldDecl",
    "LoweredOpContract",
    "TensorDataclassDecl",
    "TensorDataclassFieldDecl",
    "TorchModuleReflection",
    "render_logical_tensor_helpers_file",
    "PythonImportDecl",
    "render_python_init_file",
    "export_torch_program",
    "format_python_source",
    "instantiate_torch_module_on_meta",
    "remove_stale_files",
    "reflect_torch_module",
    "precompute_qwen3_asr_mrope",
    "logical_tensor_dataclass_from_patterns",
    "render_parameter_fields_constant",
    "render_tensor_dtype_constant",
    "render_shader_contract_variant_body",
    "render_tensor_dataclass",
    "render_tensor_dataclasses",
    "resolve_shader_symbol",
    "resolve_shader_symbol_from_target_inputs",
    "tensor_scaffold_fields_from_lowered_ops",
    "ExportedProgramLike",
    "ExportGraphArgumentLike",
    "ExportGraphInputSpecLike",
    "ExportGraphOutputSpecLike",
    "ExportGraphSignatureLike",
    "ExportOpLike",
    "FxGraphLike",
    "FxNodeLike",
    "FxTensorMetaLike",
    "iter_fx_call_function_nodes",
    "torch_ops_from_exported_program",
    "write_rendered_files",
]
