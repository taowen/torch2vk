"""Generic code-generation helpers for model-to-runtime adapter scaffolds."""

from torch2vk.export.exported_program import export_torch_program, torch_ops_from_exported_program
from torch2vk.export.contract_codegen import (
    ParamsFieldDecl,
    TensorFieldDecl,
    render_shader_contract_variant_body,
)
from torch2vk.export.ir import TorchOpPattern, TensorFieldPattern
from torch2vk.export.lowering import (
    DEFAULT_LOWERING_REGISTRY,
    OpLoweringRegistry,
    OpShaderBinding,
    resolve_shader_symbol,
)
from torch2vk.export.reflection import (
    TorchModuleReflection,
    instantiate_torch_module_on_meta,
    reflect_torch_module,
)
from torch2vk.export.rope import precompute_qwen3_asr_mrope
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
    "OpLoweringRegistry",
    "ParamsFieldDecl",
    "OpShaderBinding",
    "TorchOpPattern",
    "RenderedFile",
    "TemplateRenderer",
    "TensorFieldPattern",
    "TensorFieldDecl",
    "TorchModuleReflection",
    "export_torch_program",
    "format_python_source",
    "instantiate_torch_module_on_meta",
    "remove_stale_files",
    "reflect_torch_module",
    "precompute_qwen3_asr_mrope",
    "render_shader_contract_variant_body",
    "resolve_shader_symbol",
    "torch_ops_from_exported_program",
    "write_rendered_files",
]
