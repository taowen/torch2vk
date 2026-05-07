"""FX-node-first exporter helpers.

``torch2vk.exportv2`` treats PyTorch FX nodes as the public IR. Helper
functions project those nodes directly into the target/input/output view needed
by lowering and code generation; there is no intermediate torch2vk op object.
"""

from torch2vk.exportv2.dispatch_codegen import (
    render_dispatch_body_from_fx_nodes,
    render_dispatch_body_from_static_nodes,
    shader_symbols_from_fx_nodes,
    shader_symbols_from_static_nodes,
)
from torch2vk.exportv2.contract_codegen import (
    ParamsFieldDecl,
    TensorFieldDecl,
    render_shader_contract_variant_body,
)
from torch2vk.exportv2.frame_codegen import FrameSpec, render_frame_module
from torch2vk.exportv2.frame_dispatch import dispatch_op
from torch2vk.exportv2.fx import (
    FxNodeProjector,
    StaticNode,
    call_function_nodes,
    export_program,
    export_torch_program,
    fx_node_dtype,
    fx_node_input_names,
    fx_node_shape,
    input_names,
    iter_fx_call_function_nodes,
    mapped_node_name,
    project_fx_node,
    project_fx_nodes,
)
from torch2vk.exportv2.logical_tensor_codegen import render_logical_tensor_helpers_file
from torch2vk.exportv2.package_codegen import PythonImportDecl, render_python_init_file
from torch2vk.exportv2.protocols import (
    ExportGraphArgumentLike,
    ExportGraphInputSpecLike,
    ExportGraphOutputSpecLike,
    ExportGraphSignatureLike,
    ExportOpLike,
    ExportedProgramLike,
    FxGraphLike,
    FxNodeLike,
)
from torch2vk.exportv2.reflection import (
    TorchModuleReflection,
    instantiate_torch_module_on_meta,
    reflect_torch_module,
)
from torch2vk.exportv2.rope import precompute_qwen3_asr_mrope
from torch2vk.exportv2.module_scaffold import (
    order_tensor_fields,
    tensor_fields_from_reflected_static_nodes,
)
from torch2vk.exportv2.shader_package import (
    render_shader_package_from_source_dir,
    shader_variants_from_module,
)
from torch2vk.exportv2.tensor_codegen import (
    TensorDataclassDecl,
    TensorDataclassFieldDecl,
    logical_tensor_dataclass_from_patterns,
    render_parameter_fields_constant,
    render_tensor_dataclass,
    render_tensor_dataclasses,
    render_tensor_dtype_constant,
)
from torch2vk.exportv2.tensor_pattern import TensorFieldPattern
from torch2vk.exportv2.tensor_scaffold import (
    tensor_scaffold_fields_from_fx_nodes,
    tensor_scaffold_fields_from_static_nodes,
)
from torch2vk.exportv2.writer import (
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
    "ExportGraphArgumentLike",
    "ExportGraphInputSpecLike",
    "ExportGraphOutputSpecLike",
    "ExportGraphSignatureLike",
    "ExportOpLike",
    "ExportWriteResult",
    "ExportedProgramLike",
    "FxNodeProjector",
    "FxGraphLike",
    "FxNodeLike",
    "FrameSpec",
    "ParamsFieldDecl",
    "PythonImportDecl",
    "RenderedFile",
    "StaticNode",
    "TemplateRenderer",
    "TensorDataclassDecl",
    "TensorDataclassFieldDecl",
    "TensorFieldDecl",
    "TensorFieldPattern",
    "TorchModuleReflection",
    "call_function_nodes",
    "dispatch_op",
    "export_program",
    "export_torch_program",
    "format_python_source",
    "fx_node_dtype",
    "fx_node_input_names",
    "fx_node_shape",
    "input_names",
    "instantiate_torch_module_on_meta",
    "iter_fx_call_function_nodes",
    "logical_tensor_dataclass_from_patterns",
    "mapped_node_name",
    "order_tensor_fields",
    "precompute_qwen3_asr_mrope",
    "project_fx_node",
    "project_fx_nodes",
    "reflect_torch_module",
    "remove_stale_files",
    "render_dispatch_body_from_fx_nodes",
    "render_dispatch_body_from_static_nodes",
    "render_frame_module",
    "render_logical_tensor_helpers_file",
    "render_parameter_fields_constant",
    "render_python_init_file",
    "render_shader_package_from_source_dir",
    "render_shader_contract_variant_body",
    "render_tensor_dataclass",
    "render_tensor_dataclasses",
    "render_tensor_dtype_constant",
    "shader_symbols_from_fx_nodes",
    "shader_symbols_from_static_nodes",
    "shader_variants_from_module",
    "tensor_fields_from_reflected_static_nodes",
    "tensor_scaffold_fields_from_fx_nodes",
    "tensor_scaffold_fields_from_static_nodes",
    "write_rendered_files",
]
