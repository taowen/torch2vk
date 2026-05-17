"""Direct FX-graph-to-Vulkan export with 1:1 aten op mapping."""

from torch2vk.export.codegen_loop import (
    generate_looped_dispatch_function_source,
    generate_looped_tensor_class_sources,
)
from torch2vk.export.dispatch_codegen import (
    bind_dispatch_function_to_tensors,
    generate_dispatch_function_source,
    generate_dispatch_source,
    render_model_dispatch_module,
)
from torch2vk.export.graph import (
    KVCacheExportHint,
    KVCacheInjectHint,
    LayerLoopHint,
    export_submodule,
    inject_kv_cache,
)
from torch2vk.export.checkpoint_dtypes import (
    cast_floating_tensors,
    module_floating_dtype,
    read_checkpoint_dtypes,
    set_module_checkpoint_dtypes,
)
from torch2vk.export.reference_codegen import (
    ReferencePolicy,
    render_reference_module,
    render_streaming_compare_function,
)
from torch2vk.export.tensor_codegen import (
    TensorClassContext,
    generate_tensor_class_source,
    generate_weight_tensor_class_source,
)
from torch2vk.export.shader_codegen import (
    clear_python_modules,
    clear_shader_package,
    count_python_modules,
    rename_shader_variant,
    write_shader_file,
)

__all__ = [
    "KVCacheExportHint",
    "KVCacheInjectHint",
    "LayerLoopHint",
    "export_submodule",
    "inject_kv_cache",
    "read_checkpoint_dtypes",
    "ReferencePolicy",
    "TensorClassContext",
    "bind_dispatch_function_to_tensors",
    "cast_floating_tensors",
    "clear_python_modules",
    "clear_shader_package",
    "count_python_modules",
    "generate_dispatch_function_source",
    "generate_dispatch_source",
    "generate_looped_dispatch_function_source",
    "generate_looped_tensor_class_sources",
    "generate_tensor_class_source",
    "generate_weight_tensor_class_source",
    "module_floating_dtype",
    "rename_shader_variant",
    "render_reference_module",
    "render_streaming_compare_function",
    "render_model_dispatch_module",
    "set_module_checkpoint_dtypes",
    "write_shader_file",
]
