"""Direct FX-graph-to-Vulkan export with 1:1 aten op mapping."""

from torch2vk.export.codegen_loop import (
    generate_looped_dispatch_function_source,
    generate_looped_tensor_class_sources,
)
from torch2vk.export.dispatch_codegen import (
    generate_dispatch_function_source,
    generate_dispatch_source,
)
from torch2vk.export.graph import (
    KVCacheExportHint,
    KVCacheInjectHint,
    LayerLoopHint,
    export_submodule,
    inject_kv_cache,
)
from torch2vk.export.reference_codegen import (
    ReferencePolicy,
    render_exported_reference_function,
    render_reference_function,
    render_reference_loader,
    render_reference_module,
)
from torch2vk.export.tensor_codegen import (
    TensorClassContext,
    generate_tensor_class_source,
)

__all__ = [
    "KVCacheExportHint",
    "KVCacheInjectHint",
    "LayerLoopHint",
    "export_submodule",
    "inject_kv_cache",
    "ReferencePolicy",
    "TensorClassContext",
    "generate_dispatch_function_source",
    "generate_dispatch_source",
    "generate_looped_dispatch_function_source",
    "generate_looped_tensor_class_sources",
    "generate_tensor_class_source",
    "render_exported_reference_function",
    "render_reference_function",
    "render_reference_loader",
    "render_reference_module",
]
