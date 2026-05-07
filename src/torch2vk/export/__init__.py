"""Direct FX-graph-to-Vulkan export with 1:1 aten op mapping."""

from torch2vk.export.codegen import generate_dispatch_source
from torch2vk.export.graph import export_submodule

__all__ = [
    "export_submodule",
    "generate_dispatch_source",
]
