"""Python-first helpers for porting PyTorch models to Vulkan shaders."""

from .logical import (
    BufferSlice,
    ComparePolicy,
    LogicalTensor,
    MemoryPolicy,
    TensorLayout,
    TensorRole,
    TensorSpec,
    WeightSource,
)
from .shader import (
    Binding,
    BindingAccess,
    DispatchRecord,
    DispatchTarget,
    ShaderContract,
    ShaderVariant,
    TensorContract,
)

__all__ = [
    "Binding",
    "BindingAccess",
    "BufferSlice",
    "ComparePolicy",
    "DispatchRecord",
    "DispatchTarget",
    "LogicalTensor",
    "MemoryPolicy",
    "ShaderContract",
    "ShaderVariant",
    "TensorContract",
    "TensorLayout",
    "TensorRole",
    "TensorSpec",
    "WeightSource",
]

