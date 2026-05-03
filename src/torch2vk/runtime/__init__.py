"""RuntimeSession orchestration package."""

# Layer note: runtime is the only package that should connect LogicalTensor
# metadata to Vulkan BufferSlice materialization and dispatch records.
