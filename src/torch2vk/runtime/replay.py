"""Replay plan: pre-recorded command buffer execution for decode/prefill stages."""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vulkan import (
    VK_QUERY_RESULT_64_BIT,
    VK_QUERY_RESULT_WAIT_BIT,
    VK_QUERY_TYPE_TIMESTAMP,
    VkQueryPoolCreateInfo,
    VkSubmitInfo,
    vkCreateQueryPool,
    vkDestroyFence,
    vkDestroyQueryPool,
    vkGetQueryPoolResults,
    vkQueueSubmit,
    vkResetFences,
    vkWaitForFences,
)
from vulkan._vulkan import ffi

from torch2vk.runtime.shader import (
    ExprDim,
    ParamsBufferSpec,
    PushConstantInput,
    PushConstantType,
    TensorFieldSpec,
    eval_expr,
)
from torch2vk.vulkan.allocation import BufferAllocation
from torch2vk.vulkan.compute_pipeline import (
    BoundComputeBinding,
    ComputePipeline,
    DescriptorBufferBinding,
)

if TYPE_CHECKING:
    from torch2vk.vulkan.device import VulkanDevice


_WAIT_TIMEOUT_NS = 30_000_000_000


@dataclass(slots=True)
class ReplayReadbackSlot:
    """Host-visible buffer for reading GPU output after fence wait."""
    name: str
    allocation: BufferAllocation
    nbytes: int


@dataclass(slots=True)
class ReplayDispatchEntry:
    """Pre-resolved dispatch metadata for one shader in the replay sequence."""
    pipeline: ComputePipeline
    binding: BoundComputeBinding
    descriptors: tuple["ReplayDescriptorBinding", ...]
    push_constants: bytes | None
    dispatch_size: tuple[int, int, int]
    dispatch_formula: tuple[ExprDim, ExprDim, ExprDim]
    symbols: dict[str, int]
    indirect_offset: int | None = None
    params_buffer: BufferAllocation | None = None
    params_layout: ParamsBufferSpec | None = None
    source_dispatch_index: int | None = None
    source_frame: str | None = None
    source_shader: str | None = None
    source_logical_reads: tuple[tuple[str, str], ...] = ()
    source_logical_writes: tuple[tuple[str, str], ...] = ()


@dataclass(slots=True)
class ReplayProfileState:
    """Timestamp query resources recorded into a profiled replay command buffer."""

    query_pool: object | None
    query_count: int

    @classmethod
    def create(cls, device: "VulkanDevice", *, num_dispatches: int) -> "ReplayProfileState":
        query_count = 2 + num_dispatches * 2
        query_pool = vkCreateQueryPool(
            device.device,
            VkQueryPoolCreateInfo(
                queryType=VK_QUERY_TYPE_TIMESTAMP,
                queryCount=query_count,
            ),
            None,
        )
        return cls(query_pool=query_pool, query_count=query_count)

    def close(self, device: "VulkanDevice") -> None:
        if self.query_pool is None:
            return
        if not device.closed:
            vkDestroyQueryPool(device.device, self.query_pool, None)
        self.query_pool = None

    def read_timestamps(self, device: "VulkanDevice") -> tuple[int, ...]:
        data = ffi.new("uint64_t[]", self.query_count)
        if self.query_pool is None:
            raise RuntimeError("Replay profile query pool is closed")
        vkGetQueryPoolResults(
            device.device,
            self.query_pool,
            0,
            self.query_count,
            self.query_count * 8,
            data,
            8,
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT,
        )
        return tuple(int(data[i]) for i in range(self.query_count))


@dataclass(frozen=True, slots=True)
class ReplayDescriptorBinding:
    """One shader field descriptor recorded in a replay command buffer."""

    field: TensorFieldSpec
    tensor_name: str
    buffer: DescriptorBufferBinding
    rebindable: bool
    validate_shape: bool


@dataclass(slots=True)
class ReplayPlan:
    """Cached pre-recorded command buffer that can execute a stage for any compatible shape."""

    device: VulkanDevice
    name: str

    command_buffer: object
    fence: object

    indirect_buffer: BufferAllocation | None
    num_dispatches: int

    dispatch_entries: tuple[ReplayDispatchEntry, ...]
    params_entries: tuple[ReplayDispatchEntry, ...]
    dynamic_symbol_names: tuple[str, ...]

    readback_slots: dict[str, ReplayReadbackSlot]

    workspace_allocations: list[BufferAllocation]
    bindings: list[BoundComputeBinding]
    profile_state: ReplayProfileState | None = None
    profile_recorder: object | None = None

    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for binding in self.bindings:
            binding.close()
        if self.fence is not None:
            vkDestroyFence(self.device.device, self.fence, None)
        if self.command_buffer is not None:
            self.device.free_command_buffer(self.command_buffer)
        if self.profile_state is not None:
            self.profile_state.close(self.device)
        if self.indirect_buffer is not None:
            self.indirect_buffer.close()
        for slot in self.readback_slots.values():
            slot.allocation.close()
        for alloc in self.workspace_allocations:
            alloc.close()


def execute_replay(
    plan: ReplayPlan,
    dynamic_symbols: Mapping[str, int] | None = None,
) -> dict[str, bytes]:
    """Execute the replay plan with given dynamic symbol values.

    Static plans submit the command buffer directly. Dynamic plans update only the
    pre-bound host-visible parameter/indirect buffers before submit.
    """
    if plan._closed:
        raise RuntimeError("ReplayPlan is closed")

    symbols = {} if dynamic_symbols is None else dynamic_symbols
    unexpected_symbols = set(symbols) - set(plan.dynamic_symbol_names)
    if unexpected_symbols:
        raise ValueError(
            f"ReplayPlan {plan.name!r} got unexpected dynamic symbols: {sorted(unexpected_symbols)}"
        )

    if plan.indirect_buffer is not None:
        _write_indirect_dispatch_buffer(plan, symbols)
    if plan.params_entries:
        _write_params_buffers(plan, symbols)

    vkResetFences(plan.device.device, 1, [plan.fence])
    submit_info = VkSubmitInfo(
        commandBufferCount=1,
        pCommandBuffers=[plan.command_buffer],
    )
    vkQueueSubmit(plan.device.queue, 1, [submit_info], plan.fence)
    vkWaitForFences(plan.device.device, 1, [plan.fence], True, _WAIT_TIMEOUT_NS)
    if plan.profile_state is not None and plan.profile_recorder is not None:
        timestamps = plan.profile_state.read_timestamps(plan.device)
        record_replay_execution = getattr(plan.profile_recorder, "record_replay_execution")
        record_replay_execution(plan=plan, timestamps=timestamps)

    results: dict[str, bytes] = {}
    for name, slot in plan.readback_slots.items():
        plan.device.memory_manager.host_upload_ring.invalidate(
            allocation=slot.allocation, size=slot.nbytes,
        )
        results[name] = slot.allocation.buffer.read_bytes_at(
            slot.allocation.offset, slot.nbytes
        )
    return results


def _write_indirect_dispatch_buffer(
    plan: ReplayPlan, dynamic_symbols: Mapping[str, int]
) -> None:
    assert plan.indirect_buffer is not None
    data = bytearray(plan.num_dispatches * 12)
    for entry in plan.dispatch_entries:
        if entry.indirect_offset is None:
            continue
        symbols = {**entry.symbols, **dynamic_symbols}
        x = eval_expr(entry.dispatch_formula[0], symbols)
        y = eval_expr(entry.dispatch_formula[1], symbols)
        z = eval_expr(entry.dispatch_formula[2], symbols)
        offset = entry.indirect_offset
        struct.pack_into("<III", data, offset, x, y, z)
    plan.indirect_buffer.buffer.write_bytes_at(
        plan.indirect_buffer.offset, bytes(data)
    )
    plan.device.memory_manager.host_upload_ring.flush(allocation=plan.indirect_buffer)


def _write_params_buffers(plan: ReplayPlan, dynamic_symbols: Mapping[str, int]) -> None:
    for entry in plan.params_entries:
        if entry.params_buffer is None or entry.params_layout is None:
            raise RuntimeError("Replay params entry is missing its params buffer")
        symbols = {**entry.symbols, **dynamic_symbols}
        data = bytearray(entry.params_layout.size)
        for field in entry.params_layout.fields:
            raw = field.value
            if isinstance(raw, PushConstantInput):
                raise ValueError(f"PushConstantInput {raw.name!r} is not supported by replay params")
            if callable(raw):
                raise ValueError(f"Callable replay param {field.name!r} is not supported")
            if isinstance(raw, str):
                value = symbols[raw]
            elif isinstance(raw, int | float):
                value = raw
            else:
                value = eval_expr(raw, symbols)
            if field.dtype is PushConstantType.UINT32:
                struct.pack_into("<I", data, field.offset, int(value))
            elif field.dtype is PushConstantType.INT32:
                struct.pack_into("<i", data, field.offset, int(value))
            elif field.dtype is PushConstantType.FLOAT32:
                struct.pack_into("<f", data, field.offset, float(value))
            elif field.dtype is PushConstantType.UINT64:
                struct.pack_into("<Q", data, field.offset, int(value))
        entry.params_buffer.buffer.write_bytes_at(entry.params_buffer.offset, bytes(data))
        plan.device.memory_manager.host_upload_ring.flush(allocation=entry.params_buffer)
