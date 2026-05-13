"""Replay plan: pre-recorded command buffer execution for decode/prefill stages."""

from __future__ import annotations

import struct
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
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

from torch2vk.runtime.host_array import prepare_host_array
from torch2vk.runtime.logical import TensorRole
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
    from torch2vk.runtime.logical import LogicalTensor
    from torch2vk.runtime.session import RuntimeSession
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
    dynamic_symbol_names: tuple[str, ...] = ()
    indirect_offset: int | None = None
    params_buffer: BufferAllocation | None = None
    params_layout: ParamsBufferSpec | None = None
    source_dispatch_index: int | None = None
    source_frame: str | None = None
    source_shader: str | None = None
    source_logical_reads: tuple[tuple[str, str], ...] = ()
    source_logical_writes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ReplayDispatchTemplate:
    """Device-independent dispatch metadata captured from an eager frame."""

    shader: str
    logical_reads: tuple[tuple[str, str], ...]
    logical_writes: tuple[tuple[str, str], ...]
    symbols: tuple[tuple[str, int], ...]
    dispatch_size: tuple[int, int, int]
    dynamic_symbol_names: tuple[str, ...]
    source_dispatch_index: int | None = None
    source_frame: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayPlanTemplate:
    """Device-independent replay description reusable across RuntimeSession instances."""

    name: str
    entries: tuple[ReplayDispatchTemplate, ...]
    dynamic_symbol_names: tuple[str, ...]


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
    tensor: "LogicalTensor"
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
    template: ReplayPlanTemplate | None

    readback_slots: dict[str, ReplayReadbackSlot]

    workspace_allocations: list[BufferAllocation]
    bindings: list[BoundComputeBinding]
    profile_state: ReplayProfileState | None = None
    profile_recorder: object | None = None
    in_place_staging_rebound: bool = False

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

    host_start_ns = time.perf_counter_ns()
    profile_collection_ns = 0
    try:
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
            profile_start_ns = time.perf_counter_ns()
            timestamps = plan.profile_state.read_timestamps(plan.device)
            record_replay_execution = getattr(plan.profile_recorder, "record_replay_execution")
            record_replay_execution(plan=plan, timestamps=timestamps)
            profile_collection_ns = time.perf_counter_ns() - profile_start_ns
            record_host_event = getattr(plan.profile_recorder, "record_host_event", None)
            if record_host_event is not None:
                record_host_event(
                    name="collect_replay_profile",
                    replay_plan=plan.name,
                    elapsed_wall_ns=profile_collection_ns,
                )

        results: dict[str, bytes] = {}
        for name, slot in plan.readback_slots.items():
            plan.device.memory_manager.host_upload_ring.invalidate(
                allocation=slot.allocation, size=slot.nbytes,
            )
            results[name] = slot.allocation.buffer.read_bytes_at(
                slot.allocation.offset, slot.nbytes
            )
        return results
    finally:
        if plan.profile_recorder is not None:
            record_host_event = getattr(plan.profile_recorder, "record_host_event", None)
            if record_host_event is not None:
                record_host_event(
                    name="execute_replay",
                    replay_plan=plan.name,
                    elapsed_wall_ns=(
                        time.perf_counter_ns() - host_start_ns - profile_collection_ns
                    ),
                )


def stage_replay_step_inputs(
    rt: "RuntimeSession",
    *,
    plan: ReplayPlan,
    inputs: Mapping["LogicalTensor", np.ndarray],
    write_through: tuple["LogicalTensor", ...] = (),
) -> None:
    """Update replay step inputs and rebind plan descriptors to the latest buffers."""
    host_start_ns = time.perf_counter_ns()
    try:
        if _can_stage_replay_inputs_in_place(inputs):
            for tensor, value in inputs.items():
                tensor.validate_declaration()
                if tensor.role is not TensorRole.INPUT:
                    raise ValueError(f"{tensor.name} is not an input tensor")
                rt._inputs[tensor] = value
                _write_tensor_buffer(rt, tensor, value)
            if not plan.in_place_staging_rebound:
                rt.rebind_replay_plan(plan)
                plan.in_place_staging_rebound = True
            return

        rt.register_inputs(inputs)
        for tensor in write_through:
            value = inputs.get(tensor)
            if value is None:
                continue
            if tensor.buffer is None:
                rt._materialize_read(tensor)
            _write_tensor_buffer(rt, tensor, value)
        rt.rebind_replay_plan(plan)
        plan.in_place_staging_rebound = True
    finally:
        if rt.profiler.enabled:
            rt.profiler.record_host_event(
                name="stage_replay_step_inputs",
                replay_plan=plan.name,
                elapsed_wall_ns=time.perf_counter_ns() - host_start_ns,
            )


def _can_stage_replay_inputs_in_place(
    inputs: Mapping["LogicalTensor", np.ndarray],
) -> bool:
    for tensor in inputs:
        if tensor.buffer is None or tensor.buffer.allocation.released:
            return False
    return True


def _write_tensor_buffer(
    rt: "RuntimeSession",
    tensor: "LogicalTensor",
    data: np.ndarray,
) -> None:
    contiguous = prepare_host_array(tensor, data, context="replay write")
    if tensor.buffer is None:
        raise RuntimeError(f"Tensor {tensor.name} not materialized for replay write")
    raw_bytes = memoryview(contiguous).cast("B")
    if raw_bytes.nbytes > tensor.buffer.nbytes:
        raise ValueError(
            f"{tensor.name} replay write has {raw_bytes.nbytes} bytes, "
            f"buffer only has {tensor.buffer.nbytes}"
        )
    tensor.buffer.allocation.buffer.write_bytes_at(
        tensor.buffer.offset, raw_bytes
    )
    rt.device.memory_manager.host_upload_ring.flush(
        allocation=tensor.buffer.allocation,
        size=raw_bytes.nbytes,
    )


def _write_indirect_dispatch_buffer(
    plan: ReplayPlan, dynamic_symbols: Mapping[str, int]
) -> None:
    assert plan.indirect_buffer is not None
    data = bytearray(plan.num_dispatches * 12)
    for entry in plan.dispatch_entries:
        if entry.indirect_offset is None:
            continue
        symbols = _entry_symbols(entry, dynamic_symbols)
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
        symbols = _entry_symbols(entry, dynamic_symbols)
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


def _entry_symbols(
    entry: ReplayDispatchEntry,
    dynamic_symbols: Mapping[str, int],
) -> dict[str, int]:
    if not dynamic_symbols or not entry.dynamic_symbol_names:
        return entry.symbols
    symbols = dict(entry.symbols)
    for name in entry.dynamic_symbol_names:
        value = dynamic_symbols.get(name)
        if value is not None:
            symbols[name] = value
    return symbols
