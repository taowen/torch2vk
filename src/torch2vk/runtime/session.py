"""RuntimeSession lifecycle and compatibility facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np

from torch2vk.runtime.compare import TensorCompareResult
from torch2vk.runtime.frame import FrameContext
from torch2vk.runtime.logical import (
    LogicalTensor,
    TensorRole,
    collect_named_logical_tensors,
)
from torch2vk.runtime.shader import (
    DispatchRecord,
    IOKind,
    ParamsBufferSpec,
    PushConstantSpec,
    ShaderVariant,
    TensorFieldSpec,
)
from torch2vk.vulkan.allocation import BufferAllocation, BufferSlice
from torch2vk.vulkan.compute_pipeline import ComputePipeline
from torch2vk.vulkan.device import VulkanDevice
from torch2vk.vulkan.types import tensor_nbytes

if TYPE_CHECKING:
    from torch2vk.runtime.replay import ReplayPlan


class RuntimeSession:
    """The single runtime owner for LogicalTensor materialization and shader dispatch."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        artifact_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        profile_dir: str | Path | None = None,
        model_tensors: object | None = None,
        model_shaders: dict[str, ShaderVariant] | None = None,
    ) -> None:
        from torch2vk.runtime.profile import RuntimeProfiler

        self.device = VulkanDevice(physical_device_index=device_index)
        self.profiler = RuntimeProfiler(profile_dir)
        self.profiler.attach_device(self.device)
        self.artifact_dir = Path(
            ".cache/torch2vk/generated" if artifact_dir is None else artifact_dir
        )
        self.model_dir = None if model_dir is None else Path(model_dir).expanduser().resolve()
        if model_tensors is not None:
            collect_named_logical_tensors(model_tensors)
        self._model_tensors = model_tensors
        self._model_shaders = model_shaders
        self._inputs: dict[LogicalTensor, object] = {}
        self._frame_stack: list[FrameContext] = []
        self._frame_history: dict[str, FrameContext] = {}
        self._dispatch_records: list[DispatchRecord] = []
        self._compare_results: list[TensorCompareResult] = []
        self._pipeline_cache: dict[tuple[object, ...], ComputePipeline] = {}
        self._replay_plan_cache: dict[str, list[ReplayPlan]] = {}

        self._model_allocations: list[BufferAllocation] = []
        self._request_allocations: list[BufferAllocation] = []
        self._frame_allocations: list[tuple[LogicalTensor, BufferAllocation]] = []
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        device_index: int = 0,
        artifact_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        profile_dir: str | Path | None = None,
        model_tensors: object | None = None,
        model_shaders: dict[str, ShaderVariant] | None = None,
    ) -> "RuntimeSession":
        return cls(
            device_index=device_index,
            artifact_dir=artifact_dir,
            model_dir=model_dir,
            profile_dir=profile_dir,
            model_shaders=model_shaders,
            model_tensors=model_tensors,
        )

    def __enter__(self) -> "RuntimeSession":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def dispatch_records(self) -> tuple[DispatchRecord, ...]:
        return tuple(self._dispatch_records)

    @property
    def compare_results(self) -> tuple[TensorCompareResult, ...]:
        return tuple(self._compare_results)

    def register_inputs(self, inputs: Mapping[LogicalTensor, object]) -> None:
        for tensor, value in inputs.items():
            if not isinstance(tensor, LogicalTensor):
                raise TypeError(
                    f"register_inputs key must be LogicalTensor, got {type(tensor).__name__}"
                )
            tensor.validate_declaration()
            if tensor.role is not TensorRole.INPUT:
                raise ValueError(f"{tensor.name} is not an input tensor")
            self._inputs[tensor] = value
            self._invalidate_input_materialization(tensor)
            self._record_frame_input(tensor)

    def initialize_request_state(self, states: Mapping[LogicalTensor, object]) -> None:
        from torch2vk.runtime.request_state import initialize_request_state

        initialize_request_state(self, states)

    def read_request_state(self, tensor: LogicalTensor) -> np.ndarray:
        from torch2vk.runtime.request_state import read_request_state

        return read_request_state(self, tensor)

    def grow_request_state(
        self,
        tensor: LogicalTensor,
        new_shape: Sequence[int],
        *,
        growth: str = "geometric",
    ) -> None:
        from torch2vk.runtime.request_state import grow_request_state

        grow_request_state(self, tensor, new_shape, growth=growth)

    def release_request_state(self, tensors: Sequence[LogicalTensor] | None = None) -> None:
        from torch2vk.runtime.request_state import release_request_state

        release_request_state(self, tensors)

    @contextmanager
    def frame(self, name: str):
        if not name:
            raise ValueError("frame name must be non-empty")
        context = FrameContext(
            frame=name,
            start_dispatch_index=len(self._dispatch_records),
            end_dispatch_index=len(self._dispatch_records),
        )
        self._frame_stack.append(context)
        try:
            yield context
            context.end_dispatch_index = len(self._dispatch_records)
        finally:
            popped = self._frame_stack.pop()
            if popped is not context:
                raise RuntimeError("RuntimeSession frame stack corrupted")
            self._frame_history[context.frame] = context
            self._release_frame_allocations()

    def dispatch(self, variant: ShaderVariant, **arguments: object) -> None:
        from torch2vk.runtime.dispatcher import dispatch

        dispatch(self, variant, **arguments)

    def readback(self, tensor: LogicalTensor) -> np.ndarray:
        self._require_open()
        if tensor.buffer is None:
            if tensor_nbytes(tensor.spec) == 0:
                return self.device.empty_tensor(spec=tensor.spec)
            raise RuntimeError(f"{tensor.name} is not materialized")
        return self.device.readback_tensor(
            spec=tensor.spec, slice=tensor.buffer, layout=tensor.layout
        )

    def debug_materialization(self, tensor: LogicalTensor) -> BufferSlice | None:
        return tensor.buffer

    def build_replay_plan(
        self,
        *,
        name: str,
        frame: str,
        readback_tensors: Mapping[str, LogicalTensor] | None = None,
    ) -> "ReplayPlan":
        from torch2vk.runtime.replay_builder import build_replay_plan

        return build_replay_plan(
            self,
            name=name,
            frame=frame,
            readback_tensors=readback_tensors,
        )

    def rebind_replay_plan(
        self,
        plan: "ReplayPlan",
    ) -> None:
        from torch2vk.runtime.replay_builder import rebind_replay_plan

        rebind_replay_plan(self, plan)

    def replay_plan_compatible(self, plan: "ReplayPlan") -> bool:
        from torch2vk.runtime.replay_builder import replay_plan_compatible

        return replay_plan_compatible(self, plan)

    def cached_replay_plans(self, namespace: str) -> tuple["ReplayPlan", ...]:
        from torch2vk.runtime.replay_builder import cached_replay_plans

        return cached_replay_plans(self, namespace)

    def cache_replay_plan(self, namespace: str, plan: "ReplayPlan") -> None:
        from torch2vk.runtime.replay_builder import cache_replay_plan

        cache_replay_plan(self, namespace, plan)

    def profile_summary(self) -> dict[str, object]:
        return self.profiler.write_summary()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for plans in self._replay_plan_cache.values():
            for plan in plans:
                plan.close()
        self._replay_plan_cache.clear()
        self._release_frame_allocations()
        for pipeline in self._pipeline_cache.values():
            pipeline.close()
        self._pipeline_cache.clear()
        for allocation in reversed(self._request_allocations):
            allocation.close()
        self._request_allocations.clear()
        for allocation in reversed(self._model_allocations):
            allocation.close()
        self._model_allocations.clear()
        self.profiler.close()
        self.device.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RuntimeSession is closed")

    def _current_frame(self) -> FrameContext:
        if not self._frame_stack:
            raise RuntimeError("RuntimeSession.dispatch requires an active rt.frame(...)")
        return self._frame_stack[-1]

    def _active_frame(self) -> FrameContext | None:
        if not self._frame_stack:
            return None
        return self._frame_stack[-1]

    def _record_frame_input(self, tensor: LogicalTensor) -> None:
        frame = self._active_frame()
        if frame is not None:
            frame.registered_inputs.append(tensor)

    def _invalidate_input_materialization(self, tensor: LogicalTensor) -> None:
        if tensor.buffer is None and tensor.descriptor_nbytes is None:
            return
        with tensor.runtime_write_scope():
            tensor.buffer = None
            tensor.descriptor_nbytes = None
            tensor.alias_source = None

    def _named_model_tensors(self) -> dict[str, LogicalTensor]:
        if self._model_tensors is None:
            raise RuntimeError(
                "Replay requires RuntimeSession.open(..., model_tensors=...)"
            )
        return collect_named_logical_tensors(self._model_tensors)

    def _named_model_shaders(self) -> dict[str, ShaderVariant]:
        if self._model_shaders is None:
            raise RuntimeError(
                "Replay requires RuntimeSession.open(..., model_shaders=...)"
            )
        return self._model_shaders

    def _bind_shape_symbols(
        self,
        fields: tuple[TensorFieldSpec, ...],
        tensors: Mapping[str, LogicalTensor],
    ) -> dict[str, int]:
        from torch2vk.runtime.materialization import bind_shape_symbols

        return bind_shape_symbols(self, fields, tensors)

    def _materialize_read(self, tensor: LogicalTensor) -> None:
        from torch2vk.runtime.materialization import materialize_read

        materialize_read(self, tensor)

    def _materialize_write(self, tensor: LogicalTensor, *, io_kind: IOKind) -> None:
        from torch2vk.runtime.materialization import materialize_write

        materialize_write(self, tensor, io_kind=io_kind)

    def _materialize_weight(self, tensor: LogicalTensor) -> None:
        from torch2vk.runtime.materialization import materialize_weight

        materialize_weight(self, tensor)

    def _resolve_weight_checkpoint(self, tensor: LogicalTensor) -> Path:
        from torch2vk.runtime.materialization import resolve_weight_checkpoint

        return resolve_weight_checkpoint(self, tensor)

    def _materialize_input(self, tensor: LogicalTensor) -> None:
        from torch2vk.runtime.materialization import materialize_input

        materialize_input(self, tensor)

    def _pack_push_constants(
        self,
        spec: PushConstantSpec | None,
        *,
        tensors: Mapping[str, LogicalTensor],
        symbols: Mapping[str, int],
    ) -> tuple[bytes | None, dict[str, int | float]]:
        from torch2vk.runtime.materialization import pack_push_constants

        return pack_push_constants(self, spec, tensors=tensors, symbols=symbols)

    def _materialize_params_buffer(
        self,
        spec: ParamsBufferSpec,
        *,
        tensors: Mapping[str, LogicalTensor],
        symbols: Mapping[str, int],
    ) -> BufferAllocation:
        from torch2vk.runtime.materialization import materialize_params_buffer

        return materialize_params_buffer(self, spec, tensors=tensors, symbols=symbols)

    def _pipeline_for_variant(self, variant: ShaderVariant) -> ComputePipeline:
        from torch2vk.runtime.pipeline_cache import pipeline_for_variant

        return pipeline_for_variant(self, variant)

    def _spv_path_for_variant(self, variant: ShaderVariant) -> Path:
        from torch2vk.runtime.pipeline_cache import spv_path_for_variant

        return spv_path_for_variant(self, variant)

    def _release_frame_allocations(self) -> None:
        from torch2vk.runtime.materialization import release_frame_allocations

        release_frame_allocations(self)

    def _release_request_allocation(self, allocation: BufferAllocation) -> None:
        from torch2vk.runtime.materialization import release_request_allocation

        release_request_allocation(self, allocation)
