"""RuntimeSession lifecycle and compatibility facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np

from torch2vk.checkpoints.gguf import GGUFMmap, open_gguf_mmap
from torch2vk.checkpoints.safetensors import SafetensorsMmap, open_safetensors_mmap
from torch2vk.runtime.compare import TensorCompareResult
from torch2vk.runtime.frame import FrameContext
from torch2vk.runtime.host_array import prepare_host_array
from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
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


def _request_input_tensor(
    tensors: Mapping[str, LogicalTensor],
    name: str,
) -> LogicalTensor:
    tensor = tensors.get(name)
    if tensor is None:
        raise KeyError(f"request input {name!r} is not a named model tensor")
    return tensor


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
        get_shader: Callable[[str], ShaderVariant] | None = None,
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
        self._get_shader = get_shader
        self._inputs: dict[LogicalTensor, object] = {}
        self._request_active = False
        self._frame_stack: list[FrameContext] = []
        self._frame_history: dict[str, FrameContext] = {}
        self._dispatch_records: list[DispatchRecord] = []
        self._compare_results: list[TensorCompareResult] = []
        self._pipeline_cache: dict[tuple[object, ...], ComputePipeline] = {}
        self._replay_plan_cache: dict[str, list[ReplayPlan]] = {}
        self._checkpoint_storage_cache: dict[Path, SafetensorsMmap | GGUFMmap] = {}

        self._model_allocations: list[BufferAllocation] = []
        self._request_allocations: list[BufferAllocation] = []
        self._request_tensors: set[LogicalTensor] = set()
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
        get_shader: Callable[[str], ShaderVariant] | None = None,
    ) -> "RuntimeSession":
        return cls(
            device_index=device_index,
            artifact_dir=artifact_dir,
            model_dir=model_dir,
            profile_dir=profile_dir,
            get_shader=get_shader,
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
            if tensor.memory is MemoryClass.SESSION_TENSOR:
                raise ValueError(
                    f"{tensor.name} is a session tensor; use register_session_tensors(...)"
                )
            self._inputs[tensor] = value
            self._invalidate_input_materialization(tensor)
            self._record_frame_input(tensor)

    @contextmanager
    def request(self, **inputs: object):
        """Start a request scope and optionally register host inputs for it.

        Request keyword names are resolved against RuntimeSession.model_tensors by
        LogicalTensor.name, then registered with register_inputs().
        """
        self._require_open()
        if self._request_active:
            raise RuntimeError("RuntimeSession.request cannot be nested")
        self._request_active = True
        try:
            if inputs:
                tensors = self._named_model_tensors()
                self.register_inputs(
                    {
                        _request_input_tensor(tensors, name): value
                        for name, value in inputs.items()
                    }
                )
            yield self
        finally:
            try:
                self._close_replay_plan_cache()
                self._clear_request_state()
                self._inputs.clear()
            finally:
                self._request_active = False

    def register_session_tensors(self, tensors: Mapping[LogicalTensor, object]) -> None:
        self._require_open()
        for tensor, value in tensors.items():
            if not isinstance(tensor, LogicalTensor):
                raise TypeError(
                    f"register_session_tensors key must be LogicalTensor, got {type(tensor).__name__}"
                )
            tensor.validate_declaration()
            if (
                tensor.role is not TensorRole.INPUT
                or tensor.memory is not MemoryClass.SESSION_TENSOR
            ):
                raise ValueError(f"{tensor.name} is not a session tensor input")
            if tensor.buffer is not None or tensor.descriptor_nbytes is not None:
                raise RuntimeError(f"{tensor.name} session tensor is already registered")
            array = prepare_host_array(tensor, value, context="session tensor")
            expected = tensor_nbytes(tensor.spec)
            if array.nbytes != expected:
                raise ValueError(
                    f"{tensor.name} session tensor has {array.nbytes} bytes, expected {expected}"
                )
            if expected == 0:
                with tensor.runtime_write_scope():
                    tensor.buffer = None
                    tensor.descriptor_nbytes = 0
                    tensor.alias_source = None
                continue
            ((slice_, allocation),) = self.device.upload_numpy_arrays_with_allocations(
                [(tensor.name, array)]
            )
            with tensor.runtime_write_scope():
                tensor.buffer = slice_
                tensor.descriptor_nbytes = slice_.nbytes
                tensor.alias_source = None
            self._model_allocations.append(allocation)

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

    def materialize_model_weights(self) -> None:
        if self._model_tensors is None:
            raise RuntimeError("RuntimeSession has no model_tensors to materialize")
        for tensor in collect_named_logical_tensors(self._model_tensors).values():
            if tensor.role is TensorRole.WEIGHT:
                self._materialize_weight(tensor)

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
        self._close_replay_plan_cache()
        self._release_frame_allocations()
        for pipeline in self._pipeline_cache.values():
            pipeline.close()
        self._pipeline_cache.clear()
        self._clear_request_state()
        for allocation in reversed(self._model_allocations):
            allocation.close()
        self._model_allocations.clear()
        for storage in self._checkpoint_storage_cache.values():
            storage.close()
        self._checkpoint_storage_cache.clear()
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
            raise RuntimeError("Replay requires RuntimeSession.open(..., model_tensors=...)")
        return collect_named_logical_tensors(self._model_tensors)

    def _model_shader(self, name: str) -> ShaderVariant:
        if self._get_shader is None:
            raise RuntimeError("Replay requires RuntimeSession.open(..., get_shader=...)")
        return self._get_shader(name)

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

    def _checkpoint_storage(self, checkpoint: Path) -> SafetensorsMmap | GGUFMmap:
        self._require_open()
        resolved = checkpoint.expanduser().resolve()
        storage = self._checkpoint_storage_cache.get(resolved)
        if storage is not None:
            return storage
        if resolved.suffix == ".gguf":
            storage = open_gguf_mmap(resolved)
        else:
            storage = open_safetensors_mmap(resolved)
        self._checkpoint_storage_cache[resolved] = storage
        return storage

    def _materialize_input(self, tensor: LogicalTensor) -> None:
        from torch2vk.runtime.materialization import materialize_input

        materialize_input(self, tensor)

    def _pack_push_constants(
        self,
        spec: PushConstantSpec | None,
        *,
        tensors: Mapping[str, LogicalTensor],
        symbols: Mapping[str, int],
        push_constant_inputs: Mapping[str, object] | None = None,
    ) -> tuple[bytes | None, dict[str, int | float]]:
        from torch2vk.runtime.materialization import pack_push_constants

        return pack_push_constants(
            self,
            spec,
            tensors=tensors,
            symbols=symbols,
            push_constant_inputs=push_constant_inputs,
        )

    def _materialize_params_buffer(
        self,
        spec: ParamsBufferSpec,
        *,
        tensors: Mapping[str, LogicalTensor],
        symbols: Mapping[str, int],
        push_constant_inputs: Mapping[str, object] | None = None,
    ) -> BufferAllocation:
        from torch2vk.runtime.materialization import materialize_params_buffer

        return materialize_params_buffer(
            self,
            spec,
            tensors=tensors,
            symbols=symbols,
            push_constant_inputs=push_constant_inputs,
        )

    def _pipeline_for_variant(self, variant: ShaderVariant) -> ComputePipeline:
        from torch2vk.runtime.pipeline_cache import pipeline_for_variant

        return pipeline_for_variant(self, variant)

    def _spv_path_for_variant(self, variant: ShaderVariant) -> Path:
        from torch2vk.runtime.pipeline_cache import spv_path_for_variant

        return spv_path_for_variant(self, variant)

    def _release_frame_allocations(self) -> None:
        from torch2vk.runtime.materialization import release_frame_allocations

        release_frame_allocations(self)

    def _clear_request_state(self) -> None:
        from torch2vk.runtime.request_state import _clear_request_state

        _clear_request_state(self)

    def _close_replay_plan_cache(self) -> None:
        for plans in self._replay_plan_cache.values():
            for plan in plans:
                plan.close()
        self._replay_plan_cache.clear()

    def _release_request_allocation(self, allocation: BufferAllocation) -> None:
        from torch2vk.runtime.materialization import release_request_allocation

        release_request_allocation(self, allocation)
