"""RuntimeSession lifecycle and compatibility facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Literal

import numpy as np

from torch2vk.runtime.compare import TensorCompareResult
from torch2vk.runtime.frame import FrameContext
from torch2vk.runtime.logical import LogicalTensor, TensorRole
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
    ) -> None:
        self.device = VulkanDevice(physical_device_index=device_index)
        self.artifact_dir = Path(
            ".cache/torch2vk/generated" if artifact_dir is None else artifact_dir
        )
        self.model_dir = None if model_dir is None else Path(model_dir).expanduser().resolve()
        self._inputs: dict[LogicalTensor, object] = {}
        self._frame_stack: list[FrameContext] = []
        self._frame_history: dict[str, FrameContext] = {}
        self._dispatch_records: list[DispatchRecord] = []
        self._compare_results: list[TensorCompareResult] = []
        self._pipeline_cache: dict[tuple[object, ...], ComputePipeline] = {}
        self._replay_plan_cache: dict[str, list[ReplayPlan]] = {}
        self._pytorch_models: dict[object, object] = {}
        self._pytorch_cache_states: dict[str, object] = {}
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
    ) -> "RuntimeSession":
        return cls(device_index=device_index, artifact_dir=artifact_dir, model_dir=model_dir)

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
    def frame(
        self,
        name: str,
        *,
        pytorch_model: object | None = None,
        pytorch_model_class: object | None = None,
        pytorch_model_submodule: str | None = None,
        pytorch_args: tuple[object, ...] = (),
        pytorch_kwargs: Mapping[str, object] | None = None,
        pytorch_input_prefixes: Sequence[str] = (),
        pytorch_cache_policy: str = "none",
        pytorch_cache_namespace: str | None = None,
        pytorch_reset_cache: bool = False,
        reference_model: object | None = None,
    ):
        if not name:
            raise ValueError("frame name must be non-empty")
        cache_policy: Literal["none", "hf_dynamic"]
        if pytorch_cache_policy == "none":
            cache_policy = "none"
        elif pytorch_cache_policy == "hf_dynamic":
            cache_policy = "hf_dynamic"
        else:
            raise ValueError(f"Unsupported PyTorch cache policy: {pytorch_cache_policy!r}")
        if pytorch_model is None and pytorch_model_class is not None:
            loaded = self._load_pytorch_model(pytorch_model_class)
            if loaded is not None and pytorch_model_submodule:
                for attr in pytorch_model_submodule.split("."):
                    loaded = getattr(loaded, attr)
            pytorch_model = loaded
        context = FrameContext(
            frame=name,
            start_dispatch_index=len(self._dispatch_records),
            pytorch_model=pytorch_model,
            pytorch_args=tuple(pytorch_args),
            pytorch_kwargs={} if pytorch_kwargs is None else dict(pytorch_kwargs),
            pytorch_input_prefixes=tuple(pytorch_input_prefixes),
            pytorch_cache_policy=cache_policy,
            pytorch_cache_namespace=pytorch_cache_namespace,
            pytorch_reset_cache=pytorch_reset_cache,
            reference_model=reference_model,
        )
        self._frame_stack.append(context)
        candidate_completed = False
        try:
            yield context
            candidate_completed = True
        finally:
            try:
                if candidate_completed:
                    self._compare_frame(context)
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

    def _compare_frame(self, frame: FrameContext) -> None:
        from torch2vk.runtime.pytorch_debug import compare_frame

        compare_frame(self, frame)

    def _load_pytorch_model(self, model_class: object) -> object | None:
        from torch2vk.runtime.pytorch_debug import load_pytorch_model

        return load_pytorch_model(self, model_class)

    def build_replay_plan(
        self,
        *,
        name: str,
        frame_dispatch_records: Sequence[DispatchRecord],
        variants: Sequence[ShaderVariant],
        tensors_by_name: Mapping[str, LogicalTensor],
        dynamic_symbol_names: tuple[str, ...] = (),
        readback_tensors: Mapping[str, LogicalTensor] | None = None,
        token_feedback_source: LogicalTensor | None = None,
        token_feedback_target: LogicalTensor | None = None,
    ) -> "ReplayPlan":
        from torch2vk.runtime.replay_builder import build_replay_plan

        return build_replay_plan(
            self,
            name=name,
            frame_dispatch_records=frame_dispatch_records,
            variants=variants,
            tensors_by_name=tensors_by_name,
            dynamic_symbol_names=dynamic_symbol_names,
            readback_tensors=readback_tensors,
            token_feedback_source=token_feedback_source,
            token_feedback_target=token_feedback_target,
        )

    def rebind_replay_plan(
        self,
        plan: "ReplayPlan",
        *,
        tensors_by_name: Mapping[str, LogicalTensor],
    ) -> None:
        from torch2vk.runtime.replay_builder import rebind_replay_plan

        rebind_replay_plan(self, plan, tensors_by_name=tensors_by_name)

    def cached_replay_plans(self, namespace: str) -> tuple["ReplayPlan", ...]:
        from torch2vk.runtime.replay_builder import cached_replay_plans

        return cached_replay_plans(self, namespace)

    def cache_replay_plan(self, namespace: str, plan: "ReplayPlan") -> None:
        from torch2vk.runtime.replay_builder import cache_replay_plan

        cache_replay_plan(self, namespace, plan)

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
        self.device.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RuntimeSession is closed")

    def _current_frame(self) -> FrameContext:
        if not self._frame_stack:
            raise RuntimeError("RuntimeSession.dispatch requires an active rt.frame(...)")
        return self._frame_stack[-1]

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
