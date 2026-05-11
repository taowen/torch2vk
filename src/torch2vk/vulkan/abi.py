"""Typed boundary helpers for the dynamic ``vulkan`` CFFI package."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class _IndexableObject(Protocol):
    def __getitem__(self, index: int) -> object: ...


@runtime_checkable
class _SizedIndexableObject(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> object: ...


@dataclass(frozen=True, slots=True)
class VkDeviceNameProperties:
    device_name: bytes | str
    api_version: int


@dataclass(frozen=True, slots=True)
class VkPhysicalDeviceProperties:
    timestamp_period: int | float


@dataclass(frozen=True, slots=True)
class VkMemoryType:
    property_flags: int


@dataclass(frozen=True, slots=True)
class VkPhysicalDeviceMemoryProperties:
    memory_types: tuple[VkMemoryType, ...]

    @property
    def memory_type_count(self) -> int:
        return len(self.memory_types)


@dataclass(frozen=True, slots=True)
class VkMemoryRequirements:
    memory_type_bits: int
    size: int


class VkCooperativeMatrixProperty:
    def __init__(self, source: object) -> None:
        self.m_size = _required_int(source, "MSize")
        self.n_size = _required_int(source, "NSize")
        self.k_size = _required_int(source, "KSize")
        self.a_type = _required_int(source, "AType")
        self.b_type = _required_int(source, "BType")
        self.c_type = _required_int(source, "CType")
        self.result_type = _required_int(source, "ResultType")
        self.scope = _required_int(source, "scope")
        self.saturating_accumulation = bool(_required_bool_or_int(source, "saturatingAccumulation"))


SetDebugObjectNameProc = Callable[[object, object], int | None]
BeginDebugLabelProc = Callable[[object, object], None]
EndDebugLabelProc = Callable[[object], None]


def device_name_properties(source: object) -> VkDeviceNameProperties:
    return VkDeviceNameProperties(
        device_name=_required_name(source, "deviceName"),
        api_version=_required_int(source, "apiVersion"),
    )


def physical_device_properties(source: object) -> VkPhysicalDeviceProperties:
    limits = _required_object(source, "limits")
    return VkPhysicalDeviceProperties(timestamp_period=_required_int_or_float(limits, "timestampPeriod"))


def physical_device_memory_properties(source: object) -> VkPhysicalDeviceMemoryProperties:
    memory_type_count = _required_int(source, "memoryTypeCount")
    raw_memory_types = _required_indexable(source, "memoryTypes")
    memory_types: list[VkMemoryType] = []
    for index in range(memory_type_count):
        memory_type = raw_memory_types[index]
        memory_types.append(VkMemoryType(property_flags=_required_int(memory_type, "propertyFlags")))
    return VkPhysicalDeviceMemoryProperties(memory_types=tuple(memory_types))


def memory_requirements(source: object) -> VkMemoryRequirements:
    return VkMemoryRequirements(
        memory_type_bits=_required_int(source, "memoryTypeBits"),
        size=_required_int(source, "size"),
    )


def cooperative_matrix_properties_proc(value: object | None) -> Callable[[object], tuple[VkCooperativeMatrixProperty, ...]] | None:
    fn = _callable_proc(value)
    if fn is None:
        return None

    def query(physical_device: object) -> tuple[VkCooperativeMatrixProperty, ...]:
        raw_properties = fn(physical_device)
        if not isinstance(raw_properties, _SizedIndexableObject):
            raise TypeError("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR did not return an array")
        property_count = len(raw_properties)
        return tuple(VkCooperativeMatrixProperty(raw_properties[index]) for index in range(property_count))

    return query


def set_debug_object_name_proc(value: object | None) -> SetDebugObjectNameProc | None:
    fn = _callable_proc(value)
    if fn is None:
        return None

    def set_object_name(device: object, info: object) -> int | None:
        result = fn(device, info)
        if result is None:
            return None
        if isinstance(result, int):
            return result
        raise TypeError(f"vkSetDebugUtilsObjectNameEXT returned {type(result).__name__}")

    return set_object_name


def begin_debug_label_proc(value: object | None) -> BeginDebugLabelProc | None:
    fn = _callable_proc(value)
    if fn is None:
        return None

    def begin_label(command_buffer: object, label: object) -> None:
        fn(command_buffer, label)

    return begin_label


def end_debug_label_proc(value: object | None) -> EndDebugLabelProc | None:
    fn = _callable_proc(value)
    if fn is None:
        return None

    def end_label(command_buffer: object) -> None:
        fn(command_buffer)

    return end_label


def _callable_proc(value: object | None) -> Callable[..., object] | None:
    if value is None:
        return None
    if not callable(value):
        raise TypeError(f"Expected Vulkan procedure pointer, got {type(value).__name__}")
    return value


def _required_object(source: object, attr: str) -> object:
    try:
        return getattr(source, attr)
    except AttributeError as exc:
        raise TypeError(f"Vulkan object {type(source).__name__} is missing attribute {attr}") from exc


def _required_sequence(source: object, attr: str) -> Sequence[object]:
    value = _required_object(source, attr)
    if not isinstance(value, Sequence):
        raise TypeError(f"Vulkan attribute {attr} must be a sequence, got {type(value).__name__}")
    return value


def _required_indexable(source: object, attr: str) -> _IndexableObject:
    value = _required_object(source, attr)
    if not isinstance(value, _IndexableObject):
        raise TypeError(f"Vulkan attribute {attr} must support integer indexing, got {type(value).__name__}")
    return value


def _required_int(source: object, attr: str) -> int:
    value = _required_object(source, attr)
    if not isinstance(value, int):
        raise TypeError(f"Vulkan attribute {attr} must be int, got {type(value).__name__}")
    return int(value)


def _required_int_or_float(source: object, attr: str) -> int | float:
    value = _required_object(source, attr)
    if not isinstance(value, int | float):
        raise TypeError(f"Vulkan attribute {attr} must be int or float, got {type(value).__name__}")
    return value


def _required_bool_or_int(source: object, attr: str) -> bool | int:
    value = _required_object(source, attr)
    if not isinstance(value, bool | int):
        raise TypeError(f"Vulkan attribute {attr} must be bool or int, got {type(value).__name__}")
    return value


def _required_name(source: object, attr: str) -> bytes | str:
    value = _required_object(source, attr)
    if not isinstance(value, bytes | str):
        raise TypeError(f"Vulkan attribute {attr} must be bytes or str, got {type(value).__name__}")
    return value
