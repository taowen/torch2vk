"""Minimal Vulkan debug-utils support for profiler attribution and command labels."""

from __future__ import annotations

from dataclasses import dataclass

from vulkan import VkDebugUtilsLabelEXT, VkDebugUtilsObjectNameInfoEXT, vkGetInstanceProcAddr
from _cffi_backend import _CDataBase
from vulkan._vulkan import ExtensionNotSupportedError, ProcedureNotFoundError, ffi

from .abi import (
    BeginDebugLabelProc,
    EndDebugLabelProc,
    SetDebugObjectNameProc,
    begin_debug_label_proc,
    end_debug_label_proc,
    set_debug_object_name_proc,
)


@dataclass(frozen=True, slots=True)
class DebugUtils:
    _set_object_name: SetDebugObjectNameProc | None
    _cmd_begin_label: BeginDebugLabelProc | None
    _cmd_end_label: EndDebugLabelProc | None

    @property
    def enabled(self) -> bool:
        return self._set_object_name is not None or self.command_labels_enabled

    @property
    def command_labels_enabled(self) -> bool:
        return self._cmd_begin_label is not None and self._cmd_end_label is not None

    def set_object_name(self, *, device: object, object_type: int, handle: int | _CDataBase, name: str) -> None:
        if self._set_object_name is None or not name:
            return
        object_handle = _coerce_handle_to_uint64(handle)
        self._set_object_name(
            device,
            VkDebugUtilsObjectNameInfoEXT(
                objectType=object_type,
                objectHandle=object_handle,
                pObjectName=name,
            ),
        )

    def begin_command_label(self, *, command_buffer: object, name: str) -> None:
        if self._cmd_begin_label is None or not name:
            return
        self._cmd_begin_label(
            command_buffer,
            VkDebugUtilsLabelEXT(
                pLabelName=name,
                color=(0.0, 0.0, 0.0, 0.0),
            ),
        )

    def end_command_label(self, *, command_buffer: object) -> None:
        if self._cmd_end_label is None:
            return
        self._cmd_end_label(command_buffer)


def create_debug_utils(*, instance: object, device: object) -> DebugUtils:
    del device
    try:
        set_object_name = set_debug_object_name_proc(vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT"))
    except (ProcedureNotFoundError, ExtensionNotSupportedError):
        set_object_name = None
    try:
        cmd_begin_label = begin_debug_label_proc(vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"))
        cmd_end_label = end_debug_label_proc(vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"))
    except (ProcedureNotFoundError, ExtensionNotSupportedError):
        cmd_begin_label = None
        cmd_end_label = None
    return DebugUtils(
        _set_object_name=set_object_name,
        _cmd_begin_label=cmd_begin_label,
        _cmd_end_label=cmd_end_label,
    )


def _coerce_handle_to_uint64(handle: int | _CDataBase) -> int:
    if isinstance(handle, int):
        return handle
    return int(ffi.cast("uintptr_t", handle))
