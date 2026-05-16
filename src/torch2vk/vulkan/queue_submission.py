"""Queue submission helpers for Vulkan command buffers."""

from __future__ import annotations

from collections.abc import Callable

from vulkan import (
    VkFenceCreateInfo,
    VkSubmitInfo,
    vkCreateFence,
    vkDestroyFence,
    vkQueueSubmit,
    vkResetFences,
    vkWaitForFences,
)


DEFAULT_WAIT_TIMEOUT_NS = 10_000_000_000


def submit_and_wait(
    *,
    device_handle: object,
    queue_handle: object,
    command_buffer: object,
) -> None:
    fence = submit_with_new_fence(
        device_handle=device_handle,
        queue_handle=queue_handle,
        command_buffer=command_buffer,
    )
    try:
        wait_for_fence(device_handle=device_handle, fence=fence)
    finally:
        vkDestroyFence(device_handle, fence, None)


def submit_and_wait_with_fence(
    *,
    device_handle: object,
    queue_handle: object,
    command_buffer: object,
    fence: object,
) -> None:
    submit_with_existing_fence(
        device_handle=device_handle,
        queue_handle=queue_handle,
        command_buffer=command_buffer,
        fence=fence,
    )
    wait_for_fence(device_handle=device_handle, fence=fence)


def submit_one_shot_and_wait(
    *,
    device_handle: object,
    queue_handle: object,
    command_buffer: object,
    release_command_buffer: Callable[[object], None],
) -> None:
    try:
        submit_and_wait(
            device_handle=device_handle,
            queue_handle=queue_handle,
            command_buffer=command_buffer,
        )
    finally:
        release_command_buffer(command_buffer)


def submit_with_new_fence(
    *,
    device_handle: object,
    queue_handle: object,
    command_buffer: object,
) -> object:
    fence = vkCreateFence(device_handle, VkFenceCreateInfo(), None)
    try:
        submit_info = VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[command_buffer])
        vkQueueSubmit(queue_handle, 1, [submit_info], fence)
    except Exception:
        vkDestroyFence(device_handle, fence, None)
        raise
    return fence


def submit_with_existing_fence(
    *,
    device_handle: object,
    queue_handle: object,
    command_buffer: object,
    fence: object,
) -> None:
    vkResetFences(device_handle, 1, [fence])
    submit_info = VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[command_buffer])
    vkQueueSubmit(queue_handle, 1, [submit_info], fence)


def wait_for_fence(
    *,
    device_handle: object,
    fence: object,
) -> None:
    vkWaitForFences(device_handle, 1, [fence], True, DEFAULT_WAIT_TIMEOUT_NS)
