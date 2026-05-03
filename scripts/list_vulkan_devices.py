#!/usr/bin/env python3
"""List Vulkan physical devices visible to torch2vk."""

from __future__ import annotations

from torch2vk.vulkan_backend import create_compute_context, enumerate_physical_devices


def main() -> int:
    devices = enumerate_physical_devices()
    print(f"physical_devices={len(devices)}")
    for device in devices:
        print(f"{device.index}: {device.name} compute_queue_family={device.compute_queue_family}")
    context = create_compute_context()
    try:
        print(
            "compute_context=ok "
            f"device={context.physical_device_name} queue_family={context.compute_queue_family}"
        )
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
