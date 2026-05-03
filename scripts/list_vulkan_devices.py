#!/usr/bin/env python3
"""List Vulkan physical devices visible to torch2vk."""

from __future__ import annotations

from torch2vk.vulkan_backend import enumerate_physical_devices


def main() -> int:
    devices = enumerate_physical_devices()
    print(f"physical_devices={len(devices)}")
    for device in devices:
        print(f"{device.index}: {device.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
