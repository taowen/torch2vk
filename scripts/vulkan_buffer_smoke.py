#!/usr/bin/env python3
"""Create and destroy a host-visible Vulkan storage buffer."""

from __future__ import annotations

from torch2vk.vulkan_backend import create_compute_context


def main() -> int:
    context = create_compute_context()
    try:
        buffer = context.create_host_buffer(nbytes=4096)
        try:
            payload = b"torch2vk-buffer-smoke"
            buffer.write(payload)
            readback = buffer.read(nbytes=len(payload))
            if readback != payload:
                raise RuntimeError(f"buffer readback mismatch: {readback!r} != {payload!r}")
            print(
                "host_buffer=ok "
                f"device={context.physical_device_name} "
                f"nbytes={buffer.nbytes} allocation_nbytes={buffer.allocation_nbytes}"
            )
        finally:
            buffer.close()
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
