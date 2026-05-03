"""OmniVoice reference trace probe helpers."""

from __future__ import annotations

from torch2vk.logical import PyTorchProbe


def omnivoice_trace_probe(
    source: str,
    *,
    selector: str | None = None,
    normalize: str = "float32_contiguous",
) -> PyTorchProbe:
    return PyTorchProbe(
        kind="trace",
        source=source,
        selector=selector,
        normalize=normalize,
    )
