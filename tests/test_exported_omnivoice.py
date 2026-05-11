"""OmniVoice exported pipeline integration test."""

from __future__ import annotations

from models.exported_omnivoice.compare import compare_generation_steps


def test_exported_omnivoice_two_step_pytorch_compare() -> None:
    compare_generation_steps(num_steps=2)
