"""OmniVoice exported pipeline integration test."""

from __future__ import annotations

from models.exported_omnivoice.run import main


def test_exported_omnivoice_two_step_pytorch_compare() -> None:
    main(pytorch_compare=True, num_steps=2)
