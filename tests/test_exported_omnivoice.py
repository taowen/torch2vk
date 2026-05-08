"""OmniVoice exported pipeline integration test."""

from __future__ import annotations

from models.exported_omnivoice.run import main


def test_exported_omnivoice_single_step() -> None:
    main(pytorch_compare=True, num_steps=1)
