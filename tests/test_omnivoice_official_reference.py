"""OmniVoice official reference provider dependency behavior."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from torch2vk.logical import ComparePolicy, LogicalTensor, PyTorchProbe, TensorSpec
from torch2vk.models.omnivoice_safetensor.reference import (
    capture_official_omnivoice_trace,
    omnivoice_official_reference_provider,
)
from torch2vk.pytorch import ArtifactCache
from torch2vk.reference_trace import ReferenceTrace


class OmniVoiceOfficialReferenceTest(unittest.TestCase):
    def test_provider_construction_does_not_import_official_package(self) -> None:
        provider = omnivoice_official_reference_provider()
        self.assertEqual(provider.provider_id, "omnivoice_safetensor.official_generate.v1")

    def test_capture_reports_missing_official_package(self) -> None:
        with (
            self.assertRaisesRegex(
                RuntimeError,
                "Official OmniVoice reference capture requires the `omnivoice` package",
            ),
            patch(
                "torch2vk.models.omnivoice_safetensor.reference.import_module",
                side_effect=ModuleNotFoundError("omnivoice"),
            ),
        ):
            capture_official_omnivoice_trace({"text": "hello", "target_steps": 1})

    def test_provider_maps_final_pcm16_boundary(self) -> None:
        final_wav = LogicalTensor(
            "output.wav_pcm16",
            TensorSpec(dtype="int16", shape=(2,)),
            pytorch_probe=PyTorchProbe(kind="manual", source="output.wav_pcm16"),
            compare=ComparePolicy(kind="tensor"),
        )

        with (
            tempfile.TemporaryDirectory() as cache_dir,
            patch(
                "torch2vk.models.omnivoice_safetensor.reference.capture_official_omnivoice_trace",
                return_value=capture_official_omnivoice_trace_from_tensor(
                    torch.tensor([0, 1], dtype=torch.int16)
                ),
            ),
        ):
            provider = omnivoice_official_reference_provider()
            artifacts = provider.ensure(
                tensors=(final_wav,),
                inputs={"text": "hello", "target_steps": 1},
                cache=ArtifactCache(Path(cache_dir)),
            )

        self.assertTrue(
            artifacts["output.wav_pcm16"].equal(torch.tensor([0, 1], dtype=torch.int16))
        )

    @unittest.skipUnless(
        os.environ.get("TORCH2VK_RUN_OMNIVOICE_OFFICIAL_REFERENCE") == "1",
        "set TORCH2VK_RUN_OMNIVOICE_OFFICIAL_REFERENCE=1 to run official OmniVoice",
    )
    def test_official_capture_smoke(self) -> None:
        trace = capture_official_omnivoice_trace(
            {
                "text": "hello",
                "language": "English",
                "target_steps": 1,
                "num_steps": 1,
                "seed": 20260501,
                "position_temperature": 0.0,
                "denoise": False,
            }
        )

        self.assertIn("output.wav", trace.tensors)
        self.assertIn("output.wav_pcm16", trace.tensors)
        self.assertGreater(trace.tensors["output.wav"].numel(), 0)
        self.assertEqual(trace.tensors["output.wav_pcm16"].dtype, torch.int16)
        self.assertEqual(trace.timeline[0]["boundary"], "output.wav")


def capture_official_omnivoice_trace_from_tensor(value: torch.Tensor) -> ReferenceTrace:
    return ReferenceTrace(tensors={"output.wav_pcm16": value})
