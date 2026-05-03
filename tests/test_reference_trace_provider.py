"""ReferenceTrace provider maps external traces back to LogicalTensor names."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from torch2vk.logical import ComparePolicy, LogicalTensor, PyTorchProbe, TensorSpec
from torch2vk.models.omnivoice_safetensor.tensors.probes import omnivoice_trace_probe
from torch2vk.pytorch import ArtifactCache
from torch2vk.reference_trace import ReferenceTrace, TraceReferenceProvider


class ReferenceTraceProviderTest(unittest.TestCase):
    def test_omnivoice_trace_probe_is_a_first_class_probe_kind(self) -> None:
        probe = omnivoice_trace_probe("stage0.audio_head.logits", selector="row")

        self.assertEqual(probe.kind, "trace")
        self.assertEqual(probe.source, "stage0.audio_head.logits")
        self.assertEqual(probe.selector, "row")

    def test_uses_probe_source_and_cache_for_required_tensors_only(self) -> None:
        calls: list[Mapping[str, Any]] = []

        def capture(inputs: Mapping[str, Any]) -> ReferenceTrace:
            calls.append(dict(inputs))
            return ReferenceTrace(
                tensors={
                    "official.hidden": torch.tensor([1.0], dtype=torch.float32),
                    "unused": torch.tensor([99.0], dtype=torch.float32),
                },
                tokens={"official.token": torch.tensor([7], dtype=torch.int32)},
            )

        hidden = _tensor("logical.hidden", source="official.hidden", kind="tensor")
        token = _tensor("logical.token", source="official.token", kind="token")
        ignored = LogicalTensor("ignored", TensorSpec(dtype="float32", shape=(1,)))
        provider = TraceReferenceProvider(capture=capture, provider_id="unit.trace")

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = ArtifactCache(Path(cache_dir))
            first = provider.ensure(
                tensors=(hidden, token, ignored),
                inputs={"text": "hello"},
                cache=cache,
            )
            second = provider.ensure(
                tensors=(hidden, token, ignored),
                inputs={"text": "hello"},
                cache=cache,
            )

        self.assertEqual(calls, [{"text": "hello"}])
        self.assertEqual(set(first), {"logical.hidden", "logical.token"})
        self.assertTrue(first["logical.hidden"].equal(torch.tensor([1.0])))
        self.assertTrue(first["logical.token"].equal(torch.tensor([7], dtype=torch.int32)))
        self.assertTrue(second["logical.hidden"].equal(first["logical.hidden"]))

    def test_maps_scoped_trace_sources_to_scoped_logical_tensor_keys(self) -> None:
        calls = 0

        def capture(_inputs: Mapping[str, Any]) -> ReferenceTrace:
            nonlocal calls
            calls += 1
            return ReferenceTrace(
                tensors={
                    "generate/step=0.stage0.audio_head.logits": torch.tensor(
                        [1.0],
                        dtype=torch.float32,
                    ),
                    "generate/step=1.stage0.audio_head.logits": torch.tensor(
                        [2.0],
                        dtype=torch.float32,
                    ),
                }
            )

        logits = _tensor(
            "stage0.audio_head.logits",
            source="stage0.audio_head.logits",
            kind="tensor",
        )
        provider = TraceReferenceProvider(capture=capture, provider_id="unit.scoped")

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = ArtifactCache(Path(cache_dir))
            artifacts = provider.ensure(
                tensors=(logits,),
                inputs={},
                cache=cache,
            )
            cached = provider.ensure(
                tensors=(logits,),
                inputs={},
                cache=cache,
            )

        self.assertEqual(calls, 1)
        self.assertEqual(
            set(artifacts),
            {
                "generate/step=0.stage0.audio_head.logits",
                "generate/step=1.stage0.audio_head.logits",
            },
        )
        self.assertTrue(
            artifacts["generate/step=1.stage0.audio_head.logits"].equal(torch.tensor([2.0]))
        )
        self.assertEqual(set(cached), set(artifacts))


def _tensor(name: str, *, source: str, kind: str) -> LogicalTensor:
    return LogicalTensor(
        name,
        TensorSpec(dtype="int32" if kind == "token" else "float32", shape=(1,)),
        pytorch_probe=PyTorchProbe(kind="manual", source=source),
        compare=ComparePolicy(kind=kind),
    )
