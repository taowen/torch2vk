"""Record-first reference capture behavior for DebugContext."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from torch2vk.debug_context import DebugContext
from torch2vk.logical import ComparePolicy, LogicalTensor, PyTorchProbe, TensorSpec
from torch2vk.pytorch import ArtifactCache, TransformFn
from torch2vk.shader import DispatchRecord


class DebugContextReferenceOrderTest(unittest.TestCase):
    def test_scope_is_recorded_on_dispatch_timeline(self) -> None:
        hidden = _comparable_tensor("layer.hidden")
        with tempfile.TemporaryDirectory() as cache_dir:
            ctx = DebugContext(
                shader_dir=Path(),
                variants={},
                context=object(),
                tensors={hidden.name: hidden},
                tensor_sequence=(hidden,),
                allocations={},
                inputs={},
                cache=ArtifactCache(Path(cache_dir)),
            )
            with ctx.scope("generate", step=5), ctx.scope("stage0.audio_head", row="cond"):
                record = _record(
                    0,
                    "shader_hidden",
                    {"output": hidden.name},
                    scope=ctx.current_scope,
                )

        self.assertEqual(record.scope, "generate/step=5.stage0.audio_head/row=cond")
        self.assertEqual(
            record.artifact_key(hidden.name),
            "generate/step=5.stage0.audio_head/row=cond.layer.hidden",
        )

    def test_reference_provider_runs_after_records_define_required_tensors(self) -> None:
        hidden = _comparable_tensor("layer.hidden")
        logits = _comparable_tensor("output.logits")
        provider = _RecordingReferenceProvider(
            artifacts={
                hidden.name: torch.tensor([1.0], dtype=torch.float32),
                logits.name: torch.tensor([2.0], dtype=torch.float32),
            }
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            ctx = DebugContext(
                shader_dir=Path(),
                variants={},
                context=object(),
                tensors={hidden.name: hidden, logits.name: logits},
                tensor_sequence=(hidden, logits),
                allocations={},
                inputs={"input_ids": torch.tensor([1], dtype=torch.int64)},
                cache=ArtifactCache(Path(cache_dir)),
            )
            ctx.records.extend(
                (
                    _record(0, "shader_hidden", {"output": hidden.name}),
                    _record(1, "shader_logits", {"output": logits.name}),
                )
            )
            ctx.candidate[hidden.name] = torch.tensor([1.0], dtype=torch.float32)
            ctx.candidate[logits.name] = torch.tensor([3.0], dtype=torch.float32)

            with self.assertRaisesRegex(
                AssertionError,
                "first mismatch: output.logits\nwriter shader: shader_logits\nwriter dispatch: 1",
            ):
                ctx.compare_records(provider)

        self.assertEqual(provider.calls, ((hidden.name, logits.name),))

    def test_compare_prefers_scoped_artifacts_when_record_has_scope(self) -> None:
        hidden = _comparable_tensor("layer.hidden")
        scoped_key = "generate/step=5.layer.hidden"
        provider = _RecordingReferenceProvider(
            artifacts={
                hidden.name: torch.tensor([1.0], dtype=torch.float32),
                scoped_key: torch.tensor([2.0], dtype=torch.float32),
            }
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            ctx = DebugContext(
                shader_dir=Path(),
                variants={},
                context=object(),
                tensors={hidden.name: hidden},
                tensor_sequence=(hidden,),
                allocations={},
                inputs={},
                cache=ArtifactCache(Path(cache_dir)),
            )
            ctx.records.append(
                _record(0, "shader_hidden", {"output": hidden.name}, scope="generate/step=5")
            )
            ctx.candidate[scoped_key] = torch.tensor([3.0], dtype=torch.float32)

            with self.assertRaisesRegex(
                AssertionError,
                "first mismatch: generate/step=5.layer.hidden",
            ):
                ctx.compare_records(provider)


class _RecordingReferenceProvider:
    def __init__(self, artifacts: Mapping[str, torch.Tensor]) -> None:
        self._artifacts = dict(artifacts)
        self.calls: tuple[tuple[str, ...], ...] = ()

    def ensure(
        self,
        *,
        tensors: tuple[LogicalTensor, ...],
        inputs: Mapping[str, Any],
        cache: ArtifactCache,
        transforms: Mapping[str, TransformFn] | None = None,
        extra_fingerprint: Mapping[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        del inputs, cache, transforms, extra_fingerprint
        names = tuple(tensor.name for tensor in tensors)
        self.calls = (*self.calls, names)
        artifacts = {name: self._artifacts[name] for name in names}
        artifacts.update(
            {
                name: value
                for name, value in self._artifacts.items()
                if name not in artifacts and "." in name
            }
        )
        return artifacts


def _comparable_tensor(name: str) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype="f32", shape=(1,)),
        pytorch_probe=PyTorchProbe(kind="manual"),
        compare=ComparePolicy(kind="tensor"),
    )


def _record(
    index: int,
    shader: str,
    writes: Mapping[str, str],
    *,
    scope: str = "",
) -> DispatchRecord:
    return DispatchRecord(
        index=index,
        shader=shader,
        family="test",
        reads={},
        writes=writes,
        symbols={},
        uniforms={},
        push_constant_size=None,
        push_constants=None,
        scope=scope,
    )
