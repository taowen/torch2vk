"""End-to-end debug attribution examples for Qwen3 and OmniVoice."""

from __future__ import annotations

import unittest

import torch

from torch2vk.e2e_debug import (
    DebugBoundary,
    DrilldownResult,
    E2EDebugTrace,
    boundary_coverage,
    trace_e2e_root_cause,
)
from torch2vk.models.omnivoice_safetensor.tensors.boundaries import omnivoice_debug_boundaries
from torch2vk.models.qwen3_safetensor.tensors.boundaries import qwen3_debug_boundaries


class E2EDebugAttributionTest(unittest.TestCase):
    def test_qwen3_e2e_report_points_to_bad_logits_writer(self) -> None:
        boundaries = qwen3_debug_boundaries(layers=1)
        reference = _trace(
            steps=1,
            values={
                "qwen3.embedding": 1.0,
                "qwen3.layer.00.output": 2.0,
                "qwen3.output.logits": 3.0,
                "qwen3.output.next_token_id": 4,
            },
        )
        candidate = _trace(
            steps=1,
            values={
                "qwen3.embedding": 1.0,
                "qwen3.layer.00.output": 2.0,
                "qwen3.output.logits": 30.0,
                "qwen3.output.next_token_id": 4,
            },
        )

        report = trace_e2e_root_cause(
            boundaries=boundaries,
            steps=1,
            reference=reference,
            candidate=candidate,
            drilldown=lambda _step, boundary: DrilldownResult(
                classification="input_ok_output_bad",
                first_bad_dispatch="linear_bf16_f32",
                evidence={"boundary": boundary.name},
            ),
        )

        self.assertEqual(report.status, "root_cause_found")
        self.assertEqual(report.first_bad_step, 0)
        self.assertEqual(report.first_bad_boundary, "qwen3.output.logits")
        self.assertEqual(report.first_bad_dispatch, "linear_bf16_f32")

    def test_omnivoice_e2e_report_walks_upstream_from_late_token_mismatch(self) -> None:
        boundaries = omnivoice_debug_boundaries(llm_layers=0)
        reference = _trace(
            steps=3,
            tensors={
                "stage0.audio_embedding.output": 1.0,
                "stage0.llm.input.cond": 1.5,
                "stage0.llm.input.uncond": 1.5,
                "stage0.final_norm.input.cond": 1.7,
                "stage0.final_norm.input.uncond": 1.7,
                "stage0.final_norm.cond": 1.8,
                "stage0.final_norm.uncond": 1.8,
                "stage0.audio_head.logits.cond": 2.0,
                "stage0.audio_head.logits.uncond": 2.0,
                "state.selection_scores": 3.0,
            },
            tokens={
                "tokens.before": 0,
                "state.update_mask": 1,
                "tokens.after": 4,
                "generate.final.audio_tokens": 4,
            },
            unscoped_tensors={"stage1.decoder.waveform": 5.0, "output.wav_pcm16": 6.0},
        )
        candidate = _trace(
            steps=3,
            tensors={
                "stage0.audio_embedding.output": 1.0,
                "stage0.llm.input.cond": 1.5,
                "stage0.llm.input.uncond": 1.5,
                "stage0.final_norm.input.cond": 1.7,
                "stage0.final_norm.input.uncond": 1.7,
                "stage0.final_norm.cond": 1.8,
                "stage0.final_norm.uncond": 1.8,
                "stage0.audio_head.logits.cond": 2.0,
                "stage0.audio_head.logits.uncond": 2.0,
                "state.selection_scores": 3.0,
            },
            tokens={
                "tokens.before": 0,
                "state.update_mask": 1,
                "tokens.after": 4,
                "generate.final.audio_tokens": 4,
            },
            unscoped_tensors={"stage1.decoder.waveform": 5.0, "output.wav_pcm16": 6.0},
            token_overrides={
                "step_002.tokens.after": 40,
            },
        )

        def drilldown(_step: int | None, boundary: DebugBoundary) -> DrilldownResult:
            if boundary.name == "tokens.after":
                return DrilldownResult("input_bad_output_bad", None)
            if boundary.name == "state.update_mask":
                return DrilldownResult("input_bad_output_bad", None)
            if boundary.name == "state.selection_scores":
                return DrilldownResult("input_bad_output_bad", None)
            return DrilldownResult(
                "input_ok_output_bad",
                "omnivoice_audio_head_mat_vec_f32_f32",
                {"boundary": boundary.name},
            )

        report = trace_e2e_root_cause(
            boundaries=boundaries,
            steps=3,
            reference=reference,
            candidate=candidate,
            drilldown=drilldown,
        )

        self.assertEqual(report.status, "root_cause_found")
        self.assertEqual(report.first_bad_step, 2)
        self.assertEqual(report.first_bad_boundary, "tokens.after")
        self.assertEqual(
            report.first_bad_dispatch,
            "omnivoice_audio_head_mat_vec_f32_f32",
        )
        self.assertEqual(
            [hop.boundary for hop in report.hops],
            [
                "tokens.after",
                "state.update_mask",
                "state.selection_scores",
                "stage0.audio_head.logits",
            ],
        )
        self.assertEqual(report.hops[-1].classification, "input_ok_output_bad")

    def test_omnivoice_boundary_coverage_reports_missing_reference_artifacts(self) -> None:
        boundaries = omnivoice_debug_boundaries(llm_layers=0)
        trace = E2EDebugTrace(
            tensors={"output.wav_pcm16": torch.tensor([1], dtype=torch.int16)},
            tokens={},
        )

        report = boundary_coverage(boundaries=boundaries, steps=1, trace=trace)

        self.assertFalse(report.ok)
        self.assertNotIn("output.wav_pcm16", report.missing)
        self.assertIn("step_000.stage0.audio_head.logits.cond", report.missing)
        self.assertIn("step_000.stage0.audio_head.logits.uncond", report.missing)
        self.assertIn("generate.final.audio_tokens", report.missing)

    def test_omnivoice_reference_final_boundaries_are_not_coverage_gaps(self) -> None:
        boundaries = omnivoice_debug_boundaries(llm_layers=0)
        trace = E2EDebugTrace(
            tensors={"output.wav_pcm16": torch.tensor([1], dtype=torch.int16)},
            tokens={"generate.final.audio_tokens": torch.tensor([4], dtype=torch.int32)},
        )

        report = boundary_coverage(boundaries=boundaries, steps=1, trace=trace)

        self.assertFalse(report.ok)
        self.assertNotIn("output.wav_pcm16", report.missing)
        self.assertNotIn("generate.final.audio_tokens", report.missing)
        self.assertIn("step_000.stage0.audio_embedding.output", report.missing)


def _trace(
    *,
    steps: int,
    values: dict[str, object] | None = None,
    tensors: dict[str, object] | None = None,
    tokens: dict[str, object] | None = None,
    overrides: dict[str, object] | None = None,
    token_overrides: dict[str, object] | None = None,
    unscoped_tensors: dict[str, object] | None = None,
) -> E2EDebugTrace:
    tensor_artifacts: dict[str, torch.Tensor] = {}
    token_artifacts: dict[str, torch.Tensor] = {}
    tensor_values = tensors if tensors is not None else values or {}
    token_values = tokens or {}
    for step in range(steps):
        for name, value in tensor_values.items():
            tensor_artifacts[f"step_{step:03d}.{name}"] = _tensor(value)
        for name, value in token_values.items():
            if name.startswith("generate.final."):
                continue
            token_artifacts[f"step_{step:03d}.{name}"] = _tensor(value)
    for name, value in (overrides or {}).items():
        tensor_artifacts[name] = _tensor(value)
    for name, value in (token_overrides or {}).items():
        token_artifacts[name] = _tensor(value)
    for name, value in (unscoped_tensors or {}).items():
        tensor_artifacts[name] = _tensor(value)
    if tokens is not None and "generate.final.audio_tokens" in tokens:
        token_artifacts["generate.final.audio_tokens"] = _tensor(
            tokens["generate.final.audio_tokens"]
        )
    return E2EDebugTrace(tensors=tensor_artifacts, tokens=token_artifacts)


def _tensor(value: object) -> torch.Tensor:
    if isinstance(value, int):
        return torch.tensor([value], dtype=torch.int32)
    return torch.tensor([value], dtype=torch.float32)
