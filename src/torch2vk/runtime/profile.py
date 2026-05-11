"""Runtime dispatch/replay profiling and SQTT attribution labels."""

from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch2vk.runtime.shader import DispatchRecord

if TYPE_CHECKING:
    from torch2vk.runtime.replay import ReplayPlan
    from torch2vk.vulkan.compute_pipeline import ComputePipeline
    from torch2vk.vulkan.device import VulkanDevice


class RuntimeProfiler:
    """Write runtime dispatch manifests and lightweight performance summaries."""

    def __init__(self, root: str | Path | None = None) -> None:
        resolved = root if root is not None else os.environ.get("TORCH2VK_PROFILE_RUN_DIR")
        self.root = None if resolved is None else Path(resolved).expanduser().resolve()
        self.enabled = self.root is not None
        self.sqtt_labels_enabled = os.environ.get("TORCH2VK_SQTT_ROOT") is not None
        self._dispatch_rows: list[dict[str, Any]] = []
        self._replay_plan_samples: dict[str, list[int]] = defaultdict(list)
        self._replay_dispatch_samples: dict[tuple[str, int], list[int]] = defaultdict(list)
        self._device_info: dict[str, Any] = {}
        self._closed = False
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "dispatches.jsonl").write_text("", encoding="utf-8")
            self._write_run_file()

    @classmethod
    def disabled(cls) -> "RuntimeProfiler":
        return cls(None)

    def attach_device(self, device: "VulkanDevice") -> None:
        if not self.enabled:
            return
        self._device_info = {
            "timestamp_period_ns": device.timestamp_period_ns,
        }
        self._write_run_file()

    def record_dispatch(
        self,
        *,
        record: DispatchRecord,
        pipeline: "ComputePipeline",
        elapsed_wall_ns: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        row = _dispatch_record_row(
            record=record,
            pipeline=pipeline,
            phase="record",
            elapsed_wall_ns=elapsed_wall_ns,
        )
        self._attach_profile_shader_source(row, pipeline)
        self._append_dispatch_row(row)

    def sqtt_label(
        self,
        *,
        frame: str,
        shader: str,
        dispatch_index: int,
    ) -> str | None:
        if not self.sqtt_labels_enabled:
            return None
        payload = f"frame={frame};shader={shader};dispatch={dispatch_index}"
        return f"agentorch-profile-submit:{payload}"

    def record_replay_execution(
        self,
        *,
        plan: "ReplayPlan",
        timestamps: Sequence[int],
    ) -> None:
        if not self.enabled or plan.profile_state is None:
            return
        query_count = plan.profile_state.query_count
        if len(timestamps) != query_count:
            return
        period = plan.device.timestamp_period_ns
        plan_elapsed_ns = _elapsed_ns(timestamps[0], timestamps[-1], period)
        self._replay_plan_samples[plan.name].append(plan_elapsed_ns)

        for i, entry in enumerate(plan.dispatch_entries):
            begin = timestamps[1 + i * 2]
            end = timestamps[2 + i * 2]
            elapsed_ns = _elapsed_ns(begin, end, period)
            self._replay_dispatch_samples[(plan.name, i)].append(elapsed_ns)
            row = _replay_dispatch_row(
                plan_name=plan.name,
                replay_dispatch_index=i,
                entry=entry,
                elapsed_ns=elapsed_ns,
                sample_index=len(self._replay_dispatch_samples[(plan.name, i)]) - 1,
            )
            self._attach_profile_shader_source(row, entry.pipeline)
            self._append_dispatch_row(row)

    def summary(self) -> dict[str, Any]:
        record_rows = [row for row in self._dispatch_rows if row.get("phase") == "record"]
        replay_rows = [row for row in self._dispatch_rows if row.get("phase") == "replay"]
        summary: dict[str, Any] = {
            "dispatch_rows": len(self._dispatch_rows),
            "record": _record_summary(record_rows),
            "replay": _replay_summary(
                replay_rows=replay_rows,
                plan_samples=self._replay_plan_samples,
                dispatch_samples=self._replay_dispatch_samples,
            ),
        }
        if self._device_info:
            summary["device"] = self._device_info
        return summary

    def write_summary(self) -> dict[str, Any]:
        summary = self.summary()
        if self.root is not None:
            (self.root / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return summary

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.enabled:
            self.write_summary()

    def _append_dispatch_row(self, row: dict[str, Any]) -> None:
        self._dispatch_rows.append(row)
        if self.root is not None:
            with (self.root / "dispatches.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True))
                f.write("\n")

    def _write_run_file(self) -> None:
        if self.root is None:
            return
        payload = {
            "schema_version": 1,
            "created_unix_ns": time.time_ns(),
            "dispatches_path": "dispatches.jsonl",
            "summary_path": "summary.json",
        }
        if self._device_info:
            payload["device"] = self._device_info
        (self.root / "run.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _attach_profile_shader_source(self, row: dict[str, Any], pipeline: "ComputePipeline") -> None:
        if self.root is None:
            return
        source_path = _shader_source_path(pipeline.shader_spv_path)
        if source_path is None:
            return
        source_dir = self.root / "shaders"
        source_dir.mkdir(parents=True, exist_ok=True)
        destination = source_dir / source_path.name
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
        row["shader_glsl_path"] = str(destination)


def _dispatch_record_row(
    *,
    record: DispatchRecord,
    pipeline: "ComputePipeline",
    phase: str,
    elapsed_wall_ns: int | None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "timing_kind": "eager_dispatch_wall_ns",
        "dispatch_index": record.index,
        "frame": record.frame,
        "shader": record.shader,
        "pipeline_debug_name": pipeline.debug_name,
        "pipeline_identity_sha256": pipeline.pipeline_identity_sha256,
        "shader_spv_sha256": pipeline.shader_spv_sha256,
        "shader_spv_path": str(pipeline.shader_spv_path),
        "symbols": dict(record.symbols),
        "dispatch_size": list(record.dispatch_size),
        "push_constants": dict(record.push_constant_values),
        "reads": [
            {"field": field, "tensor": tensor}
            for field, tensor in record.logical_reads
        ],
        "writes": [
            {"field": field, "tensor": tensor}
            for field, tensor in record.logical_writes
        ],
        "descriptor_views": [
            {"field": field, "binding_index": binding, "offset": offset, "nbytes": nbytes}
            for field, binding, offset, nbytes in record.descriptor_views
        ],
        "tensors": [asdict(snapshot) for snapshot in record.tensor_snapshots],
        "elapsed_ns": None,
        "elapsed_wall_ns": elapsed_wall_ns,
    }


def _shader_source_path(shader_spv_path: Path) -> Path | None:
    for suffix in (".comp", ".glsl"):
        candidate = shader_spv_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _replay_dispatch_row(
    *,
    plan_name: str,
    replay_dispatch_index: int,
    entry: Any,
    elapsed_ns: int,
    sample_index: int,
) -> dict[str, Any]:
    return {
        "phase": "replay",
        "timing_kind": "gpu_timestamp_ns",
        "replay_plan": plan_name,
        "replay_dispatch_index": replay_dispatch_index,
        "source_dispatch_index": entry.source_dispatch_index,
        "frame": entry.source_frame,
        "shader": entry.source_shader,
        "pipeline_debug_name": entry.pipeline.debug_name,
        "pipeline_identity_sha256": entry.pipeline.pipeline_identity_sha256,
        "shader_spv_sha256": entry.pipeline.shader_spv_sha256,
        "shader_spv_path": str(entry.pipeline.shader_spv_path),
        "symbols": dict(entry.symbols),
        "dispatch_size": list(entry.dispatch_size),
        "reads": [
            {"field": field, "tensor": tensor}
            for field, tensor in entry.source_logical_reads
        ],
        "writes": [
            {"field": field, "tensor": tensor}
            for field, tensor in entry.source_logical_writes
        ],
        "output_op": _output_op_name(entry.source_logical_writes),
        "elapsed_ns": elapsed_ns,
        "sample_index": sample_index,
    }


def _elapsed_ns(begin: int, end: int, timestamp_period_ns: float) -> int:
    return int(max(0, end - begin) * timestamp_period_ns)


def _record_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wall_values = [
        int(row["elapsed_wall_ns"])
        for row in rows
        if row.get("elapsed_wall_ns") is not None
    ]
    return {
        "timing_kind": "eager_dispatch_wall_ns",
        "dispatch_count": len(rows),
        "wall_elapsed_ns_total": sum(wall_values),
        "top_shaders_by_wall_ns": _top_groups(rows, key_field="shader", value_field="elapsed_wall_ns"),
        "top_frames_by_wall_ns": _top_groups(rows, key_field="frame", value_field="elapsed_wall_ns"),
    }


def _replay_summary(
    *,
    replay_rows: Sequence[dict[str, Any]],
    plan_samples: dict[str, list[int]],
    dispatch_samples: dict[tuple[str, int], list[int]],
) -> dict[str, Any]:
    plans = {
        name: _stats(values)
        for name, values in sorted(plan_samples.items())
    }
    top_dispatches = []
    shader_groups: dict[str, list[int]] = defaultdict(list)
    output_groups: dict[str, list[int]] = defaultdict(list)
    for (plan_name, dispatch_index), values in sorted(dispatch_samples.items()):
        matching_row = next(
            (
                row
                for row in replay_rows
                if row.get("replay_plan") == plan_name
                and row.get("replay_dispatch_index") == dispatch_index
            ),
            None,
        )
        entry = {
            "replay_plan": plan_name,
            "replay_dispatch_index": dispatch_index,
            **_stats(values),
        }
        median_ns = int(entry["median_ns"])
        if matching_row is not None:
            entry["shader"] = matching_row.get("shader")
            entry["frame"] = matching_row.get("frame")
            entry["source_dispatch_index"] = matching_row.get("source_dispatch_index")
            shader_groups[str(matching_row.get("shader", ""))].append(median_ns)
            output_groups[str(matching_row.get("output_op", ""))].append(median_ns)
        top_dispatches.append(entry)
    top_dispatches.sort(key=lambda entry: int(entry.get("median_ns", 0)), reverse=True)
    return {
        "timing_kind": "gpu_timestamp_ns",
        "dispatch_samples": len(replay_rows),
        "plans": plans,
        "top_shaders_by_median_ns": _top_median_groups(shader_groups),
        "top_outputs_by_median_ns": _top_median_groups(output_groups),
        "top_dispatches_by_median_ns": top_dispatches[:20],
    }


def _output_op_name(logical_writes: Sequence[tuple[str, str]]) -> str:
    if not logical_writes:
        return "<no-write>"
    _field, tensor = logical_writes[0]
    marker = "text_decode.layers."
    if marker in tensor:
        suffix = tensor.split(marker, 1)[1]
        parts = suffix.split(".", 1)
        if len(parts) == 2:
            return parts[1]
    marker = "text_decode."
    if marker in tensor:
        return tensor.split(marker, 1)[1]
    marker = "token_select."
    if marker in tensor:
        return f"token_select.{tensor.split(marker, 1)[1]}"
    marker = "token_store"
    if marker in tensor:
        return marker
    return tensor


def _top_median_groups(
    grouped: dict[str, list[int]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    result = [
        {
            "name": name,
            "dispatch_count": len(values),
            "median_ns_total": sum(values),
            "median_ns_max": max(values) if values else 0,
        }
        for name, values in grouped.items()
    ]
    result.sort(key=lambda entry: int(entry.get("median_ns_total", 0)), reverse=True)
    return result[:limit]


def _top_groups(
    rows: Sequence[dict[str, Any]],
    *,
    key_field: str,
    value_field: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        grouped[str(row.get(key_field, ""))].append(int(value))
    result = [
        {"name": name, **_stats(values)}
        for name, values in grouped.items()
    ]
    result.sort(key=lambda entry: int(entry.get("total_ns", 0)), reverse=True)
    return result[:limit]


def _stats(values: Iterable[int]) -> dict[str, int | float]:
    samples = sorted(int(value) for value in values)
    if not samples:
        return {"count": 0, "total_ns": 0, "min_ns": 0, "median_ns": 0, "p95_ns": 0}
    return {
        "count": len(samples),
        "total_ns": sum(samples),
        "min_ns": samples[0],
        "median_ns": int(statistics.median(samples)),
        "p95_ns": _percentile(samples, 0.95),
    }


def _percentile(sorted_samples: Sequence[int], q: float) -> int:
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    index = round((len(sorted_samples) - 1) * q)
    return sorted_samples[min(max(index, 0), len(sorted_samples) - 1)]
