"""Build SPM counter and roofline summaries from RGP captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .postprocess_common import expect_int, expect_str, relative_or_str
from .rgp_capture_parser import (
    RgpAsicInfo,
    RgpDerivedSpmCounter,
    parse_rgp_capture,
)


def build_spm_counter_reports(*, root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    rgp_rows = sorted(
        rows,
        key=lambda row: (expect_int(row, "submit_ordinal"), expect_str(row, "rgp_path")),
    )
    for row in rgp_rows:
        submit_ordinal = expect_int(row, "submit_ordinal")
        rgp_path_text = expect_str(row, "rgp_path")
        key = (submit_ordinal, rgp_path_text)
        if key in seen:
            continue
        seen.add(key)
        rgp_path = Path(rgp_path_text)
        if not rgp_path.is_absolute():
            rgp_path = root / rgp_path
        try:
            capture = parse_rgp_capture(rgp_path)
            if not capture.derived_spm_chunks:
                reports.append({
                    "submit_ordinal": submit_ordinal,
                    "capture_path": relative_or_str(rgp_path, root),
                    "availability": "absent",
                    "unavailable_reason": "RGP capture has no DERIVED_SPM_DB chunk",
                })
                continue
            spm = capture.derived_spm_chunks[0]
            asic = capture.asic_info
            counter_by_name = {counter.name: counter for counter in spm.counters}
            fetch_size_bytes = _counter_sum(counter_by_name.get("Fetch size"))
            write_size_bytes = _counter_sum(counter_by_name.get("Write size"))
            total_memory_bytes = fetch_size_bytes + write_size_bytes
            capture_span_ns = _spm_capture_span_ns(asic, spm.timestamps)
            memory_busy_span_ns = _spm_counter_active_span_ns(
                asic,
                spm.timestamps,
                counter_by_name.get("Memory unity busy"),
            )
            capture_bandwidth_gb_s = _bandwidth_gb_s(
                total_memory_bytes,
                capture_span_ns,
            )
            memory_busy_bandwidth_gb_s = _bandwidth_gb_s(
                total_memory_bytes,
                memory_busy_span_ns,
            )
            trace_peak_bandwidth_gb_s = _memory_peak_bandwidth_gb_s(
                asic,
                None if asic is None else asic.trace_memory_clock_hz,
            )
            max_peak_bandwidth_gb_s = _memory_peak_bandwidth_gb_s(
                asic,
                None if asic is None else asic.max_memory_clock_hz,
            )
            compute_roofline = _compute_roofline_estimate(
                row=row,
                asic=asic,
                capture_span_ns=capture_span_ns,
                memory_busy_span_ns=memory_busy_span_ns,
                total_memory_bytes=total_memory_bytes,
                trace_peak_bandwidth_gb_s=trace_peak_bandwidth_gb_s,
                max_peak_bandwidth_gb_s=max_peak_bandwidth_gb_s,
            )
            reports.append({
                "submit_ordinal": submit_ordinal,
                "capture_path": relative_or_str(rgp_path, root),
                "dispatch_index": row.get("dispatch_index"),
                "frame": row.get("frame"),
                "shader": row.get("shader"),
                "output_op": row.get("output_op"),
                "symbols": row.get("symbols"),
                "availability": "present",
                "sample_count": spm.sample_count,
                "sample_interval": spm.sample_interval,
                "time": {
                    "spm_capture_span_ns": capture_span_ns,
                    "memory_busy_span_ns": memory_busy_span_ns,
                },
                "asic": _asic_summary(asic),
                "memory": {
                    "fetch_size_bytes": fetch_size_bytes,
                    "write_size_bytes": write_size_bytes,
                    "total_size_bytes": total_memory_bytes,
                    "local_video_memory_bytes": _counter_sum(
                        counter_by_name.get("Local video memory bytes")
                    ),
                    "pcie_bytes": _counter_sum(counter_by_name.get("PCIe bytes")),
                },
                "cache": {
                    "l2_cache_hit_avg_percent": _counter_avg(counter_by_name.get("L2 cache hit")),
                    "l2_cache_hit_max_percent": _counter_max(counter_by_name.get("L2 cache hit")),
                },
                "memory_unit": {
                    "busy_avg_percent": _counter_avg(counter_by_name.get("Memory unity busy")),
                    "busy_max_percent": _counter_max(counter_by_name.get("Memory unity busy")),
                    "stalled_avg_percent": _counter_avg(counter_by_name.get("Memory unit stalled")),
                    "stalled_max_percent": _counter_max(counter_by_name.get("Memory unit stalled")),
                },
                "roofline": {
                    "capture_bandwidth_gb_s": capture_bandwidth_gb_s,
                    "memory_busy_bandwidth_gb_s": memory_busy_bandwidth_gb_s,
                    "trace_peak_bandwidth_gb_s": trace_peak_bandwidth_gb_s,
                    "max_peak_bandwidth_gb_s": max_peak_bandwidth_gb_s,
                    "capture_trace_peak_ratio": _ratio_or_none(
                        capture_bandwidth_gb_s,
                        trace_peak_bandwidth_gb_s,
                    ),
                    "capture_max_peak_ratio": _ratio_or_none(
                        capture_bandwidth_gb_s,
                        max_peak_bandwidth_gb_s,
                    ),
                    "memory_busy_trace_peak_ratio": _ratio_or_none(
                        memory_busy_bandwidth_gb_s,
                        trace_peak_bandwidth_gb_s,
                    ),
                    "memory_busy_max_peak_ratio": _ratio_or_none(
                        memory_busy_bandwidth_gb_s,
                        max_peak_bandwidth_gb_s,
                    ),
                },
                "compute_roofline": compute_roofline,
                "counters": [
                    _counter_summary(counter)
                    for counter in spm.counters
                ],
            })
        except Exception as exc:
            reports.append({
                "submit_ordinal": submit_ordinal,
                "capture_path": relative_or_str(rgp_path, root),
                "availability": "error",
                "unavailable_reason": str(exc),
            })
    return reports


def _asic_summary(asic: object) -> dict[str, Any] | None:
    if not isinstance(asic, RgpAsicInfo):
        return None
    return {
        "gpu_name": asic.gpu_name,
        "gpu_type": asic.gpu_type,
        "gfxip_level": asic.gfxip_level,
        "shader_engines": asic.shader_engines,
        "compute_units": asic.shader_engines * asic.compute_unit_per_shader_engine,
        "simd_per_compute_unit": asic.simd_per_compute_unit,
        "vram_size_bytes": asic.vram_size_bytes,
        "vram_bus_width_bits": asic.vram_bus_width_bits,
        "l2_cache_size_bytes": asic.l2_cache_size_bytes,
        "l1_cache_size_bytes": asic.l1_cache_size_bytes,
        "gl1_cache_size_bytes": asic.gl1_cache_size_bytes,
        "instruction_cache_size_bytes": asic.instruction_cache_size_bytes,
        "scalar_cache_size_bytes": asic.scalar_cache_size_bytes,
        "mall_cache_size_bytes": asic.mall_cache_size_bytes,
        "trace_shader_core_clock_hz": asic.trace_shader_core_clock_hz,
        "trace_memory_clock_hz": asic.trace_memory_clock_hz,
        "max_shader_core_clock_hz": asic.max_shader_core_clock_hz,
        "max_memory_clock_hz": asic.max_memory_clock_hz,
        "gpu_timestamp_frequency_hz": asic.gpu_timestamp_frequency_hz,
        "memory_ops_per_clock": asic.memory_ops_per_clock,
        "memory_chip_type": asic.memory_chip_type,
    }


def _counter_summary(counter: object) -> dict[str, Any]:
    if not isinstance(counter, RgpDerivedSpmCounter):
        raise TypeError(f"counter must be RgpDerivedSpmCounter, got {type(counter).__name__}")
    return {
        "name": counter.name,
        "group": counter.group_name,
        "usage_type": counter.usage_type,
        "sample_count": len(counter.values),
        "sum": _counter_sum(counter),
        "avg": _counter_avg(counter),
        "min": _counter_min(counter),
        "max": _counter_max(counter),
        "nonzero_samples": sum(1 for value in counter.values if value != 0.0),
    }


def _counter_sum(counter: object) -> float:
    if not isinstance(counter, RgpDerivedSpmCounter):
        return 0.0
    return float(sum(counter.values))


def _counter_avg(counter: object) -> float:
    if not isinstance(counter, RgpDerivedSpmCounter) or not counter.values:
        return 0.0
    return float(sum(counter.values) / len(counter.values))


def _counter_min(counter: object) -> float:
    if not isinstance(counter, RgpDerivedSpmCounter) or not counter.values:
        return 0.0
    return float(min(counter.values))


def _counter_max(counter: object) -> float:
    if not isinstance(counter, RgpDerivedSpmCounter) or not counter.values:
        return 0.0
    return float(max(counter.values))


def _spm_capture_span_ns(asic: object, timestamps: tuple[int, ...]) -> float | None:
    if not isinstance(asic, RgpAsicInfo):
        return None
    if len(timestamps) < 2 or asic.gpu_timestamp_frequency_hz <= 0:
        return None
    return (
        float(timestamps[-1] - timestamps[0])
        * 1_000_000_000.0
        / float(asic.gpu_timestamp_frequency_hz)
    )


def _spm_counter_active_span_ns(
    asic: object,
    timestamps: tuple[int, ...],
    counter: object,
) -> float | None:
    if not isinstance(asic, RgpAsicInfo) or not isinstance(counter, RgpDerivedSpmCounter):
        return None
    if len(timestamps) < 2 or len(counter.values) != len(timestamps):
        return None
    active_indices = [index for index, value in enumerate(counter.values) if value > 0.0]
    if not active_indices or asic.gpu_timestamp_frequency_hz <= 0:
        return None
    first = active_indices[0]
    last = active_indices[-1]
    end = last + 1 if last + 1 < len(timestamps) else last
    if end <= first:
        return None
    return (
        float(timestamps[end] - timestamps[first])
        * 1_000_000_000.0
        / float(asic.gpu_timestamp_frequency_hz)
    )


def _bandwidth_gb_s(total_bytes: float, duration_ns: float | None) -> float | None:
    if duration_ns is None or duration_ns <= 0.0:
        return None
    return total_bytes / (duration_ns / 1_000_000_000.0) / 1_000_000_000.0


def _memory_peak_bandwidth_gb_s(asic: object, clock_hz: int | None) -> float | None:
    if not isinstance(asic, RgpAsicInfo) or clock_hz is None:
        return None
    if clock_hz <= 0 or asic.memory_ops_per_clock <= 0 or asic.vram_bus_width_bits <= 0:
        return None
    bytes_per_clock = float(asic.memory_ops_per_clock) * float(asic.vram_bus_width_bits) / 8.0
    return float(clock_hz) * bytes_per_clock / 1_000_000_000.0


def _compute_roofline_estimate(
    *,
    row: dict[str, Any],
    asic: object,
    capture_span_ns: float | None,
    memory_busy_span_ns: float | None,
    total_memory_bytes: float,
    trace_peak_bandwidth_gb_s: float | None,
    max_peak_bandwidth_gb_s: float | None,
) -> dict[str, Any]:
    op_estimate = _algorithmic_flop_estimate(row)
    trace_peak_tflops_s = _fp32_fma_peak_tflops_s(
        asic,
        None if not isinstance(asic, RgpAsicInfo) else asic.trace_shader_core_clock_hz,
    )
    max_peak_tflops_s = _fp32_fma_peak_tflops_s(
        asic,
        None if not isinstance(asic, RgpAsicInfo) else asic.max_shader_core_clock_hz,
    )
    if op_estimate is None:
        return {
            "availability": "unavailable",
            "unavailable_reason": "no algorithmic FLOP estimator for this shader",
            "trace_peak_tflops_s": trace_peak_tflops_s,
            "max_peak_tflops_s": max_peak_tflops_s,
        }

    flops = int(op_estimate["flops"])
    capture_tflops_s = _tflops_s(flops, capture_span_ns)
    memory_busy_tflops_s = _tflops_s(flops, memory_busy_span_ns)
    arithmetic_intensity = _ratio_or_none(float(flops), total_memory_bytes)
    trace_memory_roof_tflops_s = _memory_limited_tflops_s(
        arithmetic_intensity,
        trace_peak_bandwidth_gb_s,
    )
    max_memory_roof_tflops_s = _memory_limited_tflops_s(
        arithmetic_intensity,
        max_peak_bandwidth_gb_s,
    )
    trace_roof_tflops_s = _min_or_none(trace_peak_tflops_s, trace_memory_roof_tflops_s)
    max_roof_tflops_s = _min_or_none(max_peak_tflops_s, max_memory_roof_tflops_s)
    trace_compute_time_ns = _compute_time_ns(flops, trace_peak_tflops_s)
    trace_memory_time_ns = _memory_time_ns(total_memory_bytes, trace_peak_bandwidth_gb_s)
    max_compute_time_ns = _compute_time_ns(flops, max_peak_tflops_s)
    max_memory_time_ns = _memory_time_ns(total_memory_bytes, max_peak_bandwidth_gb_s)
    return {
        "availability": "estimated",
        "op": op_estimate["op"],
        "flops": flops,
        "flop_source": op_estimate["source"],
        "flop_counter_source": "algorithmic estimate from shader symbols",
        "arithmetic_intensity_flop_per_byte": arithmetic_intensity,
        "trace_ridge_point_flop_per_byte": _ridge_point_flop_per_byte(
            trace_peak_tflops_s,
            trace_peak_bandwidth_gb_s,
        ),
        "max_ridge_point_flop_per_byte": _ridge_point_flop_per_byte(
            max_peak_tflops_s,
            max_peak_bandwidth_gb_s,
        ),
        "capture_tflops_s": capture_tflops_s,
        "memory_busy_tflops_s": memory_busy_tflops_s,
        "trace_peak_tflops_s": trace_peak_tflops_s,
        "max_peak_tflops_s": max_peak_tflops_s,
        "trace_memory_roof_tflops_s": trace_memory_roof_tflops_s,
        "max_memory_roof_tflops_s": max_memory_roof_tflops_s,
        "trace_roof_tflops_s": trace_roof_tflops_s,
        "max_roof_tflops_s": max_roof_tflops_s,
        "trace_roof_bound": _roofline_bound(trace_peak_tflops_s, trace_memory_roof_tflops_s),
        "max_roof_bound": _roofline_bound(max_peak_tflops_s, max_memory_roof_tflops_s),
        "trace_compute_time_ns": trace_compute_time_ns,
        "trace_memory_time_ns": trace_memory_time_ns,
        "trace_roof_time_ns": _max_or_none(trace_compute_time_ns, trace_memory_time_ns),
        "max_compute_time_ns": max_compute_time_ns,
        "max_memory_time_ns": max_memory_time_ns,
        "max_roof_time_ns": _max_or_none(max_compute_time_ns, max_memory_time_ns),
        "capture_trace_roof_ratio": _ratio_or_none(
            capture_tflops_s,
            trace_roof_tflops_s,
        ),
        "capture_max_roof_ratio": _ratio_or_none(
            capture_tflops_s,
            max_roof_tflops_s,
        ),
        "memory_busy_trace_roof_ratio": _ratio_or_none(
            memory_busy_tflops_s,
            trace_roof_tflops_s,
        ),
        "memory_busy_max_roof_ratio": _ratio_or_none(
            memory_busy_tflops_s,
            max_roof_tflops_s,
        ),
        "capture_trace_peak_ratio": _ratio_or_none(
            capture_tflops_s,
            trace_peak_tflops_s,
        ),
        "capture_max_peak_ratio": _ratio_or_none(
            capture_tflops_s,
            max_peak_tflops_s,
        ),
        "memory_busy_trace_peak_ratio": _ratio_or_none(
            memory_busy_tflops_s,
            trace_peak_tflops_s,
        ),
        "memory_busy_max_peak_ratio": _ratio_or_none(
            memory_busy_tflops_s,
            max_peak_tflops_s,
        ),
    }


def _algorithmic_flop_estimate(row: dict[str, Any]) -> dict[str, Any] | None:
    shader = row.get("shader")
    symbols = row.get("symbols")
    if not isinstance(shader, str) or not isinstance(symbols, dict):
        return None
    if _is_elementwise_one_flop_shader(shader):
        elements = _symbol_element_count(symbols)
        if elements is None:
            return None
        return {
            "op": "elementwise",
            "flops": elements,
            "source": "one FLOP per output element from shape symbols",
        }

    if "linear" not in shader:
        return None

    k = _positive_int_symbol(symbols, "K")
    n = _positive_int_symbol(symbols, "N")
    if k is None or n is None:
        return None
    m = _positive_int_symbol(symbols, "M")
    if m is None and "_t1_" in shader:
        m = 1
    if m is None:
        return None

    flops = 2 * m * n * k
    if "bias" in shader and "nobias" not in shader:
        flops += m * n
    return {
        "op": "linear",
        "flops": flops,
        "source": "2*M*N*K plus bias adds when present",
    }


def _is_elementwise_one_flop_shader(shader: str) -> bool:
    return shader.startswith(("add_", "sub_", "mul_"))


def _symbol_element_count(symbols: dict[Any, Any]) -> int | None:
    count = 1
    found = False
    for value in symbols.values():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value <= 0:
            continue
        count *= value
        found = True
    if not found:
        return None
    return count


def _positive_int_symbol(symbols: dict[Any, Any], name: str) -> int | None:
    value = symbols.get(name)
    if isinstance(value, bool):
        return None
    if not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


def _fp32_fma_peak_tflops_s(asic: object, clock_hz: int | None) -> float | None:
    if not isinstance(asic, RgpAsicInfo) or clock_hz is None:
        return None
    compute_units = asic.shader_engines * asic.compute_unit_per_shader_engine
    if compute_units <= 0 or asic.simd_per_compute_unit <= 0 or clock_hz <= 0:
        return None
    lanes_per_simd = 32.0
    flops_per_fma = 2.0
    flops_per_clock = (
        float(compute_units)
        * float(asic.simd_per_compute_unit)
        * lanes_per_simd
        * flops_per_fma
    )
    return flops_per_clock * float(clock_hz) / 1_000_000_000_000.0


def _tflops_s(flops: int, duration_ns: float | None) -> float | None:
    if duration_ns is None or duration_ns <= 0.0:
        return None
    return float(flops) / (duration_ns / 1_000_000_000.0) / 1_000_000_000_000.0


def _memory_limited_tflops_s(
    arithmetic_intensity_flop_per_byte: float | None,
    bandwidth_gb_s: float | None,
) -> float | None:
    if arithmetic_intensity_flop_per_byte is None or bandwidth_gb_s is None:
        return None
    if arithmetic_intensity_flop_per_byte <= 0.0 or bandwidth_gb_s <= 0.0:
        return None
    return arithmetic_intensity_flop_per_byte * bandwidth_gb_s / 1_000.0


def _ridge_point_flop_per_byte(
    compute_peak_tflops_s: float | None,
    bandwidth_gb_s: float | None,
) -> float | None:
    if compute_peak_tflops_s is None or bandwidth_gb_s is None:
        return None
    if compute_peak_tflops_s <= 0.0 or bandwidth_gb_s <= 0.0:
        return None
    return compute_peak_tflops_s * 1_000.0 / bandwidth_gb_s


def _compute_time_ns(flops: int, peak_tflops_s: float | None) -> float | None:
    if peak_tflops_s is None or peak_tflops_s <= 0.0:
        return None
    return float(flops) / (peak_tflops_s * 1_000.0)


def _memory_time_ns(total_bytes: float, bandwidth_gb_s: float | None) -> float | None:
    if bandwidth_gb_s is None or bandwidth_gb_s <= 0.0:
        return None
    return total_bytes / bandwidth_gb_s


def _min_or_none(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _max_or_none(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _roofline_bound(compute_peak_tflops_s: float | None, memory_roof_tflops_s: float | None) -> str | None:
    if compute_peak_tflops_s is None or memory_roof_tflops_s is None:
        return None
    if memory_roof_tflops_s < compute_peak_tflops_s:
        return "memory"
    return "compute"


def _ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator
