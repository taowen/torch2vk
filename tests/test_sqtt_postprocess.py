from __future__ import annotations

from pathlib import Path


def test_compact_hotspot_focus_accepts_report_tuples(tmp_path: Path) -> None:
    from torch2vk.sqtt import report

    focus = report.compact_hotspot_focus(
        attribution_row={
            "dispatch_index": 1223,
            "dispatch_size": [9496, 1, 1],
            "driver_code_size": 428,
            "driver_exec_size": 408,
            "driver_lds_size": 1024,
            "driver_scratch_memory_size": 0,
            "driver_sgpr_count": 108,
            "driver_vgpr_count": 8,
            "driver_wave_size": 64,
            "frame": "qwen3_asr.text_decode.0000",
            "output_op": "logits",
            "pipeline_debug_name": "agp.test",
            "reads": [
                {"field": "x", "tensor": "qwen3_asr.text_decode.final_norm"},
                {"field": "weight", "tensor": "thinker.lm_head.weight"},
            ],
            "rgp_path": "rgp/submit2208.rgp",
            "shader": "qwen3_asr_text_linear_nobias_t1_f32",
            "submit_ordinal": 2208,
            "symbols": {"K": 1024, "N": 151936},
            "writes": [{"field": "output", "tensor": "qwen3_asr.text_decode.logits"}],
        },
        hotspot_payload={
            "availability": "available",
            "matched_instruction_event_count": 100,
            "total_instruction_event_count": 100,
            "zero_pc_instruction_event_count": 0,
            "source_hot_lines": (
                {
                    "glsl_path": str(tmp_path / "shader.comp"),
                    "hit_count": 64,
                    "hottest_categories": (("VALU", 32), ("IMMED", 16)),
                    "line": 41,
                    "source_text": "        for (uint k = k_lane; k < pc.K; k += 16u) {",
                    "total_cycles": 900,
                },
            ),
            "hot_ranges": (
                {
                    "categories": (("IMMED", 16),),
                    "disasm": (
                        {
                            "opcode": "s_waitcnt",
                            "text": "s_waitcnt vmcnt(0)",
                        },
                    ),
                    "hit_count": 16,
                    "line": 41,
                    "source_text": "        for (uint k = k_lane; k < pc.K; k += 16u) {",
                    "total_cycles": 800,
                },
            ),
        },
        hotspot_path="source-isa-sqtt-hotspots.json",
        root=tmp_path,
    )

    assert focus["coverage"]["matched_instruction_event_ratio"] == 1.0
    assert focus["resource_usage"]["sgpr_count"] == 108
    assert focus["top_isa_ranges"][0]["opcode"] == "s_waitcnt"
    assert focus["top_source_lines"][0]["reported_cycle_ratio"] == 1.0

    markdown = report.render_markdown(
        {
            "captured_dispatches": 1,
            "captured_submits": 1,
            "artifact_inventory": [],
            "focus": [focus],
            "limits": ["cache_hit_miss_counters"],
            "source_isa_reports": [],
        }
    )
    assert "`s_waitcnt`" in markdown
    assert "Not reported: cache_hit_miss_counters" in markdown
    assert "### Unavailable Data" not in markdown
    assert "Dominant ISA hotspot" not in markdown


def test_report_payload_keeps_debug_artifacts_optional(tmp_path: Path) -> None:
    from torch2vk.sqtt import report

    (tmp_path / "attribution.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "dispatches.jsonl").write_text("{}", encoding="utf-8")
    payload = report.build_report_payload(
        root=tmp_path,
        attribution_rows=[],
        capture_rows=[],
        source_isa_reports=[],
        focus=[],
    )
    paths = report.write_reports(root=tmp_path, report=payload)

    assert paths == {"report_json_path": "report.json", "report_path": "report.md"}
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert payload["debug_artifacts"] == {}
    assert "cache_hit_miss_counters" in payload["limits"]


def test_missing_glsl_source_text_is_not_fatal(tmp_path: Path) -> None:
    from torch2vk.sqtt.pipeline_attribution import _glsl_path_for_row
    from torch2vk.sqtt.source_isa_hotspot_report import _source_text

    missing_path = tmp_path / "missing.comp"

    assert _source_text(glsl_path=str(missing_path), line=12, source_lines_cache={}) == ""
    assert _glsl_path_for_row({"shader_spv_path": str(tmp_path / "shader.spv")}) == (
        tmp_path / "shader.comp"
    )
