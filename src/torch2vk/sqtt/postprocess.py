"""Join torch2vk runtime dispatches with RADV SQTT driver artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .postprocess_common import (
    load_jsonl,
    remove_stale_report_outputs,
    update_run_json,
    write_jsonl,
)
from .postprocess_join import build_attribution_rows
from .report import build_report_payload, write_reports
from .source_reports import build_source_isa_reports
from .spm_reports import build_spm_counter_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="SQTT run root")
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="also write full pipeline/source-ISA decoder intermediates under debug/",
    )
    args = parser.parse_args()

    try:
        postprocess(args.root.expanduser().resolve(), debug_artifacts=args.debug_artifacts)
    except Exception as exc:
        print(f"torch2vk.sqtt: {exc}", file=sys.stderr)
        return 1
    return 0


def postprocess(root: Path, *, debug_artifacts: bool = False) -> None:
    driver_dir = root / "driver"
    dispatches_path = root / "dispatches.jsonl"
    driver_dispatch_path = driver_dir / "dispatch-sequence.jsonl"
    capture_path = driver_dir / "capture-sequence.jsonl"

    manifest_rows = load_jsonl(dispatches_path)
    driver_dispatch_rows = load_jsonl(driver_dispatch_path)
    capture_rows = load_jsonl(capture_path)
    if not manifest_rows:
        raise RuntimeError(f"{dispatches_path} is empty")
    if not driver_dispatch_rows:
        raise RuntimeError(f"{driver_dispatch_path} is empty")
    if not capture_rows:
        raise RuntimeError(f"{capture_path} is empty")

    non_record = [row for row in manifest_rows if row.get("phase") != "record"]
    if non_record:
        raise RuntimeError(
            "SQTT only supports record-mode dispatches; "
            f"found {len(non_record)} non-record manifest row(s)"
        )

    rows = build_attribution_rows(
        root=root,
        manifest_rows=manifest_rows,
        driver_dispatch_rows=driver_dispatch_rows,
        capture_rows=capture_rows,
    )

    remove_stale_report_outputs(root)
    attribution_path = root / "attribution.jsonl"
    write_jsonl(attribution_path, rows)
    source_isa_reports, focus, debug_outputs = build_source_isa_reports(
        root=root,
        rows=rows,
        debug_artifacts=debug_artifacts,
    )
    spm_counter_reports = build_spm_counter_reports(root=root, rows=rows)
    report = build_report_payload(
        root=root,
        attribution_rows=rows,
        capture_rows=capture_rows,
        source_isa_reports=source_isa_reports,
        spm_counter_reports=spm_counter_reports,
        focus=focus,
        debug_artifacts=debug_outputs,
    )
    report_paths = write_reports(root=root, report=report)
    update_run_json(
        root,
        report={
            **report_paths,
            "captured_submits": report["captured_submits"],
            "captured_dispatches": report["captured_dispatches"],
            "source_isa_reports": report["source_isa_reports"],
            "spm_counter_reports": report["spm_counter_reports"],
            "debug_artifacts": report["debug_artifacts"],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
