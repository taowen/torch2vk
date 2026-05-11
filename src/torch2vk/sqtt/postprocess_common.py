"""Shared postprocess file and type helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(payload)
    return rows


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True))
            output.write("\n")


def remove_stale_report_outputs(root: Path) -> None:
    for name in (
        "sqtt-summary.json",
        "sqtt-report.md",
        "pipeline-attribution.json",
        "source-isa-sqtt-hotspots.json",
    ):
        path = root / name
        if path.is_file():
            path.unlink()
    for path in root.glob("source-isa-sqtt-hotspots-submit*.json"):
        if path.is_file():
            path.unlink()
    debug_dir = root / "debug"
    if debug_dir.is_dir():
        shutil.rmtree(debug_dir)


def update_run_json(root: Path, report: dict[str, Any]) -> None:
    path = root / "run.json"
    payload: dict[str, Any] = {}
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            payload = raw
    payload["sqtt"] = {
        "attribution_path": "attribution.jsonl",
        **report,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def expect_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string, got {type(value).__name__}")
    return value


def expect_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an int, got {type(value).__name__}")
    return value


def relative_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
