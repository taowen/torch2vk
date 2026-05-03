#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required but was not found in PATH" >&2
  exit 127
fi

echo "==> ruff"
uv run ruff check src tests

echo "==> pyright"
uv run pyright src tests

echo "==> pytest"
uv run pytest
