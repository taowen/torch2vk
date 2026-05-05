"""Command-line entrypoint for SQTT postprocessing."""

from __future__ import annotations

from .postprocess import main


if __name__ == "__main__":
    raise SystemExit(main())
