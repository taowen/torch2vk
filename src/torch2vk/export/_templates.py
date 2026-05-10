"""Shared Jinja template loader for export codegen."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


_TEMPLATE_DIR = Path(__file__).with_name("templates")
_JINJA = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    loader=FileSystemLoader(_TEMPLATE_DIR),
    lstrip_blocks=True,
    trim_blocks=True,
    undefined=StrictUndefined,
)


def render_template(template_name: str, **context) -> str:
    return _JINJA.get_template(template_name).render(**context)


def render_simple_init(docstring: str, imports: list[str]) -> str:
    return render_template(
        "simple_init.py.j2",
        docstring=docstring,
        imports_source="\n".join(imports),
    )
