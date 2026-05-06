"""Helpers for rendering shader-contract variant-body snippets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorFieldDecl:
    name: str
    io_kind: str
    role: str
    dtype: str
    shape: tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class ParamsFieldDecl:
    name: str
    dtype: str
    offset: int
    value: str | int


def render_shader_contract_variant_body(
    *,
    class_name: str,
    shader_name: str,
    tensor_fields: tuple[TensorFieldDecl, ...],
    dispatch: tuple[str | int, str | int, str | int],
    params_fields: tuple[ParamsFieldDecl, ...] = (),
    params_size: int = 0,
    params_binding_index: int | None = None,
) -> str:
    lines: list[str] = [
        "contract=ShaderContract(",
        f'    class_name="{class_name}",',
        f'    shader_name="{shader_name}",',
        "    fields=(",
    ]
    for field in tensor_fields:
        lines.extend(
            (
                "        TensorFieldSpec(",
                f'            name="{field.name}",',
                f"            io_kind=IOKind.{field.io_kind},",
                f'            role="{field.role}",',
                "            contract=TensorContract(",
                f'                dtype="{field.dtype}",',
                f"                shape={_pyrepr(field.shape)},",
                "            ),",
                "        ),",
            )
        )
    lines.append("    ),")
    lines.append(f"    dispatch={_pyrepr(dispatch)},")
    if params_fields:
        if params_binding_index is None:
            raise ValueError("params_binding_index is required when params_fields is not empty")
        lines.extend(
            (
                "    params_buffer=ParamsBufferSpec(",
                f"        size={params_size},",
                "        fields=(",
            )
        )
        for field in params_fields:
            lines.extend(
                (
                    "            ParamsBufferFieldSpec(",
                    f'                "{field.name}",',
                    f"                PushConstantType.{field.dtype},",
                    f"                {field.offset},",
                    _format_param_value(field.value) + ",",
                    "            ),",
                )
            )
        lines.extend(
            (
                "        ),",
                f"        binding_index={params_binding_index},",
                "    ),",
            )
        )
    lines.extend(("),",))
    return "\n".join(lines) + "\n"


def _format_param_value(value: str | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f'"{value}"'


def _pyrepr(value: object) -> str:
    if isinstance(value, tuple):
        if len(value) == 1:
            return f"({ _pyrepr(value[0]) },)"
        return "(" + ", ".join(_pyrepr(v) for v in value) + ")"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)

