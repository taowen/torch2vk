"""Load GLSL strings from copied Agentorch shader source modules."""

from __future__ import annotations

import ast
from functools import cache
from importlib import resources

_COPIED_PACKAGE = "torch2vk.copied.agentorch_shader_source"
type _Assignments = dict[str, ast.expr]


@cache
def copied_assignment_string(module_file: str, variable_name: str) -> str:
    assignments = _copied_module_assignments(module_file)
    try:
        return _string_value(assignments[variable_name], assignments)
    except KeyError as exc:
        raise KeyError(f"{module_file} does not define {variable_name}") from exc


@cache
def copied_shader_variant_source(module_file: str, variant_name: str) -> str:
    source = _read_copied_module(module_file)
    module = ast.parse(source, filename=module_file)
    assignments = _copied_module_assignments(module_file)
    for node in module.body:
        if not isinstance(node, ast.Assign) or not _assigns_name(node, variant_name):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        for keyword in call.keywords:
            if keyword.arg != "source":
                continue
            return _string_value(keyword.value, assignments)
    raise KeyError(f"{module_file} does not define shader source for {variant_name}")


@cache
def copied_module_shader_sources(module_file: str) -> dict[str, str]:
    source = _read_copied_module(module_file)
    module = ast.parse(source, filename=module_file)
    assignments = _copied_module_assignments(module_file)
    sources: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call) or not _is_shader_variant_call(call):
            continue
        source_expr = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "source"),
            None,
        )
        if source_expr is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                sources[target.id] = _string_value(source_expr, assignments)
    return sources


def _assigns_name(node: ast.stmt, name: str) -> bool:
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


@cache
def _copied_module_assignments(module_file: str) -> _Assignments:
    source = _read_copied_module(module_file)
    module = ast.parse(source, filename=module_file)
    assignments = _module_assignments(module)
    for node in module.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or node.module is None:
            continue
        imported_file = f"{node.module}.py"
        try:
            imported_assignments = _copied_module_assignments(imported_file)
        except FileNotFoundError:
            continue
        for alias in node.names:
            local_name = alias.asname or alias.name
            try:
                assignments[local_name] = imported_assignments[alias.name]
            except KeyError:
                continue
    return assignments


def _module_assignments(module: ast.Module) -> _Assignments:
    assignments: _Assignments = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value
    return assignments


def _is_shader_variant_call(call: ast.Call) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == "shader_variant"


def _read_copied_module(module_file: str) -> str:
    return resources.files(_COPIED_PACKAGE).joinpath(module_file).read_text(encoding="utf-8")


def _variant_source_value(variant_name: str, assignments: _Assignments) -> str:
    try:
        value = assignments[variant_name]
    except KeyError as exc:
        raise KeyError(f"Copied shader source references unknown variant {variant_name}") from exc
    if not isinstance(value, ast.Call) or not _is_shader_variant_call(value):
        raise TypeError(f"{variant_name}.source does not reference a shader variant")
    source_expr = next(
        (keyword.value for keyword in value.keywords if keyword.arg == "source"),
        None,
    )
    if source_expr is None:
        raise KeyError(f"{variant_name} does not define a shader source")
    return _string_value(source_expr, assignments)


def _string_value(node: ast.expr, assignments: _Assignments) -> str:
    if isinstance(node, ast.JoinedStr):
        return "".join(_joined_string_part(value, assignments) for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _string_value(node.left, assignments) + _string_value(node.right, assignments)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.attr == "source"
    ):
        return _variant_source_value(node.value.id, assignments)
    if isinstance(node, ast.Call):
        return _call_string_value(node, assignments)
    if isinstance(node, ast.Name):
        try:
            return _string_value(assignments[node.id], assignments)
        except KeyError as exc:
            raise KeyError(f"Copied shader source references unknown name {node.id}") from exc
    return _literal_string(node)


def _call_string_value(node: ast.Call, assignments: _Assignments) -> str:
    if (
        isinstance(node.func, ast.Attribute)
        and not node.args
        and not node.keywords
        and node.func.attr in {"lstrip", "strip"}
    ):
        value = _string_value(node.func.value, assignments)
        return value.lstrip() if node.func.attr == "lstrip" else value.strip()
    if (
        isinstance(node.func, ast.Attribute)
        and len(node.args) == 2
        and not node.keywords
        and node.func.attr == "replace"
    ):
        base = _string_value(node.func.value, assignments)
        old = ast.literal_eval(node.args[0])
        new = ast.literal_eval(node.args[1])
        if isinstance(old, str) and isinstance(new, str):
            return base.replace(old, new)
    raise TypeError("Unsupported copied shader string call")


def _literal_string(node: ast.expr) -> str:
    value = ast.literal_eval(node)
    if not isinstance(value, str):
        raise TypeError("AST expression is not a string")
    return value


def _joined_string_part(node: ast.expr, assignments: _Assignments) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.FormattedValue):
        value = node.value
        if isinstance(value, ast.Name):
            try:
                literal_value = ast.literal_eval(assignments[value.id])
            except KeyError as exc:
                message = f"Copied shader f-string references unknown name {value.id}"
                raise KeyError(message) from exc
        else:
            literal_value = ast.literal_eval(value)
        return str(literal_value)
    raise TypeError("Unsupported copied shader f-string expression")
