"""Generated PyTorch reference comparison helpers."""

from __future__ import annotations

from typing import Literal, TypeAlias

from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

from torch2vk.export._templates import render_template
from torch2vk.export.dispatch_codegen import _find_graph_outputs


ReferencePolicy: TypeAlias = (
    Literal["tensor", "q8_tensor", "q4_tensor", "token"]
    | dict[str, Literal["tensor", "q8_tensor", "q4_tensor", "token"]]
)


def render_reference_module(
    *,
    model_package: str,
    model_imports: list[str],
    model_type: str,
    reference_functions: list[str],
    loader_fields: list[str],
    loader_sources: list[str],
) -> str:
    return render_template(
        "reference_module.py.j2",
        model_package=model_package,
        model_imports_source="\n".join(model_imports),
        model_type=model_type,
        loaders=tuple({"field": field} for field in loader_fields),
        loaders_source="\n\n".join(loader_sources).rstrip(),
        reference_functions_source="\n\n".join(reference_functions).rstrip(),
    ).rstrip() + "\n"


def render_reference_loader(
    *,
    field: str,
    module_path: str,
) -> str:
    return render_template(
        "reference_loader_function.py.j2",
        field=field,
        module_path_source=repr(module_path),
    ).rstrip()


def render_reference_function(
    *,
    name: str,
    reference_source: str,
    tensors: str,
    frame_name: str,
    policy: ReferencePolicy = "tensor",
    input_bindings: dict[str, str],
    output_bindings: dict[str, str],
    needs_reference: bool,
) -> str:
    item = _reference_function_item(
        name,
        reference_source=reference_source,
        tensors=tensors,
        frame_name=frame_name,
        policy=policy,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        needs_reference=needs_reference,
    )
    return render_template("reference_function.py.j2", **item).rstrip()


def render_exported_reference_function(
    ep: ExportedProgram,
    *,
    name: str,
    reference_source: str,
    tensors: str,
    frame_name: str,
    policy: ReferencePolicy = "tensor",
    input_bindings: dict[str, str] | None = None,
    output_bindings: dict[str, str] | None = None,
) -> str:
    if input_bindings is None:
        input_bindings = {
            spec.arg.name: spec.arg.name
            for spec in ep.graph_signature.input_specs
            if spec.kind == InputKind.USER_INPUT
        }
    if output_bindings is None:
        output_bindings = {name: name for name in _find_graph_outputs(ep.graph_module.graph)}
    return render_reference_function(
        name=name,
        reference_source=reference_source,
        tensors=tensors,
        frame_name=frame_name,
        policy=policy,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        needs_reference=reference_source == "reference",
    )


def _reference_function_item(
    name: str,
    *,
    reference_source: str,
    tensors: str,
    frame_name: str,
    policy: ReferencePolicy,
    input_bindings: dict[str, str],
    output_bindings: dict[str, str],
    needs_reference: bool,
) -> dict[str, object]:
    extra_params: list[dict[str, str]] = []
    names_seen: set[str] = set()
    for param_name, param_type in (
        ("name", "str"),
        ("step", "int"),
        ("layer_idx", "int"),
    ):
        token = "{" + param_name
        if token in frame_name or param_name in tensors:
            extra_params.append({"name": param_name, "type": param_type})
            names_seen.add(param_name)
    input_names = tuple(input_bindings.keys())
    duplicate = names_seen.intersection(input_names)
    if duplicate:
        raise ValueError(f"{name} reference inputs conflict with generated params: {sorted(duplicate)}")
    return {
        "function_name": f"run_{name}",
        "extra_params": tuple(extra_params),
        "input_names": input_names,
        "name_source": _reference_name_source(frame_name),
        "reference_source": reference_source,
        "tensors_source": tensors,
        "output_bindings_source": repr(output_bindings),
        "policy_source": repr(policy),
        "needs_reference": needs_reference,
    }


def _reference_name_source(name: str) -> str:
    if "{" in name:
        return f"f{name!r}"
    return repr(name)
