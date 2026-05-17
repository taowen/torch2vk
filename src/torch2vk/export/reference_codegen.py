"""Generated PyTorch/Vulkan streaming comparison helpers."""

from __future__ import annotations

from typing import Literal, TypeAlias

from torch2vk.export._templates import render_template


ReferencePolicy: TypeAlias = (
    Literal["tensor", "q8_tensor", "q4_tensor", "token"]
    | dict[str, Literal["tensor", "q8_tensor", "q4_tensor", "token"]]
)


def render_reference_module(
    *,
    model_package: str,
    reference_functions: list[str],
    dispatch_imports: list[str] | None = None,
) -> str:
    return (
        render_template(
            "reference_module.py.j2",
            model_package=model_package,
            dispatch_imports_source="\n".join(dispatch_imports or ()),
            reference_functions_source="\n\n".join(reference_functions).rstrip(),
        ).rstrip()
        + "\n"
    )


def render_streaming_compare_function(
    *,
    name: str,
    dispatch_source: str,
    tensors: str,
    frame_name: str,
    policy: ReferencePolicy = "tensor",
    input_bindings: dict[str, str],
    output_bindings: dict[str, str],
    dispatch_args: tuple[str, ...] = (),
    dispatch_kwargs: tuple[str, ...] = (),
) -> str:
    item = _reference_function_item(
        name,
        tensors=tensors,
        frame_name=frame_name,
        policy=policy,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
    item["function_name"] = f"compare_{name}"
    item["dispatch_source"] = dispatch_source
    item["dispatch_args"] = tuple(dispatch_args)
    item["dispatch_kwargs"] = tuple(dispatch_kwargs)
    input_names = list(input_bindings)
    for kwarg in dispatch_kwargs:
        if kwarg not in input_names:
            input_names.append(kwarg)
    item["input_names"] = tuple(input_names)
    item["stage_input_names"] = tuple(input_bindings)
    return render_template("streaming_compare_function.py.j2", **item).rstrip()


def _reference_function_item(
    name: str,
    *,
    tensors: str,
    frame_name: str,
    policy: ReferencePolicy,
    input_bindings: dict[str, str],
    output_bindings: dict[str, str],
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
        raise ValueError(
            f"{name} reference inputs conflict with generated params: {sorted(duplicate)}"
        )
    return {
        "function_name": f"run_{name}",
        "extra_params": tuple(extra_params),
        "input_names": input_names,
        "name_source": _reference_name_source(frame_name),
        "tensors_source": tensors,
        "input_bindings_source": repr(input_bindings),
        "output_bindings_source": repr(output_bindings),
        "policy_source": repr(policy),
    }


def _reference_name_source(name: str) -> str:
    if "{" in name:
        return f"f{name!r}"
    return repr(name)
