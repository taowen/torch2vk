"""Shader Python module rendering."""

from __future__ import annotations

from torch2vk.export._templates import render_template
from torch2vk.runtime.shader import (
    AddExpr,
    CeilDivExpr,
    MulExpr,
    ParamsBufferSpec,
    PushConstantInput,
    ShaderVariant,
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


def render_shader_file(variant: ShaderVariant) -> str:
    contract = variant.contract
    const_name = variant.name.upper()

    needed = {"IOKind", "ShaderContract", "ShaderVariant", "TensorContract", "TensorFieldSpec"}
    if contract.push_constants:
        needed.update({"PushConstantFieldSpec", "PushConstantSpec", "PushConstantType"})
    if contract.params_buffer:
        needed.update({"ParamsBufferFieldSpec", "ParamsBufferSpec", "PushConstantType"})
    if _variant_uses_push_constant_input(variant):
        needed.add("PushConstantInput")

    def check_expr(expr) -> None:
        if isinstance(expr, CeilDivExpr):
            needed.add("ceil_div")
            check_expr(expr.lhs)
            check_expr(expr.rhs)
        elif isinstance(expr, MulExpr):
            needed.add("mul")
            check_expr(expr.lhs)
            check_expr(expr.rhs)
        elif isinstance(expr, AddExpr):
            needed.add("add")
            check_expr(expr.lhs)
            check_expr(expr.rhs)

    for dim in contract.dispatch:
        check_expr(dim)
    if contract.push_constants:
        for field in contract.push_constants.fields:
            check_expr(field.value)
    if contract.params_buffer:
        for field in contract.params_buffer.fields:
            check_expr(field.value)

    imports = ["from __future__ import annotations", "", "from torch2vk.runtime.shader import ("]
    for name in sorted(needed):
        imports.append(f"    {name},")
    imports.append(")")
    execution_requirements_source = _execution_requirements_to_source(variant.execution_requirements)
    if execution_requirements_source != "None":
        imports.append("from torch2vk.vulkan.shader_execution_requirements import (")
        imports.append("    ShaderExecutionRequirements,")
        if variant.execution_requirements is not None and variant.execution_requirements.subgroup is not None:
            imports.append("    SubgroupRequirements,")
        imports.append(")")

    fields_lines = []
    for field in contract.fields:
        fields_lines.append("            TensorFieldSpec(")
        fields_lines.append(f"                name={field.name!r},")
        fields_lines.append(f"                io_kind=IOKind.{field.io_kind.name},")
        fields_lines.append(f"                role={field.role!r},")
        fields_lines.append(
            "                "
            f"contract=TensorContract(dtype={field.contract.dtype!r}, "
            f"shape={_shape_to_source(field.contract.shape)}),"
        )
        fields_lines.append("            ),")

    dispatch_source = f"({', '.join(_expr_to_source(dim) for dim in contract.dispatch)})"

    return render_template(
        "shader_file.py.j2",
        variant_name=variant.name,
        variant_name_source=repr(variant.name),
        const_name=const_name,
        family_source=repr(variant.family),
        class_name_source=repr(contract.class_name),
        imports_source="\n".join(imports),
        fields_source="\n".join(fields_lines),
        push_constants_source=_push_constant_spec_to_source(contract.push_constants),
        params_buffer_source=_params_buffer_spec_to_source(contract.params_buffer),
        dispatch_source=dispatch_source,
        execution_requirements_source=execution_requirements_source,
        glsl=variant.source.lstrip("\n"),
    )


def _expr_to_source(expr) -> str:
    if isinstance(expr, int):
        return repr(expr)
    if isinstance(expr, str):
        return repr(expr)
    if isinstance(expr, CeilDivExpr):
        return f"ceil_div({_expr_to_source(expr.lhs)}, {_expr_to_source(expr.rhs)})"
    if isinstance(expr, MulExpr):
        return f"mul({_expr_to_source(expr.lhs)}, {_expr_to_source(expr.rhs)})"
    if isinstance(expr, AddExpr):
        return f"add({_expr_to_source(expr.lhs)}, {_expr_to_source(expr.rhs)})"
    if isinstance(expr, PushConstantInput):
        return f"PushConstantInput({expr.name!r})"
    raise TypeError(f"Unknown expr type: {type(expr)}")


def _shape_to_source(shape: tuple) -> str:
    return f"({', '.join(_expr_to_source(dim) for dim in shape)},)"


def _variant_uses_push_constant_input(variant: ShaderVariant) -> bool:
    push = variant.contract.push_constants
    if push is not None:
        for field in push.fields:
            if isinstance(field.value, PushConstantInput):
                return True
    params = variant.contract.params_buffer
    if params is not None:
        for field in params.fields:
            if isinstance(field.value, PushConstantInput):
                return True
    return False


def _push_constant_value_source(value) -> str:
    return repr(value) if isinstance(value, (int, float)) else _expr_to_source(value)


def _push_constant_spec_to_source(spec) -> str:
    if spec is None:
        return "None"
    fields = []
    for field in spec.fields:
        fields.append(
            "                "
            f"PushConstantFieldSpec({field.name!r}, PushConstantType.{field.dtype.name}, "
            f"{field.offset}, {_push_constant_value_source(field.value)}, dynamic={field.dynamic}),"
        )
    return (
        "PushConstantSpec(\n"
        f"            size={spec.size},\n"
        "            fields=(\n" + "\n".join(fields) + "\n"
        "            ),\n"
        "        )"
    )


def _params_buffer_spec_to_source(spec: ParamsBufferSpec | None) -> str:
    if spec is None:
        return "None"
    fields = []
    for field in spec.fields:
        fields.append(
            "                "
            f"ParamsBufferFieldSpec({field.name!r}, PushConstantType.{field.dtype.name}, "
            f"{field.offset}, {_push_constant_value_source(field.value)}),"
        )
    return (
        "ParamsBufferSpec(\n"
        f"            size={spec.size},\n"
        "            fields=(\n" + "\n".join(fields) + "\n"
        "            ),\n"
        f"            binding_index={spec.binding_index},\n"
        "        )"
    )


def _execution_requirements_to_source(requirements: ShaderExecutionRequirements | None) -> str:
    if requirements is None:
        return "None"
    fields = []
    if requirements.subgroup is not None:
        fields.append(
            "subgroup=SubgroupRequirements("
            f"required_size={requirements.subgroup.required_size}, "
            f"require_full_subgroups={requirements.subgroup.require_full_subgroups}"
            ")"
        )
    if requirements.cooperative_matrix is not None:
        raise NotImplementedError("generated shader files do not support cooperative matrix requirements yet")
    if requirements.require_integer_dot_product:
        fields.append("require_integer_dot_product=True")
    if requirements.require_shader_int64:
        fields.append("require_shader_int64=True")
    if requirements.require_buffer_device_address:
        fields.append("require_buffer_device_address=True")
    if requirements.require_storage_buffer_16bit_access:
        fields.append("require_storage_buffer_16bit_access=True")
    return f"ShaderExecutionRequirements({', '.join(fields)})"
