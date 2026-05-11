"""Shader Python module rendering."""

from __future__ import annotations

from pathlib import Path

from torch2vk.export._templates import render_template
from torch2vk.runtime.shader import (
    AddExpr,
    CeilDivExpr,
    MulExpr,
    ParamsBufferSpec,
    PushConstantInput,
    ShaderContract,
    ShaderVariant,
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements
from torch2vk.vulkan.types import CONTIGUOUS_LAYOUT, Q4KWordsLayout, Q8_0HalfwordsLayout


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

    layout_import_names: set[str] = set()
    for field in contract.fields:
        for dim in field.contract.shape:
            check_expr(dim)
        layout_source = _layout_to_source(field.contract.layout)
        if layout_source != "CONTIGUOUS_LAYOUT":
            layout_import_names.add(layout_source.split("(", 1)[0])

    imports = ["from __future__ import annotations", "", "from torch2vk.runtime.shader import ("]
    for name in sorted(needed):
        imports.append(f"    {name},")
    imports.append(")")
    execution_requirements_source = _execution_requirements_to_source(variant.execution_requirements)
    if execution_requirements_source != "None":
        imports.append("from torch2vk.vulkan.shader_execution_requirements import (")
        imports.append("    ShaderExecutionRequirements,")
        if (
            variant.execution_requirements is not None
            and variant.execution_requirements.cooperative_matrix is not None
        ):
            imports.append("    CooperativeMatrixRequirements,")
        if variant.execution_requirements is not None and variant.execution_requirements.subgroup is not None:
            imports.append("    SubgroupRequirements,")
        imports.append(")")
    if layout_import_names:
        imports.append("from torch2vk.vulkan.types import (")
        for name in sorted(layout_import_names):
            imports.append(f"    {name},")
        imports.append(")")

    fields_lines = []
    for field in contract.fields:
        layout_source = _layout_to_source(field.contract.layout)
        fields_lines.append("            TensorFieldSpec(")
        fields_lines.append(f"                name={field.name!r},")
        fields_lines.append(f"                io_kind=IOKind.{field.io_kind.name},")
        fields_lines.append(f"                role={field.role!r},")
        fields_lines.append(
            "                "
            f"contract=TensorContract(dtype={field.contract.dtype!r}, "
            f"shape={_shape_to_source(field.contract.shape)}{_layout_arg_source(layout_source)}),"
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


def clear_shader_package(shaders_dir: Path) -> None:
    for f in shaders_dir.glob("*.py"):
        f.unlink()


def write_shader_file(shaders_dir: Path, variant: ShaderVariant) -> None:
    (shaders_dir / f"{variant.name}.py").write_text(render_shader_file(variant))


def write_shader_init(shaders_dir: Path) -> None:
    (shaders_dir / "__init__.py").write_text('"""Generated shader package."""\n')


def rename_shader_variant(variant: ShaderVariant, new_name: str) -> ShaderVariant:
    return ShaderVariant(
        name=new_name,
        family=variant.family,
        contract=ShaderContract(
            class_name=variant.contract.class_name,
            shader_name=new_name,
            fields=variant.contract.fields,
            dispatch=variant.contract.dispatch,
            push_constants=variant.contract.push_constants,
            params_buffer=variant.contract.params_buffer,
        ),
        source=variant.source,
        precompiled_spv_path=variant.precompiled_spv_path,
        specialization_constants=variant.specialization_constants,
        include_dirs=variant.include_dirs,
        compile_defines=variant.compile_defines,
        execution_requirements=variant.execution_requirements,
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


def _layout_to_source(layout) -> str:
    if layout == CONTIGUOUS_LAYOUT:
        return "CONTIGUOUS_LAYOUT"
    if isinstance(layout, Q4KWordsLayout):
        return (
            f"q4_k_words_layout(logical_k={_expr_to_source(layout.logical_k)}, "
            f"block_size={layout.block_size}, words_per_block={layout.words_per_block})"
        )
    if isinstance(layout, Q8_0HalfwordsLayout):
        return (
            f"q8_0_halfwords_layout(logical_k={_expr_to_source(layout.logical_k)}, "
            f"block_size={layout.block_size}, halfwords_per_block={layout.halfwords_per_block})"
        )
    raise TypeError(f"Unsupported generated shader tensor layout: {layout!r}")


def _layout_arg_source(layout_source: str) -> str:
    if layout_source == "CONTIGUOUS_LAYOUT":
        return ""
    return f", layout={layout_source}"


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
        cooperative_matrix = requirements.cooperative_matrix
        fields.append(
            "cooperative_matrix=CooperativeMatrixRequirements("
            f"scope={cooperative_matrix.scope!r}, "
            f"m_size={cooperative_matrix.m_size}, "
            f"n_size={cooperative_matrix.n_size}, "
            f"k_size={cooperative_matrix.k_size}, "
            f"a_type={cooperative_matrix.a_type!r}, "
            f"b_type={cooperative_matrix.b_type!r}, "
            f"c_type={cooperative_matrix.c_type!r}, "
            f"result_type={cooperative_matrix.result_type!r}, "
            f"saturating_accumulation={cooperative_matrix.saturating_accumulation}"
            ")"
        )
    if requirements.require_integer_dot_product:
        fields.append("require_integer_dot_product=True")
    if requirements.require_shader_int64:
        fields.append("require_shader_int64=True")
    if requirements.require_buffer_device_address:
        fields.append("require_buffer_device_address=True")
    if requirements.require_storage_buffer_16bit_access:
        fields.append("require_storage_buffer_16bit_access=True")
    return f"ShaderExecutionRequirements({', '.join(fields)})"
