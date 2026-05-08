"""Generate Python dispatch code directly from torch.export.ExportedProgram."""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined
from torch.export import ExportedProgram
from torch.export.graph_signature import ExportGraphSignature, InputKind
from torch.fx import Graph, Node

from torch2vk.export.graph import SKIP_OPS, is_alias_op, node_input_names
from torch2vk.export.registry import DEFAULT_REGISTRY, ShaderRegistry
from torch2vk.runtime.shader import (
    AddExpr,
    CeilDivExpr,
    IOKind,
    MulExpr,
    ParamsBufferSpec,
    PushConstantInput,
    ShaderVariant,
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements

_JINJA = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    lstrip_blocks=True,
    trim_blocks=True,
    undefined=StrictUndefined,
)

_SHADER_FILE_TEMPLATE = '''"""Generated shader: {{ variant_name }}."""

{{ imports_source }}


{{ const_name }} = ShaderVariant(
    name={{ variant_name_source }},
    family={{ family_source }},
    contract=ShaderContract(
        class_name={{ class_name_source }},
        shader_name={{ variant_name_source }},
        fields=(
{{ fields_source }}
        ),
        push_constants={{ push_constants_source }},
        params_buffer={{ params_buffer_source }},
        dispatch={{ dispatch_source }},
    ),
    execution_requirements={{ execution_requirements_source }},
    source="""\\
{{ glsl }}""",
)
'''

_DISPATCH_FUNCTION_TEMPLATE = """def {{ function_name }}(rt: RuntimeSession, tensors: {{ class_name }}) -> None:
{% for op in ops %}
{% if op.type == "alias" %}
    _alias(rt, tensors.{{ op.src }}, tensors.{{ op.dst }})
{% elif op.type == "dispatch" %}
    {{ op.shader_const }}(rt, {{ op.args_source }})
{% elif op.type == "unsupported" %}
    raise RuntimeError({{ op.message_source }})
{% endif %}
{% endfor %}
"""

_DISPATCH_FILE_TEMPLATE = '''"""{{ docstring }}."""

from __future__ import annotations

{{ extra_imports_source }}
{% for item in shader_imports %}
from {{ shader_package }}.{{ item.shader }} import {{ item.const }}
{% endfor %}
{% for item in tensor_imports %}
from {{ tensor_package }}.{{ item.file }} import {{ item.classes_source }}
{% endfor %}
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.shader import ShaderVariant
from torch2vk.runtime.session import RuntimeSession


SHADER_VARIANTS_BY_NAME: dict[str, ShaderVariant] = {
{% for item in shader_imports %}
    {{ item.shader|tojson }}: {{ item.const }},
{% endfor %}
{% for item in extra_shader_variants %}
    {{ item.name_source }}: {{ item.const }},
{% endfor %}
}


{{ function_sources_source }}


def _alias(rt: RuntimeSession, src: LogicalTensor, dst: LogicalTensor) -> None:
    rt._materialize_read(src)
    with dst.runtime_write_scope():
        dst.buffer = src.buffer
        dst.descriptor_nbytes = src.descriptor_nbytes
        dst.version = src.version
        dst.writer = src.writer
    rt._current_frame().written_tensors.append(dst)
'''

_TENSOR_CLASS_TEMPLATE = """@dataclass(frozen=True, slots=True)
class {{ class_name }}:
{% for field in fields %}
    {{ field }}: LogicalTensor
{% endfor %}


{{ output_const }}: str = {{ output_name_source }}


{{ signature }}
    _validate_bindings(bindings, frozenset({{ tensor_names_source }}))
    _validate_request_state_outputs(request_state_outputs, frozenset(({{ output_name_source }},)))
    return {{ class_name }}(
{% for tensor in tensors %}
        {{ tensor.name }}=_bind_tensor(
            bindings,
            {{ tensor.name_source }},
            _declare_tensor(
            name={{ tensor.name_expr }},
            spec=TensorSpec(dtype={{ tensor.dtype_source }}, shape={{ tensor.shape_source }}),
            role={{ tensor.role }},
            memory={{ tensor.memory }},
            lifetime={{ tensor.lifetime }},
            request_state={{ tensor.name_source }} in request_state_outputs,
{% for extra_line in tensor.extra_lines %}
            {{ extra_line }}
{% endfor %}
            ),
        ),
{% endfor %}
    )
"""

_TENSOR_MODULE_TEMPLATE = '''"""Generated tensor declarations."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from torch2vk.runtime.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryClass,
    PyTorchProbe,
    TensorLifetime,
    TensorRole,
)
from torch2vk.vulkan.types import TensorSpec


{{ class_sources_source }}


{{ helper_source }}'''

_TENSOR_HELPERS_TEMPLATE = '''def _declare_tensor(
    *,
    name: str,
    spec: TensorSpec,
    role: TensorRole,
    memory: MemoryClass,
    lifetime: TensorLifetime,
    request_state: bool = False,
    compare: ComparePolicy | None = None,
    pytorch_probe: PyTorchProbe | None = None,
) -> LogicalTensor:
    if request_state:
        role = TensorRole.OUTPUT
        memory = MemoryClass.REQUEST_STATE
        lifetime = TensorLifetime.REQUEST
    return LogicalTensor(
        name=name,
        spec=spec,
        role=role,
        memory=memory,
        lifetime=lifetime,
        compare=compare,
        pytorch_probe=pytorch_probe,
    )


def _bind_tensor(
    bindings: Mapping[str, LogicalTensor] | None,
    field: str,
    tensor: LogicalTensor,
) -> LogicalTensor:
    if bindings is None:
        return tensor
    bound = bindings.get(field)
    if bound is None:
        return tensor
    if bound.spec != tensor.spec:
        raise ValueError(f"{field} binding spec {bound.spec} does not match {tensor.spec}")
    return bound


def _validate_bindings(
    bindings: Mapping[str, LogicalTensor] | None,
    tensor_names: frozenset[str],
) -> None:
    if bindings is None:
        return
    unknown = frozenset(bindings) - tensor_names
    if unknown:
        raise ValueError(f"unknown tensor bindings: {sorted(unknown)}")


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
'''

_SIMPLE_INIT_TEMPLATE = '''"""{{ docstring }}."""

{{ imports_source }}
'''


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

    return _render_template(
        _SHADER_FILE_TEMPLATE,
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


def render_dispatch_function(function_name: str, class_name: str, ops) -> str:
    return _render_template(
        _DISPATCH_FUNCTION_TEMPLATE,
        function_name=function_name,
        class_name=class_name,
        ops=ops,
    ).rstrip("\n")


def render_dispatch_file(
    *,
    docstring: str,
    shader_package: str,
    tensor_package: str,
    shader_imports,
    tensor_imports,
    function_sources: list[str],
    extra_imports_source: str = "",
    extra_shader_variants=(),
) -> str:
    return _render_template(
        _DISPATCH_FILE_TEMPLATE,
        docstring=docstring,
        shader_package=shader_package,
        tensor_package=tensor_package,
        shader_imports=shader_imports,
        tensor_imports=tensor_imports,
        extra_imports_source=extra_imports_source.rstrip("\n"),
        extra_shader_variants=extra_shader_variants,
        function_sources_source="\n\n\n".join(function_sources),
    )


def render_tensor_class(
    *,
    class_name: str,
    fields,
    output_const: str,
    output_name_source: str,
    signature: str,
    tensor_names_source: str,
    tensors,
) -> str:
    return _render_template(
        _TENSOR_CLASS_TEMPLATE,
        class_name=class_name,
        fields=fields,
        output_const=output_const,
        output_name_source=output_name_source,
        signature=signature,
        tensor_names_source=tensor_names_source,
        tensors=tensors,
    ).rstrip("\n")


def render_tensor_module(class_sources: list[str], helper_source: str) -> str:
    return _render_template(
        _TENSOR_MODULE_TEMPLATE,
        class_sources_source="\n\n\n".join(class_sources),
        helper_source=helper_source,
    )


def render_tensor_helpers() -> str:
    return _render_template(_TENSOR_HELPERS_TEMPLATE)


def render_simple_init(docstring: str, imports: list[str]) -> str:
    return _render_template(
        _SIMPLE_INIT_TEMPLATE,
        docstring=docstring,
        imports_source="\n".join(imports),
    )


def _render_template(source: str, **context) -> str:
    return _JINJA.from_string(source).render(**context)


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


def generate_dispatch_source(
    prog: ExportedProgram,
    *,
    class_name: str = "ExportedTensors",
    function_name: str = "run_exported",
    registry: ShaderRegistry = DEFAULT_REGISTRY,
) -> str:
    graph = prog.graph_module.graph
    sig = prog.graph_signature

    node_variants = _resolve_all_variants(graph, registry)

    lines: list[str] = []
    lines.append('"""Generated by torch2vk.export. Do not edit."""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dataclasses import dataclass")
    lines.append("")
    lines.append("from torch2vk.runtime.logical import LogicalTensor")
    lines.append("from torch2vk.runtime.session import RuntimeSession")
    lines.append("from torch2vk.runtime.shader import ShaderVariant")
    lines.append("")
    lines.append("")
    lines.extend(_generate_param_fields(sig))
    lines.append("")
    lines.append("")
    lines.extend(_generate_dataclass(graph, sig, class_name))
    lines.append("")
    lines.append("")
    lines.extend(_generate_dispatch_function(graph, class_name, function_name, node_variants))
    lines.append("")

    return "\n".join(lines)


def _resolve_all_variants(graph: Graph, registry: ShaderRegistry) -> dict[str, ShaderVariant]:
    variants: dict[str, ShaderVariant] = {}
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target in SKIP_OPS or is_alias_op(node):
            continue
        shader = registry.resolve(node)
        if shader is not None:
            variants[node.name] = shader
    return variants


def _generate_param_fields(sig: ExportGraphSignature) -> list[str]:
    lines: list[str] = []
    lines.append("PARAMETER_FIELDS = {")
    for spec in sig.input_specs:
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            lines.append(f'    "{spec.arg.name}": "{spec.target}",')
    lines.append("}")
    return lines


def _generate_dataclass(graph: Graph, sig: ExportGraphSignature, class_name: str) -> list[str]:
    lines: list[str] = []
    lines.append("@dataclass")
    lines.append(f"class {class_name}:")

    seen: set[str] = set()
    for spec in sig.input_specs:
        name = spec.arg.name
        if name not in seen:
            lines.append(f"    {name}: LogicalTensor")
            seen.add(name)

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if str(node.target) in SKIP_OPS:
            continue
        if node.name not in seen:
            lines.append(f"    {node.name}: LogicalTensor")
            seen.add(node.name)

    return lines


def _generate_dispatch_function(
    graph: Graph,
    class_name: str,
    function_name: str,
    node_variants: dict[str, ShaderVariant],
) -> list[str]:
    lines: list[str] = []
    lines.append(f"def {function_name}(rt: RuntimeSession, tensors: {class_name}, shaders: dict[str, ShaderVariant]) -> None:")

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target in SKIP_OPS:
            continue

        if is_alias_op(node):
            inputs = node_input_names(node)
            src = inputs[0] if inputs else "???"
            lines.append(f"    _alias(rt, tensors.{src}, tensors.{node.name})")
            continue

        shader = node_variants.get(node.name)
        if shader is None:
            message = f"unsupported exported op {target} ({node.name})"
            lines.append(f"    raise RuntimeError({message!r})")
            continue

        args = _build_shader_args(node, shader)
        lines.append(f'    shaders["{node.name}"](rt, {args})')

    lines.append("")
    lines.append("")
    lines.append("def _alias(rt: RuntimeSession, src: LogicalTensor, dst: LogicalTensor) -> None:")
    lines.append("    rt._materialize_read(src)")
    lines.append("    with dst.runtime_write_scope():")
    lines.append("        dst.buffer = src.buffer")
    lines.append("        dst.descriptor_nbytes = src.descriptor_nbytes")
    lines.append("        dst.version = src.version")
    lines.append("        dst.writer = src.writer")
    lines.append("    rt._current_frame().written_tensors.append(dst)")

    return lines


def _build_shader_args(node: Node, shader: ShaderVariant) -> str:
    contract = shader.contract
    input_fields = [f for f in contract.fields if f.io_kind in (IOKind.INPUT, IOKind.INOUT)]
    output_fields = [f for f in contract.fields if f.io_kind in (IOKind.OUTPUT, IOKind.INOUT)]

    inputs = node_input_names(node)
    parts: list[str] = []
    for i, field in enumerate(input_fields):
        if i < len(inputs):
            parts.append(f"{field.name}=tensors.{inputs[i]}")
    for i, field in enumerate(output_fields):
        if i == 0:
            parts.append(f"{field.name}=tensors.{node.name}")

    return ", ".join(parts)
