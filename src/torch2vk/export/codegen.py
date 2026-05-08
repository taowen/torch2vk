"""Generate Python dispatch code directly from torch.export.ExportedProgram."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

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
    ShaderContract,
    ShaderVariant,
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


class _TensorKind(Enum):
    PARAMETER = "parameter"
    USER_INPUT = "user_input"
    INTERMEDIATE = "intermediate"


@dataclass(frozen=True, slots=True)
class _TensorMeta:
    shape: tuple[int, ...]
    dtype: str
    kind: _TensorKind


@dataclass(frozen=True, slots=True)
class _AliasOp:
    src: str
    dst: str


@dataclass(frozen=True, slots=True)
class _DispatchOp:
    name: str
    variant: ShaderVariant
    bindings: dict[str, str]


@dataclass(frozen=True, slots=True)
class _UnsupportedOp:
    name: str
    message: str


_Op = _AliasOp | _DispatchOp | _UnsupportedOp


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

import sys
from typing import cast

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


def shader_variant(shader_name: str) -> ShaderVariant:
    return cast(ShaderVariant, getattr(sys.modules[__name__], shader_name.upper()))


{{ function_sources_source }}


def _alias(rt: RuntimeSession, src: LogicalTensor, dst: LogicalTensor) -> None:
    rt._materialize_read(src)
    with dst.runtime_write_scope():
        dst.buffer = src.buffer
        dst.descriptor_nbytes = src.descriptor_nbytes
        dst.version = src.version
        dst.writer = src.writer
    frame = rt._current_frame()
    frame.used_tensors.append(src)
    frame.written_tensors.append(dst)
'''

_TENSOR_CLASS_TEMPLATE = """@dataclass(frozen=True, slots=True)
class {{ class_name }}:
{% for field in fields %}
    {{ field }}: LogicalTensor
{% endfor %}


{{ output_const }}: str = {{ output_name_source }}


{{ signature }}
    _validate_request_state_outputs(request_state_outputs, frozenset(({{ output_name_source }},)))
    return {{ class_name }}(
{% for tensor in tensors %}
        {{ tensor.name }}=_bind_tensor(
            {{ tensor.name }},
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

from collections.abc import Collection
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
    bound: LogicalTensor | None,
    tensor: LogicalTensor,
) -> LogicalTensor:
    if bound is None:
        return tensor
    if bound.spec != tensor.spec:
        raise ValueError(f"{bound.name} spec {bound.spec} does not match {tensor.name} spec {tensor.spec}")
    return bound


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
) -> str:
    return _render_template(
        _DISPATCH_FILE_TEMPLATE,
        docstring=docstring,
        shader_package=shader_package,
        tensor_package=tensor_package,
        shader_imports=shader_imports,
        tensor_imports=tensor_imports,
        extra_imports_source=extra_imports_source.rstrip("\n"),
        function_sources_source="\n\n\n".join(function_sources),
    )


def render_tensor_class(
    *,
    class_name: str,
    fields,
    output_const: str,
    output_name_source: str,
    signature: str,
    tensors,
) -> str:
    return _render_template(
        _TENSOR_CLASS_TEMPLATE,
        class_name=class_name,
        fields=fields,
        output_const=output_const,
        output_name_source=output_name_source,
        signature=signature,
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


def _tensor_factory_signature(
    function_name: str,
    class_name: str,
    *,
    fields: tuple[str, ...],
    layered: bool,
) -> str:
    params = ["prefix: str"]
    if layered:
        params.append("layer_idx: int")
    params.append("*")
    params.extend(f"{field}: LogicalTensor | None = None" for field in fields)
    params.append("request_state_outputs: Collection[str] = frozenset()")
    return f"def {function_name}(\n    " + ",\n    ".join(params) + f",\n) -> {class_name}:"


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


def _generate_param_fields(sig: ExportGraphSignature, weight_prefix: str = "") -> list[str]:
    lines: list[str] = []
    lines.append("PARAMETER_FIELDS = {")
    for spec in sig.input_specs:
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            lines.append(f'    "{spec.arg.name}": "{weight_prefix}{spec.target}",')
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
    lines.append("    frame = rt._current_frame()")
    lines.append("    frame.used_tensors.append(src)")
    lines.append("    frame.written_tensors.append(dst)")

    return lines


def _build_shader_args(node: Node, shader: ShaderVariant) -> str:
    contract = shader.contract
    input_fields = [f for f in contract.fields if f.io_kind in (IOKind.INPUT, IOKind.INOUT)]
    output_fields = [f for f in contract.fields if f.io_kind in (IOKind.OUTPUT, IOKind.INOUT)]

    inputs = node_input_names(node)
    bindings: dict[str, str] = {}
    for i, field in enumerate(input_fields):
        if i < len(inputs):
            bindings[field.name] = f"tensors.{inputs[i]}"
    for field in output_fields:
        bindings[field.name] = f"tensors.{node.name}"

    return ", ".join(f"{k}={v}" for k, v in bindings.items())


def _generate_static_dispatch_function(
    graph: Graph,
    class_name: str,
    function_name: str,
    node_variants: dict[str, ShaderVariant],
) -> tuple[list[str], dict[str, str]]:
    ops = _collect_ops(graph, node_variants)
    output_names = _find_graph_outputs(graph)
    ops = _prune_dead_ops(ops, output_names)

    unsupported = [op for op in ops if isinstance(op, _UnsupportedOp)]
    if unsupported:
        details = "\n".join(f"  - {op.message}" for op in unsupported)
        raise RuntimeError(
            f"{function_name} contains unsupported ops:\n{details}"
        )

    shader_imports: dict[str, str] = {}
    lines: list[str] = []
    lines.append(f"def {function_name}(rt: RuntimeSession, tensors: {class_name}) -> None:")

    for op in ops:
        if isinstance(op, _AliasOp):
            lines.append(f"    _alias(rt, tensors.{op.src}, tensors.{op.dst})")
        elif isinstance(op, _DispatchOp):
            const_name = op.variant.name.upper()
            shader_imports[op.variant.name] = const_name
            args = ", ".join(f"{k}={v}" for k, v in op.bindings.items())
            lines.append(f"    {const_name}(rt, {args})")

    return lines, shader_imports


def _collect_ops(graph: Graph, node_variants: dict[str, ShaderVariant]) -> list[_Op]:
    ops: list[_Op] = []
    seen_variants: dict[str, ShaderVariant] = {}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target in SKIP_OPS:
            continue

        if is_alias_op(node):
            inputs = node_input_names(node)
            src = inputs[0] if inputs else "???"
            ops.append(_AliasOp(src=src, dst=node.name))
            continue

        shader = node_variants.get(node.name)
        if shader is None:
            ops.append(_UnsupportedOp(name=node.name, message=f"unsupported exported op {target} ({node.name})"))
            continue

        shader = _dedup_variant(shader, seen_variants)

        input_fields = [f for f in shader.contract.fields if f.io_kind in (IOKind.INPUT, IOKind.INOUT)]
        output_fields = [f for f in shader.contract.fields if f.io_kind in (IOKind.OUTPUT, IOKind.INOUT)]
        inputs = node_input_names(node)
        bindings: dict[str, str] = {}
        for i, field in enumerate(input_fields):
            if i < len(inputs):
                bindings[field.name] = f"tensors.{inputs[i]}"
        for field in output_fields:
            bindings[field.name] = f"tensors.{node.name}"
        ops.append(_DispatchOp(name=node.name, variant=shader, bindings=bindings))

    return ops


def _dedup_variant(
    shader: ShaderVariant,
    seen_variants: dict[str, ShaderVariant],
) -> ShaderVariant:
    shader_key = shader.name
    if shader_key in seen_variants:
        if seen_variants[shader_key].contract != shader.contract:
            shader_key = f"{shader.name}_{len(seen_variants)}"
            new_contract = ShaderContract(
                class_name=shader.contract.class_name,
                shader_name=shader_key,
                fields=shader.contract.fields,
                dispatch=shader.contract.dispatch,
                push_constants=shader.contract.push_constants,
                params_buffer=shader.contract.params_buffer,
            )
            shader = ShaderVariant(
                name=shader_key, family=shader.family, contract=new_contract,
                source=shader.source, precompiled_spv_path=shader.precompiled_spv_path,
                specialization_constants=shader.specialization_constants,
                include_dirs=shader.include_dirs, compile_defines=shader.compile_defines,
                execution_requirements=shader.execution_requirements,
            )
    if shader_key not in seen_variants:
        seen_variants[shader_key] = shader
    return shader


def _find_graph_outputs(graph: Graph) -> list[str]:
    names: list[str] = []
    for node in graph.nodes:
        if node.op == "output":
            _collect_output_node_names(node.args, names)
    return names


def _collect_output_node_names(value, names: list[str]) -> None:
    if isinstance(value, Node):
        names.append(value.name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_output_node_names(item, names)


def _prune_dead_ops(ops: list[_Op], output_names: list[str]) -> list[_Op]:
    needed = set(output_names)
    kept_reversed: list[_Op] = []
    for op in reversed(ops):
        if isinstance(op, _DispatchOp):
            if op.name not in needed:
                continue
            kept_reversed.append(op)
            for v in op.bindings.values():
                needed.add(v.removeprefix("tensors."))
        elif isinstance(op, _AliasOp):
            if op.dst not in needed:
                continue
            kept_reversed.append(op)
            needed.add(op.src)
        elif isinstance(op, _UnsupportedOp):
            if op.name not in needed:
                continue
            kept_reversed.append(op)
    return list(reversed(kept_reversed))


# ==============================================================
# Public API: split generation
# ==============================================================



def generate_tensor_class_source(
    prog: ExportedProgram,
    *,
    class_name: str = "ExportedTensors",
    function_name: str = "create_exported",
    weight_prefix: str = "",
    is_layered: bool | None = None,
    extra_lines_fn: Callable[[str], tuple[str, ...]] | None = None,
    registry: ShaderRegistry = DEFAULT_REGISTRY,
) -> str:
    """Generate tensor dataclass + factory function for a single submodule.

    Args:
        prog: Exported program to analyze.
        class_name: Name for the generated dataclass.
        function_name: Name for the factory function (e.g., "create_conv2d1").
        weight_prefix: Prefix for safetensors weight keys.
        is_layered: Whether param names contain .layers.N. patterns.
            None = auto-detect.
        extra_lines_fn: Optional callback(tensor_name) → extra source lines
            for the tensor declaration (e.g., compare policy).
        registry: Shader registry for resolving supported ops.
    """
    graph = prog.graph_module.graph
    sig = prog.graph_signature

    tensors: dict[str, _TensorMeta] = {}
    user_inputs: list[str] = []
    param_map: dict[str, str] = {}

    for spec in sig.input_specs:
        for node in graph.nodes:
            if node.name == spec.arg.name:
                tm = node.meta.get("tensor_meta")
                if tm:
                    shape = tuple(int(d) for d in tm.shape)
                    dtype = str(tm.dtype).removeprefix("torch.")
                    is_param = spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
                    tensors[spec.arg.name] = _TensorMeta(
                        shape=shape, dtype=dtype,
                        kind=_TensorKind.PARAMETER if is_param else _TensorKind.USER_INPUT,
                    )
                    if is_param:
                        param_map[spec.arg.name] = f"{weight_prefix}{spec.target}"
                    else:
                        user_inputs.append(spec.arg.name)
                break

    for node in graph.nodes:
        if node.op == "call_function" and node.name not in tensors:
            if str(node.target) in SKIP_OPS:
                continue
            tm = node.meta.get("tensor_meta")
            if tm:
                shape = tuple(int(d) for d in tm.shape)
                dtype = str(tm.dtype).removeprefix("torch.")
                tensors[node.name] = _TensorMeta(shape=shape, dtype=dtype, kind=_TensorKind.INTERMEDIATE)

    output_name = _find_output_name(graph, tensors)

    # Prune dead tensors
    node_variants = _resolve_all_variants(graph, registry)
    ops = _collect_ops(graph, node_variants)
    output_names = _find_graph_outputs(graph)
    if not output_names:
        output_names = [output_name] if output_name else []
    live_ops = _prune_dead_ops(ops, output_names)
    live_tensors = set(output_names)
    for op in live_ops:
        if isinstance(op, _DispatchOp):
            for v in op.bindings.values():
                live_tensors.add(v.removeprefix("tensors."))
        elif isinstance(op, _AliasOp):
            live_tensors.add(op.src)
            live_tensors.add(op.dst)
    live_tensors.update(name for name in user_inputs if name in live_tensors)
    live_tensors.update(param_map.keys() & live_tensors)
    tensors = {k: v for k, v in tensors.items() if k in live_tensors}
    user_inputs = [n for n in user_inputs if n in live_tensors]
    param_map = {k: v for k, v in param_map.items() if k in live_tensors}

    if is_layered is None:
        is_layered = any(re.search(r"\.layers\.\d+\.", v) for v in param_map.values())

    tensor_entries = []
    for name, meta in tensors.items():
        shape = meta.shape
        kind = meta.kind
        dtype = "bfloat16" if kind == _TensorKind.PARAMETER else (
            meta.dtype if meta.dtype in ("int64", "int32") else "float32"
        )
        if kind == _TensorKind.PARAMETER:
            role = "TensorRole.WEIGHT"
            memory = "MemoryClass.MODEL_WEIGHT"
            lifetime = "TensorLifetime.MODEL"
            safetensors_key = param_map[name]
            if is_layered:
                name_template = re.sub(r"\.layers\.(\d+)\.", ".layers.{layer_idx}.", safetensors_key)
                name_expr = f'f"{name_template}"'
            else:
                name_expr = f'"{safetensors_key}"'
        elif kind == _TensorKind.USER_INPUT:
            role = "TensorRole.INPUT"
            memory = "MemoryClass.HOST_INPUT"
            lifetime = "TensorLifetime.FRAME"
            name_expr = f'f"{{prefix}}.{name}"'
        else:
            role = "TensorRole.ACTIVATION"
            memory = "MemoryClass.FRAME_WORKSPACE"
            lifetime = "TensorLifetime.FRAME"
            name_expr = f'f"{{prefix}}.{name}"'

        extra = extra_lines_fn(name) if extra_lines_fn else ()

        tensor_entries.append({
            "name": name,
            "name_source": repr(name),
            "name_expr": name_expr,
            "dtype_source": repr(dtype),
            "shape_source": repr(shape),
            "role": role,
            "memory": memory,
            "lifetime": lifetime,
            "extra_lines": extra,
        })

    output_const = function_name.removeprefix("create_").upper() + "_OUTPUT"
    fields = tuple(tensors.keys())
    return render_tensor_class(
        class_name=class_name,
        fields=fields,
        output_const=output_const,
        output_name_source=repr(output_name),
        signature=_tensor_factory_signature(
            function_name,
            class_name,
            fields=fields,
            layered=is_layered,
        ),
        tensors=tensor_entries,
    )


def _find_output_name(graph: Graph, tensors: dict[str, _TensorMeta]) -> str | None:
    for node in graph.nodes:
        if node.op == "output":
            names = _collect_graph_output_names(node.args, tensors)
            if names:
                return names[0]
    return None


def _collect_graph_output_names(value: object, tensors: dict[str, _TensorMeta]) -> list[str]:
    names: list[str] = []
    if isinstance(value, Node):
        if value.name in tensors:
            names.append(value.name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            names.extend(_collect_graph_output_names(item, tensors))
    return names


def generate_dispatch_function_source(
    prog: ExportedProgram,
    *,
    class_name: str = "ExportedTensors",
    function_name: str = "run_exported",
    shader_package: str = "",
    registry: ShaderRegistry = DEFAULT_REGISTRY,
) -> tuple[str, dict[str, str], dict[str, ShaderVariant]]:
    """Generate dispatch function with static shader imports.

    Returns (function_source, {shader_name: CONST_NAME}, {shader_name: ShaderVariant}).
    The caller is responsible for assembling the final file with imports.
    """
    graph = prog.graph_module.graph
    node_variants = _resolve_all_variants(graph, registry)
    lines, shader_imports = _generate_static_dispatch_function(
        graph, class_name, function_name, node_variants,
    )
    # Collect actual ShaderVariant objects used (after pruning + dedup)
    all_ops = _collect_ops(graph, node_variants)
    output_names = _find_graph_outputs(graph)
    live_ops = _prune_dead_ops(all_ops, output_names)
    used_variants: dict[str, ShaderVariant] = {}
    for op in live_ops:
        if isinstance(op, _DispatchOp):
            used_variants.setdefault(op.variant.name, op.variant)
    return "\n".join(lines), shader_imports, used_variants
