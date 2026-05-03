# ShaderContract 设计说明

本文用 Python sketch 的形式定义 `torch2vk` 的 shader contract。它是文档，不是当前
runtime 的实现文件。

参考 `agentorch` 后，原有 docs 里已经清楚的部分是：

```text
LogicalTensor 声明模型里的点。
ShaderContract 声明 shader 读写哪些点，以及 Vulkan binding / dtype / shape / layout / dispatch。
RuntimeSession.dispatch() 根据 contract 做校验、materialize、descriptor 绑定、dispatch 和记录。
```

需要补清楚的细节是：

1. contract 是 shader ABI，不是模型语义，也不是显存分配计划；
2. tensor field 直接对应 GLSL input/output，Vulkan binding index 是 field 的 ABI metadata；
3. shape / dispatch / uniform / push constant 都需要同一套表达式系统；
4. shape symbol 必须有明确来源：tensor shape、tensor layout metadata 或 runtime resolver；
5. push constants 必须声明 byte size、field dtype、offset、value resolver，并做重叠和范围校验；
6. uniform buffer 可以由 runtime 从 shape symbols 自动 materialize；
7. descriptor 绑定的是 `LogicalTensor` 当前记录的 buffer slice、offset、range；
8. shader variant 还要携带 specialization constants、compile defines、include dirs 和 device capability requirements；
9. contract manifest 应该能落盘，作为 shader artifact 和 replay/debug 的稳定 ABI 摘要。

## 核心边界

`ShaderContract` 只回答这个 shader 的 ABI 问题：

```text
这个 shader 有哪些 tensor fields？
每个 field 是 read/write/read_write？
每个 field 对应 GLSL 哪个 input/output，以及它的 Vulkan binding index 是多少？
每个 tensor 的 dtype/shape/layout 约束是什么？
哪些 uniform buffer 由 runtime 自动生成？
哪些 push constants 需要打包，值从哪里来？
dispatch group count 如何从 symbol 解析？
这个 variant 需要哪些 Vulkan feature / specialization constants？
```

它不回答：

```text
LogicalTensor 从哪个 checkpoint 来。
activation 放在哪个 arena。
Frame 结束后释放什么。
和 PyTorch 比哪些 tensor。
Replay 怎么安排 storage aliasing。
```

这些仍然属于 `LogicalTensor` metadata 和 `RuntimeSession`。

## 推荐 API sketch

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence, TypeAlias


ShapeDim: TypeAlias = int | str


@dataclass(frozen=True, slots=True)
class CeilDivExpr:
    lhs: ExprDim
    rhs: ExprDim


@dataclass(frozen=True, slots=True)
class MulExpr:
    lhs: ExprDim
    rhs: ExprDim


@dataclass(frozen=True, slots=True)
class AddExpr:
    lhs: ExprDim
    rhs: ExprDim


ExprDim: TypeAlias = ShapeDim | CeilDivExpr | MulExpr | AddExpr


def ceil_div(lhs: ExprDim, rhs: ExprDim) -> CeilDivExpr:
    return CeilDivExpr(lhs=lhs, rhs=rhs)


def mul(lhs: ExprDim, rhs: ExprDim) -> MulExpr:
    return MulExpr(lhs=lhs, rhs=rhs)


def add(lhs: ExprDim, rhs: ExprDim) -> AddExpr:
    return AddExpr(lhs=lhs, rhs=rhs)


```

`ExprDim` 是 contract 的公共表达式语言。它可以出现在：

1. `TensorFieldSpec.shape`
2. `TensorLayout.params`
3. `UniformFieldSpec.value`
4. `PushConstantFieldSpec.value`
5. `ShaderContract.dispatch`

MVP 只需要 `int`、`str`、`ceil_div`、`mul`、`add`。不要在第一版引入任意 Python 表达式字符串。
如果确实需要复杂逻辑，放进显式 resolver callable，并把它标为 runtime-derived。

```python
class DescriptorType(StrEnum):
    STORAGE_BUFFER = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    UNIFORM_BUFFER = "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER"


class IOKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class PushConstantType(StrEnum):
    UINT32 = "uint32"
    INT32 = "int32"
    FLOAT32 = "float32"
    UINT64 = "uint64"


class UniformKind(StrEnum):
    IVEC4 = "ivec4"


@dataclass(frozen=True, slots=True)
class TensorLayout:
    name: str = "row_major"
    params: Mapping[str, ExprDim] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DTypeReference:
    field_name: str


def same_as(field_name: str) -> DTypeReference:
    return DTypeReference(field_name=field_name)


@dataclass(frozen=True, slots=True)
class TensorContract:
    dtype: str | tuple[str, ...] | DTypeReference
    shape: tuple[ExprDim, ...]
    layout: TensorLayout = TensorLayout()


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    name: str
    io_kind: IOKind
    binding: int
    role: str
    contract: TensorContract
    descriptor_type: DescriptorType = DescriptorType.STORAGE_BUFFER


@dataclass(frozen=True, slots=True)
class UniformFieldSpec:
    name: str
    binding: int
    kind: UniformKind
    value: tuple[ExprDim, ExprDim, ExprDim, ExprDim]


@dataclass(frozen=True, slots=True)
class PushConstantInput:
    name: str


PushConstantResolver: TypeAlias = Callable[
    [Mapping[str, object], Mapping[str, int]],
    int | float,
]
PushConstantValue: TypeAlias = (
    ExprDim | PushConstantInput | PushConstantResolver | int | float
)


@dataclass(frozen=True, slots=True)
class PushConstantFieldSpec:
    name: str
    dtype: PushConstantType
    offset: int
    value: PushConstantValue

    @property
    def size(self) -> int:
        return 8 if self.dtype is PushConstantType.UINT64 else 4


@dataclass(frozen=True, slots=True)
class PushConstantSpec:
    size: int
    fields: tuple[PushConstantFieldSpec, ...]


@dataclass(frozen=True, slots=True)
class ShaderContract:
    class_name: str
    shader_name: str
    fields: tuple[TensorFieldSpec, ...]
    uniforms: tuple[UniformFieldSpec, ...]
    push_constants: PushConstantSpec | None
    dispatch: tuple[ExprDim, ExprDim, ExprDim]

    @property
    def input_fields(self) -> tuple[TensorFieldSpec, ...]:
        return tuple(
            field
            for field in self.fields
            if field.io_kind in (IOKind.INPUT, IOKind.INOUT)
        )

    @property
    def output_fields(self) -> tuple[TensorFieldSpec, ...]:
        return tuple(
            field
            for field in self.fields
            if field.io_kind in (IOKind.OUTPUT, IOKind.INOUT)
        )


```

## Field 和 LogicalTensor 对应

`TensorFieldSpec.name` 是 shader 的 logical interface field，也是 `ShaderVariant(rt, **arguments)`
接收的参数名。模型代码传入的 `LogicalTensor` 必须精确对应这个 GLSL input/output：

```python
CONV1D_F32(rt, x=tensors.input, weight=tensors.weight, output=tensors.output)
```

这里的 `x`、`weight`、`output` 不是另一套 graph edge 名字，而是 shader field 名字。它们和
GLSL 里的 storage buffer interface 一一对应。`binding` 只是 Vulkan descriptor binding index，
是这个 field 的 ABI metadata，不是模型 adapter 需要理解或传递的对象。

推荐：

```python
fields=(
    TensorFieldSpec(name="x", io_kind=IOKind.INPUT, binding=0, role="x", ...),
    TensorFieldSpec(name="weight", io_kind=IOKind.INPUT, binding=1, role="weight", ...),
    TensorFieldSpec(name="output", io_kind=IOKind.OUTPUT, binding=2, role="output", ...),
)
```

对应 GLSL：

```glsl
layout(set = 0, binding = 0) buffer restrict readonly X { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly W { float weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly O { float output[]; };
```

如果某个 shader 需要的 view 无法精确对应现有 `LogicalTensor`，优先给 `LogicalTensor` 增加 metadata，
例如 layout params、view range、semantic、source/feed、state/lifetime、binding span 或 packed layout
信息。不要在 shader contract 里制造第二套 tensor 语义来补洞。

校验规则：

1. `field.name` 唯一且非空；
2. `field.binding` index 唯一；
3. tensor field 的 `descriptor_type` 必须是 storage buffer；
4. uniform field name 唯一且非空；
5. uniform field 的 binding index 不能和 tensor field 冲突；
6. `field.name` 必须能在 dispatch arguments 中找到对应 `LogicalTensor`；
7. `field.name` 应和 GLSL interface 名保持一致；
8. GLSL 中 `layout(set = 0, binding = N)` 必须和 `field.binding` 一致；
9. runtime 绑定 descriptor 时使用 `field.binding`，但模型 adapter 不接触这个值。

MVP 可以先靠人工 review GLSL binding；后续应在 shader build 阶段解析 GLSL 或 SPIR-V reflection
做一致性检查。

## IOKind

`IOKind` 是 shader ABI 里的读写权限，不是 `TensorRole`。

```text
INPUT   shader 只读；RuntimeSession resolve/materialize input LogicalTensor。
OUTPUT  shader 只写；RuntimeSession reserve/write output LogicalTensor，并推进它的 version。
INOUT   shader 读写同一个 descriptor；必须显式声明，不能靠同名 input/output 猜。
```

`INOUT` 的规则：

1. contract 层允许同一个 field 同时出现在 `input_fields` 和 `output_fields`；
2. runtime 必须记录 read 和 write；
3. write 后的 `LogicalTensor.version` 必须前进；
4. 是否允许物理 in-place 由 contract 的 `INOUT` 和 runtime alias policy 共同决定；
5. 普通 `INPUT` 和 `OUTPUT` 默认不允许传同一个 writable tensor view，除非 contract 显式声明 alias。

## Shape Symbol 解析

symbol 的来源只能有三类：

```text
tensor shape
  从传入 LogicalTensor 的 concrete shape 绑定，例如 B、T、C。

tensor layout metadata
  从 layout params 绑定，例如 head_count、head_dim、logical_k、stride。

runtime symbol resolver
  从 device 或 frame runtime context 绑定，例如 subgroup_size、aligned_block_count。
```

禁止出现无来源 symbol。校验时应计算：

```text
tensor_shape_symbols
tensor_layout_symbols
contract_referenced_symbols
runtime_shape_symbols
```

并要求：

```text
contract_referenced_symbols <= tensor_shape_symbols | tensor_layout_symbols | runtime_shape_symbols
runtime_shape_symbols 不允许 shadow tensor-derived symbols
runtime resolver 返回值必须刚好覆盖声明的 runtime_shape_symbols，不多不少
```

shape 绑定规则：

1. rank 必须相等；
2. concrete dim 必须相等；
3. 第一次看到 symbol 时绑定 actual dim；
4. 后续看到同名 symbol 必须等于已绑定值；
5. expression dim 必须在依赖 symbol 已解析后求值；
6. dispatch 前所有 symbol 都必须解析成 int。

这点比原有 docs 更严格：`("B", "T", "C")` 不是注释，它是运行时 contract 校验输入。

## DType 规则

`TensorContract.dtype` 支持三种形式：

```python
"float32"
("float16", "bfloat16")
same_as("x")
```

规则：

1. input field 可以允许多个 dtype，用于同一个 shader family 的有限 ABI 兼容；
2. output field 在 dispatch 前必须解析到一个 concrete dtype；
3. `same_as("x")` 只能引用已声明 field；
4. 不允许 silent cast；
5. weight dtype、LogicalTensor dtype、shader contract dtype 必须一致，除非 shader 明确是 cast/copy kernel。

## Layout 规则

layout 是 shader ABI 的一部分，不是 debug metadata。

MVP 可以先支持：

```python
TensorLayout("row_major")
TensorLayout("strided", {"stride_b": ..., "stride_t": ..., "stride_c": ...})
TensorLayout("width_major_tokens")
TensorLayout("qkv_packed_channels", {"key_dim": "KD", "value_dim": "VD", "kernel": 3})
TensorLayout("q4_k_words", {"logical_k": "K", "block_size": 256, "words_per_block": 36})
```

规则：

1. contract layout 与 actual tensor layout 必须匹配；
2. layout params 中的 symbol 参与 shape symbol 解析；
3. packed layout 必须校验 shape 与 layout params 一致；
4. layout transform 必须用显式 shader 或 compare transform 表达，不允许在 dispatch 里 silent reinterpret；
5. descriptor range 必须覆盖 layout 访问 span，而不只是 `numel * dtype_nbytes`。

## Uniform Buffer

uniform buffer 适合小的 shape-derived 参数，例如 `ivec4(SI, SO, IC, OC)`。

```python
uniforms=(
    UniformFieldSpec(
        name="sizes",
        binding=5,
        kind=UniformKind.IVEC4,
        value=("SI", "SO", "IC", "OC"),
    ),
)
```

RuntimeSession 负责：

1. resolve `value` 中的 ExprDim；
2. materialize 一个 16-byte host-visible uniform buffer；
3. 写入 `<4i`；
4. 把它追加到 descriptor buffers；
5. dispatch 完释放这个 runtime-owned uniform allocation，或从小 uniform pool 复用。

uniform buffer 是 shader ABI 的一部分，但不是模型 tensor，不进入 `LogicalTensor` tree。

## Push Constants

push constant 必须显式声明 byte layout：

```python
PushConstantSpec(
    size=80,
    fields=(
        PushConstantFieldSpec("dst_addr", UINT64, 0, device_address("output")),
        PushConstantFieldSpec("IC", UINT32, 16, "IC"),
        PushConstantFieldSpec("stride", INT32, 52, 2),
    ),
)
```

field value 来源：

```text
literal int/float
shape symbol / ExprDim
PushConstantInput("name")，由 `ShaderVariant` 调用方显式传入
resolver callable(tensors, shape_symbols)，用于 device address 这类 runtime-only 值
```

校验规则：

1. `PushConstantSpec.size >= 0`；
2. field name 唯一且非空；
3. offset 非负；
4. `offset + field.size <= size`；
5. fields 不能 byte overlap；
6. provided push constants 不能多也不能少；
7. integer field 必须是 int，float32 field 接受 int/float；
8. uint32/uint64/int32 必须做范围检查；
9. pack 使用固定 little-endian；
10. `ShaderContract.push_constants.size` 是 pipeline layout key 的一部分。

不要把 push constant 当成随手传的 dict。它是 ABI，byte offset 错了就等价于 shader 签名错了。

## Descriptor View 和 Range

RuntimeSession materialize 后，`LogicalTensor` 当前状态必须包含 descriptor 所需的 buffer view：

```text
allocation
byte_offset
nbytes
descriptor_nbytes
dtype
shape
layout
```

descriptor 绑定时必须校验：

```text
descriptor offset == tensor view byte_offset
descriptor range == tensor view descriptor_nbytes
descriptor range 不超过 allocation/buffer
descriptor range 覆盖 shader 按 layout 可能访问的 span
```

有些 shader ABI 会绑定一个比 logical tensor view 更大的 buffer range，例如 KV cache 或 llama.cpp
某些当前 row view。这个差异应该用 `LogicalTensor` 的 view/binding range metadata 表达，不能偷偷把
`LogicalTensor.shape` 改大，也不能在 `ShaderContract` 里另造一个不对应 GLSL field 的 tensor 名字。

## ShaderVariant

`ShaderContract` 描述 ABI；`ShaderVariant` 描述一个可编译、可 dispatch 的具体 shader source variant。

```python
RuntimeShapeSymbolResolver: TypeAlias = Callable[
    [Mapping[str, int]],
    Mapping[str, int],
]
RuntimeSpecializationResolver: TypeAlias = Callable[
    [Mapping[str, int]],
    Mapping[int, int] | Sequence[int] | None,
]


@dataclass(frozen=True, slots=True)
class SubgroupRequirements:
    required_size: int
    require_full_subgroups: bool = False


@dataclass(frozen=True, slots=True)
class CooperativeMatrixRequirements:
    scope: Literal["subgroup"]
    m_size: int
    n_size: int
    k_size: int
    a_type: str
    b_type: str
    c_type: str
    result_type: str
    saturating_accumulation: bool = False


@dataclass(frozen=True, slots=True)
class ShaderExecutionRequirements:
    subgroup: SubgroupRequirements | None = None
    cooperative_matrix: CooperativeMatrixRequirements | None = None
    require_integer_dot_product: bool = False
    require_shader_int64: bool = False
    require_buffer_device_address: bool = False
    require_storage_buffer_16bit_access: bool = False


@dataclass(frozen=True, slots=True)
class ShaderVariant:
    name: str
    family: str
    contract: ShaderContract
    source: str
    precompiled_spv_path: Path | None = None
    specialization_constants: tuple[tuple[int, int], ...] | None = None
    include_dirs: tuple[Path, ...] = ()
    compile_defines: tuple[str, ...] = ()
    execution_requirements: ShaderExecutionRequirements | None = None
    runtime_shape_symbols: tuple[str, ...] = ()
    runtime_shape_symbol_resolver: RuntimeShapeSymbolResolver | None = None
    runtime_specialization_resolver: RuntimeSpecializationResolver | None = None

    def __call__(self, rt: "RuntimeSession", **arguments: object) -> None:
        rt.dispatch(self, **arguments)


```

Variant 校验规则：

1. `variant.name == contract.shader_name`；
2. `family` 非空，用于 artifact grouping 和 policy；
3. `source` 必须是非空内联 GLSL；
4. specialization constants 归一化成 `(constant_id, value)`；
5. runtime specialization resolver 返回的 id/value 必须稳定可记录；
6. include dirs 和 compile defines 进入 shader artifact manifest；
7. execution requirements 在 pipeline 创建前检查 device feature；
8. variant definition site / source anchors 可选记录，方便 mismatch report 跳转到内联 shader source。

GLSL source 必须内置在 `ShaderVariant` 定义里，而不是引用 `*.comp` 文件。构建工具可以把内联 source
materialize 到 `.cache/torch2vk/generated/<variant>.glsl`，再编译出 SPIR-V；这些是 artifact，不是
source of truth。这个方向和 `agentorch` 一致：source layer 定义 named variant、contract 和 inline
GLSL，artifact layer 只负责缓存和编译结果。

## Dispatch 准备顺序

`RuntimeSession.dispatch(variant, **arguments)` 推荐顺序：

```text
1. 检查当前处于 rt.frame(...)
2. 根据 ShaderContract.fields 检查 required tensor fields 和 unexpected fields
3. resolve/materialize input fields，reserve/materialize output fields
4. 从 LogicalTensor 绑定 tensor shape symbols
5. 从 tensor layout 绑定 layout symbols
6. 校验 dtype / rank / shape / layout
7. 合并 runtime_shape_symbol_resolver 产生的 symbols
8. resolve output specs，确认 output materialization 与 contract 一致
9. materialize uniform buffers
10. validate_and_pack_push_constants
11. resolve dispatch group count
12. 根据 field/uniform binding、push constant size、specialization constants、execution requirements 取得 pipeline
13. 绑定 descriptor set 和 push constants
14. 提交 Vulkan dispatch
15. 记录 DispatchRecord
```

`DispatchRecord` 除了已有字段，还建议记录：

```text
shader variant name
contract class_name
descriptor bindings: field -> binding index / offset / range
push_constant_values
specialization_constants
shape_symbols
execution_requirements summary
```

这对 replay、debug dump 和 llama.cpp/外部 shader ABI 对齐很关键。

## Definition Validation

contract 定义阶段应该立即失败的错误：

```text
class_name / shader_name 为空
dispatch 不是 3 维
dispatch literal <= 0
field name 重复
uniform name 重复
descriptor binding index 重复
tensor field 使用了非 storage-buffer descriptor type
uniform field binding 与 tensor field binding 冲突
dtype 不被 runtime 支持
same_as 引用不存在的 field
layout rank 或 packed layout 规则不满足
push constant field offset 越界或重叠
contract referenced symbol 没有来源
runtime symbol shadow 了 tensor-derived symbol
```

dispatch 阶段应该失败的错误：

```text
缺少 tensor field
传入 unknown tensor field
read tensor 无法 materialize
write tensor role/memory/lifetime 不合法
shape symbol 绑定冲突
layout symbol 绑定冲突
dtype 不匹配
descriptor range 与 tensor view 不匹配
push constant 缺少输入或多传输入
push constant 值类型或范围不合法
runtime resolver 少返回或多返回 symbol
dispatch group count 不是 concrete positive int
device 不满足 execution requirements
```

## Manifest

每个 shader artifact 应该保存 contract manifest，至少包括：

```text
shader_name
class_name
family
fields: name / io_kind / binding / role / dtype / shape / layout
uniforms
push_constants: size / fields / offsets / value expressions
dispatch expression
descriptor bindings: field or uniform name / binding index / descriptor type
shape_symbols: tensor_shape / tensor_layout / referenced / runtime
specialization constants
compile defines
include dirs
execution requirements
source hash
```

manifest 的用途：

1. 判断 generated GLSL/SPIR-V 是否 stale；
2. replay 时确认 pipeline ABI 没漂移；
3. mismatch report 能输出 shader ABI；
4. 与外部 Vulkan baseline 对齐 descriptor / push constant / dispatch；
5. 后续 liveness planner 可消费 read/write fields。

## 和 torch2vk 现有架构的关系

`agentorch` 的 contract 直接处理执行态 tensor wrapper；`torch2vk` 第一版应该改成处理 `LogicalTensor`：

```text
模型 adapter:
  SOME_SHADER_VARIANT(rt, x=tensors.x, weight=tensors.weight, output=tensors.output)

RuntimeSession.dispatch:
  LogicalTensor.current buffer state -> DescriptorBufferBinding
```

所以 `ShaderContract` 中不应该出现 checkpoint、feed、lifetime、arena、PyTorch probe，也不应该用
默认绑定或间接名字去补模型语义。它只校验 `LogicalTensor` 当前 buffer 状态是否符合
shader ABI。LogicalTensor 的 role/memory/lifetime 和 metadata 决定怎么 materialize，ShaderContract
的 field/io/binding index 决定怎么 dispatch。

## 最小例子

```python
ELEMENTWISE_MUL_F32 = ShaderVariant(
    name="elementwise_mul_f32",
    family="toy",
    contract=ShaderContract(
        class_name="ElementwiseMulF32Program",
        shader_name="elementwise_mul_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                binding=0,
                role="x",
                contract=TensorContract("float32", ("N",)),
            ),
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                binding=1,
                role="weight",
                contract=TensorContract("float32", ("N",)),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                binding=2,
                role="output",
                contract=TensorContract("float32", ("N",)),
            ),
        ),
        uniforms=(),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, "N"),
            ),
        ),
        dispatch=(ceil_div("N", 256), 1, 1),
    ),
    source="""
#version 460

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer {
    float x[];
};

layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer {
    float weight[];
};

layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer {
    float output[];
};

layout(push_constant) uniform Params {
    uint N;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= pc.N) {
        return;
    }
    output[i] = x[i] * weight[i];
}
""".lstrip(),
)
```

上面的 inline GLSL ABI：

```glsl
layout(set = 0, binding = 0) buffer restrict readonly X { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly W { float weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly O { float output[]; };
layout(push_constant) uniform Params { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
```

## 禁止事项

```text
不要在 `ShaderVariant` 之外另写 shader 函数绕开 contract 直接传 descriptor binding。
不要在 contract 里声明 LogicalTensor lifetime 或 checkpoint。
不要让 push constants 成为无 schema 的 dict。
不要让 unresolved symbol 到 dispatch 时才变成 KeyError。
不要 silent cast dtype。
不要 silent reinterpret layout。
不要把 descriptor range 差异藏进 shape。
不要把 specialization constants 写死在 runtime 外部临时逻辑里。
不要让 replay 使用一份和 eager 不同的 shader ABI。
不要把 `*.comp` 当成 shader source of truth；`.glsl` / `.spv` 只能是从 inline source 生成的 artifact。
```
