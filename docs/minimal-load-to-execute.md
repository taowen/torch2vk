# 从声明到第一个 shader 执行的最短 MVP

本文定义 `torch2vk` 第一版可执行链路。目标不是做完整推理引擎，而是验证核心架构可以闭环：

```text
声明 LogicalTensor 点
  -> with rt.frame(...) 建立 eager execution 边界
  -> ShaderVariant 连接点和线
  -> RuntimeSession 从对应 pool/arena resolve/update LogicalTensor 当前 buffer 状态
  -> 调用第一个 Vulkan compute shader
  -> readback
  -> 和 lockstep PyTorch/CPU artifact 对拍
  -> Frame exit 释放 workspace
```

这版 MVP 必须引入 `Frame`，但只引入 `RuntimeSession.frame(...)` 这种 scope，不引入旧实现里的 `frame.workspace.*` 物理 slot tree。

## 核心结论

MVP 保留四个核心概念：

```text
LogicalTensor
  模型可见 tensor 对象。包含语义、materialization metadata，以及由 RuntimeSession 维护的当前 buffer 状态。

ShaderContract / ShaderVariant
  shader 的 typed contract。描述读写哪些 LogicalTensor、binding、shape、dtype、layout 和 dispatch。

RuntimeSession
  拥有 Vulkan device、allocation registry、dispatch、readback、compare、close 生命周期，并维护 LogicalTensor 当前 buffer 状态。

Frame
  一次 PyTorch model.forward 对齐的 eager execution scope。RuntimeSession 在 Frame 内自动分配和释放显存。
```

第一版规则：

1. 模型代码只声明 `LogicalTensor`，不手动分配显存；
2. 模型 forward 必须在 `with rt.frame(...)` 内调用 `ShaderVariant`；
3. `ShaderVariant.__call__` 只调用 `RuntimeSession.dispatch()`；
4. `RuntimeSession.dispatch()` 根据 `LogicalTensor` metadata resolve/materialize reads/writes；
5. `LogicalTensor` 记录当前 buffer 状态，但不拥有 allocation、不负责释放；
6. `RuntimeSession` 拥有所有 allocation，并按 Frame/request/model 生命周期释放；
7. dispatch records 是后续 compare、replay、liveness/aliasing 的事实来源。

所有 tensor 引用都传 `LogicalTensor` 对象本身。`LogicalTensor.name` 用于报告、artifact 路径和重复声明
检查；当 `role == WEIGHT` 时，它同时作为 checkpoint tensor key。

`LogicalTensor` metadata 只为四个目标服务：

1. 权重加载托管给框架，权重 tensor 只用 `LogicalTensor.name/spec/layout` 表达 checkpoint key、dtype、shape、layout；
2. 显存分配和释放由 runtime 根据 storage/lifetime 决定；
3. candidate 和 PyTorch/CPU 对拍时能定位 artifact、校验数值；
4. shader 调用时能校验传入 tensor 和 `ShaderContract` 匹配。

传给 `ShaderVariant` 的 `LogicalTensor` 应精确对应 GLSL input/output。缺少 layout、view range、
packed shape、state 等信息时，把这些补成 `LogicalTensor` metadata，而不是在 shader contract
里加默认 tensor 绑定。

注意：record/eager 阶段的 materialize 发生在 shader dispatch 准备阶段。它可以为了验证闭环而按需打开 checkpoint、上传权重、分配或扩容 buffer；这一阶段不追求高性能。Replay/capture 后才能基于真实 dispatch records 在 Frame enter 做权重预加载、liveness/aliasing 和 arena 准备。

## 为什么必须有 Frame

如果没有 Frame，runtime 不知道：

1. 哪些 activation 可以在一次 forward 结束后释放；
2. 哪些 written tensors 属于一次 lockstep PyTorch model.forward 的对拍边界；
3. 同一个 logical tensor name 在不同 audio token index / cond row 下的值如何区分；
4. dispatch records 应该按什么 scope 收集；
5. 后续 replay plan 应该覆盖哪段 eager execution。

所以 MVP 就引入 Frame，但必须保持它很薄：

```python
with rt.frame("toy.elementwise_mul", scope={"case": "mvp"}):
    ELEMENTWISE_MUL_F32(rt, x=x, weight=w, output=y)
```

禁止回到旧模式：

```text
不要有 frame.workspace.*.activation(name)
不要有 TensorSlot.logical_as(name)
不要让物理 slot 制造模型语义名字
不要让模型代码调用 RuntimeSession.empty/load_weight/free 管显存
```

## 最小使用形态

目标 API：

```python
with RuntimeSession.open(device_index=0, model_dir=model_dir) as rt:
    x = LogicalTensor(
        name="toy.x",
        spec=TensorSpec(dtype="float32", shape=(1024,)),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )
    w = LogicalTensor(
        name="scale",
        spec=TensorSpec(dtype="float32", shape=(1024,)),
        role=TensorRole.WEIGHT,
        memory=MemoryClass.MODEL_WEIGHT,
        lifetime=TensorLifetime.MODEL,
    )
    y = LogicalTensor(
        name="toy.y",
        spec=TensorSpec(dtype="float32", shape=(1024,)),
        role=TensorRole.OUTPUT,
        memory=MemoryClass.HOST_OUTPUT,
        lifetime=TensorLifetime.FRAME,
        compare=ComparePolicy(kind="tensor", rtol=1e-5, atol=1e-6),
    )

    rt.register_inputs({x: x_cpu})

    with rt.frame(
        "toy.elementwise_mul",
        scope={"case": "mvp"},
        reference_model=toy_reference_model,
    ):
        ELEMENTWISE_MUL_F32(rt, x=x, weight=w, output=y)
```

这里没有 binding table 暴露给模型，也没有 bound tensor tree。`x/w/y` 都是稳定 `LogicalTensor` 对象。
执行态 buffer 状态由 `RuntimeSession` 直接维护在这些 `LogicalTensor` 上。

MVP 可以先让 toy frame 传入一个最小 manual reference callable，作为当前 Frame 的 lockstep reference：

```python
def toy_reference_model(inputs: Mapping[LogicalTensor, object]) -> Mapping[str, object]:
    return {"toy.y": inputs[x] * load_cpu_weight("toy.safetensors", "scale")}
```

manual callable 只是 toy MVP 的 reference provider。真实模型路径必须使用当前 Frame 传入的 PyTorch model.forward + `LogicalTensor.pytorch_probe` 捕获 artifact。无论哪种 provider，compare targets 都由 candidate frame 实际写出的 tensors 驱动。

## 推荐目录结构

```text
src/torch2vk/
  __init__.py
  runtime/
    __init__.py
    logical.py             # LogicalTensor、TensorRole、TensorSemantic、MemoryClass、TensorLifetime、ComparePolicy
    frame.py               # FrameScope、frame context state
    materialize.py         # read/write materialization，更新 LogicalTensor 当前 buffer 状态
    session.py             # RuntimeSession：frame/register/dispatch/readback/compare/close
    checkpoint.py          # safetensors reader adapter，后续可扩展 gguf
    shader.py              # ShaderContract、ShaderVariant、DispatchRecord、contract validation
    compare.py             # candidate vs lockstep PyTorch/CPU artifact compare
    pytorch_ref.py         # PyTorch probe/hook/manual provider/cache
    shaders/
      __init__.py
      elementwise_mul_f32.py
  vulkan/
    ...
scripts/
  compile_shaders.py
  run_toy_mvp.py
tests/
  test_logical.py
  test_frame_scope.py
  test_materialize_dry_run.py
  test_shader_contract.py
  test_toy_mvp_dry_run.py
  test_toy_mvp_vulkan.py
```

当前仓库已有 `src/models` 作为模型 adapter 目录。通用执行层统一放进 `src/torch2vk/runtime`，
Vulkan 后端继续放 `src/torch2vk/vulkan`。模型 adapter 后续继续放 `src/models/<model_name>`。

## 核心类型

### TensorSpec / TensorLayout

```python
@dataclass(frozen=True, slots=True)
class TensorSpec:
    dtype: str
    shape: tuple[int | str, ...]


@dataclass(frozen=True, slots=True)
class TensorLayout:
    name: str = "row_major"
    params: Mapping[str, int | str] = field(default_factory=dict)
```

MVP 执行期只允许 concrete shape。声明和 contract 里可以出现 symbol，但 dispatch 前必须 resolve 成 int。

### LogicalTensor

```python
@dataclass(slots=True, eq=False)
class LogicalTensor:
    name: str
    spec: TensorSpec
    role: TensorRole
    memory: MemoryClass
    lifetime: TensorLifetime
    layout: TensorLayout = ROW_MAJOR
    semantic: TensorSemantic | None = None
    compare: ComparePolicy | None = None
    pytorch_probe: PyTorchProbe | None = None
    buffer: BufferSlice | None = None
    descriptor_nbytes: int | None = None
    version: int = 0
    writer: DispatchWriter | None = None
```

验收规则：

1. `name` 非空；
2. `name` 不包含 frame scope，例如 audio token index / cond row；
3. `dtype` 必须在 runtime 支持表中；
4. shape rank 固定；
5. execution 前 symbolic shape 必须 resolve；
6. `role/memory/lifetime` 组合必须合法；
7. `role == WEIGHT` 必须 `memory=MODEL_WEIGHT`、`lifetime=MODEL`；
8. `role == WEIGHT` 时 `name` 必须能作为 checkpoint tensor key；
9. `role == INPUT` 必须能通过 `register_inputs()` 绑定到运行时输入。

### TensorRole

```python
class TensorRole(StrEnum):
    INPUT = "input"
    WEIGHT = "weight"
    ACTIVATION = "activation"
    SCRATCH = "scratch"
    OUTPUT = "output"
    STATE = "state"
```

`TensorRole` 只表达粗粒度执行关系。`LOGITS`、`TOKEN`、`KV_CACHE` 这类模型语义使用可选 semantic metadata：

```python
class TensorSemantic(StrEnum):
    LOGITS = "logits"
    TOKEN = "token"
    KV_CACHE = "kv_cache"
    MASK = "mask"
    WAVEFORM = "waveform"
```

### MemoryClass

```python
class MemoryClass(StrEnum):
    MODEL_WEIGHT = "model_weight"
    REQUEST_STATE = "request_state"
    FRAME_WORKSPACE = "frame_workspace"
    OP_SCRATCH = "op_scratch"
    HOST_INPUT = "host_input"
    HOST_OUTPUT = "host_output"
```

`MemoryClass` 描述 runtime materialize 某个 `LogicalTensor` 时默认使用哪类 storage/pool。`HOST_INPUT` / `HOST_OUTPUT` 仍然是 tensor storage class，表示 shader 可绑定的 host-visible port 或 backend 内部等价 copy 路径；它们不表示 frame exit 必然自动读回。

### TensorLifetime

```python
class TensorLifetime(StrEnum):
    MODEL = "model"
    REQUEST = "request"
    FRAME = "frame"
    OP = "op"
    EXTERNAL = "external"
```

MVP 可以不实现 `EXTERNAL`。

### 运行时输入

`role == INPUT` 的 `LogicalTensor` 直接作为 `RuntimeSession.register_inputs()` 的 key：

```python
rt.register_inputs({x: x_cpu})
```

Runtime 在 materialize 输入时用同一个 `LogicalTensor` 对象查找数据；不存在额外的 input key 或输入 metadata。

### BufferSlice

```python
@dataclass(frozen=True, slots=True)
class BufferSlice:
    allocation_id: int
    offset: int
    nbytes: int
```

`BufferSlice` 只是 runtime 内部 view，不拥有 Vulkan buffer。真实 buffer 和 memory owner 归 `RuntimeSession` 内部 allocation registry。

### LogicalTensor runtime state

`RuntimeSession` 不维护一份平行的 binding table。read/write materialization 会直接更新对应
`LogicalTensor` 的当前 buffer 状态：

```text
buffer: BufferSlice | None
descriptor_nbytes: int | None
version: int
writer: DispatchWriter | None
```

模型代码只传递 `LogicalTensor`，不手动分配、释放或改写这些 runtime 字段。它们用于 runtime dispatch、
readback、compare、debug 和 replay。

## RuntimeSession API

MVP 推荐 API：

```python
class RuntimeSession:
    @classmethod
    def open(
        cls,
        *,
        device_index: int = 0,
        model_dir: Path | None = None,
    ) -> "RuntimeSession": ...

    def register_inputs(self, inputs: Mapping[LogicalTensor, object]) -> None: ...

    @contextmanager
    def frame(
        self,
        name: str,
        *,
        scope: Mapping[str, str | int] | None = None,
        pytorch_model: PyTorchFrameModel | None = None,
        reference_model: FrameReferenceProvider | None = None,
    ) -> Iterator[FrameContext]: ...

    def dispatch(self, variant: ShaderVariant, **arguments: object) -> None: ...

    def readback(self, tensor: LogicalTensor) -> object: ...

    def debug_materialization(self, tensor: LogicalTensor) -> BufferSlice | None: ...

    def close(self) -> None: ...
```

`RuntimeSession.open(...)` 返回的 session 必须支持 context manager；`__exit__` 调用幂等 `close()`。
`model_dir` 作为权重 checkpoint 的解析根目录。MVP 约定优先使用 `model.safetensors` 或
`model.safetensors.index.json`；否则目录里必须只有一个 safetensors checkpoint。

`LogicalTensor` declaration 在实际使用点校验：`register_inputs()` 校验输入 tensor，`dispatch()` 根据
shader fields 校验本次调用传入的 tensor，并把本 Frame 实际读写的 `LogicalTensor` 收集进 dispatch
records。后续 dispatch、input binding、readback 和 compare 都直接使用调用方持有的 `LogicalTensor`
对象。

Frame 入口不接收一份额外的 logical tensor 列表。Record/eager 阶段不知道也不要求提前声明本次
forward 可能读取哪些权重；权重 read、input read、activation/output write 都由对应 `dispatch()` 按需
resolve/materialize。Replay 阶段可以消费已经录制的 dispatch records，在 Frame enter 提前做权重加载、
workspace sizing、liveness/aliasing 和 arena offset 分配。这些都是 runtime 直接基于 dispatch records
推导出的执行策略，不额外引入一份模型侧存储描述。

`RuntimeSession` 负责：

1. 创建和销毁 Vulkan device/context；
2. 在 `register_inputs()` / `dispatch()` 使用点校验 `LogicalTensor` declarations；
3. 管理 frame stack；
4. 按 model/request/frame/host pool 分配或扩容 device-local、host-visible、staging、readback buffer；
5. 读取 checkpoint 并上传 weight；
6. 上传 input；
7. 根据 dispatch read/write resolve/materialize `LogicalTensor`；
8. 根据 `LogicalTensor` 当前 buffer 状态绑定 descriptor；
9. 插入必要 Vulkan barrier；
10. 记录 dispatch reads/writes；
11. Frame exit 时 readback + compare；
12. Frame exit 时释放/reuse frame workspace；
13. `close()` 时 `vkDeviceWaitIdle`，再释放所有 allocation 和 Vulkan handles。

## 权重加载

权重不再有单独的来源对象。`RuntimeSession` 从 `LogicalTensor` 自身推断 checkpoint view：

```text
checkpoint key = LogicalTensor.name
dtype          = LogicalTensor.spec.dtype
shape          = LogicalTensor.spec.shape
layout         = LogicalTensor.layout
checkpoint     = RuntimeSession.model_dir 下的 canonical safetensors
```

声明示例：

```python
w = LogicalTensor(
    name="scale",
    spec=TensorSpec(dtype="float32", shape=(1024,)),
    role=TensorRole.WEIGHT,
    memory=MemoryClass.MODEL_WEIGHT,
    lifetime=TensorLifetime.MODEL,
)
```

`RuntimeSession.dispatch()` 在 shader 第一次读到 weight 时按需加载权重：

1. 打开 checkpoint；
2. 检查 key 存在；
3. 检查 dtype 完全一致；
4. 检查 shape 完全一致；
5. 禁止 silent cast；
6. 上传到 device-local allocation；
7. 注册 model-lifetime materialization；
8. 后续 dispatch/frame 复用同一 weight。

Record/eager 阶段允许 dispatch 打开 checkpoint 并上传权重，因为这一阶段的目标是得到真实执行事实和正确性
对拍，不是热路径性能。Replay/capture 后，runtime 可以根据录制结果知道 frame 内会读哪些权重，并在 replay
Frame enter 预加载或校验这些 model-lifetime materialization。

MVP 先支持 safetensors。GGUF 后续再加 `CheckpointReader` 抽象。

## Shader 设计

### ShaderContract

```python
class IOKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


@dataclass(frozen=True, slots=True)
class TensorContract:
    dtype: str
    shape: tuple[int | str, ...]
    layout: TensorLayout = ROW_MAJOR


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    name: str
    io_kind: IOKind
    role: str
    contract: TensorContract
    descriptor_type: str = "storage_buffer"


@dataclass(frozen=True, slots=True)
class ShaderContract:
    name: str
    fields: tuple[TensorFieldSpec, ...]
    dispatch: tuple[int | str, int | str, int | str]
    push_constants: Mapping[str, str] = field(default_factory=dict)
```

contract 校验：

1. 所有 required field 都传入；
2. 不允许 unknown field；
3. dtype/rank/shape/layout 匹配；
4. symbol shape 能解析；
5. field name 不重复；
6. `OUTPUT` field 的 role/memory/lifetime 合法；
7. read/write materialization 规则可满足；
8. shader source binding 和 field 顺序一致；
9. dispatch size concrete。

### ShaderVariant

```python
@dataclass(frozen=True, slots=True)
class ShaderVariant:
    name: str
    family: str
    contract: ShaderContract
    source: str
    precompiled_spv_path: Path | None = None
    specialization_constants: Mapping[int, int] = field(default_factory=dict)

    def __call__(self, rt: RuntimeSession, **arguments: object) -> None:
        rt.dispatch(self, **arguments)
```

`RuntimeSession.dispatch()` 必须先执行：

```text
ensure inside rt.frame(...)
contract.validate(tensors)
resolve symbols
resolve/materialize reads
reserve/resolve writes from the current pool/arena
```

然后再创建/复用 pipeline、descriptor set、command buffer。

Record/eager MVP 的 dispatch 路径可以按需创建、扩容或 suballocate buffer；这是为了先把 shader 执行、
dispatch record、readback 和 compare 跑通。热路径 replay 不应该依赖这种临场分配，而应使用 capture 后的
dispatch records 和 arena high-water mark。

## 第一个 shader

第一个 shader 选 `elementwise_mul_f32`：

```text
output[i] = x[i] * weight[i]
```

原因：

1. 覆盖 input、weight、output 三类 tensor；
2. 覆盖 read materialization 和 write materialization；
3. 覆盖 weight checkpoint load；
4. 覆盖 Vulkan descriptor、pipeline、dispatch、readback；
5. PyTorch/CPU expected artifact 简单。

contract：

```text
x       float32 shape=(N,) read   parameter_index=0
weight  float32 shape=(N,) read   parameter_index=1
output  float32 shape=(N,) write  parameter_index=2
dispatch=(ceil_div(N, 256), 1, 1)
```

MVP 可以用 push constant 或 uniform 传 `N`。GLSL 必须 guard 越界 global id。

## DispatchRecord

每次 shader 调用都记录：

```python
@dataclass(frozen=True, slots=True)
class DispatchRecord:
    index: int
    frame: str
    scope: FrameScope
    shader: str
    reads: Mapping[str, LogicalTensor]
    writes: Mapping[str, LogicalTensor]
    logical_reads: Mapping[str, str]
    logical_writes: Mapping[str, str]
    symbols: Mapping[str, int]
    dispatch_size: tuple[int, int, int]
```

它用于：

1. read-before-write 检查；
2. debug 输出；
3. PyTorch compare target collection；
4. readback 精确定位；
5. 后续 liveness/aliasing；
6. 后续 replay；
7. mismatch drilldown。

`logical_reads/logical_writes` 记录 logical tensor name，`reads/writes` 记录参与本次 dispatch 的
`LogicalTensor`。

## PyTorch/CPU lockstep 对拍

MVP toy 可以先使用 manual reference callable，不急着接 module hook：

```python
def toy_reference_model(inputs: Mapping[LogicalTensor, object]) -> Mapping[str, object]:
    x_cpu = inputs[x]
    w_cpu = read_safetensors_tensor("toy.safetensors", "scale")
    return {"toy.y": x_cpu * w_cpu}
```

Frame exit 时 RuntimeSession：

```text
收集本 Frame written tensors
过滤 compare != None 的 tensor
readback candidate LogicalTensor 当前 buffer
从当前 frame 的 reference provider 取得 artifact；真实模型 provider 必须来自 PyTorch model.forward hook
compare_tensor(policy, candidate, expected)
```

manual reference provider 可以按 `LogicalTensor.name` 返回 expected artifact。真实 PyTorch provider 必须要求参与 compare 的 tensor 同时声明 `pytorch_probe`，用于在当前 frame 的 PyTorch model.forward 中捕获 artifact。

`compare_tensor()` 使用 `LogicalTensor.compare`：

```python
@dataclass(frozen=True, slots=True)
class ComparePolicy:
    kind: Literal["tensor", "token", "waveform"] = "tensor"
    rtol: float = 1e-4
    atol: float = 1e-4
    max_abs: float | None = None
```

失败报告至少包含：

```text
frame name
scope
tensor name
writer shader / dispatch index
shape/dtype mismatch if any
max_abs
max_rel if practical
first mismatch index if practical
```

后续再加：

1. `PyTorchProbe` module hook；
2. artifact cache；
3. derived transform；
4. boundary drilldown。

## 分阶段实施计划

### Phase 0：基础类型和声明校验

交付：

1. `src/torch2vk/runtime/logical.py`；
2. 复用 `src/torch2vk/vulkan/types.py` 中已有的 `TensorSpec` / layout / dtype helpers；
3. `TensorRole` / `TensorSemantic` / `MemoryClass` / `TensorLifetime`；
4. `ComparePolicy` / `PyTorchProbe`；
5. `tests/test_logical.py`。

验收：

1. name 非空；
2. dtype 支持表明确；
3. concrete shape nbytes 正确；
4. symbolic shape dispatch 前必须 resolve；
5. `role/memory/lifetime` 合法性校验；
6. weight role/memory/lifetime 不合法时报错；
7. input missing binding 在 materialize 阶段报错。

### Phase 1：Frame scope、arena 和 dry-run materialization

先不碰 Vulkan，做 fake allocation registry 和 per-lifetime arena。

交付：

1. `FrameScope`；
2. `RuntimeSession.frame()` context manager；
3. `LogicalTensor` runtime buffer state；
4. fake `BufferAllocation` / `BufferSlice`；
5. `MODEL_WEIGHT` / `REQUEST_STATE` / `FRAME_WORKSPACE` arena；
6. read/write materialization rules；
7. `tests/test_frame_scope.py`；
8. `tests/test_materialize_dry_run.py`。

验收：

1. dispatch 必须在 frame 内；
2. frame scope 生成稳定 artifact prefix；
3. write tensor 会从当前 arena reserve slice 并更新 LogicalTensor 当前 buffer 状态；
4. read-before-write 能报错；
5. weight read 会创建 model-lifetime materialization；
6. input read 会检查 runtime input binding；
7. frame exit reset frame arena；
8. model/request lifetime allocation 保留；
9. dry-run 能记录每次 dispatch 触发的 materialization 和 allocation 需求。

### Phase 2：权重加载 dry-run + safetensors

交付：

1. `checkpoint.py` safetensors reader；
2. `RuntimeSession` weight materialization；
3. toy safetensors fixture；
4. dtype/shape/key 校验。

验收：

1. key 缺失时报 `logical name + checkpoint key`；
2. dtype mismatch 不 cast，直接失败；
3. shape mismatch 直接失败；
4. 成功时创建 model-lifetime materialization；
5. weight 在多个 frame 中复用；
6. weight allocation 生命周期归 session。

### Phase 3：ShaderContract dry-run

交付：

1. `src/torch2vk/runtime/shader.py`；
2. `ShaderContract.validate()`；
3. `DispatchRecord`；
4. dry-run `RuntimeSession.dispatch()`：校验、materialize、记录，不提交 Vulkan；
5. `tests/test_shader_contract.py`；
6. `tests/test_toy_mvp_dry_run.py`。

验收：

1. 漏 field、多 field、dtype 错、shape 错都失败；
2. dispatch 不在 frame 内失败；
3. read-before-write 失败；
4. dispatch record 包含 `LogicalTensor` reads/writes；
5. frame exit 能收集 written compare tensors。

### Phase 4：Vulkan 最小执行

交付：

1. Vulkan instance/device/queue bootstrap；
2. buffer create/bind/map/copy；
3. shader module/pipeline/descriptor/command buffer/submit；
4. transfer/upload/readback barriers；
5. host-visible memory flush/invalidate；
6. `elementwise_mul_f32.py` 内置 GLSL source；
7. `scripts/compile_shaders.py` 从内置 source 生成 `.cache/torch2vk/generated/*.glsl` 和 `.spv`；
8. `scripts/run_toy_mvp.py`。

验收：

1. 能列出 compute-capable device；
2. 能上传 input 和 weight；
3. 能执行 `elementwise_mul_f32`；
4. 能 readback output；
5. output 和 CPU expected artifact 一致；
6. frame exit 后 frame workspace 释放；
7. session close 后 allocation 全释放。

### Phase 5：Frame compare

交付：

1. `src/torch2vk/runtime/compare.py`；
2. toy lockstep PyTorch/CPU model provider；
3. frame exit compare runner；
4. mismatch 报告；
5. `test_toy_mvp_vulkan.py`。

验收：

1. candidate 必须来自 Vulkan readback；
2. expected artifact 可由当前 frame 的 PyTorch/CPU model 计算；
3. allclose policy 来自 `LogicalTensor.compare`；
4. compare targets 来自 candidate written tensors；
5. mismatch 报告包含 frame、scope、tensor name、writer shader、max_abs。

### Phase 6：第一个真实 op / 子链路

Toy MVP 稳定后再引入真实 op。建议顺序：

1. `add_f32` 或 `rms_norm_f32`；
2. `linear_f32`，权重 layout 必须显式声明；
3. `embedding_lookup` / `embedding_sum`；
4. `audio_codec_decoder` 最短子链路；
5. 再考虑 attention 前的 view/reshape 支持。

每个 op 都必须经过：

```text
LogicalTensor declarations
  -> with rt.frame(...)
  -> shader contract
  -> RuntimeSession resolves LogicalTensor buffer state from pools/arenas
  -> Vulkan run
  -> dispatch record
  -> readback
  -> compare
```

## 显存规划

MVP 不需要额外准备存储描述。Record/eager 阶段可以让 shader dispatch 按需驱动 materialization 和
底层显存分配；replay 热路径直接用录制出来的 dispatch records 和 high-water mark 消除这部分开销。

第一版显存策略：

```text
MODEL_WEIGHT
  record/eager 阶段由 shader read 按需加载，grow-only，session close 释放。
  replay 阶段可根据 dispatch records 在 Frame enter 预加载或校验。

REQUEST_STATE
  request/pipeline lifetime，grow-only 或预分配，request end reset。

FRAME_WORKSPACE
  record/eager 阶段首次 dispatch 需要时创建或扩容，Frame exit 整体 reset。
  replay 阶段可在 Frame enter 根据 dispatch records/high-water mark 预分配。

OP_SCRATCH
  MVP 可以并入 FRAME_WORKSPACE。
```

只有出现这些需求时才让 runtime 在内部直接消费 dispatch records 做优化：

1. 显存峰值太高，需要 aliasing；
2. 一次性批量规划大量 activation；
3. replay 需要 stable storage fingerprint；
4. 要在运行前估算 memory footprint；
5. 要跨 shader sequence 做 liveness 分析。

这些优化不应该替代 `LogicalTensor`，也不应该成为模型目录手写的执行期查询表。

## Vulkan 风险和约束

MVP 要尽早验证：

1. instance/device 创建；
2. compute queue selection；
3. device limits；
4. storage buffer alignment；
5. shader module 创建；
6. descriptor set layout / descriptor pool / descriptor set update；
7. command pool / command buffer / submit / fence；
8. host-visible memory map/unmap；
9. host flush/invalidate；
10. transfer buffer copy；
11. shader dispatch 后 buffer memory barrier；
12. readback 前 transfer/host visibility barrier。

如果 Vulkan 路径阻塞，继续推进 dry-run materialization、shader contract、权重校验、dispatch record 和 compare，不让架构停住。

### dtype

PyTorch、checkpoint、LogicalTensor、shader contract 四方 dtype 必须显式一致。MVP 禁止 silent cast。

`float32` 作为第一版必选 dtype。`float16/bfloat16/int64` 需要根据 Vulkan device capability 和 shader extension 单独启用，不能无条件视为可执行 dtype。

### shape

执行期必须 concrete shape。symbol shape 只能存在于 declaration 或 contract 中，dispatch 前必须 resolve。

### layout

checkpoint 原始 layout 和 shader 消费 layout 必须显式一致。

如果 shader 需要 packed/reordered weight，必须通过显式 `TensorLayout` 或 preprocessing transform 表达。禁止把 packed tensor 伪装成原始 checkpoint tensor。

### 资源释放

所有 Vulkan handle owner 必须支持幂等 `close()`。`RuntimeSession.close()` 顺序：

```text
vkDeviceWaitIdle
release frame/request/model buffers
release staging/readback buffers
release pipelines/shader modules
release descriptor pools/layouts
release command pool
release device
release instance
```

## 最小里程碑

第一个可合并版本：

```text
uv run pytest \
  tests/test_logical.py \
  tests/test_frame_scope.py \
  tests/test_materialize_dry_run.py \
  tests/test_shader_contract.py \
  tests/test_toy_mvp_dry_run.py
```

第一个可执行版本：

```text
uv run python scripts/run_toy_mvp.py --device 0
```

期望输出：

```text
device: ...
frame: toy.elementwise_mul
checkpoint: toy.safetensors
shader: elementwise_mul_f32
N: 1024
max_abs: 0.0
result: ok
frame_workspace_released: true
```

这个版本就是后续真实权重、真实 shader、PyTorch module hook、replay 和 liveness planning 的基线。
