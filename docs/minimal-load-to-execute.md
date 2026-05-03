# 从声明到第一个 shader 执行的最短 MVP

本文定义 `torch2vk` 第一版可执行链路。目标不是做完整推理引擎，而是验证核心架构可以闭环：

```text
声明 LogicalTensor 点
  -> with rt.frame(...) 建立 eager execution 边界
  -> shader wrapper 连接点和线
  -> RuntimeSession 从对应 pool/arena resolve tensor instance
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
  模型可见 tensor 声明。只包含语义和 materialization metadata，不携带模型代码依赖的 storage。

ShaderContract / ShaderVariant
  shader 的 typed contract。描述读写哪些 LogicalTensor、binding、shape、dtype、layout 和 dispatch。

RuntimeSession
  拥有 Vulkan device、allocation registry、materialization table、dispatch、readback、compare、close 生命周期。

Frame
  一次 PyTorch model.forward 对齐的 eager execution scope。RuntimeSession 在 Frame 内自动分配和释放显存。
```

第一版规则：

1. 模型代码只声明 `LogicalTensor`，不手动分配显存；
2. 模型 forward 必须在 `with rt.frame(...)` 内调用 shader wrapper；
3. shader wrapper 只调用 `RuntimeSession.dispatch()`；
4. `RuntimeSession.dispatch()` 根据 `LogicalTensor` metadata resolve/materialize reads/writes；
5. `LogicalTensor` 不拥有 allocation，不负责释放；
6. `RuntimeSession` 拥有所有 allocation，并按 Frame/request/model 生命周期释放；
7. dispatch records 是后续 compare、replay、liveness/aliasing 的事实来源。

注意：这里的 materialize 不是每个 shader 现场创建 Vulkan buffer 或 `vkAllocateMemory`。底层 allocation 必须按 model/request/frame pool 管理；dispatch 只从已有 pool/arena 取得 slice 或登记 tensor instance。

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
    elementwise_mul_f32(rt, x=x, weight=w, output=y)
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
with RuntimeSession.open(device_index=0) as rt:
    x = LogicalTensor(
        name="toy.x",
        spec=TensorSpec(dtype="float32", shape=(1024,)),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
        feed=InputFeed(name="toy.x"),
    )
    w = LogicalTensor(
        name="toy.weight.scale",
        spec=TensorSpec(dtype="float32", shape=(1024,)),
        role=TensorRole.WEIGHT,
        memory=MemoryClass.MODEL_WEIGHT,
        lifetime=TensorLifetime.MODEL,
        source=WeightSource(
            checkpoint="toy.safetensors",
            key="scale",
            dtype="float32",
            shape=(1024,),
        ),
    )
    y = LogicalTensor(
        name="toy.y",
        spec=TensorSpec(dtype="float32", shape=(1024,)),
        role=TensorRole.OUTPUT,
        memory=MemoryClass.HOST_READBACK,
        lifetime=TensorLifetime.FRAME,
        compare=ComparePolicy(kind="tensor", rtol=1e-5, atol=1e-6),
        pytorch_probe=PyTorchProbe(kind="manual_hook", target="toy.y"),
    )

    rt.register_tensors([x, w, y])
    rt.register_inputs({"toy.x": x_cpu})

    with rt.frame(
        "toy.elementwise_mul",
        scope={"case": "mvp"},
        pytorch_model=toy_pytorch_model,
    ):
        elementwise_mul_f32(rt, x=x, weight=w, output=y)
```

这里没有 binding table 暴露给模型，也没有 bound tensor tree。`x/w/y` 都是声明对象。执行态 storage 存在于 `RuntimeSession` 的 materialization table。

MVP 可以先让 toy frame 传入一个最小 PyTorch/CPU model callable，作为当前 Frame 的 lockstep model：

```python
def toy_pytorch_model(feeds: Mapping[str, object]) -> Mapping[str, object]:
    return {"toy.y": feeds["toy.x"] * load_cpu_weight("toy.safetensors", "scale")}
```

但对 RuntimeSession 来说，compare 仍然由 candidate frame 实际写出的 `toy.y` 驱动。

## 推荐目录结构

```text
src/torch2vk/
  __init__.py
  types.py                 # TensorSpec、TensorLayout、dtype_nbytes、nbytes
  logical.py               # LogicalTensor、TensorRole、MemoryClass、TensorLifetime、WeightSource、ComparePolicy
  frame.py                 # FrameScope、frame context state
  memory.py                # BufferAllocation、BufferSlice、allocation registry
  materialize.py           # materialization table、TensorInstanceKey、read/write materialization
  runtime.py               # RuntimeSession：frame/register/dispatch/readback/compare/close
  checkpoint.py            # safetensors reader，后续可扩展 gguf
  shader.py                # ShaderContract、ShaderVariant、DispatchRecord、contract validation
  compare.py               # candidate vs lockstep PyTorch/CPU artifact compare
  pytorch_ref.py           # PyTorch probe/hook/manual provider/cache
  shaders/
    __init__.py
    elementwise_mul_f32.py
    glsl/
      elementwise_mul_f32.comp
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

当前仓库已有 `src/models` 作为模型 adapter 目录。通用 runtime 建议放进 `src/torch2vk`，模型 adapter 后续继续放 `src/models/<model_name>`。

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
@dataclass(frozen=True, slots=True)
class LogicalTensor:
    name: str
    spec: TensorSpec
    role: TensorRole
    memory: MemoryClass
    lifetime: TensorLifetime
    layout: TensorLayout = ROW_MAJOR
    source: WeightSource | None = None
    feed: InputFeed | None = None
    compare: ComparePolicy | None = None
    pytorch_probe: PyTorchProbe | None = None
```

验收规则：

1. `name` 非空；
2. `name` 不包含 frame scope，例如 audio token index / cond row；
3. `dtype` 必须在 runtime 支持表中；
4. shape rank 固定；
5. execution 前 symbolic shape 必须 resolve；
6. `role/memory/lifetime` 组合必须合法；
7. `role == WEIGHT` 必须有 `source`；
8. `role == INPUT` 必须能通过 `feed` 或 runtime feed 找到输入。

### TensorRole

```python
class TensorRole(StrEnum):
    INPUT = "input"
    WEIGHT = "weight"
    ACTIVATION = "activation"
    SCRATCH = "scratch"
    OUTPUT = "output"
    STATE = "state"
    KV_CACHE = "kv_cache"
    LOGITS = "logits"
    TOKEN = "token"
```

### MemoryClass

```python
class MemoryClass(StrEnum):
    MODEL_WEIGHT = "model_weight"
    REQUEST_STATE = "request_state"
    FRAME_WORKSPACE = "frame_workspace"
    OP_SCRATCH = "op_scratch"
    HOST_INPUT = "host_input"
    HOST_READBACK = "host_readback"
```

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

### InputFeed

```python
@dataclass(frozen=True, slots=True)
class InputFeed:
    name: str
    required: bool = True
```

`InputFeed.name` 是 RuntimeSession 在 `register_inputs()` 中查找输入数据的 key。MVP 可以要求所有 input 都必须显式 feed。

### BufferSlice

```python
@dataclass(frozen=True, slots=True)
class BufferSlice:
    allocation_id: int
    offset: int
    nbytes: int
```

`BufferSlice` 只是 runtime 内部 view，不拥有 Vulkan buffer。真实 buffer 和 memory owner 归 `RuntimeSession` 内部 allocation registry。

### MaterializedTensor

```python
@dataclass(frozen=True, slots=True)
class TensorInstanceKey:
    logical_name: str
    scope_key: str
    version: int


@dataclass(slots=True)
class MaterializedTensor:
    key: TensorInstanceKey
    tensor: LogicalTensor
    storage: BufferSlice
    dtype: str
    shape: tuple[int, ...]
    layout: TensorLayout
    lifetime: TensorLifetime
    writer: DispatchWriter | None = None
```

模型代码不直接持有 `MaterializedTensor`。它用于 runtime dispatch、readback、compare、debug 和 replay。

## RuntimeSession API

MVP 推荐 API：

```python
class RuntimeSession:
    @classmethod
    def open(cls, *, device_index: int = 0) -> "RuntimeSession": ...

    def register_tensors(self, tensors: Iterable[LogicalTensor]) -> None: ...

    def register_inputs(self, feeds: Mapping[str, object]) -> None: ...

    @contextmanager
    def frame(
        self,
        name: str,
        *,
        scope: Mapping[str, str | int] | None = None,
        pytorch_model: PyTorchFrameModel | None = None,
    ) -> Iterator[FrameContext]: ...

    def dispatch(self, variant: ShaderVariant, **tensors: LogicalTensor) -> None: ...

    def readback(self, key: TensorInstanceKey) -> object: ...

    def debug_materialization(self, tensor: LogicalTensor) -> MaterializedTensor | None: ...

    def close(self) -> None: ...
```

`RuntimeSession` 负责：

1. 创建和销毁 Vulkan device/context；
2. 校验并注册 `LogicalTensor` declarations；
3. 管理 frame stack；
4. 按 model/request/frame/host pool 分配或扩容 device-local、host-visible、staging、readback buffer；
5. 读取 checkpoint 并上传 weight；
6. 上传 input feed；
7. 根据 dispatch read/write resolve/materialize tensor instance；
8. 根据 materialized storage 绑定 descriptor；
9. 插入必要 Vulkan barrier；
10. 记录 dispatch reads/writes；
11. Frame exit 时 readback + compare；
12. Frame exit 时释放/reuse frame workspace；
13. `close()` 时 `vkDeviceWaitIdle`，再释放所有 allocation 和 Vulkan handles。

## 权重加载

权重声明放在 `WeightSource`：

```python
@dataclass(frozen=True, slots=True)
class WeightSource:
    checkpoint: str
    key: str
    dtype: str
    shape: tuple[int, ...]
    layout: TensorLayout = ROW_MAJOR
```

声明示例：

```python
w = LogicalTensor(
    name="toy.weight.scale",
    spec=TensorSpec(dtype="float32", shape=(1024,)),
    role=TensorRole.WEIGHT,
    memory=MemoryClass.MODEL_WEIGHT,
    lifetime=TensorLifetime.MODEL,
    source=WeightSource(
        checkpoint="toy.safetensors",
        key="scale",
        dtype="float32",
        shape=(1024,),
    ),
)
```

`RuntimeSession` 在 shader read materialization 时加载权重：

1. 打开 checkpoint；
2. 检查 key 存在；
3. 检查 dtype 完全一致；
4. 检查 shape 完全一致；
5. 禁止 silent cast；
6. 上传到 device-local allocation；
7. 注册 model-lifetime materialization；
8. 后续 frame 复用同一 weight。

MVP 先支持 safetensors。GGUF 后续再加 `CheckpointReader` 抽象。

## Shader 设计

### ShaderContract

```python
class BindingAccess(StrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass(frozen=True, slots=True)
class TensorContract:
    dtype: str
    shape: tuple[int | str, ...]
    layout: TensorLayout = ROW_MAJOR


@dataclass(frozen=True, slots=True)
class Binding:
    field: str
    binding: int
    access: BindingAccess
    descriptor_type: str = "storage_buffer"


@dataclass(frozen=True, slots=True)
class ShaderContract:
    name: str
    inputs: Mapping[str, TensorContract]
    outputs: Mapping[str, TensorContract]
    bindings: tuple[Binding, ...]
    dispatch: tuple[int | str, int | str, int | str]
    push_constants: Mapping[str, str] = field(default_factory=dict)
```

contract 校验：

1. 所有 required field 都传入；
2. 不允许 unknown field；
3. dtype/rank/shape/layout 匹配；
4. symbol shape 能解析；
5. binding 编号不重复；
6. write binding 必须对应 output；
7. read/write materialization 规则可满足；
8. shader source binding 和 contract binding 一致；
9. dispatch size concrete。

### ShaderVariant

```python
@dataclass(frozen=True, slots=True)
class ShaderVariant:
    name: str
    family: str
    contract: ShaderContract
    spirv_path: Path
    source_path: Path | None = None
    specialization_constants: Mapping[int, int] = field(default_factory=dict)

    def __call__(self, ctx: RuntimeSession, **tensors: LogicalTensor) -> None:
        ctx.dispatch(self, **tensors)
```

wrapper 保持普通 Python：

```python
def elementwise_mul_f32(
    ctx: RuntimeSession,
    *,
    x: LogicalTensor,
    weight: LogicalTensor,
    output: LogicalTensor,
) -> None:
    ELEMENTWISE_MUL_F32(ctx, x=x, weight=weight, output=output)
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

MVP 禁止在正常 dispatch 路径上对每个 tensor 调用底层 Vulkan allocation。write materialization 只能从 `FRAME_WORKSPACE` / `REQUEST_STATE` / `MODEL_WEIGHT` 等 pool 中 suballocate `BufferSlice`。Frame 结束时 whole-arena reset，而不是逐 tensor free。

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
x       float32 shape=(N,) read  binding=0
weight  float32 shape=(N,) read  binding=1
output  float32 shape=(N,) write binding=2
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
    reads: Mapping[str, TensorInstanceKey]
    writes: Mapping[str, TensorInstanceKey]
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

`logical_reads/logical_writes` 记录 logical tensor name，`reads/writes` 记录具体 tensor instance。

## PyTorch/CPU lockstep 对拍

MVP 可以先让当前 Frame 传入的 PyTorch model 是 manual callable，不急着接 module hook：

```python
def toy_pytorch_model(feeds: Mapping[str, object]) -> Mapping[str, object]:
    x_cpu = feeds["toy.x"]
    w_cpu = read_safetensors_tensor("toy.safetensors", "scale")
    return {"toy.y": x_cpu * w_cpu}
```

Frame exit 时 RuntimeSession：

```text
收集本 Frame written tensors
过滤 compare != None 的 tensor
readback candidate TensorInstanceKey
读取当前 frame pytorch_model.forward/callable 产生的 artifact
compare_tensor(policy, candidate, expected)
```

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

1. `src/torch2vk/types.py`；
2. `src/torch2vk/logical.py`；
3. `TensorRole` / `MemoryClass` / `TensorLifetime`；
4. `WeightSource` / `InputFeed` / `ComparePolicy` / `PyTorchProbe`；
5. `tests/test_logical.py`。

验收：

1. name 非空；
2. dtype 支持表明确；
3. concrete shape nbytes 正确；
4. symbolic shape dispatch 前必须 resolve；
5. `role/memory/lifetime` 合法性校验；
6. weight missing source 报错；
7. input missing feed 在 materialize 阶段报错。

### Phase 1：Frame scope、arena 和 dry-run materialization

先不碰 Vulkan，做 fake allocation registry 和 per-lifetime arena。

交付：

1. `FrameScope`；
2. `RuntimeSession.frame()` context manager；
3. `TensorInstanceKey` / `MaterializedTensor`；
4. fake `BufferAllocation` / `BufferSlice`；
5. `MODEL_WEIGHT` / `REQUEST_STATE` / `FRAME_WORKSPACE` arena；
6. read/write materialization rules；
7. `tests/test_frame_scope.py`；
8. `tests/test_materialize_dry_run.py`。

验收：

1. dispatch 必须在 frame 内；
2. frame scope 生成稳定 artifact prefix；
3. write tensor 会从当前 arena reserve slice 并创建 materialized instance；
4. read-before-write 能报错；
5. weight read 会创建 model-lifetime materialization；
6. input read 会检查 runtime feed；
7. frame exit reset frame arena；
8. model/request lifetime allocation 保留；
9. dry-run 中不会出现 per-dispatch raw allocation。

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

1. `shader.py`；
2. `ShaderContract.validate()`；
3. `DispatchRecord`；
4. dry-run `RuntimeSession.dispatch()`：校验、materialize、记录，不提交 Vulkan；
5. `tests/test_shader_contract.py`；
6. `tests/test_toy_mvp_dry_run.py`。

验收：

1. 漏 field、多 field、dtype 错、shape 错都失败；
2. dispatch 不在 frame 内失败；
3. read-before-write 失败；
4. dispatch record 包含 `TensorInstanceKey` reads/writes；
5. frame exit 能收集 written compare tensors。

### Phase 4：Vulkan 最小执行

交付：

1. Vulkan instance/device/queue bootstrap；
2. buffer create/bind/map/copy；
3. shader module/pipeline/descriptor/command buffer/submit；
4. transfer/upload/readback barriers；
5. host-visible memory flush/invalidate；
6. `elementwise_mul_f32.comp`；
7. `scripts/compile_shaders.py`；
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

1. `compare.py`；
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
  -> RuntimeSession resolves tensor instances from pools/arenas
  -> Vulkan run
  -> dispatch record
  -> readback
  -> compare
```

## 显存规划和 StoragePlan

MVP 不需要手写 `StoragePlan`，但必须有 pool/arena。不能让 shader dispatch 直接驱动底层显存分配。

第一版显存策略：

```text
MODEL_WEIGHT
  按需加载或 register 阶段加载，grow-only，session close 释放。

REQUEST_STATE
  request/pipeline lifetime，grow-only 或预分配，request end reset。

FRAME_WORKSPACE
  Frame 进入时取得 arena，dispatch 前从 arena reserve slice，Frame exit 整体 reset。

OP_SCRATCH
  MVP 可以并入 FRAME_WORKSPACE。
```

只有出现这些需求时再引入 runtime-internal plan：

1. 显存峰值太高，需要 aliasing；
2. 一次性批量规划大量 activation；
3. replay 需要 stable storage fingerprint；
4. 要在运行前估算 memory footprint；
5. 要跨 shader sequence 做 liveness 分析。

引入时也只作为 runtime 优化：

```text
StoragePlan = dispatch records + LogicalTensor declarations + FrameScope -> allocation decisions
```

它不应该替代 `LogicalTensor`，也不应该成为模型目录手写的执行期查询表。

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
