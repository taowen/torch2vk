# 从权重加载到第一个 shader 执行的最短 MVP

目标：重写 `torch2vk` 的最小可执行链路。第一版只解决一件事：

```text
声明/加载权重 -> 分配显存 -> 调用第一个 Vulkan compute shader -> 读回 -> 和 PyTorch 对拍
```

不要在 MVP 阶段提前引入 `Frame`、`Workspace`、`BindingTable`、`TensorBinding`、复杂 `StoragePlan` 或 graph IR。
这些概念会让显存管理边界变厚，重复旧实现的问题。

## 核心结论

MVP 只保留三个核心对象：

```text
RuntimeSession
  拥有 Vulkan device、allocation owner、shader dispatch、readback、close 生命周期。

LogicalTensor
  模型可见 tensor 句柄。包含语义身份，也可以在执行期直接携带 storage。

BufferSlice
  一个具体显存范围。LogicalTensor.storage 指向它。
```

第一版的规则：

1. `LogicalTensor.storage is None` 表示声明态，可用于 schema、权重声明、PyTorch probe、shape 检查。
2. shader 调用前，调用者必须保证传入的 `LogicalTensor.storage is not None`。
3. dispatcher 必须检查所有 shader 参数都已经分配显存，否则报错。
4. `LogicalTensor` 不拥有 allocation，不负责释放。
5. `RuntimeSession` 拥有所有 allocation，并在 `close()` 时统一释放。

## 为什么不保留 Frame / Workspace

旧实现里的问题不是 `FRAME_WORKSPACE` 这个内存分类错了，而是把物理 workspace tree 暴露给模型执行代码了。

旧模式类似：

```python
frame.workspace.attention.q_proj.activation("prefill.layer.03.q_proj")
```

这同时制造了两个命名空间：

```text
frame.workspace.attention.q_proj     # 物理 slot / workspace 结构
prefill.layer.03.q_proj              # 模型语义 tensor 名字
```

一旦物理 slot 能生成 logical tensor，模型语义、显存布局、debug 对拍、liveness 都会混在一起。

MVP 直接禁止这种 API：

```text
不要有 frame.workspace.*.activation(name)
不要有 TensorSlot.logical_as(name)
不要让物理 slot 制造模型语义名字
```

如果以后需要 `Frame` 或 `Workspace`，它们只能是普通 dataclass 容器，字段必须已经是 `LogicalTensor`，不能持有 allocation owner，不能提供 `.activation(name)` 这种方法。

第一版不需要它们。

## 最小使用形态

目标 API 应该接近这样：

```python
with RuntimeSession.open(device_index=0) as rt:
    x = rt.input(
        name="toy.x",
        dtype="float32",
        shape=(1024,),
        data=x_cpu,
    )
    w = rt.load_weight(
        name="toy.weight.scale",
        source=WeightSource(
            checkpoint="toy.safetensors",
            key="scale",
            dtype="float32",
            shape=(1024,),
        ),
    )
    y = rt.empty(
        name="toy.y",
        dtype="float32",
        shape=(1024,),
        role=TensorRole.OUTPUT,
        memory=MemoryClass.HOST_READBACK,
        compare=ComparePolicy(kind="tensor", rtol=1e-5, atol=1e-6),
    )

    elementwise_mul_f32(rt, x=x, weight=w, output=y)

    candidate = rt.readback(y)
    reference = x_cpu * w_cpu
    compare_tensor(y, candidate, reference)
```

这里没有额外 binding table。`x/w/y` 都是已经绑定 storage 的 `LogicalTensor`。

## 推荐目录结构

```text
src/torch2vk/
  __init__.py
  types.py                 # TensorSpec、TensorLayout、dtype_nbytes、nbytes
  logical.py               # LogicalTensor、MemoryClass、TensorRole、WeightSource、ComparePolicy
  memory.py                # BufferAllocation、BufferSlice、Vulkan buffer ownership
  runtime.py               # RuntimeSession：allocate/load_weight/dispatch/readback/close
  checkpoint.py            # safetensors reader，后续可扩展 gguf
  shader.py                # ShaderContract、ShaderVariant、dispatch record、contract validation
  compare.py               # candidate vs PyTorch/reference compare
  pytorch_ref.py           # 后续：PyTorch module hook/manual provider/cache
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
  test_runtime_allocation.py
  test_shader_contract.py
  test_toy_mvp.py
```

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

MVP 执行期只允许 concrete shape。声明里可以暂时出现 symbol，但进入 `RuntimeSession.empty()` 或 shader dispatch 前必须 resolve 成 int。

### BufferSlice

```python
@dataclass(frozen=True, slots=True)
class BufferSlice:
    allocation_id: int
    offset: int
    nbytes: int
```

`BufferSlice` 只是 view，不拥有 buffer。真实 Vulkan buffer 和 memory owner 归 `RuntimeSession` 内部的 allocation registry。

### LogicalTensor

```python
@dataclass(frozen=True, slots=True)
class LogicalTensor:
    name: str
    spec: TensorSpec
    layout: TensorLayout = ROW_MAJOR
    role: TensorRole = TensorRole.ACTIVATION
    memory: MemoryClass = MemoryClass.FRAME_WORKSPACE
    storage: BufferSlice | None = None
    source: WeightSource | None = None
    compare: ComparePolicy | None = None
    pytorch_probe: PyTorchProbe | None = None

    @property
    def is_allocated(self) -> bool:
        return self.storage is not None

    def bind(self, storage: BufferSlice) -> "LogicalTensor":
        return replace(self, storage=storage)

    def require_storage(self) -> BufferSlice:
        if self.storage is None:
            raise ValueError(f"{self.name} is not allocated")
        return self.storage
```

`bind()` 返回新对象，不原地 mutate。这样执行代码拿到的是 bound tensor，但未绑定声明仍可复用。

### MemoryClass

```python
class MemoryClass(StrEnum):
    WEIGHT = "weight"
    FRAME_WORKSPACE = "frame_workspace"
    PERSISTENT_STATE = "persistent_state"
    HOST_INPUT = "host_input"
    HOST_READBACK = "host_readback"
    STAGING = "staging"
```

`FRAME_WORKSPACE` 只表示生命周期：一次 run 内的临时 device-local tensor。它不是一个 `frame.workspace.*` 对象树。

### RuntimeSession

```python
class RuntimeSession:
    @classmethod
    def open(cls, *, device_index: int = 0) -> "RuntimeSession": ...

    def empty(
        self,
        *,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        role: TensorRole = TensorRole.ACTIVATION,
        memory: MemoryClass = MemoryClass.FRAME_WORKSPACE,
        compare: ComparePolicy | None = None,
    ) -> LogicalTensor: ...

    def input(
        self,
        *,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        data: object,
    ) -> LogicalTensor: ...

    def load_weight(self, *, name: str, source: WeightSource) -> LogicalTensor: ...

    def dispatch(self, variant: ShaderVariant, **tensors: LogicalTensor) -> None: ...

    def readback(self, tensor: LogicalTensor) -> object: ...

    def close(self) -> None: ...
```

`RuntimeSession` 负责：

1. 创建和销毁 Vulkan device/context；
2. 分配 device-local、host-visible、staging、readback buffer；
3. 上传 input 和 weight；
4. 根据 `LogicalTensor.storage` 绑定 descriptor；
5. 记录 dispatch reads/writes；
6. shader 执行前检查所有 tensor 已 allocated；
7. `close()` 时 `vkDeviceWaitIdle`，再逆序释放 allocation。

## 权重加载

权重声明直接放在 `WeightSource`：

```python
@dataclass(frozen=True, slots=True)
class WeightSource:
    checkpoint: str
    key: str
    dtype: str
    shape: tuple[int, ...]
```

调用：

```python
w = rt.load_weight(
    name="toy.weight.scale",
    source=WeightSource(
        checkpoint="toy.safetensors",
        key="scale",
        dtype="float32",
        shape=(1024,),
    ),
)
```

`load_weight()` 必须：

1. 打开 checkpoint；
2. 检查 key 存在；
3. 检查 dtype 完全一致；
4. 检查 shape 完全一致；
5. 禁止 silent cast；
6. 上传到 device-local allocation；
7. 返回 `storage != None` 的 `LogicalTensor`；
8. allocation owner 注册到 `RuntimeSession`。

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
```

contract 校验：

1. 所有 required field 都传入；
2. 不允许 unknown field；
3. dtype/rank/shape/layout 匹配；
4. symbol shape 能解析；
5. binding 编号不重复；
6. write binding 必须对应 output；
7. shader source binding 和 contract binding 一致。

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
contract.validate(tensors)
for tensor in tensors.values(): tensor.require_storage()
```

然后再创建/复用 pipeline、descriptor set、command buffer。

## 第一个 shader

第一个 shader 选 `elementwise_mul_f32`：

```text
output[i] = x[i] * weight[i]
```

原因：

1. 覆盖 input、weight、output 三类 tensor；
2. 不需要复杂 shape；
3. 不需要 reduction；
4. PyTorch reference 简单；
5. 能快速验证 Vulkan descriptor、pipeline、dispatch、readback。

contract：

```text
x       float32 shape=(N,) read  binding=0
weight  float32 shape=(N,) read  binding=1
output  float32 shape=(N,) write binding=2
dispatch=(ceil_div(N, 256), 1, 1)
```

MVP 可以用 push constant 或 uniform 传 `N`。GLSL 必须 guard 越界 global id。

## DispatchRecord

即使没有 BindingTable，也要记录 dispatch 语义：

```python
@dataclass(frozen=True, slots=True)
class DispatchRecord:
    index: int
    shader: str
    family: str
    reads: Mapping[str, str]
    writes: Mapping[str, str]
    symbols: Mapping[str, int]
    dispatch: tuple[int, int, int]
```

它用于：

1. read-before-write 检查；
2. debug 输出；
3. 后续 liveness/aliasing；
4. 后续 replay；
5. mismatch drilldown。

`reads/writes` 记录 logical tensor name，不记录物理 buffer 名字。

## PyTorch 对拍

MVP 对拍先走 manual reference，不急着接 module hook。

```python
candidate = rt.readback(y)
reference = x_cpu * w_cpu
compare_tensor(y, candidate, reference)
```

`compare_tensor()` 使用 `LogicalTensor.compare`：

```python
@dataclass(frozen=True, slots=True)
class ComparePolicy:
    kind: Literal["tensor", "token"] = "tensor"
    rtol: float = 1e-4
    atol: float = 1e-4
```

失败报告至少包含：

```text
tensor name
shape/dtype mismatch if any
max_abs
max_rel if practical
first mismatch index if practical
```

后续再加：

1. `PyTorchProbe`；
2. module input/output hook；
3. artifact cache；
4. derived transform；
5. boundary drilldown。

## 分阶段实施计划

### Phase 0：基础类型和 LogicalTensor

交付：

1. `src/torch2vk/types.py`；
2. `src/torch2vk/logical.py`；
3. `LogicalTensor.bind()` / `require_storage()`；
4. `tests/test_logical.py`。

验收：

1. name 非空；
2. dtype 支持 `float32/float16/bfloat16/int32/int64`；
3. concrete shape nbytes 正确；
4. unbound tensor `require_storage()` 报清楚 tensor name；
5. `bind()` 返回新对象。

### Phase 1：RuntimeSession dry-run allocator

先不碰 Vulkan，做一个 fake allocation registry。

交付：

1. `BufferAllocation` / `BufferSlice`；
2. `RuntimeSession.empty()`；
3. allocation owner 注册和 close 幂等；
4. `tests/test_runtime_allocation.py`。

验收：

1. `empty()` 返回 bound `LogicalTensor`；
2. offset/nbytes/alignment 合法；
3. session close 后 allocation 全释放；
4. 异常路径释放已注册 allocation。

### Phase 2：权重加载 dry-run + safetensors

交付：

1. `checkpoint.py` safetensors reader；
2. `RuntimeSession.load_weight()`；
3. toy safetensors fixture；
4. dtype/shape/key 校验。

验收：

1. key 缺失时报 `logical name + checkpoint key`；
2. dtype mismatch 不 cast，直接失败；
3. shape mismatch 直接失败；
4. 成功时返回 bound weight tensor；
5. weight allocation 生命周期归 session。

### Phase 3：ShaderContract dry-run

交付：

1. `shader.py`；
2. `ShaderContract.validate()`；
3. `DispatchRecord`；
4. dry-run `RuntimeSession.dispatch()` 只校验和记录，不提交 Vulkan；
5. `tests/test_shader_contract.py`。

验收：

1. 漏 field、多 field、dtype 错、shape 错都失败；
2. unbound tensor 传给 shader 失败；
3. dispatch record 包含 reads/writes；
4. read-before-write 检查能发现错误顺序。

### Phase 4：Vulkan 最小执行

交付：

1. Vulkan instance/device/queue bootstrap；
2. buffer create/bind/map/copy；
3. shader module/pipeline/descriptor/command buffer/submit；
4. `elementwise_mul_f32.comp`；
5. `scripts/compile_shaders.py`；
6. `scripts/run_toy_mvp.py`。

验收：

1. 能列出 compute-capable device；
2. 能上传 input 和 weight；
3. 能执行 `elementwise_mul_f32`；
4. 能 readback output；
5. output 和 CPU reference 一致；
6. session close 后 allocation 全释放。

### Phase 5：PyTorch 对拍

交付：

1. `compare.py`；
2. `test_toy_mvp.py`；
3. manual reference；
4. mismatch 报告。

验收：

1. candidate 必须来自 Vulkan readback；
2. reference 可由 PyTorch/CPU 计算；
3. allclose policy 来自 `LogicalTensor.compare`；
4. mismatch 报告包含 tensor name 和 max_abs。

### Phase 6：第一个真实 op

Toy MVP 稳定后再引入真实 op。建议顺序：

1. `add_f32` 或 `rms_norm_f32`；
2. `linear_f32`，权重必须直接消费 checkpoint 原始 layout；
3. `embedding_lookup`；
4. 再考虑 attention 前的 view/reshape 支持。

每个 op 都必须经过：

```text
LogicalTensor -> RuntimeSession allocation/load_weight -> shader contract -> Vulkan run -> readback -> compare
```

## 什么时候再引入 StoragePlan

MVP 不需要 `StoragePlan`。

只有出现这些需求时再引入：

1. 显存峰值太高，需要 aliasing；
2. 一次性批量规划大量 activation；
3. replay 需要 stable storage fingerprint；
4. 要在运行前估算 memory footprint；
5. 要跨 shader sequence 做 liveness 分析。

引入时也只作为中间计划：

```text
StoragePlan = logical tensor declarations -> allocation decisions
```

它不应该替代 `LogicalTensor.storage`，也不应该变成执行期查询表。

## 什么时候再引入 Frame / Workspace

MVP 不需要。

只有当函数参数太多，影响执行代码可读性时，才引入普通 dataclass 容器：

```python
@dataclass(frozen=True, slots=True)
class ToyTensors:
    x: LogicalTensor
    weight: LogicalTensor
    y: LogicalTensor
```

如果叫 `Frame` / `Workspace`，也必须满足：

1. 只装 `LogicalTensor`；
2. 不装 `BufferAllocation` owner；
3. 不提供 `.activation(name)`；
4. 不负责分配/释放；
5. 不把物理布局暴露给模型执行代码。

## 风险和约束

### Vulkan Python 绑定

当前依赖 Vulkan Python 绑定。MVP 要尽早验证：

1. instance/device 创建；
2. compute queue；
3. shader module 创建；
4. host-visible memory map/unmap；
5. storage buffer descriptor；
6. command submit/wait。

如果 Vulkan 路径阻塞，继续推进 dry-run allocator、shader contract、权重校验和 compare，不让架构停住。

### dtype

PyTorch、checkpoint、LogicalTensor、shader contract 四方 dtype 必须显式一致。MVP 禁止 silent cast。

### shape

执行期必须 concrete shape。symbol shape 只能存在于声明或 contract 中，dispatch 前必须 resolve。

### 资源释放

所有 Vulkan handle owner 必须支持幂等 `close()`。`RuntimeSession.close()` 顺序：

```text
vkDeviceWaitIdle
release readback/staging
release workspace/output/input
release weights
release pipelines/shader modules if owned
release command pool/device/instance
```

## 最小里程碑

第一个可合并版本：

```text
uv run pytest tests/test_logical.py tests/test_runtime_allocation.py tests/test_shader_contract.py
```

第一个可执行版本：

```text
uv run python scripts/run_toy_mvp.py --device 0
```

期望输出：

```text
device: ...
checkpoint: toy.safetensors
shader: elementwise_mul_f32
N: 1024
max_abs: 0.0
result: ok
```

这个版本就是后续真实权重、真实 shader、PyTorch module hook 和 liveness planning 的基线。
