# Frame 和 LogicalTensor

本文是 `torch2vk` 的核心架构文档。后续实现以这里的边界为准。

核心目标：

```text
tensors/ 声明点：LogicalTensor metadata
shaders/ 声明线：ShaderContract + ShaderVariant
eager execution 连接点和线：with rt.frame(...): SHADER_A(rt, ...); SHADER_B(rt, ...)
RuntimeSession 接管执行态：materialize / dispatch / record / compare / release
```

模型目录只表达模型计算。显存分配、释放、Vulkan descriptor、dispatch record、PyTorch 对拍、replay 和后续 liveness/aliasing 优化都属于通用 runtime。

`LogicalTensor` 上的 metadata 只为四件事服务：

1. 权重加载托管给框架，模型目录只声明 `WeightSource`；
2. 显存分配和释放由 runtime 根据 storage/lifetime 决定；
3. candidate 和 PyTorch 对拍时能定位 artifact、校验数值；
4. shader 调用时能校验传入 tensor 和 `ShaderContract` 匹配。

模型代码传给 `ShaderVariant` 的 `LogicalTensor` 应精确对应 GLSL 的 input/output field。若某个 GLSL
view 需要额外信息才能精确描述，优先扩展 `LogicalTensor` metadata，而不是在 `ShaderContract` 里引入
另一套 tensor 语义或默认绑定规则。

不服务于这些目标的字段不进入核心 schema。语义分类可以作为 debug/compare metadata，但不能替代 storage/lifetime 等执行规则。

## 一句话规则

```text
LogicalTensor 声明 tensor 是什么。
ShaderVariant 携带 ShaderContract，声明 op 读写什么。
Frame 声明一次 PyTorch model.forward 对齐的执行边界。
Eager execution 用普通 Python 顺序把 LogicalTensor 传给 `ShaderVariant`。
RuntimeSession 根据 LogicalTensor + Frame 自动分配、复用、释放显存，并记录执行事实。
```

## 基本对象

### LogicalTensor

`LogicalTensor` 是模型语义 tensor 的声明对象。声明阶段它没有 buffer；运行时会被 `RuntimeSession` materialize 成带 `BufferSlice` 的具体 tensor instance。

它描述：

1. tensor 的稳定语义名；
2. dtype、shape、layout；
3. role、storage class、lifetime；
4. 权重来源；
5. 输入 feed 规则；
6. PyTorch probe 和 compare policy；
7. 运行时 materialize 所需的其它 metadata。

声明阶段它不描述：

1. Vulkan buffer；
2. descriptor set；
3. allocation owner；
4. 当前 frame 中的具体 BufferSlice；
5. 当前值是否已经被某个 shader 写出。

推荐定义：

```python
@dataclass(frozen=True, slots=True)
class TensorSpec:
    dtype: str
    shape: tuple[int | str, ...]


@dataclass(frozen=True, slots=True)
class TensorLayout:
    name: str = "row_major"
    params: Mapping[str, int | str] = field(default_factory=dict)


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
    semantic: TensorSemantic | None = None
    compare: ComparePolicy | None = None
    pytorch_probe: PyTorchProbe | None = None
```

不要把 `storage` 放进模型层依赖的 `LogicalTensor` API。`storage` 是执行态事实，应该存在于 `RuntimeSession` 的 materialization table 里的 `MaterializedTensor` / tensor instance 上。

如果调试时需要查看某个 tensor 当前绑定到哪个 buffer，提供 runtime 查询接口：

```python
rt.debug_materialization(tensor, scope=...)
```

而不是让模型代码读写 `tensor.storage`。

### Frame

`Frame` 是一次 PyTorch `model.forward` 对齐的 eager execution 边界。

它同时定义：

1. Vulkan candidate forward 的执行范围；
2. dispatch records 的收集范围；
3. PyTorch hook/probe 的安装范围；
4. compare artifacts 的命名范围；
5. frame workspace 的释放或复用边界。

`Frame` 不是 tensor 容器，不是 workspace tree，也不是模型目录文件。它只是 `RuntimeSession` 上的 scope。

推荐写法是由具体 frame 文件进入 scope，并显式接收对应 PyTorch model：

```python
def run_audio_codec_decoder_frame(
    rt: RuntimeSession,
    tensors: AudioCodecDecoderTensors,
    *,
    pytorch_model: torch.nn.Module,
) -> AudioCodecDecoderOutput:
    with rt.frame(
        "audio_codec_decoder",
        scope={"domain": "audio"},
        pytorch_model=pytorch_model,
    ):
        return run_audio_codec_decoder_shader_sequence(rt, tensors)
```

`with rt.frame(...)` 负责：

```text
进入 Frame:
  设置当前 FrameScope
  设置当前 allocation lifetime context
  开始收集 dispatch records

Frame 内:
  ShaderVariant.__call__ 调 RuntimeSession.dispatch
  RuntimeSession resolve reads/writes 到 materialized tensor instance
  RuntimeSession 绑定 descriptor 并提交 Vulkan dispatch
  RuntimeSession 记录每次 dispatch 的 reads/writes/writer/scope

退出 candidate forward:
  收集本 Frame 实际写出的 LogicalTensors
  过滤 compare != None 且 pytorch_probe != None 的 tensor
  根据 probe metadata 安装 PyTorch hooks
  lockstep 执行当前 Frame 传入的 PyTorch model.forward
  readback candidate tensors
  compare candidate/PyTorch artifact
  释放或复用 FRAME/OP 生命周期资源
```

### RuntimeSession

`RuntimeSession` 是唯一拥有执行态资源的对象。

职责：

1. 创建和销毁 Vulkan instance/device/queue；
2. 管理 buffer allocation、memory mapping、staging、readback；
3. 维护 materialization table；
4. 根据 `LogicalTensor` metadata 自动加载权重、上传输入、从对应 arena/pool 取得 activation/output/state slice；
5. 根据 `ShaderContract` 校验 shader 参数；
6. 绑定 descriptor，提交 compute dispatch；
7. 记录 dispatch records；
8. 在 Frame 结束时执行 PyTorch 对拍；
9. 在 Frame/request/session 生命周期结束时释放资源；
10. 后续基于 dispatch records 生成 replay plan 和 liveness/aliasing plan。

模型代码不调用 `RuntimeSession.empty()`、`RuntimeSession.load_weight()`、`RuntimeSession.free()` 去手工管理 tensor 显存。

## 点、线和连接方式

模型接入只写三类东西：

```text
tensors/ = 点
  声明 LogicalTensor：name/spec/role/memory/lifetime/source/feed/semantic/probe/compare

shaders/ = 线
  声明 ShaderContract + ShaderVariant：读哪些点、写哪些点、参数和 dispatch 规则是什么

execution.py + 具体 frame 文件 = 连接方式
  例如 text_prefill.py、text_decode.py、audio_codec_decoder.py。
  每个文件表达一次 PyTorch model.forward 边界，并用 eager Python 顺序调用 ShaderVariant，把点和线连起来
```

对应关系：

```text
LogicalTensor 是稳定 graph node declaration。
ShaderContract 是 typed op/edge declaration。
frame 文件是 eager graph construction + immediate execution。
Frame 是一次 PyTorch model.forward 对齐的 runtime boundary。
RuntimeSession 是执行态框架。
```

模型目录只表达：

```text
有哪些 tensor 点
有哪些 shader 线
线怎么连接点
点如何映射到 PyTorch probe
```

模型目录不表达：

```text
怎么分配显存
怎么释放显存
怎么绑定 descriptor
怎么 materialize activation
怎么手写 compare
怎么做 replay
怎么做 liveness/aliasing
```

## 命名和 Scope

### LogicalTensor.name

`LogicalTensor.name` 是稳定模型语义名，不包含动态执行上下文。

好：

```text
audio_token_predictor.audio_head.logits
text_llm.decode.layer.03.output
text_llm.state.layer.03.key_cache
audio_token_selector.guided_logits
audio_codec_decoder.decoder.block2.res_unit1.output
audio_codec_decoder.decoder.waveform
output.wav_pcm16
```

坏：

```text
audio.token_005.text_llm.cond.layer.03.output
workspace.core.hidden_states
frame.attention.q_proj
buffer17.slice3
```

audio token index、text token index、chunk index、cond/uncond row、request id 等动态上下文放进 `FrameScope`，不写进 logical tensor base name。

### FrameScope

`FrameScope` 表达一次 frame 的动态上下文。

示例：

```text
frame: text_llm.decode
scope: audio.token_005/row=cond
logical tensor: text_llm.decode.layer.03.output
artifact key: audio.token_005/row=cond/text_llm.decode/text_llm.decode.layer.03.output
```

推荐结构：

```python
@dataclass(frozen=True, slots=True)
class FrameScope:
    frame: str
    values: Mapping[str, str | int]

    def artifact_prefix(self) -> str: ...
```

不要引入泛泛的 `Step` 概念。循环索引用领域名：

```text
audio_token_index
text_token_index
video_frame_index
chunk_index
sample_block_index
```

## Role、Memory、Lifetime

三者分别回答不同问题，且都必须服务于 runtime 决策。

```text
TensorRole: 这个 tensor 在执行关系里大致从哪里来、用作什么？
MemoryClass: runtime materialize 时默认使用哪类 storage/pool？
TensorLifetime: 这个 tensor 至少需要活多久？
```

### TensorRole

推荐枚举：

```text
INPUT       外部输入
WEIGHT      模型权重
ACTIVATION  中间激活
SCRATCH     op 内临时空间
OUTPUT      frame 或 pipeline 输出
STATE       request/pipeline 状态
```

`role` 主要用于默认 materialization 行为、错误信息和调试报告。不要把生命周期、storage class 或模型细分语义只藏在 `role` 里。

`LOGITS`、`TOKEN`、`KV_CACHE` 这类名称不是第一版核心 role。它们可能同时是 activation、output、state 或 input，更适合作为可选 semantic metadata：

```python
class TensorSemantic(StrEnum):
    LOGITS = "logits"
    TOKEN = "token"
    KV_CACHE = "kv_cache"
    MASK = "mask"
    WAVEFORM = "waveform"
```

### MemoryClass

推荐枚举：

```text
MODEL_WEIGHT      model lifetime，只读，通常 device local
REQUEST_STATE     request/pipeline lifetime，跨 Frame 存活
FRAME_WORKSPACE   Frame lifetime，Frame 结束可释放/复用
OP_SCRATCH        单个 shader/op 临时空间
HOST_INPUT        host-visible shader input port
HOST_OUTPUT       host-visible shader output port
```

`MemoryClass` 不是模型语义分类。它描述 runtime materialize 某个 `LogicalTensor` 时默认使用哪类 storage/pool。

`HOST_INPUT` / `HOST_OUTPUT` 仍然是 tensor storage class。它们表示 shader 可通过 descriptor 读写的 host-visible buffer/port，或者 backend 可以在内部选择等价 copy 路径。它们不表示 frame exit 必然自动读回。

### TensorLifetime

推荐枚举：

```text
MODEL      模型加载到卸载，例如 weights
REQUEST    一次 pipeline/request 内存活，例如 generated tokens、KV cache
FRAME      一次 model.forward 内存活，例如 activation
OP         单个 shader/op 内存活，例如 reduction partials
EXTERNAL   runtime 不拥有，例如外部导入 buffer，MVP 可不实现
```

`TensorLifetime` 描述何时可以释放或复用。

### Source 和 Feed metadata

权重 tensor 用 `WeightSource` 声明 checkpoint 来源：

```python
@dataclass(frozen=True, slots=True)
class WeightSource:
    checkpoint: str
    key: str
    dtype: str
    shape: tuple[int, ...]
    layout: TensorLayout = ROW_MAJOR
```

输入 tensor 用 `InputFeed` 声明运行时 feed key：

```python
@dataclass(frozen=True, slots=True)
class InputFeed:
    name: str
    required: bool = True
```

`WeightSource` 和 `InputFeed` 都只是 metadata。实际打开 checkpoint、上传权重、查找输入 feed、分配 buffer 都由 RuntimeSession 完成。

### 合法性规则

Runtime 必须校验 `role/memory/lifetime` 的组合。

推荐默认规则：

```text
source != None
  memory = MODEL_WEIGHT
  lifetime = MODEL

feed != None
  memory = HOST_INPUT 或 REQUEST_STATE
  lifetime = FRAME 或 REQUEST

role == ACTIVATION
  memory = FRAME_WORKSPACE
  lifetime = FRAME

role == SCRATCH
  memory = OP_SCRATCH
  lifetime = OP

role == OUTPUT
  memory = FRAME_WORKSPACE 或 HOST_OUTPUT
  lifetime = FRAME 或 REQUEST

role == STATE
  memory = REQUEST_STATE
  lifetime = REQUEST
```

如果声明显式覆盖默认值，RuntimeSession 必须检查是否合理。典型非法组合：

```text
source != None 但 lifetime 不是 MODEL
source != None 但 memory 不是 MODEL_WEIGHT
ACTIVATION + memory MODEL_WEIGHT
semantic == KV_CACHE 但 lifetime 不是 REQUEST
role == INPUT 但没有 feed 或 runtime feed
role == WEIGHT 但 source == None，除非它是 runtime 已注册的外部 weight
symbolic shape 在 dispatch 前仍未 resolve
```

## Materialization Table

`RuntimeSession` 维护 execution state。

核心结构可以理解为：

```python
@dataclass(frozen=True, slots=True)
class TensorInstanceKey:
    logical_name: str
    scope_key: str
    version: int


@dataclass(slots=True)
class MaterializedTensor:
    tensor: LogicalTensor
    key: TensorInstanceKey
    storage: BufferSlice
    dtype: str
    shape: tuple[int, ...]
    layout: TensorLayout
    writer: DispatchWriter | None
    lifetime: TensorLifetime
```

`BufferSlice` 是 runtime 内部 view，不拥有 allocation：

```python
@dataclass(frozen=True, slots=True)
class BufferSlice:
    allocation_id: int
    offset: int
    nbytes: int
```

allocation owner 只在 `RuntimeSession` 内部 registry 中。

## Allocation 和 Materialization 的边界

不要把 materialization 理解成每个 shader dispatch 都调用底层 Vulkan allocation。

本架构分三层：

```text
Raw allocation
  Vulkan buffer + device memory 的真实 owner。
  只在 session/model/request/frame 边界创建或扩容，不跟单个 shader 绑定。

Arena / pool suballocation
  从 MODEL_WEIGHT、REQUEST_STATE、FRAME_WORKSPACE、OP_SCRATCH 等 pool 中切 BufferSlice。
  可以是 bump allocator、free list、buddy allocator，或后续 StoragePlan 计算出的固定 offset。

Logical materialization
  把某个 LogicalTensor 在当前 FrameScope/version 下映射到一个 BufferSlice。
  dispatch 需要的是这个映射结果，用来绑定 descriptor 和记录 DispatchRecord。
```

`RuntimeSession.dispatch()` 可以触发 logical materialization，也可以从 arena 切一个 slice，但不应该在正常路径上频繁 `vkCreateBuffer` / `vkAllocateMemory`。

### Pool 策略

按 lifetime 分池，避免不同生命周期的 tensor 互相造成碎片：

```text
MODEL_WEIGHT
  model load/register 阶段加载或按需加载。
  常驻，通常 grow-only，模型卸载或 RuntimeSession close 时释放。

REQUEST_STATE
  request/pipeline 开始时创建或扩容。
  KV cache、generated tokens、跨 Frame state 放这里。
  request 结束整体释放或复用。

FRAME_WORKSPACE
  Frame 进入时创建或取得一个 workspace arena。
  activation/output 的 frame-lifetime slice 从这里切。
  Frame exit 后 whole-arena reset，不逐 tensor free。

OP_SCRATCH
  可以并入 FRAME_WORKSPACE 的一段 scratch 区，也可以单独小 arena。
  op 结束后 bump pointer 回退或由 dispatch sequence 复用。

HOST_INPUT / HOST_OUTPUT
  host-visible input/output port pool。适合小输入、控制流标量、debug 边界，或 backend 明确支持 host-visible descriptor 的场景。
  大 tensor 默认仍应放 device-local/frame/request storage，再由显式 RuntimeSession readback API 读取。
```

MVP 的 allocator 可以很简单：

```text
MODEL_WEIGHT      grow-only pool
REQUEST_STATE     grow-only pool per request
FRAME_WORKSPACE   one bump arena per frame，frame exit 整体 reset
OP_SCRATCH        先并入 FRAME_WORKSPACE
HOST_*            host-visible ports / staging buffers 复用或按需扩容
```

这样即使 write materialization 发生在 dispatch 前，也只是从当前 Frame arena 切 slice，不会造成 Vulkan allocation 级别的碎片化。

### Frame workspace 容量

Frame workspace 的容量不应该由单个 shader 随机决定。推荐顺序：

1. MVP：Frame arena 首次需要时按较大 chunk 创建，不够时扩容；Frame exit 整体 reset；
2. dry-run：先跑 contract/materialization dry-run，统计每个 Frame 的 workspace high-water mark；
3. replay：用历史 dispatch records 和 high-water mark 预分配 frame arena；
4. liveness/aliasing：用 StoragePlan 给 frame 内 tensors 分配固定 offset，减少峰值。

模型代码不关心这些策略。它只声明 `LogicalTensor` 并调用 `ShaderVariant`。

### 为什么不把 storage 放进 LogicalTensor

如果 `LogicalTensor` 是 frozen declaration，dispatch 时 resolve 出来的 `BufferSlice` 无法自然回写给后续 op。

可选方案有三个：

1. mutate `LogicalTensor.storage`；
2. 每个 `ShaderVariant` 调用返回新的 bound tensor；
3. runtime 用 materialization table 记录当前 frame/scope 下的 tensor instance。

本架构选择第三个。原因：

1. 模型表达保持纯声明；
2. 同一个 logical name 可以在不同 frame/scope/version 下有不同值；
3. replay、readback、compare、liveness 都能基于统一 runtime state；
4. 不需要模型代码携带 bound/unbound 两套对象。

## Dispatch 规则

`ShaderVariant` 本身是模型代码调用的对象：

```python
CONV1D_F32(rt, x=x, weight=weight, bias=bias, output=output)
```

它的 `__call__` 只进入 `RuntimeSession.dispatch()`，不在模型 adapter 中再包一层 shader 函数。

`RuntimeSession.dispatch()` 必须执行：

```text
1. 检查当前处于 rt.frame(...) 中
2. contract.validate(fields)
3. resolve symbolic shape
4. resolve/materialize read tensor instances
5. reserve/resolve write tensor instances from the current pool/arena
6. 检查 dtype/shape/layout/storage alignment
7. 绑定 descriptor
8. 插入必要 Vulkan barrier
9. 提交 dispatch
10. 记录 DispatchRecord
```

### Read materialization

读取 tensor 时：

```text
WEIGHT/source
  如果已加载，复用 MODEL materialization
  否则读取 checkpoint，校验 dtype/shape/layout，上传到 MODEL_WEIGHT pool
  MODEL_WEIGHT pool 可按 checkpoint/model 分块扩容，但不按 shader 临时分配

INPUT/feed
  从 runtime feed 查数据
  校验 dtype/shape
  materialize 为 HOST_INPUT/REQUEST_STATE/FRAME_WORKSPACE，具体由 tensor.memory 和 backend policy 决定

STATE 或 semantic=KV_CACHE
  从 request materialization table 查当前 instance
  不存在则根据声明初始化，或报错

ACTIVATION/OUTPUT 等上游写出值
  必须已经在当前 FrameScope 或可见上游 scope 中被写出
  未写先读时报错
```

### Write materialization

写入 tensor 时：

```text
根据 tensor.memory/tensor.lifetime 选择 pool
根据 resolved shape/layout 计算 nbytes/alignment
从对应 arena/pool reserve 一个 BufferSlice
创建新的 TensorInstanceKey version
记录 writer shader/dispatch index
注册到 materialization table
```

这里的 reserve 是 pool/arena suballocation，不是底层 Vulkan allocation。FRAME/OP 生命周期的 slice 不逐个 free，Frame exit 统一 reset。MODEL/REQUEST 生命周期的 slice 通常 grow-only，生命周期结束整体释放。

写同一个 logical tensor 时默认创建新 version。是否允许 in-place 由 shader contract 显式声明。

### 跨 Frame 可见性

Frame 内写出的 tensor 默认只在本 Frame 可见。

跨 Frame 使用必须满足至少一个条件：

1. `lifetime == REQUEST`；
2. `role == STATE`；
3. `role == OUTPUT` 且声明为 pipeline output 或 downstream input；
4. pipeline declaration 显式声明下游 Frame 引用该 tensor。

MVP 可以先只支持前两类，再逐步支持 pipeline liveness。

## ShaderContract

推荐结构：

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
    binding: int
    role: str
    contract: TensorContract
    descriptor_type: str = "storage_buffer"


@dataclass(frozen=True, slots=True)
class ShaderContract:
    name: str
    fields: tuple[TensorFieldSpec, ...]
    dispatch: tuple[int | str, int | str, int | str]
    push_constants: Mapping[str, str] = field(default_factory=dict)
    specialization_constants: Mapping[int, int] = field(default_factory=dict)
```

contract 校验至少包括：

1. required field 必须传入；
2. 不允许 unknown field；
3. dtype/rank/shape/layout 必须匹配；
4. symbolic shape 必须能在 dispatch 前 resolve；
5. field name 和 binding 编号都不重复；
6. `INPUT` field 未 materialized 时必须能按 read rules materialize；
7. `OUTPUT` field 的 role/memory/lifetime 合法；
8. `INOUT` field 必须同时记录 read 和 write；
9. shader source 里的 descriptor binding 和 contract 一致；
10. dispatch size 必须是 concrete int。

## DispatchRecord

每次 shader dispatch 都必须记录执行事实：

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

记录 `TensorInstanceKey` 是为了准确 readback、compare、replay 和 liveness。记录 logical name 是为了 debug 报告可读。

Frame 结束后：

```text
used tensors = all reads + all writes
candidate boundary tensors = writes where compare != None and pytorch_probe != None
```

默认只比较 write tensors，因为它们是 candidate 本次 forward 实际产生的值。

## PyTorch 对拍

对拍由 candidate frame 驱动：

```text
1. 进入 rt.frame(...)
2. 跑 Vulkan candidate eager forward
3. RuntimeSession 收集 dispatch records 和 written TensorInstanceKey
4. candidate forward 结束后，筛选 compare/probe tensors
5. 根据每个 LogicalTensor.pytorch_probe 安装 PyTorch hooks
6. lockstep 跑一次传入 Frame 的 PyTorch model.forward
7. hooks 收集 PyTorch artifacts
8. RuntimeSession readback 对应 candidate TensorInstanceKey
9. 按 LogicalTensor.compare 比较
10. 报告 mismatch
11. 释放或复用 Frame workspace
```

PyTorch model 不决定比较哪些 tensors。RuntimeSession 只根据 candidate 实际写出的 tensors 以及这些 `LogicalTensor` 自带的 `pytorch_probe` metadata 安装 hook/probe，然后 lockstep 执行当前 Frame 传入的 PyTorch model。

### PyTorchProbe

推荐结构：

```python
@dataclass(frozen=True, slots=True)
class PyTorchProbe:
    kind: Literal["module_input", "module_output", "manual_hook", "derived"]
    target: str
    index: int = 0
    selector: str | None = None
    transform: str | None = None
```

`PyTorchProbe` 是 `LogicalTensor` 的一部分，不是独立的平行系统。RuntimeSession 根据 candidate 写出的 LogicalTensor 读取其中的 `pytorch_probe`，再在当前 Frame 传入的 PyTorch model 上安装 hook。

禁止在任何独立 registry 或 helper 里手写 candidate 公式：

```python
# bad
return {"audio_codec_decoder.decoder.conv1": conv1d(x, w, b)}
```

PyTorch artifact 必须来自当前 Frame 传入的 PyTorch model.forward 过程中的真实 tensor。`derived` 只允许表达 dtype/layout/shape 变换等边界变换，不能重新实现模型 op。

### ComparePolicy

推荐结构：

```python
@dataclass(frozen=True, slots=True)
class ComparePolicy:
    kind: Literal["tensor", "token", "waveform"]
    rtol: float = 1e-4
    atol: float = 1e-4
    max_abs: float | None = None
```

失败报告至少包含：

```text
frame name
scope
artifact key
logical tensor name
candidate TensorInstanceKey
writer shader / dispatch index
candidate shape/dtype
PyTorch artifact shape/dtype
max_abs / max_rel
first mismatch index if practical
```

## 显存释放规则

RuntimeSession 拥有所有 allocation。LogicalTensor 不拥有 allocation。

释放边界：

```text
OP exit
  recycle OP_SCRATCH

Frame exit
  compare/readback 完成后 release/reuse FRAME_WORKSPACE
  keep REQUEST_STATE，包括 semantic=KV_CACHE 的 state
  keep MODEL_WEIGHT
  keep explicitly exported pipeline outputs

Request/Pipeline exit
  release REQUEST_STATE
  release generated token buffers
  release final host readback buffers if not returned
  keep MODEL_WEIGHT

RuntimeSession close
  vkDeviceWaitIdle
  release all buffers and memory
  release pipelines/shader modules
  release command pools
  release device/instance
```

MVP 可以先不做 tensor-level aliasing，但必须用 pool/arena 避免 per-shader raw allocation。后续 liveness/aliasing 只能优化 runtime 的 offset 分配策略，不能改变模型目录的表达方式。

## Replay 和优化边界

初期 eager execution 每次都由 Python 调 `ShaderVariant`。

后续 replay 优化由 RuntimeSession 从 dispatch records 生成：

```text
Frame eager execution
  -> dispatch records
  -> replay plan
  -> cached pipelines/descriptors/command buffers
  -> liveness/aliasing storage plan
```

Replay 只能是 runtime 优化，不允许模型目录为了 replay 改写成另一套 graph IR。模型表达仍然是点、线、eager 连接。

## MVP 规则

第一版必须坚持：

1. `LogicalTensor` 是声明对象，不携带模型代码可依赖的 `storage`；
2. 模型 forward 必须在 `with rt.frame(...)` 内执行；
3. 模型 forward 只调用 `ShaderVariant`，不手工分配/free 显存；
4. `RuntimeSession.dispatch()` resolve/materialize reads/writes，但不做 per-shader raw Vulkan allocation；
5. weights model-lifetime 常驻；
6. inputs/request state request-lifetime 或 frame-lifetime；
7. frame workspace 在 frame exit 后释放或复用；
8. 不做 aliasing；
9. 不做复杂 replay；
10. dispatch record 必须足够支持 readback、compare 和后续 replay。

## 禁止事项

```text
不要让模型 forward 调用 RuntimeSession.empty/load_weight/free
不要让模型目录写 materialize.py 管显存
不要暴露 frame.workspace.* 物理 slot tree
不要把 BufferSlice 放进模型语义命名
不要把 audio/video/text index 或 row 写进 LogicalTensor.name
不要由 PyTorch model 决定 compare tensors
不要 silent dtype cast
不要把 replay plan 当成模型表达源头
```
