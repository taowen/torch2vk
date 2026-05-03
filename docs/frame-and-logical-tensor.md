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

1. 权重加载托管给框架，权重 tensor 只用 `LogicalTensor.name/spec/layout` 表达 checkpoint key、dtype、shape、layout；
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

`LogicalTensor` 是模型语义 tensor 的稳定对象。它既是模型 adapter 传给 shader field 的声明对象，也由
`RuntimeSession` 在运行时维护当前 allocation/buffer 状态。没有一份和 `LogicalTensor` 平行维护的 binding
table。

任何需要引用 tensor 的地方都直接持有 `LogicalTensor` 对象。`LogicalTensor.name` 用于稳定报告、
debug、artifact 路径和重复声明检查；当 `role == WEIGHT` 时，它同时也是 checkpoint tensor key。

它描述：

1. tensor 的稳定语义名；
2. dtype、shape、layout；
3. role、storage class、lifetime；
4. 权重 checkpoint key 规则；
5. 运行时输入绑定规则；
6. PyTorch probe 和 compare policy；
7. 运行时 materialize 所需的其它 metadata；
8. 当前是否已经分配，以及当前对应的 buffer slice、descriptor range、writer/version 等执行态状态。

模型代码不手动维护这些执行态字段：

1. 不创建 Vulkan buffer；
2. 不绑定 descriptor set；
3. 不拥有 allocation；
4. 不手动填写或清空当前 buffer slice；
5. 不手动标记当前值是否已经被某个 shader 写出。

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

这些 runtime 字段属于 `RuntimeSession` 管理的执行态。模型 adapter 可以稳定地把同一个 `LogicalTensor`
传给 shader input/output，但不应该直接分配、释放或改写它的 buffer 状态。

如果调试时需要查看某个 tensor 当前绑定到哪个 buffer，提供 runtime 查询接口：

```python
rt.debug_materialization(tensor, scope=...)
```

查询结果来自这个 `LogicalTensor` 当前记录的执行态状态，而不是另一套 binding registry。

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
  RuntimeSession 收集本次 dispatch 实际使用的 LogicalTensor
  RuntimeSession 按 dispatch read/write resolve/materialize，并更新 LogicalTensor 当前 buffer 状态
  RuntimeSession 绑定 descriptor 并提交 Vulkan dispatch
  RuntimeSession 记录每次 dispatch 的 reads/writes/writer/scope

Replay Frame enter:
  使用 capture 得到的 dispatch records
  预加载本 Frame 会读到的权重
  预分配或绑定 frame/request arena
  根据 liveness/aliasing 直接分配或复用 arena offset

退出 candidate forward:
  收集本 Frame 实际写出的 LogicalTensors
  过滤 compare != None 且 pytorch_probe != None 的 tensor
  根据 probe metadata 安装 PyTorch hooks
  从 register_inputs() 中按 frame prefix + PyTorch forward 参数名推断 kwargs
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
3. 维护 `LogicalTensor` 当前 buffer 状态；
4. 根据 `LogicalTensor` metadata 自动加载权重、上传输入、从对应 arena/pool 取得 activation/output/state slice，并写回 `LogicalTensor`；
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
  声明 LogicalTensor：name/spec/role/memory/lifetime/semantic/probe/compare

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

### 权重和运行时输入

权重 tensor 不带单独的来源对象。`role == WEIGHT` 时，runtime 使用 `LogicalTensor` 自身字段推断
checkpoint view：

```text
checkpoint key = LogicalTensor.name
dtype          = LogicalTensor.spec.dtype
shape          = LogicalTensor.spec.shape
layout         = LogicalTensor.layout
checkpoint     = RuntimeSession.model_dir 下的 canonical safetensors
```

所以权重 `LogicalTensor.name` 必须和 checkpoint tensor key 完全一致；如果模型 adapter 想保留更短的本地
别名，应在声明阶段把别名解析成 checkpoint key，而不是把映射作为 runtime metadata 传下去。

输入 tensor 不需要额外 metadata。`role == INPUT` 的 `LogicalTensor` 直接作为运行时输入 key：

```python
rt.register_inputs({input_tensor: input_array})
```

实际打开 checkpoint、上传权重、查找输入、分配 buffer 都由 RuntimeSession 完成。

### 合法性规则

Runtime 必须校验 `role/memory/lifetime` 的组合。

推荐默认规则：

```text
role == WEIGHT
  memory = MODEL_WEIGHT
  lifetime = MODEL

role == INPUT
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
WEIGHT 但 lifetime 不是 MODEL
WEIGHT 但 memory 不是 MODEL_WEIGHT
ACTIVATION + memory MODEL_WEIGHT
semantic == KV_CACHE 但 lifetime 不是 REQUEST
role == INPUT 但没有 runtime input binding
symbolic shape 在 dispatch 前仍未 resolve
```

## LogicalTensor Runtime State

`RuntimeSession` 维护 execution state，但这个 state 记录在对应的 `LogicalTensor` 上，而不是一份平行的
materialization table。

`BufferSlice` 是 runtime 内部 view，不拥有 allocation：

```python
@dataclass(frozen=True, slots=True)
class BufferSlice:
    allocation_id: int
    offset: int
    nbytes: int
```

allocation owner 只在 `RuntimeSession` 内部 registry 中。

`LogicalTensor.buffer` 指向当前 `BufferSlice`，`descriptor_nbytes` 记录 descriptor range，`version` /
`writer` 记录当前值由哪个 dispatch 写出。读写同一个 `LogicalTensor` 时，`RuntimeSession.dispatch()`
直接检查并更新这些字段。

## Allocation 和 Materialization 的边界

不要把 materialization 理解成每个 shader dispatch 都调用底层 Vulkan allocation。

本架构分三层：

```text
Raw allocation
  Vulkan buffer + device memory 的真实 owner。
  只在 session/model/request/frame 边界创建或扩容，不跟单个 shader 绑定。

Arena / pool suballocation
  从 MODEL_WEIGHT、REQUEST_STATE、FRAME_WORKSPACE、OP_SCRATCH 等 pool 中切 BufferSlice。
  可以是 bump allocator、free list、buddy allocator，或 replay 根据 dispatch records/liveness 直接选择的 offset。

Logical materialization
  RuntimeSession 根据 LogicalTensor 的 metadata 分配或复用 BufferSlice。
  分配结果直接写回 LogicalTensor 当前 buffer 状态。
  dispatch 使用 LogicalTensor 当前 buffer 状态绑定 descriptor 和记录 DispatchRecord。
```

`RuntimeSession.dispatch()` 可以触发 logical materialization，也可以创建、扩容或从 arena 切一个 slice。record/eager 阶段不追求这里的高性能；replay 热路径应根据录制结果在 Frame enter 完成预加载和 arena 准备，避免临场 allocation。

### Pool 策略

按 lifetime 分池，避免不同生命周期的 tensor 互相造成碎片：

```text
MODEL_WEIGHT
  record/eager 阶段由 shader read 按需加载。
  replay 阶段可根据录制结果在 Frame enter 预加载或校验。
  常驻，通常 grow-only，模型卸载或 RuntimeSession close 时释放。

REQUEST_STATE
  request/pipeline 开始时创建或扩容。
  KV cache、generated tokens、跨 Frame state 放这里。
  request 结束整体释放或复用。

FRAME_WORKSPACE
  record/eager 阶段首次 dispatch 需要时创建或扩容 workspace arena。
  replay 阶段根据 capture high-water mark 或 dispatch records 在 Frame enter 预分配。
  activation/output 的 frame-lifetime slice 从这里切，或由 replay liveness 直接指定 offset。
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

record/eager 阶段可以简单按需分配来暴露真实需求；replay 阶段再用 high-water mark 和 dispatch records 把这些需求收敛成稳定的 arena 分配。

### Frame workspace 容量

Frame workspace 的容量不应该由单个 shader 随机决定。推荐顺序：

1. MVP/record：Frame arena 首次 dispatch 需要时按较大 chunk 创建，不够时扩容；Frame exit 整体 reset；
2. dry-run：先跑 contract/materialization dry-run，统计每个 Frame 的 workspace high-water mark；
3. replay：用历史 dispatch records 和 high-water mark 预分配 frame arena；
4. liveness/aliasing：直接用 dispatch records 给 frame 内 tensors 分配或复用 offset，减少峰值。

模型代码不关心这些策略。它只声明 `LogicalTensor` 并调用 `ShaderVariant`。

### 为什么把当前 buffer 状态放在 LogicalTensor 上

`LogicalTensor` 是 shader input/output 的稳定对象。RuntimeSession 在 dispatch 时把分配结果直接写回这个
对象，可以避免模型目录同时携带 bound/unbound 两套对象，也避免 runtime 维护一份和 `LogicalTensor`
平行的 binding table 或 `name -> LogicalTensor` registry。

约束是：

1. 模型 adapter 只传递 `LogicalTensor`，不手动分配、释放或改写 buffer 状态；
2. RuntimeSession 是 `LogicalTensor.buffer` / `descriptor_nbytes` / `version` / `writer` 的唯一写入者；
3. 同一个 logical name 在不同 scope 下需要不同值时，由模型目录声明不同 `LogicalTensor` 实例或由 runtime 在进入 scope 时重置/切换其当前状态；
4. replay、readback、compare、liveness 都基于 `LogicalTensor` 当前状态和 dispatch records。

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
4. resolve/materialize read `LogicalTensor`
5. reserve/resolve write `LogicalTensor` from the current pool/arena
6. 检查 dtype/shape/layout/storage alignment
7. 绑定 descriptor
8. 插入必要 Vulkan barrier
9. 提交 dispatch
10. 记录 DispatchRecord
```

### Read materialization

读取 tensor 时：

```text
WEIGHT
  如果已加载，复用 MODEL materialization
  如果未加载，record/eager dispatch 用 LogicalTensor.name 作为 checkpoint key，校验 dtype/shape/layout 并上传权重
  dispatch records 记录本次实际读取了该 weight
  replay 根据 dispatch records 可在 Frame enter 预加载或校验该 weight
  MODEL_WEIGHT pool 可按 checkpoint/model 分块扩容，热路径不应依赖临场上传

INPUT
  从 runtime input binding 查数据
  校验 dtype/shape
  materialize 为 HOST_INPUT/REQUEST_STATE/FRAME_WORKSPACE，具体由 tensor.memory 和 backend policy 决定

STATE 或 semantic=KV_CACHE
  从 LogicalTensor 当前 buffer 状态检查 request-lifetime storage
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
递增 LogicalTensor.version
记录 writer shader/dispatch index
更新 LogicalTensor 当前 buffer 状态
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
5. field name 不重复；
6. `INPUT` field 未 materialized 时必须能按 read rules materialize；
7. `OUTPUT` field 的 role/memory/lifetime 合法；
8. `INOUT` field 必须同时记录 read 和 write；
9. shader source 里的 descriptor binding 和 field 顺序一致；
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
    reads: Mapping[str, LogicalTensor]
    writes: Mapping[str, LogicalTensor]
    logical_reads: Mapping[str, str]
    logical_writes: Mapping[str, str]
    symbols: Mapping[str, int]
    dispatch_size: tuple[int, int, int]
```

记录 `LogicalTensor` 是为了准确 readback、compare、replay 和 liveness。记录 logical name 是为了 debug 报告可读。

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
3. RuntimeSession 收集 dispatch records 和 written LogicalTensors
4. candidate forward 结束后，筛选 compare/probe tensors
5. 根据每个 LogicalTensor.pytorch_probe 安装 PyTorch hooks
6. 从 `register_inputs()` 中按当前 frame name 和 PyTorch forward 签名推断 input kwargs
7. lockstep 跑一次传入 Frame 的 PyTorch model.forward
8. hooks 收集 PyTorch artifacts
9. RuntimeSession readback 对应 candidate LogicalTensor 的当前 buffer
10. 按 LogicalTensor.compare 比较
11. 报告 mismatch
12. 释放或复用 Frame workspace
```

PyTorch model 不决定比较哪些 tensors。RuntimeSession 只根据 candidate 实际写出的 tensors 以及这些 `LogicalTensor` 自带的 `pytorch_probe` metadata 安装 hook/probe，然后 lockstep 执行当前 Frame 传入的 PyTorch model。
PyTorch model 也不接收测试或 frame function 旁路传入的第二份输入。RuntimeSession 使用同一批
`register_inputs()` 输入：`role == INPUT`、logical name 以当前 frame name 为前缀、前缀后的 basename 命中
`forward` 参数名的 tensor，会作为 PyTorch kwargs 传入。例如
`qwen3_asr.audio_tower.feature_lens` 自动对应 `forward(..., feature_lens=...)`。

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
candidate LogicalTensor / current buffer
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

MVP 可以先不做 tensor-level aliasing。record/eager 可以按需 allocation；后续 liveness/aliasing 只能优化 runtime 的 offset 分配策略，不能改变模型目录的表达方式。

## Replay 和优化边界

初期 eager execution 每次都由 Python 调 `ShaderVariant`。

后续 replay 优化由 RuntimeSession 从 dispatch records 生成：

```text
Frame eager execution
  -> dispatch records
  -> replay plan
  -> weight preload from dispatch reads
  -> cached pipelines/descriptors/command buffers
  -> liveness/aliasing arena offsets
```

Replay 只能是 runtime 优化，不允许模型目录为了 replay 改写成另一套 graph IR。模型表达仍然是点、线、eager 连接。

## MVP 规则

第一版必须坚持：

1. `LogicalTensor` 是 shader IO 的稳定对象，runtime 在其上维护当前 buffer 状态；
2. 模型 forward 必须在 `with rt.frame(...)` 内执行；
3. 模型 forward 只调用 `ShaderVariant`，不手工分配/free 显存；
4. `RuntimeSession.dispatch()` resolve/materialize reads/writes；record/eager 允许按需 allocation，replay 用录制结果提前准备；
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
