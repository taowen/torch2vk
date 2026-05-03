# Frame 和 LogicalTensor

目标：定义 `torch2vk` 里显存分配/释放和 PyTorch 对拍的基本边界。

核心结论：

```text
Frame = 一次 PyTorch model.forward 对应的 Vulkan candidate forward 边界。
LogicalTensor = Frame 内外传递的模型语义 tensor 句柄。
RuntimeSession = 根据 LogicalTensor metadata 和当前 Frame 自动管理显存。
```

这里的 `Frame` 不是旧实现里的 `frame.workspace.*` 物理 slot tree。新的 `Frame` 是执行、对拍、显存生命周期的边界。

## 基本术语

### Pipeline

Pipeline 是多个 Frame 的组合，可以串联，也可以在 generation loop 中重复执行。

OmniVoice 示例：

```text
Pipeline: omnivoice audio generation

Audio token index 0:
  Frame: audio_token_predictor.forward
  Frame: text_llm_cond.decode
  Frame: text_llm_uncond.decode
  Frame: audio_token_selector.forward

Audio token index 1:
  Frame: audio_token_predictor.forward
  Frame: text_llm_cond.decode
  Frame: text_llm_uncond.decode
  Frame: audio_token_selector.forward

After audio tokens are generated:
  Frame: audio_codec_decoder.forward
```

### Frame

Frame 是一次 PyTorch `model.forward` 对应的边界。

一个 Frame 同时负责：

1. Vulkan candidate forward 的执行范围；
2. candidate dispatch records 的收集范围；
3. PyTorch hook/probe 的安装范围；
4. compare artifacts 的对齐范围；
5. frame workspace 显存的释放/复用边界。

### Domain index

不要把 `Step` 作为核心 runtime 概念。循环索引用具体领域命名，例如：

```text
audio_token_index
video_frame_index
text_token_index
chunk_index
sample_block_index
```

例如第 5 个 audio token 生成时可能有多个 Frame：

```text
audio.token_005/audio_token_predictor.forward
audio.token_005/text_llm_cond.decode
audio.token_005/text_llm_uncond.decode
audio.token_005/audio_token_selector.forward
```

这些 index 只进入 `FrameScope` / artifact key，不进入 `LogicalTensor.name`。

### Workspace

Workspace 是 Frame 内部临时显存类别，不是模型目录的一等 API。

不要暴露：

```python
frame.workspace.attention.q_proj.activation("text_llm.decode.layer.03.q_proj")
```

模型 forward 只能传递 `LogicalTensor`。


## 点、线和 eager execute

模型接入的核心抽象可以简化成：

```text
tensors/ = 点
  声明 LogicalTensor：name/spec/role/memory/lifetime/source/probe/compare

shaders/ = 线
  声明 ShaderContract + wrapper：读哪些点，写哪些点，怎么 dispatch

execution.py + submodel/forward.py = 连接方式
  用 eager Python 顺序调用 shader wrapper，把点和线连成 candidate forward
```

对应关系：

```text
LogicalTensor 是 graph node。
Shader 是 typed edge / op。
forward.py 是 eager graph construction + execution。
Frame 是一次 PyTorch model.forward 对齐的执行/对拍/lifetime 边界。
RuntimeSession 是 runtime：自动 bind/load/allocate/free、记录 dispatch、驱动 PyTorch probe、执行 compare。
```

模型目录只表达计算语义和执行顺序，不表达资源管理：

```text
写什么：有哪些 tensor 点、有哪些 shader 线、线怎么连接点、点怎么 probe 到 PyTorch。
不写什么：怎么分配显存、怎么释放显存、怎么 materialize、怎么手写 compare。
```

## Eager execute 里的 Frame

在 eager execute 中，`Frame` 表达成 `RuntimeSession` 上的 scope。

推荐写法：

```python
with rt.frame(
    "audio_codec_decoder.forward",
    scope={"domain": "audio"},
    pytorch=audio_codec_decoder_pytorch,
    probes=audio_codec_decoder_probes,
):
    waveform = run_audio_codec_decoder_forward(rt, tensors.audio_codec_decoder)
```

`forward.py` 仍然只负责 eager 调 shader：

```python
def run_audio_codec_decoder_forward(
    rt: RuntimeSession,
    tensors: AudioCodecDecoderTensors,
) -> AudioCodecDecoderOutput:
    embedding_sum_f32(
        rt,
        tokens=tensors.audio_tokens,
        weight=tensors.weights.quantizer_embed,
        output=tensors.quantizer_embed_sum,
    )
    conv1d_f32(
        rt,
        x=tensors.quantizer_embed_sum,
        weight=tensors.weights.decoder_conv1,
        bias=tensors.weights.decoder_conv1_bias,
        output=tensors.decoder.conv1,
    )
    return AudioCodecDecoderOutput(waveform=tensors.decoder.waveform)
```

`with rt.frame(...)` 负责：

```text
进入 Frame:
  设置当前 allocation lifetime context
  设置当前 artifact scope
  开始收集 dispatch records

Frame 内:
  shader dispatch lazy bind/allocate LogicalTensor
  记录 reads/writes

退出 candidate forward:
  收集本 Frame written tensors with compare/probe
  根据 collected LogicalTensors 安装 PyTorch hooks
  跑一次 PyTorch model.forward
  readback candidate tensors
  compare candidate/reference
  释放或复用 Frame workspace
```

OmniVoice pipeline 的 eager 写法示例：

```python
def run_omnivoice_pipeline(
    rt: RuntimeSession,
    tensors: OmniVoiceTensors,
    *,
    max_audio_tokens: int,
) -> LogicalTensor:
    selected = None
    for audio_token_index in range(max_audio_tokens):
        with rt.frame(
            "audio_token_predictor.forward",
            scope={"audio_token_index": audio_token_index},
            pytorch=tensors.audio_token_predictor.pytorch,
            probes=tensors.audio_token_predictor.probes,
        ):
            predictor_out = run_audio_token_predictor_forward(
                rt,
                tensors.audio_token_predictor,
                audio_token_index=audio_token_index,
            )

        with rt.frame(
            "text_llm_cond.decode",
            scope={"audio_token_index": audio_token_index, "row": "cond"},
            pytorch=tensors.text_llm.cond_pytorch,
            probes=tensors.text_llm.probes,
        ):
            cond = run_text_llm_decode_forward(rt, tensors.text_llm_cond, predictor_out)

        with rt.frame(
            "text_llm_uncond.decode",
            scope={"audio_token_index": audio_token_index, "row": "uncond"},
            pytorch=tensors.text_llm.uncond_pytorch,
            probes=tensors.text_llm.probes,
        ):
            uncond = run_text_llm_decode_forward(rt, tensors.text_llm_uncond, predictor_out)

        with rt.frame(
            "audio_token_selector.forward",
            scope={"audio_token_index": audio_token_index},
            pytorch=tensors.audio_token_selector.pytorch,
            probes=tensors.audio_token_selector.probes,
        ):
            selected = run_audio_token_selector_forward(
                rt,
                tensors.audio_token_selector,
                cond,
                uncond,
            )

        if selected.done:
            break

    with rt.frame(
        "audio_codec_decoder.forward",
        scope={"domain": "audio"},
        pytorch=tensors.audio_codec_decoder.pytorch,
        probes=tensors.audio_codec_decoder.probes,
    ):
        waveform = run_audio_codec_decoder_forward(
            rt,
            tensors.audio_codec_decoder,
            selected.tokens,
        )

    return waveform
```

`Frame` 不是 tensor 容器，不是 workspace tree，也不是模型目录文件。它只是 eager execute 里的 runtime boundary 标记。

## LogicalTensor

`LogicalTensor` 描述 tensor 的模型语义身份和 runtime materialization metadata。

建议字段：

```python
@dataclass(frozen=True, slots=True)
class LogicalTensor:
    name: str
    spec: TensorSpec
    role: TensorRole
    memory: MemoryClass
    lifetime: TensorLifetime
    storage: BufferSlice | None = None
    source: WeightSource | None = None
    compare: ComparePolicy | None = None
    pytorch_probe: PyTorchProbe | None = None
```

### name

`name` 是模型语义名，不包含 audio/video/text index、row 或 scope。

好：

```text
text_llm.decode.layer.03.output
audio_codec.decoder.block2.res_unit1.output
audio_codec.decoder.waveform
```

坏：

```text
audio.token_005.text_llm.cond.layer.03.output
workspace.core.hidden_states
buffer17.slice3
```

audio/video/text index、row、frame 信息放在 `FrameScope` 或 artifact key 中。

### role

`role` 描述 tensor 的语义角色：

```text
INPUT
WEIGHT
ACTIVATION
SCRATCH
OUTPUT
STATE
KV_CACHE
LOGITS
TOKEN
```

RuntimeSession 用 `role` 决定默认 materialization 行为。

### memory

`memory` 描述显存类别：

```text
MODEL_WEIGHT       model lifetime，通常 device local，只读
REQUEST_STATE      request lifetime，跨 Frame 存活
FRAME_WORKSPACE    frame lifetime，Frame 结束可释放/复用
FRAME_OUTPUT       Frame 输出，给后续 Frame 使用
HOST_INPUT         host input/upload
HOST_READBACK      readback/debug/最终输出
```

### lifetime

`lifetime` 描述显存生命周期：

```text
MODEL      模型加载到卸载，例如 weights
REQUEST    一次 pipeline/request 内存活，例如 KV cache、generated tokens
FRAME      一次 model.forward 内存活，例如 activation/workspace
OUTPUT     Frame 结束后仍要给下游 Frame 使用
OP         单个 shader/op 临时 scratch
```

`memory` 是放哪类池，`lifetime` 是何时可释放/复用。

### source

权重 tensor 必须带 `WeightSource`：

```python
@dataclass(frozen=True, slots=True)
class WeightSource:
    checkpoint: str
    key: str
    dtype: str
    shape: tuple[int, ...]
```

RuntimeSession 根据它自动加载权重。

### compare / pytorch_probe

需要对拍的 tensor 必须声明：

```python
compare=ComparePolicy(...)
pytorch_probe=PyTorchProbe(...)
```

candidate forward 实际写出这个 tensor 后，RuntimeSession 才会在 reference forward 时安装对应 hook/probe。

## 显存分配规则

模型代码不应该手写显存分配和释放。它只声明 `LogicalTensor`，并在 forward 里把这些 tensor 传给 shader wrapper。

RuntimeSession 在 Frame 内 lazy materialize：

```text
shader dispatch sees LogicalTensor without storage
  -> RuntimeSession checks role/memory/lifetime/source
  -> allocate/load/upload according to metadata
  -> bind BufferSlice into LogicalTensor or runtime-owned bound copy
  -> dispatch shader
```

推荐规则：

```text
role == WEIGHT or source != None
  -> load checkpoint once
  -> upload to MODEL_WEIGHT pool
  -> lifetime MODEL

role == INPUT
  -> lookup feed by logical name
  -> validate dtype/shape
  -> upload or bind host-visible memory
  -> lifetime REQUEST or FRAME, depending declaration

role == ACTIVATION or SCRATCH
  -> allocate from current Frame workspace arena
  -> lifetime FRAME or OP

role == OUTPUT
  -> allocate as Frame output
  -> retain after Frame exit if downstream needs it

role == STATE or KV_CACHE
  -> allocate/reuse request persistent arena
  -> lifetime REQUEST
```

如果 metadata 不足，RuntimeSession 必须报错，不允许模型代码临时补显存逻辑：

```text
WEIGHT missing source -> error
INPUT missing feed -> error
symbolic shape unresolved -> error
dtype mismatch -> error
shape mismatch -> error
duplicate logical name with incompatible spec -> error
```

## 显存释放规则

RuntimeSession 拥有所有 allocation。`LogicalTensor.storage` 是 view，不拥有 allocation。

释放边界：

```text
OP exit
  recycle OP scratch

Frame exit
  release/reuse FRAME_WORKSPACE
  keep FRAME_OUTPUT if exported to downstream
  keep REQUEST_STATE / KV_CACHE
  keep MODEL_WEIGHT

Request/Pipeline exit
  release REQUEST_STATE
  release generated token buffers
  release final host readback buffers if not returned
  keep MODEL_WEIGHT

Model unload / RuntimeSession close
  release MODEL_WEIGHT
  release pipelines/shader modules/device resources
```

Frame 结束时，RuntimeSession 需要知道哪些 tensor 是 Frame 输出。

Frame 输出可以通过两种方式表达：

1. `role=OUTPUT` 或 `lifetime=OUTPUT`；
2. pipeline 显式声明下游 Frame inputs 引用该 tensor。

MVP 可以先用第一种，后续再做跨 Frame liveness 分析。

## Frame scope 职责

`with rt.frame(...)` 是唯一推荐的 eager execute Frame 表达方式。它不替代 Python forward，也不隐藏 shader 调用顺序；它只给 RuntimeSession 标记一次 PyTorch `model.forward` 对齐的边界。

Frame scope 负责：

1. 设置当前 allocation lifetime context；
2. 设置当前 domain scope，例如 `audio_token_index`、`row`、`domain`；
3. 收集 dispatch records；
4. 收集 used/written LogicalTensors；
5. 在 candidate forward 结束后驱动 PyTorch probes；
6. readback 并按 `LogicalTensor.compare` 对拍；
7. 在 Frame exit 时释放/reuse workspace。

## Candidate forward

每个 submodel 的 `forward.py` 只跑 Vulkan candidate。

示例：

```python
def run_audio_codec_decoder_forward(
    rt: RuntimeSession,
    tensors: AudioCodecDecoderTensors,
) -> AudioCodecDecoderOutput:
    embedding_sum_f32(
        rt,
        tokens=tensors.audio_tokens,
        weight=tensors.weights.quantizer_embed,
        output=tensors.quantizer_embed_sum,
    )
    conv1d_f32(
        rt,
        x=tensors.quantizer_embed_sum,
        weight=tensors.weights.decoder_conv1,
        bias=tensors.weights.decoder_conv1_bias,
        output=tensors.decoder.conv1,
    )
    return AudioCodecDecoderOutput(waveform=tensors.decoder.waveform)
```

`forward.py` 禁止：

```text
不调用 PyTorch model
不做 compare
不读 checkpoint
不直接分配/free 显存
不从物理 workspace slot 生成 LogicalTensor
```

## Dispatch record

每次 shader dispatch 必须记录：

```python
@dataclass(frozen=True, slots=True)
class DispatchRecord:
    index: int
    frame: str
    scope: FrameScope
    shader: str
    reads: Mapping[str, LogicalTensor]
    writes: Mapping[str, LogicalTensor]
    symbols: Mapping[str, int]
    dispatch_size: tuple[int, int, int]
```

Frame 结束后收集：

```text
used tensors = all reads + all writes
candidate boundary tensors = writes where compare != None and pytorch_probe != None
```

默认只比较 write tensors，因为它们是 candidate 本次 forward 实际产生的值。

## PyTorch 对拍流程

对拍流程必须由 candidate 驱动：

```text
1. 进入 Frame scope
2. 跑 Vulkan candidate forward
3. RuntimeSession 收集 dispatch records 和 written LogicalTensors
4. 退出 candidate forward，但保留 Frame outputs / compare tensors
5. 根据 collected LogicalTensors 安装 PyTorch hooks/probes
6. 跑一次 PyTorch model.forward
7. hook 收集 reference artifacts
8. RuntimeSession readback candidate tensors
9. compare candidate vs reference
10. 释放/reuse Frame workspace
```

注意：PyTorch reference 不决定比较哪些 tensors。它只根据 candidate 收集到的 `LogicalTensor.pytorch_probe` 安装 hook。

## Probe 规则

`PyTorchProbe` 描述如何从 PyTorch forward 中抓 reference artifact。

示例：

```python
@dataclass(frozen=True, slots=True)
class PyTorchProbe:
    kind: Literal["module_input", "module_output", "manual_hook", "derived"]
    target: str
    index: int = 0
    selector: str | None = None
    transform: str | None = None
```

`probes.py` 负责把 probe metadata 映射到具体 PyTorch module/hook。

它不应该手写 candidate 公式：

```python
# bad
return {"audio_codec.decoder.conv1": conv1d(x, w, b)}
```

reference 必须来自 PyTorch/official model.forward 过程中的真实 tensor。

## Compare 规则

Compare 使用 `LogicalTensor.compare`。

```python
@dataclass(frozen=True, slots=True)
class ComparePolicy:
    kind: Literal["tensor", "token", "waveform"]
    rtol: float
    atol: float
    max_abs: float | None = None
```

失败报告至少包含：

```text
frame name
artifact key
logical tensor name
writer shader / dispatch index
candidate shape/dtype
reference shape/dtype
max_abs / max_rel
first mismatch index if practical
```

## Artifact key 和 Scope

`LogicalTensor.name` 不编码动态 audio/video/text index 或 row。动态上下文属于 `FrameScope`。

示例：

```text
logical name: text_llm.decode.layer.03.output
frame: text_llm_cond.decode
scope: audio.token_005/row=cond
artifact key: audio.token_005/row=cond/text_llm_cond.decode/text_llm.decode.layer.03.output
```

Audio codec decoder 没有 cond/uncond row，可以是：

```text
logical name: audio_codec.decoder.waveform
frame: audio_codec_decoder.forward
scope: audio.codec_decoder
artifact key: audio.codec_decoder/audio_codec_decoder.forward/audio_codec.decoder.waveform
```

## OmniVoice 生命周期示例

### Model lifetime

```text
audio token predictor weights
text LLM weights
audio codec decoder weights
shader modules / pipelines if cached
```

### Request/Pipeline lifetime

```text
text prompt ids
generated audio tokens
text LLM KV cache
final waveform / output handles
```

### Frame lifetime

Audio token predictor frame：

```text
audio predictor hidden
audio predictor logits
candidate audio token scratch
```

Text LLM decode frame：

```text
current hidden
q/k/v projections
attention context
mlp intermediates
logits
sampling scratch
```

Audio codec decoder frame：

```text
quantizer embed_sum
project_out intermediates
decoder conv/deconv/resblock activations
waveform output
```

### OP lifetime

```text
split-k scratch
reduction partials
temporary argmax buffers
```

## MVP 规则

MVP 可以先简单实现：

1. weights model-lifetime 常驻；
2. inputs/request state pipeline-lifetime 常驻；
3. Frame workspace 全量分配，Frame 结束统一释放；
4. 暂不做 aliasing；
5. 后续根据 dispatch records 做 liveness/aliasing 优化。

即使 MVP 不做 aliasing，也必须通过 Frame scope 建立释放边界。否则无法从全量常驻平滑演进到显存复用。

## 禁止事项

```text
不要让模型 forward 调用 RuntimeSession.empty/load_weight/free
不要让模型目录写 materialize.py 管显存
不要暴露 frame.workspace.* 物理 slot tree
不要由 PyTorch reference 决定 compare tensors
不要把 audio/video/text index 或 row 写进 LogicalTensor.name
不要 silent dtype cast
```

## 一句话规则

```text
LogicalTensor 声明 tensor 是什么。
Frame 声明一次 model.forward 活多久。
RuntimeSession 根据 LogicalTensor + Frame 自动分配和释放显存。
PyTorch 对拍由 candidate Frame 实际写出的 LogicalTensors 驱动。
```
