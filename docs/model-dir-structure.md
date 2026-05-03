# 模型目录结构规划

本文定义模型接入目录应该写什么、不写什么。核心设计以 `docs/frame-and-logical-tensor.md` 为准。

目标：

```text
模型目录 = 最精简的模型表达

tensors/     声明 LogicalTensor 点
shaders/     声明 ShaderContract 线
*.py         与 execution.py 同级的具体 frame 文件。每个文件表达一次 PyTorch model.forward 边界
execution.py 串联 frame/pipeline
```

显存分配、释放、Vulkan descriptor、dispatch record、PyTorch 对拍、replay、liveness/aliasing 都由通用 `RuntimeSession` 接管。

## 仓库边界

建议分成两层：

```text
src/torch2vk/
  通用框架：LogicalTensor、Frame、RuntimeSession、shader contract、compare、checkpoint、Vulkan backend

src/models/<model_name>/
  模型 adapter：tensor declarations、shader wrappers、frame functions
```

当前仓库已有 `src/models/omnivoice`、`src/models/qwen3_asr` 这类模型 adapter 目录。`src/omnivoice`、`src/qwen_asr` 是上游/official 代码，不应该和 adapter 目录混在一起演进。

## 目录结构

以 OmniVoice 为第一个真实目标，推荐目录：

```text
src/models/omnivoice/
  __init__.py
  execution.py
  audio_token_predictor.py
  text_prefill.py
  text_decode.py
  audio_token_selector.py
  audio_codec_decoder.py
  tensors/
    __init__.py
    spec.py
    common.py
    pipeline.py
    audio_token_predictor.py
    text_llm.py
    audio_token_selector.py
    audio_codec_decoder.py
    outputs.py
  shaders/
    __init__.py
    common.py
    embedding_sum_f32.py
    linear_f32.py
    conv1d_f32.py
    deconv1d_f32.py
    snake_f32.py
    residual_add_f32.py
    argmax_f32.py
```

如果初期只做 audio codec decoder，可以先落子集：

```text
src/models/omnivoice/
  execution.py
  audio_codec_decoder.py
  tensors/
    spec.py
    audio_codec_decoder.py
    outputs.py
  shaders/
    common.py
    embedding_sum_f32.py
    conv1d_f32.py
    deconv1d_f32.py
    snake_f32.py
    residual_add_f32.py
```

## 顶层职责

模型 adapter 只负责：

1. 通过 `tensors/` 构造完整 logical tensor tree；
2. 声明 `LogicalTensor`；
3. 声明 shader contract 和 wrapper；
4. 写每个 PyTorch model.forward 对应的 frame function；
5. 在 `LogicalTensor.pytorch_probe` 中声明 probe metadata；
6. 写 pipeline/frame 的调用顺序。

模型 adapter 不负责：

1. 创建 Vulkan device；
2. 分配/free buffer；
3. 绑定 descriptor；
4. 手动上传 input/weight；
5. 手动 readback；
6. 手写 compare；
7. 生成 replay plan；
8. 做 liveness/aliasing；
9. 维护 `frame.workspace.*` 物理 slot tree。

## Runtime materialization boundary

模型目录不提供 `materialize.py`。

模型目录输出 unbound `LogicalTensor` declarations。`RuntimeSession` 在 `rt.frame(...)` 内根据 dispatch 的 read/write 需求 resolve/materialize tensor instance。

这里的 materialize 不是每个 shader 现场创建 Vulkan allocation。显存必须由 RuntimeSession 按 model/request/frame lifetime 管理 pool/arena；dispatch 只从对应 pool/arena 取得 `BufferSlice` 并记录 tensor instance。

推荐启动形态：

```python
with RuntimeSession.open(device_index=0) as rt:
    tensors = declare_omnivoice_tensors(model_dir)

    rt.register_model(tensors, model_dir=model_dir)
    rt.register_inputs(
        {
            "pipeline.prompt_token_ids": prompt_token_ids,
            "pipeline.language_id": language_id,
        }
    )

    result = run_omnivoice_pipeline(
        rt,
        tensors,
        max_audio_tokens=max_audio_tokens,
    )
```

`rt.register_model(...)` 可以做：

1. 遍历 declarations；
2. 检查 logical name 是否重复；
3. 检查 dtype/shape/layout metadata 是否完整；
4. 记录 `WeightSource` 的 checkpoint root；
5. 从 `LogicalTensor.compare` / `LogicalTensor.pytorch_probe` 记录 compare/probe metadata；
6. 可选预加载 model-lifetime weights。

但它不应该把所有 activation 都分配成 bound tensor tree。activation/output/state 的具体 tensor instance 应由 `RuntimeSession.dispatch()` 根据当前 FrameScope 和 shader contract 触发，并从 RuntimeSession 管理的 pool/arena 中取得 slice。

## execution.py

`execution.py` 表达完整 pipeline 的 eager 顺序。它不直接写 shader sequence，也不直接管理 PyTorch hooks；它调用同级的具体 frame function 文件。

它负责：

```text
prompt/token preparation
for each audio token to generate:
  audio_token_predictor.py
  text_decode.py for cond
  text_decode.py for uncond
  audio_token_selector.py
audio_codec_decoder.py
output postprocess
```

示例：

```python
def run_omnivoice_pipeline(
    rt: RuntimeSession,
    tensors: OmniVoiceTensors,
    pytorch_models: OmniVoicePyTorchModels,
    *,
    max_audio_tokens: int,
) -> OmniVoicePipelineResult:
    selected = None

    for audio_token_index in range(max_audio_tokens):
        predictor_out = run_audio_token_predictor_frame(
            rt,
            tensors.audio_token_predictor,
            pytorch_model=pytorch_models.audio_token_predictor,
            scope={"audio_token_index": audio_token_index},
        )

        cond = run_text_decode_frame(
            rt,
            tensors.text_llm_cond,
            predictor_out,
            pytorch_model=pytorch_models.text_llm_cond,
            scope={"audio_token_index": audio_token_index, "row": "cond"},
        )

        uncond = run_text_decode_frame(
            rt,
            tensors.text_llm_uncond,
            predictor_out,
            pytorch_model=pytorch_models.text_llm_uncond,
            scope={"audio_token_index": audio_token_index, "row": "uncond"},
        )

        selected = run_audio_token_selector_frame(
            rt,
            tensors.audio_token_selector,
            cond,
            uncond,
            pytorch_model=pytorch_models.audio_token_selector,
            scope={"audio_token_index": audio_token_index},
        )

        if selected.done:
            break

    decoder_out = run_audio_codec_decoder_frame(
        rt,
        tensors.audio_codec_decoder,
        selected.tokens,
        pytorch_model=pytorch_models.audio_codec_decoder,
        scope={"domain": "audio"},
    )

    return OmniVoicePipelineResult(waveform=decoder_out.waveform)
```

`execution.py` 不应该：

```text
不写 shader kernel 细节
不直接操作 Vulkan descriptor
不手动 allocate/free tensor
不手写 PyTorch model.forward 的替代公式
不从物理 workspace slot 生成 LogicalTensor
```

## tensors/ 目录

`tensors/` 负责从 model directory 读取 `config.json`、checkpoint metadata 等声明所需信息，并构造完整 logical tensor tree。它不分配显存，不打开 Vulkan，不调用 shader。

### tensors/spec.py

`tensors/spec.py` 是 `tensors/` 内部 helper，描述创建 declarations 所需的静态配置和 shape 参数。外部调用方不直接加载 config，也不手动传 spec 给 frame/execution。

推荐结构：

```python
@dataclass(frozen=True, slots=True)
class OmniVoiceTensorSpec:
    audio_vocab_size: int
    audio_mask_token_id: int
    codebook_count: int
    codebook_size: int
    text_llm_hidden_size: int
    text_llm_num_layers: int
    text_llm_num_heads: int
    text_llm_head_dim: int
    audio_codec_decoder_channels: tuple[int, ...]
    sample_rate: int
```

职责：

1. 由 `tensors/` 内部从 `config.json` / model directory 读取创建 declarations 所需参数；
2. 定义 dtype 期望；
3. 提供各子模块 declaration helper 所需 shape；
4. 不做 dtype cast；
5. 不创建 `BufferSlice`；
6. 不启动 Vulkan。

约束：

```text
spec 只在 declaration 阶段使用。
创建 LogicalTensor 后，forward/shaders/compare 不再回头查 OmniVoiceTensorSpec。
forward 只看 LogicalTensor.spec。
外部入口只调用 declare_omnivoice_tensors(model_dir)。
```

### tensors/pipeline.py

声明 pipeline 级输入、状态、中间连接和最终输出。

示例：

```text
pipeline.prompt_token_ids
pipeline.language_id
pipeline.generated_audio_tokens
pipeline.selected_tokens
pipeline.done
output.waveform
output.wav_pcm16
output.sample_rate
```

跨 Frame 存活的 tensor 必须声明 `lifetime=REQUEST`，例如 generated tokens、KV cache、pipeline state。

### tensors/audio_token_predictor.py

声明 audio token predictor 的 tensors。

示例：

```text
audio_token_predictor.input_tokens
audio_token_predictor.audio_embedding.output
audio_token_predictor.hidden
audio_token_predictor.audio_head.logits
audio_token_predictor.audio_head.selected
```

如果内部有多个候选分支，可以继续拆 declaration helper，但不要拆出物理 workspace tree。

### tensors/text_llm.py

声明 Text LLM prefill/decode tensors。

推荐 base name：

```text
text_llm.prefill.input_ids
text_llm.prefill.layer.00.input
text_llm.prefill.layer.00.output
text_llm.decode.input_ids
text_llm.decode.layer.00.input
text_llm.decode.layer.00.output
text_llm.decode.logits
text_llm.state.layer.00.key_cache
text_llm.state.layer.00.value_cache
```

cond/uncond、audio token index 不写进 base name。它们由 `FrameScope` 表达。

KV cache 必须是 `role=KV_CACHE`、`memory=REQUEST_STATE`、`lifetime=REQUEST`。

### tensors/audio_token_selector.py

声明 classifier-free guidance、token selection、argmax/sample 等 tensors。

示例：

```text
audio_token_selector.cond_logits
audio_token_selector.uncond_logits
audio_token_selector.guided_logits
audio_token_selector.next_token
audio_token_selector.done
```

`next_token` 如果要给后续 frame 使用，应声明为 request lifetime 或 pipeline output。

### tensors/audio_codec_decoder.py

声明 audio codec decoder 相关 tensors。

这个子链路适合作为第一个真实 Vulkan 目标，因为它主要由 embedding、conv/deconv、activation、elementwise 组成。

示例：

```text
audio_codec_decoder.quantizer.tokens
audio_codec_decoder.quantizer.embed_sum
audio_codec_decoder.quantizer.project_out.hidden1024
audio_codec_decoder.quantizer.project_out.hidden256
audio_codec_decoder.decoder.conv1
audio_codec_decoder.decoder.block0.deconv
audio_codec_decoder.decoder.block0.res_unit1.conv1
audio_codec_decoder.decoder.block0.res_unit1.output
audio_codec_decoder.decoder.waveform
```

token normalization 必须作为 boundary contract 写清楚。OmniVoice audio token predictor 可能使用 `1024` 作为 audio mask token，而 audio codec decoder codebook 合法 index 是 `0..1023`。进入 decoder quantizer 前，candidate 和 PyTorch probe 必须使用一致规则，例如：

```text
normalized_token = clamp(token, 0, codebook_size - 1)
```

这个规则应存在于 tensor boundary 或 shader contract 中，不能在 candidate 和 PyTorch probe 两边各自隐式处理。

### tensors/ 禁止事项

```text
不要调用 RuntimeSession.empty/load_weight/input/free
不要持有 BufferAllocation 或 BufferSlice
不要读写 Vulkan buffer
不要读 checkpoint 内容
不要调用 shader
不要提供 frame.workspace.*.activation(name) 这种 API
不要把 audio token index / row 写进 LogicalTensor.name
```

## 权重声明

模型目录可以声明权重 tensor，但不应该手动加载权重。

推荐在对应 `tensors/<submodel>.py` 里声明：

```python
LogicalTensor(
    name="audio_codec_decoder.decoder.conv1.weight",
    spec=TensorSpec(dtype="float32", shape=(out_channels, in_channels, kernel)),
    role=TensorRole.WEIGHT,
    memory=MemoryClass.MODEL_WEIGHT,
    lifetime=TensorLifetime.MODEL,
    source=WeightSource(
        checkpoint="model.safetensors",
        key="audio_codec_decoder.decoder.conv1.weight",
        dtype="float32",
        shape=(out_channels, in_channels, kernel),
    ),
)
```

`WeightSource` 是声明 metadata。实际打开 checkpoint、校验 key/dtype/shape、上传 device local memory 由 RuntimeSession 在 read materialization 时完成。

如果未来需要 packed weight，必须显式声明新的 layout 或 preprocessing artifact，例如：

```text
layout = TensorLayout("blocked_vk_matmul", {"block_m": 16, "block_n": 16})
source = WeightSource(..., layout=checkpoint_layout)
runtime transform = WeightTransform("pack_blocked_vk_matmul", target_layout=layout)
```

`WeightTransform` 是后续扩展点，MVP 可以不实现。禁止把 packed weight 伪装成 checkpoint 原始 tensor。禁止 silent dtype cast。

## 具体 frame 文件

与 `execution.py` 同级的每个具体 frame 文件都代表一次 PyTorch `model.forward` 对齐的边界。文件名必须表达具体语义，不使用泛泛的 `forward.py`，也不额外套一层 `frames/` 子目录。

推荐命名：

```text
audio_token_predictor.py
text_prefill.py
text_decode.py
audio_token_selector.py
audio_codec_decoder.py
```

每个 frame function 必须显式接收对应的 PyTorch model。RuntimeSession 在 Vulkan shader sequence 执行完成后，用本 Frame 实际写出的 LogicalTensors 收集 compare targets，再根据这些 LogicalTensors 自带的 `pytorch_probe` metadata hook 传入的 PyTorch model，并 lockstep 执行这一次对应的 PyTorch `model.forward`。

frame function 的职责：

1. 进入 `with rt.frame(..., pytorch_model=...)`；
2. 在 frame 内调用 shader wrapper；
3. 把输入、权重、中间 activation、输出 `LogicalTensor` 传给 wrapper；
4. 返回 output dataclass；
5. 不手动安装 hooks；
6. 不维护独立 probe 列表；
7. 不手动 materialize tensor。

示例：

```python
def run_audio_codec_decoder_frame(
    rt: RuntimeSession,
    tensors: AudioCodecDecoderTensors,
    tokens: LogicalTensor,
    *,
    pytorch_model: torch.nn.Module,
    scope: Mapping[str, str | int],
) -> AudioCodecDecoderOutput:
    with rt.frame(
        "audio_codec_decoder",
        scope=scope,
        pytorch_model=pytorch_model,
    ):
        normalize_audio_tokens(
            rt,
            tokens=tokens,
            output=tensors.quantizer.normalized_tokens,
        )
        embedding_sum_f32(
            rt,
            tokens=tensors.quantizer.normalized_tokens,
            weight=tensors.weights.quantizer_embed,
            output=tensors.quantizer.embed_sum,
        )
        conv1d_f32(
            rt,
            x=tensors.quantizer.embed_sum,
            weight=tensors.weights.decoder_conv1,
            bias=tensors.weights.decoder_conv1_bias,
            output=tensors.decoder.conv1,
        )
        return AudioCodecDecoderOutput(waveform=tensors.decoder.waveform)
```

## PyTorchProbe metadata

`PyTorchProbe` 是 `LogicalTensor` 的字段。probe metadata 必须跟 tensor 声明在一起，不能在 `probes.py` 或其它平行 registry 里再维护一份。

流程：

```text
candidate frame 结束后得到 written LogicalTensors
  -> 过滤 compare != None 且 pytorch_probe != None
  -> RuntimeSession 读取每个 LogicalTensor.pytorch_probe
  -> 在当前 frame 传入的 PyTorch model 上安装 hooks
  -> lockstep 执行当前 frame 对应的 PyTorch model.forward
  -> hook 收集 PyTorch artifacts
  -> RuntimeSession 移除 hooks
```

`pytorch_probe.target` 应该直接描述 PyTorch model 内的 module path 或 hook target。不同模型确实需要不同 hook target，但这些 target 仍然属于对应 LogicalTensor 的 metadata。

禁止手写 candidate 公式：

```python
# bad
return {"audio_codec_decoder.decoder.conv1": torch.nn.functional.conv1d(x, w, b)}
```

PyTorch artifact 必须来自当前 frame 传入的 PyTorch model.forward 中捕获到的真实 tensor。模型目录不提供独立 PyTorch 子系统，也不维护另一套 PyTorch input 流程。

## shaders/ 目录

`shaders/` 放可复用的 shader contract、variant 和 wrapper。

第一批建议：

```text
embedding_sum_f32.py
linear_f32.py
conv1d_f32.py
deconv1d_f32.py
snake_f32.py
residual_add_f32.py
argmax_f32.py
```

Text LLM 后续再加：

```text
rms_norm_f32.py
rope_f32.py
attention_decode_f32.py
attention_prefill_f32.py
swiglu_f32.py
```

每个 shader wrapper 只做：

1. 暴露清晰 Python 函数；
2. 选择 `ShaderVariant`；
3. 调用 `RuntimeSession.dispatch()`。

禁止：

```text
不要加载模型权重
不要创建模型 LogicalTensor 名字
不要分配显存
不要调用 PyTorch model.forward
不要隐藏 frame 调用顺序
```

### GLSL source 组织

短期允许两种来源：

```text
1. imported_glsl/：从外部项目导入、尚未整理的 GLSL
2. shaders/*.py 或 shaders/glsl/：已经纳入 torch2vk contract 管理的 shader
```

长期目标是每个 `ShaderVariant` 明确记录 source path、compiled spirv path、contract 和 specialization constants。是否内联 GLSL 不是架构核心，核心是不要让 shader source 和 contract 脱节。

## Candidate 收集 LogicalTensors

`RuntimeSession.dispatch()` 每次 shader 调用都记录：

```text
dispatch index
frame name
FrameScope
shader name
reads: field -> TensorInstanceKey
writes: field -> TensorInstanceKey
logical_reads: field -> LogicalTensor.name
logical_writes: field -> LogicalTensor.name
symbols / dispatch dims
```

Frame 结束后：

```text
used tensors = all read tensors + all written tensors
compare tensors = written tensors where compare != None and pytorch_probe != None
```

通常只比较 written tensors，因为它们是 Vulkan candidate 产生的边界。

## 最小数据流

```text
pytest or script defines run inputs
        |
tensors/ reads config.json/checkpoint metadata and builds the logical tensor tree
        |
tensors/ declares LogicalTensors with source/feed/probe/compare metadata
        |
RuntimeSession registers model declarations and runtime feeds
        |
execution.py calls concrete frame files, such as audio_codec_decoder.py
        |
concrete frame file enters rt.frame(..., pytorch_model=...) and calls shader wrappers
        |
RuntimeSession.dispatch validates contract and resolves/materializes reads/writes from pools/arenas
        |
RuntimeSession records dispatch facts
        |
Frame exit collects written compare tensors
        |
RuntimeSession installs hooks from LogicalTensor.pytorch_probe on the frame pytorch_model
        |
RuntimeSession lockstep-runs the frame pytorch_model.forward
        |
RuntimeSession readbacks candidate TensorInstanceKeys and compares artifacts
        |
RuntimeSession releases/reuses frame workspace
```

## 命名规则

Logical tensor base name 使用模型语义：

```text
audio_token_predictor.audio_head.logits
text_llm.decode.layer.03.output
text_llm.state.layer.03.key_cache
audio_token_selector.guided_logits
audio_codec_decoder.decoder.block2.res_unit1.output
audio_codec_decoder.decoder.waveform
output.wav_pcm16
```

不要把 dynamic scope 写进 base name：

```text
bad: audio.token_005.text_llm.cond.layer.03.output
good base: text_llm.decode.layer.03.output
good scope: audio_token_index=5, row=cond
```

不要用物理位置命名 logical tensor：

```text
bad: workspace.core.hidden_states
bad: frame.attention.q_proj
bad: buffer17.slice3
```

## MVP 落地顺序

建议按这个顺序落地：

1. 建通用 `src/torch2vk` core：`LogicalTensor`、`FrameScope`、`ShaderContract`、dry-run `RuntimeSession`。
2. 建 `src/models/omnivoice/tensors/spec.py` 和 `tensors/audio_codec_decoder.py`，只生成 declarations。
3. 建 `audio_codec_decoder.py`，表达一次 audio codec decoder PyTorch model.forward 边界，并接收 `pytorch_model`。
4. 建 `shaders/embedding_sum_f32.py`、`conv1d_f32.py`、`snake_f32.py` 的 contract 和 wrapper。
5. `execution.py` 先只调用 `run_audio_codec_decoder_frame(...)`。
6. dry-run dispatch 先只校验 contract、materialization rules、dispatch record。
7. 接入 Vulkan 后执行最短 audio codec decoder 子链路。
8. candidate 跑完后从 dispatch records 收集 compare tensors。
9. RuntimeSession 根据 `LogicalTensor.pytorch_probe` 在传入的 audio codec decoder PyTorch model 上安装 hooks。
10. RuntimeSession lockstep 执行该 frame 的 PyTorch model.forward。
11. RuntimeSession frame exit readback + compare。
12. audio codec decoder 稳定后再接 audio token selector、audio token predictor、Text LLM decode/prefill。

## MVP 验收

第一个 OmniVoice 子链路完成时应满足：

1. `tensors/audio_codec_decoder.py` 可以不启动 Vulkan 生成 declarations；
2. `shaders/` 可以不加载 OmniVoice 就校验 contract；
3. `audio_codec_decoder.py` 表达一次 PyTorch model.forward 边界，并在内部使用 `with rt.frame(..., pytorch_model=...)`；
4. `execution.py` 只串联具体 frame function，不直接写 shader sequence；
5. 模型代码没有调用 `RuntimeSession.empty/load_weight/free`；
6. dispatch 时 RuntimeSession 能 resolve/materialize read/write tensor instances，但不做 per-shader raw Vulkan allocation；
7. candidate forward 后能从 dispatch records 收集 written LogicalTensors；
8. 当前 frame 传入的 PyTorch model 根据 collected LogicalTensors 动态安装 hooks 并 lockstep 执行；
9. compare 只比较 candidate 实际写出且声明了 compare/probe 的 tensors；
10. 没有 `frame.workspace.*.activation(name)` 这种 API；
11. 权重 dtype/shape/layout 必须和 checkpoint 声明一致，不做 silent cast；
12. Frame exit 后 FRAME_WORKSPACE 被释放或复用，MODEL/REQUEST 生命周期资源保留。
