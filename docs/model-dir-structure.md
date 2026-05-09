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

模型目录给 `LogicalTensor` 附加 metadata 只为四个目标：

1. 声明权重 tensor 的 name/spec/layout，让 runtime 推断 checkpoint key 并托管加载和校验；
2. 声明 storage/lifetime，让 runtime 托管显存分配和释放；
3. 生成 `ReferenceSpec`，让 runtime 输出和 PyTorch reference 对拍；
4. 声明 spec/layout，让 runtime 校验 shader contract 匹配。

不影响这四件事的模型语义不要放进核心执行规则。`LOGITS`、`TOKEN`、`KV_CACHE` 等可以作为 semantic metadata 服务 debug、compare 和报告。

## 仓库边界

建议分成两层：

```text
src/torch2vk/
  通用框架：LogicalTensor、Frame、RuntimeSession、shader contract、compare、checkpoint、Vulkan backend

src/models/<model_name>/
  模型 adapter：tensor declarations、shader variants、frame functions
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
3. 声明 shader contract 和 `ShaderVariant`；
4. 写每个 PyTorch model.forward 对应的 frame function；
5. 在生成的 tensor/spec 中声明 reference binding metadata；
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

模型目录输出稳定的 `LogicalTensor` tree。`RuntimeSession` 在 `rt.frame(...)` 内根据 dispatch 的 read/write
需求 resolve/materialize，并直接更新对应 `LogicalTensor` 当前 buffer 状态。

这里的 materialize 不是每个 shader 现场创建 Vulkan allocation。显存必须由 RuntimeSession 按
model/request/frame lifetime 管理 pool/arena；dispatch 只从对应 pool/arena 取得 `BufferSlice` 并写回
`LogicalTensor` 当前状态。

推荐启动形态：

```python
with RuntimeSession.open(device_index=0, model_dir=model_dir) as rt:
    tensors = declare_omnivoice_tensors(model_dir)

    rt.register_inputs(
        {
            tensors.pipeline.prompt_token_ids: prompt_token_ids,
            tensors.pipeline.language_id: language_id,
        }
    )

    result = run_omnivoice_pipeline(
        rt,
        tensors,
        max_audio_tokens=max_audio_tokens,
    )
```

`register_inputs()` 的 key 是声明好的 `LogicalTensor` 对象，不是 logical name 字符串。这样输入绑定和
shader dispatch 使用的是同一批 tensor 对象，不需要额外的 input key 映射。

PyTorch lockstep 对拍也应使用同一份业务输入，但不由 RuntimeSession 自动调用 PyTorch。模型 `run.py`
显式推进 reference state，得到 expected dict 后调用 `compare_expected_with_spec()`。这样长流程生成任务可以让
PyTorch reference 和 Vulkan candidate 同步推进，而不是依赖 frame exit 的隐式行为。

`RuntimeSession.open(..., model_dir=...)` 设置权重 checkpoint 根目录。`LogicalTensor` 的
name/dtype/shape/layout、role、memory、lifetime 等声明组合在实际使用点校验：`register_inputs()` 校验输入 tensor，
`RuntimeSession.dispatch()` 校验本次 shader 调用传入的 tensor，并在 record/eager 阶段按需 materialize
本次实际读写的 tensors。

Runtime 不维护 `name -> LogicalTensor` registry，也不把 compare/debug metadata 复制到另一份表里。后续
dispatch、readback、compare 和 replay 都必须继续拿着原始 `LogicalTensor` 对象走。Frame 中实际执行了哪些
tensor，由 `ShaderVariant(rt, ...)` 调用在运行时收集到 dispatch records。

具体 frame function 不需要把本次 forward 可能读取的权重和外部依赖再作为一份 tensor 列表传给
`rt.frame(...)`。实际读写了哪些 tensor 只由 shader 调用过程收集到 dispatch records。Replay 可以消费这些
records，在 Frame enter 提前做权重预加载、workspace sizing、liveness/aliasing 和 arena offset 分配。

Runtime 不应该预先把所有 activation 都分配出来。activation/output/state 的当前 buffer 状态应由
`RuntimeSession.dispatch()` 根据当前 frame 和 shader contract 触发，并从 RuntimeSession 管理的
pool/arena 中取得 slice 后写回对应 `LogicalTensor`。

## execution.py

`execution.py` 表达完整 pipeline 的 eager 顺序。它不直接写 shader sequence，也不直接管理 PyTorch reference；它调用同级的具体 frame function 文件。

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
            frame_name=f"audio_token_predictor.audio_token_{audio_token_index:03d}",
        )

        cond = run_text_decode_frame(
            rt,
            tensors.text_llm_cond,
            predictor_out,
            pytorch_model=pytorch_models.text_llm_cond,
            frame_name=f"text_llm.decode.audio_token_{audio_token_index:03d}.cond",
        )

        uncond = run_text_decode_frame(
            rt,
            tensors.text_llm_uncond,
            predictor_out,
            pytorch_model=pytorch_models.text_llm_uncond,
            frame_name=f"text_llm.decode.audio_token_{audio_token_index:03d}.uncond",
        )

        selected = run_audio_token_selector_frame(
            rt,
            tensors.audio_token_selector,
            cond,
            uncond,
            pytorch_model=pytorch_models.audio_token_selector,
            frame_name=f"audio_token_selector.audio_token_{audio_token_index:03d}",
        )

        if selected.done:
            break

    decoder_out = run_audio_codec_decoder_frame(
        rt,
        tensors.audio_codec_decoder,
        selected.tokens,
        pytorch_model=pytorch_models.audio_codec_decoder,
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

声明 pipeline 级输入、状态、中间连接和最终输出。pipeline 级输入的 `LogicalTensor` 会作为
`RuntimeSession.register_inputs()` 的 key。

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

cond/uncond、audio token index 进入本次 invocation 的 frame name，并由对应的 `LogicalTensor` tree 承载。

KV cache 必须是 `role=STATE`、`semantic=KV_CACHE`、`memory=REQUEST_STATE`、`lifetime=REQUEST`。

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

`next_token` 如果要给后续 frame 使用，应声明为 request lifetime 或 pipeline output，并可标记 `semantic=TOKEN`。

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
)
```

权重 `LogicalTensor.name` 必须和 checkpoint tensor key 一致；dtype、shape、layout 从 `spec/layout`
推断。实际打开 checkpoint、校验 key/dtype/shape、上传 device local memory 由 RuntimeSession 在
record/eager 阶段的 dispatch read path 按需完成。dispatch 读到 weight 时，如果已有
model-lifetime materialization 就复用；否则打开 checkpoint、校验并上传。Replay 可以根据录制过的 dispatch
reads 在 Frame enter 预加载或校验这些权重。

如果未来需要 packed weight，必须显式声明新的 layout 或 preprocessing artifact，例如：

```text
layout = TensorLayout("blocked_vk_matmul", {"block_m": 16, "block_n": 16})
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

每个 frame function 只表达 Vulkan shader sequence。PyTorch reference 由模型 `run.py` 或同级 helper 显式执行，
再通过 `ReferenceSpec.output_bindings` 和 `compare_expected_with_spec()` 对齐 Vulkan 输出。

frame function 的职责：

1. 进入 `with rt.frame(...)`；
2. 在 frame 内调用 `ShaderVariant`；
3. 把输入、权重、中间 activation、输出 `LogicalTensor` 传给 `ShaderVariant`；
4. 返回 output dataclass；
5. 不维护独立 compare binding 列表；
6. 不绕过 `ReferenceSpec` 手写 tensor path；
7. 不手动 materialize tensor。

示例：

```python
def run_audio_codec_decoder_frame(
    rt: RuntimeSession,
    tensors: AudioCodecDecoderTensors,
    tokens: LogicalTensor,
    *,
    pytorch_model: torch.nn.Module,
    frame_name: str = "audio_codec_decoder",
) -> AudioCodecDecoderOutput:
    with rt.frame(
        frame_name,
        pytorch_model=pytorch_model,
    ):
        NORMALIZE_AUDIO_TOKENS(
            rt,
            tokens=tokens,
            output=tensors.quantizer.normalized_tokens,
        )
        EMBEDDING_SUM_F32(
            rt,
            tokens=tensors.quantizer.normalized_tokens,
            weight=tensors.weights.quantizer_embed,
            output=tensors.quantizer.embed_sum,
        )
        CONV1D_F32(
            rt,
            x=tensors.quantizer.embed_sum,
            weight=tensors.weights.decoder_conv1,
            bias=tensors.weights.decoder_conv1_bias,
            output=tensors.decoder.conv1,
        )
        return AudioCodecDecoderOutput(waveform=tensors.decoder.waveform)
```

## ReferenceSpec metadata

`ReferenceSpec` 由 export 生成，描述 reference key 到 tensor dataclass 字段路径的绑定。它必须和生成的 tensor
tree 一起维护，不能在 `probes.py` 或其它平行 registry 里再维护一份。

流程：

```text
Vulkan frame 写出 LogicalTensors
  -> run.py 显式执行 PyTorch reference
  -> compare_expected_with_spec() 读取 ReferenceSpec.output_bindings
  -> RuntimeSession readback 对应 LogicalTensor
  -> compare helper 写出 pass/fail 和 artifacts
```

`ReferenceSpec.program` 有值时表示该 reference 可以从 `.pt2` 加载；为空时表示调用方负责显式计算 expected。

禁止手写 candidate 公式：

```python
# bad
return {"audio_codec_decoder.decoder.conv1": torch.nn.functional.conv1d(x, w, b)}
```

PyTorch artifact 必须来自当前 frame 传入的 PyTorch model.forward 中捕获到的真实 tensor。模型目录不提供独立 PyTorch 子系统，也不维护另一套 PyTorch input 流程。
PyTorch input 同样不维护另一套 mapping registry；输入绑定由 `register_inputs()` 和 logical name 规则决定。

## shaders/ 目录

`shaders/` 放可复用的 shader contract 和 `ShaderVariant`。

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

每个 shader 文件只做：

1. 定义 `ShaderContract`；
2. 定义可调用的 `ShaderVariant`；
3. 不再为 variant 额外包一层同名 Python 函数。

禁止：

```text
不要加载模型权重
不要创建模型 LogicalTensor 名字
不要分配显存
不要调用 PyTorch model.forward
不要隐藏 frame 调用顺序
```

### GLSL source 组织

`ShaderVariant` 的 Python 定义是 shader source of truth。GLSL 应内置在对应 `shaders/*.py` 文件的
`source="""..."""` 字段里，contract、variant metadata 和 GLSL 放在同一个 source-layer 定义点。

推荐组织：

```text
shaders/
  embedding_sum_f32.py     # ShaderContract + ShaderVariant(source="...")
  conv1d_f32.py            # ShaderContract + ShaderVariant(source="...")

.cache/torch2vk/generated/
  embedding_sum_f32.glsl   # generated artifact
  embedding_sum_f32.spv    # generated artifact
```

`docs/models/omnivoice/imported_glsl_reference/` 可以保留外部项目导入、尚未整理的 GLSL baseline，但纳入 torch2vk 管理的 shader 必须迁移成
inline GLSL variant。构建工具负责从 `ShaderVariant.source` 生成 `.cache/torch2vk/generated/*.glsl`
和 `.spv`，并把 source hash、contract manifest、compile defines、include dirs 和 specialization
constants 记录进 artifact manifest。不要让 `*.comp` 或 generated `.glsl` 成为 source of truth。

## Candidate 收集 LogicalTensors

`RuntimeSession.dispatch()` 每次 shader 调用都记录：

```text
dispatch index
frame name
shader name
reads: field -> LogicalTensor
writes: field -> LogicalTensor
logical_reads: field -> LogicalTensor.name
logical_writes: field -> LogicalTensor.name
symbols / dispatch dims
```

Frame 结束后：

```text
used tensors = all read tensors + all written tensors
written tensors = shader outputs in this frame
dispatch records = replay/debug/liveness source of truth
```

是否对拍由调用方显式决定。调用方执行 PyTorch reference 后，用 `ReferenceSpec.output_bindings` 定位
candidate tensor，并传入明确的 `ComparePolicy`。

## 最小数据流

```text
pytest or script defines run inputs
        |
tensors/ reads config.json/checkpoint metadata and builds the logical tensor tree
        |
tensors/ declares LogicalTensors and generated reference specs
        |
RuntimeSession registers model declarations and runtime inputs
        |
execution.py calls concrete frame files, such as audio_codec_decoder.py
        |
concrete frame file enters rt.frame(...) and calls ShaderVariant objects
        |
RuntimeSession.dispatch validates contract and resolves/materializes reads/writes from pools/arenas
        |
RuntimeSession records dispatch facts
        |
run.py computes PyTorch expected values at the same logical step
        |
compare_expected_with_spec locates output tensors from ReferenceSpec
        |
RuntimeSession readbacks candidate LogicalTensors and compares artifacts
        |
On mismatch, RuntimeSession drills down through writer.reads on demand
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

不要用物理位置命名 logical tensor：

```text
bad: workspace.core.hidden_states
bad: frame.attention.q_proj
bad: buffer17.slice3
```

动态 invocation 身份由 frame name 和本次执行创建的 tensor tree 表达：

```text
text_llm.decode.audio_token_005.cond
text_llm.decode.audio_token_005.cond.layer.03.output
```

## MVP 落地顺序

建议按这个顺序落地：

1. 建通用 `src/torch2vk` core：`LogicalTensor`、`FrameContext`、`ShaderContract`、dry-run `RuntimeSession`。
2. 建 `src/models/omnivoice/tensors/spec.py` 和 `tensors/audio_codec_decoder.py`，只生成 declarations。
3. 建 `audio_codec_decoder.py`，表达一次 audio codec decoder Vulkan frame 边界。
4. 建 `shaders/embedding_sum_f32.py`、`conv1d_f32.py`、`snake_f32.py` 的 contract 和 `ShaderVariant`。
5. `execution.py` 先只调用 `run_audio_codec_decoder_frame(...)`。
6. dry-run dispatch 先只校验 contract、materialization rules、dispatch record。
7. 接入 Vulkan 后执行最短 audio codec decoder 子链路。
8. candidate 跑完后显式执行同粒度 PyTorch reference。
9. 用 `ReferenceSpec.output_bindings` 定位 candidate tensor。
10. 调用 `compare_expected_with_spec()` 完成 readback + compare。
11. mismatch 时报告 writer shader 和 artifact path。
12. audio codec decoder 稳定后再接 audio token selector、audio token predictor、Text LLM decode/prefill。

## MVP 验收

第一个 OmniVoice 子链路完成时应满足：

1. `tensors/audio_codec_decoder.py` 可以不启动 Vulkan 生成 declarations；
2. `shaders/` 可以不加载 OmniVoice 就校验 contract；
3. `audio_codec_decoder.py` 表达一次 Vulkan frame 边界，并在内部使用 `with rt.frame(...)`；
4. `execution.py` 只串联具体 frame function，不直接写 shader sequence；
5. 模型代码没有调用 `RuntimeSession.empty/load_weight/free`；
6. dispatch 时 RuntimeSession 能 resolve/materialize read/write LogicalTensors；record/eager 可按需 allocation，replay 用录制结果优化；
7. candidate forward 后能从 dispatch records 收集 written LogicalTensors；
8. 调用方可以显式执行 PyTorch reference 并通过 `ReferenceSpec` 对拍；
9. compare 输出包含 candidate、expected、summary artifacts；
10. 没有 `frame.workspace.*.activation(name)` 这种 API；
11. 权重 dtype/shape/layout 必须和 checkpoint 声明一致，不做 silent cast；
12. Frame exit 后 FRAME_WORKSPACE 被释放或复用，MODEL/REQUEST 生命周期资源保留。

## Application Layer（补充设计）

Docs 原始设计只覆盖了模型 adapter 的内部结构。实际接入 Qwen3-ASR 后发现需要一个 application layer 来处理 docs 没有规划的职责：

### 输入准备

模型特有的输入准备（processor 调用、音频归一化、prompt 模板构造、RoPE cos/sin 预计算）放在模型 adapter 目录内。它们不属于 `tensors/`、`shaders/`、或单个 frame 文件；Qwen3-ASR 这类没有独立 application 层的 adapter 直接放在 `execution.py`。

RoPE 预计算这类可复用数学逻辑可以单独放在 `rope.py`。

### PyTorch 模型加载

`pytorch_model` 参数必须由某处提供。PyTorch 模型加载代码放在模型 adapter 目录内，通常在 `pytorch/` 子目录中。这些代码只用于 lockstep 对拍，不参与 Vulkan 执行路径。

### execution.py 与入口分工

```text
execution.py    = adapter entry: 输入准备, frame 间输入注册, 循环控制, 状态增长
audio_tower.py  = frame function: shader 调用序列
text_prefill.py = frame function: shader 调用序列
text_decode.py  = frame function: shader 调用序列
token_select.py = frame function: shader 调用序列
```

### REQUEST_STATE 初始化

KV cache 等 REQUEST_STATE tensor 需要在首次使用前初始化为零。由 frame function 在 `with rt.frame(...)` 之前调用 `rt.initialize_request_state({tensor: np.zeros(...)})` 完成。

### 动态 shape 增长

自回归解码中 KV cache 和 generated_tokens 每步增长。由 `execution.py` 在 decode 循环中调用 `rt.grow_request_state(tensor, new_shape)` 完成。Runtime 使用 geometric growth 策略避免频繁重分配。

### 不需要 pytorch_model 的 Frame

并非所有 frame 都对应一个 PyTorch model.forward。纯 runtime 操作（如 token_select、state copy）使用 `pytorch_model=None` 的 frame。RuntimeSession 在 frame exit 时跳过对拍。
