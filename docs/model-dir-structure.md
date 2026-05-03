# OmniVoice 模型目录结构规划

目标：以 OmniVoice 为第一个真实目标，规划 `torch2vk` 的模型目录。执行和对拍流程固定为：

```text
OmniVoice pipeline 先跑 Vulkan candidate
  -> 每个 submodel 的 forward 调用 Vulkan compute shader
  -> RuntimeSession.dispatch 记录实际 read/write 的 LogicalTensors
  -> candidate forward 结束后收集本次实际使用的 LogicalTensors
  -> 再跑 PyTorch/official OmniVoice reference
  -> 根据 collected LogicalTensors 动态安装 hook/probe
  -> 收集 reference artifacts
  -> readback candidate tensors 并 compare
```

关键原则：**对拍需要哪些 tensor，由 Vulkan candidate forward 实际写出的 `LogicalTensor` 决定，不由 PyTorch reference 预先决定。**

显存分配、释放、dispatch、readback 统一归 `RuntimeSession`。OmniVoice 目录只负责：tensor 声明、权重声明、shader wrapper、submodel forward、完整 pipeline、PyTorch probe 映射和 compare 编排。

## 推荐目录

```text
src/torch2vk/models/omnivoice/
  __init__.py
  execution.py
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
  audio_token_predictor/
    __init__.py
    forward.py
    weights.py
    pytorch.py
    probes.py
  text_llm/
    __init__.py
    prefill_forward.py
    decode_forward.py
    weights.py
    pytorch.py
    probes.py
  audio_token_selector/
    __init__.py
    forward.py
    probes.py
  audio_codec_decoder/
    __init__.py
    forward.py
    weights.py
    pytorch.py
    probes.py
  shaders/
    __init__.py
    common.py
    embedding_sum_f32.py
    linear_f32.py
    rms_norm_f32.py
    conv1d_f32.py
    deconv1d_f32.py
    snake_f32.py
    argmax_f32.py
```

如果初期只做 audio_codec_decoder decode，可以先落这个子集：

```text
src/torch2vk/models/omnivoice/
  __init__.py
  execution.py
  tensors/
    __init__.py
    spec.py
    audio_codec_decoder.py
    outputs.py
  audio_codec_decoder/
    __init__.py
    forward.py
    weights.py
    pytorch.py
    probes.py
  shaders/
    __init__.py
    conv1d_f32.py
    deconv1d_f32.py
    snake_f32.py
```

## 顶层文件

测试输入和 debug 参数不放进模型目录。pytest 应该在测试函数或 fixture 里内联构造输入数据，例如 token ids、`max_audio_tokens`、`guidance_scale`、forced tokens 和 compare boundary。模型目录只保留可复用的 runtime 结构。

### Runtime materialization boundary

OmniVoice 模型目录不提供 `materialize.py`。模型接入方不应该手写“把哪些 tensor 分配到哪块显存、什么时候释放”的代码。

模型目录只输出 unbound `LogicalTensor` declarations。`RuntimeSession` 负责统一 materialize：

```python
with RuntimeSession.open(device_index=0) as rt:
    tensors = omnivoice_tensors(tensor_spec, batch=batch, max_audio_tokens=max_audio_tokens)
    bound = rt.prepare(tensors, inputs={...}, model_dir=model_dir)
    run_omnivoice_pipeline(rt, bound, max_audio_tokens=max_audio_tokens)
```

`RuntimeSession.prepare()` 的职责：

1. 遍历 `LogicalTensor` declarations；
2. 根据 `role/memory/source` 自动决定 input、weight、activation、output 的分配和上传；
3. 根据 `WeightSource` 自动加载权重；
4. 返回同结构、但每个 `LogicalTensor.storage != None` 的 bound tensor tree；
5. 记录 allocation owner，并在 session close 时统一释放。

模型代码可以声明 memory policy，但不能直接调用 `RuntimeSession.empty/load_weight/input` 去逐个分配 tensor。这样 `LogicalTensor` 才有意义。

### execution.py

`execution.py` 是完整 OmniVoice pipeline，不是某个模型 forward。

它负责串联：

```text
prompt/token preparation
for each audio token to generate:
  audio_token_predictor.forward
  text_llm.prefill/decode forward for cond/uncond if needed
  audio_token_selector.forward
audio_codec_decoder.forward
output postprocess
```

示例形态：

```python
def run_omnivoice_pipeline(
    rt: RuntimeSession,
    state: OmniVoiceRuntimeState,
    *,
    max_audio_tokens: int,
) -> OmniVoicePipelineResult:
    for audio_token_index in range(max_audio_tokens):
        audio_token_predictor_out = run_audio_token_predictor_forward(
            rt,
            state.audio_token_predictor,
            audio_token_index=audio_token_index,
        )
        text_llm_cond = run_text_llm_decode_forward(
            rt,
            state.text_llm_cond,
            audio_token_predictor_out,
            audio_token_index=audio_token_index,
        )
        text_llm_uncond = run_text_llm_decode_forward(
            rt,
            state.text_llm_uncond,
            audio_token_predictor_out,
            audio_token_index=audio_token_index,
        )
        selected = run_audio_token_selector_forward(
            rt,
            state.audio_token_selector,
            text_llm_cond,
            text_llm_uncond,
            audio_token_index=audio_token_index,
        )
        if selected.done:
            break
    audio_codec_decoder_out = run_audio_codec_decoder_forward(rt, state.audio_codec_decoder, selected.tokens)
    return OmniVoicePipelineResult(waveform=audio_codec_decoder_out.waveform)
```

`execution.py` 可以管理 audio token generation loop 和 submodel 调用顺序，但不应该：

```text
不写 shader kernel 细节
不直接操作 Vulkan descriptor
不手写 PyTorch reference 计算
不从物理 workspace slot 生成 LogicalTensor
```

### Frame compare boundary

OmniVoice 模型目录不提供 `compare.py`。对拍编排属于 RuntimeSession 的 Frame runner。

原因：`LogicalTensor` 已经声明了 `compare` 和 `pytorch_probe`，candidate forward 的 dispatch records 也能告诉 RuntimeSession 本次实际写出了哪些 tensor。因此 Frame 结束时 runtime 可以统一完成：

1. 从本 Frame dispatch records 收集 written `LogicalTensor`；
2. 过滤 `compare != None` 且 `pytorch_probe != None` 的 tensors；
3. 按 submodel 调用对应 `probes.py` 安装 PyTorch hooks；
4. 执行 PyTorch/official `model.forward`；
5. readback candidate tensors；
6. 按 `LogicalTensor.compare` 比较并报告 mismatch。

模型目录只提供 `probes.py`，不提供独立 compare 编排。

## tensors/ 目录

`tensors/` 只负责根据 spec/config 创建 `LogicalTensor` declarations 和 probe metadata。它不分配显存，不打开 checkpoint，不调用 shader。

### tensors/spec.py

`tensors/spec.py` 描述 OmniVoice 的静态配置和 shape 参数，只服务于创建 `LogicalTensor` declarations。

建议包含：

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

1. 从 config/model directory 读取创建 tensor declarations 所需的 shape/dtype 参数；
2. 提供 `tensors/audio_token_predictor.py`、`tensors/text_llm.py`、`tensors/audio_codec_decoder.py` 的 declaration helper 所需参数；
3. 定义 dtype 期望，不做 dtype cast；
4. 不打开 Vulkan，不创建 `LogicalTensor.storage`。

约束：

```text
spec 只在 tensors/ declaration 阶段使用。
根据 spec 创建出一批 LogicalTensor 之后，execution/forward/shaders/compare 不再引用 spec。
forward 只看 LogicalTensor.spec，不回头查 OmniVoiceTensorSpec。
```

### tensors/pipeline.py

声明 pipeline 级输入、token buffer、中间连接和最终输出。

示例职责：

```text
prompt token ids
language ids
generated audio tokens
selected tokens
done flag
waveform
wav pcm16 output
```

### tensors/audio_token_predictor.py

声明 audio_token_predictor 相关 tensors。

可能包括：

```text
audio_token_predictor.input_tokens
audio_token_predictor.audio_embedding.output
audio_token_predictor.hidden
audio_token_predictor.audio_head.logits
audio_token_predictor.audio_head.selected
```

如果 audio_token_predictor 内部有多个候选模型分支，也可以继续拆 helper，但不要拆出物理 workspace tree。

### tensors/text_llm.py

声明 Text LLM text model 的 prefill/decode tensors。

建议区分 base name 和 execution scope：

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

cond/uncond、audio token index 不直接写进 base name。它们由 run scope/artifact key 表达。

### tensors/audio_token_selector.py

声明 classifier-free guidance / token selection / argmax 等 tensors。

可能包括：

```text
audio_token_selector.cond_logits
audio_token_selector.uncond_logits
audio_token_selector.guided_logits
audio_token_selector.next_token
audio_token_selector.done
```

### tensors/audio_codec_decoder.py

声明 audio tokenizer / decoder / waveform 相关 tensors。

优先覆盖 audio_codec_decoder，因为它比较适合作为第一个真实 Vulkan 子链路：

```text
audio_codec_decoder.quantizer.tokens
audio_codec_decoder.quantizer.embed_sum
audio_codec_decoder.quantizer.project_out.hidden1024
audio_codec_decoder.quantizer.project_out.hidden256
audio_codec_decoder.decoder.conv1
audio_codec_decoder.decoder.block0.deconv
audio_codec_decoder.decoder.block0.res_unit1.conv1
audio_codec_decoder.decoder.block0.res_unit1.output
...
audio_codec_decoder.decoder.waveform
```

注意 token normalization 要作为 boundary contract 写清楚。OmniVoice audio_token_predictor 可能使用 `1024` 作为 audio mask token，但 audio_codec_decoder codebook 合法 index 是 `0..1023`。进入 audio_codec_decoder quantizer 边界时，candidate 和 PyTorch probe 必须使用一致规则，例如：

```text
clamp(token, 0, codebook_size - 1)
```

### tensors/outputs.py

声明 pipeline 最终输出：

```text
output.waveform
output.wav_pcm16
output.sample_rate
```

### tensors/ 禁止做的事

```text
不要调用 RuntimeSession.empty/load_weight/input
不要持有 BufferAllocation
不要读写 Vulkan buffer
不要读 checkpoint 内容
不要调用 shader
不要提供 frame.workspace.*.activation(name) 这种 API
```

## submodel forward 目录

### audio_token_predictor/forward.py

audio_token_predictor 的 Vulkan candidate forward。

职责：

1. 调用 audio_token_predictor 所需 shader wrapper；
2. 写出 audio_token_predictor logits/token/hidden 等 `LogicalTensor`；
3. 返回 audio_token_predictor output dataclass；
4. 不调用 PyTorch reference。

### text_llm/prefill_forward.py 和 text_llm/decode_forward.py

Text LLM 子模型分 prefill 和 decode 两条 forward。

职责：

1. 调用 embedding、rms_norm、linear、attention、mlp、logits 等 shader wrapper；
2. 读写 KV cache `LogicalTensor`；
3. 支持 cond/uncond 由 scope 或 state 区分；
4. 不在 logical tensor base name 里硬编码 audio token index/row。

### audio_token_selector/forward.py

audio_token_selector 不是 PyTorch 大模型，但仍按 submodel 处理。

职责：

1. 读取 cond/uncond logits；
2. 执行 guidance、mask、argmax/sample；
3. 写出 next token / done flag；
4. 记录这些 tensors 供对拍或 token 边界检查。

### audio_codec_decoder/forward.py

audio_codec_decoder 的 Vulkan candidate forward。

职责：

1. token normalize；
2. quantizer embedding sum；
3. project out；
4. decoder conv/deconv/residual/snake；
5. 写出 waveform。

audio_codec_decoder 适合作为首个真实落地子链路，因为它主要是 conv/activation/elementwise，边界比 Text LLM attention 更容易收敛。

## submodel weights.py

每个带权重的 submodel 都有自己的 `weights.py`：

```text
audio_token_predictor/weights.py
text_llm/weights.py
audio_codec_decoder/weights.py
```

职责：

1. 声明或收集该 submodel 的 weight tensors；
2. 根据 `WeightSource` 校验 checkpoint key/dtype/shape；
3. 调用 `RuntimeSession.load_weight()`；
4. 返回 bound weight dataclass。

禁止：

```text
不做隐式 dtype cast
不做权重 packing/reorder
不调用 shader
不做 PyTorch forward
```

如果未来需要 packed weight，必须显式建 preprocessing audio token 或专门 weight layout，不能伪装成 checkpoint 原始 tensor。

## submodel pytorch.py

每个 submodel 的 `pytorch.py` 负责加载/持有 PyTorch 或 official reference model。

```text
audio_token_predictor/pytorch.py
text_llm/pytorch.py
audio_codec_decoder/pytorch.py
```

职责：

1. 构建 official/PyTorch module；
2. 加载 reference 权重；
3. 准备和 candidate 对齐的 PyTorch inputs；
4. 暴露 model object 或 `run_*_pytorch_forward()`。

它不决定比较哪些 tensors。比较目标来自 candidate forward 收集到的 `LogicalTensor`。

## submodel probes.py

每个 submodel 的 `probes.py` 负责把 `LogicalTensor.pytorch_probe` 映射成 PyTorch hook/probe。

```text
audio_token_predictor/probes.py
text_llm/probes.py
audio_codec_decoder/probes.py
```

流程：

```text
candidate forward 结束后得到 collected LogicalTensors
  -> 按 submodel 过滤 tensors
  -> 过滤 compare != None 且 pytorch_probe != None
  -> 根据 probe metadata 在 PyTorch model 上安装 hook
  -> 执行 PyTorch forward
  -> hook 收集 reference artifacts: artifact_key -> torch.Tensor
  -> 移除 hook
```

`probes.py` 不应该手写 candidate 公式，例如不要写：

```python
return {"audio_codec_decoder.decoder.conv1": conv1d(x, w, b)}
```

reference 必须来自 PyTorch/official model forward 中捕获到的真实 tensor。

## shaders/ 目录

`shaders/` 放 OmniVoice 复用的 shader contract、variant、wrapper 和内联 GLSL source。

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

每个 shader Python 文件内联自己的 GLSL source，例如 `GLSL_SOURCE = """..."""`。不要再拆 `glsl/` 子目录。

每个 shader wrapper 只做:

1. 校验 tensor contract；
2. 选择具体 `ShaderVariant`；
3. 调用 `RuntimeSession.dispatch()`。

禁止：

```text
不要加载模型权重
不要创建模型 LogicalTensor 名字
不要分配显存
不要调用 PyTorch reference
不要隐藏 submodel forward 顺序
```

## Candidate 收集 LogicalTensors

`RuntimeSession.dispatch()` 每次 shader 调用都应该记录：

```text
dispatch index
shader name
reads: field -> LogicalTensor
writes: field -> LogicalTensor
symbols / dispatch dims
scope if available
```

candidate forward 结束后：

```text
used tensors = all read tensors + all written tensors
compare tensors = written tensors where compare != None and pytorch_probe != None
```

通常优先比较 written tensors，因为它们是 Vulkan candidate 产生的边界。

对于 audio token generation loop，dispatch record 需要携带 scope：

```text
audio.token_005/row=cond
audio_codec_decoder.decoder
```

artifact key 由 scope + logical tensor base name 组成。

## 最小数据流

```text
pytest or script defines inline run inputs
        |
tensors/spec.py loads OmniVoiceTensorSpec
        |
tensors/ declares unbound LogicalTensors with compare/probe metadata
        |
RuntimeSession.prepare binds input/weights/activations/output from declarations
        |
execution.py runs full OmniVoice candidate pipeline
        |
audio_token_predictor/text_llm/audio_token_selector/audio_codec_decoder forward.py call shaders/ wrappers
        |
RuntimeSession.dispatch records used LogicalTensors
        |
RuntimeSession frame runner collects compare tensors from dispatch records
        |
submodel pytorch.py builds/runs reference models
        |
submodel probes.py installs hooks based on collected LogicalTensors
        |
RuntimeSession frame runner readbacks candidate tensors and compares artifacts
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

不要把 audio token index/row 写进 base name：

```text
bad: audio.token_005.text_llm.cond.layer.03.output
good base: text_llm.decode.layer.03.output
good scope: audio.token_005/row=cond
```

不要用物理位置命名 logical tensor：

```text
bad: workspace.core.hidden_states
bad: frame.attention.q_proj
bad: buffer17.slice3
```

## MVP 落地顺序

建议按这个顺序落地：

1. 建 `tensors/spec.py`、`tensors/audio_codec_decoder.py`、`audio_codec_decoder/forward.py`、`audio_codec_decoder/weights.py`、`audio_codec_decoder/probes.py`，并让 `RuntimeSession.prepare()` 能自动 materialize 这些 declarations。
2. 写 audio_codec_decoder 最短链路：tokens -> embed_sum -> project_out -> conv/deconv subset -> waveform 或中间 boundary。
3. 写 `shaders/conv1d_f32.py`、`deconv1d_f32.py`、`snake_f32.py` 的 contract 和 wrapper。
4. `execution.py` 先只调用 audio_codec_decoder forward。
5. candidate 跑完后从 dispatch records 收集 compare tensors。
6. `audio_codec_decoder/pytorch.py` 加载 official/PyTorch audio_codec_decoder。
7. `audio_codec_decoder/probes.py` 根据 collected tensors 安装 hook。
8. RuntimeSession Frame runner readback + compare。
9. audio_codec_decoder 稳定后再接 audio_token_selector、audio_token_predictor、Text LLM decode/prefill。

## MVP 验收

第一个 OmniVoice 子链路完成时应满足：

1. `tensors/audio_codec_decoder.py` 可以不启动 Vulkan 生成 declarations；
2. `shaders/` 可以不加载 OmniVoice 就校验 contract；
3. `audio_codec_decoder/forward.py` 只表达 Vulkan shader forward；
4. `execution.py` 表达 pipeline，即使初期只有 audio_codec_decoder；
5. candidate forward 后能从 dispatch records 收集 used LogicalTensors；
6. PyTorch/official forward 根据 collected LogicalTensors 动态安装 hook；
7. compare 只比较 candidate 实际写出且声明了 compare/probe 的 tensors；
8. 没有 `frame.workspace.*.activation(name)` 这种 API；
9. 权重 dtype/shape 必须和 checkpoint 声明一致，不做 silent cast。
