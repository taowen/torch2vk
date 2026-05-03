# Record / Replay 机制

本文是 `torch2vk` 的 runtime replay 设计文档。核心边界延续
`docs/frame-and-logical-tensor.md`：模型目录只写 eager execution，replay 只能由 `RuntimeSession`
从真实 dispatch facts 生成，不能成为另一套模型 graph、IR 或 replay-only execution。

核心规则：

```text
模型目录只写 eager execution。
RuntimeSession 在 eager execution 中记录已经 materialize 的 dispatch stream。
Replay 复用这条 dispatch stream 的 Vulkan command buffer、descriptor binding、push constants 和 storage fingerprint。
```

Replay 不是重新解释模型。Replay 是把一次真实 eager 运行中发出的 shader dispatch 序列录下来，在同一个
shape/control-flow regime 下重复提交。

## 一句话规则

```text
Python eager execution 是唯一模型程序。
Record 捕获 eager 实际发出的 dispatch 事实。
Replay 提交捕获后的 Vulkan command sequence。
```

所以模型 adapter 不需要知道 replay 存在。它只需要继续这样写：

```python
def run_audio_codec_decoder_frame(rt, tensors, *, pytorch_model):
    with rt.frame("audio_codec_decoder", scope={"domain": "audio"}, pytorch_model=pytorch_model):
        SHADER_A(rt, x=tensors.x, weight=tensors.w0, output=tensors.h0)
        SHADER_B(rt, x=tensors.h0, weight=tensors.w1, output=tensors.waveform)
        return AudioCodecDecoderOutput(waveform=tensors.waveform)
```

同一段 eager 代码可以被三种 runtime policy 消费：

```text
普通 eager
  ShaderVariant.__call__ -> RuntimeSession.dispatch -> 立即提交 Vulkan dispatch

record capture
  ShaderVariant.__call__ -> RuntimeSession.dispatch -> materialize/prepare -> 追加 PreparedDispatch 到 ReplayCapture

replay
  不再运行 Python model code；直接提交已录制 command buffer
```

## Capture 捕获什么

一次 dispatch 在 eager 中经过这些阶段：

```text
LogicalTensor declarations
  -> RuntimeSession materialize reads/writes
  -> ShaderContract validate
  -> resolve shape symbols
  -> materialize uniforms
  -> pack push constants
  -> resolve specialization constants
  -> get/create compute pipeline
  -> bind descriptor set
  -> submit dispatch
  -> record DispatchRecord
```

Replay capture 保存的是这里的执行态结果，而不是高层 Python 调用：

```text
shader variant / SPIR-V artifact
pipeline layout key
descriptor buffers: allocation + offset + range
push constants bytes and decoded values
specialization constants
dispatch group count
read buffer ranges
write buffer ranges
frame/scope/logical read-write metadata
```

这些内容足够重放 Vulkan command。模型目录里的 `execution.py`、frame 函数和 `ShaderVariant.__call__` 在 replay
热路径中不再执行。

## PreparedDispatch

推荐把 capture 的基本单位叫 `PreparedDispatch`：

```python
@dataclass(slots=True)
class PreparedDispatch:
    variant: ShaderVariant
    contract: ShaderContract
    tensors: Mapping[str, MaterializedTensor]
    descriptor_buffers: tuple[DescriptorBufferBinding, ...]
    pipeline: ComputePipeline
    binding: BoundComputeBinding
    push_constants: bytes | None
    push_constant_values: Mapping[str, int | float]
    specialization_constants: Mapping[int, int] | None
    dispatch: tuple[int, int, int]
    read_ranges: tuple[BufferRange, ...]
    write_ranges: tuple[BufferRange, ...]
    dispatch_record: DispatchRecord
```

`PreparedDispatch` 是 runtime 内部对象，不是模型 DSL。它已经把 `LogicalTensor` 解析成了具体
`MaterializedTensor` 和 descriptor buffer range。

模型目录永远不应该构造 `PreparedDispatch`。

## ReplayCapture

`RuntimeSession` 可以提供 capture scope：

```python
with RuntimeSession.open(device_index=0) as rt:
    rt.register_model(tensors, model_dir=model_dir)
    rt.register_inputs(feeds)

    with rt.capture_replay(
        name="omnivoice.tts.full_pipeline",
        regime={"text_len": 64, "max_audio_tokens": 256, "speaker": "zero_shot"},
    ) as capture:
        result = run_omnivoice_pipeline(
            rt,
            tensors,
            pytorch_models=pytorch_models,
            max_audio_tokens=256,
        )

    replay = capture.finalize()
    replay.validate()
    replay.unsafe_replay()
```

在 `capture_replay(...)` 内，模型仍然走普通 eager execution。差别只是 `RuntimeSession.dispatch()`
在每次 dispatch 后把 prepared dispatch 追加到 capture。

捕获结束时：

```text
PreparedDispatch list
  -> 计算读写 buffer range
  -> 自动插入 compute barriers
  -> 录制 Vulkan command buffer
  -> 计算 shader fingerprint
  -> 计算 resource fingerprint
  -> 生成 ReplaySession
```

## RecordedSequence

Replay command buffer 是按 prepared dispatch 顺序录制的：

```text
for dispatch in prepared_dispatches:
  if pending writes 与当前 read/write ranges overlap:
    vkCmdPipelineBarrier(shader write -> shader read/write)
  vkCmdBindPipeline
  vkCmdBindDescriptorSets
  vkCmdPushConstants
  vkCmdDispatch(group_x, group_y, group_z)
  pending_writes += dispatch.write_ranges
```

barrier 来自 buffer range overlap，而不是模型语义。只要 contract 正确标了 `INPUT` / `OUTPUT` /
`INOUT`，runtime 就能从 descriptor ranges 推导同步需求。

MVP 可以先在每个 dispatch 之间插 conservative compute barrier。后续再用 read/write range overlap
减少 barrier。

## ReplaySession

ReplaySession 至少持有：

```python
@dataclass(slots=True)
class ReplaySession:
    name: str
    regime_key: RegimeKey
    sequence: RecordedSequence
    prepared_dispatches: tuple[PreparedDispatch, ...]
    shader_fingerprint: str
    resource_fingerprint: str
```

对外提供：

```python
def validate(self) -> None: ...
def replay(self) -> None: ...
def unsafe_replay(self) -> None: ...
def close(self) -> None: ...
```

`replay()` 是安全入口：

```text
检查 command buffer 仍有效
检查 shader artifact / dispatch topology 没变
检查 descriptor resource fingerprint 没变
检查 replay 没有触发 allocation growth
提交 command buffer
```

`unsafe_replay()` 是热路径入口：

```text
不做 fingerprint
不做 debug
不做 compare
不做 readback
不做 allocation
只提交已录制 command buffer
```

只有在 stage/pipeline 已经确认当前 request 仍匹配同一个 regime 时，才能调用 `unsafe_replay()`。

## Fingerprint

Replay 的关键不是“这段 Python 看起来一样”，而是“底层 Vulkan ABI 和资源绑定没变”。

推荐两类 fingerprint：

```text
shader_fingerprint
  SPIR-V 内容 hash
  shader variant name
  dispatch topology
  descriptor binding ABI
  tensor field dtype/shape/layout
  specialization constants
  push constant size

resource_fingerprint
  device identity
  descriptor buffer allocation identity
  descriptor offset/range
  materialized tensor slice allocation/offset/nbytes
  model/request/state buffer identity
```

如果 shader 重新编译、shape 变了、descriptor range 变了、frame arena 被 reset 后重新分配到了不同
buffer，旧 replay session 必须失效。

## RegimeKey

Replay 是 regime-specialized，不是对任意未来输入都无条件有效。

`RegimeKey` 描述这条 replay capture 适用的 shape 和控制流条件：

```python
RegimeKey(
    name="omnivoice.tts.full_pipeline",
    values=(
        ("text_len", 64),
        ("max_audio_tokens", 256),
        ("codec_window", 2048),
        ("dtype", "f16_weights_f32_acc"),
    ),
)
```

哪些东西应该进入 regime：

```text
影响 dispatch 数量的参数
影响 dispatch group count 的 shape
影响 descriptor range 的 shape
影响 specialization constants 的值
影响 pipeline 分支的模式
影响 loop 次数的生成上限或固定输出长度
影响 workspace high-water mark 的尺寸
```

哪些东西不一定进入 regime：

```text
同 shape 的输入数值
同 shape 的权重数值，但权重 buffer identity 必须在 resource fingerprint 中稳定
request 内当前语义状态，只要它写入的是 replay 绑定的稳定 state buffer
```

如果 TTS 生成提前遇到 stop token，导致实际 audio token 数少于 capture 时的 loop 次数，那么完整 pipeline
replay 不再适用。可以改用固定 `max_audio_tokens` 的 capture，或者按 stage/step 捕获更小的 replay session。

## 为什么复杂长 pipeline 也能 replay

长 pipeline 本质上仍然只是更长的 dispatch stream。

例如 TTS 可能串联多个模型：

```text
text/token preparation
  -> audio token predictor
  -> text LLM cond decode
  -> text LLM uncond decode
  -> audio token selector
  -> audio codec decoder
  -> waveform postprocess
```

模型目录中会写成多个 frame 和多个模型 adapter 的 eager 调用：

```python
def run_omnivoice_pipeline(rt, tensors, pytorch_models, *, max_audio_tokens):
    selected = None

    with rt.request("omnivoice.tts", scope={"request_id": request_id}):
        run_text_prefill_frame(rt, tensors.text, pytorch_model=pytorch_models.text)

        for audio_token_index in range(max_audio_tokens):
            predictor = run_audio_token_predictor_frame(
                rt,
                tensors.predictor,
                pytorch_model=pytorch_models.predictor,
                scope={"audio_token_index": audio_token_index},
            )
            cond = run_text_decode_frame(
                rt,
                tensors.text_cond,
                predictor,
                pytorch_model=pytorch_models.text_cond,
                scope={"audio_token_index": audio_token_index, "row": "cond"},
            )
            uncond = run_text_decode_frame(
                rt,
                tensors.text_uncond,
                predictor,
                pytorch_model=pytorch_models.text_uncond,
                scope={"audio_token_index": audio_token_index, "row": "uncond"},
            )
            selected = run_audio_token_selector_frame(rt, tensors.selector, cond, uncond)

        return run_audio_codec_decoder_frame(
            rt,
            tensors.codec,
            selected.tokens,
            pytorch_model=pytorch_models.codec,
        )
```

Capture 不关心这是一个模型还是多个模型。它只看到：

```text
dispatch 000 frame=text_prefill         shader=...
dispatch 001 frame=text_prefill         shader=...
dispatch 057 frame=audio_predictor      shader=...
dispatch 081 frame=text_decode cond     shader=...
dispatch 113 frame=text_decode uncond   shader=...
dispatch 132 frame=audio_selector       shader=...
...
dispatch 900 frame=audio_codec_decoder  shader=...
```

只要这些 dispatch 的 descriptor 读写关系和 resource lifetime 是稳定的，runtime 就可以把它们录成一条
大的 `RecordedSequence`。

## 跨模型数据如何衔接

跨模型衔接靠 `LogicalTensor` 和 request/pipeline lifetime，不靠 replay 特殊规则。

```text
audio_token_selector.selected_tokens
  role=STATE 或 OUTPUT
  memory=REQUEST_STATE
  lifetime=REQUEST

audio_codec_decoder.codes
  role=INPUT
  feed/downstream source 指向 selected_tokens
```

eager execution 中，selector frame 写出 request-lifetime tensor；decoder frame 读取同一个 materialized
state。Capture 记录的是这两个 dispatch range：

```text
selector dispatch writes selected_tokens buffer range
decoder dispatch reads selected_tokens buffer range
```

Replay 时同一条 command buffer 先写后读。barrier 由 buffer range overlap 推导出来。它不需要知道
“selector”和“decoder”是两个模型。

## Frame、Request、Pipeline 的 replay 边界

Replay 可以按不同粒度捕获：

```text
Frame replay
  适合单个 PyTorch forward 边界调试和 microbenchmark。

Stage replay
  适合 decode step、audio codec decoder、encoder block 这类 shape 稳定的重复阶段。

Pipeline replay
  适合固定输入 shape / 固定 loop 上限 / 固定分支的完整 request。
```

三者机制相同，差别只是 capture scope 包住多少 eager dispatch。

推荐第一版优先支持：

```text
Frame replay
  最小闭环，容易验证。

Stage replay
  性能收益最大，例如 autoregressive decode 的单 token step。
```

完整 pipeline replay 可以在 request/state lifetime 和 dynamic control flow 规则清楚后再打开。

## 动态循环和条件分支

Replay 只能复现 capture 时的 dispatch topology。

如果 eager execution 中有：

```python
for i in range(max_audio_tokens):
    ...
    if selected.done:
        break
```

那么 replay 有三种策略：

```text
固定 topology
  capture 固定 max_audio_tokens，不在 replay 内提前 break。适合 benchmark 或固定长度生成。

step replay
  每个 decode step 捕获一个 replay session。Python 仍负责循环和 stop 判断，但 step 内 shader 序列用 replay。

multi-regime replay cache
  为不同 loop count / branch outcome 建不同 ReplaySession，用 RegimeKey 选择。
```

不要假装一条 capture 可以自动适配不同分支。不同 dispatch topology 必须对应不同 replay session。

## 输入更新

Replay command buffer 绑定的是稳定 buffer，不是稳定输入值。

如果下一次 request 的输入 shape 相同，可以复用 replay session，但必须在 replay 前把新输入写入相同的
HOST_INPUT / REQUEST_STATE buffer：

```text
upload new prompt tokens to existing prompt_token_ids buffer
upload new acoustic features to existing mel/features buffer
reset request state buffers
unsafe_replay()
read final output
```

如果输入 buffer 重新分配了，resource fingerprint 会变，必须重新 capture。为了复用 replay，runtime 应
优先在 request/pipeline materialization 阶段复用同一批 stable slots。

## State 更新

Replay 允许 state buffer 的内容变化，但不允许 state buffer 的绑定身份变化。

例如 autoregressive decode：

```text
KV cache buffer 固定
cache_position buffer 固定
token input buffer 固定
logits output buffer 固定
```

每步 Python 更新 token/cache_position 的内容，然后提交 step replay。command buffer 仍然绑定同一批
buffer range，shader 会读到新内容。

所以要区分：

```text
semantic state
  当前 token、当前长度、stop flag，可以变。

resource state
  buffer handle、offset、range、shape regime，replay 期间必须稳定。
```

## Allocation 纪律

Replay 能稳定工作的前提是 capture 后不再发生资源漂移：

1. replay session 持有或引用的 buffers 不能在 session 关闭前释放；
2. frame workspace 如果参与 replay，不能在 frame exit 后 reset 掉；
3. request/pipeline state 适合 replay，因为 lifetime 足够长；
4. stage-owned frame slots 适合 replay，因为它们为某个 regime 预分配并长期复用；
5. replay 热路径不能触发 allocation growth。

这意味着普通 debug frame 的 `FRAME_WORKSPACE` 不一定能直接被长生命周期 replay 引用。要 replay 一个
stage，需要 runtime 把该 stage 的 workspace 提升为 stage/request lifetime 的 stable slots，或者在 capture
结束后保证 replay session owns those allocations。

## 和 PyTorch 对拍的关系

PyTorch 对拍只发生在 eager candidate frame 中。

Replay 热路径不做：

```text
PyTorch model.forward
hook/probe
candidate readback
compare
debug dump
artifact cache
```

推荐流程：

```text
1. eager + PyTorch compare 验证正确性
2. eager capture replay session
3. safe replay 验证 fingerprint 和输出
4. hot path unsafe_replay
```

如果 replay 输出错了，先回到 eager compare 定位。不要给 replay 写第二套 debug model。

## Replay 和 liveness/aliasing

Replay capture 初期可以不做 aliasing，只记录实际 materialized buffer ranges。

后续优化顺序：

```text
eager dispatch records
  -> liveness analysis
  -> StoragePlan
  -> stable offset assignment
  -> capture replay with planned storage
```

liveness planner 消费的是 dispatch read/write edges，不是模型目录里的额外 graph IR。模型 adapter
仍然只写 eager execution。

## 失败和失效条件

以下情况必须重新 capture：

```text
shader SPIR-V 变了
ShaderContract manifest 变了
dispatch 数量或顺序变了
dispatch group count 变了
specialization constants 变了
push constant byte layout 变了
descriptor binding index/type/range 变了
buffer allocation identity 变了
shape regime 变了
控制流走了不同分支
stage workspace 被释放或重新分配
```

以下情况可以复用 replay：

```text
同 shape 下输入数值变了
同一 weight buffer 中权重内容保持或被显式更新后仍使用同一 buffer
request state buffer 内容变了
当前 token/position 等小输入内容变了
```

## 禁止事项

```text
不要在 models/<model_name>/ 下写 replay-only execution。
不要让 replay 绕开 ShaderContract。
不要把 replay plan 当成模型语义源头。
不要在 unsafe_replay 中做 PyTorch compare、readback 或 allocation。
不要让 command buffer 引用已经会在 frame exit reset 的 workspace。
不要用一条 capture replay 不同 loop count 或不同 branch outcome。
不要为了 replay 把动态 scope 写进 LogicalTensor.name。
不要让 replay session 在 shader/resource fingerprint mismatch 后继续执行。
```

## MVP

第一版建议这样落地：

1. `RuntimeSession.capture_replay(name, regime)` scope；
2. dispatch 时在 eager 执行之外追加 `PreparedDispatch`；
3. `ReplayCapture.finalize()` 录制 command buffer 并生成 `ReplaySession`；
4. 每个 dispatch 之间先插 conservative compute barrier；
5. 记录 shader/resource fingerprint；
6. 支持 `replay()` 和 `unsafe_replay()`；
7. 先支持 frame/stage replay，不急着支持完整动态 pipeline replay；
8. 测试用 toy frame 和一个两 frame pipeline 验证跨 frame write/read；
9. 再用 OmniVoice audio codec decoder 或 Qwen3-ASR audio encoder 做真实 stage replay。

这个 MVP 已经能证明关键架构：模型目录只定义 eager execution，通用 runtime 可以从 eager dispatch
事实生成 replay。
