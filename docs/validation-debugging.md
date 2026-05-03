# 验证与调试设计

现在的 debug 目标不是“单个 shader 能跑”，而是端到端生成能定位错误。以
OmniVoice 为例，它不是一个模型：

```text
text prompt
  -> tokenizer
  -> stage0 audio embedding
  -> Qwen3 text decoder cond/uncond prefill
  -> stage0 audio head logits
  -> iterative token selection, repeat N steps
  -> stage1 audio tokenizer / decoder
  -> waveform
  -> wav postprocess
```

任何一段都可能错；错误也不一定发生在第一个 token，可能到第 `n` 个 generation step、
第 `m` 个 codebook、某个 cond/uncond row、某个 decoder block 才发散。所以验证入口只能
是一个 text -> wav 的完整 run。单 shader synthetic 用例可以帮助开发某个 kernel，但不能
进入“集成测试”的概念，也不能作为 debug 机制有效的证据。

## 单一集成入口

集成测试只有一个：端到端 case。

```python
def test_omnivoice_text_to_wav_matches_reference_or_reports_shader():
    case = OmniVoiceDebugCase(text="hello world", target_steps=8, num_steps=8)
    report = run_e2e_debug_case(
        family="omnivoice_safetensor",
        case=case,
        model_dir=resolve_model_dir(),
    )
    report.raise_for_mismatch()
```

这一个入口内部可以做很多阶段的定位，但对测试调用方只暴露端到端结果：

```text
capture_reference  跑 official/PyTorch，缓存完整端到端 trace
locate_candidate   跑 Vulkan eager，轻量记录边界、token、dispatch timeline
compare_boundaries 比较 reference/candidate，找到第一个坏 step + boundary
drilldown          只针对坏 boundary 重跑或读回细节，定位 input_ok_output_bad 的 shader
root_cause         如果 input 已经坏，沿 upstream boundary 自动回溯
```

因此单 kernel synthetic 用例只能作为临时开发辅助，不应进入端到端集成测试集合，也不能证明
debug 机制成立。真正的 integration test 必须从文本开始，最终覆盖 audio tokens/wav，并在
失败时自动归因到 shader。

boundary 顺序必须由模型 family 的 tensor schema 提供，而不是由测试临时手写。torch2vk 中
Qwen3 和 OmniVoice 分别通过 `tensors/boundaries.py` 声明端到端 boundary、compare policy、
step-scoped/global scope，以及一个 boundary 对应的实际 artifact names。测试只调用 schema，
避免测试层变成第二套 debug 设计。OmniVoice 的 `stage0.audio_head.logits` boundary 不是单个
artifact，而是同时覆盖 `stage0.audio_head.logits.cond` 和
`stage0.audio_head.logits.uncond`；LLM input/layer/final norm boundary 也按 cond/uncond 成对
声明。

候选侧必须使用 torch2vk 自己的执行框架：`DebugIntegrationCase` 负责 storage、资源和
`ReferenceProvider` 绑定；`DebugContext` 负责 eager dispatch、candidate readback、dispatch
timeline 记录和 run 结束后的对拍。模型代码只做普通 Python 函数调用。agentorch 只能作为设计
参考，不能作为候选 runtime，也不能用 bridge 绕过 torch2vk 的 LogicalTensor、storage plan、
Vulkan dispatch 和 readback。

execution 函数不传 PyTorch model。LogicalTensor 的收集来自 shader 函数的参数：

```python
OMNIVOICE_AUDIO_HEAD_MAT_VEC_F32_F32(
    ctx,
    weight=weights.stage0.audio_heads,
    x=tensors.audio_head_hidden,
    output=tensors.audio_head_logits,
)
```

`ShaderVariant.__call__` 只接收 `ctx` 和具名 `LogicalTensor`，然后进入
`DebugContext.dispatch(variant, tensors)`。dispatch record 记录本次 shader 的 reads/writes；
写出的 comparable tensors 被 readback 到 candidate artifacts。PyTorch/official reference
只作为 `ReferenceProvider` 绑定在集成 case 上，等 Vulkan run 完成、实际 comparable writes
已经确定后再按需 cache-first 捕获。

Correctness 不能靠换成低性能 shader 解决。debug eager 的 shader 调用仍然是候选实现本身；
如果 checkpoint 权重是 `float32`，shader ABI 必须直接消费 `float32`，但不能把热路径退化成
每个输出一个线程串行扫 hidden 维度的 scalar kernel。发现 dtype/ABI 不一致时，正确处理是：

```text
1. 在 LogicalTensor weight tree 和 safetensor verifier 中锁定真实 dtype/shape；
2. 暂时只把已满足性能/ABI 要求的边界接入端到端 debug；
3. 对热路径补匹配真实 dtype 的并行 shader，再接入对拍；
4. 不保留未使用、命名误导或只为正确性存在的慢 shader。
```

同理，当前 debug run 只绑定已经执行的 concrete tensors。最终 tree 可以声明 waveform 这类
符号 shape 边界，但在 stage1 decoder 尚未接入前，不能把整个 tree 交给 storage planner；
否则会把 schema 中的未来目标误判成当前必须分配的 Vulkan buffer。

OmniVoice 的 stage0 audio head 可能产生 mask token `1024`，而 stage1 codebook embedding
只有 `0..1023`。stage1 quantizer shader 的 contract 是 clamp 到 `[0, V-1]` 后再查表；
PyTorch/reference artifact 也必须按同一 contract 归一化，否则 reference 侧会先越界，而不是
暴露 Vulkan 计算差异。

当前 torch2vk debug smoke 已覆盖到 stage1 quantizer 的 concrete 边界：

```text
stage1.quantizer.embed_sum
stage1.quantizer.project_out_sum.hidden1024
stage1.quantizer.project_out_sum.hidden256
stage1.decoder.conv1
stage1.decoder.block0.deconv
stage1.decoder.block0.res_unit1.conv1
stage1.decoder.block0.res_unit1.output
stage1.decoder.block0.res_unit2.conv1
stage1.decoder.block0.res_unit2.output
stage1.decoder.block0.res_unit3.conv1
stage1.decoder.block0.res_unit3.output
stage1.decoder.block1.deconv
stage1.decoder.block1.res_unit1.conv1
stage1.decoder.block1.res_unit1.output
stage1.decoder.block1.res_unit2.conv1
stage1.decoder.block1.res_unit2.output
stage1.decoder.block1.res_unit3.conv1
stage1.decoder.block1.res_unit3.output
stage1.decoder.block2.deconv
stage1.decoder.block2.res_unit1.conv1
stage1.decoder.block2.res_unit1.output
stage1.decoder.block2.res_unit2.conv1
stage1.decoder.block2.res_unit2.output
stage1.decoder.block2.res_unit3.conv1
stage1.decoder.block2.res_unit3.output
stage1.decoder.block3.deconv
stage1.decoder.block3.res_unit1.conv1
stage1.decoder.block3.res_unit1.output
stage1.decoder.block3.res_unit2.conv1
stage1.decoder.block3.res_unit2.output
stage1.decoder.block3.res_unit3.conv1
stage1.decoder.block3.res_unit3.output
stage1.decoder.block4.deconv
stage1.decoder.block4.res_unit1.conv1
stage1.decoder.block4.res_unit1.output
stage1.decoder.block4.res_unit2.conv1
stage1.decoder.block4.res_unit2.output
stage1.decoder.block4.res_unit3.conv1
stage1.decoder.block4.res_unit3.output
stage1.decoder.waveform
```

这一段已经覆盖到 waveform。剩余端到端 gap 不在 stage1 decoder 内，而在完整
OmniVoice 文本生成 loop：当前 smoke 仍用 deterministic debug token/hidden fixture，没有跑
official tokenizer + Qwen3 cond/uncond iterative generation 的完整 text -> wav case。

## 自动归因

可借鉴的经验是：端到端测试不手动选择“测哪个 shader”。它先比较高层边界，再对第一个坏
边界做 drilldown；但执行和记录都发生在 torch2vk runtime 内。

```text
reference trace:
  tensors.npz / tokens.npz / timeline.json

candidate locate:
  tensors.npz / tokens.npz / timeline.json / dispatches.json / checkpoints.json

debug report:
  first_bad_step
  first_bad_boundary
  first_bad_dispatch
  classification
  hops
```

drilldown 的分类必须能指导下一步：

```text
input_ok_output_bad
  当前 boundary 的 writer shader 是根因候选，报告 first_bad_dispatch

input_bad_output_bad
  当前 boundary 的输入已经坏，自动跳到 upstream boundary

boundary_output_match
  locate 阶段的边界不再复现，报告 state transition / instrumentation gap

boundary_coverage_insufficient
  没有足够边界或 checkpoint，说明 LogicalTensor tree/probe 设计缺口
```

这个过程是集成测试的一部分，不是另一个测试层级。

## 端到端 Run

Debug 入口应该描述一次完整 run，而不是一个孤立 forward：

```python
case = OmniVoiceDebugCase(
    text="hello world",
    language="English",
    target_steps=8,
    num_steps=8,
    seed=20260501,
    position_temperature=0.0,
    denoise=False,
)

ctx = DebugContext.recording(...)
run_omnivoice_debug(
    ctx,
    case=case,
    tensors=omnivoice_tensors(case, spec),
    weights=weights,
)
ctx.compare_records(reference_provider)
```

`run_omnivoice_debug()` 仍然是普通 Python eager 代码，shader 仍然像函数一样调用。但这个
函数的粒度是端到端 generation loop，而不是单个模型 forward。

```python
def run_omnivoice_debug(ctx, *, case, tensors, weights):
    prompt = tokenize(case.text, case.language)
    audio_tokens = init_masked_audio_tokens(case)

    for step, unmask_count in enumerate(unmask_schedule(case)):
        scope = tensors.step(step)

        run_stage0_audio_embedding(
            ctx,
            tensors=scope.stage0,
            weights=weights.stage0,
            audio_tokens=audio_tokens,
        )
        run_qwen3_prefill(
            ctx,
            tensors=scope.cond_prefill,
            weights=weights.llm,
            row="cond",
        )
        run_qwen3_prefill(
            ctx,
            tensors=scope.uncond_prefill,
            weights=weights.llm,
            row="uncond",
        )
        run_stage0_audio_head(
            ctx,
            tensors=scope.audio_head,
            weights=weights.stage0,
        )
        audio_tokens = select_next_audio_tokens(
            ctx,
            tensors=scope.selection,
            current=audio_tokens,
            unmask_count=unmask_count,
        )

    run_stage1_decoder(
        ctx,
        tensors=tensors.stage1,
        weights=weights.stage1,
        audio_tokens=audio_tokens,
    )
    write_wav_postprocess(ctx, tensors=tensors.output)
```

这仍然没有第二套 PyTorch eager loop。Vulkan 是按同一 run 的 Python 调用顺序 eager 执行；
PyTorch/official reference 是 Vulkan run 结束后按本次实际 dispatch records 生成的 artifact
provider。这样不会在 LogicalTensor 还没收集完整时提前执行 PyTorch。

## Reference Capture

单模型场景可以用 `PyTorchForwardCapture(model).run()`。OmniVoice 需要的是
`ReferenceRunCapture`：它捕获完整 generation trace，并按 LogicalTensor scope 落盘。

```python
class ReferenceRunCapture:
    def run(self, required):
        trace = {}
        official = load_official_omnivoice()
        official.generate(
            text=case.text,
            language=case.language,
            target_steps=case.target_steps,
            num_steps=case.num_steps,
            seed=case.seed,
            callbacks=ReferenceCallbacks(trace, required),
        )
        return normalize_artifacts(trace, required)
```

关键点：

```text
cache key = checkpoint identity + config + case args + probe schema version
artifact key = LogicalTensor.name + scope
scope = phase / step / row / layer / submodel
```

如果 cache 命中，端到端 PyTorch/official run 可以跳过；如果 cache miss，必须一次性捕获完整
reference trace。`required` 来自 Vulkan eager 已经跑出的 dispatch timeline 和 comparable
LogicalTensor writes，而不是在第一个 shader 前预先猜测。不能为了 Vulkan 每个 shader 调用再
单独跑一段 PyTorch。

torch2vk 里的对应抽象是 `TraceReferenceProvider`：capture 函数返回 official/PyTorch
端到端 trace，provider 再按每个 `LogicalTensor.pytorch_probe.source` 映射成
`LogicalTensor.name -> torch.Tensor` artifact。reference cache 面向 torch2vk 的
LogicalTensor 名字，而不是 official runtime 的内部命名。当前 OmniVoice 已有
`omnivoice_official_reference_provider()` 懒加载入口；provider 构造不会加载官方模型。默认 CI
只验证这个懒加载和缺依赖错误路径；设置
`TORCH2VK_RUN_OMNIVOICE_OFFICIAL_REFERENCE=1` 时会额外跑最小 official text -> wav capture smoke。
`ReferenceTrace` 已经包含 `timeline` 字段；当前 official smoke 至少记录 final wav boundary，
并同时产出 `output.wav` 和与最终 boundary 对齐的 `output.wav_pcm16`。后续接入 layer/token probes
时继续向同一个 trace 追加 step events。
`boundary_coverage()` 可直接对 schema 和 trace 做覆盖检查；当前能明确报告
`output.wav_pcm16` 已覆盖，而 stage0/Qwen/token 边界仍缺 reference artifacts。

## Scope

只用 tensor 名不够。OmniVoice 的同一个语义 tensor 会在多个 generation step 出现：

```text
generate.step_000.stage0.audio_head.logits.cond
generate.step_001.stage0.audio_head.logits.cond
generate.step_007.stage0.audio_head.logits.cond
```

因此 dispatch record 和 artifact cache 都必须记录 scope：

```python
with ctx.scope("generate", step=step), ctx.scope("stage0.audio_head", row="cond"):
    OMNIVOICE_AUDIO_HEAD_MAT_VEC(
        ctx,
        weight=weights.stage0.audio_heads,
        x=tensors.stage0.audio_head_hidden,
        output=tensors.stage0.audio_head_logits,
    )
```

最终 artifact key 可以由框架生成：

```python
artifact_key = record.artifact_key(tensor.name)
```

不要把 `step`、`row`、`layer` 硬编码进所有 tensor base name。base name 表示语义位置；
scope 表示这次端到端 run 中第几次出现。

当前 `DebugContext.scope(...)` 已经把 scope 写进 `DispatchRecord.scope`，并提供
`DispatchRecord.artifact_key(tensor.name)` 生成 scoped artifact key。candidate readback 已按
这个 key 落 artifact；compare 也优先使用 scoped key，找不到时才回退到旧的 unscoped key。
Qwen3 单 prefill smoke 仍使用空 scope；OmniVoice debug smoke 已经用 `debug/step=0` scope
验证 scoped candidate/reference artifact 能对齐。完整 generation loop 接入后，reference cache
必须继续产出同样的 scoped key，避免第 0 步和第 n 步同名 tensor 互相覆盖。
`TraceReferenceProvider` 已支持这条路径：official trace 中的 `generate/step=1.<probe.source>`
会映射成 `generate/step=1.<LogicalTensor.name>`。

## 对拍策略

OmniVoice 这类动态 eager run 使用 record-first 对拍。DebugContext 每次 shader 函数调用后
立即执行 Vulkan、记录 dispatch，并 readback 当前写出的 comparable tensors；但不立刻跑
PyTorch reference：

```python
def dispatch(self, variant, tensors):
    record = self.record(variant, tensors, scope=self.scope)
    self.vulkan.run(record)
    self.records.append(record)

    for tensor in comparable_writes(record):
        key = record.artifact_key(tensor.name)
        self.candidate[key] = self.readback(tensor)
        self.required_reference.add(tensor.pytorch_probe)
```

Vulkan eager run 结束后，框架再根据实际记录生成 reference manifest，cache-first 捕获
PyTorch/official artifacts，然后按 dispatch timeline 比较：

```python
ctx.ensure_reference(reference_provider, tensors=ctx.comparable_written_tensors())
for record in ctx.records:
    for tensor in comparable_writes(record):
        key = record.artifact_key(tensor.name)
        compare(reference[key], ctx.candidate[key], tensor.compare).raise_for_mismatch(
            shader=record.shader,
            dispatch_index=record.index,
            scope=record.scope,
            tensor=tensor.name,
        )
```

失败报告必须能回答：

```text
哪个端到端 case
第几个 generation step
哪个 submodel / phase
cond 还是 uncond
哪个 LogicalTensor
哪个 shader 写坏
dispatch index
reference/candidate shape dtype
max_abs 或 token diff
```

示例：

```text
first mismatch:
  case: omnivoice hello-world target_steps=8 num_steps=8
  scope: generate.step_005/stage0.audio_head/row=cond
  tensor: stage0.audio_head.logits
  writer shader: omnivoice_audio_head_mat_vec_f32_f32
  dispatch: 1842
  reason: value mismatch max_abs=0.03125
```

## Token Divergence

端到端 token 生成有一个特殊风险：最终 wav 错不代表最后一个 shader 错。第一个坏点通常在
更早的 token selection。

因此 OmniVoice 的 reference trace 必须至少捕获：

```text
tokens.before(step)
stage0.audio_embedding.output(step)
qwen3 cond/uncond layer outputs(step, layer)
final_norm cond/uncond(step)
audio_head logits cond/uncond(step)
guided scores(step)
selected token/update mask(step)
tokens.after(step)
stage1 hidden / waveform
final wav metadata
```

定位流程：

```python
for step in generation_steps:
    compare(tokens.before(step))
    compare(stage0.audio_embedding.output(step))
    compare(prefill layer boundaries)
    compare(audio_head logits)
    compare(selection scores)
    compare(tokens.after(step))
compare(stage1.waveform)
compare(final.wav)
```

这样即使最终 wav 在第 8 步才错，也可以报告第一个坏 step 和对应 writer shader。

## 集成测试形态

OmniVoice 集成测试应该很短，但必须只表达端到端 case：

```python
def test_omnivoice_text_to_wav_matches_reference_or_reports_shader():
    case = OmniVoiceDebugCase(text="hello world", target_steps=8, num_steps=8)
    report = run_e2e_debug_case(
        family="omnivoice_safetensor",
        case=case,
        model_dir=resolve_model_dir(),
    )
    report.raise_for_mismatch()
```

短是因为通用框架负责：

```text
load reference cache or capture official/PyTorch trace
run Vulkan eager candidate trace
compare scoped token/tensor/wav boundaries
find first bad step + boundary
drill down through dispatches at that boundary
walk upstream if boundary inputs are already bad
report first bad shader or explicit coverage gap
persist artifacts and debug report
```

测试短不等于覆盖少。测试入口必须覆盖 text -> wav 的完整调用链；单 shader 和固定中间输入
都只是开发辅助，不能替代这个入口。

## Replay

Vulkan eager debug 顺便可以录制 replay，但 replay 不是主语义。主语义是“shader 裸调用立即
执行并对拍”。Replay 使用同一批 scoped `DispatchRecord`：

```python
sequence = VulkanReplaySequence.capture(ctx.records, tensors=bound_tensors)
sequence.replay()
```

Replay 只复现 Vulkan 侧；reference artifact 来自端到端 cache。
