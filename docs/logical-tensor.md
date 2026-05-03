# LogicalTensor Tree 设计

`LogicalTensor` 是模型可见 tensor 的语义句柄。它不是 storage slot，不是 hook id，也不是
某一次 dispatch 的临时名字。它描述一个 tensor 在模型算法里的身份，以及如何和 reference
artifact 对齐。

之前的设计过于偏向单模型 forward。对于 OmniVoice 这种多模型串联、循环生成的系统，最终
设计必须同时表达：

```text
submodel: stage0 / qwen3 cond / qwen3 uncond / stage1 / wav postprocess
generation step: 第 0..N 步
row: cond / uncond
layer: 第几层
tensor role: input / activation / logits / token / waveform
reference probe: 从 official/PyTorch trace 的哪里取
compare policy: tensor / token / waveform
```

这个 tree 服务的是 torch2vk runtime 本身。候选执行必须走 torch2vk 的
`DebugIntegrationCase`、`DebugContext`、storage plan、Vulkan dispatch 和 readback；
外部项目的 trace 只能作为 reference/设计材料，不能替代候选侧的 LogicalTensor tree。

LogicalTensor tree 还要保护 shader ABI，不允许为了快速对拍把 checkpoint 权重转换成另一个
dtype 去适配旧 shader。以 OmniVoice stage0 为例，`audio_embeddings.weight` 和
`audio_heads.weight` 是 `float32`，`codebook_layer_offsets` 是 `int64`；tree、verifier 和
shader contract 都必须反映这个事实。性能敏感边界如果暂时没有匹配真实 dtype 的并行 shader，
宁可先标记为 coverage gap，也不要接入一个慢 scalar kernel 伪装成候选实现。

## Base Name 和 Scope

`LogicalTensor.name` 表示语义位置，不直接编码“第几次出现”。

```text
stage0.audio_embedding.output
qwen3.layer.03.output
stage0.audio_head.logits
selection.guided_scores
stage1.decoder.waveform
output.wav_pcm16
```

端到端 run 中的动态维度由 scope 表示：

```text
generate.step_005/stage0.audio_head/row=cond + stage0.audio_head.logits
generate.step_005/qwen3.prefill/row=uncond/layer=03 + qwen3.layer.03.output
stage1.decoder + stage1.decoder.waveform
```

框架生成 artifact key：

```python
artifact_key = scope.key(tensor.name)
```

不要为每一步创建完全不同的字符串表：

```text
bad: generate.step_005.stage0.audio_head.logits.cond
good: scope(step=5, phase="stage0.audio_head", row="cond") + tensor.name
```

这样同一个 tensor tree 可以在 generation loop 中复用。

## Core API

理想核心对象：

```python
@dataclass(frozen=True, slots=True)
class LogicalTensor:
    name: str
    spec: TensorSpec
    layout: TensorLayout = TensorLayout.row_major()
    role: TensorRole = TensorRole.ACTIVATION
    memory: MemoryPolicy = MemoryPolicy.FRAME_WORKSPACE
    storage: BufferSlice | None = None
    source: WeightSource | None = None
    pytorch_probe: PyTorchProbe | None = None
    compare: ComparePolicy | None = None
```

需要新增的 run-scope 概念不应该塞进 `LogicalTensor.name`：

```python
@dataclass(frozen=True, slots=True)
class RunScope:
    path: tuple[str, ...] = ()
    labels: Mapping[str, int | str] = {}

    def child(self, name: str, **labels) -> RunScope: ...
    def key(self, tensor_name: str) -> str: ...
```

`LogicalTensor` 保持纯语义；`RunScope` 描述一次端到端执行中的位置。

execution 代码只消费这些 tree 对象，不持有 PyTorch model：

```python
def run_stage1_decoder(ctx, *, tensors, weights):
    OMNIVOICE_STAGE1_CONV1D_K7_F32(
        ctx,
        x=tensors.project_out_sum_hidden256,
        weight=weights.decoder_conv1_weight,
        bias=weights.decoder_conv1_bias,
        output=tensors.decoder_conv1,
    )
```

这里 `x/weight/bias/output` 本身就是 LogicalTensor 映射关系。框架从 shader contract 得到
哪些字段是 read、哪些字段是 write，再从实际 dispatch timeline 收集需要对拍的 writes。
PyTorch/official reference 不参与收集 LogicalTensor，只负责在候选 run 结束后为这些 tensor
names/probes 产出 reference artifact。

## Tensor Tree 目录

每个模型 family 仍然需要 `tensors/` 子目录，但 OmniVoice 不能只有一个平铺 tree。

```text
src/torch2vk/models/omnivoice_safetensor/
  execution.py
  tensors/
    __init__.py
    case.py
    run.py
    stage0.py
    qwen3.py
    stage1.py
    weights.py
    probes.py
```

职责：

```text
case.py
  Debug/generation case 参数：text、language、target_steps、num_steps、seed

run.py
  端到端 run tree，组合 stage0/qwen3/stage1/output

stage0.py
  audio embedding、audio head、selection 相关 tensor

qwen3.py
  cond/uncond prefill/decode 使用的 LLM tensor tree，可复用 qwen3_safetensor 的 layer tree

stage1.py
  audio tokenizer / decoder / waveform tensor

weights.py
  generator、llm、audio tokenizer 权重 tree

probes.py
  official/PyTorch trace source 到 LogicalTensor 的 probe helper

boundaries.py
  端到端 debug 的 boundary 顺序、compare policy 和 step/global scope

reference.py
  把 official/PyTorch 端到端 trace 接成 `ReferenceProvider`
```

## OmniVoice Run Tree

端到端 tree 应该表达完整生成流程：

```python
@dataclass(frozen=True, slots=True)
class OmniVoiceStepTensors:
    tokens_before: LogicalTensor
    stage0: OmniVoiceStage0Tensors
    qwen3_cond: Qwen3PrefillTensors
    qwen3_uncond: Qwen3PrefillTensors
    selection: OmniVoiceSelectionTensors
    tokens_after: LogicalTensor


@dataclass(frozen=True, slots=True)
class OmniVoiceRunTensors:
    prompt_ids: LogicalTensor
    steps: tuple[OmniVoiceStepTensors, ...]
    stage1: OmniVoiceStage1Tensors
    waveform: LogicalTensor
    wav_pcm16: LogicalTensor
```

`steps` 的长度来自 debug case 的 `num_steps`。每个 step 复用相同的 base names，但执行时用
`RunScope(step=i)` 区分 artifact。

这个 tree 不是为了拆出多层测试。集成测试仍然只跑端到端 case。tree 的职责是给端到端
自动归因提供边界、checkpoint 和 dispatch 归属。

实际执行时不能无条件把整棵最终 tree 都交给 storage planner。尚未落地的边界可能有符号 shape
（例如 `stage1.decoder.waveform: (B, samples, 1)`），它们应该留在 schema 中表达最终目标，
但不进入当前 run 的 `DebugIntegrationCase.tensors`。每个 debug run 只绑定本次真正执行和比较的
LogicalTensor；否则会把“未来覆盖范围”错误地变成当前 storage/dispatch 要求。

Tree 还应该把 token normalization 作为边界 contract 写清楚。OmniVoice stage0 使用
`1024` 作为 audio mask token，但 stage1 quantizer codebook 的 vocab 是 1024，合法 embedding
index 是 `0..1023`。进入 stage1 quantizer 边界时，candidate shader 和 reference probe 都按
`clamp(token, 0, vocab - 1)` 对齐。

当前已接入的 stage1 concrete tensors 包括 `stage1.quantizer.embed_sum`、
`stage1.quantizer.project_out_sum.hidden1024`、
`stage1.quantizer.project_out_sum.hidden256`、`stage1.decoder.conv1`、
`stage1.decoder.block0.deconv`、`stage1.decoder.block0.res_unit{1,2,3}.conv1`、
`stage1.decoder.block0.res_unit{1,2,3}.output`、`stage1.decoder.block1.deconv`、
`stage1.decoder.block1.res_unit{1,2,3}.conv1` 和
`stage1.decoder.block1.res_unit{1,2,3}.output`、`stage1.decoder.block2.deconv`、
`stage1.decoder.block2.res_unit{1,2,3}.conv1` 和
`stage1.decoder.block2.res_unit{1,2,3}.output`、`stage1.decoder.block3.deconv`、
`stage1.decoder.block3.res_unit{1,2,3}.conv1` 和
`stage1.decoder.block3.res_unit{1,2,3}.output`、`stage1.decoder.block4.deconv`、
`stage1.decoder.block4.res_unit{1,2,3}.conv1`、
`stage1.decoder.block4.res_unit{1,2,3}.output` 和 `stage1.decoder.waveform`。当前 debug
case 用 concrete `steps * 960` samples 绑定 waveform；最终 tree 仍可以用符号 `samples`
表达通用形态。

## Boundary 和 Checkpoint

端到端 debug 需要 first-class boundary。Boundary 是“可以比较、可以回溯、可以 drilldown”
的语义边界，不是独立测试层级。

```python
@dataclass(frozen=True, slots=True)
class DebugBoundary:
    name: str
    order: int
    scope: BoundaryScope
    artifacts: tuple[str, ...]
    tensors: tuple[LogicalTensor, ...]
    tokens: tuple[LogicalTensor, ...] = ()
    checkpoint: LogicalTensor | None = None
    writer_dispatch: bool = False
```

OmniVoice 需要的边界类似：

```text
tokens.before
stage0.audio_embedding.output
qwen3.prefill.input
qwen3.layer.00.output
...
qwen3.layer.27.output
qwen3.final_norm
stage0.audio_head.logits
selection.guided_scores
tokens.after
stage1.decoder.waveform
output.wav
```

`name` 是归因报告里的边界名，`artifacts` 是实际需要比较的 artifact 名。比如
`stage0.audio_head.logits` 这个 boundary 应同时比较
`stage0.audio_head.logits.cond` 和 `stage0.audio_head.logits.uncond`；LLM layer boundary
同理比较 cond/uncond 两侧。这样集成测试仍然只有一个边界顺序，但不会把多 row 的模型状态压扁
成单个 tensor。

`order` 用于同一个 step 内回溯；step N 的开头依赖 step N-1 的 `tokens.after`。如果某个
boundary drilldown 后发现 `input_bad_output_bad`，框架按 `(step, order)` 自动找上游
boundary，而不是让测试作者手动决定下一个测什么。

Checkpoint 是 drilldown 的重跑入口：

```text
boundary: stage0.audio_head.logits
checkpoint: stage0.audio_head.input_hidden
metadata:
  step=5
  row=cond
  tokens_before=...
```

没有 checkpoint 的 boundary 只能 locate，不能高效 drilldown。文档里的 tensor tree 必须
明确哪些 boundary 能提供 checkpoint，否则端到端失败时只能报告“覆盖不足”。

## 归因状态

LogicalTensor tree 和 boundary schema 应支持这些状态：

```text
match
  端到端 reference 和 Vulkan candidate 一致

input_ok_output_bad
  boundary 输入对，输出错；writer shader 是根因候选

input_bad_output_bad
  boundary 输入已经错；自动回溯上游 boundary

boundary_output_match
  locate 发现错，但 drilldown 不复现；说明 state transition 或 instrumentation gap

boundary_coverage_insufficient
  缺少可比较 tensor、checkpoint 或 dispatch 归属
```

这些状态属于同一个端到端集成测试的自动诊断流程。

## Probe

单模型 forward 的 probe 不够，需要支持端到端 trace source：

```python
PyTorchProbe(
    kind="trace",
    source="audio_head.logits",
    selector="cond",
    normalize="float32_contiguous",
)
```

建议 probe 语义：

```text
module_output
  单个 PyTorch/HF module forward hook，适合 Qwen3 单模型

module_input
  单个 module 的输入 hook

manual
  从 forward/generate 返回对象取值

derived
  由其他 artifact 计算

trace
  从端到端 reference trace 按当前 RunScope 取值
```

示例：

```python
audio_head_logits = activation_tensor(
    "stage0.audio_head.logits",
    dtype="float32",
    shape=(batch, codebooks, target_steps, codebook_vocab),
    pytorch_probe=PyTorchProbe(
        kind="trace",
        source="stage0.audio_head.logits",
        selector="row",
    ),
    compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
)
```

`selector="row"` 表示 reference capture 用当前 scope 的 `row=cond/uncond` 取对应 trace。

## Compare Policy

不同 artifact 需要不同策略：

```python
ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4)
ComparePolicy(kind="token")
ComparePolicy(kind="token_sequence")
ComparePolicy(kind="waveform", rtol=1e-4, atol=1e-4, max_abs=1e-3)
```

Token 边界必须是 first-class tensor：

```python
tokens_before = activation_tensor(
    "tokens.before",
    dtype="int32",
    shape=(codebooks, target_steps),
    role=TensorRole.TOKEN,
    pytorch_probe=trace_probe("tokens.before"),
    compare=ComparePolicy(kind="token_sequence"),
)
```

如果只比较最终 wav，定位信息太晚；必须在每一步 token 更新前后都能对拍。

## Execution 不拼名字

Execution source 只拿 tree 对象，不构造名字：

```python
with ctx.scope("generate", step=step):
    step_tensors = tensors.steps[step]
    run_stage0_audio_embedding(ctx, tensors=step_tensors.stage0, weights=weights.stage0)
```

不要这样：

```python
name = f"generate.step_{step:03d}.stage0.audio_embedding.output"
workspace.tensor(name)
```

名字属于 tree factory，scope 属于 execution context。

## 多模型权重 Tree

OmniVoice 权重不是单一 `weights`：

```python
@dataclass(frozen=True, slots=True)
class OmniVoiceWeights:
    stage0: OmniVoiceStage0Weights
    llm: Qwen3Weights
    stage1: OmniVoiceStage1Weights
```

checkpoint source key 可以不同，但 logical key 必须稳定：

```text
weights.stage0.audio_embeddings
weights.llm.layer.03.self_attn.q_proj
weights.stage1.decoder.block3.deconv.weight
```

Qwen3 单模型仍然可以使用：

```text
weights.layer.03.self_attn.q_proj
```

但 OmniVoice 组合模型里应该带 `weights.llm` 前缀，避免和 stage0/stage1 混淆。

## 通用收集

框架需要能递归收集 dataclass tree：

```python
collect_logical_tensors(tensors.run)
collect_logical_tensors(weights)
```

但 reference required probes 不能只看“所有带 probe 的 tensor”。端到端 run 中应该按
boundary schema 和实际执行记录收集：

```python
required = debug_schema.required_reference_keys(case)
candidate_boundaries = ctx.executed_boundaries()
```

第一版可以在 run 前根据 tree + case 展开全部 boundary scope；更稳的实现是在 locate mode
记录 candidate boundary timeline，再用 boundary/tensor keys 反推 required reference。
这仍然是一个端到端 run，不是分层测试。

## 失败报告

`LogicalTensor` 设计必须服务于诊断，而不是只服务于 storage planning。Mismatch report 至少包含：

```text
case fingerprint
scope key
base tensor name
writer shader
dispatch index
submodel / phase
step / row / layer
compare policy
reference artifact path
candidate artifact path
```

如果缺少 scope，OmniVoice 第 0 步和第 7 步的同名 tensor 会互相覆盖，debug 结果不可信。

## 对 Qwen3 的影响

Qwen3 prefill/decode 是单模型，所以可以用空 scope 或简单 scope：

```text
qwen3.prefill + output.logits
qwen3.decode/step=12 + output.next_token_id
```

这不会破坏现有裸调用体验，只是让同一套 DebugContext 能覆盖 OmniVoice 这种多模型端到端 run。
