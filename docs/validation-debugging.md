# 验证与调试指南

`torch2vk` 的验证入口应该像一次普通模型调用。Vulkan shader 仍然是裸函数调用，但每次
调用都同时把 `ctx` 和 `pytorch_model` 传进去：

```python
run_qwen3_prefill(
    ctx,
    pytorch_model,
    tensors=prefill_tensors,
    weights=weights,
    spec=spec,
)
```

这里的 `pytorch_model` 是 PyTorch/HuggingFace eager reference model。shader 函数本身
不直接读取它，只把它交给 `ctx.dispatch()`。`ctx` 在第一次 dispatch 时用这个
`pytorch_model` 做一次 forward capture，按 `LogicalTensor.pytorch_probe` 捕获 reference
artifacts 并落盘。之后 Vulkan shader 按 Python 调用顺序 eager 执行、读回和对拍。

不要求 PyTorch 和 Vulkan 逐 shader 同步。shader 调用后，`ctx` 会从 dispatch writes
里自然收集需要对拍的 `LogicalTensor`；对齐粒度是 `LogicalTensor.name`。

## 核心原则

所有可对拍值都以 `LogicalTensor.name` 为唯一 key。

好的 key：

```text
decode.layer.03.input_norm
decode.layer.03.self_attn.q_proj
decode.layer.03.self_attn.q_rope
decode.layer.03.attention.output
decode.layer.03.mlp.output
output.logits
output.next_token_id
```

不接受这些 key：

```text
torch_hook_17
candidate_qproj
workspace.hidden_a
slot4
buffer_12_offset_4096
```

物理 storage、buffer offset、hook id 都只是实现细节，不能成为 reference 和 candidate
的对齐依据。

## LogicalTensor Probe

`LogicalTensor` 集中声明它如何从 PyTorch eager 获得 reference。

```python
LogicalTensor(
    name="decode.layer.03.self_attn.q_proj",
    spec=TensorSpec(dtype="float32", shape=("B", "S", "Q")),
    pytorch_probe=PyTorchProbe(
        kind="module_output",
        target="model.layers.3.self_attn.q_proj",
        normalize="float32_contiguous",
    ),
    compare=ComparePolicy(kind="tensor", rtol=1e-2, atol=1e-2),
)
```

Probe 类型：

```python
PyTorchProbe(kind="module_output", target="model.layers.3.self_attn.q_proj")
PyTorchProbe(kind="module_input", target="model.layers.3.self_attn.q_proj", index=0)
PyTorchProbe(kind="manual", source="next_token_id")
PyTorchProbe(kind="derived", inputs=(...), transform="apply_rope_ref")
```

Probe 只负责 reference artifact 映射，不参与 Vulkan 执行顺序。

## 自动收集对拍 Tensor

默认规则很简单：shader dispatch 写出的 output tensor 如果声明了 `pytorch_probe` 和
`compare`，就自动进入对拍。调用完 shader 后，哪些 tensor 需要对拍已经由
`record.writes` 自然产生。

```python
LINEAR_BF16_F32(
    ctx,
    pytorch_model,
    x=layer.input_norm,
    weight=layer_weights.self_attn.q_proj,
    output=layer.q_proj,
)
```

这个调用写出 `layer.q_proj`。如果 `layer.q_proj.pytorch_probe` 存在，`ctx.dispatch()`
会自动 readback `layer.q_proj`，并和 `reference[layer.q_proj.name]` 比较。

## 裸调用 Shader

Vulkan shader 在 execution source 里应该像普通函数一样调用。`pytorch_model` 每次都显式
出现在调用点，让 reference side 和 candidate side 都在场。

```python
def run_qwen3_prefill(ctx, pytorch_model, *, tensors, weights, spec):
    EMBEDDING_LOOKUP_BF16_F32(
        ctx,
        pytorch_model,
        input_ids=tensors.input_ids,
        weight=weights.embed_tokens,
        output=tensors.hidden,
    )

    for layer, layer_weights in zip(tensors.layers, weights.layers, strict=True):
        RMS_NORM_F32(
            ctx,
            pytorch_model,
            x=layer.input,
            weight=layer_weights.input_layernorm,
            output=layer.input_norm,
        )
        LINEAR_BF16_F32(
            ctx,
            pytorch_model,
            x=layer.input_norm,
            weight=layer_weights.self_attn.q_proj,
            output=layer.q_proj,
        )
```

每个 shader callable 只做一件事：把 variant、`pytorch_model` 和 logical tensors 交给
`ctx.dispatch()`。

```python
class ShaderFunction:
    variant: ShaderVariant

    def __call__(self, ctx, pytorch_model, **tensors):
        ctx.dispatch(self.variant, pytorch_model, tensors)
```

`ctx.dispatch()` 是 eager debug 的核心：

```python
class DebugContext:
    def dispatch(self, variant, pytorch_model, tensors):
        self.ensure_reference(pytorch_model)

        record = self.record(variant, tensors)
        self.vulkan.run(record)
        self.records.append(record)

        for tensor in self.comparable_writes(record):
            self.candidate[tensor.name] = self.readback(tensor)
            compare_tensor(
                tensor,
                reference=self.reference,
                candidate=self.candidate,
                dispatch_records=self.records,
            ).raise_for_mismatch()
```

这保持了“shader 当函数调用”的体验，同时每次调用都立即执行、记录、读回和对拍。
`DispatchRecord` 是 debug eager 执行的副产物，可用于诊断和 replay。

## 单入口流程

调用方创建一个 Vulkan/debug `ctx`，然后直接调用模型 execution function。`pytorch_model`
不是 `ctx` 的构造参数，而是 shader 调用链上的显式参数。

```python
def validate_model_prefill(
    *,
    pytorch_model,
    ctx,
    tensors,
    weights,
    spec,
):
    run_qwen3_prefill(
        ctx,
        pytorch_model,
        tensors=tensors,
        weights=weights,
        spec=spec,
    )

    return ctx.records
```

`ctx` 第一次 dispatch 前自动准备 reference。required probes 也不需要外部传入：
`ctx` 可以从 tensor tree 中收集所有带 `pytorch_probe` 的 tensor，或者只收集本次执行中
实际写出的 comparable tensors。对调用方来说，PyTorch reference capture 和 Vulkan eager
execution 是同一次裸调用的一部分。

```python
class DebugContext:
    def ensure_reference(self, pytorch_model):
        if self.reference_ready:
            return

        required = self.required_probe_names()
        fingerprint = make_run_fingerprint(
            pytorch_model,
            self.tensors,
            self.inputs,
            required,
        )
        cached = self.cache.load(fingerprint)
        if cached.complete(required):
            self.reference = cached
        else:
            self.reference = PyTorchForwardCapture(
                pytorch_model,
                self.tensors,
                self.inputs,
            ).run(required)
            self.cache.store(fingerprint, self.reference)

        self.reference_ready = True
```

第一版 PyTorch capture 可以就是一次完整 forward：

```python
class PyTorchForwardCapture:
    def run(self, names):
        artifacts = {}
        handles = install_hooks(self.pytorch_model, self.tensors, names, artifacts)
        try:
            with torch.no_grad():
                output = self.pytorch_model(**self.inputs)
            collect_manual_probes(output, self.tensors, artifacts)
            run_derived_probes(self.tensors, artifacts)
        finally:
            for handle in handles:
                handle.remove()
        return artifacts
```

这消除了测试里额外维护一套 PyTorch eager loop。PyTorch 只是同一次 shader 裸调用入口里的
reference artifact provider。

## Artifact Cache

PyTorch reference 是语义 oracle，但不应该每次 Vulkan shader 调试都重新跑。

cache key 必须覆盖所有会影响 PyTorch reference 的输入：

```text
model_family
checkpoint identity
model config / spec
prompt input ids
position ids
attention mask
generation mode: prefill / decode
max_seq_len
probe schema version
probe names
transform implementation version
torch dtype / reference dtype policy
```

落盘 artifact 仍然以 `LogicalTensor.name` 为 key：

```text
artifacts/
  qwen3_prefill_<fingerprint>/
    manifest.json
    decode.embedding.pt
    decode.layer.00.input_norm.pt
    decode.layer.00.self_attn.q_proj.pt
    ...
    output.logits.pt
    output.next_token_id.pt
```

读取 cache 时必须验证：

1. artifact 存在；
2. artifact key 等于 `LogicalTensor.name`；
3. dtype 和 normalize policy 一致；
4. shape 和 `LogicalTensor` 一致；
5. manifest fingerprint 等于当前 run fingerprint。

cache 只缓存 PyTorch reference artifacts，不缓存 Vulkan candidate artifacts。Vulkan
candidate 是当前 shader 代码的被测对象，调试时必须重新执行并重新 readback。

## Mismatch Drilldown

每个 dispatch 都记录 logical reads 和 writes：

```text
dispatch 84 flash_attn_f32_f16
  reads  q=decode.layer.03.self_attn.q_rope
  reads  key_cache=decode.layer.03.key_cache
  reads  value_cache=decode.layer.03.value_cache
  writes output=decode.layer.03.attention.output
```

失败报告先给语义信息：

```text
first mismatch: decode.layer.03.attention.output
writer shader: flash_attn_f32_f16
writer dispatch: 84
matching inputs: q, key_cache, value_cache
divergent output: output
max_abs: 0.137
```

物理信息只能作为附加诊断：

```text
storage: allocation=qwen3-prefill offset=1048576 nbytes=8192
descriptor: binding=2
```

## 精度策略

每个可比较 tensor 都必须有明确 `ComparePolicy`。

```python
ComparePolicy(kind="tensor", rtol=1e-2, atol=1e-2)
ComparePolicy(kind="tensor", rtol=0.0, atol=0.5)
ComparePolicy(kind="token")
```

策略应当跟 tensor 语义绑定，而不是跟测试绑定。

## Replay Validation

Replay 只能在 eager debug 正确之后启用。

Replay 不应该：

1. 跑 PyTorch capture；
2. 做 debug readback；
3. 重新推断 liveness；
4. 重新解释模型语义；
5. 改变 `LogicalTensor` 声明。

Replay 的验证方式是：同一输入下，replay 输出必须等于 Vulkan eager debug 的输出，并继续
满足 PyTorch reference 的最终边界。
