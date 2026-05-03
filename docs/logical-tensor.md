# LogicalTensor Tree 设计

`LogicalTensor` 是 `torch2vk` 的模型可见 tensor 句柄。最终形态应该是一棵集中声明的
LogicalTensor tree：它描述模型里所有值得被命名、存储、读写、对拍、缓存和诊断的 tensor。

执行顺序不属于 LogicalTensor tree。执行顺序来自 Python execution source 里 shader
像函数一样被 eager 调用的顺序。

## 目标目录

每个模型 family 应该有一个 `tensors/` 子目录，集中放 logical tensor tree。

```text
src/torch2vk/models/qwen3_safetensor/
  execution.py
  tensors/
    __init__.py
    prefill.py
    decode.py
    weights.py
    probes.py
```

职责划分：

```text
tensors/prefill.py
  prefill dataclass tree 和 prefill tensor factory

tensors/decode.py
  decode dataclass tree 和 decode tensor factory

tensors/weights.py
  structured weight tree 和 checkpoint source

tensors/probes.py
  PyTorch probe helper 和路径约定

execution.py
  使用 tensor tree，按 eager 顺序调用 Vulkan shader
```

## Core API

理想的 `LogicalTensor` 形态：

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

`name` 是唯一语义 key。`storage` 是物理 backing。两者必须分离。

同一个 `LogicalTensor.name` 会被这些系统使用：

1. Vulkan shader dispatch 的 reads/writes；
2. PyTorch probe artifact；
3. Vulkan readback artifact；
4. artifact cache；
5. liveness planner；
6. replay fingerprint；
7. mismatch report。

不要再引入第二套模型可见 tensor id，例如 `TensorSlot`、`WorkspaceTensor`、
`candidate_qproj`。

## Tensor Tree

模型应该暴露结构化 tree，而不是散落的字符串表。

```python
@dataclass(frozen=True, slots=True)
class Qwen3LayerTensors:
    input: LogicalTensor
    input_norm: LogicalTensor
    q_proj: LogicalTensor
    k_proj: LogicalTensor
    v_proj: LogicalTensor
    q_rope: LogicalTensor
    key_cache: LogicalTensor
    value_cache: LogicalTensor
    attention_context: LogicalTensor
    attention_o_proj: LogicalTensor
    attention_residual: LogicalTensor
    post_attention_norm: LogicalTensor
    mlp_gate: LogicalTensor
    mlp_up: LogicalTensor
    mlp_gated: LogicalTensor
    mlp_down: LogicalTensor
    output: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3PrefillTensors:
    input_ids: LogicalTensor
    position_ids: LogicalTensor
    attention_mask: LogicalTensor
    hidden: LogicalTensor
    layers: tuple[Qwen3LayerTensors, ...]
    final_norm: LogicalTensor
    logits: LogicalTensor
    next_token_id: LogicalTensor
```

访问 tensor 时应该走 tree：

```python
LINEAR_BF16_F32(
    ctx,
    pytorch_model,
    x=layer.input_norm,
    weight=weights.layers[3].self_attn.q_proj,
    output=layer.q_proj,
)
```

不要在 execution 里拼 tensor 名：

```python
name = f"decode.layer.{i:02d}.self_attn.q_proj"
workspace.tensor(name)
```

名字属于 tensor tree 构造阶段，execution 只使用已经声明好的对象。

## Prefill 和 Decode

Prefill 和 decode 必须有不同的 tree factory。

```python
def qwen3_prefill_tensors(
    *,
    batch: int,
    steps: int,
    spec: Qwen3Spec,
    max_seq_len: int,
) -> Qwen3PrefillTensors: ...


def qwen3_decode_tensors(
    *,
    batch: int,
    spec: Qwen3Spec,
    max_seq_len: int,
    step_index: int,
) -> Qwen3DecodeTensors: ...
```

原因：

```text
prefill:
  steps = prompt length
  attention mask 是 prompt 内的 causal mask
  KV cache 批量写入多行

decode:
  steps = 1
  attention 读取历史 cache
  KV cache 追加当前 token
  cache position / row index / state checkpoint 是 step 级状态
```

可以共享 `Qwen3LayerTensors` 的一部分字段，但不要用一个 factory 同时隐式处理 prefill
和 decode 的 shape 语义。

## PyTorch Probe

`LogicalTensor` 应该集中声明它如何和 PyTorch eager 对齐。

```python
layer.q_proj = activation_tensor(
    "decode.layer.03.self_attn.q_proj",
    dtype="float32",
    shape=(batch, steps, spec.q_proj_out_features),
    pytorch_probe=module_output(
        "model.layers.3.self_attn.q_proj",
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

Probe 的职责：

1. 从 PyTorch eager 中捕获 reference tensor；
2. 归一化 dtype、shape、contiguity；
3. 以 `LogicalTensor.name` 写入 artifact cache；
4. 不参与 Vulkan 执行顺序。

PyTorch probe 不是 execution node。它只是 `LogicalTensor` 到 PyTorch eager artifact 的映射。

## 自动对拍

默认情况下，不需要单独列出“要对拍哪些 tensor”。shader 调用写出的 output
`LogicalTensor` 如果声明了 `pytorch_probe` 和 `compare`，就自然是可对拍 tensor。

Debug loop 的顺序来自 shader 函数调用产生的 `DispatchRecord`。最小形态是直接比较
当前 dispatch 写出的 comparable tensors：

```python
for record in dispatch_records:
    vulkan_runner.run_one(record)
    for tensor in comparable_writes(record):
        candidate[tensor.name] = readback(tensor)
        compare(tensor, reference, candidate)
```

## Factory API

推荐给模型 tensor tree 提供小而明确的 factory。

```python
def A(
    name: str,
    *,
    dtype: str,
    shape: tuple[int, ...],
    probe: PyTorchProbe | None = None,
    compare: ComparePolicy | None = None,
    role: TensorRole = TensorRole.ACTIVATION,
    memory: MemoryPolicy = MemoryPolicy.FRAME_WORKSPACE,
) -> LogicalTensor: ...


def I(name: str, *, dtype: str, shape: tuple[int, ...]) -> LogicalTensor: ...
def O(name: str, *, dtype: str, shape: tuple[int, ...]) -> LogicalTensor: ...
def K(name: str, *, dtype: str, shape: tuple[int, ...]) -> LogicalTensor: ...
```

示例：

```python
def _layer_tensors(
    *,
    layer_index: int,
    batch: int,
    steps: int,
    spec: Qwen3Spec,
    layer_input: LogicalTensor,
) -> Qwen3LayerTensors:
    prefix = f"decode.layer.{layer_index:02d}"
    torch_prefix = f"model.layers.{layer_index}"

    return Qwen3LayerTensors(
        input=layer_input,
        input_norm=A(
            f"{prefix}.input_norm",
            dtype="float32",
            shape=(batch, steps, spec.hidden_size),
            probe=module_output(f"{torch_prefix}.input_layernorm"),
        ),
        q_proj=A(
            f"{prefix}.self_attn.q_proj",
            dtype="float32",
            shape=(batch, steps, spec.q_proj_out_features),
            probe=module_output(f"{torch_prefix}.self_attn.q_proj"),
        ),
        ...
    )
```

## Views

Views are allowed, but they must not create hidden semantics.

同一个值的 ABI 视图可以保留同一个 name：

```python
q_heads = q_proj.view_as(
    q_proj.name,
    spec=TensorSpec(
        dtype="float32",
        shape=(batch, steps, spec.num_attention_heads, spec.head_dim),
    ),
)
```

新的模型可见值必须使用新的 name：

```python
qkv_packed = packed.view_as(
    "decode.layer.03.self_attn.qkv_packed",
    spec=TensorSpec(dtype="float32", shape=(batch, steps, qkv_width)),
)
```

`view_as()` 必须保留 `pytorch_probe`、`compare`、role、memory，除非调用方显式覆盖。

## Binding

LogicalTensor declaration 和 bound tensor 是同一个 identity 的两个阶段。

```text
declaration:
  name, spec, role, layout, probe, compare

bound tensor:
  declaration + BufferSlice
```

绑定 storage 不能改变 `name`、`spec`、`pytorch_probe`、`compare`。

```python
unbound = qwen3_collect_logical_tensors(tensors)
plan = plan_storage(unbound, allocation_id="qwen3-prefill")
bound = bind_storage(unbound, plan)
```

允许 planner 让生命周期不重叠的 logical tensors 共享物理 slot。共享 storage 不改变
logical identity。

## Weights

权重也应该在 tree 中有结构化访问方式。

```python
@dataclass(frozen=True, slots=True)
class Qwen3LayerWeights:
    input_layernorm: LogicalTensor
    q_proj: LogicalTensor
    k_proj: LogicalTensor
    v_proj: LogicalTensor
    o_proj: LogicalTensor
    post_attention_layernorm: LogicalTensor
    gate_proj: LogicalTensor
    up_proj: LogicalTensor
    down_proj: LogicalTensor
```

权重 `LogicalTensor` 必须声明 checkpoint source：

```python
W(
    "weights.layer.03.self_attn.q_proj",
    safetensor_key="model.layers.3.self_attn.q_proj.weight",
    dtype="bfloat16",
    shape=(spec.q_proj_out_features, spec.hidden_size),
)
```

execution 不应该知道 safetensors key。

## Diagnostics

错误报告必须先给 logical identity：

```text
first mismatch: decode.layer.03.attention.output
writer shader: flash_attn_f32_f16
writer dispatch: 84
matching inputs: q, key_cache, value_cache
divergent output: output
```

物理信息只能作为附加内容：

```text
storage: allocation=qwen3-prefill offset=1048576 nbytes=8192
descriptor: binding=2
```

## 禁止事项

不要让 `tensors/` 子目录承担执行职责。

禁止：

```python
tensors.prefill.run()
tensors.layer[3].q_proj.execute()
tensors.schedule
```

允许：

```python
run_qwen3_prefill(ctx, pytorch_model, spec=spec, tensors=tensors, weights=weights)
run_qwen3_decode_step(ctx, pytorch_model, spec=spec, tensors=tensors, weights=weights)
```

`LogicalTensor` tree 是模型语义和对拍映射。execution source 才是执行语义。
