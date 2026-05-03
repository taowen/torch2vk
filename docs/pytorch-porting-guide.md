# PyTorch Model Porting Guide

This guide describes how to port a PyTorch model into `torch2vk`.

The target shape is:

```text
One model family.
One Python execution source.
One model-visible tensor handle.
Explicit shader variants.
Explicit weight/state semantics.
Continuous comparison against PyTorch reference.
```

`torch2vk` is not a PyTorch exporter. Do not start by capturing a graph and
lowering it. Start by reading the PyTorch model and writing the model family's
execution path directly in Python.

## Porting Order

Use this order for every new model family:

1. Read the PyTorch modules and checkpoint layout.
2. Define model constants in `spec.py`.
3. Declare all model-visible tensors in a model schema.
4. Declare weights with their checkpoint source and runtime layout.
5. Implement the reference Vulkan execution path in Python.
6. Write the first simple shader variants, even if they are not fast.
7. Compare outputs and selected intermediate tensors against PyTorch eager.
8. Add optimized shader variants one at a time.
9. Add weight conversion only when an optimized variant requires a new layout.
10. Add replay or command recording only after eager execution is correct.

Correctness comes before replay, packing, quantization, and fusion.

## Recommended Model Directory

A model family should be organized around source-visible execution:

```text
models/<family>/
  spec.py
  schema.py
  weights.py
  state.py
  execution.py
  reference.py
  validation.py
  shaders/
    rms_norm_bf16.py
    linear_bf16_raw.py
    linear_bf16_tiled.py
    rotary_embedding.py
```

The names are not mandatory, but the responsibility split is.

`spec.py` contains static model parameters and supported runtime regimes.
`schema.py` declares logical tensors and boundaries. `weights.py` materializes
checkpoint tensors into `LogicalTensor` objects. `state.py` owns persistent
state such as KV cache. `execution.py` is the readable Python shader call
sequence. `reference.py` collects PyTorch eager reference values.

Do not put model semantics in a generic runtime scheduler.

## Execution Source

The execution file should look like model code, not like an IR interpreter:

```python
def run_decode_layer(ctx, layer, state, x):
    norm = layer.tensors.input_norm
    q = layer.tensors.q_proj
    k = layer.tensors.k_proj
    v = layer.tensors.v_proj
    out = layer.tensors.output

    rms_norm_bf16(ctx, x=x, weight=layer.weights.input_norm, output=norm)
    linear_bf16_raw(ctx, x=norm, weight=layer.weights.q_proj, output=q)
    linear_bf16_raw(ctx, x=norm, weight=layer.weights.k_proj, output=k)
    linear_bf16_raw(ctx, x=norm, weight=layer.weights.v_proj, output=v)
    rotary_embedding(ctx, q=q, k=k, positions=state.positions, q_out=q, k_out=k)
    sdpa_decode(ctx, q=q, k=k, v=v, cache=state.kv_cache, output=out)
    return out
```

This code is the model execution truth. It should be readable by a developer
who knows the PyTorch module.

Avoid:

```python
for node in exported_graph.nodes:
    runtime.execute(node)
```

Avoid dynamic semantic names in execution:

```python
name = f"decode.layer.{i}.q_proj"
workspace.tensor(name)
```

Names belong in the schema construction step. Execution should receive already
declared objects.

## LogicalTensor Is The Model Handle

Every meaningful runtime value should be represented by one `LogicalTensor`.

This includes:

1. inputs;
2. outputs;
3. checkpoint weights;
4. activations worth comparing;
5. optimized-only intermediate tensors;
6. persistent state;
7. temporary scratch whose liveness matters;
8. host-visible input and output ports.

The model execution layer should pass `LogicalTensor` objects to shaders.

Do not expose a second model-visible tensor handle such as `TensorSlot`,
`DeviceTensor`, `TensorView`, or raw `BufferSlice`. Those may exist inside the
allocator or Vulkan backend, but they are physical details.

## Physical Storage

A `LogicalTensor` may be unbound during schema construction and bound after
allocation or weight materialization.

The important split is:

```text
LogicalTensorDecl or schema object
  semantic identity, spec, role, reference rule, checkpoint source

LogicalTensor
  semantic identity plus optional physical storage

BufferSlice / allocation
  physical memory only
```

Do not introduce a named `TensorSlot` tree that mirrors the model tensor tree.
That recreates a second namespace. If the allocator needs slots, they should be
anonymous physical slots chosen from liveness:

```text
decode.layer.03.q_proj live 17..18 -> physical slot 4
decode.layer.04.q_proj live 25..26 -> physical slot 4
```

The model should never call `slot.activation("decode.layer.03.q_proj")`.
Instead, the schema should already contain `decode.layer.03.q_proj`, and the
planner should bind it to storage.

## Weight Loading

Weights are declared with logical names and checkpoint sources:

```python
q_proj = W(
    "weights.layer.03.self_attn.q_proj",
    source="model.layers.3.self_attn.q_proj.weight",
    spec=TensorSpec(dtype="bf16", shape=(hidden, hidden)),
    layout=Layout.row_major(),
)
```

The materializer should:

1. open the checkpoint;
2. validate source shape and dtype;
3. apply explicit conversion only when declared;
4. allocate or upload storage;
5. return `LogicalTensor` weights keyed by logical name.

No execution file should know safetensors keys.

## Reference Execution

PyTorch eager is the reference interpreter. The reference path should produce
artifacts keyed by `LogicalTensor.name`.

For direct module outputs, use hooks. For tensors that exist only in Vulkan
execution, derive the reference from other declared tensors:

```python
qkv_packed = A(
    "decode.layer.03.qkv_packed",
    spec=...,
    ref=DerivedRef(inputs=(q_proj, k_proj, v_proj), fn=pack_qkv_ref),
)
```

Reference logic should not restate the Vulkan execution sequence. It should
describe how to obtain the comparable PyTorch value for a declared tensor.

## Optimized Variants

An optimized path should be another visible Python execution version, not the
hidden result of a pass:

```python
def run_decode_layer_reference(ctx, layer, state, x): ...
def run_decode_layer_packed_qkv(ctx, layer, state, x): ...
def run_decode_layer_fused_attention(ctx, layer, state, x): ...
```

Each optimized version must answer:

1. which tensors are shared with the reference path;
2. which tensors are new optimized-only intermediates;
3. which weights keep checkpoint layout;
4. which weights require conversion;
5. which comparison boundaries prove the rewrite.

## Acceptance Criteria

A port is not done until:

1. model execution passes only `LogicalTensor` objects to shaders;
2. execution code does not construct semantic names dynamically;
3. weights are materialized from schema declarations;
4. PyTorch reference artifacts and Vulkan artifacts share `LogicalTensor.name`;
5. selected intermediate tensors compare against PyTorch;
6. persistent state stays device-resident across steps;
7. optimized variants are explicit and separately comparable;
8. replay, if present, records already-correct shader calls and does not become a new semantic layer.

