# Contract Validation

Contracts are the checks that keep Python execution, Vulkan shaders, logical
tensors, and PyTorch reference aligned.

There are two contract layers:

1. shader contract validation: `LogicalTensor` objects must match the shader
   variant's declared inputs, outputs, bindings, shapes, dtypes, layouts, and
   dispatch requirements;
2. reference contract validation: `LogicalTensor` declarations and Vulkan
   artifacts must match the corresponding PyTorch tensor semantics.

Both layers are required. Shader validation can prove a dispatch is well formed;
it cannot prove the shader computes the PyTorch value.

## Shader Contract

Every shader variant should declare a contract:

```python
ShaderContract(
    name="linear_bf16_raw",
    inputs={
        "x": TensorContract(dtype="bf16", shape=("B", "S", "K"), layout="row_major"),
        "weight": TensorContract(dtype="bf16", shape=("N", "K"), layout="row_major"),
    },
    outputs={
        "output": TensorContract(dtype="bf16", shape=("B", "S", "N"), layout="row_major"),
    },
    bindings=[
        StorageBuffer("x", binding=0, readonly=True),
        StorageBuffer("weight", binding=1, readonly=True),
        StorageBuffer("output", binding=2, writeonly=True),
        Uniform("sizes", binding=3),
    ],
    dispatch=(ceil_div("N", 16), "B*S", 1),
)
```

The contract is not optional documentation. It is runtime validation input.

## Dispatch Validation

Before dispatching a shader, validate:

1. all required input tensors are provided;
2. all required output tensors are provided;
3. no unknown tensor fields are passed unless explicitly allowed;
4. dtype matches;
5. rank and shape match;
6. layout matches;
7. residency permits the shader access;
8. descriptor range is large enough;
9. output storage is writable;
10. read-only tensors are not bound to write-only fields;
11. push constants and uniforms match declared type and size;
12. dispatch geometry resolves to positive integer workgroups.

Failure should report the shader field and logical tensor name:

```text
linear_bf16_raw.weight expected dtype bf16, got f32
  tensor: weights.layer.03.self_attn.q_proj
```

## Shape Symbols

Contracts may use symbols:

```text
x      shape (B, S, K)
weight shape (N, K)
output shape (B, S, N)
```

Validation resolves symbols from provided tensors:

```text
B = x.shape[0] = output.shape[0]
S = x.shape[1] = output.shape[1]
K = x.shape[2] = weight.shape[1]
N = weight.shape[0] = output.shape[2]
```

If one symbol resolves to conflicting values, validation fails.

Do not let GLSL infer a different shape from push constants than the contract
validated. Shape metadata sent to the shader must come from the same resolved
symbols.

## Layout Contract

Shape equality is not enough. Layout must match too.

Examples:

```text
row_major
strided
qkv_packed_rows
kv_cache_paged
linear_weight_tiled
q4_k_words
```

If a shader expects packed weights, the input tensor must declare that packed
layout. Do not silently reinterpret a checkpoint weight as packed.

A physical descriptor range can be wider than the logical view, but the logical
layout still has to match what the shader reads.

## Binding Contract

The Python contract and GLSL bindings must match exactly.

If the contract says:

```text
binding 0: output
binding 1: input_ids
binding 2: weight
binding 3: sizes uniform
```

The GLSL must use the same binding numbers:

```glsl
layout(set = 0, binding = 0) writeonly buffer OutputBuffer { ... };
layout(set = 0, binding = 1) readonly buffer InputIdsBuffer { ... };
layout(set = 0, binding = 2) readonly buffer WeightBuffer { ... };
layout(set = 0, binding = 3) uniform Sizes { ... };
```

This should be checked by code review and, when possible, generated manifest
checks.

## Read/Write Contract

Each shader field has an access mode:

```text
read
write
read_write
```

The dispatcher records logical edges from this contract:

```text
reads:
  x -> decode.layer.03.input_norm
  weight -> weights.layer.03.self_attn.q_proj
writes:
  output -> decode.layer.03.q_proj
```

These edges drive:

1. liveness;
2. replay fingerprints;
3. debug drilldown;
4. first-writer reports;
5. accidental overwrite detection.

If a shader updates persistent state in place, mark the tensor as `read_write`.

## PyTorch Reference Contract

Every comparable `LogicalTensor` should declare how it corresponds to PyTorch:

```python
LogicalTensorDecl(
    name="decode.layer.03.input_norm",
    spec=TensorSpec(dtype="bf16", shape=("B", "S", "H")),
    ref=HookRef(lambda model: model.layers[3].input_layernorm),
    compare=ComparePolicy(kind="tensor", rtol=1e-2, atol=1e-2),
)
```

For Vulkan-only values:

```python
LogicalTensorDecl(
    name="decode.layer.03.qkv_packed",
    spec=TensorSpec(dtype="bf16", shape=("B", "S", "QKV")),
    ref=DerivedRef(inputs=(q_proj, k_proj, v_proj), fn=pack_qkv_ref),
)
```

The reference contract validates:

1. PyTorch tensor exists;
2. PyTorch shape maps to logical shape;
3. PyTorch dtype maps to logical dtype or declared compare dtype;
4. any transpose, slice, pack, or cast is explicit;
5. comparison tolerance is declared.

## Tensor Matching Against PyTorch

When comparing a Vulkan `LogicalTensor` to PyTorch:

1. read back the Vulkan tensor using its logical shape and dtype;
2. obtain the PyTorch value from the reference rule;
3. normalize dtype only if declared;
4. compare with the tensor's `ComparePolicy`;
5. report mismatch under `LogicalTensor.name`.

Do not compare an implicit flattened byte buffer to a PyTorch tensor and call it
done. Shape, dtype, and layout are part of the contract.

## Weight Contract

A checkpoint weight declaration should state:

1. logical name;
2. checkpoint key;
3. source dtype and shape if known;
4. runtime dtype and shape;
5. runtime layout;
6. shader variants allowed to consume it.

Raw checkpoint weight:

```text
weights.layer.03.self_attn.q_proj
  source: model.layers.3.self_attn.q_proj.weight
  runtime layout: row_major
  consumed by: linear_bf16_raw
```

Do not introduce a second converted weight:

```text
weights_packed.layer.03.self_attn.q_proj.linear_bf16_tiled  # forbidden
```

If a shader cannot consume the raw declared layout, do not use that shader in
the safetensors port.

## State Contract

Persistent state must declare:

1. lifetime;
2. shape and max capacity;
3. update shader;
4. read shader;
5. whether updates are in-place;
6. PyTorch reference rule if comparable.

KV cache is not just a buffer. It has logical identity:

```text
cache.layer.03.key
cache.layer.03.value
```

Shaders that update cache must record write or read-write edges to those names.

## Contract Timing

Use different validation levels:

```text
schema validation:
  names, duplicate declarations, checkpoint sources, reference rules

dispatch validation:
  shader contract vs bound LogicalTensor objects

compare validation:
  Vulkan artifact vs PyTorch artifact

replay validation:
  recorded contract and storage fingerprint vs current tensors
```

Eager debug should run full validation. Hot replay may run a cheaper fingerprint
check after a session has been validated.

## Error Reporting

Errors should be specific:

```text
contract error in sdpa_decode_kv_packed.k_cache
  expected shape: (B, KVH, MAX_S, D)
  actual shape:   (B, QH, MAX_S, D)
  tensor: cache.layer.03.key
```

For PyTorch mismatch:

```text
reference mismatch at decode.layer.03.attention.output
  compare: rtol=0.01 atol=0.01
  max_abs: 0.084
  writer: sdpa_decode_kv_packed
```

Start with semantic identity. Add physical buffer details only after that.

## Minimum Acceptance

A model family should not be considered ported until:

1. every shader dispatch validates its contract in eager mode;
2. every shader contract can emit logical reads and writes;
3. every comparable tensor has a PyTorch reference rule;
4. every compare boundary has a tolerance policy;
5. every weight-consuming shader matches the raw safetensors dtype, shape, and layout;
6. replay validates regime and storage fingerprint before hot use;
7. contract failures and numeric mismatches report `LogicalTensor.name`.
