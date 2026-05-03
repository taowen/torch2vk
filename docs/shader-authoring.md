# Shader Authoring Guide

Shaders are the core execution units in `torch2vk`.

A shader should represent one stable, named computation boundary. It should be
specific enough that a developer can understand its tensor semantics from its
name and contract.

## Shader Variant Shape

Each shader variant should have:

1. a stable variant name;
2. a shader family name;
3. a typed tensor contract;
4. explicit descriptor bindings;
5. explicit push constants or uniforms;
6. explicit dispatch geometry;
7. GLSL source or a checked-in generated source path;
8. a Python callable wrapper.

Example shape:

```python
LINEAR_BF16_RAW = ShaderVariant(
    name="linear_bf16_raw",
    family="linear_bf16",
    inputs={
        "x": TensorContract(dtype="bf16", shape=("B", "S", "K")),
        "weight": TensorContract(dtype="bf16", shape=("N", "K")),
    },
    outputs={
        "output": TensorContract(dtype="bf16", shape=("B", "S", "N")),
    },
    bindings=[
        StorageBuffer("x", binding=0, readonly=True),
        StorageBuffer("weight", binding=1, readonly=True),
        StorageBuffer("output", binding=2, writeonly=True),
        Uniform("sizes", binding=3),
    ],
    dispatch=(ceil_div("N", 16), "B*S", 1),
    source=...,
)
```

The exact Python API can change, but the information must exist.

## Naming

Use names that describe semantics and important storage assumptions:

```text
rms_norm_bf16
linear_bf16_raw
linear_bf16_tiled
linear_bf16_weight_packed
rotary_embedding
sdpa_decode_kv_cache_update
sdpa_prefill_blocked
argmax_last_logits_i32
```

Do not name shaders after implementation accidents:

```text
kernel1
fast_linear
new_attention
test_shader
```

## Contracts

A shader contract is the boundary between Python execution and Vulkan. It must
state:

1. which tensor fields are read;
2. which tensor fields are written;
3. dtype and shape requirements;
4. layout requirements;
5. descriptor binding order;
6. push constant format;
7. dispatch geometry.

The dispatcher should be able to record:

```text
shader linear_bf16_raw
  reads  x=decode.layer.03.input_norm
  reads  weight=weights.layer.03.self_attn.q_proj
  writes output=decode.layer.03.q_proj
```

This record is used for validation, liveness, replay, and mismatch drilldown.

## Python Wrapper

The wrapper should accept `LogicalTensor` objects, validate model-level
conditions, and call the variant:

```python
def linear_bf16_raw(ctx, *, x: LogicalTensor, weight: LogicalTensor, output: LogicalTensor) -> None:
    if weight.spec.dtype != "bf16":
        raise ValueError(f"linear_bf16_raw expects bf16 weight, got {weight.spec.dtype}")
    LINEAR_BF16_RAW(ctx, x=x, weight=weight, output=output)
```

The wrapper may choose between variants:

```python
def linear_bf16(ctx, *, x, weight, output, scratch=None):
    if x.shape[1] <= 8:
        return LINEAR_BF16_VEC(ctx, x=x, weight=weight, output=output)
    if scratch is None:
        return LINEAR_BF16_RAW(ctx, x=x, weight=weight, output=output)
    return LINEAR_BF16_TILED(ctx, x=x, weight=weight, scratch=scratch, output=output)
```

Variant selection is normal Python code. It should remain visible.

## GLSL Rules

Use explicit bindings:

```glsl
layout(set = 0, binding = 0) readonly buffer XBuffer { uint x[]; };
layout(set = 0, binding = 1) readonly buffer WBuffer { uint weight[]; };
layout(set = 0, binding = 2) writeonly buffer OBuffer { uint output[]; };
layout(set = 0, binding = 3) uniform Sizes { uvec4 sizes; };
```

Keep indexing formulas local and named:

```glsl
uint offset_bsh(uint b, uint s, uint h, uint S, uint H) {
    return (b * S + s) * H + h;
}
```

Guard all global IDs that can exceed logical shape:

```glsl
if (h >= H || s >= S || b >= B) {
    return;
}
```

For reductions, document the reduction axis and accumulation dtype in the
shader comment or contract.

## Push Constants And Uniforms

Use push constants for small per-dispatch values that are not naturally shape
metadata. Use uniform buffers for structured shape metadata when it simplifies
descriptor preparation or replay.

Do not hide semantic state in push constants. For example, current decode
position may be a scalar parameter, but the KV cache tensor and its logical
identity must still be explicit.

## Layout And Descriptor Range

A tensor layout is part of the shader contract. If a shader expects packed QKV
or tiled weights, say so in the contract and in the variant name.

Descriptor range can be larger than the logical view when the shader ABI needs
access to a wider backing buffer, such as a KV cache row view. Treat this as
physical ABI, not a new semantic tensor.

## Weight Conversion

Only introduce weight conversion when the shader truly cannot consume the
checkpoint layout.

Good:

```text
linear_bf16_raw consumes checkpoint weight directly.
linear_bf16_weight_packed consumes packed_weight produced by pack_linear_bf16().
```

Bad:

```text
All weights pass through conversion because the runtime always has a converter.
```

Converted weights need their own logical names:

```text
weights.layer.03.self_attn.q_proj
weights_packed.layer.03.self_attn.q_proj.linear_bf16_tiled
```

## Reference First, Optimize Second

For a new op, first write the most direct shader that matches PyTorch semantics.
Then add optimized variants.

Each optimized variant should have a comparison plan:

1. compare its output against the reference variant;
2. compare the same boundary against PyTorch eager;
3. document any intentional precision difference;
4. record whether it changes weight or state layout.

## Review Checklist

Before accepting a shader:

1. name describes semantics and layout assumptions;
2. contract lists every read and write tensor;
3. dtype, shape, and layout are validated before dispatch;
4. GLSL binding order matches contract binding order;
5. out-of-range global IDs are guarded;
6. output tensor is fully written or intentionally partially updated;
7. reference comparison exists for at least one realistic shape;
8. optimized variant does not silently change weight or state semantics.

