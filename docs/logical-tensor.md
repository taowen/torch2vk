# LogicalTensor Design

`LogicalTensor` is the model-visible tensor handle in `torch2vk`.

Its job is to keep semantic identity attached to the storage that shaders read
and write. It is not a compiler node, and it is not just a buffer view.

## Core Contract

Every meaningful model runtime value has one logical identity:

```python
@dataclass(frozen=True, slots=True)
class LogicalTensor:
    name: str
    spec: TensorSpec
    layout: TensorLayout
    role: TensorRole
    memory: MemoryPolicy
    storage: BufferSlice | None = None
    source: WeightSource | None = None
    ref: ReferenceRule | None = None
    compare: ComparePolicy | None = None
```

`name` is the semantic key. `storage` is physical backing. Keep those two
concepts separate.

The same `LogicalTensor.name` should be used by:

1. shader dispatch read/write records;
2. PyTorch reference artifacts;
3. Vulkan readback artifacts;
4. liveness planning;
5. replay storage fingerprints;
6. error reports.

## Roles

Use explicit roles so runtime decisions remain inspectable:

```text
input
output
weight
activation
scratch
kv_cache
recurrent_state
mask
logits
debug
```

Role is semantic. Residency is physical policy. Do not encode one through the
other.

## Memory Policy

A memory policy says where the tensor should live and how long it lives:

```text
device_local
host_visible_input
host_visible_output
persistent_state
frame_workspace
step_temporary
debug_readback
```

Persistent state must not alias frame workspace. Frame workspace may alias
other frame workspace tensors only when liveness proves their lifetimes do not
overlap.

## Declarations And Bound Tensors

It is useful to distinguish declaration from materialized tensor:

```text
LogicalTensor declaration:
  name, spec, role, layout, source/ref/compare

Bound LogicalTensor:
  declaration plus BufferSlice
```

This is a phase distinction, not two competing identities. The declaration and
bound tensor must share the same name.

Good:

```python
decl = A("decode.layer.03.q_proj", spec=...)
q_proj = allocator.bind(decl)
linear_bf16_raw(ctx, x=input_norm, weight=q_weight, output=q_proj)
```

Bad:

```python
slot = frame.full_attention.q_proj
q_proj = slot.activation("decode.layer.03.q_proj")
```

The bad shape makes a physical slot responsible for creating semantic identity.

## Views

Views are allowed when a shader ABI needs a different shape, byte range, or
descriptor range. A view must preserve or explicitly rename logical identity.

Use the same name when the view is only a different physical interpretation of
the same value:

```python
q_as_heads = q.view(shape=(batch, heads, head_dim))
```

Use a new declared name when the view represents a new model-visible value:

```python
q_packed = qkv_buffer.view_as("decode.layer.03.qkv_packed", ...)
```

Descriptor range is physical ABI. It should not create a new logical tensor by
itself.

## No TensorSlot At The Model Layer

Avoid a model-visible `TensorSlot`.

`TensorSlot` usually starts as an allocation convenience, but it tends to grow
`name`, `spec`, `shape`, `logical`, `activation`, `input`, and `output`
helpers. Once that happens it duplicates `LogicalTensor`.

Allowed internal allocator shape:

```python
@dataclass(frozen=True)
class PhysicalSlot:
    id: int
    storage: BufferSlice
    nbytes: int
```

Not allowed in model execution:

```python
class TensorSlot:
    name: str
    def logical(self) -> LogicalTensor: ...
```

Physical slots are planner output. Logical tensors are model API.

## Naming Rules

Names should be stable, hierarchical, and model-specific:

```text
input.input_ids
weights.layer.03.self_attn.q_proj
decode.layer.03.input_norm
decode.layer.03.self_attn.q_rope
cache.layer.03.key
output.next_token_id
```

Names should not include physical allocation details:

```text
workspace.hidden_a
slot4
buffer_17
tmp_0
```

If a temporary tensor matters for debug, liveness, or replay, give it a semantic
name. If it does not matter, keep it inside the shader.

## Liveness

Liveness is inferred from shader read/write records:

```text
dispatch 17 writes decode.layer.03.q_proj
dispatch 18 reads  decode.layer.03.q_proj
dispatch 18 writes decode.layer.03.q_rope
```

The planner may bind multiple logical tensors to the same physical slot when
their lifetimes do not overlap. This does not change their logical identity.

## Diagnostics

Error messages should report logical names first:

```text
first mismatch:
  tensor: decode.layer.03.attention.output
  writer: sdpa_decode_kv_packed
  dispatch: 84
```

Physical details are secondary:

```text
storage: device_local allocation 7, offset 1048576, bytes 8192
```

