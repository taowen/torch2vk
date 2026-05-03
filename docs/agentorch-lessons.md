# Lessons From Agentorch Investigation

This note records design lessons from reading Agentorch's LogicalTensor porting
guide and implementation. It is not a compatibility target for `torch2vk`.

## Useful Ideas

Agentorch has several useful ideas worth keeping:

1. make logical tensor names the common key for weights, dispatch traces,
   reference artifacts, debug artifacts, replay, and liveness;
2. pass logical tensor objects to shader wrappers so dispatch can record reads
   and writes automatically;
3. keep shader variants as explicit Python objects with contracts, bindings,
   source, dispatch geometry, and runtime specialization;
4. make model execution ordinary Python that calls shaders in source order;
5. materialize weights from schema declarations rather than hand-written
   checkpoint maps in execution code;
6. compare against PyTorch eager at named boundaries;
7. use liveness from logical read/write edges to choose physical aliasing;
8. treat replay as captured execution, not as a new semantic source.

Those are aligned with `torch2vk`.

## Main Problem To Avoid

Agentorch's newer guide says the only model-visible tensor value should be
`LogicalTensor`, but its current implementation still has model-visible
`TensorSlot` trees.

The problematic shape is:

```python
frame.workspace.full_attention.q_proj.activation("decode.layer.03.q_proj")
```

This means:

1. the frame slot tree has its own model-shaped field structure;
2. each slot has a name, spec, shape, and allocation;
3. the slot can manufacture a `LogicalTensor`;
4. execution code or logical-view code maps physical slot fields to semantic
   logical names.

That recreates two namespaces:

```text
frame.workspace.full_attention.q_proj
decode.layer.03.q_proj
```

The first namespace is physical/workspace-shaped. The second is semantic. Once
both are visible to model code, they can diverge.

## torch2vk Rule

`torch2vk` should use this rule:

```text
LogicalTensor is model API.
PhysicalSlot is allocator output.
BufferSlice is Vulkan storage.
```

No object other than `LogicalTensor` should have both:

1. a model-semantic name; and
2. methods that produce model-visible tensors.

An allocator may produce anonymous physical slots:

```python
@dataclass(frozen=True)
class PhysicalSlot:
    id: int
    storage: BufferSlice
```

But model execution should not see:

```python
slot.activation("decode.layer.03.q_proj")
```

The schema should declare `decode.layer.03.q_proj` first, and the planner should
bind storage to that logical tensor.

## Declaration Versus Runtime Handle

It is still reasonable to separate schema declarations from bound runtime
handles:

```text
LogicalTensorDecl
  name, spec, role, source/ref/compare

LogicalTensor
  same identity plus storage
```

This is a phase distinction. It is not a second namespace.

Use it to support:

1. schema construction before device allocation;
2. weight materialization;
3. reference capture;
4. liveness planning;
5. validation without Vulkan.

Do not add a third model-visible tensor object.

## Shader Variant Pattern

Agentorch's shader pattern is valuable:

```text
ShaderVariant
  name
  family
  contract
  source
  specialization constants
  runtime shape resolver
  dispatch geometry
```

Calling a variant with logical tensors lets the dispatcher derive:

```text
logical_reads  from input fields
logical_writes from output fields
```

`torch2vk` should adopt that behavior. It makes debug and liveness almost free
once execution is written correctly.

## Replay Boundary

Agentorch's replay idea is also useful if kept narrow:

```text
run Python execution once
record concrete Vulkan dispatch sequence
validate storage fingerprint
replay without Python model execution
```

Replay should never introduce new names, new tensor semantics, or a hidden
workflow abstraction. It is a performance tool for stable shader sequences.

