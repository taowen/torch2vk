# Validation And Debugging Guide

`torch2vk` should keep PyTorch eager as the semantic reference while Vulkan
executes the candidate path.

The goal is not just to know that output differs. The goal is to find the first
logical tensor or shader boundary where it starts to differ.

## Artifact Keys

All reference and candidate artifacts should be keyed by `LogicalTensor.name`.

Good:

```text
decode.layer.03.input_norm
decode.layer.03.self_attn.q_proj
decode.layer.03.self_attn.q_rope
decode.layer.03.attention.output
output.next_token_id
```

Bad:

```text
torch_hook_17
candidate_qproj
workspace.hidden_a
slot4
```

If the same value appears in a generation step, prefix it with the step:

```text
generate.step_003.decode.layer.03.attention.output
```

The base logical tensor name should still be visible.

## Compare Boundaries

Do not compare every tensor by default. Declare useful boundaries:

1. model input embedding;
2. per-layer attention output;
3. per-layer MLP output;
4. final norm;
5. logits;
6. sampled token;
7. persistent state updates when relevant.

Each boundary should reference logical tensors, not strings copied from another
table:

```python
Boundary(
    name="decode.layer.03.attention",
    tensors=(layer.input_norm, layer.q_proj, layer.k_proj, layer.attention_output),
    compare=ComparePolicy(rtol=1e-2, atol=1e-2),
)
```

## Reference Rules

Use the simplest PyTorch reference source that corresponds to the logical
tensor.

Use direct hooks for module boundaries:

```python
A("decode.layer.03.input_norm", ref=HookRef(lambda m: m.layers[3].input_layernorm))
```

Use derived references for Vulkan-only tensors:

```python
A(
    "decode.layer.03.qkv_packed",
    ref=DerivedRef(inputs=(q_proj, k_proj, v_proj), fn=pack_qkv_ref),
)
```

Use step references for generation policy values:

```python
O("generate.next_token", ref=StepRef(fn=sample_next_token_ref))
```

Do not put boundary ordering, checkpoint restore, or readback policy inside a
single tensor reference rule. Those are boundary-level policies.

## Candidate Trace

Every shader dispatch should record logical reads and writes:

```text
dispatch 84 sdpa_decode_kv_packed
  reads  q=decode.layer.03.q_rope
  reads  key_cache=cache.layer.03.key
  reads  value_cache=cache.layer.03.value
  writes output=decode.layer.03.attention.output
```

This trace is enough to answer:

1. who wrote this tensor;
2. which tensors were read by that writer;
3. what to read back after a mismatch;
4. which physical buffers must remain stable for replay.

## Mismatch Drilldown

Normal compare flow:

```text
run Vulkan eager
run PyTorch eager reference
compare declared boundaries in order
stop at first mismatch
walk candidate dispatch trace backward
read writer inputs and output
report first shader whose inputs match and output diverges
```

The report should start with semantic names:

```text
first mismatch: decode.layer.03.attention.output
writer shader: sdpa_decode_kv_packed
writer dispatch: 84
matching inputs: q, key_cache, value_cache
divergent output: output
```

Only then include physical details such as buffer offsets.

## Precision Contracts

Every compare point needs a policy:

```python
ComparePolicy(kind="tensor", rtol=1e-2, atol=1e-2)
ComparePolicy(kind="relative_l2", threshold=1e-4)
ComparePolicy(kind="token")
```

The policy should reflect the operation. Attention softmax and matmul may need
different tolerances. Token equality is stricter than logits tolerance.

When an optimized shader changes accumulation order, document the expected
numeric effect next to the variant or boundary.

## Checkpoint And Rerun

For expensive multi-step workflows, a boundary may declare a checkpoint tensor
or payload that lets debug rerun a smaller region.

Example:

```python
Boundary(
    name="decode.layer.03.mlp",
    tensors=(layer.mlp_down, layer.output),
    checkpoint=Checkpoint(tensor=layer.post_attention_norm),
)
```

Checkpointing is a debug policy. It must not become a second model execution
path.

## Replay Validation

Replay is allowed only after eager execution is correct.

Replay should reuse an already captured Vulkan submission sequence. It should
not:

1. run PyTorch reference;
2. do debug readback;
3. infer liveness again;
4. allocate new model tensors;
5. reinterpret model semantics.

Validate replay by comparing its outputs against eager Vulkan and PyTorch
reference for the same regime.

## Integration Test Shape

Each model family should have one high-value integration test:

1. load a small checkpoint or fixture;
2. run PyTorch eager reference;
3. run Vulkan eager candidate;
4. compare declared boundaries;
5. run replay if supported;
6. compare replay output;
7. print first semantic mismatch on failure.

Small unit tests are useful for utilities, but the integration test is the
contract that the model port still means the same thing as PyTorch.

