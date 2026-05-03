# Record Replay Design

Record/replay is a performance mechanism for stable shader sequences.

It is not a model representation, not an IR, and not a workflow engine. The
semantic source remains the Python execution code that calls shaders with
`LogicalTensor` objects.

## Purpose

Python eager execution is the readable path:

```text
Python model code
  -> shader calls
  -> Vulkan dispatches
```

Replay exists for repeated workloads where CPU-side dispatch preparation and
submission overhead becomes visible:

```text
run Python execution once
  -> record concrete Vulkan submission sequence
  -> validate tensor/storage fingerprint
  -> replay same sequence many times
```

Typical replay targets:

1. LLM decode step for a fixed batch and cache regime;
2. repeated audio encoder blocks with fixed shapes;
3. vocoder blocks with stable tensor shapes;
4. benchmark loops for one known workload.

Do not use replay to make an unverified model fast. First make eager correct.

## Non-Goals

Replay must not:

1. infer model semantics;
2. choose shader variants;
3. allocate new model tensors;
4. perform PyTorch comparison;
5. perform debug readback;
6. mutate logical tensor names;
7. hide a different execution order from the Python source.

If replay output differs from eager output, replay is wrong.

## Record Input

The recorder consumes an already executed or dry-run Python shader sequence.

Each recorded dispatch should contain:

```text
shader variant name
descriptor bindings
push constants
specialization constants
workgroup counts
logical reads
logical writes
physical storage identity
```

The logical metadata is retained for validation and diagnostics. The replay
runtime should not need it for hot execution.

Example record:

```text
dispatch 84 sdpa_decode_kv_packed
  workgroups (8, 1, 4)
  reads  q=decode.layer.03.q_rope
  reads  key_cache=cache.layer.03.key
  reads  value_cache=cache.layer.03.value
  writes output=decode.layer.03.attention.output
  descriptors:
    0 q buffer=12 offset=4096 bytes=8192
    1 key_cache buffer=19 offset=0 bytes=67108864
    2 value_cache buffer=20 offset=0 bytes=67108864
    3 output buffer=12 offset=12288 bytes=8192
```

## Regime Key

Replay is valid only for a declared regime.

A regime key should include every value that can change the recorded sequence:

1. model family and variant;
2. phase, such as prefill or decode;
3. batch size;
4. sequence length or decode position class;
5. cache capacity;
6. tensor shapes;
7. shader variant choices;
8. specialization constants;
9. descriptor range requirements;
10. relevant memory policy.

Example:

```python
ReplayRegime(
    model="qwen",
    phase="decode",
    batch=4,
    token_steps=1,
    max_seq_len=4096,
    hidden=4096,
    kv_heads=8,
    head_dim=128,
)
```

If the regime changes, record again.

## Storage Fingerprint

Replay must validate that the physical storage still matches the recorded
sequence.

A storage fingerprint should include:

1. Vulkan buffer identity;
2. allocation generation or stable allocation id;
3. byte offset;
4. descriptor range bytes;
5. dtype;
6. logical shape;
7. layout;
8. role and memory policy when relevant.

This prevents replay from writing into a different buffer after allocator reuse.

The fingerprint is physical. It does not replace `LogicalTensor.name`.

## Replay Object Lifecycle

The replay object owns prepared Vulkan resources:

```text
ReplaySession
  recorded command buffers
  prepared descriptor bindings
  pipeline references
  storage fingerprint
  regime key
```

Close order should be explicit:

1. wait for in-flight work or require caller synchronization;
2. destroy recorded command buffers;
3. release prepared descriptor bindings;
4. release replay-owned temporary allocations;
5. clear session state.

Replay must not own model weights or persistent state. The model/runtime owns
those tensors.

## Capture Flow

Recommended capture flow:

```text
1. Build model schema.
2. Materialize weights and persistent state.
3. Allocate/bind logical tensors for the regime.
4. Run Python execution through a recording dispatch target.
5. Validate every shader contract.
6. Build liveness plan if needed.
7. Prepare descriptors and pipelines.
8. Record command buffers.
9. Store regime key and storage fingerprint.
10. Optionally run one replay and compare against eager output.
```

The recording dispatch target should expose the same `.run(shader, **tensors)`
surface as eager execution.

## Hot Replay Flow

Hot replay should be small:

```text
1. Check regime key.
2. Check storage fingerprint, or rely on a previously checked stable session.
3. Update allowed dynamic inputs.
4. Submit recorded commands.
5. Read output only if the public API requires host output.
```

It should not rebuild descriptor sets on every call unless the backend has no
stable descriptor strategy yet.

## Dynamic Inputs

Replay can support changing input values if their storage identity is stable.

Examples:

1. writing new token ids into the same host-visible input tensor;
2. updating a small uniform or push constant when the recorded command buffer
   strategy supports it;
3. writing generation state into already-bound device buffers before replay.

Changing tensor shape, descriptor range, or shader variant invalidates replay.

## Push Constants

Push constants are tricky because many Vulkan command buffers bake them at
record time.

Use one of these policies:

1. record separate sessions for different push constant values;
2. keep dynamic values in a small uniform/storage buffer;
3. record command buffers only for regimes where push constants are stable.

Do not silently replay with stale push constants.

## Liveness And Aliasing

Replay can use physical aliasing chosen by liveness planning, but it must record
the resulting storage fingerprint.

Rules:

1. no aliasing across overlapping logical lifetimes;
2. no aliasing between persistent state and frame workspace;
3. no aliasing between weights and mutable workspace;
4. no aliasing change after replay capture unless the session is invalidated.

If a logical tensor is rebound to a different physical slot, replay must be
discarded or recaptured.

## Debug Mode

Replay debug should be opt-in and separate from hot replay.

Allowed debug replay tools:

1. dump recorded dispatch list;
2. compare replay descriptors against eager descriptors;
3. run replay once and compare final outputs;
4. bisect by recapturing shorter sequences.

Hot replay should not contain debug readback or PyTorch reference work.

## Correctness Contract

Replay is correct when:

```text
Vulkan eager output == replay output
Vulkan eager selected boundaries == PyTorch reference selected boundaries
```

If eager and PyTorch differ, fix eager or shader semantics first. If eager and
replay differ, fix replay recording, storage identity, synchronization, or
dynamic input policy.

