# Replay From Cache

本文记录 `torch2vk` replay cache 的关键决策。目标不是建立另一套 graph 或执行器，而是让真实 eager
运行产生的 Vulkan dispatch stream 可以被缓存，并在后续兼容请求中复用。

核心目标：

```text
第一次：eager warmup -> materialize -> record command buffer -> cache ReplayPlan
后续：rebind descriptors / update params -> submit cached command buffer
```

热路径不应该重新执行 Python model code、不应该重新 `vkBeginCommandBuffer` / `vkCmd*` / `vkEndCommandBuffer`，
也不应该发生普通 Vulkan allocation/free。

## 基本边界

模型目录仍然只写 eager execution。Replay 只能从 `RuntimeSession` 已经观察到的 dispatch facts 生成：

```text
ShaderVariant.__call__
  -> RuntimeSession.dispatch
  -> materialize tensors
  -> resolve shader contract symbols
  -> bind pipeline/descriptors
  -> submit eager dispatch
  -> append DispatchRecord
```

`ReplayPlan` 从这些 `DispatchRecord` 和对应 `ShaderVariant` 构建。模型代码不构造 replay IR，也不维护另一套
replay-only graph。

## ReplayPlan 持有什么

一个 cached `ReplayPlan` 持有：

```text
command_buffer
fence
dispatch_entries
bound descriptor sets
params buffers
optional indirect dispatch buffer
replay-owned workspace allocations
optional readback slots
```

command buffer 中记录的是：

```text
vkCmdBindPipeline
vkCmdBindDescriptorSets
vkCmdPushConstants for static constants
vkCmdDispatch or vkCmdDispatchIndirect
conservative compute barriers
optional readback copy commands
```

后续 replay 只提交这条 command buffer。descriptor set handle 被记录在 command buffer 中，但 descriptor set
内容可以在 submit 前更新。因此同一个 command buffer 可以在兼容请求之间复用，只要 descriptor layout、
shader ABI 和 command topology 不变。

## 不每次 Record

replay cache 的关键收益是消除 CPU 侧 command recording：

```text
命中 cache 时不再：
  run Python decode frame
  validate ShaderContract for every dispatch
  resolve every shape symbol for command recording
  pack static push constants
  allocate descriptor set
  begin/end command buffer
  issue vkCmdBind*/vkCmdDispatch*
```

命中 cache 时允许做：

```text
write small host-visible control buffers
flush params / indirect dispatch buffer
update existing descriptor sets when buffer identity/range changed
vkQueueSubmit cached command buffer
wait/readback only when caller explicitly needs host-visible output
```

`mode` 语义：

```text
default       use compatible cached plan, otherwise warmup and record
require_cache fail on cache miss
force_record  ignore cache and build a fresh plan
```

测试应该用 `force_record` 建立第一条 plan，再用 `require_cache` 验证第二个兼容请求确实没有重新 record。

## 显存和 Buffer 生命周期

Replay 热路径不能依赖 frame allocator 每次重新分配出相同地址。当前决策是：

```text
MODEL_WEIGHT      常驻，不能 rebind
FRAME_WORKSPACE   replay plan owns capture-time allocations
OP_SCRATCH        replay plan owns capture-time allocations
REQUEST_STATE     可 rebind，适合 KV cache / logits / generated tokens
HOST_INPUT        可 rebind 或写入 replay-owned host-visible allocation
HOST_OUTPUT       可 rebind
```

`FRAME_WORKSPACE` 和 `OP_SCRATCH` 被视为 replay-owned storage。这样 replay 不需要在每次 submit 前重新走
workspace allocation，也不会依赖 frame scope reset 后的临时 buffer。

`REQUEST_STATE` 是跨请求/跨 step 的状态，例如 KV cache、token、logits。它允许在 cache 命中时 rebind 到新请求的
buffer。grow 后 buffer handle 或 descriptor range 可能改变，因此 cache 命中时必须先调用
`rebind_replay_plan(...)` 更新 descriptor set。

## Descriptor Rebind

Vulkan command buffer 记录 descriptor set handle，不记录 descriptor set 里的 buffer address。利用这一点：

```text
record:
  bind descriptor set A into command buffer

replay prepare:
  update descriptor set A to point at current request buffers

replay:
  submit same command buffer
```

rebind 必须检查：

```text
plan belongs to same device
plan is still alive
descriptor tensor names are present
new tensors can be materialized
static shape symbols match recorded values
dynamic shape symbols are explicitly declared dynamic
```

如果只变 buffer 内容，不需要 descriptor update。如果 buffer handle、offset、range 变了，需要 update existing
descriptor set，但不需要重新 record command buffer。

## 动态长度

不同 wav 长度会改变 prompt length、audio feature length 和 KV capacity。Replay 不应该靠按长度 bucket 规避
问题，也不应该为每个 exact length record 一条 command buffer。

当前采用的方向是类似 DynamicCache 的连续 KV storage：

```text
KV storage capacity belongs to REQUEST_STATE buffer shape
active KV length comes from cache_position
shader attention loop uses active length
descriptor binds current capacity buffer
capacity stride S is a dynamic symbol
```

因此长度变化时：

```text
不重新 record command buffer
rebind KV descriptors to the current request buffers
write S into params buffer
write indirect dispatch counts if dispatch formula references dynamic symbols
submit cached command buffer
```

注意不要让 shader 按 capacity 做无效 attention。Qwen3-ASR decode attention 仍使用：

```glsl
cache_len = uint(cache_position[0]) + 1u
for key_pos in 0 .. cache_len
```

`S` 只作为 KV cache row stride，不作为 active loop length。

## Push Constants 和 Params Buffer

Push constants 被录进 command buffer。会随请求长度或 grow 后 capacity 改变的值不能继续放在 push constants 中。

决策：

```text
static constants stay in push constants
dynamic scalar values move to params buffer
dispatch group counts that depend on dynamic symbols use vkCmdDispatchIndirect
```

Qwen3-ASR decode 已经把 KV capacity stride `S` 从 decode attention / decode KV write 的 push constants 移到
params buffer。`S` 被声明为 replay dynamic symbol，因此不同长度请求可以复用同一条 replay command buffer。

## Indirect Dispatch

如果 `dynamic_symbol_names` 非空，`ReplayPlan` 使用一个 host-visible indirect dispatch buffer：

```text
num_dispatches * sizeof(uint32[3])
```

每次 replay 前根据 dispatch formula 和当前 dynamic symbols 写入 group count，然后 command buffer 中使用
`vkCmdDispatchIndirect`。

这避免了因为 dispatch group count 改变而重新 record command buffer。对 Qwen3-ASR decode，很多 dispatch
维度本身固定，但启用 dynamic symbol 后统一走 indirect dispatch，保持 replay 路径简单。

## Token Feedback

自回归 decode 的下一步 token 不应该 CPU readback 后再写入下一步 input。Replay plan 支持：

```text
token_feedback_source = token_select.next_token
token_feedback_target = decode.input_ids
```

构建 replay descriptor 时，`decode.input_ids` 的 binding 可以指向 `token_select.next_token` 的 GPU buffer。
这样下一步 decode 直接读取上一步 token_select 的输出。

generated tokens 预分配到 REQUEST_STATE buffer，token_store 按 index 写入，并维护：

```text
generated_length
stopped
```

EOS 后 replay loop 可以通过很小的 host readback 检查 `stopped` 并提前退出。长期可以把 done flag readback
进一步延迟或搬到 GPU-side loop，但当前优先保证正确性和避免每步重新 record。

## Qwen3-ASR Decode Replay 决策

Qwen3-ASR 目前 replay 的单位是单个 decode step，包含：

```text
text_decode
token_select
token_store
```

Prefill 和 audio tower 仍走 eager。原因是 decode step 是循环热点，且 command topology 固定，最适合先做
replay from cache。

Qwen3-ASR decode cache 命中条件：

```text
same device
same stop_on_eos replay namespace
same shader/pipeline ABI
same command topology
same descriptor set layout
static symbols unchanged
dynamic symbol S allowed to change
request tensors can be rebound
```

不同 wav 长度允许复用同一条 decode replay plan，因为 KV capacity stride `S` 是 dynamic symbol，active length
来自 `cache_position`。

## Debug 和 Compare

Replay 热路径不做 PyTorch compare，不做 full readback。常规正确性测试应该检查完整 transcript 和 EOS。

Frame-level PyTorch compare 仍然有价值，但它是调试工具：

```text
frame exit
  run PyTorch model forward
  capture frame boundary output such as logits
  compare against Vulkan readback
```

失败时再使用现有 writer drilldown 读取相关 dispatch 的输入输出。不要把 shader-level drilldown 放进 replay hot
path。

## 测试要求

关键测试应覆盖：

```text
first request force_record
second request require_cache
second request has different wav length / prompt length / audio feature length
cached command_buffer identity unchanged
second request adds no text_decode DispatchRecord
both requests decode to EOS
both requests match full transcript
```

这样才能证明 replay cache 同时解决了：

```text
显存 buffer identity/range 变化后的 descriptor rebind
动态 KV capacity 后的 params update
不同长度请求不重复 record
完整 ASR 输出正确
```
