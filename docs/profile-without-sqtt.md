# 不依赖 SQTT 的 Profiler

这份文档定义 `torch2vk` 的普通 profiler。它不依赖 Mesa fork，不录 RGP，也不要求 AMD GPU。
它的核心产物是 runtime dispatch manifest：一份由 `torch2vk` 自己写出的 dispatch 语义记录。

这份 manifest 既能独立用于普通性能分析，也会被 SQTT profiler 复用。

## 它回答什么

普通 profiler 要回答：

1. 一次运行里有哪些 frame 和 dispatch。
2. 每个 dispatch 对应哪个 shader、pipeline、SPIR-V。
3. 每个 dispatch 的 shape symbols、dispatch size、push constants。
4. 每个 dispatch 读写了哪些 logical tensor，以及当时的 shape/dtype/descriptor 范围。
5. 如果开启 timestamp，每个 dispatch 或 replay dispatch 的 GPU 时间是多少。

它不回答 ISA、wave stall、cache miss 等硬件原因。这些属于 SQTT profiler。

## 当前代码基础

当前 `torch2vk` 已经有足够信息写 manifest：

1. `RuntimeSession.dispatch()` 是 eager dispatch 的入口。
2. `DispatchRecord` 已记录 `index`、`frame`、`shader`、logical reads/writes、symbols、
   `dispatch_size`、descriptor views、tensor snapshots、push constants。
3. `ComputePipeline` 已记录 `debug_name`、`shader_spv_sha256`、`pipeline_identity_sha256`。
4. `ReplayPlan.dispatch_entries` 保存 replay 中每个 dispatch 的 pipeline、binding、dispatch size 和
   dynamic params 信息。
5. `VulkanDevice.timestamp_period_ns` 可用于 timestamp query 结果换算。

重要原则：

```text
RuntimeSession.dispatch_records 是语义事实源。
driver trace、RGP、timestamp 只能补充信息，不能反推模型语义。
```

## 最终形态

普通 profiler 只需要两个层次。

第一层永远可用：写 manifest，不插入 GPU query。

第二层按需开启：在 command buffer 中写 Vulkan timestamp query，补充耗时。

推荐 run 目录：

```text
.cache/torch2vk/profile/<run-id>/
  run.json
  dispatches.jsonl
  summary.json
```

其中 `dispatches.jsonl` 是最重要的文件。每行对应一个 runtime dispatch，推荐字段：

```json
{
  "dispatch_index": 42,
  "frame": "qwen3_asr.text_decode",
  "phase": "eager",
  "shader": "text_attention_decode_f32",
  "pipeline_debug_name": "agp.text_attention_decode_f32.0123456789abcdef",
  "pipeline_identity_sha256": "...",
  "shader_spv_sha256": "...",
  "symbols": {"B": 1, "T": 128},
  "dispatch_size": [32, 1, 1],
  "push_constants": {"token_pos": 17},
  "reads": [],
  "writes": [],
  "elapsed_ns": null
}
```

`elapsed_ns` 可以为空。这样 manifest-only 模式和 timestamp 模式使用同一种文件结构。

## eager profiling

当前 eager dispatch 是一条 dispatch 一个 command buffer，一个 submit 后等待完成。因此 eager profiling 很直接：

```text
RuntimeSession.dispatch()
  -> 算出 dispatch_index
  -> 解析 tensor/symbol/push constants
  -> 获取 ComputePipeline
  -> 可选：在 dispatch 前后写 timestamp
  -> 提交并等待
  -> append DispatchRecord
  -> 写一行 dispatches.jsonl
```

eager 时间适合找粗热点和做调试，但它不是最终吞吐，因为 eager 每个 dispatch 都 submit/wait。

## replay profiling

replay 是更接近最终推理的形态。普通执行仍使用当前 `execute_replay(plan)`。需要 profiling 时，使用同一组
`ReplayPlan.dispatch_entries` 录一条 profile command buffer：

```text
for entry in plan.dispatch_entries:
  timestamp begin
  record dispatch or indirect dispatch
  timestamp end
  barrier
submit profile command buffer
read query results
```

这个 profile command buffer 可以临时生成，不应该替换 `ReplayPlan.command_buffer`。这样普通 replay 不承担
timestamp query 成本。

高性能 replay profile 的关键是保持 replay 的提交形态，而不是退回到 eager：

```text
warmup N 次 execute/profile submit，不读结果
repeat M 次 profile submit
一次 submit 覆盖整条 replay plan
一次性读取 query 结果
按 dispatch 聚合 min / median / p95
```

不要为了测每个 dispatch 而把 replay 拆成多个 submit。拆开会改变 queue 调度、barrier 成本和 cache 行为，
测到的就不再是 replay。正确做法是在同一个 replay command buffer 内给每个 dispatch 前后写 timestamp。

profile command buffer 应尽量复用 replay 已经解析好的对象：

1. 复用 `ReplayPlan.dispatch_entries` 的 pipeline、binding、push constants、dispatch size。
2. dynamic replay 复用 `execute_replay()` 的参数更新逻辑：submit 前更新 params buffer 和 indirect buffer。
3. barrier 顺序和正常 replay 保持一致。
4. readback copy 不放进 dispatch 计时区间；如果要测 output copy，单独作为 plan 级时间记录。
5. query pool、profile command buffer、临时 fence 都按 replay plan 缓存，避免每次 profile 重建。

query 布局保持简单：

```text
query 0: plan begin
query 1: dispatch 0 begin
query 2: dispatch 0 end
query 3: dispatch 1 begin
query 4: dispatch 1 end
...
last query: plan end
```

`plan begin/end` 给出整条 replay 的 GPU 时间；每个 dispatch 的 begin/end 给出 dispatch 内部时间。
两者都要记录，因为 dispatch 时间之和不一定等于 plan 时间，中间还有 barrier、indirect dispatch 读取、
cache flush 等成本。

结果写回时，`dispatches.jsonl` 保留每个 replay dispatch 的静态元数据和聚合耗时，例如：

```json
{
  "phase": "replay",
  "replay_plan": "text_decode",
  "replay_dispatch_index": 3,
  "source_dispatch_index": 42,
  "elapsed_ns": 11840,
  "elapsed_ns_min": 11200,
  "elapsed_ns_p95": 12600
}
```

`summary.json` 记录 plan 级时间：

```json
{
  "replay_plan": "text_decode",
  "warmup": 5,
  "repeat": 50,
  "plan_elapsed_ns_median": 842000,
  "plan_elapsed_ns_min": 821000,
  "plan_elapsed_ns_p95": 870000
}
```

这样既能看单个 shader 的热点，也能看整条 replay 的真实收益。

## 和 SQTT 的关系

SQTT profiler 必须复用这份 manifest。它需要这些字段做 join：

1. `dispatch_index`
2. `frame`
3. `phase`
4. `shader`
5. `pipeline_debug_name`
6. `pipeline_identity_sha256`
7. `shader_spv_sha256`

因此普通 profiler 即使不开 timestamp，也应该能生成完整 dispatch manifest。SQTT 只消费
`phase="record"` 的 dispatch 行；`phase="replay"` 的行只用于普通 profiler 的 replay timestamp 分析。

SQTT debug label 不需要新的 tag 字段。它直接由 manifest 里的 `frame`、`shader` 和 `dispatch_index`
生成：

```text
agentorch-profile-submit:frame=qwen3_asr.text_decode;shader=text_attention_decode_f32;dispatch=42
```

`LogicalTensor.name`、tensor version、shape、descriptor range、pipeline identity 等完整语义仍只保存在
`dispatches.jsonl`，不要塞进 label。

## 实现顺序

推荐按这个顺序实现：

1. 写 manifest-only recorder：不改 command buffer，只在 `RuntimeSession.dispatch()` 后写
   `dispatches.jsonl`。
2. 给 record dispatch 加可选 timestamp。
3. 给 replay 增加临时 profile command buffer。
4. 输出按 frame/shader 聚合的 `summary.json`。

这四步完成后，普通 profiler 已经完整可用；SQTT profiler 也有了可靠的语义输入。

## 不变量

1. profiler 关闭时，runtime 和 replay 行为完全不变。
2. manifest 默认不保存 tensor 原始值，只保存 shape/dtype/name/descriptor 元数据。
3. timestamp 失败不能写假时间；应写 `elapsed_ns: null` 和失败原因。
4. dispatch 顺序必须和 `RuntimeSession.dispatch_records` 一致。
5. pipeline join 以 hash 为准，不以文件名或耗时猜测为准。
