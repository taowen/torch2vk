# Record / Replay 需要回答的问题

本文只列待决问题。

出发点：

```text
热路径性能要高。
显存分配和释放不能成为热路径成本。
```

## 目标和度量

- record/replay 要消灭哪些具体成本？
- 哪些成本允许留在 replay prepare 阶段？
- 哪些成本不能留在 replay hot path？
- Python frame/model code 是否还执行？
- ShaderContract validate 是否还执行？
- shape symbol resolve 是否还执行？
- push constant pack 是否还执行？
- pipeline lookup 是否还执行？
- descriptor set allocation 是否还执行？
- descriptor set update 是否每步执行？
- command buffer begin/end/record 是否还执行？
- `vkQueueSubmit` 是每 dispatch 一次，还是每 stage 一次？
- `vkWaitForFences` / readback 是否每 step 一次？
- Vulkan allocation/free 是否还执行？
- `FRAME_WORKSPACE` allocation 是否还执行？
- `REQUEST_STATE` grow/copy 是否还执行？
- 如何统计 hot path 中的 allocation 次数？
- 如何统计 hot path 中的 command buffer record 次数？
- 如何统计 hot path 中的 queue submit 次数？
- replay 的 CPU wall time、GPU time、allocation count、descriptor update count 分别如何测量？

## Replay 单位

- replay 的最小有效单位是什么？
- replay 单位候选是 frame、stage、decode step，还是完整 pipeline？
- 单位太小时 submit 次数是否仍然过多？
- 单位太大时动态 shape、分支、EOS、workspace high-water mark 是否导致命中率过低？
- 对 Qwen3-ASR，优先验证的 replay 目标单位是什么？
- `token_select` 是否和 decode step 放在同一个 replay 单位里？
- audio tower 是否需要 replay？
- text prefill 是否需要 replay？
- 完整 decode loop 是否需要 replay？
- 如果只 replay decode step，外层 loop 的 CPU 成本是否还能接受？
- 如果 replay K-step chunk，token 依赖如何处理？
- 如果 replay full pipeline，不同 decode length 如何处理？

## Command Buffer

- Vulkan command buffer 里哪些内容是固定的？
- 哪些动态性会被录进 `vkCmdPushConstants`？
- 哪些动态性会被录进 `vkCmdDispatch(x, y, z)`？
- 哪些动态性会影响 command 数量和顺序？
- 哪些动态性会影响 descriptor set layout？
- 哪些动态性会影响 descriptor range？
- 哪些动态性会影响 buffer offset、stride、shape 解释？
- replay 时 command buffer 是否必须完全不重新 record？
- safe replay 和 unsafe replay 对 command buffer 的要求是否不同？
- command buffer 是否按 frame、stage、request 或 replay session 持有？
- command buffer 生命周期如何和 GPU fence/timeline 绑定？
- command buffer 引用的 descriptor set 是否允许 replay 前更新？

## 动态长度

- prompt length 放在哪里？
- audio length 放在哪里？
- cache position 放在哪里？
- KV active length 放在哪里？
- KV storage capacity 放在哪里？
- decode step index 放在哪里？
- generated token count 放在哪里？
- done/eos flag 放在哪里？
- 哪些动态值现在在 push constants 里？
- 哪些动态值现在参与 direct dispatch group count？
- 哪些动态值现在来自 `LogicalTensor.spec.shape`？
- 哪些动态值需要成为 shader 可读取的数据？
- 哪些动态值改变时必须重新 record？
- 哪些动态值改变时只需要更新 buffer 内容？
- 哪些动态值改变时只需要更新 descriptor set？
- 哪些动态值改变时需要更新 indirect dispatch buffer？

## Dispatch

- direct dispatch 是否足够？
- 哪些 shader 的 dispatch group count 依赖 prompt length？
- 哪些 shader 的 dispatch group count 依赖 KV length？
- 哪些 shader 的 dispatch group count 依赖 audio length？
- 哪些 shader 的 dispatch group count 在 decode step 中固定？
- 是否需要 `vkCmdDispatchIndirect`？
- indirect dispatch 参数由 CPU 写还是 GPU 写？
- indirect dispatch buffer 如何同步？
- 固定 envelope + shader guard 的无效 work 成本是多少？
- bucket 化 dispatch 的命中率和浪费比例如何测？
- exact length capture 的 replay cache 数量是否可接受？

## Descriptor Set

- replay command buffer 绑定 descriptor set handle 后，是否允许 submit 前更新 descriptor 内容？
- ReplaySession 是否持有长期存活的 descriptor set？
- descriptor set 是 per replay session、per request，还是 per in-flight slot？
- descriptor update 是每 step、每 request，还是 resource identity 改变时才做？
- descriptor update 成本是否会抵消 replay 收益？
- 如何避免每 dispatch 更新 descriptor？
- 是否需要 descriptor set cache？
- 是否需要 descriptor ring 处理多 outstanding submit？
- 如何保证 descriptor set 更新时没有上一次 submit 正在使用？
- 如果 buffer handle 变了，哪些 descriptor binding 需要更新？
- 如果 buffer range 变了，哪些 descriptor binding 需要更新？
- 如果只变 buffer 内容，是否可以不更新 descriptor？
- descriptor range 绑定 active shape 还是 storage shape？

## Params Buffer

- 是否需要统一 params buffer？
- params buffer 的 ABI 如何定义？
- 每个 replay unit 一个 params buffer，还是全局 params buffer？
- params buffer 是 host-visible，还是 device-local + staging upload？
- 每步写 params buffer 的成本是多少？
- params buffer 写入后是否需要 flush？
- params buffer 如何避免和 GPU in-flight 读冲突？
- params buffer 是否需要 ring？
- 哪些现有 push constants 应迁移到 params buffer？
- 哪些 push constants 可以保留？
- params buffer 改变后 replay command buffer 是否仍可复用？

## LogicalTensor Shape

- `LogicalTensor.spec.shape` 当前同时承担了哪些含义？
- 是否需要区分 active shape、storage shape、capacity shape？
- ShaderContract 绑定的是 active shape 还是 storage shape？
- descriptor range 来自 active shape 还是 storage shape？
- dispatch symbols 来自 active shape、storage shape，还是 params buffer？
- compare/readback 用 active shape 还是 storage shape？
- request state grow 后 `LogicalTensor.spec.shape` 如何更新？
- replay 中 shape 改变是否一定导致 replay 失效？
- 如何表示“同一个 storage，不同 active length”？
- 如何避免 prompt_len/kv_len 改变时破坏 replay 命中？

## Allocator

- Raw Vulkan allocation 何时发生？
- replay hot path 是否允许 raw allocation？
- MODEL_WEIGHT 是否启动时一次性加载并常驻？
- REQUEST_STATE 是否按 request arena/slab 分配？
- FRAME_WORKSPACE 是否按 replay high-water mark 预分配？
- OP_SCRATCH 是否并入 frame workspace？
- HOST_INPUT / HOST_OUTPUT 是否使用 ring 或 pool？
- allocation grow 策略是什么？
- grow 是否需要 copy 旧内容？
- grow 的 copy 成本如何测？
- free 是逐 tensor free，还是 request/frame 整体 reset？
- frame workspace 是否可以在 replay session 生命周期内保持稳定？
- ReplaySession 是否 owns workspace allocations？
- ReplaySession 是否只引用外部 workspace arena？
- workspace offset 是 capture 后固定，还是每次 replay 动态分配？
- 多个 replay session 是否共享同一个 workspace arena？
- 不同 stage 的 workspace 是否能复用同一块大 arena？
- GPU 还在使用 allocation 时，allocator 如何避免 recycle？

## Liveness 和 Aliasing

- replay finalize 是否需要生成 liveness interval？
- liveness 基于 dispatch read/write edge，还是 LogicalTensor 声明？
- 如何计算 activation 最后一次使用？
- 哪些 tensor 可以 alias？
- 哪些 tensor 不能 alias？
- INOUT tensor 如何处理 alias？
- compare/debug target 是否阻止 alias？
- request state 是否禁止 alias？
- alias plan 是 per replay regime，还是全局？
- alias 后 descriptor offset 如何固定？
- alias 后 debug/readback 如何定位 tensor 内容？
- liveness/aliasing 对显存峰值降低多少？

## KV Cache

- KV cache 可以在哪些阶段 grow？
- KV cache 是否允许在 replayed command buffer 内 grow？
- grow 后 buffer handle 变了怎么办？
- grow 后 descriptor set 怎么更新？
- grow 后 stride/capacity 从哪里读？
- capacity 是否还在 push constants 里？
- capacity 是否需要进入 params buffer？
- contiguous KV grow 是否需要 copy old KV？
- old KV copy 成本在长序列下是否可接受？
- 是否需要 paged/slab KV？
- paged KV 的 page table 如何表示？
- page table 是 descriptor、params，还是普通 storage buffer？
- shader 里 page table lookup 成本是多少？
- paged KV 是否能减少 descriptor update？
- paged KV 是否能避免 old KV copy？
- Qwen3-ASR 当前序列长度下 contiguous KV 是否已经是瓶颈？

## Prompt / Prefill

- 不同请求 prompt length 不一致时，prefill 是否需要 replay？
- prefill shader sequence 是否随 prompt length 变化？
- prefill dispatch group count 是否随 prompt length 变化？
- prefill workspace size 是否随 prompt length 变化？
- RoPE buffer shape 是否随 prompt length 变化？
- attention cache/mask 是否随 prompt length 变化？
- audio scatter 是否随 prompt length 变化？
- exact prompt length replay 的 cache 数量是否可接受？
- prompt bucket replay 的无效 work 成本是多少？
- indirect dispatch 是否能覆盖 prefill 动态长度？
- prefill replay 是否需要先统一 params buffer？
- prefill replay 是否需要 active/storage shape 分离？

## Decode Loop

- 不同请求 decode length 不一致时，完整 loop 是否可 replay？
- decode length 改变是否改变 command 数量？
- EOS 提前 stop 如何处理？
- 是否固定跑 max_new_tokens？
- 是否 replay 单步 decode？
- 是否 replay K-step chunk？
- 每 step 一个 queue submit 是否够快？
- K-step chunk 中下一步 token 如何从上一步 token_select 传入？
- 是否需要 GPU-side loop？
- 是否需要 command buffer chain？
- done flag 是否每步 CPU readback？
- done flag 是否可延迟 readback？

## Token Feedback

- 下一步 input token 是否必须 CPU readback？
- `decode.input_ids` 能否直接绑定 `token_select.next_token` 的 GPU buffer？
- `token_select.next_token` 是否需要是 REQUEST_STATE？
- 当前 generated tokens 是否必须每步 grow？
- generated tokens 是否可以预分配并按 position 写入？
- cache_position 由 CPU 写，还是 GPU 自增？
- 如果 CPU 每步写 cache_position，成本是多少？
- 如果 GPU 自增 cache_position，如何处理 reset 和多请求？
- token feedback 留在 GPU 内后，debug/compare 如何读取边界结果？

## Synchronization

- replay command buffer 内 barrier 策略是什么？
- 是否每 dispatch 后插 conservative barrier？
- 什么时候做基于 buffer range overlap 的 barrier elimination？
- DispatchRecord 是否记录足够的 read/write ranges？
- INOUT tensor 的 read/write range 如何表示？
- 不同 buffer allocation 间是否可以省 barrier？
- descriptor update 和 command submit 之间需要什么同步？
- host-visible params/input 写入后需要什么 flush/invalidate？
- request state recycle 依赖什么 fence/timeline？
- 多 in-flight replay 如何避免覆盖仍被 GPU 读取的 buffer？

## Safe Replay / Unsafe Replay

- safe replay 要检查哪些条件？
- unsafe replay 不检查哪些条件？
- shader/pipeline ABI 如何检查？
- descriptor set layout 如何检查？
- command topology 如何检查？
- current LogicalTensor buffer 是否满足 binding 要求？
- workspace/request buffers 是否仍然 alive？
- params buffer 是否已写入？
- safe replay 允许多慢？
- prepare 阶段允许多慢？
- unsafe replay mismatch 如何 fallback 到 eager compare？

## Pipeline / Shader Cache

- SPIR-V compile 是否可能进入 request path？
- pipeline create 是否可能进入 replay path？
- descriptor pool growth 是否可能进入 replay path？
- capture 前是否需要 warmup 所有 pipeline？
- ReplaySession 是否强引用 pipeline？
- shader artifact 改变后 replay 如何失效？
- ShaderContract 改变后 replay 如何失效？
- pipeline layout 改变后 replay 如何失效？

## 并发和生命周期

- ReplaySession close 时是否需要 wait idle？
- 是否用 fence/timeline 追踪 in-flight command buffer？
- 多请求并发是否需要 per in-flight workspace？
- 多请求并发是否需要 per in-flight descriptor set？
- host input ring 如何和 in-flight submit 绑定？
- params buffer ring 如何和 in-flight submit 绑定？
- request state recycle 如何和 in-flight submit 绑定？
- ReplaySession 引用的 buffers 是否可以被外部 release？
- RuntimeSession close 时如何释放 replay resources？

## Debug / Compare

- replay path 是否做 PyTorch compare？
- replay path 是否做 readback？
- replay mismatch 如何定位？
- 是否先 eager compare，再 capture replay？
- replay 输出错时如何回到 eager drilldown？
- aliasing 后 debug dump 如何映射 LogicalTensor？
- params buffer 中的动态值是否写入 debug artifact？
- descriptor update 后的实际 binding 是否写入 debug artifact？

## 测试

- toy two-dispatch frame 如何测试 replay correctness？
- descriptor update 后 replay 读取新 input 如何测试？
- params buffer 改 prompt_len/cache_position 如何测试？
- fixed guard 如何测试？
- indirect dispatch 如何测试？
- request state grow 后 replay prepare 更新 descriptor 如何测试？
- workspace arena replay 后不 dangling 如何测试？
- multi-submit fence/recycle 如何测试？
- decode step replay 如何测试？
- token feedback 不回 CPU 如何测试？

## Benchmark

- eager per-dispatch baseline 如何测？
- frame command-buffer replay 如何测？
- decode-step replay 如何测？
- descriptor update cost 如何单独测？
- params buffer update cost 如何单独测？
- workspace allocation count 如何测？
- request-state grow count 如何测？
- queue submit count 如何测？
- CPU wall time per token 如何测？
- GPU time per token 如何测？
- memory peak 如何测？
- allocation/free 次数如何测？

## 需要先回答的 Qwen3-ASR 问题

- Qwen3-ASR 当前 decode step 有多少 dispatch？
- decode step 中哪些 dispatch 的 shape 真正动态？
- decode step 中哪些 push constants 是动态值？
- decode step 中哪些 descriptor binding 每步必须变化？
- `cache_position` 是否已经完全从 push constants 移出？
- KV capacity 是否还被 push constants 假定？
- decode input token 是否还来自 HOST_INPUT？
- token_select 输出是否可以直接作为下一步 decode input？
- generated_tokens 是否还在每步 grow？
- done flag 是否每步 readback？
- text prefill 的 prompt length 是否影响 replay 命中？
- audio tower 是否值得 replay？
- frame workspace 的 allocation/free 是否是热点？
- request state grow/copy 是否是热点？
