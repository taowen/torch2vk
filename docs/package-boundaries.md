# Package Boundaries

`torch2vk` 按执行语义从高到低分层：

```text
runtime
  连接 LogicalTensor metadata、materialization、dispatch record、debug/compare/replay。

logical
  模型面对的稳定 tensor declarations。可以记录当前 Vulkan BufferSlice 状态，但不能依赖 dispatch/runtime。

vulkan
  Vulkan device、buffer、allocator、pipeline、descriptor binding、queue submission、barrier、readback/copy/fill 等操作。
  这一层不能 import logical/runtime/dispatch；允许 import checkpoints 以标注 checkpoint-backed upload。
  BufferAllocation、BufferSlice、GPU timeline point 属于这一层。

host/types/kernel/checkpoints
  host tensor、稳定 tensor metadata、shader requirements、checkpoint readers。
```

关键边界：

1. 模型 adapter 只传 `LogicalTensor`。
2. `RuntimeSession` 是唯一把 `LogicalTensor` materialize 成 `BufferSlice` 的层。
3. `vulkan.compute_pipeline` 只把 `BufferSlice` 降成 descriptor buffer binding。
4. `vulkan` driver 不感知 `LogicalTensor`、probe、compare、lifetime；checkpoint 只作为 typed host data source。
5. 不再引入执行态 tensor wrapper；运行时对象只保留 `LogicalTensor` 和 `BufferSlice` 两层。

这些方向由 `import-linter` 强制：

```bash
uv run lint-imports
```
