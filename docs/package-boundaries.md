# Package Boundaries

`torch2vk` 只保留两层执行边界：

```text
runtime
  模型 adapter 面对的唯一通用层。
  包含 LogicalTensor、Frame、ShaderContract、materialization、RuntimeSession、dispatch record、debug/compare/replay。

vulkan
  Vulkan device、buffer、allocator、pipeline、descriptor binding、queue submission、barrier、readback/copy/fill 等操作。
  这一层不能 import runtime；允许 import checkpoints 以标注 checkpoint-backed upload。
  BufferAllocation、BufferSlice、GPU timeline point 属于这一层。
```

关键边界：

1. 模型 adapter 只 import `torch2vk.runtime` 和传 `LogicalTensor`。
2. `RuntimeSession` 是唯一把 `LogicalTensor` materialize 成 `BufferSlice` 的层。
3. 运行时输入绑定使用 `Mapping[LogicalTensor, object]`；不引入单独的 input key 或输入别名 metadata。
4. Runtime 不维护 `name -> LogicalTensor` registry；`LogicalTensor.name` 只用于报告和声明校验。
5. `vulkan.compute_pipeline` 只把 `BufferSlice` 降成 descriptor buffer binding。
6. `vulkan` driver 不感知 `LogicalTensor`、probe、compare、lifetime；checkpoint 只作为 typed host data source。
7. 不再引入执行态 tensor wrapper；运行时对象只保留 `LogicalTensor` 和 `BufferSlice` 两层。

`torch2vk.runtime` 内部可以按文件拆分实现，例如 `logical.py`、`frame.py`、`shader.py`、`session.py`，
但这些都是同一个 runtime 包的内部组织，不是新的架构层。模型 adapter 需要哪个对象，就从定义它的
runtime 子模块导入，不通过包级 `__init__.py` 聚合导出。

这些方向由 `import-linter` 强制：

```bash
uv run lint-imports
```
