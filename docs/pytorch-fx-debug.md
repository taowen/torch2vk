# PyTorch FX Graph 对拍

本文记录当前 `torch2vk` 使用 `torch.export` / FX graph 做 reference 的方式，以及它和手写 callable
reference 的边界。

## 目标模型

Vulkan shader 的边界由 shader contract 强制定义；PyTorch `nn.Module` 边界只是代码组织方式，两者不保证
一一对应。对拍应绑定到导出阶段可确认的 tensor graph：

```text
ExportedProgram / FX node output
  -> ReferenceSpec.output_bindings
  -> LogicalTensor field path
  -> generated reference.run_xxx(...)
```

## `.pt2` reference program

对于无外部状态的子图，export 阶段保存 `.pt2`：

```python
ReferenceSpec(
    program="reference_programs/decode_norm.pt2",
    tensors="model_tensors().decode_norm",
    name="spike.decode.{step:04d}.norm",
    policy="tensor",
    input_bindings={"hidden_states": "hidden_states"},
    output_bindings={"mul_1": "mul_1"},
)
```

runtime debug 通过 `load_exported_reference(base_dir, spec, state_dict=...)` 加载：

```python
ref = load_exported_reference(base_dir, reference_specs.TEXT_NORM_SPEC, state_dict=norm.state_dict())
expected = ref.execute({"hidden_states": hidden_np})
```

`ExportedProgramReference` 使用 `torch.fx.Interpreter` 执行 `ep.graph_module`，并捕获每个 tensor node 的输出。
因此 expected key 是 FX node name，例如 `"mul_1"` 或 `"linear"`。

`.pt2` 是生成产物，路径写在 `reference_specs.py`，文件本身不提交到 git。

## 显式 callable reference

有状态或暂时无法导出为单个 graph 的逻辑使用显式 callable reference，由模型 `run.py` 推进状态：

```text
Qwen3-ASR text layer: PyTorch layer + DynamicCache
Qwen3-ASR token store: numpy reference state
OmniVoice LLM step: tensor module + running tokens
OmniVoice token score/update: tensor-only PyTorch module
```

这类 reference 仍然通过同一个 `ReferenceSpec.output_bindings` 对拍。区别只是 expected 由显式 callable
计算，而不是 `ExportedProgramReference.execute()` 计算。调用点仍然走生成的 `reference.run_xxx(...)`。

## 生成阶段保存什么

export/codegen 负责生成：

1. `LogicalTensor.reference_key`：记录 tensor 来自哪个 FX node；
2. `reference_specs.py`：记录 reference input/output key、tensor root、compare name 和 policy；
3. `reference_setup.py`：生成 `.pt2` graph reference loader；
4. `reference.py`：生成 `run_xxx(...)` wrapper，统一执行 reference 和 compare；
5. 可选 `.pt2`：保存无状态 exported graph reference。

运行时不再从 tensor name 猜 reference key，也不把 compare/probe metadata 放到另一套表里。

## 同步推进

长流程对拍必须同步推进 Vulkan 和 PyTorch：

```text
for step in generation:
    run Vulkan subgraph(s)
    run generated reference.run_xxx(...) with PyTorch state
    update PyTorch reference state
```

OmniVoice 的 32 步 masked decoding 和 Qwen3-ASR 的 prefill/decode 都按这个方式工作。这样后续步骤的累计漂移
不会被单步 wrapper 掩盖。

## 不做的事

```text
不为了 debug 拆 Runtime frame 语义。
不把 PyTorch module boundary 当作默认对拍边界。
不维护手写 reference graph 来平行复制 FX graph。
不让 replay builder 理解自回归 token 反馈等模型业务。
```
