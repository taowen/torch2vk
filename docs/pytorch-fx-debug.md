# PyTorch FX Graph 对拍

本文记录当前 `torch2vk` 使用 `torch.export` / FX graph 确定 reference binding 的方式，以及它和手写 callable
reference 的边界。运行时对拍直接调用当前 PyTorch 模型里的 submodule，不保存独立 graph artifact。

## 目标模型

Vulkan shader 的边界由 shader contract 强制定义；PyTorch `nn.Module` 边界只是代码组织方式，两者不保证
一一对应。对拍应绑定到导出阶段可确认的 tensor graph：

```text
ExportedProgram / FX node output
  -> generated reference.py output_bindings
  -> LogicalTensor field path
  -> generated reference.run_xxx(...)
```

## 直接 module reference

对于无外部状态的子图，export 阶段只保存 input/output binding，不再保存独立 reference graph 文件：

```python
reference.run_decode_norm(rt, hidden_states=hidden_np)
```

生成的 `reference.py` 在运行时从当前 PyTorch 模型 lazy load 对应 submodule：

```python
_load_decode_norm() == _require_model().get_submodule("thinker.model.norm")
```

`reference.run_xxx(...)` 负责把 numpy 输入转成 CUDA tensor，调用这个 submodule，再按 export 阶段生成的
`output_bindings` 和 Vulkan 输出比较。权重仍然来自同一个已经加载的 PyTorch 模型，不复制到独立 reference artifact。

## 显式 callable reference

有状态或暂时无法导出为单个 graph 的逻辑使用显式 callable reference，由模型 `run.py` 推进状态：

```text
Qwen3-ASR text layer: PyTorch layer + DynamicCache
Qwen3-ASR token store: numpy reference state
OmniVoice LLM step: tensor module + running tokens
OmniVoice token score/update: tensor-only PyTorch module
```

这类 reference 仍然通过生成的 `reference.py` 内联 `output_bindings` 对拍。区别只是 expected 由显式 callable
计算，而不是直接调用一个无状态 submodule。调用点仍然走生成的 `reference.run_xxx(...)`。

## 生成阶段保存什么

export/codegen 负责生成：

1. `LogicalTensor.reference_key`：记录 tensor 来自哪个 FX node；
2. `reference.py`：生成 `run_xxx(...)` wrapper，并内联 input/output key、tensor root、compare name、policy 和 lazy submodule loader。

运行时不再从 tensor name 猜 reference key，也不把 compare metadata 放到另一套表里。

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
