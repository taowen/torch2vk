# PyTorch 对拍调试

本文定义当前 `torch2vk` 的 PyTorch/Vulkan lockstep 对拍方式。对拍边界不再绑在
`rt.frame(...)` 的 PyTorch hook 上，而是由模型运行代码显式推进 reference 计算，再用生成阶段产出的
`ReferenceSpec` 绑定 Vulkan 输出。

## 核心规则

```text
Vulkan candidate: rt.frame(...) 内按 dispatch 顺序执行 shader。
PyTorch reference: 生成的 reference_setup.py 创建 reference 对象，run.py 调用生成的 reference.run_xxx(...)
同步执行同粒度 tensor reference。
绑定关系: ReferenceSpec.output_bindings 把 reference output key 映射到 LogicalTensor 字段。
比较动作: 生成的 reference.py 调用 compare_expected_with_spec(...)。
```

RuntimeSession 只负责 shader dispatch、materialization、readback、compare artifact 和 replay 记录。它不安装
PyTorch hook，不推断 PyTorch forward 参数，也不维护另一套 probe registry。

## ReferenceSpec

`ReferenceSpec` 由 export 阶段生成：

```python
ReferenceSpec(
    program="reference_programs/text_norm.pt2",
    tensors="model_tensors().text_norm",
    name="spike.text.norm",
    policy="tensor",
    input_bindings={"hidden_states": "hidden_states"},
    output_bindings={"mul_1": "mul_1"},
)
```

字段语义：

1. `program`: 可选 `.pt2` 路径。有值表示可以用 `load_exported_reference()` 加载 torch.export graph reference；
   `None` 表示调用方用显式 PyTorch callable 计算 expected。
2. `tensors`: compare 时作为 root 的 generated tensor 表达式。
3. `name`: compare artifact/frame 名称，可以包含 `{step}`、`{layer_idx}` 或 `{name}`。
4. `policy`: `"tensor"`、`"token"`，或按 output key 指定的 policy dict。
5. `input_bindings`: reference input key 到 tensor dataclass 字段路径的映射，也是生成 wrapper 参数的来源。
6. `output_bindings`: reference output key 到 tensor dataclass 字段路径的映射，是实际对拍依据。

## 对拍入口

`torch2vk.export.render_reference_setup_module()` 生成 `reference_setup.py`，负责把已加载的 PyTorch 模型转成
对拍所需的 reference 对象。`torch2vk.export.render_reference_module()` 根据 `ReferenceSpec` 生成
`reference.py`。模型 `run.py` 在 Vulkan dispatch 后调用生成 wrapper：

```python
dispatch.run_text_norm(rt)
expected = reference.run_text_norm(
    rt,
    refs.text_norm,
    hidden_states=hidden,
)
```

多输出 shader 可以按 output key 指定不同 policy：

```python
ReferenceSpec(
    program=None,
    tensors="model_tensors()",
    name="omnivoice.step.{step:04d}.token_score",
    policy={"candidate_tokens": "token", "candidate_scores": "tensor"},
    input_bindings={...},
    output_bindings={...},
)
```

## 有状态 reference

生成式模型必须由调用方显式维护 PyTorch state。例如 Qwen3-ASR decode reference 持有 `DynamicCache`，
OmniVoice masked decoding reference 持有 `tokens` 和 `batch_input_ids`。Vulkan 每执行一步，PyTorch
reference 也执行同一步，并用同一个 token/cache 状态推进。

这能覆盖长流程累计漂移：如果前面某步 token update 出错，下一步 PyTorch reference 会在自己的正确状态上继续，
后续对拍会在对应子图暴露 mismatch。

## 输入与权重

输入应来自同一份业务数据。模型运行代码负责把 numpy/torch 输入同时喂给 Vulkan 和 PyTorch reference。Runtime
不会接受第二套隐藏输入，也不会在 frame exit 自动调用 PyTorch。

权重必须对齐到同一 checkpoint 语义：

```text
Vulkan: LogicalTensor.checkpoint_key -> checkpoint tensor
PyTorch: loaded module state_dict or .pt2 lifted parameters
```

如果 Vulkan 路径使用 bf16 checkpoint 而 PyTorch reference 用 fp32 原始权重，token argmax 这类离散输出可能漂移。
调用方需要在 compare setup 阶段显式做同等 dtype rounding。

## Artifact 和失败报告

`compare_expected_with_spec()` 会：

1. 用 `ReferenceSpec.output_bindings` 找到 Vulkan `LogicalTensor`；
2. readback candidate；
3. 将 expected 转成 numpy；
4. 按 `ComparePolicy` 比较；
5. mismatch 时写入 `.cache/torch2vk/generated/debug/...` 的 candidate、expected 和 summary。

失败报告必须能定位 frame name、logical tensor、writer shader、shape/dtype、最大误差和首个 mismatch。

## 不做的事

```text
不在 RuntimeSession 内安装 PyTorch hook。
不让 frame 自动推断 PyTorch forward kwargs。
不维护和 LogicalTensor tree 平行的 probe registry。
不在 replay builder 中硬编码 token feedback 等业务语义。
不在 compare 阶段 silent reshape / transpose / cast 来掩盖差异。
```
