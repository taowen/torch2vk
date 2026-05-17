# PyTorch 对拍调试

当前对拍采用 PyTorch 主流程的 streaming compare：模型 `compare.py` 先执行同粒度 PyTorch
reference，拿到 expected；再调用生成的 `reference.compare_xxx(...)` 或直接调用
`compare_vulkan_stage(...)`，在同一个阶段内运行 Vulkan candidate 并比较输出。

## 核心规则

```text
PyTorch reference: compare.py 显式推进，是长流程的主控。
Vulkan candidate: compare_vulkan_stage() 在 rt.request/rt.frame 内运行一个 stage。
绑定关系: export 生成 reference.compare_xxx(...)，内联 input_bindings/output_bindings。
比较动作: compare_vulkan_stage() 调 compare_expected()。
```

RuntimeSession 只负责 Vulkan dispatch、materialization、readback、compare artifact 和 replay 记录。它不安装
PyTorch hook，不推断 PyTorch forward 参数，也不维护另一套 compare registry。

## 生成代码

export 阶段会为可规则表达的 Vulkan stage 生成 `reference.py`：

```python
def compare_text_norm(rt, *, expected, hidden_states):
    return compare_vulkan_stage(
        rt,
        name="spike.text.norm",
        run=lambda: _dispatch_text_norm(rt),
        tensors=model_tensors().text_norm,
        input_bindings={"hidden_states": "hidden_states"},
        output_bindings={"mul_1": "mul_1"},
        policy=_policy("tensor"),
        inputs={"hidden_states": hidden_states},
        expected=expected,
    )
```

字段语义：

1. `run`: 当前 stage 的 Vulkan dispatch 函数。
2. `tensors`: compare 时作为 root 的 generated tensor 对象。
3. `name`: compare artifact/frame 名称，可以包含 `{step}`、`{layer_idx}`。
4. `policy`: `"tensor"`、`"token"`，或按 output key 指定的 policy dict。
5. `input_bindings`: reference input key 到 `LogicalTensor` 字段路径的映射。
6. `output_bindings`: expected output key 到 `LogicalTensor` 字段路径的映射。

不规则 stage 可以在 `compare.py` 里直接调用 `compare_vulkan_stage(...)`。例如 token score/update、LM head
argmax、optimized fused tail 这类逻辑本来就不是单个 exported graph。

## 长流程状态

生成式模型的对拍不再以 Vulkan readback 推进 PyTorch。正确方向是：

```text
1. PyTorch reference 执行当前 stage，得到 expected。
2. Vulkan candidate 执行同一个 stage。
3. compare_vulkan_stage() 比较输出，并把 expected 按 Vulkan LogicalTensor dtype/shape 规范化后返回。
4. compare.py 用返回值推进下一段 PyTorch reference。
```

这样既能保持 PyTorch 主流程完整，又能让 dtype/shape 与 Vulkan 阶段一致。如果某一段 Vulkan 输出错误，会在该段
对拍中暴露；如果量化误差需要进入下一段，下一段使用的也是已规范化的 stage output。

## 输入与权重

入口输入应来自同一份业务数据。stage 内的中间输入由 `compare.py` 显式传给 `compare_vulkan_stage()`，不再由
Runtime 从 frame 自动推断。

权重必须对齐到同一 checkpoint 语义：

```text
Vulkan: LogicalTensor.checkpoint_key -> checkpoint/gguf tensor
PyTorch: compare.py 里的 reference module state_dict
```

如果 Vulkan 路径使用量化权重，PyTorch reference 仍用原始权重，容差必须表达这个差异；不能在 runtime 内静默
reshape、transpose 或 cast 来掩盖问题。

## Artifact 和失败报告

`compare_expected()` 会：

1. 用 `output_bindings` 找到 Vulkan `LogicalTensor`；
2. readback candidate；
3. 将 expected 转成 numpy；
4. 按 `ComparePolicy` 比较；
5. mismatch 时写入 `.cache/torch2vk/generated/debug/...` 的 candidate、expected 和 summary。

失败报告必须能定位 frame name、logical tensor、writer shader、shape/dtype、最大误差和首个 mismatch。

## 不做的事

```text
不在 RuntimeSession 内安装 PyTorch hook。
不让 frame 自动推断 PyTorch forward kwargs。
不维护和 LogicalTensor tree 平行的 compare registry。
不生成单独执行 PyTorch 的 run wrapper。
不在 replay builder 中硬编码 token feedback 等业务语义。
不在 compare 阶段 silent reshape / transpose / cast 来掩盖差异。
```
