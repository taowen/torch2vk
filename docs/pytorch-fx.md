# PyTorch FX Export 能够提供的信息

`torch.export.export` 对模型子模块导出后，FX graph 已经包含了 exportv2 中大量手写维护的信息。本文通过实验结果说明哪些信息可以直接从 PyTorch 模型获取。

## 1. 参数路径 (Parameter Paths)

`graph_signature.input_specs` 直接给出每个参数的模块路径：

```python
prog = torch.export.export(layer, (hidden_states,), kwargs=..., strict=False)
for spec in prog.graph_signature.input_specs:
    print(spec.kind, spec.target)
```

输出：

```
PARAMETER  self_attn.q_proj.weight
PARAMETER  self_attn.k_proj.weight
PARAMETER  self_attn.v_proj.weight
PARAMETER  self_attn.o_proj.weight
PARAMETER  self_attn.q_norm.weight
PARAMETER  self_attn.k_norm.weight
PARAMETER  mlp.gate_proj.weight
PARAMETER  mlp.up_proj.weight
PARAMETER  mlp.down_proj.weight
PARAMETER  input_layernorm.weight
PARAMETER  post_attention_layernorm.weight
USER_INPUT hidden_states
USER_INPUT position_ids
USER_INPUT position_embeddings_0
USER_INPUT position_embeddings_1
```

这与手写的 `_text_layer_parameter_sources()` 完全一致：

```python
def _text_layer_parameter_sources() -> dict[str, str]:
    return {
        "input_layernorm_weight": "input_layernorm.weight",
        "q_proj_weight": "self_attn.q_proj.weight",
        ...
    }
```

同样，顶层模型的 `named_parameters()` + `named_buffers()` 提供了 `_text_parameter_fields()` 中的全部信息：

```
llm.embed_tokens.weight      → 手写为 embed_tokens_weight
audio_embeddings.weight      → 手写为 audio_embeddings_weight
codebook_layer_offsets       → 手写为 codebook_layer_offsets (buffer)
llm.norm.weight              → 手写为 norm_weight
audio_heads.weight           → 手写为 audio_heads_weight
```

## 2. 模块边界 (Module Boundaries)

每个 FX node 的 `meta["nn_module_stack"]` 记录了产生该 node 的完整模块层级：

```python
for node in graph.nodes:
    if node.op == 'call_function':
        stack = node.meta.get('nn_module_stack', {})
        # stack 的最后一个 entry 是最内层的子模块
```

实验结果显示 77 个原始 op 可以按模块自动分组：

```
--- input_layernorm (Qwen3RMSNorm) ---
  to, pow, mean, add, rsqrt, mul, to, mul          # 对应手写的 aten.rms_norm.default

--- self_attn.q_proj (Linear) ---
  linear                                            # 对应手写的 aten.linear.default

--- self_attn.q_norm (Qwen3RMSNorm) ---
  to, pow, mean, add, rsqrt, mul, to, mul          # 对应手写的 aten.torch2vk.text_qk_norm.default

--- self_attn (Qwen3Attention) ---
  unsqueeze, mul, slice, neg, cat, mul, add         # rope for q
  unsqueeze, mul, slice, neg, cat, mul, add         # rope for k
  scaled_dot_product_attention                      # 对应手写的 aten.torch2vk.text_attention.default
  transpose, contiguous, reshape

--- self_attn.o_proj (Linear) ---
  linear                                            # 对应手写的 aten.linear.default

--- (Qwen3DecoderLayer) ---
  add                                               # residual, 对应手写的 aten.add.Tensor

--- post_attention_layernorm (Qwen3RMSNorm) ---
  to, pow, mean, add, rsqrt, mul, to, mul          # 对应手写的 aten.rms_norm.default

--- mlp.gate_proj (Linear) ---
  linear

--- mlp.act_fn (SiLUActivation) ---
  silu                                              # }
--- mlp (Qwen3MLP) ---                             # } 对应手写的 aten.torch2vk.text_swiglu.default
  mul                                               # }

--- mlp.down_proj (Linear) ---
  linear

--- (Qwen3DecoderLayer) ---
  add                                               # residual
```

手写的 18 个 `_text_layer_nodes()` 就是对上述 77 个原始 op 按模块边界做的人工聚合。

## 3. Tensor Shape 和 Dtype

每个 node 的 `meta["tensor_meta"]` 提供了精确的 shape 和 dtype：

```python
tensor_meta = node.meta.get("tensor_meta")
tensor_meta.shape  # e.g. torch.Size([1, 4, 1024])
tensor_meta.dtype  # e.g. torch.float32
```

## 4. 手写的 18 个节点 vs 导出的 77 个节点

| 手写 StaticNode | 对应的模块类型 | 导出的原始 op 数量 |
|---|---|---|
| `aten.rms_norm.default` (×4) | `Qwen3RMSNorm` | 每个 ~10 ops |
| `aten.linear.default` (×7) | `torch.nn.Linear` | 每个 1 op |
| `aten.torch2vk.text_qk_norm.default` (×2) | `Qwen3RMSNorm` (在 self_attn 内) | 每个 ~10 ops |
| `aten.torch2vk.text_rope.default` (×2) | `Qwen3Attention` 中的 rope 部分 | 每个 ~8 ops |
| `aten.torch2vk.text_attention.default` | `scaled_dot_product_attention` + reshapes | ~4 ops |
| `aten.torch2vk.text_swiglu.default` | `SiLUActivation` + `Qwen3MLP` 中的 mul | 2 ops |
| `aten.add.Tensor` (×2) | `Qwen3DecoderLayer` 顶层 residual | 每个 1 op |
| `aten.torch2vk.text_kv_cache_write.default` | **不存在于导出结果中** | runtime 概念 |

## 5. 真正需要额外提供的信息

以下信息无法从 PyTorch 模型导出中获得，是 Vulkan runtime 特有的：

| 信息 | 原因 |
|---|---|
| 模块类型 → fused op 名称的映射 | `Qwen3RMSNorm` → `aten.rms_norm.default` 是人为定义的聚合规则 |
| KV cache 读写 | 推理时的状态管理，不在模型 forward 中 |
| Shader 源码和绑定 | Vulkan compute shader 的实现细节 |
| Dispatch 策略 | workgroup size、tiling 等 GPU 执行参数 |

## 6. 新 torch2vk.export 的目标

基于以上实验，新的 `torch2vk.export` 模块要达成以下目标：

### 完成标准

1. **给定一个 PyTorch 子模块 + example inputs，自动导出可执行的 Vulkan dispatch 代码**
   - 调用 `export_submodule(layer, args, kwargs)` 即可得到完整计算图
   - 不需要手写 op 列表、参数映射、tensor field 声明

2. **每个 aten 原始 op 一一对应一个 shader**
   - 不做任何 fusion（rms_norm 拆为 pow+mean+rsqrt+mul，swiglu 拆为 silu+mul）
   - FX graph 输出什么 op，就 dispatch 什么 shader
   - Shape-only ops（view, unsqueeze, reshape, contiguous, transpose）做 buffer alias，不需要 shader

3. **以 qwen3_asr text decoder layer 为首个验证目标**
   - 导出一个完整的 Qwen3 decoder layer（77 个 aten ops）
   - 生成的 dispatch 函数可通过 RuntimeSession 执行
   - 参数路径与模型 `named_parameters()` 一致

4. **验证方式**
   - 对比导出的 op 序列覆盖所有 77 个 FX call_function nodes
   - 对比生成的参数路径与 `graph_signature.input_specs` 的 target 字段
   - 生成的代码能被 pyright 静态检查通过

### 需要实现的 aten shader（text decoder layer 所需）

| aten op | 出现次数 | 说明 |
|---|---|---|
| `aten.linear.default` | 7 | 矩阵乘法 |
| `aten.mul.Tensor` | 13 | elementwise 乘 |
| `aten.add.Tensor` | 8 | elementwise 加 |
| `aten.pow.Tensor_Scalar` | 4 | elementwise 幂（x²） |
| `aten.mean.dim` | 4 | 沿维度求均值（reduction） |
| `aten.rsqrt.default` | 4 | elementwise 1/sqrt |
| `aten.silu.default` | 1 | silu 激活 |
| `aten.neg.default` | 2 | elementwise 取负 |
| `aten.cat.default` | 2 | tensor 拼接 |
| `aten.slice.Tensor` | 4 | tensor 切片 |
| `aten.scaled_dot_product_attention.default` | 1 | 注意力 |
| `aten.embedding.default` | — | embedding lookup（后续扩展） |

### 非目标

- 不替代 exportv2 的全部功能（Jinja2 模板、generation loop 编排等）
- 不处理 KV cache（runtime 概念，不在 FX graph 中）
- 不做性能优化（fusion、tiling 策略等留给后续）

## 7. 实现进展与待解决问题

### 已完成

`src/torch2vk/export/` 包已实现核心流程，直接操作 PyTorch FX 对象（`torch.fx.Node`、`torch.export.ExportedProgram`），无中间表示：

```python
from torch2vk.export import export_submodule, generate_dispatch_source

prog = export_submodule(layer, args=(hidden_states,), kwargs={"position_embeddings": (cos, sin)})
source = generate_dispatch_source(prog, class_name="TextLayerTensors", function_name="run_text_layer")
```

- `graph.py` — `export_submodule()` 返回 `torch.export.ExportedProgram`，提供 `is_alias_op(node)` / `node_input_names(node)` 工具函数
- `shaders/` — 11 种 aten op 的 GLSL shader
- `registry.py` — `ShaderRegistry` 接收 `torch.fx.Node`，按 target 匹配 ShaderVariant
- `codegen.py` — 遍历 `prog.graph_module.graph.nodes` 和 `prog.graph_signature` 生成 Python dispatch 代码

验证结果：
- 69 个 FX ops 全覆盖（50 shader dispatch + 19 alias），0 TODO
- 参数路径与手写 `TEXT_DECODER_LAYER_PARAMETER_FIELDS` 100% 一致

### 已解决：ShaderVariant 按需生成

shader 从静态单例改为工厂函数。每个工厂根据 `node.meta["tensor_meta"].shape` 生成 contract shape 匹配实际 tensor rank 的 `ShaderVariant`：

```python
# shaders/mul_f32.py
def make_mul_variant(node: Node) -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    y_shape = node_input_shape(node, 1)
    # 根据 broadcast pattern 选择 GLSL + 生成匹配的 contract
```

验证结果：**50/50 compute ops 全部通过 `bind_shape_symbols` 验证**。

### 已解决：Broadcasting

4 种 GLSL variant 覆盖所有 mul broadcast 模式：

| 模式 | GLSL 索引 | contract 示例 |
|---|---|---|
| same_shape | `output[i] = x[i] * y[i]` | x=(1,"T","H"), y=(1,"T","H") |
| broadcast_last | `output[i] = x[i] * y[i / H]` | x=(1,"T","H"), y=(1,"T",1) |
| left_broadcast | `output[i] = x[i % H] * y[i]` | x=("D",), y=(1,"T","H","D") |
| broadcast_inner | `y_idx = (idx/STRIDE)/REPEAT*STRIDE + idx%STRIDE` | x=(1,"T","H","D"), y=(1,1,"H","D") |

### 已解决：Scalar 参数

`make_add_variant` 检查 `node.args[1]` 是否为 Node：
- 是 Node → 同 shape add（2 input tensor shader）
- 是 scalar → add_scalar shader（scalar 作为 push constant）

### 已验证：Vulkan 端到端执行 — 数值完全正确

- **pyright 0 errors**
- **单 op 测试（add）**：max diff = 0 ✓
- **RMS norm（6 ops: pow+mean+add_scalar+rsqrt+mul_broadcast+mul_left）**：max diff = 5.96e-08 ✓
- **RoPE（slice+neg+cat+mul+add）**：max diff = 0 ✓
- **SDPA（causal attention）**：max diff = 2.98e-08 ✓
- **完整 decoder layer with real weights（69 ops）**：max diff = 1.55e-06 ✓

### 阻塞项：Audio Tower 无法通过 torch.export 导出

audio tower 的 forward 中有 `chunk_num.sum().item()`（数据依赖的动态控制流），导致 torch.export 失败。但 **audio encoder layer** 可以单独导出（28 ops）。

audio tower 的 conv 预处理 + chunking 逻辑必须手写或复用 exportv2 的实现。

### 端到端转录的剩余工作

1. audio tower 编排（conv2d 预处理 + encoder layer loop + post-encoder projection）—— 需要手写或复用 exportv2
2. text prefill 编排（embedding + scatter audio features + decoder layer loop + lm_head）
3. text decode 编排（embedding + decoder layer with KV cache + lm_head）
4. generation loop（token select + token store + 循环控制）
5. 新增 shader：`aten.layer_norm.default`, `aten.gelu.default`, `aten.conv2d.default`, `aten.embedding.default`（audio tower 用）
