# 直接导出上游 PyTorch Module

核心原则：**export 的输入必须是上游 `torch.nn.Module`，不做任何包装或改写。**

如果需要自定义 `torch.nn.Module` 来适配导出，说明框架有问题，不是模型有问题。

## 实验结果（2026-05-08）

对 `Qwen3ASRForConditionalGeneration` 上游模型直接调用 `torch.export.export(module, args, strict=False)`:

| 导出目标 | 结果 | 节点数 | 失败原因 |
|---------|------|--------|---------|
| `audio_tower.layers[0]` | **成功** | 28 ops | — |
| `model.layers[0]` (text, past_key_values=None) | **成功** | 77 ops | — |
| `model.layers[0]` (text, past_key_values=StaticCache) | 失败 | — | StaticCache 不是 pytree 类型 |
| `model` (全量 text model) | 失败 | — | rotary_emb 里 torch.autocast 对 meta device 不兼容 |
| `audio_tower` (完整) | 失败 | — | `.item()` 造成 GuardOnDataDependentSymNode |

## 可直接导出的上游 Module

```python
# audio encoder layer — 直接从上游取
layer = model.thinker.audio_tower.layers[0]
prog = torch.export.export(layer, (hidden_states, cu_seqlens),
    kwargs={"attention_mask": mask}, strict=False)
# → 28 call_function nodes, 16 parameters

# text decoder layer — 直接从上游取
layer = model.thinker.model.layers[0]
prog = torch.export.export(layer, (hidden_states, position_embeddings),
    kwargs={"past_key_values": None, "attention_mask": None}, strict=False)
# → 77 call_function nodes, 11 parameters
```

这些就是 `optimized_qwen3_asr` 对拍框架所对齐的同一批上游 Module：
- `pytorch_model_submodule="thinker.audio_tower"` → 对拍整个 audio tower
- `pytorch_model_submodule="thinker"` → 对拍整个 text model

## 不能导出的部分及其原因

### Audio tower 的动态前向

`Qwen3ASRAudioEncoder.forward()` 第 725 行：
```python
chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
chunk_lengths = torch.tensor([self.n_window * 2] * int(chunk_num.sum().item()), ...)
```

`.item()` 将 tensor 值提取为 Python int 用于控制流 → `torch.export` 无法处理。这是上游为 eager mode 写的代码，不可修改。

**解决方式**：audio tower 的编排逻辑（pad → conv → compact → layer loop → proj）在 runtime 层实现，不通过 export。`optimized_qwen3_asr/audio_tower.py` 就是这个模式 — 直接调用 shader 序列，对拍验证正确性。

### KV Cache

`torch.export` 不支持 `StaticCache` 作为输入（不是 pytree 类型）。但 KV cache 本就不属于模型 forward 的计算图 — 它是推理时的状态管理。

**解决方式**：export 层拿到的是 `past_key_values=None` 的纯计算图（attention 只用当前 step 的 q/k/v）。KV cache 的 write (index_copy) 和 decode-phase 的 cache read 作为 runtime 额外插入的操作。这和 `optimized_qwen3_asr` 的做法一致 — `QWEN3_ASR_TEXT_KV_CACHE_WRITE_F32` 是独立的 shader，不在模型 forward 里。

### 全量 text model（所有层 unroll）

`Qwen3ASRThinkerTextModel.forward()` 里的 `rotary_emb` 使用了 `torch.autocast`，对 meta device 不兼容。如果用真实 device（需要 GPU 内存加载所有权重），有可能成功 — 但未验证。

此外，28 层 unroll 后 FX graph 会有 ~2000 个 op node、~308 个 parameter。实用性存疑。

**当前做法**：export 单层，runtime 循环调用。这也是 `optimized_qwen3_asr` 的做法。

## 架构对齐：export vs optimized

| 维度 | `optimized_qwen3_asr` | `exported_qwen3_asr` (目标) |
|------|----------------------|---------------------------|
| 对拍目标 | `thinker.audio_tower`, `thinker` | 同 |
| 模块边界 | 手动对齐上游 Module 结构 | torch.export 自动从上游 Module 获取 |
| shader 来源 | 手写 GLSL（fused, 高性能） | 从 aten op 1:1 自动生成（正确性优先） |
| tensor 连接 | 手动在 dispatch 函数中 wire | FX graph 内部自动提供 |
| KV cache | 独立 shader (kv_cache_write) | runtime 层处理，不在 export 图中 |
| 验证方式 | 运行上游 Module 对拍输出 | 同 |

## 不应该做的事

1. **不要定义自定义 `torch.nn.Module`**
   - ~~`_AudioConvStack`~~, ~~`_AudioProj`~~, ~~`_TextLayerPrefillWithCache`~~ — 全部删除
   - export 的输入只能是 `model.xxx` 链条上直接取到的上游 Module

2. **不要把 KV cache 塞进 export**
   - `past_key_values=None` 是正确的 export 调用方式
   - cache write/read 是 runtime 职责

3. **不要强行 export 有动态控制流的 forward**
   - audio tower 的 pad/compact/chunking 就不是 export 能处理的
   - 这些逻辑用 runtime 编排（和 `optimized_qwen3_asr` 一样）

## export 的实际价值

export 有价值的场景：**一个 Module 内部有大量 op 需要按顺序 dispatch，且它们之间的 tensor 连接是由 FX graph 自动提供的**。

- text decoder layer: 77 个 op，11 个参数，内部复杂的 RoPE/attention/MLP 连接 — 适合 export
- audio encoder layer: 28 个 op，16 个参数 — 适合 export
- `nn.Conv2d`: 1 个 op — 不值得 export 的开销，直接调 shader
- `nn.Linear`: 1 个 op — 同上

## 导出后得到什么

```python
prog = torch.export.export(layer, args, kwargs, strict=False)
```

FX graph 提供：
1. **参数路径** — `graph_signature.input_specs` 给出 `self_attn.q_proj.weight` 等
2. **op 序列** — 按执行顺序排列的 aten 调用
3. **tensor 连接** — node A 的输出是 node B 的输入（同一个 FX Node）
4. **shape/dtype** — `node.meta["tensor_meta"]` 每个中间 tensor 的精确形状
5. **模块层级** — `node.meta["nn_module_stack"]` 每个 op 属于哪个子模块

codegen 从这些信息自动生成：
- `TensorClass`（所有参数 + 中间 tensor 的 dataclass 声明）
- `dispatch_function`（按序调用 shader 的 Python 函数）
- `shader` 文件（GLSL compute shader）

## 当前 shader 覆盖（18 种 aten op）

text decoder layer 的 77 个 op 使用以下 aten 原语：

| aten op | 出现次数 | shader |
|---------|---------|--------|
| `aten.linear.default` | 7 | linear_nobias / linear_bias |
| `aten.mul.Tensor` | 13 | mul (4 种 broadcast) |
| `aten.add.Tensor` | 8 | add / add_scalar |
| `aten.pow.Tensor_Scalar` | 4 | pow_scalar |
| `aten.mean.dim` | 4 | mean_dim |
| `aten.rsqrt.default` | 4 | rsqrt |
| `aten.silu.default` | 1 | silu |
| `aten.neg.default` | 2 | neg |
| `aten.cat.default` | 2 | cat |
| `aten.slice.Tensor` | 4 | slice |
| `aten.scaled_dot_product_attention` | 1 | sdpa (4 种变体) |
| `aten.transpose.int` | 4 | transpose / alias |
| `aten.to.dtype` | 8 | alias (same dtype) |
| `aten.view/reshape/unsqueeze/contiguous` | 6 | alias (zero-copy) |

audio encoder layer 额外需要：
| aten op | shader |
|---------|--------|
| `aten.layer_norm.default` | layer_norm |
| `aten.gelu.default` | gelu |

audio tower 编排额外需要：
| aten op | shader |
|---------|--------|
| `aten.conv2d.default` | conv2d |
| `aten.embedding.default` | embedding |
| `aten.index_select.default` | index_select |
| `aten.index_copy.default` | index_copy (KV cache write) |
