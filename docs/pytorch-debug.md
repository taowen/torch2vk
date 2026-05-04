# PyTorch 对拍调试

本文定义 `torch2vk` 如何和 PyTorch reference 做 lockstep 对拍。核心边界延续
`docs/frame-and-logical-tensor.md`：

```text
Vulkan candidate 由 Frame 内的 shader eager execution 产生。
PyTorch reference 由同一个 Frame 传入的 pytorch_model.forward 产生。
RuntimeSession 根据 candidate 实际写出的 LogicalTensor 决定比较什么。
模型 adapter 只声明 probe/compare metadata，不手写 compare，不手写 PyTorch 替代公式。
```

## 目标

PyTorch 对拍不是单独跑一遍完整模型再人工比输出，而是服务于 shader 迁移过程中的逐层定位：

1. 每个 frame 对齐一次 PyTorch `model.forward` 边界；
2. frame 内 Vulkan shader 先按 adapter 写好的 eager 顺序执行；
3. `RuntimeSession` 收集本 frame 的 dispatch records 和 written tensors；
4. frame 退出时先选择少量边界 tensor 作为本次 active compare target；
5. 临时在当前 `pytorch_model` 上安装 hook/probe；
6. lockstep 执行当前 PyTorch `model.forward`；
7. readback active candidate tensor，与 hook 捕获的 PyTorch artifact 做数值比较；
8. mismatch 后沿 writer dispatch 的 read tensors 自动 drilldown，按需 readback 已声明可对拍的直接输入；
9. mismatch 报告必须能定位到 frame、logical tensor、writer shader 和 dispatch index。

这样可以先从一个 kernel、一个 block、一个 frame 开始对拍，再逐步扩大到完整 pipeline。

## 一句话规则

```text
candidate 写什么，RuntimeSession 才可能把什么选为 active compare 或 drilldown target。
PyTorch artifact 从当前 frame 的真实 PyTorch forward 捕获。
LogicalTensor 只声明怎么捕获和怎么比较，不代表每次都 readback。
RuntimeSession 决定本次 active target、执行 hook、readback、compare 和报告。
```

禁止反过来让 PyTorch model 决定 compare targets。也禁止在 debug helper 里重写一份 candidate
公式作为 reference。reference 必须来自当前 frame 传入的 PyTorch model 的真实执行过程。

## 对拍入口

每个模型 adapter 的具体 frame 文件负责进入 `rt.frame(...)`，并显式接收与这个 frame 对齐的
PyTorch module 或 callable：

```python
def run_audio_codec_decoder_frame(
    rt: RuntimeSession,
    tensors: AudioCodecDecoderTensors,
    *,
    pytorch_model: torch.nn.Module,
) -> AudioCodecDecoderOutput:
    with rt.frame(
        "audio_codec_decoder",
        pytorch_model=pytorch_model,
    ):
        AUDIO_CODEC_DECODER_QUANTIZER_EMBED_SUM_F32(
            rt,
            codes=tensors.codes,
            embed=tensors.quantizer_embed,
            output=tensors.quantizer_sum,
        )
        AUDIO_CODEC_DECODER_CONV1D_K7_F32(
            rt,
            x=tensors.quantizer_sum,
            weight=tensors.decoder.conv_in.weight,
            bias=tensors.decoder.conv_in.bias,
            output=tensors.decoder.conv_in.output,
        )
        return AudioCodecDecoderOutput(waveform=tensors.waveform)
```

这里的 `pytorch_model` 是 reference provider。`ShaderVariant.__call__` 只调用 `rt.dispatch(...)`，不读写
PyTorch，也不做 compare。

PyTorch model.forward 的输入不由 frame function 额外传 `pytorch_input`、`pytorch_args` 或
`pytorch_kwargs`。RuntimeSession 只使用 `rt.register_inputs({logical_tensor: value})` 中的输入，并按规则
推断 PyTorch kwargs：

```text
当前 frame name: qwen3_asr.audio_tower
LogicalTensor.name: qwen3_asr.audio_tower.input_features
PyTorch forward 参数: input_features
=> pytorch_model.forward(input_features=<registered value>)
```

如果某个 input tensor 的 logical name 属于当前 frame，但 basename 不在 PyTorch forward 签名里，它不会传给
PyTorch；这允许迁移阶段保留 Vulkan 暂时消费的中间态输入，例如 `padded_feature`。后续实现上游
padding/chunk shader 后，再让 Vulkan 改读和 PyTorch 相同的 input tensor。

toy MVP 可以先用 `reference_model` manual callable：

```python
with rt.frame(
    "toy.elementwise_mul",
    reference_model=toy_reference_model,
):
    ELEMENTWISE_MUL_F32(rt, x=x, weight=w, output=y)
```

manual callable 只适合最小链路验证。真实模型迁移必须使用 `pytorch_model` + `PyTorchProbe`。

## LogicalTensor 需要声明什么

可参与对拍的 candidate tensor 必须同时声明：

1. `compare`：数值如何比较；
2. `pytorch_probe`：PyTorch forward 中到哪里捕获 reference artifact；
3. `semantic`：可选，只服务 debug 报告，不参与执行规则。

示例：

```python
hidden = LogicalTensor(
    name="audio_codec_decoder.decoder.block0.res_unit0.output",
    spec=TensorSpec(dtype="float32", shape=("B", "C", "T")),
    role=TensorRole.ACTIVATION,
    memory=MemoryClass.FRAME_WORKSPACE,
    lifetime=TensorLifetime.FRAME,
    compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
    pytorch_probe=PyTorchProbe(
        kind="module_output",
        target="decoder.blocks.0.res_units.0",
        index=0,
    ),
)
```

没有 `compare` 的 tensor 不能作为数值对拍点。没有 `pytorch_probe` 的 tensor 在真实 PyTorch model 路径下也不能
自动取得 reference artifact，因为 runtime 不知道 reference artifact 来自哪里。

声明 `compare/pytorch_probe` 只表示这个 tensor 可被对拍，不表示每次 frame exit 都要 readback。RuntimeSession
会先比较默认边界 target；只有当该 target mismatch 并且 writer drilldown 走到某个已声明 tensor 时，才按需
readback 这个 tensor。

## PyTorchProbe

推荐结构：

```python
@dataclass(frozen=True, slots=True)
class PyTorchProbe:
    kind: Literal["module_input", "module_output", "manual_hook", "derived"]
    target: str
    index: int = 0
    selector: str | None = None
    transform: str | None = None
```

字段含义：

```text
kind
  module_output 表示捕获 target module 的输出。
  module_input 表示捕获 target module 的输入。
  manual_hook 表示 adapter 提供一个很薄的 hook 点，用于 official 模型结构不方便按 module 名字定位的情况。
  derived 表示从已捕获 artifact 做边界变换，例如 transpose、reshape、slice、dtype normalization。

target
  PyTorch module path 或 adapter 定义的 hook target。它必须能在当前 frame 的 pytorch_model 内解析。

index
  当 input/output 是 tuple/list 时选择第几个元素。普通 tensor 通常为 0。

selector
  对复杂返回值做字段选择，例如 "hidden_states"、"logits"。selector 只做取值，不实现计算。

transform
  只允许表达 Vulkan/PyTorch 边界差异，例如 layout 转换、去 padding、cast 到 float32 后比较。
```

`derived` 和 `transform` 不能重新实现模型 op。比如 `permute_nchw_to_nhwc`、`slice_valid_tokens` 是可以的；
`conv1d_again`、`attention_again`、`gelu_again` 是不允许的。

## ComparePolicy

推荐结构：

```python
@dataclass(frozen=True, slots=True)
class ComparePolicy:
    kind: Literal["tensor", "token", "waveform"]
    rtol: float = 1e-4
    atol: float = 1e-4
    max_abs: float | None = None
```

默认 tensor 比较使用：

```text
abs(candidate - expected) <= atol + rtol * abs(expected)
```

不同输出可以使用不同 policy：

```text
float32 accumulation kernel
  rtol=1e-5, atol=1e-6 起步。

float16/bfloat16 shader 或 mixed precision 路径
  rtol=1e-3, atol=1e-3 起步，再按误差来源收紧。

token / argmax
  先比较 token id 完全一致；不一致时附带比较 logits top-k 和 margin。

waveform
  不只看逐点误差，也可以报告 peak、RMS、SNR；但 frame 内定位仍应优先比较 decoder 中间 tensor。
```

如果一个 tensor 在 PyTorch 中是 `float16`/`bfloat16`，但 shader readback 是 `float32`，比较前可以按
`transform` 统一到 `float32`。这种 cast 必须发生在 compare 边界，不能掩盖权重加载或 shader contract
中的 dtype mismatch。

## Artifact key

每次 invocation 都有自己的 frame name 和 `LogicalTensor` tree。动态上下文直接进入 frame name，
不再有单独的动态上下文 map。

artifact key 推荐由两部分组成：

```text
<frame-name>/<logical-tensor-name>
```

示例：

```text
text_llm.decode.audio_token_005.cond/text_llm.decode.audio_token_005.cond.layer.03.output
audio_codec_decoder/audio_codec_decoder.decoder.waveform
toy.elementwise_mul/toy.y
```

这个 key 同时用于：

1. PyTorch artifact cache；
2. candidate readback 文件；
3. mismatch 报告；
4. 后续 replay/debug drilldown。

同一次 forward 需要执行多少次，就创建多少份带 invocation 身份的 frame/tensor tree。

PyTorch artifact cache 是磁盘缓存，不是 PyTorch model 决定 compare targets。RuntimeSession 仍然先根据
本 frame 实际 written tensors 选择少量 active targets；对 active target 和 drilldown target 用 artifact key 加上
model/checkpoint fingerprint、register_inputs 输入 fingerprint、probe target/selector/transform 和 tensor
shape/dtype/layout 生成 cache key。如果 expected artifact 已存在且 metadata 完全匹配，RuntimeSession 直接
读取缓存，不重跑 PyTorch forward；否则只对缺失 target 安装 hook，lockstep 执行一次当前 frame 的
`pytorch_model.forward`，然后把捕获到的 expected artifact 落盘。

缓存命中不改变正确性边界：只要输入、权重、probe 或 tensor metadata 变了，就必须 cache miss。缓存文件只是
真实 PyTorch forward 捕获结果的持久化副本，不允许由 adapter 手写 reference 公式生成。

## Frame 退出时的执行顺序

`RuntimeSession` 在 frame exit 做完整对拍：

```text
1. 停止收集 candidate dispatch records
2. 从 records 中收集本 frame written LogicalTensors
3. 从实际 written 且声明了 compare/probe 的 tensors 中选择默认 active target，通常是最后一个边界输出
4. 按 artifact cache key 查找 PyTorch expected artifact
5. 对 cache miss 的 target 根据 pytorch_probe 在 pytorch_model 上安装临时 hook
6. 用当前 frame 的 inputs/state lockstep 执行 pytorch_model.forward
7. hook 捕获 PyTorch artifact，并应用 selector/transform，然后写入磁盘缓存
8. readback candidate LogicalTensor 当前 buffer，并应用 readback transform
9. 检查 shape、dtype、layout
10. 按 ComparePolicy 计算 max_abs、max_rel、first mismatch
11. 如果 mismatch，沿 failed tensor.writer 的 dispatch.reads 自动选择已声明的直接输入做 drilldown compare
12. 输出 compare summary 或 mismatch report
13. 移除 PyTorch hooks
14. 释放或复用 FRAME/OP 生命周期资源
```

hook 必须是 frame-local 的。frame 退出后不保留 PyTorch hook，避免后续 frame 捕获到错误 artifact。

## Mismatch 报告

失败报告至少包含：

```text
frame name
artifact key
logical tensor name
candidate LogicalTensor / current buffer
writer shader
dispatch index
candidate shape/dtype/layout
PyTorch artifact shape/dtype/layout
ComparePolicy
max_abs
max_rel
first mismatch index
candidate value at first mismatch
expected value at first mismatch
```

推荐附带：

```text
top-k token/logit 差异
NaN/Inf 数量
candidate min/max/mean
expected min/max/mean
误差最大的前 N 个 index
上游最近的 compared tensor
```

定位时优先看第一个 mismatch，而不是最后输出。完整 pipeline 输出错了，通常要从 frame 内最近的
activation 对拍点向前二分：

```text
final output mismatch
  -> compare frame output
  -> compare block output
  -> compare op output
  -> compare shader input
  -> inspect weight/input layout
```

每个 `DispatchRecord` 是一次 shader 函数调用记录。即便 shader 是 fused kernel，也可以把它看作：

```text
shader(inputs, weights, push_constants, layout) -> outputs
```

对应的 PyTorch reference 可能是一个 module output、module input、root output，或由真实 PyTorch artifact
做有限边界 transform 后得到的聚合函数结果。定位到某个 failed tensor 后，RuntimeSession 应沿现有对象图倒查：

```text
failed LogicalTensor
  -> tensor.writer
  -> dispatch record
  -> dispatch.reads / dispatch.writes
  -> 对 read tensors 查看最近 compare result 或继续沿 read.writer 倒查
```

失败时应持久化当前 writer dispatch 的 `dispatch.json`、所有已 materialized read/write tensor dump，以及
`drilldown.json`。RuntimeSession 用迭代式 drilldown，不用层层递归异常表达路径：

```text
failed output
  -> dump writer IO
  -> compare declared direct reads
  -> direct reads passed: input_ok_output_bad
  -> direct read failed: continue from that read.writer
  -> direct read missing compare/probe: missing_reference_probe
```

如果当前 dispatch 的直接输入都已有通过的 compare，而 output mismatch，则可以把问题收敛到这一次 shader
调用或它的 PyTorch reference boundary；如果某个直接输入已声明 compare/probe，RuntimeSession 自动对它做
按需 readback 和 PyTorch cache/probe compare，并继续沿它的 writer 往上游倒查。没有声明 compare/probe
的非权重 tensor 会在报告中列为 `missing_reference_tensors`，不能自动数值对拍。权重 tensor 不要求声明
probe，权重正确性由 checkpoint key、dtype、shape、layout 和 binding 检查覆盖。

## 权重、输入和随机性

对拍必须保证 PyTorch reference 和 Vulkan candidate 使用同一份输入与权重：

1. 权重由 `LogicalTensor.name/spec/layout` 推断 checkpoint key、dtype、shape 和 layout，runtime 读取 checkpoint 并校验，不允许 silent cast；
2. 输入由 `register_inputs({logical_tensor: array})` 提供，PyTorch frame forward 使用同一批输入数组；PyTorch kwargs 由 RuntimeSession 按 logical name 和 forward 签名推断，不允许测试或 adapter 传第二份旁路输入；
3. dropout、采样、随机噪声必须关闭或固定 seed；
4. PyTorch model 应进入 `eval()`；
5. 生成式模型的 token 选择如果含随机采样，先对拍 logits，再对拍 deterministic selector；
6. 对拍 kernel 时使用小 shape 固定样例，扩大范围前先保证 op-level 误差可解释。

当前仓库已有 official PyTorch 入口：

```text
src/models/omnivoice/pytorch/example.py
src/models/qwen3_asr/pytorch/example.py
tests/test_omnivoice_qwen3_asr_roundtrip.py
```

这些入口适合验证上游 PyTorch 模型能跑通，以及端到端输入输出是否合理。它们不是 shader 对拍的替代品。
shader 对拍仍然要落到 frame 内的 `LogicalTensor.compare` / `PyTorchProbe`。

## Adapter 职责

模型 adapter 应该做：

1. 在 `tensors/` 中给需要调试的 LogicalTensor 增加 `compare` 和 `pytorch_probe`；
2. 在具体 frame 文件里传入正确的 `pytorch_model`；
3. 在 frame 内直接调用 `ShaderVariant`，不要再包一层 shader 函数；
4. 对 PyTorch/Vulkan layout 差异声明明确 transform；
5. 用 frame name 表达动态上下文，例如 `audio_token_index`、`row`、`chunk_index`。

模型 adapter 不应该做：

```text
不手写 readback
不手写 compare loop
不在模型目录里维护 artifact registry
不把 PyTorch hook 作为全局状态常驻
不在 debug helper 中重写 conv/attention/gelu 作为 reference
不把动态上下文写进 LogicalTensor.name
不为了通过比较而 silent cast、silent reshape 或 silent transpose
```

## RuntimeSession 职责

`RuntimeSession` 应该集中实现：

1. frame-local hook 安装和卸载；
2. PyTorch artifact 磁盘缓存，缓存 key 包含 frame/tensor、输入、模型和 probe metadata；
3. candidate readback；
4. dtype/shape/layout 校验；
5. compare policy 执行；
6. mismatch summary 和详细报告；
7. dispatch record 到 artifact key 的映射；
8. dispatch-level writer IO dump/reload 所需 metadata；
9. 沿 `LogicalTensor.writer` / `DispatchRecord.reads` 的倒查报告；
10. debug dump 开关，例如只 dump mismatch tensor 或 dump 某个 frame。

这样不同模型 adapter 可以复用同一套对拍能力。

## 推荐调试流程

迁移新 shader 或新 frame 时，按这个顺序推进：

1. 先声明最小输入、权重、输出 tensor，跑通 contract/materialization dry-run；
2. 给真实 PyTorch 边界存在的 LogicalTensor 声明 `compare` / `pytorch_probe`；
3. 优先保证 frame 最终输出和关键中间边界有声明；
4. 如果 output mismatch，让 RuntimeSession 自动沿 writer graph drilldown；
5. RuntimeSession 报 `missing_reference_probe` 时，再补对应 LogicalTensor 的 probe 声明；
6. 当前 shader 输入都通过而输出失败后，再检查 shader GLSL、push constants、layout 和 binding；
7. frame 通过后，再进入 pipeline 级输出或生成循环。

可以声明很多 compare/probe 点，因为声明本身不会触发 readback。RuntimeSession 默认只激活边界 target，mismatch
后沿 writer graph 自动按需激活直接输入。调试策略应是：

```text
关键边界作为默认 active target
mismatch 后按需打开上游 probe
定位完成后保留最有价值的 regression compare 点
```

## MVP 到完整实现

MVP 阶段可以先支持：

1. manual `reference_model`；
2. `module_output` hook；
3. tensor compare；
4. mismatch summary；
5. readback active candidate output，并在 mismatch 后按需 readback writer inputs。

随后再扩展：

1. `module_input`；
2. `selector` 和有限 `transform`；
3. token/logit top-k 报告；
4. waveform/SNR 报告；
5. artifact dump/reload；
6. replay plan 上的定点复现；
7. 自动二分最近 mismatch compare 点。

这些扩展都属于通用 runtime，不应该改变模型 adapter 的基本表达方式。
