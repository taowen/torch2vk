# torch2vk：把 AI 模型搬进任意 GPU


---

# 第一部分：为什么需要 torch2vk

## 你的显卡，凭什么不能跑大模型？

你买了一台笔记本，里面有一颗 AMD 核显。你想在本地跑一个语音识别模型，把会议录音转成文字。你搜了一圈，发现几乎所有教程都写着：

> 请确保你有一张 NVIDIA 显卡。

为什么？AI 模型本质上就是一堆矩阵乘法和简单的数学运算，你的 AMD 显卡明明也能算，凭什么只有 NVIDIA 能跑？

答案是软件生态。NVIDIA 的 CUDA 平台从 2007 年起步，花了十几年建立起了对 AI 计算的垄断：PyTorch 默认用 CUDA 加速，模型发布时默认假设你有 CUDA，推理引擎（如 TensorRT）只给 NVIDIA 写。你的 AMD 显卡、Intel 集显、高通手机 GPU，硬件上完全有能力做同样的计算，但没有人给它们写完整的 AI 推理软件栈。

有没有一种跨厂商的 GPU 编程标准？有——**Vulkan**。

Vulkan 是 Khronos 组织（OpenGL 的维护者）在 2016 年推出的新一代图形和计算 API。和 CUDA 只支持 NVIDIA 不同，Vulkan 几乎在所有现代 GPU 上都有实现：

- AMD 显卡（包括核显）：通过 Mesa RADV 或 AMDVLK 驱动
- Intel 集显：通过 Mesa ANV 驱动
- 高通 Adreno（安卓手机）：厂商驱动
- 苹果 GPU：通过 MoltenVK 转译层
- 甚至纯 CPU 软件实现（lavapipe）都存在

Vulkan 1.0 开始就包含 **compute shader** 能力——不渲染图形，只做通用计算。这正是跑 AI 模型所需要的。

**torch2vk 就是来解决这个问题的。** 它把 PyTorch 模型翻译成 Vulkan compute shader，这样任何支持 Vulkan 的 GPU 都能跑推理。不需要 NVIDIA，不需要 CUDA，不需要等各家厂商分别适配。


## GPU 编程到底是怎么回事？

在理解 torch2vk 怎么工作之前，我们需要先搞清楚一件事：GPU 是怎么执行计算的？

### CPU vs GPU：两种完全不同的思路

CPU 擅长处理复杂的逻辑分支——if/else 套 if/else，各种跳转和递归。它的核心少（通常 4～16 个），但每个核心很强，能快速处理各种不同的任务。

GPU 则完全相反。一颗现代 GPU 有成百上千个小核心，每个核心都比 CPU 核心弱得多，但它们可以同时做同一件事。打个比方：

- **CPU** 像一个数学教授，一个人解一道复杂的微分方程。
- **GPU** 像一千个小学生，每人算一道简单的加法，一千道题一秒钟同时出结果。

AI 模型恰好就是"一千道简单加法"的类型——矩阵乘法里的每一个乘法-累加是独立的，激活函数对每一个元素是独立的，归一化里的每一行是独立的。所以 GPU 天然适合跑 AI 推理。

### Compute Shader：告诉 GPU 每个工人干什么

那怎么让 GPU 做计算呢？你需要写一个叫 **compute shader** 的小程序。这段程序不是给 CPU 的——它会被同时复制到 GPU 的上千个核心上执行。每个核心通过一个内置变量知道自己是"第几号工人"，然后到对应的位置读取数据、计算、写回结果。

以 SiLU 激活函数为例——对每个数 v 计算 `v / (1 + e^(-v))`——写成 Vulkan compute shader 只需要几行：

```glsl
void main() {
    uint idx = gl_GlobalInvocationID.x;  // 我是第几号工人？
    if (idx < N) {
        float v = x[idx];                // 从输入数组取出我负责的那个数
        output[idx] = v / (1.0 + exp(-v)); // 算完放回输出数组
    }
}
```

上千个 GPU 核心同时执行这段完全相同的代码，每个核心通过 `gl_GlobalInvocationID.x` 知道自己负责第几个元素，各算各的，互不干扰。这就是 GPU 并行计算的核心思想——术语叫 **SIMT**（Single Instruction, Multiple Threads）。

### Workgroup 和 Dispatch：组织工人

GPU 不是把一千个核心直接撒出去。它把工人分成小组（**workgroup**），每组通常 64 或 256 个工人。同组的工人可以通过**共享内存（shared memory）**交换数据，这比去主显存读写快几十倍。

比如矩阵乘法的 shader，会把矩阵的一小块（tile）先加载到共享内存里，让组内的工人都从共享内存读取，避免每个工人各自去显存读一遍。

**Dispatch** 就是 CPU 告诉 GPU "我要启动多少组工人"的命令。比如一个 1024 维的向量要做 SiLU，每组 256 个工人，那就 dispatch 4 个 workgroup，总共 1024 个工人，每人处理一个元素。

这就是 GPU 编程的全貌：**写一段 shader 描述每个工人的任务，然后 dispatch 足够多的 workgroup 来覆盖所有数据。** 接下来的问题是：怎么从一个 PyTorch 模型自动生成这些 shader？



---

# 第二部分：从 PyTorch 到 Vulkan 的翻译过程

## 第一步：torch.export 把模型摊平

### PyTorch 模型长什么样

一个 PyTorch 模型是嵌套的 Python 对象。比如一个简化版的 Transformer FFN（前馈网络）：

```python
class TinyFFN(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(64)
        self.gate = nn.Linear(64, 128, bias=False)
        self.up   = nn.Linear(64, 128, bias=False)
        self.down = nn.Linear(128, 64, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return self.down(silu(self.gate(h)) * self.up(h))
```

这段代码里，`nn.LayerNorm`、`nn.Linear` 各自又是一个 Module，里面还有权重矩阵。GPU 没法直接执行这种嵌套的 Python 对象——它需要一张扁平的、每一步运算都明确的指令序列。

### FX Graph：扁平的运算图

PyTorch 提供了一个工具 `torch.export`，可以追踪 `forward()` 的执行过程，把所有嵌套层展开成一张扁平的运算图（**FX Graph**）。图里的每个节点是一个 **aten op**——PyTorch 的底层原子操作。

torch2vk 调用 `export_submodule()` 完成这个追踪：

```python
prog = export_submodule(model, args=(dummy_input,))
```

上面那个 TinyFFN 会被展开成这样的图：

```
[input] norm.weight          → 形状 (64,)
[input] norm.bias            → 形状 (64,)
[input] gate.weight          → 形状 (128, 64)
[input] up.weight            → 形状 (128, 64)
[input] down.weight          → 形状 (64, 128)
[input] x                    → 形状 (1, 8, 64)

layer_norm(x, weight, bias)  → 形状 (1, 8, 64)
linear(layer_norm, gate.w)   → 形状 (1, 8, 128)
silu(linear)                 → 形状 (1, 8, 128)
linear_1(layer_norm, up.w)   → 形状 (1, 8, 128)
mul(silu, linear_1)          → 形状 (1, 8, 128)
linear_2(mul, down.w)        → 形状 (1, 8, 64)

[output] linear_2
```

原来嵌套在 `nn.LayerNorm` 和 `nn.Linear` 里的逻辑，现在变成了 6 个清晰的 aten op：`layer_norm` → `linear` → `silu` → `linear` → `mul` → `linear`。每个 op 的输入输出张量的形状和数据类型都已经确定。

### 真实模型有多大的图

上面的例子只有 6 个 op，但真实模型要大得多。Qwen3-ASR 的一个 decoder layer（共 28 个 layer）展开后，FX Graph 里有几十个 aten op——包括 RMSNorm 拆分出来的 pow、mean、rsqrt、mul 系列，RoPE 的 slice、neg、cat、mul、add 系列，以及 QKV 投影、attention、FFN 等。整个音频编码器 + 文本解码器展开后有数百个 op。

但不管多大，结构都是一样的：一张由 aten op 节点组成的扁平有向图，每个节点有确定的输入、输出和形状。这张图就是 torch2vk 后续一切工作的起点。


## 第二步：aten op → Vulkan shader 的映射

FX Graph 里的每个 aten op 都需要翻译成一个 Vulkan compute shader。torch2vk 用一张查找表——**ShaderRegistry**——来完成这个翻译。

### ShaderRegistry：翻译字典

ShaderRegistry 的结构很简单：一个 aten op 名称到 shader 工厂函数的映射列表。当前支持 27 种 aten op：

| 类别 | 支持的 aten op |
|---|---|
| 线性代数 | linear（带 bias / 不带 bias） |
| 逐元素二元 | mul, add, sub |
| 逐元素一元 | silu, gelu, neg, rsqrt, reciprocal, sin, pow |
| 归约 | mean, max, argmax |
| 变换 | transpose, permute, slice, cat |
| 归一化 | layer_norm |
| 注意力 | scaled_dot_product_attention |
| 嵌入 | embedding |
| 卷积 | conv1d, conv2d, conv_transpose1d |
| 索引 | index_copy, index_select, select |

遇到不在表里的 op，导出会直接报错。比如 `nn.RMSNorm` 导出为 `aten.rms_norm`，它不在 registry 中——torch2vk 的做法是手动把 RMSNorm 拆成 pow + mean + rsqrt + mul 等已支持的 op 组合，或者直接写一个融合 shader（后面会讲到）。

### 工厂函数：从 FX 节点构造 ShaderVariant

Registry 里存的不是 shader 本身，而是**工厂函数**。每个工厂函数接收一个 FX Graph 节点，提取出张量的形状、数据类型等信息，然后构造出一个完整的 **ShaderVariant** 对象。

为什么需要工厂函数，不能直接存一个固定的 shader？因为同一种 op 在不同上下文里形状不同。比如 `aten.linear` 有时输入是 `(1, 128, 4096)`，有时是 `(1, 1, 4096)`；权重有时是 float32，有时是 bfloat16。工厂函数会根据这些信息动态生成对应的 shader 代码和参数。

### ShaderVariant 的组成

一个 ShaderVariant 包含运行一个 GPU 计算所需的全部信息：

**1. ShaderContract——shader 的接口规格**

类似函数签名，声明这个 shader 读什么、写什么：

```python
ShaderContract(
    fields=(
        TensorFieldSpec("x",      IOKind.INPUT,  TensorContract(dtype="float16", shape=("B", "T", "K"))),
        TensorFieldSpec("weight", IOKind.INPUT,  TensorContract(dtype="float32", shape=("N", "K"))),
        TensorFieldSpec("output", IOKind.OUTPUT, TensorContract(dtype="float16", shape=("B", "T", "N"))),
    ),
    push_constants=PushConstantSpec(
        fields=(
            PushConstantFieldSpec("M", UINT32, offset=0, value=mul("B", "T")),
            PushConstantFieldSpec("K", UINT32, offset=4, value="K"),
            PushConstantFieldSpec("N", UINT32, offset=8, value="N"),
        ),
    ),
    dispatch=(ceil_div(mul("B", "T"), 16), ceil_div("N", 64), 1),
)
```

这里有几个关键概念：

- **fields** 声明了三个张量：两个输入（x 和 weight）、一个输出。shape 里的 `"B"`, `"T"`, `"K"`, `"N"` 是**符号维度**——它们在导出时还不知道具体值，运行时才会从实际传入的张量推断出来。
- **push_constants** 是通过 GPU 寄存器传给 shader 的小块数据（通常不超过 128 字节），用来告诉 shader 矩阵有多大。`mul("B", "T")` 表示把 batch 和 sequence 维度乘在一起。这比通过显存传递快得多。
- **dispatch** 声明了要启动多少组工人：`ceil_div(M, 16)` 个组在 X 方向，`ceil_div(N, 64)` 个组在 Y 方向。这是符号表达式，运行时会代入实际数值。

**2. GLSL 源码——实际的 GPU 程序**

以矩阵乘法 shader 为例，它使用 16×64 的 tile 策略和共享内存来高效计算 `output = x @ weight^T`。GLSL 源码里用模板占位符（如 `{{ACTIVATION_TYPE}}`）来适配不同数据类型，工厂函数在构造时把占位符替换成实际的 GLSL 类型（如 `float16_t` 或 `float`）。

**3. 执行要求**

某些 shader 需要特定的 GPU 能力，比如：
- 16 位存储访问（`require_storage_buffer_16bit_access`）：使用 float16 张量时需要
- 64 位整数（`require_shader_int64`）：处理 int64 类型的索引时需要
- 子组操作（`subgroup`）：某些归约和注意力 shader 需要同组工人间的快速通信

### Alias Op：什么都不做的"翻译"

FX Graph 里还有一类操作：`view`、`unsqueeze`、`reshape`、`contiguous`。它们不改变数据本身，只改变"怎么看待这块内存"的方式——比如把 `(1, 8, 64)` 看成 `(8, 64)`。

torch2vk 把这些识别为 **alias op**：不生成 shader，不做计算，不分配显存。两个 LogicalTensor 指向同一块 GPU buffer，零开销。这在注意力机制里很常见——QKV 投影后的 reshape 和 transpose 有一部分可以用 alias 消除。


## 第三步：代码生成——tensor 声明和 dispatch 函数

ShaderRegistry 解决了"一个 aten op 用哪个 shader"的问题。但要真正执行，还需要回答：每个中间结果的显存从哪来？shader 按什么顺序调用？这就是代码生成（codegen）的工作。

### LogicalTensor：torch2vk 的核心抽象

在讨论生成的代码之前，需要先理解 torch2vk 最重要的概念——**LogicalTensor**。

LogicalTensor **不是 GPU buffer 本身**。它是一张"声明"，描述了一个张量的一切 metadata：

```python
@dataclass
class LogicalTensor:
    spec: TensorSpec         # 数据类型和形状，如 dtype="float16", shape=(1, 8, 64)
    role: TensorRole         # 语义角色
    memory: MemoryClass      # 显存分配策略
    lifetime: TensorLifetime # 生命周期
    layout: TensorLayout     # 内存布局（连续、量化打包等）
    # --- 运行时状态，由 RuntimeSession 维护 ---
    buffer: BufferSlice | None    # 当前绑定的 GPU buffer（如果已分配）
    version: int                  # 被写入过几次
    writer: DispatchWriter | None # 最后一次写入它的是哪个 shader
```

类比：LogicalTensor 是**建筑图纸**（描述这块空间多大、什么用途、什么时候拆），实际的 GPU buffer 是**建成的房间**。图纸可以先画好，房间等需要时再建。

#### 五种角色（TensorRole）

每个 LogicalTensor 有一个语义角色，告诉运行时这个张量从哪来、做什么用：

| 角色 | 含义 | 例子 |
|---|---|---|
| **WEIGHT** | 模型权重，从 checkpoint 加载 | `gate_proj.weight` |
| **INPUT** | 外部输入，从 CPU 上传 | 音频特征、token ID |
| **ACTIVATION** | 中间计算结果 | layer_norm 的输出 |
| **SCRATCH** | 某个 shader 内部的临时空间 | 归约时的 partial sum |
| **OUTPUT** | 最终输出，需要读回 CPU | 生成的 token、转录文字 |

#### 六种显存类别（MemoryClass）

角色决定了语义，显存类别决定了**从哪个显存池分配**：

| 显存类别 | 分配策略 | 对应角色 |
|---|---|---|
| **MODEL_WEIGHT** | 常驻 GPU，只读，模型加载时一次性上传 | WEIGHT |
| **HOST_INPUT** | Host-visible buffer，CPU 可直接写入 | INPUT |
| **FRAME_WORKSPACE** | 临时池，Frame 结束时整体回收 | ACTIVATION |
| **OP_SCRATCH** | 更短暂的临时池，单个 op 结束就回收 | SCRATCH |
| **REQUEST_STATE** | 跨 Frame 存活，请求结束时释放 | KV cache、已生成的 token |
| **HOST_OUTPUT** | Host-visible，CPU 可直接读取 | OUTPUT |

#### 四种生命周期（TensorLifetime）

显存类别决定了"从哪分配"，生命周期决定了"什么时候可以释放"：

| 生命周期 | 释放时机 | 典型用途 |
|---|---|---|
| **MODEL** | 模型卸载或会话结束 | 权重 |
| **REQUEST** | 一次推理请求完成 | KV cache |
| **FRAME** | 一个 Frame 执行完毕 | 中间激活 |
| **OP** | 一个 shader 执行完毕 | 临时 scratch |

**关键设计原则**：模型代码永远不手动分配或释放显存。模型代码只创建 LogicalTensor、把它们传给 shader，运行时（RuntimeSession）负责在需要时自动分配 buffer、用完后自动释放。

### 生成的代码长什么样

代码生成器从 FX Graph 提取信息，为每个子模块生成两样东西：

#### Tensor 声明（tensors/ 目录）

一个 frozen dataclass，包含该子模块用到的所有 LogicalTensor：

```python
@dataclass(frozen=True, slots=True)
class TextNormTensors:
    p_weight: LogicalTensor      # 权重：LayerNorm 的 weight
    hidden_states: LogicalTensor  # 输入：上一层的输出
    pow_1: LogicalTensor          # 中间激活：x^2
    mean: LogicalTensor           # 中间激活：均值
    rsqrt: LogicalTensor          # 中间激活：1/sqrt(var+eps)
    mul_1: LogicalTensor          # 输出：归一化后的结果
```

每个字段的 shape、dtype、role、memory、lifetime 都在工厂函数里填好了。比如 `p_weight` 会被声明为 `TensorRole.WEIGHT, MemoryClass.MODEL_WEIGHT, TensorLifetime.MODEL`，而 `pow_1` 会被声明为 `TensorRole.ACTIVATION, MemoryClass.FRAME_WORKSPACE, TensorLifetime.FRAME`。

#### Dispatch 函数（dispatch/ 目录）

一个按图的拓扑顺序排列的 shader 调用序列：

```python
def run_text_norm(rt: RuntimeSession, tensors: TextNormTensors) -> None:
    POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    MUL_LEFT_BROADCAST(rt, x=tensors.p_weight, y=tensors.to_1, output=tensors.mul_1)
```

每一行就是一次 GPU dispatch：把指定的 LogicalTensor 传给指定的 ShaderVariant。运行时会自动处理 buffer 分配、数据上传、descriptor 绑定等一切细节。

### 完整的导出流程

把上面的内容串起来，一个 `export.py` 脚本做的事情是：

```
PyTorch Module
  ↓ torch.export
FX Graph (aten ops + 形状信息)
  ↓ ShaderRegistry 匹配
每个 op 对应一个 ShaderVariant
  ↓ codegen
shaders/   → ShaderVariant 的 Python 声明文件（包含 GLSL 源码和 contract）
tensors/   → LogicalTensor 的 dataclass + 工厂函数
dispatch/  → shader 调用序列
```

整个过程是全自动的。输入一个 PyTorch module 和一组 dummy input（用于追踪形状），输出三个目录的 Python 源文件。这些文件就是模型在 Vulkan 上运行所需的全部声明。



---

# 第三部分：在 GPU 上真正跑起来

## Vulkan 驱动层：与 GPU 对话的基础设施

前面几节讲的是"声明"——声明有哪些 tensor、哪些 shader、怎么连接。这一节开始讲"执行"——这些声明怎么变成 GPU 上的真实计算。

torch2vk 的 Vulkan 驱动层（`torch2vk/vulkan/`）封装了与 GPU 通信的全部底层细节。

### VulkanDevice：连接 GPU

一切从发现 GPU 开始。`VulkanDevice` 负责：

1. 创建 Vulkan 实例（相当于"注册"到 GPU 驱动）
2. 枚举物理设备（找到你的 AMD 核显 / Intel 集显 / 独立显卡）
3. 查询能力（支持 float16？支持 64 位整数？子组大小是多少？）
4. 创建逻辑设备和计算队列（建立通信通道）

能力查询很重要。不同 GPU 支持的特性不同——比如 AMD RDNA 架构的子组大小是 64，而某些 Intel GPU 的子组大小是 32。某些高级 shader（如量化矩阵乘法）会要求特定的子组大小或协作矩阵（cooperative matrix）支持。如果 GPU 不满足 shader 的执行要求，运行时会在 dispatch 前报错，而不是给出错误结果。

### 显存管理：分层设计

GPU 的显存管理和 CPU 的 malloc/free 很不一样。频繁的小块分配/释放在 GPU 上开销极大——每次都要和驱动通信。torch2vk 用了一个分层的显存管理器来解决这个问题：

**DeviceLocalArena**——大块预分配，内部切片。类似于你先租下整层写字楼，然后自己隔成办公室分配给各个部门，而不是每个部门各自去找一间房子租。Arena 按 chunk 增长，内部用 free list 管理子区域，支持合并（coalescing）相邻的空闲区域。

**HostRing**——环形缓冲区，专门用于 CPU→GPU 的数据上传。CPU 往 ring 里写数据，GPU 读完了这块空间就可以被下一次写入复用。因为推理是流式的（一个 token 一个 token），这种环形结构非常高效。

**TemporaryTensorPool**——中间激活张量的缓存。一个 Frame 的中间结果（FRAME_WORKSPACE）在 Frame 结束后不是真的释放回操作系统，而是放进池子里。下一个 Frame 如果需要相同形状的张量，直接从池子里拿，免去重新分配的开销。

**RetiredAllocationQueue**——延迟回收队列。GPU 执行是异步的——CPU 提交了一批命令后，GPU 可能还在执行。你不能在 GPU 还在读一块内存的时候就释放它。这个队列跟踪每块内存的"最后使用时间"，只有确认 GPU 已经用完了才允许回收。

### ComputePipeline：编译 shader

GLSL 源码不能直接给 GPU 执行。需要先编译成 SPIR-V（Vulkan 的中间字节码），然后和 descriptor 布局、push constant 配置一起打包成一个 **ComputePipeline**。

管线编译的开销不小，所以 torch2vk 会缓存它们——同一个 shader 变体只编译一次。缓存的 key 包括：SPIR-V 内容的 SHA256 哈希、descriptor 数量、特化常量值、push constant 大小。

管线编译好之后，每次 dispatch 只需要：

1. 从 pipeline cache 拿到已编译的管线
2. 分配一个 **descriptor set**（告诉 GPU "binding 0 对应这块 buffer、binding 1 对应那块"）
3. 填入 push constants（矩阵维度等参数）
4. 提交 dispatch 命令

其中 descriptor set 的分配很轻量，可以频繁做。

### 一次 dispatch 的完整旅程

把上面的组件串起来，一次 shader dispatch 在 Vulkan 层面实际发生的事情是：

```
1. 查 pipeline cache → 命中，拿到 ComputePipeline
2. 从 DeviceLocalArena 切一块 buffer 给输出张量
3. 把输入 buffer 和输出 buffer 包装成 DescriptorBufferBinding
4. 从 descriptor pool 分配一个 descriptor set，写入这些 binding
5. 把矩阵维度等打包成 push constants（几十个字节）
6. 录入 command buffer：
   - vkCmdBindPipeline（绑定管线）
   - vkCmdBindDescriptorSets（绑定数据）
   - vkCmdPushConstants（传入参数）
   - vkCmdDispatch(groupX, groupY, groupZ)（启动计算）
7. vkQueueSubmit → GPU 开始执行
8. vkWaitForFences → CPU 等待 GPU 完成
```

在 eager 模式下（默认），每次 dispatch 都完整走一遍这个流程。后面会讲到 replay cache 如何跳过大部分步骤。


## RuntimeSession：把一切串起来的运行时

Vulkan 驱动层提供了与 GPU 通信的原语。但谁来决定"什么时候分配显存"、"权重从哪加载"、"中间结果什么时候可以释放"？答案是 **RuntimeSession**——torch2vk 运行时的核心对象。

### RuntimeSession 是唯一拥有执行状态的对象

这是一条硬约束。模型代码不手动分配 buffer、不手动绑定 descriptor、不手动释放显存。模型代码只做一件事：把 LogicalTensor 传给 ShaderVariant 调用。RuntimeSession 接管一切底层细节。

```python
rt = RuntimeSession.open(
    device_index=0,            # 第几块 GPU
    model_dir=gguf_path,       # 模型权重目录
    model_tensors=tensors,     # 所有 LogicalTensor 的声明
    get_shader=get_shader,     # shader 加载函数
)
```

### Frame：执行的逻辑边界

模型推理不是一个扁平的 shader 列表——它有结构。一次 prefill 是一个阶段，每个 decode step 是一个阶段，音频编码是另一个阶段。torch2vk 用 **Frame** 来表达这种结构：

```python
with rt.frame("qwen3.decode.0042"):
    run_decode_embed(rt)
    for layer_idx in range(28):
        run_decode_layer(rt, layer_idx)
    run_decode_norm(rt)
    run_lm_head_select(rt)
    run_token_store(rt)
```

`with rt.frame(...)` 做四件事：

1. **标记执行范围**：所有在这个 with 块内发生的 dispatch 都属于这个 frame
2. **收集 dispatch 记录**：每次 shader 调用的输入输出、形状、dispatch 大小都会被记录（DispatchRecord），为后续的 replay 和调试提供数据
3. **管理临时显存**：frame 结束时，所有 FRAME_WORKSPACE 类的 buffer 整体回收到临时池，不需要逐个释放
4. **提供调试锚点**：profile 和 compare 的结果都按 frame name 组织

### Materialization：声明变成现实

当一个 shader 被调用时，RuntimeSession 需要把 LogicalTensor 的"声明"变成"实际的 GPU buffer"。这个过程叫 **materialization**，根据 tensor 的角色走不同路径：

**权重（WEIGHT）**：第一次用到时从 checkpoint 文件（safetensors 或 GGUF）读取并上传到 GPU。之后常驻 MODEL_WEIGHT 显存池，不再重复加载。

**输入（INPUT）**：从用户通过 `rt.request(inputs=...)` 或 `rt.register_host_inputs()` 提供的 numpy 数组上传。每次新的输入数据都需要重新上传。

**中间激活（ACTIVATION）**：从 FRAME_WORKSPACE 池分配一块临时 buffer。Frame 结束后这块 buffer 会被回收到 TemporaryTensorPool，下一个 Frame 的同形状张量可以直接复用它。

**请求状态（REQUEST_STATE）**：KV cache、已生成的 token 列表等跨 Frame 存活的张量。从 REQUEST_STATE 池分配，整个推理请求完成后才释放。KV cache 还支持动态增长——随着生成的 token 变多，cache 容量可以自动扩大。

### 一次 shader dispatch 的 10 个步骤

当模型代码调用 `SILU_F32(rt, x=tensors.linear, output=tensors.silu)` 时，RuntimeSession 内部执行：

```
 1. 检查当前处于 rt.frame(...) 中 → 否则报错
 2. 校验 ShaderContract：传入的 tensor 数量、名称和 contract 声明匹配
 3. 解析符号维度：从实际 tensor 的 shape 推断出 "B"=1, "T"=8, "H"=128 等
 4. Materialize 输入：确保 x 有对应的 GPU buffer（权重则从 checkpoint 加载）
 5. Materialize 输出：为 output 从 FRAME_WORKSPACE 分配一块 buffer
 6. 打包 push constants：把符号值代入表达式，序列化成字节
 7. 计算 dispatch 维度：ceil_div(8*128, 256) = 4 个 workgroup
 8. 提交 GPU 计算：绑定 pipeline + descriptor + push constants → dispatch
 9. 记录 DispatchRecord：shader 名、读了哪些 tensor、写了哪些、dispatch 大小
10. 更新 tensor 状态：output.version += 1，output.writer = "silu_f32"
```

模型代码看到的只是一行 `SILU_F32(rt, x=..., output=...)`。背后这 10 步全部由 RuntimeSession 自动完成。

### Alias 的实现

前面提到 FX Graph 里的 view/reshape 是 alias op。在 RuntimeSession 层面，alias 的实现很简单：两个 LogicalTensor 共享同一个 `BufferSlice`。当 RuntimeSession materialize 一个 alias tensor 的读取时，它直接用源 tensor 的 buffer，不做任何数据拷贝或显存分配。

```python
bind_logical_tensor_alias(src=tensors.hidden_states, dst=tensors.to)
```

声明阶段建立 alias 关系后，运行时 materialize `tensors.to` 时直接复用 `tensors.hidden_states` 的 buffer。

这使得 torch.export 产生的大量 view/reshape/unsqueeze 操作不产生任何运行时开销。



---

# 第四部分：性能优化

## 为什么自动导出的版本慢？

前面几节描述了一条完整的路径：PyTorch module → FX Graph → shader + tensor + dispatch → Vulkan 执行。走完这条路径，你得到了一个**功能正确但性能朴素**的 Vulkan 实现。

它能跑，但比优化后的版本慢好几倍。问题出在哪？

### 一个 decoder layer 的实际调度次数

以 Qwen3 大语言模型的一个 decoder layer 为例。自动导出后，FX Graph 展开成几十个 aten op，每个 op 对应一次 GPU dispatch。让我们数一数：

```
RMSNorm（输入归一化）：
  pow → mean → add → rsqrt → mul → mul    = 6 次 dispatch

Q 投影 + QNorm + RoPE：
  linear → pow → mean → add → rsqrt → mul → mul → transpose
  → slice → neg → cat → mul → mul → add   = 14 次 dispatch

K 投影 + KNorm + RoPE：类似 Q，又是十几次

V 投影 + transpose：3 次

KV cache write：2 次

Attention (SDPA)：1 次

Output 投影 + 残差连接：2 次

FFN 的 RMSNorm + gate/up/down 投影 + SiLU + mul：约 14 次
```

总计：一个 decoder layer **约 56 次 GPU dispatch**。

### 两种开销

每次 dispatch 都有两种开销：

**1. Dispatch 开销（CPU 侧）**

每次 dispatch，CPU 都要：准备 descriptor set、填入 push constants、录入 command buffer、提交到 GPU 队列、等待完成。这些步骤的耗时和计算量无关——算 1 个数也是这么多步骤，算 100 万个数也是这么多步骤。

56 次 dispatch 就是 56 次这样的固定开销。

**2. 显存带宽开销（GPU 侧）**

这是更大的问题。GPU 的计算速度远远快于显存的读写速度。现代 GPU 每秒能做上万亿次浮点运算，但显存带宽只有几百 GB/s。

当你把 RMSNorm 拆成 6 个独立的 shader 时：

```
pow：   从显存读 x → 计算 x^2 → 写回显存
mean：  从显存读 x^2 → 计算均值 → 写回显存
add：   从显存读均值 → 加 epsilon → 写回显存
rsqrt： 从显存读结果 → 计算倒数平方根 → 写回显存
mul：   从显存读 x 和 rsqrt → 相乘 → 写回显存
mul：   从显存读结果和 weight → 相乘 → 写回显存
```

每一步的计算只是简单的乘法或加法，但每一步都要完整地读和写一次显存。6 步加起来，同一份数据在显存里进进出出了 12 次。如果把这 6 步合并成一个 shader，数据只需要从显存读一次、写一次——读写量减少到原来的 1/6。

这就是所谓的 **memory bound**（内存瓶颈）：GPU 大部分时间不是在计算，而是在等显存数据传输完成。

### 超市买菜的比喻

你要做一道菜，需要土豆、葱、肉三种食材。

**朴素版（56 次 dispatch）**：去超市买土豆，回家放好；再去超市买葱，回家放好；再去超市买肉，回家放好。每趟路上花的时间远超挑菜的时间。

**优化版（13 次 dispatch）**：列个清单，一趟把三样都买回来。路上只花一次时间。

接下来要讲的 shader 融合，就是"列清单一趟买完"。


## Shader 融合：多步合一

shader 融合的思路很直接：把多个小 shader 合并成一个大 shader，让数据在 GPU 的寄存器或共享内存里完成全部计算，只在最开始读一次显存、最后写一次显存。

让我们逐个看 Qwen3 优化版里实际做了哪些融合。

### 融合 1：RMSNorm + 权重乘法

**导出版**（6 次 dispatch）：
```
pow → mean → add → rsqrt → mul → mul
```

**优化版**（1 次 dispatch）：
```
RMS_NORM_MUL_F16_F32
```

一个 shader 里完成全部操作：读入 x 和 weight，在寄存器里计算 `x * weight / sqrt(mean(x^2) + eps)`，写出结果。中间的 `x^2`、均值、`rsqrt` 都不需要写回显存。

### 融合 2：RMSNorm + RoPE + 转置

**导出版**（约 14 次 dispatch）：
```
RMSNorm（6步）→ transpose → slice → neg → cat → mul → mul → add
```

**优化版**（1 次 dispatch）：
```
RMS_NORM_ROPE_TRANSPOSE_F16
```

RoPE（旋转位置编码）对 Q/K 向量施加和位置相关的旋转。导出版里它被拆成了 slice（取前半/后半）、neg（取负）、cat（拼接）、mul（乘以 cos/sin）、add 这些碎步。融合后，归一化、旋转和转置在一个 shader 内一气呵成。

### 融合 3：Q/K/V 三路投影

**导出版**（3 次 dispatch）：
```
linear(x, q_weight) → Q
linear(x, k_weight) → K
linear(x, v_weight) → V
```

**优化版**（1 次 dispatch）：
```
LINEAR_NOBIAS_Q4_K_QKV_MATVEC_F32
```

三次矩阵乘法的输入 `x` 是完全一样的。导出版每次都从显存读一遍 `x`，读了三次。融合版只读一次 `x`，在 shader 内部同时计算 Q、K、V 三个投影，分别写到三个输出 buffer。显存读取量减少为原来的 1/3。

同理，gate_proj 和 up_proj（FFN 里的两个投影）也被融合成 `LINEAR_NOBIAS_Q4_K_QK_MATVEC_F32`——读一次 x，同时算两个输出。

### 融合 4：SiLU 门控乘法

**导出版**（2 次 dispatch）：
```
silu(gate_output) → mul(silu_output, up_output)
```

**优化版**（1 次 dispatch）：
```
SWIGLU_F16
```

SwiGLU 是 LLM 里标准的 FFN 激活方式：`silu(gate) * up`。两步合一步，少一次显存读写。

### 融合 5：注意力 + KV cache 写入

**导出版**（3 次 dispatch）：
```
kv_cache_write(key_cache, new_k)
kv_cache_write(value_cache, new_v)
sdpa(q, k, v)
```

**优化版**（1 次 dispatch）：
```
SDPA_DECODE_CACHE_WRITE_F32
```

decode 阶段的注意力有一个特点：每次只有一个新 token 的 Q/K/V（sequence length = 1），但需要和 cache 里所有历史 token 的 K/V 做 attention。融合版在同一个 shader 里先把新 K/V 写入 cache 对应位置，然后直接用完整 cache 做 attention，省去了独立的 cache 写入 dispatch。

### 融合 6：矩阵乘法 + 残差连接

**导出版**（2 次 dispatch）：
```
linear(x, o_proj_weight) → add(residual, linear_output)
```

**优化版**（1 次 dispatch）：
```
LINEAR_NOBIAS_Q4_K_MATVEC_ADD_F32
```

矩阵乘法的输出直接加上残差，不需要中间写回显存再读出来做 add。

### 总体效果

| | 导出版 | 优化版 |
|---|---|---|
| 每层 dispatch 次数 | ~56 | ~13 |
| 显存读写次数 | 每个中间结果读写两次 | 大部分中间结果不过显存 |
| 额外收益 | — | 量化权重、decode 专用 matvec |

在 AMD Radeon 890M 核显上的实际数据：优化版 Qwen3 的 decode 速度是 **7.4 ms/token**（约 135 tokens/秒），这对一块笔记本集成显卡来说是相当不错的。

### 这些融合不是编译器做的

值得强调：**每一个融合 shader 都是人手写的 GLSL 程序**。`SDPA_DECODE_CACHE_WRITE_F32` 有 168 行 Python 声明 + 内嵌 GLSL 代码。`LINEAR_NOBIAS_Q4_K_QKV_MATVEC_F32` 需要同时理解量化数据布局和多路矩阵乘法。

torch2vk 不依赖编译器自动做 fusion。它的策略是：

1. 自动导出生成一个**正确的基线**（56 次 dispatch 版本）
2. 人工（或 AI coding agent）阅读基线，识别融合机会
3. 手写融合 shader，替换掉基线中的多步调用

这个策略的好处是完全透明——每个 shader 都是可读的 GLSL 代码，可以独立测试、独立调优。代价是需要人写 shader。但这恰好是 AI coding agent 可以胜任的工作：基线代码结构清晰，每个 shader 的输入输出有 contract 明确声明，融合后的 shader 可以用同样的对拍框架验证正确性。


## 量化：用更少的内存装更多的模型

### 为什么需要量化

一个大语言模型的绝大部分体积是权重矩阵。以 Qwen3-0.6B 为例：6 亿个参数，如果用 float32 存储，每个参数 4 字节，光权重就要 2.4 GB。更大的模型（7B、14B）动辄几十 GB。

笔记本核显的显存通常只有 4～8 GB，还要和系统共享。不压缩权重，根本塞不下。

### 基本原理：用整数近似浮点数

量化的核心思想：一个矩阵里的数值通常分布在一个有限范围内。如果用 8 bit 整数（256 个级别）均匀覆盖这个范围，大多数情况下精度损失很小。

最简单的量化方式（Q8_0）：

```
原始权重（32 个 float32）：[0.12, -0.35, 0.08, ..., 0.44]

1. 找到绝对值最大的数 max_abs = 0.44
2. 缩放因子 scale = max_abs / 127
3. 量化值 = round(原始值 / scale)，范围 [-128, 127]
4. 存储：1 个 float16 scale + 32 个 int8 = 34 字节

原始大小：32 × 4 = 128 字节
量化后：34 字节 → 约 4× 压缩
```

还原时只需 `value = quantized_int × scale`，在 shader 里做矩阵乘法时实时还原。

### torch2vk 支持的量化格式

torch2vk 使用和 llama.cpp 兼容的 GGUF 量化格式，支持三种精度：

| 格式 | 比特数 | 块大小 | 每块字节 | 压缩比 | 用途 |
|---|---|---|---|---|---|
| **Q8_0** | 8 bit | 32 | 34 字节 | ~4× | 卷积等特殊层 |
| **Q6_K** | 6 bit | 256 | 210 字节 | ~5× | 关键层（注意力 V 投影、输出头） |
| **Q4_K_M** | 4 bit | 256 | 144 字节 | ~8× | 大多数层 |

Q4_K_M 是最激进的压缩。256 个浮点数被压缩到 144 字节，每个数平均只占 4.5 bit。它使用分层缩放结构：把 256 个值分成 8 个子块，每个子块有自己的 scale 和 min 值，再加上全局的 scale。这比简单的均匀量化多了一些 overhead，但精度好很多。

### 混合精度策略：不同层用不同精度

不是所有层对量化的敏感度都一样。torch2vk 的策略是：

- **前 1/8 的层**（靠近输入）：用 Q6_K（更高精度），因为它们编码的是低频特征，误差会被后续层放大
- **后 1/8 的层**（靠近输出）：用 Q6_K，因为它们直接影响最终输出质量
- **中间层**：大多用 Q4_K_M，但每隔 3 层穿插一层 Q6_K
- **输出头（lm_head）**：用 Q6_K，因为它直接决定 token 选择
- **小张量**（如 LayerNorm 的 weight）：不量化，保持原始精度

这种混合策略在 4× 到 8× 的压缩比之间取得平衡。Qwen3-0.6B 量化后只需约 400MB 显存存储权重。

### 量化 shader 和普通 shader 的区别

量化不只是换个存储格式。shader 里的矩阵乘法也要重写——需要在计算过程中实时解码量化数据。

普通的矩阵乘法 shader 直接读 float 权重：
```glsl
float w = weight[row * K + k];
acc += x_value * w;
```

Q4_K 量化的矩阵乘法 shader 需要从压缩格式里解码：
```glsl
// 从 144 字节的块里提取：全局 scale、子块 scale、4-bit 量化值
uint block_word = weight[row * blocks_per_row * 36 + block * 36];
vec2 dm = unpackHalf2x16(block_word);  // scale 和 min
// ... 解码 4-bit 值、乘以 scale、减去 min、累加 ...
```

decode 阶段还有一个特点：batch size = 1（每次只生成一个 token），所以矩阵乘法退化为**向量-矩阵乘法（matvec）**。torch2vk 为 matvec 写了专用的 shader，利用子组（subgroup）内的快速归约，比通用矩阵乘法更高效。

### 量化过程本身也在 GPU 上

一个有趣的细节：权重量化（从 float32 打包成 Q4_K/Q6_K/Q8_0）本身也是用 Vulkan compute shader 完成的。torch2vk 会：

1. 从 HuggingFace 下载原始 safetensors 格式的权重
2. 根据量化配置，对每个权重矩阵调用 Vulkan 量化 shader
3. 把量化后的结果写入 GGUF 文件
4. 后续推理直接从 GGUF 加载，通过 mmap 零拷贝映射到内存

第一次运行时会多花几分钟做量化，之后的运行直接用缓存的 GGUF 文件。


## Replay Cache：录一次，重放无数次

### 自回归生成的瓶颈

大语言模型生成文本时，一个 token 一个 token 地输出。每个 token 的 decode step 执行完全相同的 shader 序列——28 层 decoder layer + norm + lm_head + token select——只是输入数据不同（新 token 的 embedding、更新的 KV cache position）。

在 eager 模式下，每个 decode step 都要：

```
Python 层：运行模型代码，依次调用每个 shader
  ↓
RuntimeSession：为每个 shader 做 materialization、contract 校验、符号解析
  ↓
Vulkan 层：为每个 shader 分配 descriptor、录入 command buffer、提交、等待
```

生成 64 个 token 就重复 64 次。Python 层和 Vulkan 准备层的开销在 decode 阶段变得显著——GPU 算得快，但 CPU 准备慢。

### Replay 的核心思路

如果 shader 序列不变、buffer 布局不变，那 command buffer 的结构也不变。能不能**录一次 command buffer，后续直接重放**？

这就是 Replay Cache 的设计：

```
第一次 decode step（record）：
  正常走 eager 流程 → 记录所有 dispatch → 构建 ReplayPlan → 缓存

后续 decode step（replay）：
  跳过 Python 模型代码
  跳过 contract 校验、符号解析
  跳过 command buffer 录制
  只更新变化的数据 → 直接 vkQueueSubmit 缓存的 command buffer
```

### ReplayPlan 里有什么

一个 ReplayPlan 是一条预录制的 GPU 命令序列，包含：

- **Command buffer**：已经录制好的 Vulkan 命令（bind pipeline、bind descriptor、push constants、dispatch），可以直接提交
- **Descriptor set**：每个 shader 的 buffer 绑定。Vulkan 的 command buffer 记录的是 descriptor set 的 handle，不是 buffer 的地址——所以可以在不重新录制 command buffer 的情况下，更新 descriptor set 里的 buffer 指向
- **Fence**：GPU 同步原语，用于等待执行完成
- **Workspace allocations**：中间激活张量的 buffer（replay 自己持有，不依赖 Frame allocator）

### 什么可以变，什么不能变

Replay 的关键约束：command buffer 的**结构**不能变，但 buffer 的**内容和地址**可以变。

**不能变的**（会导致 cache miss，需要重新 record）：
- shader 序列（哪些 shader、什么顺序）
- 每个 shader 的 pipeline 和 descriptor 布局
- 静态的 push constant 值（如 attention head 数量、hidden size）

**可以变的**（replay 时动态更新）：
- buffer 内容（新 token 的 embedding、新的 cache position）
- buffer 地址（KV cache 增长后重新分配了更大的 buffer）
- 动态符号值（如 KV cache 的容量步长 S）

### 动态符号和间接 dispatch

有些值在不同请求之间会变化。比如 Qwen3-ASR 处理不同长度的音频时，KV cache 的容量不同。如果把容量值写死在 push constants 里（push constants 被录进 command buffer），不同长度就没法复用同一个 replay plan。

解决方案是把这类值从 push constants 移到 **params buffer**——一块 host-visible 的 GPU buffer。replay 前 CPU 更新 params buffer 的内容，command buffer 里的 shader 从 buffer 里读取参数。

类似地，dispatch 维度（workgroup 数量）如果依赖动态符号，就用 `vkCmdDispatchIndirect`——把 workgroup 数量写在一个 buffer 里，GPU 从 buffer 读取 dispatch 大小，而不是从 command buffer 的常量里读。

这样，**同一条 command buffer 可以服务不同长度的请求**，只要 shader 序列和 descriptor 布局不变。

### 实际效果

Replay 消除了 decode 热路径上的所有 Python 开销和大部分 Vulkan 准备开销。每个 decode step 只需要：

```
1. 把新的 cache_position、token_index 等写入 host-visible buffer
2. 更新 params buffer 里的动态符号值
3. 更新 indirect dispatch buffer 里的 workgroup 数量
4. vkQueueSubmit（提交缓存的 command buffer）
5. vkWaitForFences（等待完成）
```

步骤 1-3 只涉及 CPU 端的少量内存写入，步骤 4-5 是不可避免的 GPU 交互。跳过了整个 Python 模型代码执行、每个 shader 的 materialization 和 command buffer 录制。

torch2vk 的测试验证了 replay cache 的正确性：用两段不同长度的音频分别做 ASR，第一段触发 record，第二段命中 cache 走 replay，两段都必须得到正确的转录结果。



---

# 第五部分：保证正确性

## 保证正确性：PyTorch 对拍和集成测试

几十种手写 shader，涉及量化解码、注意力机制、位置编码旋转——怎么保证数值正确？

### PyTorch 对拍：逐张量比较

torch2vk 的导出过程不仅生成 shader 和 dispatch 代码，还会自动生成 **reference 函数**——一组 Python 包装器，可以用原始 PyTorch 模型跑同样的计算，然后逐张量对比 Vulkan 的输出和 PyTorch 的输出。

对拍流程：

```
1. Vulkan 跑一步（比如一个 decoder layer）
2. 读回 Vulkan 的输出到 CPU（readback）
3. 用相同的输入跑 PyTorch 的对应子模块
4. 逐元素比较两个输出
```

比较时允许小误差——GPU 的 float16 计算和 CPU 的 float32 计算不可能完全一致。torch2vk 使用 **ComparePolicy** 定义容差：

| 比较模式 | 典型容差 | 用途 |
|---|---|---|
| **tensor**（浮点张量） | rtol=1e-2, atol=1.5 | 中间激活、注意力输出 |
| **token**（整数 token） | 精确匹配 | 生成的 token ID 必须一致 |

不匹配时，报告精确到：哪个 frame、哪个 tensor、哪个 shader 最后写了它、第几个元素出了问题、最大绝对误差和相对误差是多少。

这个机制是**显式触发**的——通过 `compare.py` 脚本运行，不会混入推理热路径。你可以在开发融合 shader 时频繁对拍，确认无误后关掉对拍直接跑推理。

### 有状态的对拍：decode 循环

decode 阶段有一个难点：它是有状态的。每一步都读写 KV cache，下一步依赖上一步的结果。如果只对比单步的输入输出，可能错过累积误差。

torch2vk 现在把 Vulkan/request-state tensor 作为长流程状态的 single source of truth：
- KV cache、generated tokens、stopped 标志都保存在 Vulkan 侧；
- compare.py 在每个子图执行前，从 Vulkan 当前 tensor/readback 读取 PyTorch reference 的输入；
- PyTorch reference 只验证当前子图，不维护另一份独立 token/cache 状态。

这样仍然能覆盖累计漂移：如果前一步 Vulkan 写错了 token 或 KV cache，下一步 reference 会用这个错误状态作为输入，
并在对应子图的输出比较中暴露 mismatch，而不是让 PyTorch 在自己的正确状态上继续跑出另一条路径。

### 集成测试：端到端验证

torch2vk 的项目原则是不做单元测试，做**端到端集成测试**。原因是：单元测试的被测接口经常随优化而改变（比如 shader 融合会改变整个 dispatch 序列），测试价值不大。端到端测试验证的是最终输出，不依赖内部实现细节。

实际的测试用例：

**文本生成**：
```python
def test_optimized_qwen3_generates_vulkan_text():
    result = main(max_new_tokens=8)
    assert "Vulkan" in result.text
```
提示词要求模型谈论 Vulkan compute shader。如果量化或 shader 有 bug 导致生成乱码，"Vulkan" 几乎不可能出现在输出里。

**语音识别**：
```python
def test_exported_qwen3_asr_transcribes_asknot():
    assert main() == "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
```
输入 JFK 的演讲音频，必须精确转录出完整句子。任何 shader 的数值误差积累到足以影响 token 选择时，转录文本就会出错。

**Replay cache 跨长度复用**：
```python
def test_replay_cache_hits_across_wav_lengths():
    # 第一段音频：触发 record，生成 "And so"
    result1 = main(wav="asknot.wav", max_new_tokens=2)
    assert result1 == "And so"
    
    # 第二段音频（不同长度）：必须命中 cache 走 replay
    result2 = main(wav="different.wav", max_new_tokens=2)
    # 检查 dispatch 记录：第二次不应该有 record 阶段的 dispatch
    # 但必须有 replay 阶段的 dispatch
```
这个测试同时验证了两件事：replay 的 descriptor 更新机制正确，以及动态符号更新后的数值结果正确。



---

# 第六部分：性能调优——深入 GPU 内部

## SQTT：看清 GPU 在忙什么

Shader 融合解决了"减少趟数"的问题。但每趟内部效率如何？一个融合后的 shader 到底花了多少时间在计算、多少时间在等内存？瓶颈在哪条 GLSL 源码行？

要回答这些问题，需要**指令级别**的 GPU 性能数据。

### SQTT 是什么

SQTT（Shader Queue Thread Trace）是 AMD GPU 的硬件级追踪机制。它不是采样（每隔一段时间看一下 GPU 在干嘛），而是**逐指令录制**——每条 GPU 汇编指令的执行时间、停顿原因、使用的硬件单元，都被完整记录下来。

这些数据的精度是 CPU 侧的 profiling 工具无法比拟的。CPU 只能在 dispatch 前后打时间戳，看到的是"这个 shader 总共花了 X 微秒"。SQTT 能告诉你："这个 shader 的第 42 行源码对应的那条 VMEM 加载指令，平均等了 150 个 cycle 才拿到数据——你的内存访问模式有问题。"

### 从 GLSL 源码到 GPU 指令

理解 SQTT 的归因链路，需要知道一行 GLSL 代码经过了多少层翻译才到达硬件：

```
GLSL 源码（你写的高级代码）
  ↓ glslang 编译器
SPIR-V（Vulkan 的中间字节码）
  ↓ AMD 驱动内的编译器（基于 LLVM）
ISA 汇编（AMD GPU 的原生指令集）
  ↓ GPU 硬件执行
SQTT 事件流（每条指令的执行记录）
```

SQTT 记录的是最底层的 ISA 指令事件。要把它映射回你关心的 GLSL 源码行，需要驱动编译时保留的 debug info——从 ISA 指令的 PC（程序计数器）地址映射到 SPIR-V offset，再映射到 GLSL 源文件的行号和列号。

### torch2vk 的 profiling 工作流

torch2vk 把 SQTT 封装成了一个完整的工作流：

#### 1. Capture：选择性录制

SQTT 的数据量很大，不能对整个推理过程全局开启。torch2vk 通过 Vulkan debug label 实现选择性录制：

- 运行时在每次 dispatch 时打上标签，如 `frame=qwen3.decode.0042;shader=sdpa_decode_cache_write;dispatch=137`
- 定制的 Mesa RADV 驱动（项目的 `third_party/mesa` fork）识别这些标签
- 只有匹配 filter 的 dispatch 才会触发 SQTT 录制
- 输出 `.rgp` 文件（AMD Radeon GPU Profiler 格式），每个录制的 dispatch 一个文件

同时，运行时写出 `dispatches.jsonl`——每个 dispatch 的元数据（shader 名、frame、dispatch 大小、读写了哪些 tensor）。

#### 2. Decode：解析硬件追踪

`.rgp` 文件是二进制格式。torch2vk 的 SQTT 模块用 ROCm 的 trace decoder 库（通过 Python FFI）解析它，提取出指令级事件：

- **PC 地址**：这条指令在 shader 二进制里的位置
- **指令类别**：VALU（向量计算）、VMEM（显存读写）、SMEM（标量内存）、LDS（共享内存）、SALU（标量计算）等
- **持续周期数**：这条指令花了多少个 GPU clock cycle
- **停顿周期数**：其中多少是在等待（等内存返回数据、等前一条指令的结果）

#### 3. Attribution：归因到源码

把指令事件映射回 GLSL 源码行。torch2vk 从驱动编译产物中提取 debug info：

- `pipeline-debug.json`：每个管线的寄存器使用量、ISA 指令到源码行的映射表
- `compiler-native-disasm.s`：完整的 ISA 汇编代码

然后把 SQTT 事件按 `(pipeline, 源码行)` 聚合，得到每行 GLSL 源码的总周期数和指令类别分布。

#### 4. Report：热点报告

最终生成可读的分析报告：

**Top 热点源码行**——按周期数排序，最耗时的 GLSL 源码行排在前面：
```
Line 42: acc += src[i] * weight[i];
  Total: 1250 cycles  |  VALU: 850  |  VMEM: 350  |  Hits: 1000
```

**Roofline 分析**——判断 shader 是计算瓶颈还是内存瓶颈：
```
Arithmetic intensity: 2.3 FLOP/byte → Memory bound
Achieved bandwidth: 180 GB/s / 256 GB/s peak → 70% utilization
```

如果一个 shader 是 memory bound，优化方向是减少显存访问（更多融合、更好的数据重用）。如果是 compute bound，优化方向是减少指令数（更高效的算法、用 cooperative matrix 等硬件加速）。

### Profile diff：和 llama.cpp 比快慢

llama.cpp 是另一个成熟的 Vulkan 推理实现。torch2vk 提供了 profile diff 工具，可以：

1. 对比两者在相同 op 类别（attention、linear、norm 等）上的耗时
2. 找到 torch2vk 比 llama.cpp 慢的 op
3. 针对性地优化那些 shader

这些 profiling 数据形成闭环：写 shader → 跑 SQTT → 找热点 → 优化 shader → 再跑 SQTT 验证效果。



---

# 第七部分：全景回顾

## 架构全图和设计哲学

### 四层积木

把前面所有内容串起来，torch2vk 是四层积木：

```
┌──────────────────────────────────────────────────────────────┐
│  模型适配层（models/）                                         │
│  每个模型一个目录，包含：                                        │
│  · tensors/  — LogicalTensor 声明                             │
│  · shaders/  — ShaderVariant 声明（GLSL + contract）           │
│  · dispatch/ — shader 调用序列                                 │
│  · run.py    — 推理入口（prefill + decode loop）               │
│  · compare.py — PyTorch 对拍入口                               │
├──────────────────────────────────────────────────────────────┤
│  导出层（export/）                                              │
│  · torch.export → FX Graph                                    │
│  · ShaderRegistry：aten op → ShaderVariant 匹配               │
│  · codegen：生成 tensors/ + shaders/ + dispatch/ 代码          │
│  · reference codegen：生成 PyTorch 对拍的 reference 函数        │
├──────────────────────────────────────────────────────────────┤
│  运行时（runtime/）                                             │
│  · RuntimeSession：执行状态的唯一所有者                           │
│  · Frame：执行边界 + 显存生命周期管理                             │
│  · Materialization：LogicalTensor → GPU buffer                 │
│  · Replay cache：command buffer 录制和重放                      │
│  · Profiler：dispatch 记录 + SQTT 集成                         │
├──────────────────────────────────────────────────────────────┤
│  Vulkan 驱动层（vulkan/）                                       │
│  · VulkanDevice：GPU 发现、能力查询、队列管理                     │
│  · MemoryManager：Arena + Ring + Pool 分层显存管理              │
│  · ComputePipeline：GLSL → SPIR-V → 管线编译和缓存              │
│  · Queue submission：command buffer 提交和同步                  │
└──────────────────────────────────────────────────────────────┘
```

每一层只和相邻层通信：

- **模型层**不直接调用 Vulkan API，只使用 LogicalTensor 和 ShaderVariant
- **导出层**不知道运行时如何分配显存，只生成声明代码
- **运行时**不知道模型的业务逻辑（什么是 attention、什么是 FFN），只知道 tensor 的 role/memory/lifetime 和 shader 的 contract
- **Vulkan 层**不知道 tensor 或 shader 的概念，只知道 buffer、pipeline、descriptor

项目的 import linter 强制了这些层级约束：模型层不能直接 import vulkan 模块，vulkan 层不能 import runtime 模块。

### 数据流

一个模型从 PyTorch 到在 GPU 上运行的完整数据流：

```
PyTorch nn.Module + dummy inputs
  ↓ export.py（一次性运行）
FX Graph
  ↓ ShaderRegistry + codegen
shaders/  +  tensors/  +  dispatch/  （生成的 Python 源文件）
  ↓ run.py（每次推理运行）
RuntimeSession.open()
  ↓ 加载 checkpoint → materialize 权重 → 上传到 GPU
  ↓ 注册输入 → register_host_inputs()
  ↓ 执行 prefill frame → 录制 → 构建 replay plan
  ↓ 执行 decode loop → replay cached command buffer
  ↓ 读回输出 → read_request_state()
文本 / 转录结果
```

### 为什么不用编译器做优化

传统的 AI 推理优化路径是让编译器自动做 fusion——TVM、Triton、XLA 都是这个方向。编译器分析计算图，识别可融合的模式，自动生成融合后的 kernel。

torch2vk 选择了不同的路径。它的编译器（导出层）只做 1:1 的翻译：一个 aten op 生成一个 shader，不做任何自动融合。所有性能优化都是手写的 GLSL shader。

**为什么这样选择？**

**1. 透明性**

编译器是黑盒。当 TVM 生成的 kernel 性能不好时，你看到的是一堆编译器内部 IR 和调度策略——很难理解为什么它做了这样的决策，更难修改。

torch2vk 生成的代码是人可读的 Python + GLSL。每个 shader 的输入输出有 contract 声明，dispatch 维度有符号表达式，GLSL 源码就在 Python 文件里。出了问题，直接看代码。

**2. 可控性**

编译器的 fusion 策略是通用的——它不知道 decode 阶段 batch=1 所以应该用 matvec，不知道 QKV 投影共享输入所以应该合并，不知道 KV cache write 和 attention 在语义上紧密关联所以应该融合。

手写 shader 可以利用这些领域知识做出编译器做不到的优化决策。

**3. 跨硬件可移植性**

Vulkan compute shader 是跨硬件的。同一个 GLSL shader 在 AMD、Intel、高通的 GPU 上都能运行（虽然性能特征不同）。如果某个 GPU 需要特殊优化，只需要替换对应的 shader 文件，不需要重写编译器后端。

**4. AI coding agent 友好**

这是最有意思的一点。自动导出的基线代码结构清晰：

- 每个 shader 是独立的 GLSL 文件
- 输入输出有类型化的 contract
- 可以用同样的对拍框架验证任何 shader 修改
- 融合的模式是重复出现的（RMSNorm + X、linear + residual、Q/K/V merge）

这些特征让 shader 融合变成了一个结构化的编程任务——而不是调编译器旋钮的黑魔法。一个足够理解 GLSL 和 transformer 架构的 AI coding agent 可以：阅读导出基线 → 识别融合机会 → 生成优化 shader → 用对拍验证正确性。

**这个选择的代价**是需要人（或 agent）写融合 shader。每个新模型架构的优化需要专门的工作。**收益**是完全透明、完全可控、完全可调试。

### 试一试

如果你想亲手体验这个系统：

```bash
# 量化 Qwen3 文本生成（Q4_K_M，在核显上约 7 ms/token）
uv run python -m models.optimized_qwen3.run

# Qwen3-ASR 语音识别（从 WAV 文件转录文字）
uv run python -m models.exported_qwen3_asr.run

# 观察导出过程：PyTorch module → 生成 shaders/ + tensors/ + dispatch/
uv run python -m models.exported_qwen3_asr.export
```

第一次运行会自动下载模型权重并做量化（需要几分钟）。之后的运行直接使用缓存。

整个过程只需要一台有 Vulkan 支持的 GPU 的电脑——不需要 NVIDIA，不需要 CUDA。
