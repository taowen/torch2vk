# torch2vk 科普文章提纲

目标读者：对 AI 有兴趣但没有 GPU 编程经验的开发者。全文用"提出问题→解释原理→展示 torch2vk 怎么做"的结构串联。

---

## 第一部分：为什么需要 torch2vk

### 1. 你的显卡，凭什么不能跑大模型？
- NVIDIA CUDA 对 AI 生态的垄断现状
- AMD / Intel / 高通 / 手机 GPU 硬件上完全能算，但缺软件
- Vulkan：一个跨厂商的 GPU 编程标准
- torch2vk 的一句话定位：把 PyTorch 模型翻译成 Vulkan compute shader，让任意 GPU 都能跑推理

### 2. GPU 编程到底是怎么回事？
- CPU vs GPU 的核心差异：少量强核 vs 海量弱核
- 比喻：数学天才 vs 一千个小学生
- AI 计算为什么天然适合 GPU：矩阵乘法、逐元素运算，每个元素独立可算
- **Compute shader 是什么**：一段告诉 GPU "每个工人干什么"的小程序
- 用 SiLU 激活函数举例：10 行 GLSL 代码 + `gl_GlobalInvocationID` 的含义
- workgroup 和 dispatch 的概念：把工作分成小组，每组内的工人可以共享数据

---

## 第二部分：从 PyTorch 到 Vulkan 的翻译过程

### 3. 第一步：torch.export 把模型摊平
- PyTorch 模型是嵌套的 Python 对象（nn.Module 套 nn.Module）
- torch.export 把 forward() 追踪成一张扁平的运算图（FX Graph）
- 图里每个节点是一个 aten op（PyTorch 的底层原子操作）
- 用一个 TinyFFN（LayerNorm + SiLU 门控 MLP）举例，展示 6 个 aten op 的图

### 4. 第二步：aten op → Vulkan shader 的映射
- **ShaderRegistry**：一张"aten op → shader 工厂"的查找表
- 当前支持的 27 种 aten op 一览（linear、mul、add、silu、sdpa、embedding、conv 等）
- 工厂函数做什么：从 FX node 提取 shape/dtype 信息，构造 ShaderVariant
- **ShaderVariant 的组成**：
  - ShaderContract：声明 shader 的输入/输出张量、数据类型、符号化的形状
  - GLSL 源码：实际的 GPU 程序
  - Push constants：传给 shader 的运行时参数（矩阵维度、epsilon 等）
  - Dispatch 维度：告诉 GPU 要派多少个工作组（用符号表达式，如 `ceil_div(M, 16)`）
- 遇到不支持的 op 会怎样：报错，需要手写一个新 shader

### 5. 第三步：代码生成——tensor 声明和 dispatch 函数
- **LogicalTensor**：torch2vk 的核心抽象
  - 不是显存 buffer 本身，而是一个"声明"——描述张量的形状、类型、角色、生命周期
  - 类比：建筑图纸 vs 实际建筑物
  - 五种角色（TensorRole）：WEIGHT / INPUT / ACTIVATION / SCRATCH / OUTPUT
  - 六种显存类别（MemoryClass）：MODEL_WEIGHT / HOST_INPUT / FRAME_WORKSPACE / REQUEST_STATE / ...
  - 四种生命周期（TensorLifetime）：MODEL（常驻）/ REQUEST（跨帧）/ FRAME（单帧释放）/ OP（单算子）
  - 模型代码永远不手动分配/释放显存，只传递 LogicalTensor，RuntimeSession 负责一切
- 生成的 tensor dataclass：每个子模块一个，自动从 FX graph 提取所有张量的 metadata
- 生成的 dispatch 函数：一行一个 shader 调用，按图的拓扑顺序排列
- 整个过程全自动：一个 `export.py` 脚本，输入 PyTorch module，输出 shaders/ + tensors/ + dispatch/ 三个目录

---

## 第三部分：在 GPU 上真正跑起来

### 6. Vulkan 驱动层：与 GPU 对话的基础设施
- VulkanDevice：发现 GPU、建立连接、查询能力
- 显存管理的分层设计：
  - DeviceLocalArena：大块预分配，内部切片，避免频繁分配
  - HostRing：环形缓冲区，用于 CPU→GPU 数据上传
  - TemporaryTensorPool：中间激活的缓存复用
- ComputePipeline：编译 GLSL → SPIR-V → GPU 可执行的管线
  - 管线编译代价高，所以要缓存
  - Descriptor set：告诉 GPU "哪块显存绑定到 shader 的哪个参数"
- 一次 dispatch 的完整流程：绑定管线 → 绑定 descriptor → 设置 push constants → 提交 → GPU 执行 → 等待完成

### 7. RuntimeSession：把一切串起来的运行时
- RuntimeSession 是唯一拥有执行状态的对象
- **Frame 的概念**：一段逻辑执行范围
  - `with rt.frame("decode.0042")` 划定边界
  - Frame 结束时自动释放 FRAME_WORKSPACE 类的临时显存
  - Frame 内记录所有 dispatch 历史（DispatchRecord）
- **Materialization 过程**：LogicalTensor 声明 → 实际 GPU buffer
  - 权重：从 checkpoint 文件读取并上传到 GPU（只上传一次，常驻）
  - 输入：从 CPU numpy 数组上传到 host-visible buffer
  - 中间激活：从 FRAME_WORKSPACE pool 分配临时 buffer
  - KV cache：REQUEST_STATE 分配，跨多个 Frame 存活
- 一次 shader dispatch 的 10 个步骤：
  1. 校验在 frame 内
  2. 校验 shader contract 和传入 tensor 匹配
  3. 解析符号化的 shape（如 "B"=1, "T"=128）
  4. Materialize 输入（读）
  5. Materialize 输出（写——分配 buffer）
  6. 打包 push constants
  7. 计算 dispatch 维度
  8. 提交 GPU 计算
  9. 记录 DispatchRecord
  10. 更新 tensor 的 version 和 writer

### 8. Alias：零开销的 tensor 视图
- torch.export 会产生 view/unsqueeze/reshape 等操作
- 这些操作不改变数据，只改变"怎么看这块内存"
- torch2vk 把它们识别为 alias op，不生成 shader，只在 LogicalTensor 之间建立 alias 关系
- 两个 LogicalTensor 共享同一块 GPU buffer，零显存、零计算开销

---

## 第四部分：性能优化——从"能跑"到"好用"

### 9. 为什么自动导出的版本慢？
- 每个 aten op 一个 shader = 大量小 dispatch
- 实际例子：一个 decoder layer，自动导出 56 次调度
- 两种开销：
  - **dispatch 开销**：CPU→GPU 的指令传递、管线切换
  - **显存带宽开销**：中间结果写回显存又读出来（memory bound）
- 类比：买菜跑 56 趟 vs 一趟买完

### 10. Shader 融合：多步合一
- RMSNorm 融合：pow → mean → add → rsqrt → mul → mul 合并成 `RMS_NORM_MUL`（6→1）
- RoPE 融合：RMSNorm + RoPE + transpose 合并成 `RMS_NORM_ROPE_TRANSPOSE`
- QKV 投影融合：3 个独立 linear 合并成 `LINEAR_Q4_K_QKV_MATVEC`（读一次 x，算三个输出）
- SiLU 门控融合：silu + mul 合并成 `SWIGLU`
- 注意力融合：SDPA + KV cache write 合并成 `SDPA_DECODE_CACHE_WRITE`
- 矩阵乘法 + 残差连接融合：linear + add 合并成 `LINEAR_MATVEC_ADD`
- 总计从 56 次 → 约 13 次
- **关键洞察**：这些融合不是编译器自动做的，而是人写的——每个融合 shader 是一个独立的 GLSL 程序
- 这个架构为什么对 AI coding agent 友好：生成的代码结构清晰（Python + GLSL），输入输出有明确的 contract 声明

### 11. 量化：用更少的内存装更多的模型
- 为什么需要量化：大模型权重太大，核显显存有限
- 基本原理：把 float32 的权重压缩成 4/6/8 bit 整数 + 缩放因子
- torch2vk 支持的量化格式（兼容 GGUF / llama.cpp）：
  - Q4_K_M（4 bit，256 元素一块，8x 压缩）：大多数层
  - Q6_K（6 bit，256 元素一块，5x 压缩）：关键层（注意力、输出头）
  - Q8_0（8 bit，32 元素一块，4x 压缩）：卷积等特殊层
- 混合精度策略：不同层用不同量化级别
  - 前 1/8 和后 1/8 的层用更高精度（Q6_K）
  - 中间层大多用 Q4_K
  - 小张量不量化（保持 float32）
- 量化 shader 和普通 shader 的区别：
  - 需要在计算中实时解码——读取压缩数据、乘以缩放因子、还原为 float
  - Dispatch 策略不同：decode 阶段用 matvec（向量-矩阵乘法，因为 batch=1）
- Vulkan GPU 量化：权重打包本身也在 GPU 上完成（用 Vulkan compute shader 做量化）
- checkpoint 系统：支持 safetensors（原始 HuggingFace 格式）和 GGUF（量化后格式），通过 mmap 零拷贝加载

### 12. Replay Cache：录一次，重放无数次
- 自回归生成的特点：decode 阶段重复执行相同的 shader 序列，只是数据不同
- 问题：每次都要 Python 层准备 + GPU 录制命令——CPU 成了瓶颈
- Replay 的思路：
  - 第一次（record）：完整执行 Python，录制 GPU command buffer
  - 后续（replay）：跳过 Python 层，只更新数据，重放录好的 command buffer
- ReplayPlan 的组成：
  - 预录制的 Vulkan command buffer
  - 绑定的 descriptor set（可更新内容而不重建）
  - 可变参数的 params buffer（如 KV cache 的容量步长）
  - 间接 dispatch buffer（动态长度时的 workgroup 数量）
- 什么可以变，什么不能变：
  - 不能变：shader 序列、管线、descriptor 布局
  - 可以变：buffer 内容、buffer 地址（通过 descriptor 更新）、动态符号值（通过 params buffer）
- 不同音频长度也能复用同一个 replay plan：KV cache 的容量作为 dynamic symbol 传递
- 实际效果：decode 热路径不再执行 Python 模型代码、不再重新录制命令

---

## 第五部分：保证正确性

### 13. PyTorch 对拍：Vulkan 算得对不对？
- 问题：几十种手写 shader，怎么保证数值正确？
- 方案：每个导出的子模块自动生成 PyTorch reference 函数
- 对拍流程：
  1. Vulkan 跑一步
  2. PyTorch 用完全相同的输入跑同样的计算
  3. 逐元素比较输出（容忍小误差：rtol/atol）
  4. 不匹配时精确报告哪个 tensor、哪个 shader、第几个元素出了问题
- ComparePolicy：tensor 级（浮点容差）、token 级（必须精确一致）、waveform 级（音频特殊策略）
- 这个机制不是自动运行的——通过 compare.py 显式触发，不混入推理热路径

### 14. 集成测试：端到端验证
- 不做单元测试，做端到端集成测试（项目原则）
- 测试例子：
  - 文本生成：Qwen3 生成的文本里必须包含 "Vulkan"
  - 语音识别：JFK 演讲音频必须转录出完整的 "ask not what your country can do for you..."
- Replay cache 正确性测试：
  - 第一个请求（cache miss）：完整录制
  - 第二个请求（不同长度的音频，cache hit）：用缓存的 replay plan
  - 两次都要得到正确的转录结果

---

## 第六部分：性能调优——深入 GPU 内部

### 15. SQTT：看清 GPU 在忙什么
- 上一层的融合是"减少趟数"，但每趟内部效率如何？
- 需要 instruction 级别的数据：哪条 GPU 指令花了多少周期、卡在哪里
- SQTT（Shader Queue Thread Trace）：AMD GPU 的硬件级追踪机制
  - 不是采样，是逐指令录制——精确到每条 GPU 汇编指令
  - 记录执行时间、停顿原因（等内存、等依赖、资源不足）
- torch2vk 的 profiling 工作流：
  1. **Capture**：用定制的 Mesa RADV 驱动录制 SQTT trace
     - 通过 debug label 过滤，只录制你关心的 dispatch
     - 输出 .rgp 文件（AMD Radeon GPU Profiler 格式）
  2. **Decode**：解析 .rgp 文件，提取指令级事件
     - 使用 ROCm 的 trace decoder 库（FFI 调用）
  3. **Attribution**：把指令级数据归因到 GLSL 源码行
     - 完整链路：GLSL → SPIR-V → AMD ISA → 硬件执行
     - 驱动编译时保留 debug info（PC 地址 → 源码行号映射）
  4. **Report**：生成热点报告
     - 每个 shader 的 top 热点源码行和 GPU 周期数
     - Roofline 模型分析：计算瓶颈 vs 内存瓶颈
- 这些信息指导下一轮 shader 优化：知道了瓶颈在哪，才能精准融合

### 16. Profile diff：和 llama.cpp 比快慢
- llama.cpp 是成熟的 Vulkan 推理实现，适合作为性能对照
- profile_diff 工具：对比两者在相同 op 类别上的耗时
- 找到 torch2vk 比 llama.cpp 慢的 op → 针对性优化 shader

---

## 第七部分：全景回顾

### 17. 架构全图
- 四层积木图（模型适配层 / 导出层 / 运行时 / Vulkan 驱动层）
- 数据流：PyTorch module → FX Graph → shader + tensor + dispatch 代码 → Vulkan 执行 → 输出
- 每一层的职责边界：
  - 模型代码只声明"有什么 tensor"和"怎么连接 shader"
  - 导出层负责翻译
  - 运行时负责显存管理、dispatch 调度、replay
  - 驱动层负责和 GPU 通信

### 18. 设计哲学：为什么不用编译器
- 传统方案：用编译器自动做 fusion（如 TVM、Triton、XLA）
  - 优点：自动化程度高
  - 缺点：编译器是黑盒，出问题难调试，跨硬件兼容性差
- torch2vk 的选择：生成对人可读的 Python + GLSL 代码
  - 自动导出保证正确性基线
  - 手动（或 AI agent）做 shader 融合优化
  - 每个 shader 都是独立可理解、可测试、可替换的
  - 优化过程是"编程"，不是"调编译器参数"
- 这个选择的代价：需要人（或 agent）写融合 shader
- 这个选择的收益：完全透明，完全可控，跨硬件只需要重写 shader

### 19. 实际跑一跑
- 环境要求：有 Vulkan 支持的 GPU + Python 3.12 + uv
- 三个可以直接运行的命令：
  - `uv run python -m models.optimized_qwen3.run` — 量化 Qwen3 文本生成
  - `uv run python -m models.exported_qwen3_asr.run` — 语音识别
  - `uv run python -m models.exported_qwen3_asr.export` — 观察完整的导出过程
- 预期的输出和关键指标（prefill 速度、decode ms/token、显存占用）
