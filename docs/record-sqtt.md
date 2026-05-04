# SQTT 录制设计

本文记录对 `agentorch-mesa` 中 RADV SQTT/RGP 录制路径的调查结论，并定义 `torch2vk`
这边最终想要的 SQTT 录制形态。

核心结论：

```text
SQTT 录制不要在 torch2vk 里重写。
torch2vk 负责打开 RADV RGP/SQTT、给 dispatch 打稳定标签、收集 driver 产物、把 RGP 文件映射回 RuntimeSession.dispatch_records。
真正的 thread trace start/stop/readback/dump 继续交给 Mesa RADV。
```

## Mesa 侧调查结论

`third_party/mesa` 来自 `~/projects/agentorch/third_party/mesa`，submodule URL 是
`https://github.com/taowen/agentorch-mesa.git`，当前提交是
`39485e5803d48790bbf0946411782200ddc168cd`。

RADV 的 SQTT 入口不是 Vulkan 公共 API，而是 Mesa driver 内部的 RGP trace mode：

```text
MESA_VK_TRACE=rgp
  -> vk_instance.trace_mode 包含 RADV_TRACE_MODE_RGP
  -> radv_device_init_rgp()
  -> radv_sqtt_init()
  -> device dispatch table 挂上 sqtt_device_entrypoints
```

关键文件：

```text
third_party/mesa/src/vulkan/runtime/vk_instance.c
third_party/mesa/src/amd/vulkan/radv_instance.c
third_party/mesa/src/amd/vulkan/radv_device.c
third_party/mesa/src/amd/vulkan/radv_sqtt.c
third_party/mesa/src/amd/vulkan/layers/radv_sqtt_layer.c
third_party/mesa/src/amd/common/ac_sqtt.c
third_party/mesa/src/amd/common/ac_rgp.c
```

### 初始化

`radv_sqtt_init()` 做三件事：

1. 分配 SQTT buffer。默认每个 shader engine 32 MiB，可由 `RADV_THREAD_TRACE_BUFFER_SIZE`
   覆盖；dGPU 上会使用 staging buffer 提高 CPU readback 速度。
2. 初始化 queue event/timestamp command pool，用于给 submit/present 事件插 GPU timestamp。
3. 初始化 `ac_sqtt` 元数据表，包括 code object、loader event、PSO correlation、queue event、
   clock calibration。

这个 fork 里 `ac_check_profile_state()` 直接返回 `false`，所以不会因为 GPU 不在 profiling pstate
而拒绝捕获。

### 开始录制

`radv_sqtt_start_capturing(queue)`：

1. reserve VMID；
2. 采样 CPU/GPU calibrated timestamp 和当前 SCLK/MCLK；
3. 录制并提交一个 start command buffer；
4. 标记 `device->sqtt_enabled = true`。

start command buffer 里主要做：

```text
wait idle
disable clock gating
enable SQG/SPI events
optional SPM setup
ac_sqtt_emit_start()
optional SPM start
```

`ac_sqtt_emit_start()` 是真正配置硬件 thread trace 的地方。它逐个 SE 写
`SQ_THREAD_TRACE_BUF*`、mask、token mask、control register，然后发 `THREAD_TRACE_START`
事件。不同 GFX generation 的寄存器路径在 `ac_sqtt.c` 里分支处理。

### 停止录制

`radv_sqtt_stop_capturing(queue, submit_ordinal, profile_tag)`：

1. 录制并提交 stop command buffer；
2. `QueueWaitIdle` 等待 stop 和 staging copy 完成；
3. unreserve VMID；
4. `radv_get_sqtt_trace()` 从 buffer 中解析每个 SE 的 info/data；
5. `ac_dump_rgp_capture()` 写 `.rgp` 文件；
6. 清理本次 capture 的 clock calibration、queue event、timestamp cmdbuf。

stop command buffer 里主要做：

```text
wait idle
optional SPM stop
ac_sqtt_emit_stop()
ac_sqtt_emit_wait()
disable SQG/SPI events
restore clock gating
optional copy SQTT/SPM buffer to staging buffer
```

`ac_sqtt_emit_wait()` 等待 `FINISH_DONE` / `BUSY` 状态，并用 `COPY_DATA` 把
WPTR/status/counter 写回 SQTT info buffer。`radv_get_sqtt_trace()` 后面就靠这些 info 判断
buffer 是否完整、每个 SE 写了多少 thread trace 字节。

### RGP 文件写出

`ac_dump_rgp_capture()` 写到 `/tmp/<process>_<timestamp>_frameN.rgp` 或
`/tmp/<process>_<timestamp>_submitN.rgp`。

文件内容由 `ac_sqtt_dump_data()` 组织，主要 chunk 是：

```text
SQTT file header
CPU info
ASIC info
API info
code object database
code object loader events
PSO correlation
queue event timings
clock calibration
每个 SE 的 SQTT desc + SQTT data
optional SPM / derived SPM
```

硬件 thread trace 原始数据大小是 `info->cur_offset * 32` 字节。

### QueueSubmit 路径

`MESA_VK_TRACE_PER_SUBMIT=true` 时，`sqtt_QueueSubmit2()` 会在每个 submit 前后自动 start/stop
SQTT，并为该 submit 产出一个 `.rgp`。它还会把原始 submit 包装成：

```text
pre timestamp cmdbuf
用户 command buffer
post timestamp cmdbuf
```

这样 RGP 可以显示 queue event timings。

`torch2vk` 当前用的是 `vkQueueSubmit`，但 Mesa common implementation 会把 `vkQueueSubmit`
转成 `device->dispatch_table.QueueSubmit2(...)`。实际接入时仍要做一次 smoke test，确认我们的
Python Vulkan binding 走到了 RADV SQTT layer 的 `sqtt_QueueSubmit2()`。

### Frame trigger 不适合 torch2vk 主路径

Mesa 的非 per-submit capture 主要由 WSI present 路径触发：

```text
MESA_VK_TRACE_FRAME
MESA_VK_TRACE_TRIGGER
hotkey trigger
  -> wsi_common_queue_present()
  -> device->capture_trace()
  -> device->sqtt_triggered = true
  -> 下次 present 周期 start/stop
```

`torch2vk` 是 compute-only，没有 swapchain present。因此我们不能把 `MESA_VK_TRACE_FRAME`
作为主路径。MVP 应以 `MESA_VK_TRACE_PER_SUBMIT=true` 为主。

### Agentorch fork 增加的归因能力

这个 Mesa fork 额外写出 driver artifacts：

```text
AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR/
  driver-session.json
  capture-sequence.jsonl
  dispatch-sequence.jsonl
  <pipeline_hash>/
    pipeline-debug.json
    compiler-native-disasm.s
```

关键逻辑：

1. `CmdBeginDebugUtilsLabelEXT` 如果 label 前缀是
   `agentorch-profile-submit:`，就把后面的 payload 存到 command buffer 的
   `agentorch_profile_tag` / `agentorch_profile_submit_tag`。
2. 每个 compute dispatch 会追加一条 `dispatch-sequence.jsonl`，记录
   `submit_ordinal`、`dispatch_index`、`pipeline_hash`、`pipeline_name`、`profile_tag`。
3. 每个 RGP capture 会追加一条 `capture-sequence.jsonl`，记录
   `submit_ordinal`、`capture_path`、`profile_tag`。
4. pipeline 创建/命名时导出 pipeline debug JSON 和 compiler-native disassembly。

这正好是 `torch2vk` 需要的桥：RGP 是底层性能文件，`dispatch-sequence.jsonl` 和
`capture-sequence.jsonl` 能把它映射回 Python runtime 的 frame/dispatch/shader。

## 对 torch2vk 的判断

我们这边不应该做的事：

1. 不在 Python/CFFI 里直接写 AMD SQTT 寄存器。
2. 不 fork 一套和 RADV 重复的 RGP file writer。
3. 不让模型目录感知 SQTT。
4. 不把 SQTT capture 设计成 replay 的必要条件。

我们应该做的事：

1. 用 `MESA_VK_TRACE=rgp` 和 `MESA_VK_TRACE_PER_SUBMIT=true` 打开 RADV SQTT。
2. 在 `RuntimeSession` / `ComputePipeline` 层给 Vulkan command buffer 打 debug label。
3. 使用稳定的 profile tag，把 driver 产物和 `DispatchRecord` 关联起来。
4. 把 Mesa fork 的 driver artifacts 归档到 `artifact_dir` 下。
5. 生成一个 `torch2vk` 自己的 manifest，作为查询入口。

重要约束：

1. 环境变量必须在创建 Vulkan instance/device 之前设置。
2. 只支持 AMD RADV + 带 SQTT/RGP 的 Mesa build。
3. Mesa 需要启用 libelf，否则 `ac_dump_rgp_capture()` 不能写 RGP。
4. RGP/SQTT 开销很大，只能作为 profiling/debug mode。
5. per-submit capture 会让每个 submit 都被 start/stop 包围，不能用于性能基准，只用于归因分析。

## 最终理想形态

最终用户体验应该是：

```python
from pathlib import Path

from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.sqtt import SqttCaptureConfig


sqtt = SqttCaptureConfig(
    enabled=True,
    root=Path(".cache/torch2vk/sqtt/qwen3-asr-decode"),
    mode="per_submit",
    trace_buffer_mib=256,
    queue_events=True,
    instruction_timing=True,
    export_driver_artifacts=True,
)

with RuntimeSession.open(device_index=0, model_dir=model_dir, sqtt=sqtt) as rt:
    run_qwen3_asr_decode(rt, tensors, replay_mode="force_record")

manifest = rt.sqtt_manifest()
```

这个 API 的含义：

```text
RuntimeSession.open(..., sqtt=...)
  -> 在 Vulkan instance 创建前设置 Mesa/RADV/Agentorch env
  -> 创建 VulkanDevice
  -> 开启 debug utils label support
  -> 在 dispatch/replay command recording 期间自动打 label
  -> 结束时 flush/collect driver artifacts
  -> 写 torch2vk-sqtt-session.json
```

推荐的 artifact layout：

```text
.cache/torch2vk/sqtt/<run-id>/
  torch2vk-sqtt-session.json
  torch2vk-dispatch-map.jsonl
  driver/
    driver-session.json
    capture-sequence.jsonl
    dispatch-sequence.jsonl
    <pipeline_hash>/
      pipeline-debug.json
      compiler-native-disasm.s
  rgp/
    submit-000001.rgp
    submit-000002.rgp
    ...
```

`driver/` 是 `AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR` 指向的目录。`rgp/` 可以先不强制搬运；
第一版 manifest 记录 `/tmp/...rgp` 的原始路径即可。第二版再把文件复制/硬链接到 run artifact
目录，避免 `/tmp` 被清理。

### Tag 规范

profile tag 要稳定、可读、可 join。建议第一版使用 JSON 不方便，因为 label 长度和可读性都不好。
用分号分隔的 key-value 字符串即可：

```text
frame=qwen3_asr.text_decode;phase=eager;dispatch=42;shader=text_attention_decode_f32
```

实际传给 Vulkan debug label 的字符串：

```text
agentorch-profile-submit:frame=qwen3_asr.text_decode;phase=eager;dispatch=42;shader=text_attention_decode_f32
```

规则：

1. `frame` 来自当前 `FrameContext.frame`。
2. `phase` 是 `eager`、`record`、`replay` 或 `warmup`。
3. `dispatch` 是 `RuntimeSession` 的全局 dispatch index，replay 中可同时记录
   `replay_dispatch`。
4. `shader` 是 `ShaderVariant.name`。
5. 如果 tag 太长，保留完整信息到 `torch2vk-dispatch-map.jsonl`，label 中使用短 hash：
   `tag=<sha1-12>`。

### Capture 模式

我们需要三个层次：

```text
per_submit
  每个 vkQueueSubmit 一个 RGP。MVP 主路径。

per_dispatch
  eager dispatch 当前就是一个 dispatch 一个 command buffer 一个 submit，因此 per_submit 等价于 per_dispatch。
  如果未来 eager 合并 submit，则需要 RuntimeSession 临时禁用合并。

replay_submit
  replay plan 整体录成一个 command buffer/submit，一个 RGP 包含整条 replay dispatch sequence。
  通过 command labels 和 dispatch-sequence.jsonl 做内部归因。
```

不把 `frame` 作为第一版主模式，因为 compute-only 没有 present trigger。

## 实现路径

### Phase 0：手工验证 Mesa fork

目标是证明当前 Python 进程能触发 RADV SQTT layer。

仓库提供两个入口脚本：

```bash
scripts/build-mesa-radv.sh
scripts/profile-sqtt.sh --root .cache/torch2vk/sqtt/manual -- <command>
```

`scripts/build-mesa-radv.sh` 默认把 `third_party/mesa` 编译并安装到：

```text
.cache/torch2vk/mesa-build-venv
.cache/torch2vk/vulkan-sdk
.cache/torch2vk/mesa-radv-build
.cache/torch2vk/mesa-radv
```

在 Fedora Atomic/Bazzite 这类不适合 `dnf install *-devel` 的系统上，做法和
`agentorch` 一样：不修改宿主系统。`scripts/build-mesa-radv.sh` 内部用 `dnf download`
下载 RPM，然后把 `vulkan-headers`、`vulkan-loader-devel`、`spirv-tools-devel`、
`spirv-headers-devel`、`libdrm/libdrm-devel`、`expat/expat-devel` 解到
`.cache/torch2vk/vulkan-sdk`，把 `glslc/glslangValidator` 解到 `.cache/torch2vk/bin`。
同一个脚本再创建 `.cache/torch2vk/mesa-build-venv`，安装
`meson / ninja / mako / PyYAML / packaging / setuptools`，并在 Meson 环境里设置：

```bash
export PKG_CONFIG_PATH=<repo>/.cache/torch2vk/vulkan-sdk/usr/lib64/pkgconfig:<repo>/.cache/torch2vk/vulkan-sdk/usr/share/pkgconfig:${PKG_CONFIG_PATH:-}
export PKG_CONFIG_SYSROOT_DIR=<repo>/.cache/torch2vk/vulkan-sdk
export PATH=<repo>/.cache/torch2vk/mesa-build-venv/bin:<repo>/.cache/torch2vk/bin:${PATH}
```

运行 profile 时还要带上 Mesa 链接到的 LLVM runtime libdir。例如当前 Bazzite/Fedora
Atomic 环境会自动发现：

```bash
export LLVM_CONFIG=/usr/lib64/rocm/llvm/bin/llvm-config
export LD_LIBRARY_PATH=/usr/lib64/rocm/llvm/lib:<repo>/.cache/torch2vk/vulkan-sdk/usr/lib64:<repo>/.cache/torch2vk/mesa-radv/lib:${LD_LIBRARY_PATH:-}
```

Mesa 配置是 RADV-only：

```text
-Dvulkan-drivers=amd
-Dgallium-drivers=
-Dopengl=false
-Dglx=disabled
-Degl=disabled
-Dgbm=disabled
-Dllvm=enabled
-Dshared-llvm=enabled
```

`scripts/profile-sqtt.sh` 会检查本地 RADV ICD 是否存在；默认缺失时自动调用
`scripts/build-mesa-radv.sh`。运行用户命令前，它会设置：

```bash
export VK_DRIVER_FILES=<repo>/.cache/torch2vk/mesa-radv/share/vulkan/icd.d/radeon_icd*.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export LD_LIBRARY_PATH=<repo>/.cache/torch2vk/mesa-radv/lib:<repo>/.cache/torch2vk/mesa-radv/lib64:${LD_LIBRARY_PATH:-}
export MESA_VK_TRACE=rgp
export MESA_VK_TRACE_PER_SUBMIT=true
export RADV_THREAD_TRACE_BUFFER_SIZE=$((256 * 1024 * 1024))
export RADV_THREAD_TRACE_QUEUE_EVENTS=true
export RADV_THREAD_TRACE_INSTRUCTION_TIMING=true
export AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR=<root>/driver
export AGENTORCH_RADV_EXPORT_TAG=<run-id>
export MESA_SHADER_CACHE_DIR=<root>/mesa-shader-cache
```

然后运行一个最小 compute dispatch，验证：

```text
.cache/torch2vk/sqtt/manual/rgp/submit1.rgp 存在且非空
.cache/torch2vk/sqtt/manual/driver/capture-sequence.jsonl 存在
.cache/torch2vk/sqtt/manual/driver/dispatch-sequence.jsonl 存在
dispatch-sequence.jsonl 能看到 pipeline_hash / pipeline_name
.cache/torch2vk/sqtt/manual/driver/<pipeline_hash>/pipeline-debug.json 存在
.cache/torch2vk/sqtt/manual/driver/<pipeline_hash>/compiler-native-disasm.s 存在
```

如果没有 RGP：

1. 用 `scripts/profile-sqtt.sh --print-env --dry-run -- <command>` 确认 `VK_DRIVER_FILES` 指向
   `.cache/torch2vk/mesa-radv/.../radeon_icd*.json`。
2. 确认进程加载的是 `third_party/mesa` build 出来的 RADV，而不是系统 Mesa。
3. 确认 `MESA_VK_TRACE=rgp` 在 Vulkan instance 创建前已经设置。
4. 确认 Mesa build 启用了 libelf。
5. 确认 `vkQueueSubmit` 是否真的转到了 `sqtt_QueueSubmit2()`；必要时把
   `queue_submission.py` 升级为直接调用 `vkQueueSubmit2`。

### Phase 1：最小 SqttRecorder

新增 runtime 内部模块：

```text
src/torch2vk/runtime/sqtt.py
```

推荐对象：

```python
@dataclass(frozen=True, slots=True)
class SqttCaptureConfig:
    enabled: bool = False
    root: Path = Path(".cache/torch2vk/sqtt/default")
    mode: Literal["per_submit", "per_dispatch", "replay_submit"] = "per_submit"
    trace_buffer_mib: int = 256
    queue_events: bool = True
    instruction_timing: bool = True
    export_driver_artifacts: bool = True


class SqttRecorder:
    def prepare_environment(self) -> None: ...
    def profile_tag_for_dispatch(self, record_context: DispatchTagContext) -> str: ...
    def begin_label(self, device: VulkanDevice, command_buffer: object, tag: str) -> None: ...
    def end_label(self, device: VulkanDevice, command_buffer: object) -> None: ...
    def collect(self) -> SqttManifest: ...
```

`prepare_environment()` 只允许在 `VulkanDevice` 创建前调用。如果发现 device 已经创建，直接报错。
不要悄悄设置无效环境。

环境变量映射：

```text
MESA_VK_TRACE=rgp
MESA_VK_TRACE_PER_SUBMIT=true
RADV_THREAD_TRACE_BUFFER_SIZE=<trace_buffer_mib MiB in bytes>
RADV_THREAD_TRACE_QUEUE_EVENTS=true/false
RADV_THREAD_TRACE_INSTRUCTION_TIMING=true/false
AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR=<root>/driver
AGENTORCH_RADV_EXPORT_TAG=<run id>
```

### Phase 2：给 eager dispatch 打 label

当前 `ComputePipeline.dispatch()` 自己分配 command buffer、begin/end、submit。最小改法是在
`record_dispatch()` 前后插入 label：

```text
vkBeginCommandBuffer
device.begin_command_label("agentorch-profile-submit:<tag>")
record_dispatch
device.end_command_label
record_eager_completion_barrier
vkEndCommandBuffer
submit
```

但 tag 需要 `RuntimeSession` 的 frame/dispatch/shader 信息，而 `ComputePipeline.dispatch()`
现在不知道这些。所以推荐把 eager dispatch 改为由 `RuntimeSession` 显式录制 command buffer：

```text
RuntimeSession.dispatch()
  -> resolve/materialize/bind
  -> allocate command buffer
  -> begin command buffer
  -> sqtt begin label
  -> pipeline.record_dispatch(...)
  -> sqtt end label
  -> pipeline.record_eager_completion_barrier(...)
  -> end command buffer
  -> submit
  -> append DispatchRecord
```

这样 `RuntimeSession` 能在 append `DispatchRecord` 前就知道即将使用的 dispatch index。
如果要保持 `ComputePipeline.dispatch()`，也可以给它加可选参数：

```python
debug_label: str | None = None
```

MVP 可以先走可选参数，后续 replay/capture 再统一到 RuntimeSession recording path。

### Phase 3：生成 torch2vk manifest

`SqttRecorder.collect()` 读取：

```text
<root>/driver/capture-sequence.jsonl
<root>/driver/dispatch-sequence.jsonl
RuntimeSession.dispatch_records
```

输出：

```text
<root>/torch2vk-sqtt-session.json
<root>/torch2vk-dispatch-map.jsonl
```

`torch2vk-dispatch-map.jsonl` 每行建议包含：

```json
{
  "torch2vk_dispatch_index": 42,
  "frame": "qwen3_asr.text_decode",
  "shader": "text_attention_decode_f32",
  "profile_tag": "frame=qwen3_asr.text_decode;phase=eager;dispatch=42;shader=text_attention_decode_f32",
  "submit_ordinal": 43,
  "rgp_path": "/tmp/python_2026.05.05_12.00.00_submit43.rgp",
  "pipeline_hash": 123456789,
  "pipeline_name": "torch2vk:text_attention_decode_f32:..."
}
```

Join 规则：

```text
capture-sequence.jsonl.profile_tag == dispatch-sequence.jsonl.profile_tag
dispatch-sequence.jsonl.profile_tag == torch2vk generated profile_tag
```

如果 label 因为长度被 hash，join 使用 `tag=<sha1>`，完整 tag 存在 `torch2vk-dispatch-map.jsonl`。

### Phase 4：replay capture

Replay plan 录制 command buffer 时也要打 label。区别是 replay 通常一个 submit 包含多个 dispatch：

```text
begin replay command buffer
for replay dispatch:
  begin label(agentorch-profile-submit:<tag>)
  record_dispatch or record_indirect_dispatch
  end label
  barrier
end command buffer
submit once
```

在 `MESA_VK_TRACE_PER_SUBMIT=true` 下，这会生成一个 replay submit 的 RGP。RGP 内部的 user event
和 `dispatch-sequence.jsonl` 用于区分每个 replay dispatch。

Replay manifest 需要额外记录：

```text
replay_plan_id
replay_dispatch_index
source_record_dispatch_index
source_frame
source_shader
```

### Phase 5：CLI / pytest 集成

推荐提供一个显式入口，而不是要求用户记住所有 env：

```bash
uv run torch2vk-sqtt \
  --root .cache/torch2vk/sqtt/qwen3-asr-decode \
  --trace-buffer-mib 256 \
  -- pytest tests/test_qwen3_asr.py -k decode
```

CLI 做两件事：

1. 设置 env；
2. exec 用户命令，保证 env 在 Python import Vulkan 之前生效。

pytest 可以只测非硬件部分：

```text
SqttCaptureConfig -> env mapping
profile tag generation
driver JSONL join
manifest output
debug label calls can be mocked
```

硬件 smoke test 单独标记，例如 `pytest -m sqtt_hardware`，默认不跑。

## MVP 验收标准

MVP 完成时应满足：

1. `RuntimeSession.open(..., sqtt=SqttCaptureConfig(enabled=True, ...))` 能设置正确 env，并要求在
   Vulkan device 创建前生效。
2. eager compute dispatch 能生成 `.rgp`。
3. `AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR` 下能生成 `capture-sequence.jsonl` 和
   `dispatch-sequence.jsonl`。
4. `torch2vk-sqtt-session.json` 能列出每个 `.rgp` 对应的 frame、dispatch index、shader。
5. 关闭 SQTT 时，普通 runtime 行为完全不变。

## 风险和待确认

1. Python Vulkan binding 的 `vkQueueSubmit` 是否稳定走 Mesa common `QueueSubmit2` 转换，需要 smoke test。
2. 运行环境是否能保证加载 submodule build 出来的 RADV，而不是系统 RADV。
3. `/tmp` 下 RGP 文件生命周期不稳定，需要尽快把 capture 文件复制或硬链接进 artifact root。
4. 每 dispatch 一个 RGP 的数据量会很大，decode 长序列需要采样或只抓指定范围。
5. Label payload 长度如果过长，RGP/driver artifact 是否截断需要实测；准备 hash fallback。
6. `MESA_VK_TRACE_PER_SUBMIT=true` 下 start/stop 成本很高，任何性能数字都不能当真实运行性能。

## 推荐先做的最短路径

第一步不要改 replay。先让 eager dispatch 可录：

```text
1. 先用 `scripts/profile-sqtt.sh` 建立外部 profile 入口，确保 profile 时走本地 Mesa RADV。
2. 加 SqttCaptureConfig / SqttRecorder，把脚本里的 env mapping 收进 Python/CLI。
3. RuntimeSession.open(..., sqtt=...) 在 VulkanDevice 前设置 env。
4. ComputePipeline.dispatch(debug_label=...) 在 command buffer 内包 debug label。
5. RuntimeSession.dispatch() 生成 agentorch-profile-submit tag 并传下去。
6. SqttRecorder.collect() join driver JSONL，输出 torch2vk manifest。
7. 用一个单 dispatch shader 做硬件 smoke test。
```

这个路径最短，因为当前 eager dispatch 已经是 one command buffer + one queue submit；配合
`MESA_VK_TRACE_PER_SUBMIT=true`，天然就是 per-dispatch RGP capture。
