# SQTT 录制调查

本文只记录一件事：`torch2vk` 如何借助本仓库的 Mesa RADV fork 录制 SQTT/RGP。

核心结论：

```text
SQTT start/stop/readback/RGP 写文件都交给 Mesa RADV。
torch2vk 只负责：
1. 确保 profile 进程使用本地 Mesa RADV。
2. 在 command buffer 里打稳定 label。
3. 保存 runtime dispatch manifest。
4. 把 Mesa 写出的 RGP 和 driver artifacts 归档到一次 run 目录。
```

## Mesa 侧机制

RADV 没有通过 Vulkan 公共 API 暴露 SQTT。它通过 Mesa 的 RGP trace mode 打开：

```text
MESA_VK_TRACE=rgp
  -> RADV 初始化 RGP/SQTT
  -> queue submit 路径被 SQTT layer 包装
```

关键代码在 Mesa：

```text
third_party/mesa/src/amd/vulkan/radv_sqtt.c
third_party/mesa/src/amd/vulkan/layers/radv_sqtt_layer.c
third_party/mesa/src/amd/common/ac_sqtt.c
third_party/mesa/src/amd/common/ac_rgp.c
```

`MESA_VK_TRACE_PER_SUBMIT=true` 时，每次 `vkQueueSubmit` 会被包成：

```text
start SQTT
submit user command buffers
stop SQTT
dump one .rgp file
```

`torch2vk` 是 compute-only，没有 swapchain present，所以不适合依赖
`MESA_VK_TRACE_FRAME`。实际录制主路径应使用 per-submit capture。

## agentorch Mesa fork 增加的产物

这个 fork 除了写 `/tmp/*.rgp`，还会在指定目录写 driver artifacts：

```text
AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR/
  driver-session.json
  capture-sequence.jsonl
  dispatch-sequence.jsonl
  <pipeline_hash>/
    pipeline-debug.json
    compiler-native-disasm.s
```

这些文件的作用：

1. `capture-sequence.jsonl`：submit ordinal 到 RGP 文件的映射。
2. `dispatch-sequence.jsonl`：submit 内每个 compute dispatch 的 pipeline 和 label。
3. `pipeline-debug.json` / `compiler-native-disasm.s`：pipeline 的 driver 侧信息和反汇编。

fork 还识别 debug utils label：

```text
agentorch-profile-submit:frame=<frame>;shader=<shader>;dispatch=<index>
```

如果 command buffer 中出现这个 label，driver artifacts 会记录 label payload。payload 直接使用已有的
`FrameContext.frame`、`ShaderVariant.name` 和 `DispatchRecord.index`，不再引入单独的 tag 概念。
这就是把 RGP 映射回 `RuntimeSession.dispatch_records` 的桥。

## 当前脚本

本仓库保留两个脚本。

`scripts/build-mesa-radv.sh`：

1. 编译 `third_party/mesa`。
2. 安装到 `.cache/torch2vk/mesa-radv`。
3. 只构建 RADV Vulkan ICD，不构建 OpenGL/Gallium 其它驱动。
4. 在 Fedora Atomic/Bazzite 上不安装系统依赖，而是用 `dnf download` 下载 RPM，并解包到
   `.cache/torch2vk/vulkan-sdk`。

`scripts/profile-sqtt.sh`：

1. 找到或自动构建本地 RADV ICD。
2. 在运行用户命令前设置 Vulkan/Mesa/SQTT 环境变量。
3. 设置 `AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR=<root>/driver`。
4. 运行命令后把新增 `/tmp/*.rgp` 复制到 `<root>/rgp/`。

关键环境变量：

```bash
export VK_DRIVER_FILES=<repo>/.cache/torch2vk/mesa-radv/share/vulkan/icd.d/radeon_icd*.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export LD_LIBRARY_PATH=<repo>/.cache/torch2vk/mesa-radv/lib:${LD_LIBRARY_PATH:-}
export MESA_VK_TRACE=rgp
export MESA_VK_TRACE_PER_SUBMIT=true
export RADV_THREAD_TRACE_BUFFER_SIZE=$((256 * 1024 * 1024))
export RADV_THREAD_TRACE_QUEUE_EVENTS=true
export RADV_THREAD_TRACE_INSTRUCTION_TIMING=true
export AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR=<root>/driver
export AGENTORCH_RADV_EXPORT_TAG=<run-id>
export MESA_SHADER_CACHE_DIR=<root>/mesa-shader-cache
```

这些环境变量必须在 Vulkan instance 创建前设置。已经创建 `VulkanDevice` 后再设置是无效的。

## 验证标准

一次有效 SQTT 录制至少应产生：

```text
<root>/rgp/submit1.rgp
<root>/driver/driver-session.json
<root>/driver/capture-sequence.jsonl
<root>/driver/dispatch-sequence.jsonl
<root>/driver/<pipeline_hash>/pipeline-debug.json
<root>/driver/<pipeline_hash>/compiler-native-disasm.s
```

`dispatch-sequence.jsonl` 中应该能看到 pipeline 信息；接入 runtime label 后，还应该能看到类似下面的
label payload：

```text
frame=qwen3_asr.text_decode;shader=text_attention_decode_f32;dispatch=42
```

## 常见问题

如果没有 RGP：

1. 用 `scripts/profile-sqtt.sh --print-env --dry-run -- <command>` 检查 `VK_DRIVER_FILES`。
2. 确认环境变量在 Python 创建 Vulkan instance 前已经设置。
3. 确认加载的是 `.cache/torch2vk/mesa-radv`，不是系统 Mesa。
4. 确认命令确实触发了 `vkQueueSubmit`。
5. 确认 Mesa build 有 libelf 支持，否则 RGP 写出会失败。

如果有 RGP 但无法映射回 dispatch：

1. 检查 command buffer 是否打了 `agentorch-profile-submit:frame=...;shader=...;dispatch=...`。
2. 检查 `dispatch-sequence.jsonl` 是否记录了同一个 label payload。
3. 检查 `torch2vk` 是否写出了 runtime dispatch manifest。
