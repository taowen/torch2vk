# 使用 SQTT 的 Profiler

这份文档定义 `torch2vk` 如何录制 SQTT，并把 RGP/driver artifacts 映射回 runtime dispatch。
它建立在 `docs/profile-without-sqtt.md` 的 dispatch manifest 之上。

核心目标：

```text
给定一次 record 模式下的 profile run，
能知道每个 RGP submit 对应 torch2vk 的哪个 frame / dispatch / shader / pipeline。
```

更深入的 RGP UI 分析、ISA 热点分析可以在这个映射之后做，但不是录制链路本身的一部分。

当前只支持 record 模式录制 SQTT。replay 不支持 SQTT；replay 性能分析走普通 profiler 的 timestamp
路径。

SQTT 不能全局打开后跑完整 workload。它的开销和 RGP dump 都太大。实际录制必须精确到目标
record dispatch：runtime 给每个 record dispatch 打稳定 label，Mesa 只对匹配 filter 的 submit
执行 SQTT start/stop。

## 输入和输出

输入：

1. 本地 Mesa RADV fork。
2. `scripts/profile-sqtt.sh` 启动的用户命令。
3. `torch2vk` runtime 写出的 `dispatches.jsonl`。
4. command buffer 里的 `agentorch-profile-submit:frame=...;shader=...;dispatch=...` label。

输出：

```text
.cache/torch2vk/sqtt/<run-id>/
  run.json
  dispatches.jsonl
  attribution.jsonl
  report.md
  report.json
  rgp/
    submit1.rgp
    submit2.rgp
  driver/
    driver-session.json
    capture-sequence.jsonl
    dispatch-sequence.jsonl
    <pipeline_hash>/
      pipeline-debug.json
      compiler-native-disasm.s
  debug/                         # 只有 --debug-artifacts 时生成
    pipeline-attribution.json
    source-isa-sqtt-hotspots*.json
```

`dispatches.jsonl` 来自普通 profiler，是 runtime 语义。`driver/` 和 `rgp/` 来自 Mesa。
`attribution.jsonl` 是 join 后的结果。默认要看的入口是 `report.md`：它只保留目标 dispatch、
coverage、资源用量、top source line、top ISA range 和产物清单。`report.json` 是同一份信息的结构化
版本。`debug/` 下的完整 pipeline attribution 和 source/ISA decoder JSON 只在需要追查 postprocess
细节时用 `python -m torch2vk.sqtt --root <root> --debug-artifacts` 生成。

## 运行方式

SQTT 环境变量必须在 Vulkan instance 创建前设置，所以主入口应该是外部脚本：

```bash
scripts/profile-sqtt.sh \
  --root .cache/torch2vk/sqtt/qwen3-decode-linear \
  --capture-filter 'frame=qwen3_asr.text_decode.0000;shader=qwen3_asr_text_linear_nobias_t1_f32;dispatch=123' \
  -- \
  uv run pytest tests/test_qwen3_asr.py::test_qwen3_asr_record_decode_one_step_for_sqtt -s
```

这个脚本负责：

1. 确保本地 RADV ICD 存在，不存在就调用 `scripts/build-mesa-radv.sh`。
2. 设置 `VK_DRIVER_FILES` / `VK_ICD_FILENAMES` 指向本地 Mesa。
3. 设置 `MESA_VK_TRACE=rgp` 和 `MESA_VK_TRACE_PER_SUBMIT=true`。
4. 设置 `AGENTORCH_RADV_SQTT_PROFILE_TAG_FILTER=<filter>`，只录匹配 label 的 submit。
5. 设置 `AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR=<root>/driver`。
6. 设置 `TORCH2VK_PROFILE_RUN_DIR=<root>`，让 runtime manifest 写到同一个目录。
7. 运行用户命令。
8. 复制 RGP 到 `<root>/rgp/`，生成 `attribution.jsonl`、`report.md` 和 `report.json`。

如果只知道 shader，decode 层内 GEMM 可以先用 `--shader qwen3_asr_text_linear_nobias_t1_f32`，
decode lm_head greedy selection 可以用 `--shader qwen3_asr_text_lm_head_select_partial_t1_f32`。
如果 RGP 仍然太多，先跑普通 profiler 找到目标 `dispatch_index`，再用完整 `--capture-filter` 精确到单个 dispatch。
脚本不接受空 filter，避免误把 SQTT 对完整 workload 全局打开。

## runtime 需要做什么

SQTT 打开时，`torch2vk` runtime 仍只做两件简单的事。

第一，写 dispatch manifest。字段见 `docs/profile-without-sqtt.md`。

第二，在 command buffer 中包 debug label：

```text
agentorch-profile-submit:frame=<frame>;shader=<shader>;dispatch=<index>
```

record 模式下，每个 `RuntimeSession.dispatch()` 录制和提交自己的 command buffer：

```text
begin command buffer
begin label(agentorch-profile-submit:frame=<frame>;shader=<shader>;dispatch=<index>)
record dispatch
end label
barrier
end command buffer
submit
```

label payload 只使用已有字段：

```text
frame=qwen3_asr.text_decode;shader=text_attention_decode_f32;dispatch=42
```

不要把 `LogicalTensor.name`、shape、pipeline hash 等信息塞进 label。这些完整语义已经在
`dispatches.jsonl` 里，label 只负责把 driver 记录映射回 manifest。

Mesa fork 看到 label 后不会立刻录制。它先用
`AGENTORCH_RADV_SQTT_PROFILE_TAG_FILTER` 做 substring match；只有匹配的 submit 才会 start SQTT、
提交 command buffer、stop SQTT、dump RGP。未匹配 submit 直接走正常 `QueueSubmit2`。

## join 规则

postprocess 只需要做一个明确的 join，不需要猜测：

```text
parse driver/dispatch-sequence.jsonl.profile_tag
  -> frame / shader / dispatch

dispatches.jsonl.frame == parsed.frame
dispatches.jsonl.shader == parsed.shader
dispatches.jsonl.dispatch_index == parsed.dispatch

driver/dispatch-sequence.jsonl.submit_ordinal
  == driver/capture-sequence.jsonl.submit_ordinal

driver/dispatch-sequence.jsonl.pipeline_hash
  -> driver/<pipeline_hash>/pipeline-debug.json
```

这里的 `profile_tag` 只是 Mesa fork 在 JSONL 里的字段名；`torch2vk` 不再设计单独的 tag。
字段内容就是 debug label payload。

join 成功后，每行 `attribution.jsonl` 应包含：

```json
{
  "dispatch_index": 42,
  "frame": "qwen3_asr.text_decode",
  "phase": "record",
  "shader": "text_attention_decode_f32",
  "submit_ordinal": 7,
  "driver_dispatch_index": 0,
  "pipeline_hash": "123456",
  "pipeline_debug_name": "agp.text_attention_decode_f32.0123456789abcdef",
  "pipeline_identity_sha256": "...",
  "shader_spv_sha256": "...",
  "rgp_path": "rgp/submit7.rgp",
  "pipeline_debug_path": "driver/123456/pipeline-debug.json",
  "disasm_path": "driver/123456/compiler-native-disasm.s"
}
```

不要用 RGP 文件排序、耗时相似度或 shader 文件名作为主要 join key。

## 只支持 record 模式

SQTT profiler 只支持 record 模式。这里的 record 模式指 runtime 真实执行 `RuntimeSession.dispatch()`，
并在该 dispatch 的 command buffer 里写 label。

record 模式当前是一 dispatch 一 submit。配合 `MESA_VK_TRACE_PER_SUBMIT=true`，通常就是一 dispatch
一个 RGP，driver artifacts 中的 submit、dispatch 和 `DispatchRecord.index` 能直接 join。

replay 不支持 SQTT。原因是 replay 使用预录 command buffer，和 record 阶段的 `DispatchRecord` 不是同一个
录制时机；即使能录出一个 replay submit 的 RGP，也不能作为当前 SQTT profiler 的受支持输出。需要分析 replay
性能时，使用不依赖 SQTT 的普通 profiler timestamp；需要硬件归因时，在 record 模式下复现同一段 workload。

## 失败条件

这些情况应直接失败，而不是生成空报告：

1. `VK_DRIVER_FILES` 没有指向本地 `.cache/torch2vk/mesa-radv`。
2. `MESA_VK_TRACE` 不是 `rgp`。
3. `MESA_VK_TRACE_PER_SUBMIT` 没有打开。
4. filter 没有匹配任何 record dispatch。
5. `dispatches.jsonl` 缺失或为空。
6. `driver/capture-sequence.jsonl` 缺失或为空。
7. `driver/dispatch-sequence.jsonl` 缺失或为空。
8. RGP 文件缺失或为空。
9. driver label payload 无法解析，或无法按 frame/shader/dispatch join 到 manifest。
10. pipeline hash 能 join，但 pipeline name 对不上。
11. manifest 中出现 `phase != "record"` 的 dispatch。

失败信息写进 `run.json` 或 stderr，便于直接定位。

## 实现顺序

推荐顺序：

1. `scripts/profile-sqtt.sh` 设置 `TORCH2VK_PROFILE_RUN_DIR=<root>` 和 capture filter。
2. 普通 profiler 写 manifest。
3. record 模式的 `RuntimeSession.dispatch()` 录制 debug label。
4. Mesa 只 capture 匹配 filter 的 submit。
5. postprocess 读 manifest、driver JSONL、RGP 路径，输出 `attribution.jsonl`。
6. `src/torch2vk/sqtt` 解析 RGP/SQTT stream，并按 driver source map 抽取一份事实型 `report.md`。
7. 如果 profile run 进入 replay，直接报错，提示 SQTT 只支持 record 模式。

完成这条链路后，`torch2vk` 已经能稳定回答：某个 RGP 里的 dispatch 对应哪段 runtime 代码、哪个 shader、
哪个 pipeline。后续更深入的分析只应在 `attribution.jsonl` 之上继续扩展。
