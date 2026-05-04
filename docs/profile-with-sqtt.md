# 依赖 SQTT 的深入 Profiler

这份文档定义 `torch2vk` 的深度 profiler。它依赖本仓库的 Mesa RADV fork，通过
`MESA_VK_TRACE=rgp` 和 RADV SQTT 生成 RGP，同时导出 driver artifacts。它的职责不是替代轻量 profiler，
而是在轻量 profiler 产出的 runtime manifest 基础上解释 GPU 端为什么慢。

相关背景见 `docs/record-sqtt.md`。那份文档记录 Mesa/RADV SQTT 机制和手工验证路径；本文定义 `torch2vk`
最终产品形态。

## 目标

SQTT profiler 要回答这些问题：

1. 某个 `torch2vk` dispatch 在 RGP 里对应哪个 submit、pipeline、code object。
2. 某个 shader 的热点 ISA 指令、源代码位置和硬件事件是什么。
3. 慢点主要来自 memory、wait、occupancy、VALU/SALU、barrier、cache 或 wave scheduling 的哪一类。
4. replay submit 内部多个 dispatch 的顺序和 RGP/SQTT 事件如何对应。
5. 两次 profile run 的热点和硬件事件是否发生变化。

它不负责给正常推理做低开销计时。`MESA_VK_TRACE_PER_SUBMIT=true` 会显著改变运行形态，RGP/SQTT 结果用于
归因，不用于日常性能门禁。日常 benchmark 使用 `docs/profile-without-sqtt.md` 里的 timestamp profiler。

## 当前代码基础

仓库已有这些基础：

1. `third_party/mesa` 是 Mesa fork submodule。
2. `scripts/build-mesa-radv.sh` 是合并后的单入口，负责编译本地 RADV-only Mesa，并在 Fedora Atomic/Bazzite
   上通过 `dnf download` + RPM 解包准备本地 SDK，不修改宿主系统。
3. `scripts/profile-sqtt.sh` 负责在用户命令启动前选择本地 RADV ICD，设置 `MESA_VK_TRACE=rgp`、
   `MESA_VK_TRACE_PER_SUBMIT=true`、`RADV_THREAD_TRACE_*`、`AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR`、
   `AGENTORCH_RADV_EXPORT_TAG`，并把 `/tmp/*.rgp` 复制进 profile root。
4. `VulkanDevice` 已启用 `VK_EXT_debug_utils`，并提供 `begin_command_label()` /
   `end_command_label()`。
5. `ComputePipeline` 已有 `debug_name`、`shader_spv_sha256`、`pipeline_identity_sha256`。
6. `RuntimeSession.dispatch_records` 已保存 frame、dispatch index、shader、tensor、symbol、descriptor 语义。
7. `ReplayPlan.dispatch_entries` 已保存 replay command buffer 中每个 dispatch 的 pipeline、binding、dispatch
   size、dynamic params 信息。

当前缺口：

1. runtime 还没有生成稳定 `runtime-manifest.json`。
2. eager/replay command buffer 还没有打 `agentorch-profile-submit:` debug label。
3. 还没有把 runtime manifest、driver artifacts、RGP/SQTT 解码结果做 postprocess join。
4. Mesa fork 里的环境变量仍叫 `AGENTORCH_RADV_*`。短期沿用，长期可以在 Mesa fork 里加
   `TORCH2VK_RADV_*` alias。

## 分层模型

最终 SQTT profiler 必须按四层 join：

```text
torch2vk runtime manifest
  frame / dispatch_index / shader / tensor / shape / pipeline_identity_sha256

shader manifest
  GLSL / SPIR-V / source hash / SPIR-V hash / compile options

RADV driver artifacts
  driver-session.json / capture-sequence.jsonl / dispatch-sequence.jsonl
  pipeline-debug.json / compiler-native-disasm.s

RGP / SQTT
  submit trace / user events / code objects / instruction events / timing packets
```

runtime manifest 是语义事实源。driver artifacts 和 RGP 只提供底层执行事实。任何 postprocessor 都不能只靠
RGP 文件名、耗时相似度或 pipeline 名称猜测模型语义。

## 最终用户形态

外部 CLI 是主入口，因为 RADV SQTT 环境变量必须在 Vulkan instance 创建前设置：

```bash
uv run torch2vk-sqtt \
  --root .cache/torch2vk/sqtt/qwen3-asr-decode \
  --trace-buffer-mib 256 \
  --runtime-profile replay \
  --warmup 3 \
  --repeat 10 \
  -- pytest tests/test_qwen3_asr.py -k decode
```

低层脚本继续保留：

```bash
scripts/profile-sqtt.sh --root .cache/torch2vk/sqtt/manual -- <command>
```

Python 配置用于被 CLI 或测试注入 runtime：

```python
from pathlib import Path

from torch2vk.runtime.profile import ProfileConfig
from torch2vk.runtime.sqtt import SqttProfileConfig
from torch2vk.runtime.session import RuntimeSession


runtime_profile = ProfileConfig(
    enabled=True,
    root=Path(".cache/torch2vk/sqtt/qwen3-asr-decode"),
    mode="replay",
    warmup=3,
    repeat=10,
)

sqtt = SqttProfileConfig(
    enabled=True,
    root=Path(".cache/torch2vk/sqtt/qwen3-asr-decode"),
    mode="per_submit",
    trace_buffer_mib=256,
    require_local_mesa=True,
    export_driver_artifacts=True,
    runtime_profile=runtime_profile,
)

with RuntimeSession.open(device_index=0, model_dir=model_dir, profile=runtime_profile, sqtt=sqtt) as rt:
    run_qwen3_asr_decode(rt, tensors)

manifest = rt.sqtt_manifest()
```

`SqttProfileConfig` 不应该在 device 已创建后偷偷设置环境变量。如果 runtime 发现 Vulkan instance 已经存在，
必须报错。

## 运行流程

### 1. 构建或验证本地 Mesa

`scripts/profile-sqtt.sh` 默认在本地 RADV ICD 不存在时调用：

```bash
scripts/build-mesa-radv.sh --prefix .cache/torch2vk/mesa-radv
```

`build-mesa-radv.sh` 做这些事：

1. 检查 `third_party/mesa/meson.build`。
2. 准备 `.cache/torch2vk/vulkan-sdk`，包括 Vulkan headers、Vulkan loader devel、SPIR-V headers/tools、
   libdrm、expat。
3. 准备 `.cache/torch2vk/bin/glslc` 和 `glslangValidator`。
4. 准备 `.cache/torch2vk/mesa-build-venv`，安装 meson/ninja/mako/PyYAML 等 build tools。
5. 配置 RADV-only Mesa：

```text
-Dvulkan-drivers=amd
-Dgallium-drivers=
-Dplatforms=
-Dopengl=false
-Dglx=disabled
-Degl=disabled
-Dgbm=disabled
-Dllvm=enabled
-Dshared-llvm=enabled
```

这条路径适配 Fedora Atomic/Bazzite，因为它不要求 `dnf install *-devel` 到宿主系统。

### 2. 在进程启动前设置环境

profile command 启动前必须设置：

```bash
export VK_DRIVER_FILES=<repo>/.cache/torch2vk/mesa-radv/share/vulkan/icd.d/radeon_icd*.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export LD_LIBRARY_PATH=<repo>/.cache/torch2vk/mesa-radv/lib:<repo>/.cache/torch2vk/vulkan-sdk/usr/lib64:${LD_LIBRARY_PATH:-}
export MESA_VK_TRACE=rgp
export MESA_VK_TRACE_PER_SUBMIT=true
export RADV_THREAD_TRACE_BUFFER_SIZE=$((256 * 1024 * 1024))
export RADV_THREAD_TRACE_QUEUE_EVENTS=true
export RADV_THREAD_TRACE_INSTRUCTION_TIMING=true
export AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR=<root>/driver
export AGENTORCH_RADV_EXPORT_TAG=<run-id>
export MESA_SHADER_CACHE_DIR=<root>/mesa-shader-cache
export TORCH2VK_SQTT_ROOT=<root>
export TORCH2VK_PROFILE_RUN_DIR=<root>
```

`AGENTORCH_RADV_*` 是当前 Mesa fork 已实现的 artifact 导出接口。`TORCH2VK_*` 是 `torch2vk`
runtime/CLI 自己的接口。

### 3. runtime 生成语义 manifest

SQTT run 必须同时打开轻量 profiler，至少是 `mode="session"`：

```text
RuntimeSession.dispatch()
  -> 生成 dispatch_index / frame / shader / profile_submit_tag
  -> 记录 DispatchRecord
  -> 写 runtime-manifest.json
```

如果 `runtime-manifest.json` 缺失或为空，SQTT postprocess 必须失败。没有 runtime manifest 的 RGP 只能说明
GPU 做了什么，不能说明模型哪一步慢。

### 4. command buffer 打 label

Mesa fork 通过 debug utils label 把 Python runtime 语义传给 RADV：

```text
agentorch-profile-submit:<payload>
```

payload 建议短而稳定：

```text
run=qwen3-asr-decode;phase=replay;frame=text_decode;dispatch=42;shader=text_attention_decode_f32
```

如果 payload 过长，使用短 tag：

```text
agentorch-profile-submit:tag=8f6d1e2a9b31
```

完整映射写入 `runtime-manifest.json`：

```json
{
  "profile_submit_tag": "tag=8f6d1e2a9b31",
  "full_profile_tag": "run=qwen3-asr-decode;phase=replay;frame=text_decode;dispatch=42;shader=text_attention_decode_f32"
}
```

eager dispatch 当前是一 dispatch 一 submit，所以 per-submit SQTT 基本等价于 per-dispatch SQTT。

replay dispatch 通常是一个 submit 内多个 dispatch。replay command buffer 需要在每个 dispatch 周围打 label：

```text
begin command buffer
for dispatch in replay_plan.dispatch_entries:
  begin label(agentorch-profile-submit:<tag>)
  record dispatch or indirect dispatch
  end label
  barrier
end command buffer
submit once
```

这样一个 RGP 可以包含整条 replay sequence，而 `dispatch-sequence.jsonl` 能给出 submit 内部 dispatch 顺序。

### 5. RADV 导出 artifacts

Mesa fork 应导出：

```text
<root>/driver/
  driver-session.json
  capture-sequence.jsonl
  dispatch-sequence.jsonl
  <pipeline_hash>/
    pipeline-debug.json
    compiler-native-disasm.s
```

`scripts/profile-sqtt.sh` 还会把新的 `/tmp/*.rgp` 复制到：

```text
<root>/rgp/submit1.rgp
<root>/rgp/submit2.rgp
...
```

长期应由 postprocessor 根据 `capture-sequence.jsonl` 的原始路径复制或硬链接，保留稳定文件名：

```text
rgp/submit-000001.rgp
rgp/submit-000002.rgp
```

### 6. postprocess join

推荐新增：

```text
src/torch2vk/profiler/sqtt_postprocess.py
src/torch2vk/profiler/radv_artifacts.py
src/torch2vk/profiler/rgp.py
src/torch2vk/profiler/source_map.py
```

join 顺序：

```text
runtime-manifest.json
  join profile_submit_tag
driver/capture-sequence.jsonl
  join submit_ordinal
driver/dispatch-sequence.jsonl
  join pipeline_hash / dispatch_index within submit
driver/<pipeline_hash>/pipeline-debug.json
  join code object / shader hash
RGP/SQTT decoded events
  join code object / pipeline / event PC
compiler-native-disasm.s
  join PC / symbol / ISA line
GLSL/SPIR-V source manifest
  join shader_spv_sha256 / source hash
```

硬性 join key：

1. `profile_submit_tag`。
2. `submit_ordinal`。
3. driver submit 内 `dispatch_index`。
4. `pipeline_hash`。
5. `pipeline_debug_name`。
6. `pipeline_identity_sha256`。
7. `shader_spv_sha256`。
8. `runtime dispatch_index`。

不要用 RGP 文件排序或耗时相近作为主要 join 依据。

## Artifact layout

推荐完整 bundle：

```text
.cache/torch2vk/sqtt/<run-id>/
  command.txt
  command.log
  environment.json
  mesa-build.json
  profile-run.json
  runtime-manifest.json
  dispatch-profiles.jsonl
  shader-manifest.json
  rgp/
    submit-000001.rgp
    submit-000002.rgp
  driver/
    driver-session.json
    capture-sequence.jsonl
    dispatch-sequence.jsonl
    <pipeline_hash>/
      pipeline-debug.json
      compiler-native-disasm.s
  pipeline-attribution.json
  dispatch-sequence-attribution.json
  source-isa-sqtt-hotspots.json
  shader-optimization-report.json
  optimization-focus.md
  summary.json
  stage-timings.json
```

文件职责：

1. `runtime-manifest.json`：`torch2vk` 语义事实源。
2. `dispatch-profiles.jsonl`：轻量 profiler 的 per-dispatch timestamp 结果。SQTT run 可以只写 session
   信息，也可以同时做 replay timestamp。
3. `shader-manifest.json`：source/SPIR-V/hash/compile options。
4. `driver/`：Mesa fork 原始产物。
5. `rgp/`：稳定归档的 RGP。
6. `pipeline-attribution.json`：pipeline hash 到 shader/runtime identity 的映射。
7. `dispatch-sequence-attribution.json`：每个 runtime dispatch 到 submit/RGP/pipeline 的映射。
8. `source-isa-sqtt-hotspots.json`：源代码、ISA、SQTT event 的热点 join 结果。
9. `optimization-focus.md`：给人读的优化建议，只引用可追溯的证据。

## SqttProfileConfig

推荐 runtime 配置：

```python
@dataclass(frozen=True, slots=True)
class SqttProfileConfig:
    enabled: bool = False
    root: Path = Path(".cache/torch2vk/sqtt/default")
    mode: Literal["per_submit", "replay_submit"] = "per_submit"
    trace_buffer_mib: int = 256
    queue_events: bool = True
    instruction_timing: bool = True
    require_local_mesa: bool = True
    export_driver_artifacts: bool = True
    runtime_profile: ProfileConfig | None = None
```

`per_dispatch` 不需要单独作为主模式。当前 eager path 在 `MESA_VK_TRACE_PER_SUBMIT=true` 下已经是
per-dispatch；replay path 更应该保留一个 replay submit，靠 driver dispatch sequence 和 debug label 归因。

## Failure Modes

SQTT profiler 遇到这些情况要明确失败：

1. `VK_DRIVER_FILES` 没有指向 `.cache/torch2vk/mesa-radv/.../radeon_icd*.json`，但配置要求 local Mesa。
2. `MESA_VK_TRACE` 不是 `rgp`。
3. `MESA_VK_TRACE_PER_SUBMIT` 未打开，且当前模式依赖 per-submit capture。
4. `driver-session.json` 缺失。
5. `capture-sequence.jsonl` 缺失或为空。
6. `dispatch-sequence.jsonl` 缺失或为空。
7. `runtime-manifest.json` 缺失或为空。
8. `profile_submit_tag` 在 runtime 和 driver artifacts 中无法 join。
9. `pipeline_hash` 能 join，但 `pipeline_identity_sha256` 或 `shader_spv_sha256` 不匹配。
10. RGP 文件缺失、为空或无法解析。
11. SQTT decoder 没有 instruction event、PC 全零、code object 无法映射。

这些失败都应该写入 `summary.json`，并让 CLI 返回非零退出码，避免用户拿到看似完整但实际无法归因的报告。

## 实现阶段

### Phase 0：脚本和硬件 smoke

已具备：

1. `scripts/build-mesa-radv.sh` 编译本地 RADV。
2. `scripts/profile-sqtt.sh` 选择本地 RADV 并打开 RGP capture。
3. smoke run 能生成 RGP 和 driver artifacts。

### Phase 1：轻量 profiler

实现 `docs/profile-without-sqtt.md` 的 manifest-only 至 replay timestamp。SQTT profiler 后续只接受有
`runtime-manifest.json` 的 run。

### Phase 2：debug label 接入

1. eager dispatch 支持 `agentorch-profile-submit:<tag>`。
2. replay command buffer 每个 dispatch 支持 label。
3. tag 生成使用 runtime dispatch index、frame、shader、phase。
4. tag 过长时使用 hash fallback。

### Phase 3：SQTT session manifest

新增：

```text
src/torch2vk/runtime/sqtt.py
```

负责：

1. 校验 env 是否在 Vulkan device 前生效。
2. 写 `environment.json`。
3. 收集 driver artifacts 和 RGP path。
4. 写 `torch2vk-sqtt-session.json` 或合并进 `summary.json`。

### Phase 4：postprocess attribution

1. 解析 driver JSONL。
2. 归档 RGP。
3. join runtime dispatch 和 driver dispatch。
4. 输出 `pipeline-attribution.json` 和 `dispatch-sequence-attribution.json`。

### Phase 5：RGP/SQTT 解码和报告

1. 接入 RGP parser/SQTT decoder。
2. join code object、ISA、source map。
3. 输出热点 JSON 和 markdown 报告。
4. 支持 run diff。

## 不变量

1. SQTT profiler 必须依赖轻量 profiler 的 runtime manifest。
2. SQTT profiler 不在 Python 里写 AMD SQTT 寄存器，也不自己生成 RGP 文件。
3. 模型代码和模型目录不感知 SQTT。
4. SQTT 关闭时，普通 runtime/replay 行为完全不变。
5. debug label 是归因 metadata，不应该改变 shader、descriptor、barrier 或 replay plan 语义。
6. profile build 的 shader/cache 可以和普通运行分开，避免 debug info、cache key 或 Mesa trace 影响正常运行。
7. 所有 report 里的优化建议必须能追溯到 runtime manifest、driver artifacts 或 SQTT event。

## 和轻量 profiler 的边界

轻量 profiler 给出稳定、低开销的“哪一步慢”。SQTT profiler 只在需要回答“为什么慢”时运行。

推荐工作流：

```text
1. 用 torch2vk-profile 找到慢 frame/shader/dispatch。
2. 只对目标 frame 或 replay plan 运行 torch2vk-sqtt。
3. 用 SQTT report 定位 ISA/source/hardware event。
4. 改 shader。
5. 回到 torch2vk-profile 做低开销回归验证。
```

这条边界很重要。否则所有 profiling 都走 SQTT，会让数据量、运行时间和解释成本都失控。

## MVP 验收标准

1. `scripts/profile-sqtt.sh --print-env --dry-run -- <command>` 能清楚显示本地 Mesa ICD 和 SQTT env。
2. 单 dispatch smoke 能生成非空 `.rgp`。
3. `driver/capture-sequence.jsonl`、`driver/dispatch-sequence.jsonl` 非空。
4. runtime manifest 中每个 dispatch 都有 `profile_submit_tag`。
5. postprocess 能把至少一个 runtime dispatch join 到 RGP、submit ordinal、pipeline hash、pipeline debug 文件。
6. 缺少 runtime manifest 或 driver artifact 时 CLI 失败，而不是生成空报告。
