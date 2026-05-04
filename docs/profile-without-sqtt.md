# 不依赖 SQTT 的轻量 Profiler

这份文档定义 `torch2vk` 的基础 profiler。它不依赖 Mesa fork，不打开 RGP/SQTT，也不要求 AMD
RADV。它的职责是把 `torch2vk` runtime 已经知道的语义信息，和 Vulkan timestamp query 得到的 GPU
耗时合成一个稳定 manifest。后续依赖 SQTT 的深度 profiler 必须建立在这个 manifest 之上。

## 目标

轻量 profiler 要回答这些问题：

1. 哪个 frame、dispatch、shader 慢。
2. 同一个 replay plan 里每个 dispatch 的 GPU 时间占比。
3. eager 路径和 replay 路径的时间差异。
4. 每个 dispatch 的输入输出 tensor、shape、dtype、descriptor 范围、push constants、shape symbols 是什么。
5. 某个 shader 在不同 shape 或不同 replay plan 下是否退化。
6. 根据现有 shader contract 做粗粒度工作量估算，例如读写字节数、dispatch group 数、近似带宽。

它不回答这些问题：

1. 某条 ISA 指令为什么慢。
2. wavefront stall、cache miss、VALU/SALU 利用率、等待原因。
3. source line 到硬件事件的精确映射。

这些是 SQTT profiler 的范围，见 `docs/profile-with-sqtt.md`。

## 当前代码基础

当前代码已经具备轻量 profiler 的主要锚点：

1. `RuntimeSession.dispatch()` 是 eager dispatch 的语义入口。它解析 shape symbols，materialize read/write
   tensor，绑定 descriptor，打包 push constants，执行 `ComputePipeline.dispatch()`，然后追加
   `DispatchRecord`。
2. `DispatchRecord` 已经包含 `index`、`frame`、`shader`、logical reads/writes、concrete
   symbols、`dispatch_size`、descriptor views、tensor snapshots、push constant values。
3. `ComputePipeline` 已经计算 `shader_spv_sha256`、`pipeline_identity_sha256` 和 `debug_name`。
   这些字段可以作为 shader/pipeline 归因的稳定 join key。
4. `ComputePipeline.record_dispatch()` 和 `record_indirect_dispatch()` 是 replay command buffer 的统一录制点。
5. `ReplayPlan` 已经保存 `dispatch_entries`、`params_entries`、dynamic symbol names、readback slots 和绑定好的
   command buffer。
6. `VulkanDevice.timestamp_period_ns` 已经暴露物理设备 timestamp period，但还缺 query pool 封装和 queue
   family 的 timestamp 支持检查。

关键判断：`DispatchRecord` 是 `torch2vk` 语义事实源，timestamp query 只补充耗时。不要反过来从 driver
或 trace 文件推断模型语义。

## 最终用户形态

Python API 应该允许用户在创建 `RuntimeSession` 时显式打开 profiler：

```python
from pathlib import Path

from torch2vk.runtime.profile import ProfileConfig
from torch2vk.runtime.session import RuntimeSession


profile = ProfileConfig(
    enabled=True,
    root=Path(".cache/torch2vk/profile/qwen3-asr-decode"),
    mode="replay",
    warmup=5,
    repeat=50,
    collect_tensor_values=False,
)

with RuntimeSession.open(device_index=0, model_dir=model_dir, profile=profile) as rt:
    run_qwen3_asr_decode(rt, tensors)

manifest = rt.profile_manifest()
```

也需要一个外部 CLI，保证测试、benchmark 和模型脚本不用手写环境变量：

```bash
uv run torch2vk-profile \
  --root .cache/torch2vk/profile/qwen3-asr-decode \
  --mode replay \
  --warmup 5 \
  --repeat 50 \
  -- pytest tests/test_qwen3_asr.py -k decode
```

CLI 的职责只是设置 `TORCH2VK_PROFILE_RUN_DIR`、`TORCH2VK_PROFILE_MODE` 等环境变量并 exec 用户命令。
真正的 runtime 记录必须在 Python 进程内完成，因为只有 runtime 知道 frame、tensor、shader variant 和
replay plan。

## 模式

### eager 模式

eager 模式围绕 `RuntimeSession.dispatch()` 单次 dispatch 记录 timestamp。当前 eager dispatch 本身就是：

```text
bind descriptor set
allocate command buffer
record one dispatch
record completion barrier
submit
wait fence
```

因此 eager profiler 能得到每个 dispatch 的 GPU 时间，并天然按 `DispatchRecord.index` 对齐。它适合做
smoke、正确性定位、首次热点排序。

限制是 eager 每个 dispatch 一个 submit，CPU submit/wait 和 GPU submit 粒度都不是最终推理形态。所以 eager
时间不能代表最终吞吐。

### replay 模式

replay 模式是主要形态。`ReplayPlan` 是 steady-state 推理应优化的对象，因为它复用 command buffer、descriptor
set 和 workspace，只更新 dynamic symbol/params/indirect buffers。

推荐实现一个 sidecar profile plan，而不是修改正常 `ReplayPlan.command_buffer`：

```text
ReplayPlan
  正常执行路径，保持现在的 command buffer。

ReplayProfilePlan
  共享 ReplayPlan 的 pipeline/binding/dispatch_entries。
  额外持有 query pool、profile command buffer、query index map。
  command buffer 中在每个 dispatch 前后写 timestamp。
```

这样 profiler 不会污染生产 replay command buffer，也不会让普通 replay 多出 query 指令。

### session 模式

session 模式只记录 runtime manifest，不插入 timestamp query。它用于：

1. 给 SQTT profiler 生成语义 manifest。
2. 在没有 timestamp query 支持的设备上保留 dispatch/tensor/pipeline 归因。
3. 低开销记录一整次模型运行的 dispatch 序列。

## 数据模型

推荐新增模块：

```text
src/torch2vk/runtime/profile.py
src/torch2vk/vulkan/timestamp_query.py
```

核心对象：

```python
@dataclass(frozen=True, slots=True)
class ProfileConfig:
    enabled: bool = False
    root: Path = Path(".cache/torch2vk/profile/default")
    mode: Literal["session", "eager", "replay"] = "replay"
    warmup: int = 3
    repeat: int = 20
    collect_tensor_values: bool = False
    session_name: str | None = None
    fail_if_timestamps_unavailable: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeDispatchProfile:
    run_id: str
    profile_submit_tag: str
    execution_kind: Literal["eager", "replay"]
    frame: str
    dispatch_index: int
    replay_plan: str | None
    replay_dispatch_index: int | None
    shader: str
    shader_family: str | None
    pipeline_debug_name: str
    pipeline_identity_sha256: str
    shader_spv_sha256: str
    symbols: dict[str, int]
    dispatch_size: tuple[int, int, int]
    push_constants: dict[str, int | float]
    tensor_bindings: list[dict[str, object]]
    elapsed_ns: int | None
    elapsed_ms: float | None
    work_estimate: dict[str, int | float]
```

run 级 manifest：

```json
{
  "schema_version": 1,
  "run_id": "qwen3-asr-decode",
  "tool": "torch2vk-profile",
  "mode": "replay",
  "device": {
    "name": "...",
    "timestamp_period_ns": 1.0
  },
  "command": ["pytest", "tests/test_qwen3_asr.py", "-k", "decode"],
  "environment": {
    "TORCH2VK_PROFILE_RUN_DIR": "..."
  },
  "dispatch_profile_path": "dispatch-profiles.jsonl",
  "summary_path": "summary.json"
}
```

每个 dispatch profile 应该保留这些 join key：

1. `dispatch_index`：`RuntimeSession` 全局 dispatch 顺序。
2. `frame`：当前 `FrameContext.frame`。
3. `shader`：`ShaderVariant.name`。
4. `pipeline_debug_name`：当前 `ComputePipeline.debug_name`。
5. `pipeline_identity_sha256`：pipeline 创建参数的稳定 hash。
6. `shader_spv_sha256`：SPIR-V 内容 hash。
7. `profile_submit_tag`：未来和 SQTT driver artifacts join 的短标签。

## Artifact layout

推荐布局：

```text
.cache/torch2vk/profile/<run-id>/
  profile-run.json
  runtime-manifest.json
  dispatch-profiles.jsonl
  frame-summary.json
  shader-summary.json
  replay-summary.json
  environment.json
  command.log
```

`runtime-manifest.json` 是语义事实源，结构应稳定，并被 SQTT profiler 复用。`dispatch-profiles.jsonl`
可以包含重复测量样本，summary 文件只保存派生统计。

## Timestamp query 设计

新增 `TimestampQueryRecorder` 封装 Vulkan query pool：

```text
create VkQueryPool(queryType=VK_QUERY_TYPE_TIMESTAMP, queryCount=2 * dispatch_count * repeat)
reset query pool
write timestamp before dispatch
record dispatch
write timestamp after dispatch
get query pool results
elapsed_ns = (end - begin) * device.timestamp_period_ns
```

实现要求：

1. 在 device 创建时读取 compute queue family 的 `timestampValidBits`，为 0 时降级到 session 模式或按配置报错。
2. query pool 只在 profile path 使用，不放进正常 replay command buffer。
3. 每个 repeat 使用独立 query index，避免复用 query 造成结果混淆。
4. warmup 的结果不写入 summary，但可以写到 raw samples，标记 `sample_kind="warmup"`。
5. 对 timestamp wraparound 做基本处理，至少在 manifest 中记录 `timestamp_valid_bits`。

## Runtime 集成点

### RuntimeSession

`RuntimeSession.open()` 增加可选 `profile: ProfileConfig | None` 参数。初始化顺序：

```text
RuntimeSession.__init__
  -> 创建 ProfileRecorder
  -> 创建 VulkanDevice
  -> ProfileRecorder.attach_device(device)
```

`RuntimeSession.dispatch()` 在执行前就能确定下一条 dispatch index：

```text
dispatch_index = len(self._dispatch_records)
profile_submit_tag = recorder.tag_for_dispatch(frame, variant, dispatch_index, phase="eager")
```

随后执行 dispatch，并把 `pipeline` 的 hash/name 写入 profile record。当前 `DispatchRecord` 在执行后才 append；
实现时可以先计算 index/tag，执行后再写 `DispatchRecord` 和 `RuntimeDispatchProfile`，但 tag 不能依赖执行后的
副作用。

### ComputePipeline

最小 eager 改法是在 `ComputePipeline.dispatch()` 增加可选 profile 参数：

```python
def dispatch(..., profile_scope: DispatchProfileScope | None = None) -> None:
    ...
    profile_scope.record_begin(command_buffer)
    self.record_dispatch(...)
    profile_scope.record_end(command_buffer)
```

长期更好的形态是把 eager command buffer 录制上移到 `RuntimeSession.dispatch()`，这样 runtime 可以同时控制
debug label、timestamp query、barrier 和 submit tag。

### ReplayPlan

不要修改 `execute_replay(plan)` 的默认行为。新增：

```python
def profile_replay(
    plan: ReplayPlan,
    *,
    config: ProfileConfig,
    dynamic_symbols: Mapping[str, int] | None = None,
) -> ReplayProfileResult:
    ...
```

`profile_replay()` 复用 `_write_indirect_dispatch_buffer()` 和 `_write_params_buffers()`，然后提交 profile command
buffer。这样动态 replay 的 shape 更新逻辑和正常执行保持一致。

## 工作量估算

第一版不需要精确 FLOPs。更有价值的是稳定、可比较的粗指标：

1. `dispatch_groups = x * y * z`。
2. `logical_read_bytes` 和 `logical_write_bytes`：来自 tensor snapshots 的 shape/dtype 和 descriptor nbytes。
3. `descriptor_read_bytes` 和 `descriptor_write_bytes`：来自 descriptor view 的 range。
4. `bytes_per_ms`：粗略带宽指标。
5. `groups_per_ms`：同 shader 不同 shape 的归一化指标。

如果 shader contract 以后增加 `work_estimate` 字段，再补充 shader 自己声明的 FLOPs、element count、
attention token count 等模型相关指标。

## 和 SQTT profiler 的关系

轻量 profiler 是 SQTT profiler 的前置层：

```text
RuntimeSession / ReplayPlan semantics
  -> runtime-manifest.json
  -> timestamp dispatch-profiles.jsonl
  -> SQTT profiler join driver artifacts and RGP
```

SQTT profiler 不能替代这层，原因是 RGP/SQTT 不知道 `LogicalTensor`、`FrameContext`、shape symbols、模型阶段、
request state 版本，也不应该从 GPU trace 里反推这些语义。

## 实现阶段

### Phase 1：manifest-only

1. 新增 `ProfileConfig`、`ProfileRecorder`、manifest writer。
2. `RuntimeSession.open(..., profile=...)` 接入 recorder。
3. 每次 dispatch 后把 `DispatchRecord`、pipeline hash/name、shader metadata 写到 `runtime-manifest.json`。
4. 不做 timestamp query。

验收：一次 smoke dispatch 能产出非空 `runtime-manifest.json`，关闭 profile 时无文件写入。

### Phase 2：eager timestamp

1. 新增 query pool wrapper。
2. 在 eager command buffer 中围绕 dispatch 写 timestamp。
3. 写 `dispatch-profiles.jsonl` 和 frame/shader summary。

验收：单 dispatch 的 `elapsed_ns` 非空，重复运行不会改变 dispatch/tensor 语义。

### Phase 3：replay timestamp

1. 新增 `ReplayProfilePlan` 或 `profile_replay()`。
2. 每个 replay dispatch 独立 timestamp。
3. 支持 warmup/repeat/median/min/p95 summary。
4. 支持动态 symbols 和 indirect dispatch。

验收：profile dispatch 数量和 `ReplayPlan.num_dispatches` 一致，顺序和 `dispatch_entries` 一致。

### Phase 4：CLI 和回归对比

1. `torch2vk-profile` CLI。
2. `profile-diff.json` / `profile-diff.md`。
3. 可选性能门禁，例如某 shader median 时间回退超过阈值时报错。

## 不变量

1. profile 关闭时，普通 runtime 和 replay 行为完全不变。
2. profiler 不保存 raw tensor value，除非用户显式 `collect_tensor_values=True`。
3. timestamp query 失败时不能 silently 写假数据；要写 `elapsed_ns=null` 和明确原因，或按配置失败。
4. `runtime-manifest.json` 的 dispatch 顺序必须和 `RuntimeSession.dispatch_records` 一致。
5. `pipeline_identity_sha256` 和 `shader_spv_sha256` 是主要 pipeline join key，不依赖文件名或耗时近似匹配。
6. replay profiler 不能修改正常 `ReplayPlan.command_buffer`。

## 测试建议

1. 单 shader smoke：manifest 里有一条 dispatch，frame/shader/tensor/push constants 正确。
2. 关闭 profile：不创建 profile root。
3. eager timestamp：`elapsed_ns > 0` 或在不支持 timestamp 的设备上记录明确 skip reason。
4. replay timestamp：dispatch 数量、顺序、shader 名和原 `frame_dispatch_records` 一致。
5. dynamic replay：改变 dynamic symbols 后，profile path 和正常 `execute_replay()` 结果一致。
6. summary：median/min/p95 由 raw samples 派生，不能手写不一致的 summary。
