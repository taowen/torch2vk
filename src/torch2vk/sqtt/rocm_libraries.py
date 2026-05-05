"""ROCm profiler shared-library discovery for flatpak-safe local installs."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


ROCM_PROFILER_CACHE_DIR = Path(".cache/torch2vk/profiler/rocm-libs")
AGENTORCH_COMPAT_PROFILER_CACHE_DIR = Path(".cache/agentorch/profiler/rocm-libs")
TRACE_DECODER_LIBRARY = "librocprof-trace-decoder.so"
ROCPROFILER_SDK_LIBRARY = "librocprofiler-sdk.so"
AQLPROFILE_LIBRARY = "libhsa-amd-aqlprofile64.so.1"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def local_rocm_profiler_cache_dir() -> Path:
    return repository_root() / ROCM_PROFILER_CACHE_DIR


def find_trace_decoder_library() -> Path:
    candidates: list[Path] = []
    env_path = os.environ.get("TORCH2VK_TRACE_DECODER_SO")
    if env_path:
        candidates.append(Path(env_path))
    env_path = os.environ.get("AGENTORCH_TRACE_DECODER_SO")
    if env_path:
        candidates.append(Path(env_path))
    for directory in _trace_decoder_library_dirs(None):
        candidates.append(directory / TRACE_DECODER_LIBRARY)
    return _first_existing_file(TRACE_DECODER_LIBRARY, candidates)


def find_trace_decoder_library_dir(explicit_dir: Path | None) -> Path:
    candidates = _trace_decoder_library_dirs(explicit_dir)
    for candidate in candidates:
        if (candidate / TRACE_DECODER_LIBRARY).exists():
            return candidate
    raise FileNotFoundError(_missing_message(TRACE_DECODER_LIBRARY, candidates))


def find_rocprofiler_sdk_library(explicit_path: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    env_path = os.environ.get("TORCH2VK_ROCPROFILER_SDK_SO")
    if env_path:
        candidates.append(Path(env_path))
    env_path = os.environ.get("AGENTORCH_ROCPROFILER_SDK_SO")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(_local_sdk_library_candidates())
    candidates.append(Path("/opt/rocm/lib") / ROCPROFILER_SDK_LIBRARY)
    return _first_existing_file(ROCPROFILER_SDK_LIBRARY, candidates)


def preload_rocprofiler_sdk_dependencies(sdk_library_path: Path) -> None:
    sdk_dir = sdk_library_path.parent
    for library_name in (AQLPROFILE_LIBRARY,):
        candidate = sdk_dir / library_name
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)


def _trace_decoder_library_dirs(explicit_dir: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_dir is not None:
        candidates.append(explicit_dir)
    env_dir = os.environ.get("TORCH2VK_TRACE_DECODER_LIBRARY_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    env_dir = os.environ.get("AGENTORCH_TRACE_DECODER_LIBRARY_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    cache_dir = local_rocm_profiler_cache_dir()
    agentorch_compat_cache_dir = repository_root().parent / "agentorch" / AGENTORCH_COMPAT_PROFILER_CACHE_DIR
    candidates.extend(
        [
            cache_dir / "opt/rocm/lib",
            agentorch_compat_cache_dir / "opt/rocm/lib",
            Path("/home/taowen/projects/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64"),
            Path("/opt/rocm/lib"),
        ]
    )
    return candidates


def _local_sdk_library_candidates() -> list[Path]:
    cache_dir = local_rocm_profiler_cache_dir()
    candidates = sorted(cache_dir.glob("opt/rocm-*/lib/librocprofiler-sdk.so"), reverse=True)
    candidates.append(cache_dir / "opt/rocm/lib/librocprofiler-sdk.so")
    agentorch_compat_cache_dir = repository_root().parent / "agentorch" / AGENTORCH_COMPAT_PROFILER_CACHE_DIR
    candidates.extend(sorted(agentorch_compat_cache_dir.glob("opt/rocm-*/lib/librocprofiler-sdk.so"), reverse=True))
    candidates.append(agentorch_compat_cache_dir / "opt/rocm/lib/librocprofiler-sdk.so")
    return candidates


def _first_existing_file(library_name: str, candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(_missing_message(library_name, candidates))


def _missing_message(library_name: str, candidates: list[Path]) -> str:
    searched = ", ".join(str(item) for item in candidates)
    return f"Could not find {library_name}. Searched: {searched}"
