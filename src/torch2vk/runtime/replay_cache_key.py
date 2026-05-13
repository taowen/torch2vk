"""Helpers for replay cache keys."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch2vk.runtime.replay import ReplayPlan
    from torch2vk.runtime.session import RuntimeSession


def source_tree_digest(source_file: str | Path) -> str:
    root = Path(source_file).parent
    hasher = hashlib.sha256()
    for path in (
        Path(source_file),
        *sorted((root / "dispatch").glob("*.py")),
        *sorted((root / "shaders").glob("*.py")),
        *sorted((root / "tensors").glob("*.py")),
    ):
        hasher.update(str(path.relative_to(root)).encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def replay_cache_namespace(
    *,
    name: str,
    source_digest: str,
    model_dir: str | Path,
    shape_key: str | None = None,
) -> str:
    namespace = f"{name}:{source_digest}:{Path(model_dir).expanduser().resolve()}"
    if shape_key is None:
        return namespace
    return f"{namespace}:{shape_key}"


def cached_replay_plan(
    rt: "RuntimeSession",
    *,
    namespace: str,
) -> "ReplayPlan | None":
    for plan in rt.cached_replay_plans(namespace):
        return plan
    return None


def build_cached_replay_plan(
    rt: "RuntimeSession",
    *,
    namespace: str,
    name: str,
    frame: str,
    readback_error: str,
) -> "ReplayPlan":
    plan = rt.build_replay_plan(name=name, frame=frame)
    if plan.readback_slots:
        plan.close()
        raise RuntimeError(readback_error)
    rt.cache_replay_plan(namespace, plan)
    return plan
