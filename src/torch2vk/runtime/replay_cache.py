"""Replay template cache and cache compatibility."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.replay import ReplayPlan, ReplayPlanTemplate
from torch2vk.runtime.replay_descriptor import (
    canonical_replay_descriptor_tensor,
    replay_descriptor_rebindable,
)
from torch2vk.runtime.replay_instantiation import instantiate_replay_template
from torch2vk.runtime.replay_rebind import replay_plan_compatible, replay_symbols_compatible

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


_REPLAY_TEMPLATE_CACHE: dict[str, list[ReplayPlanTemplate]] = {}
_REPLAY_TEMPLATE_CACHE_DIR = "replay_templates_v3"


def cached_replay_plans(rt: RuntimeSession, namespace: str) -> tuple[ReplayPlan, ...]:
    rt._require_open()
    plans = rt._replay_plan_cache.get(namespace, [])
    live_plans = [plan for plan in plans if not plan._closed and replay_plan_compatible(rt, plan)]
    templates = _cached_replay_templates(rt, namespace)
    for template in templates:
        if any(plan.template == template for plan in live_plans):
            continue
        if not replay_template_compatible(rt, template):
            continue
        live_plans.append(
            instantiate_replay_template(
                rt,
                template=template,
                logical_tensors=rt._named_model_tensors(),
            )
        )
    rt._replay_plan_cache[namespace] = live_plans
    if not live_plans and (any(not plan._closed for plan in plans) or templates):
        raise RuntimeError(
            f"Replay cache {namespace!r} exists but is incompatible with current model tensors"
        )
    return tuple(live_plans)


def cache_replay_plan(rt: RuntimeSession, namespace: str, plan: ReplayPlan) -> None:
    rt._require_open()
    if plan._closed:
        raise RuntimeError(f"Cannot cache closed ReplayPlan {plan.name!r}")
    if plan.device is not rt.device:
        raise ValueError("ReplayPlan belongs to a different RuntimeSession device")
    plans = rt._replay_plan_cache.setdefault(namespace, [])
    if not any(existing is plan for existing in plans):
        plans.append(plan)
    if plan.template is not None:
        templates = _cached_replay_templates(rt, namespace)
        if plan.template not in templates:
            templates.append(plan.template)
            _write_replay_templates(rt, namespace, templates)


def _cached_replay_templates(
    rt: RuntimeSession,
    namespace: str,
) -> list[ReplayPlanTemplate]:
    cached = _REPLAY_TEMPLATE_CACHE.get(namespace)
    if cached is not None:
        return cached
    path = _replay_template_cache_path(rt, namespace)
    if not path.is_file():
        _REPLAY_TEMPLATE_CACHE[namespace] = []
        return _REPLAY_TEMPLATE_CACHE[namespace]
    with path.open("rb") as handle:
        loaded: object = pickle.load(handle)
    if not isinstance(loaded, list):
        raise TypeError(f"Replay template cache {path} did not contain a list")
    templates: list[ReplayPlanTemplate] = []
    for item in loaded:
        if not isinstance(item, ReplayPlanTemplate):
            raise TypeError(
                f"Replay template cache {path} contained {type(item).__name__}, "
                "expected ReplayPlanTemplate"
            )
        templates.append(item)
    _REPLAY_TEMPLATE_CACHE[namespace] = templates
    return templates


def _write_replay_templates(
    rt: RuntimeSession,
    namespace: str,
    templates: Sequence[ReplayPlanTemplate],
) -> None:
    path = _replay_template_cache_path(rt, namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(list(templates), handle, protocol=pickle.HIGHEST_PROTOCOL)


def _replay_template_cache_path(rt: RuntimeSession, namespace: str) -> Path:
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    return rt.artifact_dir.parent / _REPLAY_TEMPLATE_CACHE_DIR / f"{digest}.pkl"


def replay_template_compatible(rt: RuntimeSession, template: ReplayPlanTemplate) -> bool:
    logical_tensors = rt._named_model_tensors()
    for entry in template.entries:
        source_variant = rt._model_shader(entry.shader)
        logical_by_field = dict(entry.logical_reads)
        logical_by_field.update(entry.logical_writes)
        field_tensors: dict[str, LogicalTensor] = {}
        for field in source_variant.contract.fields:
            tensor_name = logical_by_field.get(field.name)
            if tensor_name is None:
                return False
            tensor = logical_tensors.get(tensor_name)
            if tensor is None:
                return False
            descriptor_tensor = canonical_replay_descriptor_tensor(
                tensor=tensor,
                logical_tensors=logical_tensors,
            )
            if replay_descriptor_rebindable(descriptor_tensor) and descriptor_tensor is tensor:
                field_tensors[field.name] = tensor
        try:
            rebound_symbols = rt._bind_shape_symbols(
                tuple(
                    field for field in source_variant.contract.fields if field.name in field_tensors
                ),
                field_tensors,
            )
        except ValueError:
            return False
        if not replay_symbols_compatible(
            plan_name=template.name,
            entry_symbols=dict(entry.symbols),
            dynamic_symbol_names=entry.dynamic_symbol_names,
            rebound_symbols=rebound_symbols,
        ):
            return False
    return True
