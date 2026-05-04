"""Compute pipeline cache and inline shader compilation helpers."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from torch2vk.runtime.shader import ShaderVariant
from torch2vk.vulkan.compute_pipeline import ComputePipeline

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


def pipeline_for_variant(rt: RuntimeSession, variant: ShaderVariant) -> ComputePipeline:
    spv_path = spv_path_for_variant(rt, variant)
    descriptor_count = len(variant.contract.fields)
    if variant.contract.params_buffer is not None:
        descriptor_count += 1
    key = (
        str(spv_path),
        descriptor_count,
        variant.specialization_constants,
        0 if variant.contract.push_constants is None else variant.contract.push_constants.size,
        variant.execution_requirements,
    )
    pipeline = rt._pipeline_cache.get(key)
    if pipeline is not None:
        return pipeline
    pipeline = ComputePipeline(
        rt.device,
        shader_spv_path=spv_path,
        storage_buffer_count=descriptor_count,
        specialization_constants=None
        if variant.specialization_constants is None
        else dict(variant.specialization_constants),
        push_constant_size=0
        if variant.contract.push_constants is None
        else variant.contract.push_constants.size,
        execution_requirements=variant.execution_requirements,
    )
    rt._pipeline_cache[key] = pipeline
    return pipeline


def spv_path_for_variant(rt: RuntimeSession, variant: ShaderVariant) -> Path:
    if variant.precompiled_spv_path is not None:
        return variant.precompiled_spv_path
    compiler = shutil.which("glslc")
    if compiler is None:
        raise RuntimeError("glslc is required to compile inline ShaderVariant source")
    compile_args = (
        "-fshader-stage=compute",
        "--target-env=vulkan1.3",
        "-O",
        "-g",
    )
    source_hash = hashlib.sha256(
        "\n".join(
            (
                variant.source,
                repr(compile_args),
                repr(variant.include_dirs),
                repr(variant.compile_defines),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    stem = f"{variant.name}.{source_hash}"
    glsl_path = rt.artifact_dir / f"{stem}.comp"
    spv_path = rt.artifact_dir / f"{stem}.spv"
    rt.artifact_dir.mkdir(parents=True, exist_ok=True)
    if not spv_path.is_file():
        glsl_path.write_text(variant.source, encoding="utf-8")
        include_args: list[str | Path] = []
        for include_dir in variant.include_dirs:
            include_args.extend(("-I", include_dir))
        define_args = [f"-D{define}" for define in variant.compile_defines]
        subprocess.run(
            [
                compiler,
                *compile_args,
                *include_args,
                *define_args,
                glsl_path.name,
                "-o",
                spv_path.name,
            ],
            check=True,
            cwd=str(glsl_path.parent),
        )
    return spv_path
