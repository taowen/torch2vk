"""Shader package rendering from ShaderVariant source modules."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import ModuleType

from torch2vk.exportv2.writer import (
    RenderedFile,
    format_python_source,
    normalize_generated_python,
)
from torch2vk.runtime.shader import ShaderVariant


@dataclass(frozen=True, slots=True)
class _ShaderSourceModule:
    module: str
    constants: tuple[str, ...]
    source: str


def render_shader_package_from_source_dir(
    source_dir: str | Path,
    *,
    constant_prefix: str,
    package_import: str,
    output_package_dir: str | Path,
    docstring: str,
    relative_dir: str | Path = "shaders",
) -> list[RenderedFile]:
    output_root = Path(output_package_dir)
    package_root = Path(relative_dir)
    modules = _shader_source_modules(
        Path(source_dir),
        constant_prefix=constant_prefix,
        output_package_dir=output_root,
        relative_dir=package_root,
    )

    files = [
        RenderedFile(
            package_root / "__init__.py",
            normalize_generated_python(
                format_python_source(
                    _render_shader_init_source(
                        modules,
                        package_import=package_import,
                        docstring=docstring,
                    ),
                    filename=output_root / package_root / "__init__.py",
                )
            ),
        )
    ]
    files.extend(
        RenderedFile(
            package_root / f"{module.module}.py", normalize_generated_python(module.source)
        )
        for module in modules
    )
    return files


def shader_variants_from_module(module: ModuleType) -> dict[str, ShaderVariant]:
    names = getattr(module, "__all__", ())
    if not isinstance(names, tuple | list):
        raise TypeError(f"{module.__name__}.__all__ must be a tuple or list")

    variants: dict[str, ShaderVariant] = {}
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"{module.__name__}.__all__ entries must be strings")
        value = getattr(module, name)
        if isinstance(value, ShaderVariant):
            variants[name] = value
    return variants


def _shader_source_modules(
    source_dir: Path,
    *,
    constant_prefix: str,
    output_package_dir: Path,
    relative_dir: Path,
) -> tuple[_ShaderSourceModule, ...]:
    modules: list[_ShaderSourceModule] = []
    for path in sorted(source_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        modules.append(
            _ShaderSourceModule(
                module=path.stem,
                constants=_shader_constants(path, constant_prefix=constant_prefix),
                source=format_python_source(
                    path.read_text(encoding="utf-8").rstrip() + "\n",
                    filename=output_package_dir / relative_dir / path.name,
                ),
            )
        )
    return tuple(modules)


def _shader_constants(path: Path, *, constant_prefix: str) -> tuple[str, ...]:
    module = _load_shader_source_module(path)
    constants = tuple(
        name
        for name, value in vars(module).items()
        if name.startswith(constant_prefix) and isinstance(value, ShaderVariant)
    )
    if not constants:
        raise ValueError(f"{path} does not define any {constant_prefix}* ShaderVariant constants")
    return constants


def _load_shader_source_module(path: Path) -> ModuleType:
    module_name = f"_torch2vk_exportv2_shader_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shader module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render_shader_init_source(
    modules: tuple[_ShaderSourceModule, ...],
    *,
    package_import: str,
    docstring: str,
) -> str:
    lines = [f'"""{docstring}"""', ""]
    for module in modules:
        lines.append(f"from {package_import}.{module.module} import (")
        lines.extend(f"    {constant}," for constant in module.constants)
        lines.append(")")
    lines.append("")
    lines.append("__all__ = [")
    for module in modules:
        lines.extend(f"    {constant!r}," for constant in module.constants)
    lines.append("]")
    return "\n".join(lines) + "\n"
