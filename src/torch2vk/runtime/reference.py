"""Reference execution for subgraph comparison."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import Interpreter, Node


class ExportedProgramReference:
    """Reference that uses torch.fx.Interpreter to capture all intermediate node outputs."""

    def __init__(
        self,
        ep: ExportedProgram,
        state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self._graph_module = ep.graph_module
        self._input_specs = list(ep.graph_signature.input_specs)
        self._state_dict: dict[str, torch.Tensor] = {}
        for spec in self._input_specs:
            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                if spec.target is None:
                    raise ValueError(f"{spec.kind} input spec has no target")
                param = state_dict.get(spec.target) if state_dict is not None else None
                if param is None:
                    param = ep.state_dict.get(spec.target)
                if param is None:
                    param = dict(ep.graph_module.named_parameters()).get(spec.target)
                if param is None:
                    param = dict(ep.graph_module.named_buffers()).get(spec.target)
                if param is not None:
                    if param.is_meta:
                        raise ValueError(
                            f"reference parameter {spec.target!r} is a meta tensor; "
                            "pass the loaded module state_dict"
                        )
                    self._state_dict[spec.target] = param.cuda()

    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        class _CapturingInterpreter(Interpreter):
            def __init__(self, module: object) -> None:
                super().__init__(module)  # type: ignore[arg-type]
                self.captured: dict[str, torch.Tensor] = {}

            def run_node(self, n: Node) -> object:
                if (
                    n.op == "call_function"
                    and str(n.target) == "aten._assert_tensor_metadata.default"
                ):
                    return None
                result = super().run_node(n)
                if isinstance(result, torch.Tensor):
                    self.captured[n.name] = result
                return result

        all_args: list[torch.Tensor] = []
        for spec in self._input_specs:
            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                if spec.target is None:
                    raise ValueError(f"{spec.kind} input spec has no target")
                all_args.append(self._state_dict[spec.target])
            else:
                arr = inputs.get(spec.arg.name)
                if arr is None:
                    raise KeyError(
                        f"ExportedProgramReference: missing input '{spec.arg.name}', "
                        f"available: {sorted(inputs.keys())}"
                    )
                all_args.append(torch.from_numpy(np.ascontiguousarray(arr)).cuda())

        interp = _CapturingInterpreter(self._graph_module)
        with torch.no_grad():
            interp.run(*all_args)
        return cast(dict[str, object], interp.captured)


def load_exported_reference(
    base_dir: Path,
    program: str,
    *,
    state_dict: Mapping[str, torch.Tensor],
) -> ExportedProgramReference:
    ep = torch.export.load(base_dir / program)
    return ExportedProgramReference(cast(ExportedProgram, ep), state_dict=state_dict)
