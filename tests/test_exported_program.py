"""PyTorch ExportedProgram graph import coverage."""

from __future__ import annotations

import torch
from torch import nn

from torch2vk.export.exported_program import (
    export_torch_program,
    torch_ops_from_exported_program,
)


def test_torch_ops_from_exported_program_preserves_fx_node_contract() -> None:
    class ToyEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(16, 4, device="meta")

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            token_ids = input_ids[:, 0, :]
            return self.embedding(token_ids)

    exported_program = export_torch_program(
        ToyEmbedding().eval(),
        (torch.zeros((1, 2, 3), dtype=torch.long, device="meta"),),
        strict=False,
    )
    ops = torch_ops_from_exported_program(
        exported_program,
        tensor_name_map={
            "p_embedding_weight": "embedding_weight",
            "input_ids": "input_ids",
            "select": "token_ids",
            "embedding": "output",
        },
    )

    assert [(op.name, op.op, op.target, op.inputs, op.outputs) for op in ops] == [
        ("select", "call_function", "aten.select.int", ("input_ids",), ("token_ids",)),
        (
            "embedding",
            "call_function",
            "aten.embedding.default",
            ("embedding_weight", "token_ids"),
            ("output",),
        ),
    ]
    assert ops[0].args == ("input_ids", 1, 0)
    assert ops[0].shape == (1, 3)
    assert ops[0].dtype == "int64"
    assert ops[1].shape == (1, 3, 4)
    assert ops[1].dtype == "float32"
