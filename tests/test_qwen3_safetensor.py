from __future__ import annotations

import unittest

from torch2vk.models.qwen3_safetensor.execution import (
    qwen3_execution_tensors,
    record_qwen3_minimal_prefill,
)
from torch2vk.models.qwen3_safetensor.schema import qwen3_model_schema, qwen3_weight_tensors
from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec
from torch2vk.shader import DispatchTarget


def tiny_spec() -> Qwen3Spec:
    return Qwen3Spec(
        model_type="qwen3",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )


class Qwen3SafetensorTests(unittest.TestCase):
    def test_qwen3_weight_schema_names_and_shapes(self) -> None:
        spec = tiny_spec()
        weights = qwen3_weight_tensors(spec)
        by_name = {weight.name: weight for weight in weights}

        self.assertEqual(len(weights), 3 + spec.num_hidden_layers * 11)
        self.assertEqual(by_name["weights.embed_tokens"].shape, (128, 64))
        self.assertEqual(by_name["weights.layer.00.self_attn.q_proj"].shape, (64, 64))
        self.assertEqual(by_name["weights.layer.00.self_attn.k_proj"].shape, (32, 64))
        down_source = by_name["weights.layer.01.mlp.down_proj"].source
        self.assertIsNotNone(down_source)
        assert down_source is not None
        self.assertEqual(
            down_source.key,
            "model.layers.1.mlp.down_proj.weight",
        )

    def test_qwen3_model_schema_contains_ordered_boundaries(self) -> None:
        schema = qwen3_model_schema(tiny_spec())
        self.assertEqual(schema.model, "qwen3_safetensor")
        self.assertEqual(
            [boundary.name for boundary in schema.ordered_boundaries()],
            [
                "generate.prompt.tokens",
                "state.tokens.before",
                "model.next_token",
                "state.tokens.after",
                "generate.final.generated_tokens",
                "generate.final.tokens",
            ],
        )

    def test_qwen3_minimal_prefill_records_logical_edges(self) -> None:
        spec = tiny_spec()
        tensors = qwen3_execution_tensors(batch=1, steps=4, spec=spec)
        target = DispatchTarget()

        record_qwen3_minimal_prefill(target, spec=spec, tensors=tensors)

        self.assertEqual(
            [record.shader for record in target.records],
            [
                "embedding_lookup_bf16_f32",
                "rms_norm_f32",
                "linear_bf16_f32",
            ],
        )
        self.assertEqual(
            target.records[0].reads,
            {
                "input_ids": "input.input_ids",
                "weight": "weights.embed_tokens",
            },
        )
        self.assertEqual(target.records[0].writes, {"output": "decode.embedding"})
        self.assertEqual(target.records[2].writes, {"output": "output.logits"})

    def test_contract_validation_reports_logical_tensor_name(self) -> None:
        spec = tiny_spec()
        tensors = qwen3_execution_tensors(batch=1, steps=4, spec=spec)
        bad_tensors = qwen3_execution_tensors(batch=1, steps=5, spec=spec)
        target = DispatchTarget()

        with self.assertRaisesRegex(ValueError, "decode.embedding"):
            record_qwen3_minimal_prefill(
                target,
                spec=spec,
                tensors=type(tensors)(
                    input_ids=bad_tensors.input_ids,
                    hidden=tensors.hidden,
                    final_norm=tensors.final_norm,
                    logits=tensors.logits,
                ),
            )


if __name__ == "__main__":
    unittest.main()
