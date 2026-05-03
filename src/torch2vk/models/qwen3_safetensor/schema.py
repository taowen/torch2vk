"""Logical schema for Qwen3 safetensors."""

from __future__ import annotations

from torch2vk.logical import ComparePolicy, LogicalTensor, ReferenceRule, TensorRole, TensorSpec
from torch2vk.schema import BoundaryRule, ModelSchema, StepScope, W, token

from .spec import Qwen3Spec

GENERATE_STEP = StepScope(name="generate", prefix="generate.step_{step:03d}")

PROMPT_TOKENS = token("generate.prompt.tokens", ref_source="prompt_token_ids")
TOKENS_BEFORE = token("tokens.before", ref_source="tokens_before", step_scope=GENERATE_STEP)
MODEL_NEXT_TOKEN = token("model.next_token", ref_source="next_token_id", step_scope=GENERATE_STEP)
TOKENS_AFTER = token("tokens.after", ref_source="tokens_after", step_scope=GENERATE_STEP)
FINAL_GENERATED_TOKENS = token(
    "generate.final.generated_tokens",
    ref_source="generated",
)
FINAL_TOKENS = token("generate.final.tokens", ref_source="full_tokens")
MODEL_LOGITS = LogicalTensor(
    name=f"{GENERATE_STEP.prefix}.model.logits",
    spec=TensorSpec(dtype="float32", shape=("B", "V")),
    role=TensorRole.LOGITS,
    ref=ReferenceRule(source="logits"),
    compare=ComparePolicy(kind="tensor", rtol=1e-2, atol=1e-2),
)


def qwen3_weight_tensors(spec: Qwen3Spec) -> tuple[LogicalTensor, ...]:
    weights = [
        W(
            "weights.embed_tokens",
            safetensor_key="model.embed_tokens.weight",
            dtype="bfloat16",
            shape=(spec.vocab_size, spec.hidden_size),
        ),
        W(
            "weights.norm",
            safetensor_key="model.norm.weight",
            dtype="bfloat16",
            shape=(spec.hidden_size,),
        ),
        W(
            "weights.lm_head",
            safetensor_key="lm_head.weight",
            dtype="bfloat16",
            shape=(spec.vocab_size, spec.hidden_size),
        ),
    ]
    for layer_index in range(spec.num_hidden_layers):
        checkpoint_prefix = f"model.layers.{layer_index}"
        logical_prefix = f"weights.layer.{layer_index:02d}"
        weights.extend(
            (
                W(
                    f"{logical_prefix}.input_layernorm",
                    safetensor_key=f"{checkpoint_prefix}.input_layernorm.weight",
                    dtype="bfloat16",
                    shape=(spec.hidden_size,),
                ),
                W(
                    f"{logical_prefix}.post_attention_layernorm",
                    safetensor_key=f"{checkpoint_prefix}.post_attention_layernorm.weight",
                    dtype="bfloat16",
                    shape=(spec.hidden_size,),
                ),
                W(
                    f"{logical_prefix}.self_attn.q_proj",
                    safetensor_key=f"{checkpoint_prefix}.self_attn.q_proj.weight",
                    dtype="bfloat16",
                    shape=(spec.q_proj_out_features, spec.hidden_size),
                ),
                W(
                    f"{logical_prefix}.self_attn.k_proj",
                    safetensor_key=f"{checkpoint_prefix}.self_attn.k_proj.weight",
                    dtype="bfloat16",
                    shape=(spec.kv_proj_out_features, spec.hidden_size),
                ),
                W(
                    f"{logical_prefix}.self_attn.v_proj",
                    safetensor_key=f"{checkpoint_prefix}.self_attn.v_proj.weight",
                    dtype="bfloat16",
                    shape=(spec.kv_proj_out_features, spec.hidden_size),
                ),
                W(
                    f"{logical_prefix}.self_attn.o_proj",
                    safetensor_key=f"{checkpoint_prefix}.self_attn.o_proj.weight",
                    dtype="bfloat16",
                    shape=(spec.hidden_size, spec.q_proj_out_features),
                ),
                W(
                    f"{logical_prefix}.self_attn.q_norm",
                    safetensor_key=f"{checkpoint_prefix}.self_attn.q_norm.weight",
                    dtype="bfloat16",
                    shape=(spec.head_dim,),
                ),
                W(
                    f"{logical_prefix}.self_attn.k_norm",
                    safetensor_key=f"{checkpoint_prefix}.self_attn.k_norm.weight",
                    dtype="bfloat16",
                    shape=(spec.head_dim,),
                ),
                W(
                    f"{logical_prefix}.mlp.gate_proj",
                    safetensor_key=f"{checkpoint_prefix}.mlp.gate_proj.weight",
                    dtype="bfloat16",
                    shape=(spec.intermediate_size, spec.hidden_size),
                ),
                W(
                    f"{logical_prefix}.mlp.up_proj",
                    safetensor_key=f"{checkpoint_prefix}.mlp.up_proj.weight",
                    dtype="bfloat16",
                    shape=(spec.intermediate_size, spec.hidden_size),
                ),
                W(
                    f"{logical_prefix}.mlp.down_proj",
                    safetensor_key=f"{checkpoint_prefix}.mlp.down_proj.weight",
                    dtype="bfloat16",
                    shape=(spec.hidden_size, spec.intermediate_size),
                ),
            )
        )
    return tuple(weights)


def qwen3_boundaries() -> tuple[BoundaryRule, ...]:
    token_compare = ComparePolicy(kind="token")
    return (
        BoundaryRule(
            name="generate.prompt.tokens",
            phase="input",
            order=100,
            tokens=(PROMPT_TOKENS,),
            compare=token_compare,
            checkpoint=PROMPT_TOKENS,
        ),
        BoundaryRule(
            name="state.tokens.before",
            phase="state",
            order=200,
            tokens=(TOKENS_BEFORE,),
            compare=token_compare,
            checkpoint=TOKENS_BEFORE,
        ),
        BoundaryRule(
            name="model.next_token",
            phase="model",
            order=300,
            tensors=(MODEL_LOGITS,),
            tokens=(MODEL_NEXT_TOKEN,),
            compare=token_compare,
            checkpoint=MODEL_NEXT_TOKEN,
        ),
        BoundaryRule(
            name="state.tokens.after",
            phase="state",
            order=400,
            tokens=(TOKENS_AFTER,),
            compare=token_compare,
            checkpoint=TOKENS_AFTER,
        ),
        BoundaryRule(
            name="generate.final.generated_tokens",
            phase="final",
            order=900,
            tokens=(FINAL_GENERATED_TOKENS,),
            compare=token_compare,
            checkpoint=FINAL_GENERATED_TOKENS,
        ),
        BoundaryRule(
            name="generate.final.tokens",
            phase="final",
            order=1000,
            tokens=(FINAL_TOKENS,),
            compare=token_compare,
            checkpoint=FINAL_TOKENS,
        ),
    )


def qwen3_model_schema(spec: Qwen3Spec) -> ModelSchema:
    return ModelSchema(
        model="qwen3_safetensor",
        weights=qwen3_weight_tensors(spec),
        boundaries=qwen3_boundaries(),
    )
