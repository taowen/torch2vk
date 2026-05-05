"""Qwen3-ASR single-token text decode frame boundary."""

from __future__ import annotations

from models.qwen3_asr.shaders.text_add_3d_f32 import QWEN3_ASR_TEXT_ADD_3D_F32
from models.qwen3_asr.shaders.text_attention_decode_f32 import QWEN3_ASR_TEXT_ATTENTION_DECODE_F32
from models.qwen3_asr.shaders.text_embed_lookup_f32 import QWEN3_ASR_TEXT_EMBED_LOOKUP_F32
from models.qwen3_asr.shaders.text_gate_up_swiglu_t1_f32 import (
    QWEN3_ASR_TEXT_GATE_UP_SWIGLU_T1_F32,
)
from models.qwen3_asr.shaders.text_kv_cache_write_f32 import (
    QWEN3_ASR_TEXT_KV_CACHE_WRITE_DECODE_F32,
)
from models.qwen3_asr.shaders.text_linear_nobias_f32 import QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32
from models.qwen3_asr.shaders.text_linear_nobias_t1_f32 import QWEN3_ASR_TEXT_LINEAR_NOBIAS_T1_F32
from models.qwen3_asr.shaders.text_linear_nobias_t1_splitk4_f32 import (
    QWEN3_ASR_TEXT_LINEAR_NOBIAS_T1_SPLITK4_F32,
)
from models.qwen3_asr.shaders.text_lm_head_select_t1_f32 import (
    QWEN3_ASR_TEXT_LM_HEAD_SELECT_PARTIAL_T1_F32,
    QWEN3_ASR_TEXT_LM_HEAD_SELECT_REDUCE_T1_F32,
)
from models.qwen3_asr.shaders.text_qk_norm_f32 import QWEN3_ASR_TEXT_QK_NORM_F32
from models.qwen3_asr.shaders.text_qkv_proj_t1_f32 import QWEN3_ASR_TEXT_QKV_PROJ_T1_F32
from models.qwen3_asr.shaders.text_rms_norm_f32 import QWEN3_ASR_TEXT_RMS_NORM_F32
from models.qwen3_asr.shaders.text_rope_f32 import QWEN3_ASR_TEXT_ROPE_F32
from models.qwen3_asr.shaders.text_swiglu_f32 import QWEN3_ASR_TEXT_SWIGLU_F32
from models.qwen3_asr.tensors.text import Qwen3AsrTextDecodeTensors, Qwen3AsrTokenSelectTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


def run_qwen3_asr_text_decode(
    rt: RuntimeSession,
    tensors: Qwen3AsrTextDecodeTensors,
    *,
    step: int,
    pytorch_compare: bool = True,
    token_select: Qwen3AsrTokenSelectTensors | None = None,
) -> None:
    """Frame boundary for one cached decode step."""
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if pytorch_compare and token_select is not None:
        raise ValueError("lm-head token selection fusion requires pytorch_compare=False")
    if pytorch_compare:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        frame_scope = rt.frame(
            f"qwen3_asr.text_decode.{step:04d}",
            pytorch_model_class=Qwen3ASRForConditionalGeneration,
            pytorch_model_submodule="thinker",
            pytorch_input_prefixes=("qwen3_asr.text_decode",),
            pytorch_cache_policy="hf_dynamic",
            pytorch_cache_namespace="qwen3_asr.text",
        )
    else:
        frame_scope = rt.frame(f"qwen3_asr.text_decode.{step:04d}")

    with frame_scope:
        QWEN3_ASR_TEXT_EMBED_LOOKUP_F32(
            rt,
            input_ids=tensors.input_ids,
            embed_tokens_weight=tensors.embed_tokens_weight,
            output=tensors.inputs_embeds,
        )

        hidden = tensors.inputs_embeds
        for layer in tensors.layers:
            QWEN3_ASR_TEXT_RMS_NORM_F32(
                rt, x=hidden, weight=layer.input_layernorm_weight, output=layer.input_layernorm,
            )
            _run_qkv_proj_decode(
                rt,
                x=layer.input_layernorm,
                q_weight=layer.q_proj_weight,
                k_weight=layer.k_proj_weight,
                v_weight=layer.v_proj_weight,
                q_output=layer.q_proj,
                k_output=layer.k_proj,
                v_output=layer.v_proj,
                pytorch_compare=pytorch_compare,
            )
            QWEN3_ASR_TEXT_QK_NORM_F32(
                rt, x=layer.q_proj, weight=layer.q_norm_weight, output=layer.q_normed,
            )
            QWEN3_ASR_TEXT_QK_NORM_F32(
                rt, x=layer.k_proj, weight=layer.k_norm_weight, output=layer.k_normed,
            )
            QWEN3_ASR_TEXT_ROPE_F32(
                rt, x=layer.q_normed, cos=tensors.rope_cos, sin=tensors.rope_sin,
                output=layer.q_roped,
            )
            QWEN3_ASR_TEXT_ROPE_F32(
                rt, x=layer.k_normed, cos=tensors.rope_cos, sin=tensors.rope_sin,
                output=layer.k_roped,
            )
            QWEN3_ASR_TEXT_KV_CACHE_WRITE_DECODE_F32(
                rt, k=layer.k_roped, v=layer.v_proj,
                cache_position=tensors.cache_position,
                key_cache=layer.key_cache, value_cache=layer.value_cache,
            )
            QWEN3_ASR_TEXT_ATTENTION_DECODE_F32(
                rt, q=layer.q_roped,
                key_cache=layer.key_cache, value_cache=layer.value_cache,
                cache_position=tensors.cache_position,
                output=layer.attention,
            )
            _run_linear_nobias_decode(
                rt,
                x=layer.attention,
                weight=layer.o_proj_weight,
                output=layer.o_proj,
                pytorch_compare=pytorch_compare,
            )
            QWEN3_ASR_TEXT_ADD_3D_F32(
                rt, x=hidden, y=layer.o_proj, output=layer.attn_residual,
            )
            QWEN3_ASR_TEXT_RMS_NORM_F32(
                rt, x=layer.attn_residual, weight=layer.post_attention_layernorm_weight,
                output=layer.post_attention_layernorm,
            )
            _run_mlp_gate_up_swiglu_decode(
                rt,
                x=layer.post_attention_layernorm,
                gate_weight=layer.gate_proj_weight,
                up_weight=layer.up_proj_weight,
                gate=layer.gate_proj,
                up=layer.up_proj,
                output=layer.swiglu,
                pytorch_compare=pytorch_compare,
            )
            _run_down_proj_decode(
                rt,
                x=layer.swiglu,
                weight=layer.down_proj_weight,
                output=layer.down_proj,
                pytorch_compare=pytorch_compare,
            )
            QWEN3_ASR_TEXT_ADD_3D_F32(
                rt, x=layer.attn_residual, y=layer.down_proj, output=layer.output,
            )
            hidden = layer.output

        QWEN3_ASR_TEXT_RMS_NORM_F32(
            rt, x=hidden, weight=tensors.norm_weight, output=tensors.final_norm,
        )
        if pytorch_compare:
            QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(
                rt, x=tensors.final_norm, weight=tensors.lm_head_weight, output=tensors.logits,
            )
        elif token_select is not None:
            QWEN3_ASR_TEXT_LM_HEAD_SELECT_PARTIAL_T1_F32(
                rt,
                x=tensors.final_norm,
                weight=tensors.lm_head_weight,
                scratch=tensors.lm_head_select_scratch,
            )
            QWEN3_ASR_TEXT_LM_HEAD_SELECT_REDUCE_T1_F32(
                rt,
                scratch=tensors.lm_head_select_scratch,
                eos_token_ids=token_select.eos_token_ids,
                next_token=token_select.next_token,
                done=token_select.done,
            )
        else:
            QWEN3_ASR_TEXT_LINEAR_NOBIAS_T1_F32(
                rt, x=tensors.final_norm, weight=tensors.lm_head_weight, output=tensors.logits,
            )


def _run_linear_nobias_decode(
    rt: RuntimeSession,
    *,
    x: LogicalTensor,
    weight: LogicalTensor,
    output: LogicalTensor,
    pytorch_compare: bool,
) -> None:
    variant = QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32 if pytorch_compare else QWEN3_ASR_TEXT_LINEAR_NOBIAS_T1_F32
    variant(rt, x=x, weight=weight, output=output)


def _run_down_proj_decode(
    rt: RuntimeSession,
    *,
    x: LogicalTensor,
    weight: LogicalTensor,
    output: LogicalTensor,
    pytorch_compare: bool,
) -> None:
    variant = (
        QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32
        if pytorch_compare
        else QWEN3_ASR_TEXT_LINEAR_NOBIAS_T1_SPLITK4_F32
    )
    variant(rt, x=x, weight=weight, output=output)


def _run_mlp_gate_up_swiglu_decode(
    rt: RuntimeSession,
    *,
    x: LogicalTensor,
    gate_weight: LogicalTensor,
    up_weight: LogicalTensor,
    gate: LogicalTensor,
    up: LogicalTensor,
    output: LogicalTensor,
    pytorch_compare: bool,
) -> None:
    if pytorch_compare:
        QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(rt, x=x, weight=gate_weight, output=gate)
        QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(rt, x=x, weight=up_weight, output=up)
        QWEN3_ASR_TEXT_SWIGLU_F32(rt, gate=gate, up=up, output=output)
        return
    QWEN3_ASR_TEXT_GATE_UP_SWIGLU_T1_F32(
        rt,
        x=x,
        gate_weight=gate_weight,
        up_weight=up_weight,
        output=output,
    )


def _run_qkv_proj_decode(
    rt: RuntimeSession,
    *,
    x: LogicalTensor,
    q_weight: LogicalTensor,
    k_weight: LogicalTensor,
    v_weight: LogicalTensor,
    q_output: LogicalTensor,
    k_output: LogicalTensor,
    v_output: LogicalTensor,
    pytorch_compare: bool,
) -> None:
    if pytorch_compare:
        QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(rt, x=x, weight=q_weight, output=q_output)
        QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(rt, x=x, weight=k_weight, output=k_output)
        QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(rt, x=x, weight=v_weight, output=v_output)
        return
    QWEN3_ASR_TEXT_QKV_PROJ_T1_F32(
        rt,
        x=x,
        q_weight=q_weight,
        k_weight=k_weight,
        v_weight=v_weight,
        q_output=q_output,
        k_output=k_output,
        v_output=v_output,
    )
