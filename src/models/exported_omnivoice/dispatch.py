"""Generated dispatch functions for OmniVoice."""

from __future__ import annotations

from models.exported_omnivoice.tensors.model import model_tensors
from torch2vk.runtime.rope_table import run_rope_table_f32
from models.exported_omnivoice.shaders.add_f32 import ADD_F32
from models.exported_omnivoice.shaders.add_f32_37 import ADD_F32_37
from models.exported_omnivoice.shaders.add_f32_43 import ADD_F32_43
from models.exported_omnivoice.shaders.add_scalar import ADD_SCALAR
from models.exported_omnivoice.shaders.add_scalar_17 import ADD_SCALAR_17
from models.exported_omnivoice.shaders.add_scalar_9 import ADD_SCALAR_9
from models.exported_omnivoice.shaders.audio_head_linear_nobias_f32 import AUDIO_HEAD_LINEAR_NOBIAS_F32
from models.exported_omnivoice.shaders.cat_f32 import CAT_F32
from models.exported_omnivoice.shaders.cat_f32_32 import CAT_F32_32
from models.exported_omnivoice.shaders.linear_nobias_f32 import LINEAR_NOBIAS_F32
from models.exported_omnivoice.shaders.linear_nobias_f32_14 import LINEAR_NOBIAS_F32_14
from models.exported_omnivoice.shaders.linear_nobias_f32_22 import LINEAR_NOBIAS_F32_22
from models.exported_omnivoice.shaders.linear_nobias_f32_36 import LINEAR_NOBIAS_F32_36
from models.exported_omnivoice.shaders.linear_nobias_f32_38 import LINEAR_NOBIAS_F32_38
from models.exported_omnivoice.shaders.linear_nobias_f32_40 import LINEAR_NOBIAS_F32_40
from models.exported_omnivoice.shaders.linear_nobias_f32_42 import LINEAR_NOBIAS_F32_42
from models.exported_omnivoice.shaders.mean_dim_f32 import MEAN_DIM_F32
from models.exported_omnivoice.shaders.mean_dim_f32_16 import MEAN_DIM_F32_16
from models.exported_omnivoice.shaders.mean_dim_f32_8 import MEAN_DIM_F32_8
from models.exported_omnivoice.shaders.mul_broadcast_inner import MUL_BROADCAST_INNER
from models.exported_omnivoice.shaders.mul_broadcast_inner_29 import MUL_BROADCAST_INNER_29
from models.exported_omnivoice.shaders.mul_broadcast_inner_33 import MUL_BROADCAST_INNER_33
from models.exported_omnivoice.shaders.mul_broadcast_last import MUL_BROADCAST_LAST
from models.exported_omnivoice.shaders.mul_broadcast_last_11 import MUL_BROADCAST_LAST_11
from models.exported_omnivoice.shaders.mul_broadcast_last_19 import MUL_BROADCAST_LAST_19
from models.exported_omnivoice.shaders.mul_f32 import MUL_F32
from models.exported_omnivoice.shaders.mul_left_broadcast import MUL_LEFT_BROADCAST
from models.exported_omnivoice.shaders.mul_left_broadcast_12 import MUL_LEFT_BROADCAST_12
from models.exported_omnivoice.shaders.mul_left_broadcast_20 import MUL_LEFT_BROADCAST_20
from models.exported_omnivoice.shaders.neg_f32 import NEG_F32
from models.exported_omnivoice.shaders.omnivoice_cfg_score_f32 import OMNIVOICE_CFG_SCORE_F32
from models.exported_omnivoice.shaders.omnivoice_input_embed_f32 import OMNIVOICE_INPUT_EMBED_F32
from models.exported_omnivoice.shaders.omnivoice_token_update_topk_f32 import OMNIVOICE_TOKEN_UPDATE_TOPK_F32
from models.exported_omnivoice.shaders.pow_scalar_f32 import POW_SCALAR_F32
from models.exported_omnivoice.shaders.pow_scalar_f32_15 import POW_SCALAR_F32_15
from models.exported_omnivoice.shaders.pow_scalar_f32_7 import POW_SCALAR_F32_7
from models.exported_omnivoice.shaders.rsqrt_f32 import RSQRT_F32
from models.exported_omnivoice.shaders.rsqrt_f32_10 import RSQRT_F32_10
from models.exported_omnivoice.shaders.rsqrt_f32_18 import RSQRT_F32_18
from models.exported_omnivoice.shaders.sdpa_masked_f32 import SDPA_MASKED_F32
from models.exported_omnivoice.shaders.silu_f32 import SILU_F32
from models.exported_omnivoice.shaders.slice_f32 import SLICE_F32
from models.exported_omnivoice.shaders.slice_f32_25 import SLICE_F32_25
from models.exported_omnivoice.shaders.slice_f32_30 import SLICE_F32_30
from models.exported_omnivoice.shaders.slice_f32_31 import SLICE_F32_31
from models.exported_omnivoice.shaders.transpose_f32_45de1e4f84 import TRANSPOSE_F32_45DE1E4F84
from models.exported_omnivoice.shaders.transpose_f32_c943282b28 import TRANSPOSE_F32_C943282B28
from models.exported_omnivoice.shaders.transpose_f32_f3e8fdf2d4 import TRANSPOSE_F32_F3E8FDF2D4
from models.exported_omnivoice.tensors.audio_head import AudioHeadTensors
from models.exported_omnivoice.tensors.llm_forward import LlmForwardTensors
from torch2vk.runtime.session import RuntimeSession


def _run_llm_forward_with_tensors(rt: RuntimeSession, tensors: LlmForwardTensors) -> None:
    for layer_t in tensors.layers:
        POW_SCALAR_F32(rt, x=layer_t.to, output=layer_t.pow_1)
        MEAN_DIM_F32(rt, x=layer_t.pow_1, output=layer_t.mean)
        ADD_SCALAR(rt, x=layer_t.mean, output=layer_t.add)
        RSQRT_F32(rt, x=layer_t.add, output=layer_t.rsqrt)
        MUL_BROADCAST_LAST(rt, x=layer_t.to, y=layer_t.rsqrt, output=layer_t.mul)
        MUL_LEFT_BROADCAST(rt, x=layer_t.p_layers_0_input_layernorm_weight, y=layer_t.to_1, output=layer_t.mul_1)
        LINEAR_NOBIAS_F32(rt, x=layer_t.mul_1, weight=layer_t.p_layers_0_self_attn_q_proj_weight, output=layer_t.linear)
        POW_SCALAR_F32_7(rt, x=layer_t.to_2, output=layer_t.pow_2)
        MEAN_DIM_F32_8(rt, x=layer_t.pow_2, output=layer_t.mean_1)
        ADD_SCALAR_9(rt, x=layer_t.mean_1, output=layer_t.add_1)
        RSQRT_F32_10(rt, x=layer_t.add_1, output=layer_t.rsqrt_1)
        MUL_BROADCAST_LAST_11(rt, x=layer_t.to_2, y=layer_t.rsqrt_1, output=layer_t.mul_2)
        MUL_LEFT_BROADCAST_12(rt, x=layer_t.p_layers_0_self_attn_q_norm_weight, y=layer_t.to_3, output=layer_t.mul_3)
        TRANSPOSE_F32_F3E8FDF2D4(rt, x=layer_t.mul_3, output=layer_t.transpose)
        LINEAR_NOBIAS_F32_14(rt, x=layer_t.mul_1, weight=layer_t.p_layers_0_self_attn_k_proj_weight, output=layer_t.linear_1)
        POW_SCALAR_F32_15(rt, x=layer_t.to_4, output=layer_t.pow_3)
        MEAN_DIM_F32_16(rt, x=layer_t.pow_3, output=layer_t.mean_2)
        ADD_SCALAR_17(rt, x=layer_t.mean_2, output=layer_t.add_2)
        RSQRT_F32_18(rt, x=layer_t.add_2, output=layer_t.rsqrt_2)
        MUL_BROADCAST_LAST_19(rt, x=layer_t.to_4, y=layer_t.rsqrt_2, output=layer_t.mul_4)
        MUL_LEFT_BROADCAST_20(rt, x=layer_t.p_layers_0_self_attn_k_norm_weight, y=layer_t.to_5, output=layer_t.mul_5)
        TRANSPOSE_F32_C943282B28(rt, x=layer_t.mul_5, output=layer_t.transpose_1)
        LINEAR_NOBIAS_F32_22(rt, x=layer_t.mul_1, weight=layer_t.p_layers_0_self_attn_v_proj_weight, output=layer_t.linear_2)
        TRANSPOSE_F32_C943282B28(rt, x=layer_t.view_2, output=layer_t.transpose_2)
        MUL_BROADCAST_INNER(rt, x=layer_t.transpose, y=layer_t.unsqueeze, output=layer_t.mul_6)
        SLICE_F32(rt, x=layer_t.transpose, output=layer_t.slice_1)
        SLICE_F32_25(rt, x=layer_t.transpose, output=layer_t.slice_2)
        NEG_F32(rt, x=layer_t.slice_2, output=layer_t.neg)
        CAT_F32(rt, a=layer_t.neg, b=layer_t.slice_1, output=layer_t.cat)
        MUL_BROADCAST_INNER(rt, x=layer_t.cat, y=layer_t.unsqueeze_1, output=layer_t.mul_7)
        ADD_F32(rt, x=layer_t.mul_6, y=layer_t.mul_7, output=layer_t.add_3)
        MUL_BROADCAST_INNER_29(rt, x=layer_t.transpose_1, y=layer_t.unsqueeze, output=layer_t.mul_8)
        SLICE_F32_30(rt, x=layer_t.transpose_1, output=layer_t.slice_3)
        SLICE_F32_31(rt, x=layer_t.transpose_1, output=layer_t.slice_4)
        NEG_F32(rt, x=layer_t.slice_4, output=layer_t.neg_1)
        CAT_F32_32(rt, a=layer_t.neg_1, b=layer_t.slice_3, output=layer_t.cat_1)
        MUL_BROADCAST_INNER_33(rt, x=layer_t.cat_1, y=layer_t.unsqueeze_1, output=layer_t.mul_9)
        ADD_F32(rt, x=layer_t.mul_8, y=layer_t.mul_9, output=layer_t.add_4)
        SDPA_MASKED_F32(rt, q=layer_t.add_3, k=layer_t.add_4, v=layer_t.transpose_2, mask=tensors.attention_mask, output=layer_t.scaled_dot_product_attention)
        TRANSPOSE_F32_45DE1E4F84(rt, x=layer_t.scaled_dot_product_attention, output=layer_t.transpose_3)
        LINEAR_NOBIAS_F32_36(rt, x=layer_t.reshape, weight=layer_t.p_layers_0_self_attn_o_proj_weight, output=layer_t.linear_3)
        ADD_F32_37(rt, x=layer_t.to, y=layer_t.linear_3, output=layer_t.add_5)
        POW_SCALAR_F32(rt, x=layer_t.to_6, output=layer_t.pow_4)
        MEAN_DIM_F32(rt, x=layer_t.pow_4, output=layer_t.mean_3)
        ADD_SCALAR(rt, x=layer_t.mean_3, output=layer_t.add_6)
        RSQRT_F32(rt, x=layer_t.add_6, output=layer_t.rsqrt_3)
        MUL_BROADCAST_LAST(rt, x=layer_t.to_6, y=layer_t.rsqrt_3, output=layer_t.mul_10)
        MUL_LEFT_BROADCAST(rt, x=layer_t.p_layers_0_post_attention_layernorm_weight, y=layer_t.to_7, output=layer_t.mul_11)
        LINEAR_NOBIAS_F32_38(rt, x=layer_t.mul_11, weight=layer_t.p_layers_0_mlp_gate_proj_weight, output=layer_t.linear_4)
        SILU_F32(rt, x=layer_t.linear_4, output=layer_t.silu)
        LINEAR_NOBIAS_F32_40(rt, x=layer_t.mul_11, weight=layer_t.p_layers_0_mlp_up_proj_weight, output=layer_t.linear_5)
        MUL_F32(rt, x=layer_t.silu, y=layer_t.linear_5, output=layer_t.mul_12)
        LINEAR_NOBIAS_F32_42(rt, x=layer_t.mul_12, weight=layer_t.p_layers_0_mlp_down_proj_weight, output=layer_t.linear_6)
        ADD_F32_43(rt, x=layer_t.to_6, y=layer_t.linear_6, output=layer_t.add_7)
    POW_SCALAR_F32(rt, x=tensors.to_224, output=tensors.pow_113)
    MEAN_DIM_F32(rt, x=tensors.pow_113, output=tensors.mean_112)
    ADD_SCALAR(rt, x=tensors.mean_112, output=tensors.add_224)
    RSQRT_F32(rt, x=tensors.add_224, output=tensors.rsqrt_112)
    MUL_BROADCAST_LAST(rt, x=tensors.to_224, y=tensors.rsqrt_112, output=tensors.mul_364)
    MUL_LEFT_BROADCAST(rt, x=tensors.p_norm_weight, y=tensors.to_225, output=tensors.mul_365)


def _run_audio_head_with_tensors(rt: RuntimeSession, tensors: AudioHeadTensors) -> None:
    AUDIO_HEAD_LINEAR_NOBIAS_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_rope_table(rt: RuntimeSession, *, frame_name: str) -> None:
    rope_t = model_tensors().rope
    run_rope_table_f32(
        rt,
        start_position=rope_t.start_position,
        theta=rope_t.theta,
        cos=rope_t.cos,
        sin=rope_t.sin,
        frame_name=frame_name,
    )


def run_input_embed(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    OMNIVOICE_INPUT_EMBED_F32(
        rt,
        text_weight=tensors.text_embedding_weight,
        audio_weight=tensors.audio_embedding_weight,
        batch_input_ids=tensors.batch_input_ids,
        batch_audio_mask=tensors.batch_audio_mask,
        hidden_states=tensors.llm_forward.hidden_states,
    )


def run_llm_forward(rt: RuntimeSession) -> None:
    _run_llm_forward_with_tensors(rt, model_tensors().llm_forward)


def run_audio_head(rt: RuntimeSession) -> None:
    _run_audio_head_with_tensors(rt, model_tensors().audio_head)


def run_token_score(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    OMNIVOICE_CFG_SCORE_F32(
        rt,
        logits=tensors.audio_head.linear,
        tokens=tensors.tokens,
        audio_mask_id=tensors.audio_mask_id,
        rng_seed=tensors.rng_seed,
        step_index=tensors.step_index,
        candidate_tokens=tensors.candidate_tokens,
        candidate_scores=tensors.candidate_scores,
    )


def run_token_update(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    OMNIVOICE_TOKEN_UPDATE_TOPK_F32(
        rt,
        candidate_tokens=tensors.candidate_tokens,
        candidate_scores=tensors.candidate_scores,
        unmask_count=tensors.unmask_count,
        tokens=tensors.tokens,
        batch_input_ids=tensors.batch_input_ids,
    )


def run_generation_step(rt: RuntimeSession, *, step: int) -> None:
    with rt.frame(f"omnivoice.step.{step:04d}"):
        run_input_embed(rt)
        run_llm_forward(rt)
        run_audio_head(rt)
        run_token_score(rt)
        run_token_update(rt)
