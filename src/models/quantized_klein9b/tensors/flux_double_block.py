"""Generated tensor declarations."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    bind_logical_tensor_alias,
    bind_logical_tensor_names,
)
from torch2vk.vulkan.types import (
    CONTIGUOUS_LAYOUT,
    TensorLayout,
    TensorSpec,
    q4_k_words_layout,
    q6_k_halfwords_layout,
    q8_0_halfwords_layout,
)


@dataclass(frozen=True, slots=True)
class FluxDoubleBlockTensors:
    p_img_attn_qkv_weight: LogicalTensor
    p_img_attn_norm_query_norm_scale: LogicalTensor
    p_img_attn_norm_key_norm_scale: LogicalTensor
    p_img_attn_proj_weight: LogicalTensor
    p_img_mlp_0_weight: LogicalTensor
    p_img_mlp_2_weight: LogicalTensor
    p_txt_attn_qkv_weight: LogicalTensor
    p_txt_attn_norm_query_norm_scale: LogicalTensor
    p_txt_attn_norm_key_norm_scale: LogicalTensor
    p_txt_attn_proj_weight: LogicalTensor
    p_txt_mlp_0_weight: LogicalTensor
    p_txt_mlp_2_weight: LogicalTensor
    img: LogicalTensor
    txt: LogicalTensor
    pe: LogicalTensor
    pe_ctx: LogicalTensor
    img_mod1_shift: LogicalTensor
    img_mod1_scale: LogicalTensor
    img_mod1_gate: LogicalTensor
    img_mod2_shift: LogicalTensor
    img_mod2_scale: LogicalTensor
    img_mod2_gate: LogicalTensor
    txt_mod1_shift: LogicalTensor
    txt_mod1_scale: LogicalTensor
    txt_mod1_gate: LogicalTensor
    txt_mod2_shift: LogicalTensor
    txt_mod2_scale: LogicalTensor
    txt_mod2_gate: LogicalTensor
    layer_norm: LogicalTensor
    add: LogicalTensor
    mul: LogicalTensor
    add_1: LogicalTensor
    linear: LogicalTensor
    reshape: LogicalTensor
    permute: LogicalTensor
    getitem: LogicalTensor
    getitem_1: LogicalTensor
    getitem_2: LogicalTensor
    rms_norm: LogicalTensor
    rms_norm_1: LogicalTensor
    to: LogicalTensor
    to_1: LogicalTensor
    layer_norm_1: LogicalTensor
    add_2: LogicalTensor
    mul_1: LogicalTensor
    add_3: LogicalTensor
    linear_1: LogicalTensor
    reshape_1: LogicalTensor
    permute_1: LogicalTensor
    getitem_3: LogicalTensor
    getitem_4: LogicalTensor
    getitem_5: LogicalTensor
    rms_norm_2: LogicalTensor
    rms_norm_3: LogicalTensor
    to_2: LogicalTensor
    to_3: LogicalTensor
    cat: LogicalTensor
    cat_1: LogicalTensor
    cat_2: LogicalTensor
    cat_3: LogicalTensor
    to_4: LogicalTensor
    reshape_2: LogicalTensor
    to_5: LogicalTensor
    reshape_3: LogicalTensor
    select: LogicalTensor
    select_1: LogicalTensor
    mul_2: LogicalTensor
    select_2: LogicalTensor
    select_3: LogicalTensor
    mul_3: LogicalTensor
    add_4: LogicalTensor
    select_4: LogicalTensor
    select_5: LogicalTensor
    mul_4: LogicalTensor
    select_6: LogicalTensor
    select_7: LogicalTensor
    mul_5: LogicalTensor
    add_5: LogicalTensor
    reshape_4: LogicalTensor
    type_as: LogicalTensor
    reshape_5: LogicalTensor
    type_as_1: LogicalTensor
    scaled_dot_product_attention: LogicalTensor
    permute_2: LogicalTensor
    reshape_6: LogicalTensor
    slice_1: LogicalTensor
    slice_2: LogicalTensor
    linear_2: LogicalTensor
    mul_6: LogicalTensor
    add_6: LogicalTensor
    add_7: LogicalTensor
    layer_norm_2: LogicalTensor
    mul_7: LogicalTensor
    add_8: LogicalTensor
    linear_3: LogicalTensor
    getitem_6: LogicalTensor
    getitem_7: LogicalTensor
    silu: LogicalTensor
    mul_8: LogicalTensor
    linear_4: LogicalTensor
    mul_9: LogicalTensor
    add_9: LogicalTensor
    linear_5: LogicalTensor
    mul_10: LogicalTensor
    add_10: LogicalTensor
    add_11: LogicalTensor
    layer_norm_3: LogicalTensor
    mul_11: LogicalTensor
    add_12: LogicalTensor
    linear_6: LogicalTensor
    getitem_8: LogicalTensor
    getitem_9: LogicalTensor
    silu_1: LogicalTensor
    mul_12: LogicalTensor
    linear_7: LogicalTensor
    mul_13: LogicalTensor
    add_13: LogicalTensor


FLUX_DOUBLE_BLOCK_OUTPUT: str = 'add_9'


def create_flux_double_block(
    prefix: str,
    layer_idx: int,
    *,
    image_seq_len: int,
    text_seq_len: int,
    p_img_attn_qkv_weight: LogicalTensor | None = None,
    p_img_attn_norm_query_norm_scale: LogicalTensor | None = None,
    p_img_attn_norm_key_norm_scale: LogicalTensor | None = None,
    p_img_attn_proj_weight: LogicalTensor | None = None,
    p_img_mlp_0_weight: LogicalTensor | None = None,
    p_img_mlp_2_weight: LogicalTensor | None = None,
    p_txt_attn_qkv_weight: LogicalTensor | None = None,
    p_txt_attn_norm_query_norm_scale: LogicalTensor | None = None,
    p_txt_attn_norm_key_norm_scale: LogicalTensor | None = None,
    p_txt_attn_proj_weight: LogicalTensor | None = None,
    p_txt_mlp_0_weight: LogicalTensor | None = None,
    p_txt_mlp_2_weight: LogicalTensor | None = None,
    img: LogicalTensor | None = None,
    txt: LogicalTensor | None = None,
    pe: LogicalTensor | None = None,
    pe_ctx: LogicalTensor | None = None,
    img_mod1_shift: LogicalTensor | None = None,
    img_mod1_scale: LogicalTensor | None = None,
    img_mod1_gate: LogicalTensor | None = None,
    img_mod2_shift: LogicalTensor | None = None,
    img_mod2_scale: LogicalTensor | None = None,
    img_mod2_gate: LogicalTensor | None = None,
    txt_mod1_shift: LogicalTensor | None = None,
    txt_mod1_scale: LogicalTensor | None = None,
    txt_mod1_gate: LogicalTensor | None = None,
    txt_mod2_shift: LogicalTensor | None = None,
    txt_mod2_scale: LogicalTensor | None = None,
    txt_mod2_gate: LogicalTensor | None = None,
    layer_norm: LogicalTensor | None = None,
    add: LogicalTensor | None = None,
    mul: LogicalTensor | None = None,
    add_1: LogicalTensor | None = None,
    linear: LogicalTensor | None = None,
    reshape: LogicalTensor | None = None,
    permute: LogicalTensor | None = None,
    getitem: LogicalTensor | None = None,
    getitem_1: LogicalTensor | None = None,
    getitem_2: LogicalTensor | None = None,
    rms_norm: LogicalTensor | None = None,
    rms_norm_1: LogicalTensor | None = None,
    to: LogicalTensor | None = None,
    to_1: LogicalTensor | None = None,
    layer_norm_1: LogicalTensor | None = None,
    add_2: LogicalTensor | None = None,
    mul_1: LogicalTensor | None = None,
    add_3: LogicalTensor | None = None,
    linear_1: LogicalTensor | None = None,
    reshape_1: LogicalTensor | None = None,
    permute_1: LogicalTensor | None = None,
    getitem_3: LogicalTensor | None = None,
    getitem_4: LogicalTensor | None = None,
    getitem_5: LogicalTensor | None = None,
    rms_norm_2: LogicalTensor | None = None,
    rms_norm_3: LogicalTensor | None = None,
    to_2: LogicalTensor | None = None,
    to_3: LogicalTensor | None = None,
    cat: LogicalTensor | None = None,
    cat_1: LogicalTensor | None = None,
    cat_2: LogicalTensor | None = None,
    cat_3: LogicalTensor | None = None,
    to_4: LogicalTensor | None = None,
    reshape_2: LogicalTensor | None = None,
    to_5: LogicalTensor | None = None,
    reshape_3: LogicalTensor | None = None,
    select: LogicalTensor | None = None,
    select_1: LogicalTensor | None = None,
    mul_2: LogicalTensor | None = None,
    select_2: LogicalTensor | None = None,
    select_3: LogicalTensor | None = None,
    mul_3: LogicalTensor | None = None,
    add_4: LogicalTensor | None = None,
    select_4: LogicalTensor | None = None,
    select_5: LogicalTensor | None = None,
    mul_4: LogicalTensor | None = None,
    select_6: LogicalTensor | None = None,
    select_7: LogicalTensor | None = None,
    mul_5: LogicalTensor | None = None,
    add_5: LogicalTensor | None = None,
    reshape_4: LogicalTensor | None = None,
    type_as: LogicalTensor | None = None,
    reshape_5: LogicalTensor | None = None,
    type_as_1: LogicalTensor | None = None,
    scaled_dot_product_attention: LogicalTensor | None = None,
    permute_2: LogicalTensor | None = None,
    reshape_6: LogicalTensor | None = None,
    slice_1: LogicalTensor | None = None,
    slice_2: LogicalTensor | None = None,
    linear_2: LogicalTensor | None = None,
    mul_6: LogicalTensor | None = None,
    add_6: LogicalTensor | None = None,
    add_7: LogicalTensor | None = None,
    layer_norm_2: LogicalTensor | None = None,
    mul_7: LogicalTensor | None = None,
    add_8: LogicalTensor | None = None,
    linear_3: LogicalTensor | None = None,
    getitem_6: LogicalTensor | None = None,
    getitem_7: LogicalTensor | None = None,
    silu: LogicalTensor | None = None,
    mul_8: LogicalTensor | None = None,
    linear_4: LogicalTensor | None = None,
    mul_9: LogicalTensor | None = None,
    add_9: LogicalTensor | None = None,
    linear_5: LogicalTensor | None = None,
    mul_10: LogicalTensor | None = None,
    add_10: LogicalTensor | None = None,
    add_11: LogicalTensor | None = None,
    layer_norm_3: LogicalTensor | None = None,
    mul_11: LogicalTensor | None = None,
    add_12: LogicalTensor | None = None,
    linear_6: LogicalTensor | None = None,
    getitem_8: LogicalTensor | None = None,
    getitem_9: LogicalTensor | None = None,
    silu_1: LogicalTensor | None = None,
    mul_12: LogicalTensor | None = None,
    linear_7: LogicalTensor | None = None,
    mul_13: LogicalTensor | None = None,
    add_13: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> FluxDoubleBlockTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('add_9', 'add_13')))
    tensors = FluxDoubleBlockTensors(
        p_img_attn_qkv_weight=_bind_tensor(
            p_img_attn_qkv_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.img_attn.qkv.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.img_attn.qkv.weight", dtype='float32', shape=(12288, 4096)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.img_attn.qkv.weight", dtype='float32', shape=(12288, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_img_attn_qkv_weight' in request_state_outputs,
            ),
        ),
        p_img_attn_norm_query_norm_scale=_bind_tensor(
            p_img_attn_norm_query_norm_scale,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.img_attn.norm.query_norm.scale",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.img_attn.norm.query_norm.scale", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.img_attn.norm.query_norm.scale", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_img_attn_norm_query_norm_scale' in request_state_outputs,
            ),
        ),
        p_img_attn_norm_key_norm_scale=_bind_tensor(
            p_img_attn_norm_key_norm_scale,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.img_attn.norm.key_norm.scale",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.img_attn.norm.key_norm.scale", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.img_attn.norm.key_norm.scale", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_img_attn_norm_key_norm_scale' in request_state_outputs,
            ),
        ),
        p_img_attn_proj_weight=_bind_tensor(
            p_img_attn_proj_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.img_attn.proj.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.img_attn.proj.weight", dtype='float32', shape=(4096, 4096)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.img_attn.proj.weight", dtype='float32', shape=(4096, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_img_attn_proj_weight' in request_state_outputs,
            ),
        ),
        p_img_mlp_0_weight=_bind_tensor(
            p_img_mlp_0_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.img_mlp.0.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.img_mlp.0.weight", dtype='float32', shape=(24576, 4096)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.img_mlp.0.weight", dtype='float32', shape=(24576, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_img_mlp_0_weight' in request_state_outputs,
            ),
        ),
        p_img_mlp_2_weight=_bind_tensor(
            p_img_mlp_2_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.img_mlp.2.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.img_mlp.2.weight", dtype='float32', shape=(4096, 12288)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.img_mlp.2.weight", dtype='float32', shape=(4096, 12288)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_img_mlp_2_weight' in request_state_outputs,
            ),
        ),
        p_txt_attn_qkv_weight=_bind_tensor(
            p_txt_attn_qkv_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.txt_attn.qkv.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.txt_attn.qkv.weight", dtype='float32', shape=(12288, 4096)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.txt_attn.qkv.weight", dtype='float32', shape=(12288, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_txt_attn_qkv_weight' in request_state_outputs,
            ),
        ),
        p_txt_attn_norm_query_norm_scale=_bind_tensor(
            p_txt_attn_norm_query_norm_scale,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.txt_attn.norm.query_norm.scale",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.txt_attn.norm.query_norm.scale", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.txt_attn.norm.query_norm.scale", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_txt_attn_norm_query_norm_scale' in request_state_outputs,
            ),
        ),
        p_txt_attn_norm_key_norm_scale=_bind_tensor(
            p_txt_attn_norm_key_norm_scale,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.txt_attn.norm.key_norm.scale",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.txt_attn.norm.key_norm.scale", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.txt_attn.norm.key_norm.scale", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_txt_attn_norm_key_norm_scale' in request_state_outputs,
            ),
        ),
        p_txt_attn_proj_weight=_bind_tensor(
            p_txt_attn_proj_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.txt_attn.proj.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.txt_attn.proj.weight", dtype='float32', shape=(4096, 4096)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.txt_attn.proj.weight", dtype='float32', shape=(4096, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_txt_attn_proj_weight' in request_state_outputs,
            ),
        ),
        p_txt_mlp_0_weight=_bind_tensor(
            p_txt_mlp_0_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.txt_mlp.0.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.txt_mlp.0.weight", dtype='float32', shape=(24576, 4096)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.txt_mlp.0.weight", dtype='float32', shape=(24576, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_txt_mlp_0_weight' in request_state_outputs,
            ),
        ),
        p_txt_mlp_2_weight=_bind_tensor(
            p_txt_mlp_2_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"double_blocks.{layer_idx}.txt_mlp.2.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"double_blocks.{layer_idx}.txt_mlp.2.weight", dtype='float32', shape=(4096, 12288)),
                layout=_quantized_weight_layout(f"double_blocks.{layer_idx}.txt_mlp.2.weight", dtype='float32', shape=(4096, 12288)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_txt_mlp_2_weight' in request_state_outputs,
            ),
        ),
        img=_bind_tensor(
            img,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='img' in request_state_outputs,
            ),
        ),
        txt=_bind_tensor(
            txt,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='txt' in request_state_outputs,
            ),
        ),
        pe=_bind_tensor(
            pe,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, image_seq_len, 64, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='pe' in request_state_outputs,
            ),
        ),
        pe_ctx=_bind_tensor(
            pe_ctx,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len, 64, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='pe_ctx' in request_state_outputs,
            ),
        ),
        img_mod1_shift=_bind_tensor(
            img_mod1_shift,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='img_mod1_shift' in request_state_outputs,
            ),
        ),
        img_mod1_scale=_bind_tensor(
            img_mod1_scale,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='img_mod1_scale' in request_state_outputs,
            ),
        ),
        img_mod1_gate=_bind_tensor(
            img_mod1_gate,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='img_mod1_gate' in request_state_outputs,
            ),
        ),
        img_mod2_shift=_bind_tensor(
            img_mod2_shift,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='img_mod2_shift' in request_state_outputs,
            ),
        ),
        img_mod2_scale=_bind_tensor(
            img_mod2_scale,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='img_mod2_scale' in request_state_outputs,
            ),
        ),
        img_mod2_gate=_bind_tensor(
            img_mod2_gate,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='img_mod2_gate' in request_state_outputs,
            ),
        ),
        txt_mod1_shift=_bind_tensor(
            txt_mod1_shift,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='txt_mod1_shift' in request_state_outputs,
            ),
        ),
        txt_mod1_scale=_bind_tensor(
            txt_mod1_scale,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='txt_mod1_scale' in request_state_outputs,
            ),
        ),
        txt_mod1_gate=_bind_tensor(
            txt_mod1_gate,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='txt_mod1_gate' in request_state_outputs,
            ),
        ),
        txt_mod2_shift=_bind_tensor(
            txt_mod2_shift,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='txt_mod2_shift' in request_state_outputs,
            ),
        ),
        txt_mod2_scale=_bind_tensor(
            txt_mod2_scale,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='txt_mod2_scale' in request_state_outputs,
            ),
        ),
        txt_mod2_gate=_bind_tensor(
            txt_mod2_gate,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='txt_mod2_gate' in request_state_outputs,
            ),
        ),
        layer_norm=_bind_tensor(
            layer_norm,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='layer_norm',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_norm' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            add,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add' in request_state_outputs,
            ),
        ),
        mul=_bind_tensor(
            mul,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul' in request_state_outputs,
            ),
        ),
        add_1=_bind_tensor(
            add_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_1' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            linear,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear' in request_state_outputs,
            ),
        ),
        reshape=_bind_tensor(
            reshape,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 3, 32, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape' in request_state_outputs,
            ),
        ),
        permute=_bind_tensor(
            permute,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(3, 1, 32, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute' in request_state_outputs,
            ),
        ),
        getitem=_bind_tensor(
            getitem,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem' in request_state_outputs,
            ),
        ),
        getitem_1=_bind_tensor(
            getitem_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_1' in request_state_outputs,
            ),
        ),
        getitem_2=_bind_tensor(
            getitem_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_2' in request_state_outputs,
            ),
        ),
        rms_norm=_bind_tensor(
            rms_norm,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rms_norm' in request_state_outputs,
            ),
        ),
        rms_norm_1=_bind_tensor(
            rms_norm_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rms_norm_1' in request_state_outputs,
            ),
        ),
        to=_bind_tensor(
            to,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to' in request_state_outputs,
            ),
        ),
        to_1=_bind_tensor(
            to_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_1' in request_state_outputs,
            ),
        ),
        layer_norm_1=_bind_tensor(
            layer_norm_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='layer_norm_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_norm_1' in request_state_outputs,
            ),
        ),
        add_2=_bind_tensor(
            add_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_2' in request_state_outputs,
            ),
        ),
        mul_1=_bind_tensor(
            mul_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_1' in request_state_outputs,
            ),
        ),
        add_3=_bind_tensor(
            add_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_3' in request_state_outputs,
            ),
        ),
        linear_1=_bind_tensor(
            linear_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_1' in request_state_outputs,
            ),
        ),
        reshape_1=_bind_tensor(
            reshape_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 3, 32, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_1' in request_state_outputs,
            ),
        ),
        permute_1=_bind_tensor(
            permute_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(3, 1, 32, text_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute_1' in request_state_outputs,
            ),
        ),
        getitem_3=_bind_tensor(
            getitem_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_3' in request_state_outputs,
            ),
        ),
        getitem_4=_bind_tensor(
            getitem_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_4' in request_state_outputs,
            ),
        ),
        getitem_5=_bind_tensor(
            getitem_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_5' in request_state_outputs,
            ),
        ),
        rms_norm_2=_bind_tensor(
            rms_norm_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rms_norm_2' in request_state_outputs,
            ),
        ),
        rms_norm_3=_bind_tensor(
            rms_norm_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rms_norm_3' in request_state_outputs,
            ),
        ),
        to_2=_bind_tensor(
            to_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_2' in request_state_outputs,
            ),
        ),
        to_3=_bind_tensor(
            to_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_3' in request_state_outputs,
            ),
        ),
        cat=_bind_tensor(
            cat,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len + image_seq_len, 64, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat' in request_state_outputs,
            ),
        ),
        cat_1=_bind_tensor(
            cat_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat_1' in request_state_outputs,
            ),
        ),
        cat_2=_bind_tensor(
            cat_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat_2' in request_state_outputs,
            ),
        ),
        cat_3=_bind_tensor(
            cat_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat_3' in request_state_outputs,
            ),
        ),
        to_4=_bind_tensor(
            to_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_4' in request_state_outputs,
            ),
        ),
        reshape_2=_bind_tensor(
            reshape_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_2' in request_state_outputs,
            ),
        ),
        to_5=_bind_tensor(
            to_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_5' in request_state_outputs,
            ),
        ),
        reshape_3=_bind_tensor(
            reshape_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_3' in request_state_outputs,
            ),
        ),
        select=_bind_tensor(
            select,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select' in request_state_outputs,
            ),
        ),
        select_1=_bind_tensor(
            select_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_1' in request_state_outputs,
            ),
        ),
        mul_2=_bind_tensor(
            mul_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_2' in request_state_outputs,
            ),
        ),
        select_2=_bind_tensor(
            select_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_2' in request_state_outputs,
            ),
        ),
        select_3=_bind_tensor(
            select_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_3' in request_state_outputs,
            ),
        ),
        mul_3=_bind_tensor(
            mul_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_3' in request_state_outputs,
            ),
        ),
        add_4=_bind_tensor(
            add_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_4' in request_state_outputs,
            ),
        ),
        select_4=_bind_tensor(
            select_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_4' in request_state_outputs,
            ),
        ),
        select_5=_bind_tensor(
            select_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_5' in request_state_outputs,
            ),
        ),
        mul_4=_bind_tensor(
            mul_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_4' in request_state_outputs,
            ),
        ),
        select_6=_bind_tensor(
            select_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_6' in request_state_outputs,
            ),
        ),
        select_7=_bind_tensor(
            select_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_7' in request_state_outputs,
            ),
        ),
        mul_5=_bind_tensor(
            mul_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_5' in request_state_outputs,
            ),
        ),
        add_5=_bind_tensor(
            add_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_5' in request_state_outputs,
            ),
        ),
        reshape_4=_bind_tensor(
            reshape_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_4' in request_state_outputs,
            ),
        ),
        type_as=_bind_tensor(
            type_as,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='type_as',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='type_as' in request_state_outputs,
            ),
        ),
        reshape_5=_bind_tensor(
            reshape_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_5' in request_state_outputs,
            ),
        ),
        type_as_1=_bind_tensor(
            type_as_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='type_as_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='type_as_1' in request_state_outputs,
            ),
        ),
        scaled_dot_product_attention=_bind_tensor(
            scaled_dot_product_attention,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='scaled_dot_product_attention',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='scaled_dot_product_attention' in request_state_outputs,
            ),
        ),
        permute_2=_bind_tensor(
            permute_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 32, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute_2' in request_state_outputs,
            ),
        ),
        reshape_6=_bind_tensor(
            reshape_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_6' in request_state_outputs,
            ),
        ),
        slice_1=_bind_tensor(
            slice_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='slice_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='slice_1' in request_state_outputs,
            ),
        ),
        slice_2=_bind_tensor(
            slice_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='slice_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='slice_2' in request_state_outputs,
            ),
        ),
        linear_2=_bind_tensor(
            linear_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_2' in request_state_outputs,
            ),
        ),
        mul_6=_bind_tensor(
            mul_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_6' in request_state_outputs,
            ),
        ),
        add_6=_bind_tensor(
            add_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_6' in request_state_outputs,
            ),
        ),
        add_7=_bind_tensor(
            add_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_7' in request_state_outputs,
            ),
        ),
        layer_norm_2=_bind_tensor(
            layer_norm_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='layer_norm_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_norm_2' in request_state_outputs,
            ),
        ),
        mul_7=_bind_tensor(
            mul_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_7' in request_state_outputs,
            ),
        ),
        add_8=_bind_tensor(
            add_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_8' in request_state_outputs,
            ),
        ),
        linear_3=_bind_tensor(
            linear_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 24576)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_3' in request_state_outputs,
            ),
        ),
        getitem_6=_bind_tensor(
            getitem_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_6' in request_state_outputs,
            ),
        ),
        getitem_7=_bind_tensor(
            getitem_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_7' in request_state_outputs,
            ),
        ),
        silu=_bind_tensor(
            silu,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='silu',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='silu' in request_state_outputs,
            ),
        ),
        mul_8=_bind_tensor(
            mul_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_8' in request_state_outputs,
            ),
        ),
        linear_4=_bind_tensor(
            linear_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_4' in request_state_outputs,
            ),
        ),
        mul_9=_bind_tensor(
            mul_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_9',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_9' in request_state_outputs,
            ),
        ),
        add_9=_bind_tensor(
            add_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_9',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_9' in request_state_outputs,
            ),
        ),
        linear_5=_bind_tensor(
            linear_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_5' in request_state_outputs,
            ),
        ),
        mul_10=_bind_tensor(
            mul_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_10',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_10' in request_state_outputs,
            ),
        ),
        add_10=_bind_tensor(
            add_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_10',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_10' in request_state_outputs,
            ),
        ),
        add_11=_bind_tensor(
            add_11,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_11',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_11' in request_state_outputs,
            ),
        ),
        layer_norm_3=_bind_tensor(
            layer_norm_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='layer_norm_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_norm_3' in request_state_outputs,
            ),
        ),
        mul_11=_bind_tensor(
            mul_11,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_11',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_11' in request_state_outputs,
            ),
        ),
        add_12=_bind_tensor(
            add_12,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_12',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_12' in request_state_outputs,
            ),
        ),
        linear_6=_bind_tensor(
            linear_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 24576)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_6' in request_state_outputs,
            ),
        ),
        getitem_8=_bind_tensor(
            getitem_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_8' in request_state_outputs,
            ),
        ),
        getitem_9=_bind_tensor(
            getitem_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_9',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_9' in request_state_outputs,
            ),
        ),
        silu_1=_bind_tensor(
            silu_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='silu_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='silu_1' in request_state_outputs,
            ),
        ),
        mul_12=_bind_tensor(
            mul_12,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_12',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_12' in request_state_outputs,
            ),
        ),
        linear_7=_bind_tensor(
            linear_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_7' in request_state_outputs,
            ),
        ),
        mul_13=_bind_tensor(
            mul_13,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_13',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_13' in request_state_outputs,
            ),
        ),
        add_13=_bind_tensor(
            add_13,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_13',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_13' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.linear, tensors.reshape)
    _bind_alias_source(tensors.rms_norm, tensors.to)
    _bind_alias_source(tensors.rms_norm_1, tensors.to_1)
    _bind_alias_source(tensors.linear_1, tensors.reshape_1)
    _bind_alias_source(tensors.rms_norm_2, tensors.to_2)
    _bind_alias_source(tensors.rms_norm_3, tensors.to_3)
    _bind_alias_source(tensors.cat_1, tensors.to_4)
    _bind_alias_source(tensors.to_4, tensors.reshape_2)
    _bind_alias_source(tensors.cat_2, tensors.to_5)
    _bind_alias_source(tensors.to_5, tensors.reshape_3)
    _bind_alias_source(tensors.add_4, tensors.reshape_4)
    _bind_alias_source(tensors.reshape_4, tensors.type_as)
    _bind_alias_source(tensors.add_5, tensors.reshape_5)
    _bind_alias_source(tensors.reshape_5, tensors.type_as_1)
    _bind_alias_source(tensors.permute_2, tensors.reshape_6)
    return tensors


_F16_TENSOR_NAMES = frozenset(())
_F16_TENSOR_PREFIXES = ()
_Q6_TENSOR_NAMES = frozenset(())
_Q6_TENSOR_PREFIXES = ()
_Q8_TENSOR_NAMES = frozenset(())
_Q8_TENSOR_PREFIXES = ('',)


def _quantized_weight_spec(checkpoint_key: str, *, dtype: str, shape: tuple[int, ...]) -> TensorSpec:
    force_f16 = checkpoint_key in _F16_TENSOR_NAMES or checkpoint_key.startswith(_F16_TENSOR_PREFIXES)
    if force_f16:
        return TensorSpec(dtype="float16", shape=shape)
    if dtype not in ("float32", "float16", "bfloat16"):
        return TensorSpec(dtype=dtype, shape=shape)
    force_q8 = checkpoint_key in _Q8_TENSOR_NAMES or checkpoint_key.startswith(_Q8_TENSOR_PREFIXES)
    force_q6 = checkpoint_key in _Q6_TENSOR_NAMES or checkpoint_key.startswith(_Q6_TENSOR_PREFIXES)
    if force_q8 and len(shape) >= 2:
        n, k = _quantized_matrix_shape(shape)
        padded_k = _round_up(k, 32)
        return TensorSpec(dtype="uint16", shape=(n, padded_k // 32 * 17))
    if force_q6 and len(shape) >= 2:
        n, k = _quantized_matrix_shape(shape)
        if k % 256 != 0:
            raise ValueError(f"Q6_K tensor {checkpoint_key} requires K to be divisible by 256, got {k}")
        return TensorSpec(dtype="uint16", shape=(n, k // 256 * 105))
    if len(shape) != 2:
        return TensorSpec(dtype=dtype, shape=shape)
    n, k = shape
    if k % 256 != 0:
        if k % 32 != 0:
            return TensorSpec(dtype="float32", shape=shape)
        return TensorSpec(dtype="uint16", shape=(n, k // 32 * 17))
    return TensorSpec(dtype="uint32", shape=(n, k // 256 * 36))


def _quantized_weight_layout(checkpoint_key: str, *, dtype: str, shape: tuple[int, ...]) -> TensorLayout:
    force_f16 = checkpoint_key in _F16_TENSOR_NAMES or checkpoint_key.startswith(_F16_TENSOR_PREFIXES)
    if force_f16:
        return CONTIGUOUS_LAYOUT
    if dtype not in ("float32", "float16", "bfloat16"):
        return CONTIGUOUS_LAYOUT
    force_q8 = checkpoint_key in _Q8_TENSOR_NAMES or checkpoint_key.startswith(_Q8_TENSOR_PREFIXES)
    force_q6 = checkpoint_key in _Q6_TENSOR_NAMES or checkpoint_key.startswith(_Q6_TENSOR_PREFIXES)
    if force_q8 and len(shape) >= 2:
        _, k = _quantized_matrix_shape(shape)
        return q8_0_halfwords_layout(logical_k=k)
    if force_q6 and len(shape) >= 2:
        _, k = _quantized_matrix_shape(shape)
        if k % 256 != 0:
            raise ValueError(f"Q6_K tensor {checkpoint_key} requires K to be divisible by 256, got {k}")
        return q6_k_halfwords_layout(logical_k=k)
    if len(shape) != 2:
        return CONTIGUOUS_LAYOUT
    _, k = shape
    if k % 256 != 0:
        if k % 32 != 0:
            return CONTIGUOUS_LAYOUT
        return q8_0_halfwords_layout(logical_k=k)
    return q4_k_words_layout(logical_k=k)


def _quantized_matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    rows = shape[0]
    cols = 1
    for dim in shape[1:]:
        cols *= dim
    return rows, cols


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _declare_tensor(
    *,
    spec: TensorSpec,
    role: TensorRole,
    memory: MemoryClass,
    lifetime: TensorLifetime,
    layout: TensorLayout = CONTIGUOUS_LAYOUT,
    checkpoint: str | None = None,
    checkpoint_key: str | None = None,
    reference_key: str | None = None,
    layer: str | None = None,
    request_state: bool = False,
) -> LogicalTensor:
    if request_state:
        role = TensorRole.OUTPUT
        memory = MemoryClass.REQUEST_STATE
        lifetime = TensorLifetime.REQUEST
    return LogicalTensor(
        spec=spec,
        role=role,
        memory=memory,
        lifetime=lifetime,
        checkpoint=checkpoint,
        checkpoint_key=checkpoint_key,
        reference_key=reference_key,
        layer=layer,
        layout=layout,
    )


def _bind_tensor(
    bound: LogicalTensor | None,
    tensor: LogicalTensor,
) -> LogicalTensor:
    if bound is None:
        return tensor
    if bound.spec != tensor.spec:
        bound_name = bound.name or "<bound>"
        tensor_name = tensor.name or "<declared>"
        raise ValueError(f"{bound_name} spec {bound.spec} does not match {tensor_name} spec {tensor.spec}")
    return bound


def _bind_alias_source(src: LogicalTensor, dst: LogicalTensor) -> None:
    bind_logical_tensor_alias(src, dst)


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
