"""Generated dispatch function for run_audio_decode."""

from __future__ import annotations

from models.optimized_omnivoice.tensors.model import model_tensors
from models.optimized_omnivoice.shaders.audio_decode_add_f32 import AUDIO_DECODE_ADD_F32
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32 import CONV1D_Q8_0W_F32B_F32
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_21 import CONV1D_Q8_0W_F32B_F32_21
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_22 import CONV1D_Q8_0W_F32B_F32_22
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_23 import CONV1D_Q8_0W_F32B_F32_23
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_24 import CONV1D_Q8_0W_F32B_F32_24
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_25 import CONV1D_Q8_0W_F32B_F32_25
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_26 import CONV1D_Q8_0W_F32B_F32_26
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_28 import CONV1D_Q8_0W_F32B_F32_28
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_29 import CONV1D_Q8_0W_F32B_F32_29
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_30 import CONV1D_Q8_0W_F32B_F32_30
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_31 import CONV1D_Q8_0W_F32B_F32_31
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_32 import CONV1D_Q8_0W_F32B_F32_32
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_34 import CONV1D_Q8_0W_F32B_F32_34
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_35 import CONV1D_Q8_0W_F32B_F32_35
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_36 import CONV1D_Q8_0W_F32B_F32_36
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_37 import CONV1D_Q8_0W_F32B_F32_37
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_38 import CONV1D_Q8_0W_F32B_F32_38
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_39 import CONV1D_Q8_0W_F32B_F32_39
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_41 import CONV1D_Q8_0W_F32B_F32_41
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_42 import CONV1D_Q8_0W_F32B_F32_42
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_43 import CONV1D_Q8_0W_F32B_F32_43
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_44 import CONV1D_Q8_0W_F32B_F32_44
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_45 import CONV1D_Q8_0W_F32B_F32_45
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_46 import CONV1D_Q8_0W_F32B_F32_46
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_48 import CONV1D_Q8_0W_F32B_F32_48
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_49 import CONV1D_Q8_0W_F32B_F32_49
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_50 import CONV1D_Q8_0W_F32B_F32_50
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_51 import CONV1D_Q8_0W_F32B_F32_51
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_52 import CONV1D_Q8_0W_F32B_F32_52
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_53 import CONV1D_Q8_0W_F32B_F32_53
from models.optimized_omnivoice.shaders.conv1d_q8_0w_f32b_f32_54 import CONV1D_Q8_0W_F32B_F32_54
from models.optimized_omnivoice.shaders.conv_transpose1d_q8_0w_f32b_f32 import CONV_TRANSPOSE1D_Q8_0W_F32B_F32
from models.optimized_omnivoice.shaders.conv_transpose1d_q8_0w_f32b_f32_27 import CONV_TRANSPOSE1D_Q8_0W_F32B_F32_27
from models.optimized_omnivoice.shaders.conv_transpose1d_q8_0w_f32b_f32_33 import CONV_TRANSPOSE1D_Q8_0W_F32B_F32_33
from models.optimized_omnivoice.shaders.conv_transpose1d_q8_0w_f32b_f32_40 import CONV_TRANSPOSE1D_Q8_0W_F32B_F32_40
from models.optimized_omnivoice.shaders.conv_transpose1d_q8_0w_f32b_f32_47 import CONV_TRANSPOSE1D_Q8_0W_F32B_F32_47
from models.optimized_omnivoice.shaders.embedding_q8_0_f32 import EMBEDDING_Q8_0_F32
from models.optimized_omnivoice.shaders.linear_bias_q8_0w_f32b_f32 import LINEAR_BIAS_Q8_0W_F32B_F32
from models.optimized_omnivoice.shaders.omnivoice_snake_f16 import OMNIVOICE_SNAKE_F16
from models.optimized_omnivoice.shaders.permute_f32_c87693fdcf import PERMUTE_F32_C87693FDCF
from models.optimized_omnivoice.shaders.select_int64 import SELECT_INT64
from models.optimized_omnivoice.shaders.select_int64_10 import SELECT_INT64_10
from models.optimized_omnivoice.shaders.select_int64_4 import SELECT_INT64_4
from models.optimized_omnivoice.shaders.select_int64_5 import SELECT_INT64_5
from models.optimized_omnivoice.shaders.select_int64_6 import SELECT_INT64_6
from models.optimized_omnivoice.shaders.select_int64_7 import SELECT_INT64_7
from models.optimized_omnivoice.shaders.select_int64_8 import SELECT_INT64_8
from models.optimized_omnivoice.shaders.select_int64_9 import SELECT_INT64_9
from models.optimized_omnivoice.shaders.transpose_f32_0ae87925be import TRANSPOSE_F32_0AE87925BE
from models.optimized_omnivoice.shaders.transpose_f32_7d30938139 import TRANSPOSE_F32_7D30938139
from models.optimized_omnivoice.tensors.audio_decode import AudioDecodeTensors
from torch2vk.runtime.session import RuntimeSession


def _run_audio_decode_with_tensors(rt: RuntimeSession, tensors: AudioDecodeTensors) -> None:
    SELECT_INT64(rt, x=tensors.audio_codes, output=tensors.select)
    EMBEDDING_Q8_0_F32(rt, weight=tensors.b_quantizer_quantizers_0_codebook_embed, indices=tensors.to, output=tensors.embedding)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.embedding, weight=tensors.p_quantizer_quantizers_0_project_out_weight, bias=tensors.p_quantizer_quantizers_0_project_out_bias, output=tensors.linear)
    PERMUTE_F32_C87693FDCF(rt, x=tensors.linear, output=tensors.permute)
    SELECT_INT64_4(rt, x=tensors.audio_codes, output=tensors.select_1)
    EMBEDDING_Q8_0_F32(rt, weight=tensors.b_quantizer_quantizers_1_codebook_embed, indices=tensors.to_1, output=tensors.embedding_1)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.embedding_1, weight=tensors.p_quantizer_quantizers_1_project_out_weight, bias=tensors.p_quantizer_quantizers_1_project_out_bias, output=tensors.linear_1)
    PERMUTE_F32_C87693FDCF(rt, x=tensors.linear_1, output=tensors.permute_1)
    SELECT_INT64_5(rt, x=tensors.audio_codes, output=tensors.select_2)
    EMBEDDING_Q8_0_F32(rt, weight=tensors.b_quantizer_quantizers_2_codebook_embed, indices=tensors.to_2, output=tensors.embedding_2)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.embedding_2, weight=tensors.p_quantizer_quantizers_2_project_out_weight, bias=tensors.p_quantizer_quantizers_2_project_out_bias, output=tensors.linear_2)
    PERMUTE_F32_C87693FDCF(rt, x=tensors.linear_2, output=tensors.permute_2)
    SELECT_INT64_6(rt, x=tensors.audio_codes, output=tensors.select_3)
    EMBEDDING_Q8_0_F32(rt, weight=tensors.b_quantizer_quantizers_3_codebook_embed, indices=tensors.to_3, output=tensors.embedding_3)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.embedding_3, weight=tensors.p_quantizer_quantizers_3_project_out_weight, bias=tensors.p_quantizer_quantizers_3_project_out_bias, output=tensors.linear_3)
    PERMUTE_F32_C87693FDCF(rt, x=tensors.linear_3, output=tensors.permute_3)
    SELECT_INT64_7(rt, x=tensors.audio_codes, output=tensors.select_4)
    EMBEDDING_Q8_0_F32(rt, weight=tensors.b_quantizer_quantizers_4_codebook_embed, indices=tensors.to_4, output=tensors.embedding_4)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.embedding_4, weight=tensors.p_quantizer_quantizers_4_project_out_weight, bias=tensors.p_quantizer_quantizers_4_project_out_bias, output=tensors.linear_4)
    PERMUTE_F32_C87693FDCF(rt, x=tensors.linear_4, output=tensors.permute_4)
    SELECT_INT64_8(rt, x=tensors.audio_codes, output=tensors.select_5)
    EMBEDDING_Q8_0_F32(rt, weight=tensors.b_quantizer_quantizers_5_codebook_embed, indices=tensors.to_5, output=tensors.embedding_5)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.embedding_5, weight=tensors.p_quantizer_quantizers_5_project_out_weight, bias=tensors.p_quantizer_quantizers_5_project_out_bias, output=tensors.linear_5)
    PERMUTE_F32_C87693FDCF(rt, x=tensors.linear_5, output=tensors.permute_5)
    SELECT_INT64_9(rt, x=tensors.audio_codes, output=tensors.select_6)
    EMBEDDING_Q8_0_F32(rt, weight=tensors.b_quantizer_quantizers_6_codebook_embed, indices=tensors.to_6, output=tensors.embedding_6)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.embedding_6, weight=tensors.p_quantizer_quantizers_6_project_out_weight, bias=tensors.p_quantizer_quantizers_6_project_out_bias, output=tensors.linear_6)
    PERMUTE_F32_C87693FDCF(rt, x=tensors.linear_6, output=tensors.permute_6)
    SELECT_INT64_10(rt, x=tensors.audio_codes, output=tensors.select_7)
    EMBEDDING_Q8_0_F32(rt, weight=tensors.b_quantizer_quantizers_7_codebook_embed, indices=tensors.to_7, output=tensors.embedding_7)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.embedding_7, weight=tensors.p_quantizer_quantizers_7_project_out_weight, bias=tensors.p_quantizer_quantizers_7_project_out_bias, output=tensors.linear_7)
    PERMUTE_F32_C87693FDCF(rt, x=tensors.linear_7, output=tensors.permute_7)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.permute, y=tensors.permute_1, output=tensors.add)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add, y=tensors.permute_2, output=tensors.add_1)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_1, y=tensors.permute_3, output=tensors.add_2)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_2, y=tensors.permute_4, output=tensors.add_3)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_3, y=tensors.permute_5, output=tensors.add_4)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_4, y=tensors.permute_6, output=tensors.add_5)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_5, y=tensors.permute_7, output=tensors.add_6)
    TRANSPOSE_F32_7D30938139(rt, x=tensors.add_6, output=tensors.transpose)
    LINEAR_BIAS_Q8_0W_F32B_F32(rt, x=tensors.transpose, weight=tensors.p_fc2_weight, bias=tensors.p_fc2_bias, output=tensors.linear_8)
    TRANSPOSE_F32_0AE87925BE(rt, x=tensors.linear_8, output=tensors.transpose_1)
    CONV1D_Q8_0W_F32B_F32(rt, x=tensors.transpose_1, weight=tensors.p_acoustic_decoder_conv1_weight, bias=tensors.p_acoustic_decoder_conv1_bias, output=tensors.conv1d)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_0_snake1_alpha, x=tensors.reshape, output=tensors.add_8)
    CONV_TRANSPOSE1D_Q8_0W_F32B_F32(rt, x=tensors.reshape_1, weight=tensors.p_acoustic_decoder_block_0_conv_t1_weight, bias=tensors.p_acoustic_decoder_block_0_conv_t1_bias, output=tensors.conv_transpose1d)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_0_res_unit1_snake1_alpha, x=tensors.reshape_2, output=tensors.add_10)
    CONV1D_Q8_0W_F32B_F32_21(rt, x=tensors.reshape_3, weight=tensors.p_acoustic_decoder_block_0_res_unit1_conv1_weight, bias=tensors.p_acoustic_decoder_block_0_res_unit1_conv1_bias, output=tensors.conv1d_1)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_0_res_unit1_snake2_alpha, x=tensors.reshape_4, output=tensors.add_12)
    CONV1D_Q8_0W_F32B_F32_22(rt, x=tensors.reshape_5, weight=tensors.p_acoustic_decoder_block_0_res_unit1_conv2_weight, bias=tensors.p_acoustic_decoder_block_0_res_unit1_conv2_bias, output=tensors.conv1d_2)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.conv_transpose1d, y=tensors.conv1d_2, output=tensors.add_13)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_0_res_unit2_snake1_alpha, x=tensors.reshape_6, output=tensors.add_15)
    CONV1D_Q8_0W_F32B_F32_23(rt, x=tensors.reshape_7, weight=tensors.p_acoustic_decoder_block_0_res_unit2_conv1_weight, bias=tensors.p_acoustic_decoder_block_0_res_unit2_conv1_bias, output=tensors.conv1d_3)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_0_res_unit2_snake2_alpha, x=tensors.reshape_8, output=tensors.add_17)
    CONV1D_Q8_0W_F32B_F32_24(rt, x=tensors.reshape_9, weight=tensors.p_acoustic_decoder_block_0_res_unit2_conv2_weight, bias=tensors.p_acoustic_decoder_block_0_res_unit2_conv2_bias, output=tensors.conv1d_4)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_13, y=tensors.conv1d_4, output=tensors.add_18)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_0_res_unit3_snake1_alpha, x=tensors.reshape_10, output=tensors.add_20)
    CONV1D_Q8_0W_F32B_F32_25(rt, x=tensors.reshape_11, weight=tensors.p_acoustic_decoder_block_0_res_unit3_conv1_weight, bias=tensors.p_acoustic_decoder_block_0_res_unit3_conv1_bias, output=tensors.conv1d_5)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_0_res_unit3_snake2_alpha, x=tensors.reshape_12, output=tensors.add_22)
    CONV1D_Q8_0W_F32B_F32_26(rt, x=tensors.reshape_13, weight=tensors.p_acoustic_decoder_block_0_res_unit3_conv2_weight, bias=tensors.p_acoustic_decoder_block_0_res_unit3_conv2_bias, output=tensors.conv1d_6)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_18, y=tensors.conv1d_6, output=tensors.add_23)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_1_snake1_alpha, x=tensors.reshape_14, output=tensors.add_25)
    CONV_TRANSPOSE1D_Q8_0W_F32B_F32_27(rt, x=tensors.reshape_15, weight=tensors.p_acoustic_decoder_block_1_conv_t1_weight, bias=tensors.p_acoustic_decoder_block_1_conv_t1_bias, output=tensors.conv_transpose1d_1)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_1_res_unit1_snake1_alpha, x=tensors.reshape_16, output=tensors.add_27)
    CONV1D_Q8_0W_F32B_F32(rt, x=tensors.reshape_17, weight=tensors.p_acoustic_decoder_block_1_res_unit1_conv1_weight, bias=tensors.p_acoustic_decoder_block_1_res_unit1_conv1_bias, output=tensors.conv1d_7)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_1_res_unit1_snake2_alpha, x=tensors.reshape_18, output=tensors.add_29)
    CONV1D_Q8_0W_F32B_F32_28(rt, x=tensors.reshape_19, weight=tensors.p_acoustic_decoder_block_1_res_unit1_conv2_weight, bias=tensors.p_acoustic_decoder_block_1_res_unit1_conv2_bias, output=tensors.conv1d_8)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.conv_transpose1d_1, y=tensors.conv1d_8, output=tensors.add_30)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_1_res_unit2_snake1_alpha, x=tensors.reshape_20, output=tensors.add_32)
    CONV1D_Q8_0W_F32B_F32_29(rt, x=tensors.reshape_21, weight=tensors.p_acoustic_decoder_block_1_res_unit2_conv1_weight, bias=tensors.p_acoustic_decoder_block_1_res_unit2_conv1_bias, output=tensors.conv1d_9)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_1_res_unit2_snake2_alpha, x=tensors.reshape_22, output=tensors.add_34)
    CONV1D_Q8_0W_F32B_F32_30(rt, x=tensors.reshape_23, weight=tensors.p_acoustic_decoder_block_1_res_unit2_conv2_weight, bias=tensors.p_acoustic_decoder_block_1_res_unit2_conv2_bias, output=tensors.conv1d_10)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_30, y=tensors.conv1d_10, output=tensors.add_35)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_1_res_unit3_snake1_alpha, x=tensors.reshape_24, output=tensors.add_37)
    CONV1D_Q8_0W_F32B_F32_31(rt, x=tensors.reshape_25, weight=tensors.p_acoustic_decoder_block_1_res_unit3_conv1_weight, bias=tensors.p_acoustic_decoder_block_1_res_unit3_conv1_bias, output=tensors.conv1d_11)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_1_res_unit3_snake2_alpha, x=tensors.reshape_26, output=tensors.add_39)
    CONV1D_Q8_0W_F32B_F32_32(rt, x=tensors.reshape_27, weight=tensors.p_acoustic_decoder_block_1_res_unit3_conv2_weight, bias=tensors.p_acoustic_decoder_block_1_res_unit3_conv2_bias, output=tensors.conv1d_12)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_35, y=tensors.conv1d_12, output=tensors.add_40)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_2_snake1_alpha, x=tensors.reshape_28, output=tensors.add_42)
    CONV_TRANSPOSE1D_Q8_0W_F32B_F32_33(rt, x=tensors.reshape_29, weight=tensors.p_acoustic_decoder_block_2_conv_t1_weight, bias=tensors.p_acoustic_decoder_block_2_conv_t1_bias, output=tensors.conv_transpose1d_2)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_2_res_unit1_snake1_alpha, x=tensors.reshape_30, output=tensors.add_44)
    CONV1D_Q8_0W_F32B_F32_34(rt, x=tensors.reshape_31, weight=tensors.p_acoustic_decoder_block_2_res_unit1_conv1_weight, bias=tensors.p_acoustic_decoder_block_2_res_unit1_conv1_bias, output=tensors.conv1d_13)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_2_res_unit1_snake2_alpha, x=tensors.reshape_32, output=tensors.add_46)
    CONV1D_Q8_0W_F32B_F32_35(rt, x=tensors.reshape_33, weight=tensors.p_acoustic_decoder_block_2_res_unit1_conv2_weight, bias=tensors.p_acoustic_decoder_block_2_res_unit1_conv2_bias, output=tensors.conv1d_14)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.conv_transpose1d_2, y=tensors.conv1d_14, output=tensors.add_47)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_2_res_unit2_snake1_alpha, x=tensors.reshape_34, output=tensors.add_49)
    CONV1D_Q8_0W_F32B_F32_36(rt, x=tensors.reshape_35, weight=tensors.p_acoustic_decoder_block_2_res_unit2_conv1_weight, bias=tensors.p_acoustic_decoder_block_2_res_unit2_conv1_bias, output=tensors.conv1d_15)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_2_res_unit2_snake2_alpha, x=tensors.reshape_36, output=tensors.add_51)
    CONV1D_Q8_0W_F32B_F32_37(rt, x=tensors.reshape_37, weight=tensors.p_acoustic_decoder_block_2_res_unit2_conv2_weight, bias=tensors.p_acoustic_decoder_block_2_res_unit2_conv2_bias, output=tensors.conv1d_16)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_47, y=tensors.conv1d_16, output=tensors.add_52)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_2_res_unit3_snake1_alpha, x=tensors.reshape_38, output=tensors.add_54)
    CONV1D_Q8_0W_F32B_F32_38(rt, x=tensors.reshape_39, weight=tensors.p_acoustic_decoder_block_2_res_unit3_conv1_weight, bias=tensors.p_acoustic_decoder_block_2_res_unit3_conv1_bias, output=tensors.conv1d_17)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_2_res_unit3_snake2_alpha, x=tensors.reshape_40, output=tensors.add_56)
    CONV1D_Q8_0W_F32B_F32_39(rt, x=tensors.reshape_41, weight=tensors.p_acoustic_decoder_block_2_res_unit3_conv2_weight, bias=tensors.p_acoustic_decoder_block_2_res_unit3_conv2_bias, output=tensors.conv1d_18)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_52, y=tensors.conv1d_18, output=tensors.add_57)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_3_snake1_alpha, x=tensors.reshape_42, output=tensors.add_59)
    CONV_TRANSPOSE1D_Q8_0W_F32B_F32_40(rt, x=tensors.reshape_43, weight=tensors.p_acoustic_decoder_block_3_conv_t1_weight, bias=tensors.p_acoustic_decoder_block_3_conv_t1_bias, output=tensors.conv_transpose1d_3)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_3_res_unit1_snake1_alpha, x=tensors.reshape_44, output=tensors.add_61)
    CONV1D_Q8_0W_F32B_F32_41(rt, x=tensors.reshape_45, weight=tensors.p_acoustic_decoder_block_3_res_unit1_conv1_weight, bias=tensors.p_acoustic_decoder_block_3_res_unit1_conv1_bias, output=tensors.conv1d_19)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_3_res_unit1_snake2_alpha, x=tensors.reshape_46, output=tensors.add_63)
    CONV1D_Q8_0W_F32B_F32_42(rt, x=tensors.reshape_47, weight=tensors.p_acoustic_decoder_block_3_res_unit1_conv2_weight, bias=tensors.p_acoustic_decoder_block_3_res_unit1_conv2_bias, output=tensors.conv1d_20)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.conv_transpose1d_3, y=tensors.conv1d_20, output=tensors.add_64)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_3_res_unit2_snake1_alpha, x=tensors.reshape_48, output=tensors.add_66)
    CONV1D_Q8_0W_F32B_F32_43(rt, x=tensors.reshape_49, weight=tensors.p_acoustic_decoder_block_3_res_unit2_conv1_weight, bias=tensors.p_acoustic_decoder_block_3_res_unit2_conv1_bias, output=tensors.conv1d_21)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_3_res_unit2_snake2_alpha, x=tensors.reshape_50, output=tensors.add_68)
    CONV1D_Q8_0W_F32B_F32_44(rt, x=tensors.reshape_51, weight=tensors.p_acoustic_decoder_block_3_res_unit2_conv2_weight, bias=tensors.p_acoustic_decoder_block_3_res_unit2_conv2_bias, output=tensors.conv1d_22)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_64, y=tensors.conv1d_22, output=tensors.add_69)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_3_res_unit3_snake1_alpha, x=tensors.reshape_52, output=tensors.add_71)
    CONV1D_Q8_0W_F32B_F32_45(rt, x=tensors.reshape_53, weight=tensors.p_acoustic_decoder_block_3_res_unit3_conv1_weight, bias=tensors.p_acoustic_decoder_block_3_res_unit3_conv1_bias, output=tensors.conv1d_23)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_3_res_unit3_snake2_alpha, x=tensors.reshape_54, output=tensors.add_73)
    CONV1D_Q8_0W_F32B_F32_46(rt, x=tensors.reshape_55, weight=tensors.p_acoustic_decoder_block_3_res_unit3_conv2_weight, bias=tensors.p_acoustic_decoder_block_3_res_unit3_conv2_bias, output=tensors.conv1d_24)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_69, y=tensors.conv1d_24, output=tensors.add_74)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_4_snake1_alpha, x=tensors.reshape_56, output=tensors.add_76)
    CONV_TRANSPOSE1D_Q8_0W_F32B_F32_47(rt, x=tensors.reshape_57, weight=tensors.p_acoustic_decoder_block_4_conv_t1_weight, bias=tensors.p_acoustic_decoder_block_4_conv_t1_bias, output=tensors.conv_transpose1d_4)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_4_res_unit1_snake1_alpha, x=tensors.reshape_58, output=tensors.add_78)
    CONV1D_Q8_0W_F32B_F32_48(rt, x=tensors.reshape_59, weight=tensors.p_acoustic_decoder_block_4_res_unit1_conv1_weight, bias=tensors.p_acoustic_decoder_block_4_res_unit1_conv1_bias, output=tensors.conv1d_25)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_4_res_unit1_snake2_alpha, x=tensors.reshape_60, output=tensors.add_80)
    CONV1D_Q8_0W_F32B_F32_49(rt, x=tensors.reshape_61, weight=tensors.p_acoustic_decoder_block_4_res_unit1_conv2_weight, bias=tensors.p_acoustic_decoder_block_4_res_unit1_conv2_bias, output=tensors.conv1d_26)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.conv_transpose1d_4, y=tensors.conv1d_26, output=tensors.add_81)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_4_res_unit2_snake1_alpha, x=tensors.reshape_62, output=tensors.add_83)
    CONV1D_Q8_0W_F32B_F32_50(rt, x=tensors.reshape_63, weight=tensors.p_acoustic_decoder_block_4_res_unit2_conv1_weight, bias=tensors.p_acoustic_decoder_block_4_res_unit2_conv1_bias, output=tensors.conv1d_27)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_4_res_unit2_snake2_alpha, x=tensors.reshape_64, output=tensors.add_85)
    CONV1D_Q8_0W_F32B_F32_51(rt, x=tensors.reshape_65, weight=tensors.p_acoustic_decoder_block_4_res_unit2_conv2_weight, bias=tensors.p_acoustic_decoder_block_4_res_unit2_conv2_bias, output=tensors.conv1d_28)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_81, y=tensors.conv1d_28, output=tensors.add_86)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_4_res_unit3_snake1_alpha, x=tensors.reshape_66, output=tensors.add_88)
    CONV1D_Q8_0W_F32B_F32_52(rt, x=tensors.reshape_67, weight=tensors.p_acoustic_decoder_block_4_res_unit3_conv1_weight, bias=tensors.p_acoustic_decoder_block_4_res_unit3_conv1_bias, output=tensors.conv1d_29)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_block_4_res_unit3_snake2_alpha, x=tensors.reshape_68, output=tensors.add_90)
    CONV1D_Q8_0W_F32B_F32_53(rt, x=tensors.reshape_69, weight=tensors.p_acoustic_decoder_block_4_res_unit3_conv2_weight, bias=tensors.p_acoustic_decoder_block_4_res_unit3_conv2_bias, output=tensors.conv1d_30)
    AUDIO_DECODE_ADD_F32(rt, x=tensors.add_86, y=tensors.conv1d_30, output=tensors.add_91)
    OMNIVOICE_SNAKE_F16(rt, alpha=tensors.p_acoustic_decoder_snake1_alpha, x=tensors.reshape_70, output=tensors.add_93)
    CONV1D_Q8_0W_F32B_F32_54(rt, x=tensors.reshape_71, weight=tensors.p_acoustic_decoder_conv2_weight, bias=tensors.p_acoustic_decoder_conv2_bias, output=tensors.conv1d_31)


def run_audio_decode_with_tensors(rt: RuntimeSession, tensors: AudioDecodeTensors) -> None:
    _run_audio_decode_with_tensors(rt, tensors)


def run_audio_decode(rt: RuntimeSession) -> None:
    _run_audio_decode_with_tensors(rt, model_tensors().audio_decode)
