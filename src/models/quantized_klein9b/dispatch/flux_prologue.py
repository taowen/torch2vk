"""Generated dispatch function for run_flux_prologue."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.arange_f32 import ARANGE_F32
from models.quantized_klein9b.shaders.arange_i64 import ARANGE_I64
from models.quantized_klein9b.shaders.cat_4_f32 import CAT_4_F32
from models.quantized_klein9b.shaders.cos_f32 import COS_F32
from models.quantized_klein9b.shaders.cos_f32_32 import COS_F32_32
from models.quantized_klein9b.shaders.cos_f32_36 import COS_F32_36
from models.quantized_klein9b.shaders.cos_f32_41 import COS_F32_41
from models.quantized_klein9b.shaders.cos_f32_44 import COS_F32_44
from models.quantized_klein9b.shaders.cos_f32_48 import COS_F32_48
from models.quantized_klein9b.shaders.cos_f32_51 import COS_F32_51
from models.quantized_klein9b.shaders.cos_f32_55 import COS_F32_55
from models.quantized_klein9b.shaders.cos_f32_58 import COS_F32_58
from models.quantized_klein9b.shaders.cos_f32_62 import COS_F32_62
from models.quantized_klein9b.shaders.cos_f32_65 import COS_F32_65
from models.quantized_klein9b.shaders.cos_f32_69 import COS_F32_69
from models.quantized_klein9b.shaders.cos_f32_72 import COS_F32_72
from models.quantized_klein9b.shaders.cos_f32_76 import COS_F32_76
from models.quantized_klein9b.shaders.cos_f32_79 import COS_F32_79
from models.quantized_klein9b.shaders.cos_f32_83 import COS_F32_83
from models.quantized_klein9b.shaders.cos_f32_86 import COS_F32_86
from models.quantized_klein9b.shaders.div_scalar import DIV_SCALAR
from models.quantized_klein9b.shaders.div_scalar_27 import DIV_SCALAR_27
from models.quantized_klein9b.shaders.div_scalar_39 import DIV_SCALAR_39
from models.quantized_klein9b.shaders.div_scalar_46 import DIV_SCALAR_46
from models.quantized_klein9b.shaders.div_scalar_53 import DIV_SCALAR_53
from models.quantized_klein9b.shaders.div_scalar_60 import DIV_SCALAR_60
from models.quantized_klein9b.shaders.div_scalar_67 import DIV_SCALAR_67
from models.quantized_klein9b.shaders.div_scalar_74 import DIV_SCALAR_74
from models.quantized_klein9b.shaders.div_scalar_81 import DIV_SCALAR_81
from models.quantized_klein9b.shaders.einsum_outer_f32 import EINSUM_OUTER_F32
from models.quantized_klein9b.shaders.exp_f32 import EXP_F32
from models.quantized_klein9b.shaders.flux_prologue_cat_2_f32 import FLUX_PROLOGUE_CAT_2_F32
from models.quantized_klein9b.shaders.flux_prologue_mul_broadcast import FLUX_PROLOGUE_MUL_BROADCAST
from models.quantized_klein9b.shaders.flux_prologue_neg_f32 import FLUX_PROLOGUE_NEG_F32
from models.quantized_klein9b.shaders.flux_prologue_silu_f32 import FLUX_PROLOGUE_SILU_F32
from models.quantized_klein9b.shaders.linear_nobias_q8_0_f32_act_f32 import LINEAR_NOBIAS_Q8_0_F32_ACT_F32
from models.quantized_klein9b.shaders.linear_nobias_q8_0_matvec_f32_act_f32 import LINEAR_NOBIAS_Q8_0_MATVEC_F32_ACT_F32
from models.quantized_klein9b.shaders.mul_scalar import MUL_SCALAR
from models.quantized_klein9b.shaders.mul_scalar_2 import MUL_SCALAR_2
from models.quantized_klein9b.shaders.mul_scalar_30 import MUL_SCALAR_30
from models.quantized_klein9b.shaders.mul_scalar_40 import MUL_SCALAR_40
from models.quantized_klein9b.shaders.mul_scalar_47 import MUL_SCALAR_47
from models.quantized_klein9b.shaders.mul_scalar_54 import MUL_SCALAR_54
from models.quantized_klein9b.shaders.mul_scalar_61 import MUL_SCALAR_61
from models.quantized_klein9b.shaders.mul_scalar_68 import MUL_SCALAR_68
from models.quantized_klein9b.shaders.mul_scalar_75 import MUL_SCALAR_75
from models.quantized_klein9b.shaders.mul_scalar_82 import MUL_SCALAR_82
from models.quantized_klein9b.shaders.pow_base_scalar_f32 import POW_BASE_SCALAR_F32
from models.quantized_klein9b.shaders.reciprocal_f32 import RECIPROCAL_F32
from models.quantized_klein9b.shaders.select_int64 import SELECT_INT64
from models.quantized_klein9b.shaders.select_int64_38 import SELECT_INT64_38
from models.quantized_klein9b.shaders.select_int64_45 import SELECT_INT64_45
from models.quantized_klein9b.shaders.select_int64_52 import SELECT_INT64_52
from models.quantized_klein9b.shaders.select_int64_66 import SELECT_INT64_66
from models.quantized_klein9b.shaders.select_int64_73 import SELECT_INT64_73
from models.quantized_klein9b.shaders.select_int64_80 import SELECT_INT64_80
from models.quantized_klein9b.shaders.sin_f32 import SIN_F32
from models.quantized_klein9b.shaders.sin_f32_33 import SIN_F32_33
from models.quantized_klein9b.shaders.sin_f32_35 import SIN_F32_35
from models.quantized_klein9b.shaders.sin_f32_42 import SIN_F32_42
from models.quantized_klein9b.shaders.sin_f32_43 import SIN_F32_43
from models.quantized_klein9b.shaders.sin_f32_49 import SIN_F32_49
from models.quantized_klein9b.shaders.sin_f32_50 import SIN_F32_50
from models.quantized_klein9b.shaders.sin_f32_56 import SIN_F32_56
from models.quantized_klein9b.shaders.sin_f32_57 import SIN_F32_57
from models.quantized_klein9b.shaders.sin_f32_63 import SIN_F32_63
from models.quantized_klein9b.shaders.sin_f32_64 import SIN_F32_64
from models.quantized_klein9b.shaders.sin_f32_70 import SIN_F32_70
from models.quantized_klein9b.shaders.sin_f32_71 import SIN_F32_71
from models.quantized_klein9b.shaders.sin_f32_77 import SIN_F32_77
from models.quantized_klein9b.shaders.sin_f32_78 import SIN_F32_78
from models.quantized_klein9b.shaders.sin_f32_84 import SIN_F32_84
from models.quantized_klein9b.shaders.sin_f32_85 import SIN_F32_85
from models.quantized_klein9b.shaders.stack_4_f32 import STACK_4_F32
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32 import TUPLE_GETITEM_SLICE_F32
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_12 import TUPLE_GETITEM_SLICE_F32_12
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_13 import TUPLE_GETITEM_SLICE_F32_13
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_14 import TUPLE_GETITEM_SLICE_F32_14
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_15 import TUPLE_GETITEM_SLICE_F32_15
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_16 import TUPLE_GETITEM_SLICE_F32_16
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_17 import TUPLE_GETITEM_SLICE_F32_17
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_18 import TUPLE_GETITEM_SLICE_F32_18
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_19 import TUPLE_GETITEM_SLICE_F32_19
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_20 import TUPLE_GETITEM_SLICE_F32_20
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_21 import TUPLE_GETITEM_SLICE_F32_21
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_22 import TUPLE_GETITEM_SLICE_F32_22
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_23 import TUPLE_GETITEM_SLICE_F32_23
from models.quantized_klein9b.tensors.flux_prologue import FluxPrologueTensors
from torch2vk.runtime.session import RuntimeSession


def _run_flux_prologue_with_tensors(rt: RuntimeSession, tensors: FluxPrologueTensors) -> None:
    MUL_SCALAR(rt, x=tensors.timesteps, output=tensors.mul)
    ARANGE_F32(rt, output=tensors.arange)
    MUL_SCALAR_2(rt, x=tensors.arange, output=tensors.mul_1)
    rt.release_frame_workspace(tensors.arange)
    DIV_SCALAR(rt, x=tensors.mul_1, output=tensors.div)
    rt.release_frame_workspace(tensors.mul_1)
    EXP_F32(rt, x=tensors.div, output=tensors.exp)
    rt.release_frame_workspace(tensors.div)
    FLUX_PROLOGUE_MUL_BROADCAST(rt, x=tensors.to, y=tensors.unsqueeze_1, output=tensors.mul_2)
    rt.release_frame_workspace(tensors.exp)
    rt.release_frame_workspace(tensors.mul)
    COS_F32(rt, x=tensors.mul_2, output=tensors.cos)
    SIN_F32(rt, x=tensors.mul_2, output=tensors.sin)
    rt.release_frame_workspace(tensors.mul_2)
    FLUX_PROLOGUE_CAT_2_F32(rt, x0=tensors.cos, x1=tensors.sin, output=tensors.cat)
    rt.release_frame_workspace(tensors.cos)
    rt.release_frame_workspace(tensors.sin)
    LINEAR_NOBIAS_Q8_0_MATVEC_F32_ACT_F32(rt, x=tensors.to_1, weight=tensors.p_time_in_in_layer_weight, output=tensors.linear)
    rt.release_frame_workspace(tensors.cat)
    FLUX_PROLOGUE_SILU_F32(rt, x=tensors.linear, output=tensors.silu)
    rt.release_frame_workspace(tensors.linear)
    LINEAR_NOBIAS_Q8_0_MATVEC_F32_ACT_F32(rt, x=tensors.silu, weight=tensors.p_time_in_out_layer_weight, output=tensors.linear_1)
    rt.release_frame_workspace(tensors.silu)
    FLUX_PROLOGUE_SILU_F32(rt, x=tensors.linear_1, output=tensors.silu_1)
    LINEAR_NOBIAS_Q8_0_MATVEC_F32_ACT_F32(rt, x=tensors.silu_1, weight=tensors.p_double_stream_modulation_img_lin_weight, output=tensors.linear_2)
    rt.release_frame_workspace(tensors.silu_1)
    TUPLE_GETITEM_SLICE_F32(rt, x=tensors.unsqueeze_2, output=tensors.getitem)
    TUPLE_GETITEM_SLICE_F32_12(rt, x=tensors.unsqueeze_2, output=tensors.getitem_1)
    TUPLE_GETITEM_SLICE_F32_13(rt, x=tensors.unsqueeze_2, output=tensors.getitem_2)
    TUPLE_GETITEM_SLICE_F32_14(rt, x=tensors.unsqueeze_2, output=tensors.getitem_3)
    TUPLE_GETITEM_SLICE_F32_15(rt, x=tensors.unsqueeze_2, output=tensors.getitem_4)
    TUPLE_GETITEM_SLICE_F32_16(rt, x=tensors.unsqueeze_2, output=tensors.getitem_5)
    rt.release_frame_workspace(tensors.linear_2)
    FLUX_PROLOGUE_SILU_F32(rt, x=tensors.linear_1, output=tensors.silu_2)
    LINEAR_NOBIAS_Q8_0_MATVEC_F32_ACT_F32(rt, x=tensors.silu_2, weight=tensors.p_double_stream_modulation_txt_lin_weight, output=tensors.linear_3)
    rt.release_frame_workspace(tensors.silu_2)
    TUPLE_GETITEM_SLICE_F32(rt, x=tensors.unsqueeze_3, output=tensors.getitem_6)
    TUPLE_GETITEM_SLICE_F32_17(rt, x=tensors.unsqueeze_3, output=tensors.getitem_7)
    TUPLE_GETITEM_SLICE_F32_18(rt, x=tensors.unsqueeze_3, output=tensors.getitem_8)
    TUPLE_GETITEM_SLICE_F32_19(rt, x=tensors.unsqueeze_3, output=tensors.getitem_9)
    TUPLE_GETITEM_SLICE_F32_20(rt, x=tensors.unsqueeze_3, output=tensors.getitem_10)
    TUPLE_GETITEM_SLICE_F32_21(rt, x=tensors.unsqueeze_3, output=tensors.getitem_11)
    rt.release_frame_workspace(tensors.linear_3)
    FLUX_PROLOGUE_SILU_F32(rt, x=tensors.linear_1, output=tensors.silu_3)
    LINEAR_NOBIAS_Q8_0_MATVEC_F32_ACT_F32(rt, x=tensors.silu_3, weight=tensors.p_single_stream_modulation_lin_weight, output=tensors.linear_4)
    rt.release_frame_workspace(tensors.silu_3)
    TUPLE_GETITEM_SLICE_F32(rt, x=tensors.unsqueeze_4, output=tensors.getitem_12)
    TUPLE_GETITEM_SLICE_F32_22(rt, x=tensors.unsqueeze_4, output=tensors.getitem_13)
    TUPLE_GETITEM_SLICE_F32_23(rt, x=tensors.unsqueeze_4, output=tensors.getitem_14)
    rt.release_frame_workspace(tensors.linear_4)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.x, weight=tensors.p_img_in_weight, output=tensors.linear_5)
    LINEAR_NOBIAS_Q8_0_F32_ACT_F32(rt, x=tensors.ctx, weight=tensors.p_txt_in_weight, output=tensors.linear_6)
    SELECT_INT64(rt, x=tensors.x_ids, output=tensors.select)
    ARANGE_I64(rt, output=tensors.arange_1)
    DIV_SCALAR_27(rt, x=tensors.arange_1, output=tensors.div_1)
    rt.release_frame_workspace(tensors.arange_1)
    POW_BASE_SCALAR_F32(rt, x=tensors.div_1, output=tensors.pow_1)
    rt.release_frame_workspace(tensors.div_1)
    RECIPROCAL_F32(rt, x=tensors.pow_1, output=tensors.reciprocal)
    rt.release_frame_workspace(tensors.pow_1)
    MUL_SCALAR_30(rt, x=tensors.reciprocal, output=tensors.mul_3)
    rt.release_frame_workspace(tensors.reciprocal)
    EINSUM_OUTER_F32(rt, x=tensors.select, y=tensors.mul_3, output=tensors.einsum)
    rt.release_frame_workspace(tensors.mul_3)
    rt.release_frame_workspace(tensors.select)
    COS_F32_32(rt, x=tensors.einsum, output=tensors.cos_1)
    SIN_F32_33(rt, x=tensors.einsum, output=tensors.sin_1)
    FLUX_PROLOGUE_NEG_F32(rt, x=tensors.sin_1, output=tensors.neg)
    rt.release_frame_workspace(tensors.sin_1)
    SIN_F32_35(rt, x=tensors.einsum, output=tensors.sin_2)
    COS_F32_36(rt, x=tensors.einsum, output=tensors.cos_2)
    rt.release_frame_workspace(tensors.einsum)
    STACK_4_F32(rt, x0=tensors.cos_1, x1=tensors.neg, x2=tensors.sin_2, x3=tensors.cos_2, output=tensors.stack)
    rt.release_frame_workspace(tensors.cos_1)
    rt.release_frame_workspace(tensors.cos_2)
    rt.release_frame_workspace(tensors.neg)
    rt.release_frame_workspace(tensors.sin_2)
    SELECT_INT64_38(rt, x=tensors.x_ids, output=tensors.select_1)
    ARANGE_I64(rt, output=tensors.arange_2)
    DIV_SCALAR_39(rt, x=tensors.arange_2, output=tensors.div_2)
    rt.release_frame_workspace(tensors.arange_2)
    POW_BASE_SCALAR_F32(rt, x=tensors.div_2, output=tensors.pow_2)
    rt.release_frame_workspace(tensors.div_2)
    RECIPROCAL_F32(rt, x=tensors.pow_2, output=tensors.reciprocal_1)
    rt.release_frame_workspace(tensors.pow_2)
    MUL_SCALAR_40(rt, x=tensors.reciprocal_1, output=tensors.mul_4)
    rt.release_frame_workspace(tensors.reciprocal_1)
    EINSUM_OUTER_F32(rt, x=tensors.select_1, y=tensors.mul_4, output=tensors.einsum_1)
    rt.release_frame_workspace(tensors.mul_4)
    rt.release_frame_workspace(tensors.select_1)
    COS_F32_41(rt, x=tensors.einsum_1, output=tensors.cos_3)
    SIN_F32_42(rt, x=tensors.einsum_1, output=tensors.sin_3)
    FLUX_PROLOGUE_NEG_F32(rt, x=tensors.sin_3, output=tensors.neg_1)
    rt.release_frame_workspace(tensors.sin_3)
    SIN_F32_43(rt, x=tensors.einsum_1, output=tensors.sin_4)
    COS_F32_44(rt, x=tensors.einsum_1, output=tensors.cos_4)
    rt.release_frame_workspace(tensors.einsum_1)
    STACK_4_F32(rt, x0=tensors.cos_3, x1=tensors.neg_1, x2=tensors.sin_4, x3=tensors.cos_4, output=tensors.stack_1)
    rt.release_frame_workspace(tensors.cos_3)
    rt.release_frame_workspace(tensors.cos_4)
    rt.release_frame_workspace(tensors.neg_1)
    rt.release_frame_workspace(tensors.sin_4)
    SELECT_INT64_45(rt, x=tensors.x_ids, output=tensors.select_2)
    ARANGE_I64(rt, output=tensors.arange_3)
    DIV_SCALAR_46(rt, x=tensors.arange_3, output=tensors.div_3)
    rt.release_frame_workspace(tensors.arange_3)
    POW_BASE_SCALAR_F32(rt, x=tensors.div_3, output=tensors.pow_3)
    rt.release_frame_workspace(tensors.div_3)
    RECIPROCAL_F32(rt, x=tensors.pow_3, output=tensors.reciprocal_2)
    rt.release_frame_workspace(tensors.pow_3)
    MUL_SCALAR_47(rt, x=tensors.reciprocal_2, output=tensors.mul_5)
    rt.release_frame_workspace(tensors.reciprocal_2)
    EINSUM_OUTER_F32(rt, x=tensors.select_2, y=tensors.mul_5, output=tensors.einsum_2)
    rt.release_frame_workspace(tensors.mul_5)
    rt.release_frame_workspace(tensors.select_2)
    COS_F32_48(rt, x=tensors.einsum_2, output=tensors.cos_5)
    SIN_F32_49(rt, x=tensors.einsum_2, output=tensors.sin_5)
    FLUX_PROLOGUE_NEG_F32(rt, x=tensors.sin_5, output=tensors.neg_2)
    rt.release_frame_workspace(tensors.sin_5)
    SIN_F32_50(rt, x=tensors.einsum_2, output=tensors.sin_6)
    COS_F32_51(rt, x=tensors.einsum_2, output=tensors.cos_6)
    rt.release_frame_workspace(tensors.einsum_2)
    STACK_4_F32(rt, x0=tensors.cos_5, x1=tensors.neg_2, x2=tensors.sin_6, x3=tensors.cos_6, output=tensors.stack_2)
    rt.release_frame_workspace(tensors.cos_5)
    rt.release_frame_workspace(tensors.cos_6)
    rt.release_frame_workspace(tensors.neg_2)
    rt.release_frame_workspace(tensors.sin_6)
    SELECT_INT64_52(rt, x=tensors.x_ids, output=tensors.select_3)
    ARANGE_I64(rt, output=tensors.arange_4)
    DIV_SCALAR_53(rt, x=tensors.arange_4, output=tensors.div_4)
    rt.release_frame_workspace(tensors.arange_4)
    POW_BASE_SCALAR_F32(rt, x=tensors.div_4, output=tensors.pow_4)
    rt.release_frame_workspace(tensors.div_4)
    RECIPROCAL_F32(rt, x=tensors.pow_4, output=tensors.reciprocal_3)
    rt.release_frame_workspace(tensors.pow_4)
    MUL_SCALAR_54(rt, x=tensors.reciprocal_3, output=tensors.mul_6)
    rt.release_frame_workspace(tensors.reciprocal_3)
    EINSUM_OUTER_F32(rt, x=tensors.select_3, y=tensors.mul_6, output=tensors.einsum_3)
    rt.release_frame_workspace(tensors.mul_6)
    rt.release_frame_workspace(tensors.select_3)
    COS_F32_55(rt, x=tensors.einsum_3, output=tensors.cos_7)
    SIN_F32_56(rt, x=tensors.einsum_3, output=tensors.sin_7)
    FLUX_PROLOGUE_NEG_F32(rt, x=tensors.sin_7, output=tensors.neg_3)
    rt.release_frame_workspace(tensors.sin_7)
    SIN_F32_57(rt, x=tensors.einsum_3, output=tensors.sin_8)
    COS_F32_58(rt, x=tensors.einsum_3, output=tensors.cos_8)
    rt.release_frame_workspace(tensors.einsum_3)
    STACK_4_F32(rt, x0=tensors.cos_7, x1=tensors.neg_3, x2=tensors.sin_8, x3=tensors.cos_8, output=tensors.stack_3)
    rt.release_frame_workspace(tensors.cos_7)
    rt.release_frame_workspace(tensors.cos_8)
    rt.release_frame_workspace(tensors.neg_3)
    rt.release_frame_workspace(tensors.sin_8)
    CAT_4_F32(rt, x0=tensors.to_2, x1=tensors.to_3, x2=tensors.to_4, x3=tensors.to_5, output=tensors.cat_1)
    rt.release_frame_workspace(tensors.stack)
    rt.release_frame_workspace(tensors.stack_1)
    rt.release_frame_workspace(tensors.stack_2)
    rt.release_frame_workspace(tensors.stack_3)
    SELECT_INT64(rt, x=tensors.ctx_ids, output=tensors.select_4)
    ARANGE_I64(rt, output=tensors.arange_5)
    DIV_SCALAR_60(rt, x=tensors.arange_5, output=tensors.div_5)
    rt.release_frame_workspace(tensors.arange_5)
    POW_BASE_SCALAR_F32(rt, x=tensors.div_5, output=tensors.pow_5)
    rt.release_frame_workspace(tensors.div_5)
    RECIPROCAL_F32(rt, x=tensors.pow_5, output=tensors.reciprocal_4)
    rt.release_frame_workspace(tensors.pow_5)
    MUL_SCALAR_61(rt, x=tensors.reciprocal_4, output=tensors.mul_7)
    rt.release_frame_workspace(tensors.reciprocal_4)
    EINSUM_OUTER_F32(rt, x=tensors.select_4, y=tensors.mul_7, output=tensors.einsum_4)
    rt.release_frame_workspace(tensors.mul_7)
    rt.release_frame_workspace(tensors.select_4)
    COS_F32_62(rt, x=tensors.einsum_4, output=tensors.cos_9)
    SIN_F32_63(rt, x=tensors.einsum_4, output=tensors.sin_9)
    FLUX_PROLOGUE_NEG_F32(rt, x=tensors.sin_9, output=tensors.neg_4)
    rt.release_frame_workspace(tensors.sin_9)
    SIN_F32_64(rt, x=tensors.einsum_4, output=tensors.sin_10)
    COS_F32_65(rt, x=tensors.einsum_4, output=tensors.cos_10)
    rt.release_frame_workspace(tensors.einsum_4)
    STACK_4_F32(rt, x0=tensors.cos_9, x1=tensors.neg_4, x2=tensors.sin_10, x3=tensors.cos_10, output=tensors.stack_4)
    rt.release_frame_workspace(tensors.cos_10)
    rt.release_frame_workspace(tensors.cos_9)
    rt.release_frame_workspace(tensors.neg_4)
    rt.release_frame_workspace(tensors.sin_10)
    SELECT_INT64_66(rt, x=tensors.ctx_ids, output=tensors.select_5)
    ARANGE_I64(rt, output=tensors.arange_6)
    DIV_SCALAR_67(rt, x=tensors.arange_6, output=tensors.div_6)
    rt.release_frame_workspace(tensors.arange_6)
    POW_BASE_SCALAR_F32(rt, x=tensors.div_6, output=tensors.pow_6)
    rt.release_frame_workspace(tensors.div_6)
    RECIPROCAL_F32(rt, x=tensors.pow_6, output=tensors.reciprocal_5)
    rt.release_frame_workspace(tensors.pow_6)
    MUL_SCALAR_68(rt, x=tensors.reciprocal_5, output=tensors.mul_8)
    rt.release_frame_workspace(tensors.reciprocal_5)
    EINSUM_OUTER_F32(rt, x=tensors.select_5, y=tensors.mul_8, output=tensors.einsum_5)
    rt.release_frame_workspace(tensors.mul_8)
    rt.release_frame_workspace(tensors.select_5)
    COS_F32_69(rt, x=tensors.einsum_5, output=tensors.cos_11)
    SIN_F32_70(rt, x=tensors.einsum_5, output=tensors.sin_11)
    FLUX_PROLOGUE_NEG_F32(rt, x=tensors.sin_11, output=tensors.neg_5)
    rt.release_frame_workspace(tensors.sin_11)
    SIN_F32_71(rt, x=tensors.einsum_5, output=tensors.sin_12)
    COS_F32_72(rt, x=tensors.einsum_5, output=tensors.cos_12)
    rt.release_frame_workspace(tensors.einsum_5)
    STACK_4_F32(rt, x0=tensors.cos_11, x1=tensors.neg_5, x2=tensors.sin_12, x3=tensors.cos_12, output=tensors.stack_5)
    rt.release_frame_workspace(tensors.cos_11)
    rt.release_frame_workspace(tensors.cos_12)
    rt.release_frame_workspace(tensors.neg_5)
    rt.release_frame_workspace(tensors.sin_12)
    SELECT_INT64_73(rt, x=tensors.ctx_ids, output=tensors.select_6)
    ARANGE_I64(rt, output=tensors.arange_7)
    DIV_SCALAR_74(rt, x=tensors.arange_7, output=tensors.div_7)
    rt.release_frame_workspace(tensors.arange_7)
    POW_BASE_SCALAR_F32(rt, x=tensors.div_7, output=tensors.pow_7)
    rt.release_frame_workspace(tensors.div_7)
    RECIPROCAL_F32(rt, x=tensors.pow_7, output=tensors.reciprocal_6)
    rt.release_frame_workspace(tensors.pow_7)
    MUL_SCALAR_75(rt, x=tensors.reciprocal_6, output=tensors.mul_9)
    rt.release_frame_workspace(tensors.reciprocal_6)
    EINSUM_OUTER_F32(rt, x=tensors.select_6, y=tensors.mul_9, output=tensors.einsum_6)
    rt.release_frame_workspace(tensors.mul_9)
    rt.release_frame_workspace(tensors.select_6)
    COS_F32_76(rt, x=tensors.einsum_6, output=tensors.cos_13)
    SIN_F32_77(rt, x=tensors.einsum_6, output=tensors.sin_13)
    FLUX_PROLOGUE_NEG_F32(rt, x=tensors.sin_13, output=tensors.neg_6)
    rt.release_frame_workspace(tensors.sin_13)
    SIN_F32_78(rt, x=tensors.einsum_6, output=tensors.sin_14)
    COS_F32_79(rt, x=tensors.einsum_6, output=tensors.cos_14)
    rt.release_frame_workspace(tensors.einsum_6)
    STACK_4_F32(rt, x0=tensors.cos_13, x1=tensors.neg_6, x2=tensors.sin_14, x3=tensors.cos_14, output=tensors.stack_6)
    rt.release_frame_workspace(tensors.cos_13)
    rt.release_frame_workspace(tensors.cos_14)
    rt.release_frame_workspace(tensors.neg_6)
    rt.release_frame_workspace(tensors.sin_14)
    SELECT_INT64_80(rt, x=tensors.ctx_ids, output=tensors.select_7)
    ARANGE_I64(rt, output=tensors.arange_8)
    DIV_SCALAR_81(rt, x=tensors.arange_8, output=tensors.div_8)
    rt.release_frame_workspace(tensors.arange_8)
    POW_BASE_SCALAR_F32(rt, x=tensors.div_8, output=tensors.pow_8)
    rt.release_frame_workspace(tensors.div_8)
    RECIPROCAL_F32(rt, x=tensors.pow_8, output=tensors.reciprocal_7)
    rt.release_frame_workspace(tensors.pow_8)
    MUL_SCALAR_82(rt, x=tensors.reciprocal_7, output=tensors.mul_10)
    rt.release_frame_workspace(tensors.reciprocal_7)
    EINSUM_OUTER_F32(rt, x=tensors.select_7, y=tensors.mul_10, output=tensors.einsum_7)
    rt.release_frame_workspace(tensors.mul_10)
    rt.release_frame_workspace(tensors.select_7)
    COS_F32_83(rt, x=tensors.einsum_7, output=tensors.cos_15)
    SIN_F32_84(rt, x=tensors.einsum_7, output=tensors.sin_15)
    FLUX_PROLOGUE_NEG_F32(rt, x=tensors.sin_15, output=tensors.neg_7)
    rt.release_frame_workspace(tensors.sin_15)
    SIN_F32_85(rt, x=tensors.einsum_7, output=tensors.sin_16)
    COS_F32_86(rt, x=tensors.einsum_7, output=tensors.cos_16)
    rt.release_frame_workspace(tensors.einsum_7)
    STACK_4_F32(rt, x0=tensors.cos_15, x1=tensors.neg_7, x2=tensors.sin_16, x3=tensors.cos_16, output=tensors.stack_7)
    rt.release_frame_workspace(tensors.cos_15)
    rt.release_frame_workspace(tensors.cos_16)
    rt.release_frame_workspace(tensors.neg_7)
    rt.release_frame_workspace(tensors.sin_16)
    CAT_4_F32(rt, x0=tensors.to_6, x1=tensors.to_7, x2=tensors.to_8, x3=tensors.to_9, output=tensors.cat_2)
    rt.release_frame_workspace(tensors.stack_4)
    rt.release_frame_workspace(tensors.stack_5)
    rt.release_frame_workspace(tensors.stack_6)
    rt.release_frame_workspace(tensors.stack_7)


def run_flux_prologue(rt: RuntimeSession) -> None:
    tensors = model_tensors().flux_prologue
    _run_flux_prologue_with_tensors(rt, tensors)
