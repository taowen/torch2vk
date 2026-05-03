"""OmniVoice safetensor torch2vk debug integration smoke."""

from __future__ import annotations

import unittest
from pathlib import Path

from torch2vk.models.omnivoice_safetensor.model_directory import resolve_omnivoice_model_dir
from torch2vk.models.omnivoice_safetensor.runtime import run_omnivoice_text_to_audio_tokens_debug
from torch2vk.models.omnivoice_safetensor.spec import load_omnivoice_spec
from torch2vk.models.omnivoice_safetensor.tensors.case import default_omnivoice_debug_case
from torch2vk.models.omnivoice_safetensor.tensors.run import omnivoice_run_tensors

SHADER_DIR = Path("build/shaders/omnivoice_safetensor")


class OmniVoiceSafetensorIntegrationTest(unittest.TestCase):
    def test_run_tree_declares_qwen3_cond_uncond_rows(self) -> None:
        model_dir = resolve_omnivoice_model_dir()
        spec = load_omnivoice_spec(model_dir)
        case = default_omnivoice_debug_case()

        tensors = omnivoice_run_tensors(case=case, spec=spec)
        step = tensors.steps[0]

        self.assertEqual(step.qwen3_cond.row, "cond")
        self.assertEqual(step.qwen3_uncond.row, "uncond")
        self.assertEqual(step.qwen3_cond.prefill.logits.name, "output.logits")
        self.assertEqual(step.qwen3_uncond.prefill.logits.name, "output.logits")
        self.assertEqual(tensors.stage1.waveform.name, "stage1.decoder.waveform")
        self.assertEqual(tensors.wav_pcm16.name, "output.wav_pcm16")

    def test_text_to_stage1_decoder_waveform_uses_torch2vk_debug_runtime(self) -> None:
        model_dir = resolve_omnivoice_model_dir()
        spec = load_omnivoice_spec(model_dir)
        case = default_omnivoice_debug_case()

        self.assertEqual(spec.num_audio_codebook, 8)
        self.assertEqual(case.target_steps, case.num_steps)
        if not SHADER_DIR.exists():
            self.skipTest("OmniVoice shaders are not compiled")

        result = run_omnivoice_text_to_audio_tokens_debug(
            spec=spec,
            case=case,
            model_dir=model_dir,
        )

        self.assertEqual({record.scope for record in result.records}, {"debug/step=0"})
        self.assertEqual(
            [record.shader for record in result.records],
            [
                "omnivoice_audio_embedding_sum_f32",
                "omnivoice_audio_head_mat_vec_f32_f32",
                "omnivoice_codebook_argmax_f32",
                "omnivoice_stage1_quantizer_embed_sum_f32",
                "omnivoice_stage1_quantizer_embed_project_out_sum_f32",
                "omnivoice_stage1_conv1d_k1_f32",
                "omnivoice_stage1_conv1d_k7_f32",
                "omnivoice_stage1_snake_deconv1d_block0_f32",
                "omnivoice_stage1_snake_conv1d_k7_d1_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d3_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d9_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_deconv1d_block1_f32",
                "omnivoice_stage1_snake_conv1d_k7_d1_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d3_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d9_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_deconv1d_block2_f32",
                "omnivoice_stage1_snake_conv1d_k7_d1_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d3_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d9_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_deconv1d_block3_f32",
                "omnivoice_stage1_snake_conv1d_k7_d1_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d3_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d9_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_deconv1d_block4_f32",
                "omnivoice_stage1_snake_conv1d_k7_d1_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d3_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d9_f32",
                "omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
                "omnivoice_stage1_snake_conv1d_k7_d1_f32",
            ],
        )
