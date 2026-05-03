"""Torch2VK-native OmniVoice debug runtime helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from torch.nn import functional

from torch2vk.integration import (
    DebugIntegrationCase,
    first_tensor,
    run_debug_integration_case,
)
from torch2vk.logical import LogicalTensor
from torch2vk.models.omnivoice_safetensor.debug import omnivoice_debug_initial_tensors
from torch2vk.models.omnivoice_safetensor.execution import run_omnivoice_debug_audio_token_step
from torch2vk.models.omnivoice_safetensor.spec import OmniVoiceSpec
from torch2vk.models.omnivoice_safetensor.tensors.case import OmniVoiceDebugCase
from torch2vk.models.omnivoice_safetensor.tensors.stage0 import (
    OmniVoiceStage0Tensors,
    omnivoice_stage0_tensors,
)
from torch2vk.models.omnivoice_safetensor.tensors.stage1 import (
    OmniVoiceStage1Tensors,
    omnivoice_stage1_tensors,
)
from torch2vk.models.omnivoice_safetensor.tensors.weights import (
    OmniVoiceWeights,
    omnivoice_weights,
)
from torch2vk.reference_trace import ReferenceTrace, TraceReferenceProvider
from torch2vk.shader import DispatchRecord
from torch2vk.storage import tensor_nbytes
from torch2vk.vulkan_backend import VulkanBuffer
from torch2vk.vulkan_runner import LogicalTensorLookup, write_bound_tensor_bytes


@dataclass(frozen=True, slots=True)
class OmniVoiceDebugRun:
    records: tuple[DispatchRecord, ...]


def run_omnivoice_text_to_audio_tokens_debug(
    *,
    spec: OmniVoiceSpec,
    case: OmniVoiceDebugCase,
    model_dir: str | Path | None = None,
) -> OmniVoiceDebugRun:
    tensors = omnivoice_stage0_tensors(batch=1, steps=case.target_steps, spec=spec)
    stage1_tensors = omnivoice_stage1_tensors(batch=1, steps=case.target_steps)
    weights = omnivoice_weights(spec)
    fixture = _OmniVoiceDebugFixture.from_case(spec=spec, case=case, model_dir=model_dir)
    result = run_debug_integration_case(
        DebugIntegrationCase(
            shader_dir=Path("build/shaders/omnivoice_safetensor"),
            shader_package="torch2vk.models.omnivoice_safetensor.shaders",
            allocation_id="omnivoice-debug-audio-token",
            tensors=(
                tensors,
                stage1_tensors.quantizer_embed_sum,
                stage1_tensors.project_out_sum_hidden1024,
                stage1_tensors.project_out_sum_hidden256,
                stage1_tensors.decoder_conv1,
                stage1_tensors.decoder_block0_deconv,
                stage1_tensors.decoder_block0_res1_conv1,
                stage1_tensors.decoder_block0_res1_output,
                stage1_tensors.decoder_block0_res2_conv1,
                stage1_tensors.decoder_block0_res2_output,
                stage1_tensors.decoder_block0_res3_conv1,
                stage1_tensors.decoder_block0_res3_output,
                stage1_tensors.decoder_block1_deconv,
                stage1_tensors.decoder_block1_res1_conv1,
                stage1_tensors.decoder_block1_res1_output,
                stage1_tensors.decoder_block1_res2_conv1,
                stage1_tensors.decoder_block1_res2_output,
                stage1_tensors.decoder_block1_res3_conv1,
                stage1_tensors.decoder_block1_res3_output,
                stage1_tensors.decoder_block2_deconv,
                stage1_tensors.decoder_block2_res1_conv1,
                stage1_tensors.decoder_block2_res1_output,
                stage1_tensors.decoder_block2_res2_conv1,
                stage1_tensors.decoder_block2_res2_output,
                stage1_tensors.decoder_block2_res3_conv1,
                stage1_tensors.decoder_block2_res3_output,
                stage1_tensors.decoder_block3_deconv,
                stage1_tensors.decoder_block3_res1_conv1,
                stage1_tensors.decoder_block3_res1_output,
                stage1_tensors.decoder_block3_res2_conv1,
                stage1_tensors.decoder_block3_res2_output,
                stage1_tensors.decoder_block3_res3_conv1,
                stage1_tensors.decoder_block3_res3_output,
                stage1_tensors.decoder_block4_deconv,
                stage1_tensors.decoder_block4_res1_conv1,
                stage1_tensors.decoder_block4_res1_output,
                stage1_tensors.decoder_block4_res2_conv1,
                stage1_tensors.decoder_block4_res2_output,
                stage1_tensors.decoder_block4_res3_conv1,
                stage1_tensors.decoder_block4_res3_output,
                stage1_tensors.waveform,
            ),
            weights=weights,
            initial_tensors=omnivoice_debug_initial_tensors(tensors=tensors, weights=weights),
            inputs={"text": case.text, "target_steps": case.target_steps},
            reference_provider=TraceReferenceProvider(
                capture=lambda _inputs: fixture.reference_trace(scope="debug/step=0"),
                provider_id="omnivoice_safetensor.fixture_trace.v1",
            ),
            run=lambda debug_context: _run_scoped_debug_audio_token_step(
                debug_context=debug_context,
                tensors=tensors,
                stage1_tensors=stage1_tensors,
                weights=weights,
            ),
            write_initial_tensors=lambda lookup, allocations, initial: _write_fixture_tensors(
                lookup,
                allocations,
                initial,
                tensors=tensors,
                weights=weights,
                fixture=fixture,
            ),
            extra_fingerprint={
                "omnivoice_debug_fixture": fixture.fingerprint,
                "seed": case.seed,
            },
        )
    )
    return OmniVoiceDebugRun(records=result.records)


def _run_scoped_debug_audio_token_step(
    *,
    debug_context: Any,
    tensors: OmniVoiceStage0Tensors,
    stage1_tensors: OmniVoiceStage1Tensors,
    weights: OmniVoiceWeights,
) -> None:
    with debug_context.scope("debug", step=0):
        run_omnivoice_debug_audio_token_step(
            debug_context,
            tensors=tensors,
            stage1_tensors=stage1_tensors,
            weights=weights,
        )


@dataclass(frozen=True, slots=True)
class _OmniVoiceDebugFixture:
    audio_ids: torch.Tensor
    audio_embeddings: torch.Tensor
    audio_heads: torch.Tensor
    codebook_offsets: torch.Tensor
    audio_head_hidden: torch.Tensor
    audio_head_logits: torch.Tensor
    stage1_codebook_embeds: tuple[torch.Tensor, ...]
    stage1_project_out_weights: tuple[torch.Tensor, ...]
    stage1_project_out_biases: tuple[torch.Tensor, ...]
    stage1_fc2_weight: torch.Tensor
    stage1_fc2_bias: torch.Tensor
    stage1_decoder_conv1_weight: torch.Tensor
    stage1_decoder_conv1_bias: torch.Tensor
    stage1_decoder_block0_snake_alpha: torch.Tensor
    stage1_decoder_block0_deconv_weight: torch.Tensor
    stage1_decoder_block0_deconv_bias: torch.Tensor
    stage1_decoder_block0_resunit_snake1_alpha: tuple[torch.Tensor, ...]
    stage1_decoder_block0_resunit_conv1_weight: tuple[torch.Tensor, ...]
    stage1_decoder_block0_resunit_conv1_bias: tuple[torch.Tensor, ...]
    stage1_decoder_block0_resunit_snake2_alpha: tuple[torch.Tensor, ...]
    stage1_decoder_block0_resunit_conv2_weight: tuple[torch.Tensor, ...]
    stage1_decoder_block0_resunit_conv2_bias: tuple[torch.Tensor, ...]
    stage1_decoder_block1_snake_alpha: torch.Tensor
    stage1_decoder_block1_deconv_weight: torch.Tensor
    stage1_decoder_block1_deconv_bias: torch.Tensor
    stage1_decoder_block1_resunit_snake1_alpha: tuple[torch.Tensor, ...]
    stage1_decoder_block1_resunit_conv1_weight: tuple[torch.Tensor, ...]
    stage1_decoder_block1_resunit_conv1_bias: tuple[torch.Tensor, ...]
    stage1_decoder_block1_resunit_snake2_alpha: tuple[torch.Tensor, ...]
    stage1_decoder_block1_resunit_conv2_weight: tuple[torch.Tensor, ...]
    stage1_decoder_block1_resunit_conv2_bias: tuple[torch.Tensor, ...]
    stage1_decoder_block2_snake_alpha: torch.Tensor
    stage1_decoder_block2_deconv_weight: torch.Tensor
    stage1_decoder_block2_deconv_bias: torch.Tensor
    stage1_decoder_block2_resunit_snake1_alpha: tuple[torch.Tensor, ...]
    stage1_decoder_block2_resunit_conv1_weight: tuple[torch.Tensor, ...]
    stage1_decoder_block2_resunit_conv1_bias: tuple[torch.Tensor, ...]
    stage1_decoder_block2_resunit_snake2_alpha: tuple[torch.Tensor, ...]
    stage1_decoder_block2_resunit_conv2_weight: tuple[torch.Tensor, ...]
    stage1_decoder_block2_resunit_conv2_bias: tuple[torch.Tensor, ...]
    stage1_decoder_block3_snake_alpha: torch.Tensor
    stage1_decoder_block3_deconv_weight: torch.Tensor
    stage1_decoder_block3_deconv_bias: torch.Tensor
    stage1_decoder_block3_resunit_snake1_alpha: tuple[torch.Tensor, ...]
    stage1_decoder_block3_resunit_conv1_weight: tuple[torch.Tensor, ...]
    stage1_decoder_block3_resunit_conv1_bias: tuple[torch.Tensor, ...]
    stage1_decoder_block3_resunit_snake2_alpha: tuple[torch.Tensor, ...]
    stage1_decoder_block3_resunit_conv2_weight: tuple[torch.Tensor, ...]
    stage1_decoder_block3_resunit_conv2_bias: tuple[torch.Tensor, ...]
    stage1_decoder_block4_snake_alpha: torch.Tensor
    stage1_decoder_block4_deconv_weight: torch.Tensor
    stage1_decoder_block4_deconv_bias: torch.Tensor
    stage1_decoder_block4_resunit_snake1_alpha: tuple[torch.Tensor, ...]
    stage1_decoder_block4_resunit_conv1_weight: tuple[torch.Tensor, ...]
    stage1_decoder_block4_resunit_conv1_bias: tuple[torch.Tensor, ...]
    stage1_decoder_block4_resunit_snake2_alpha: tuple[torch.Tensor, ...]
    stage1_decoder_block4_resunit_conv2_weight: tuple[torch.Tensor, ...]
    stage1_decoder_block4_resunit_conv2_bias: tuple[torch.Tensor, ...]
    stage1_decoder_final_snake_alpha: torch.Tensor
    stage1_decoder_final_conv_weight: torch.Tensor
    stage1_decoder_final_conv_bias: torch.Tensor
    stage1_embed_sum: torch.Tensor
    stage1_project_out_sum: torch.Tensor
    stage1_project_out_hidden256: torch.Tensor
    stage1_decoder_conv1: torch.Tensor
    stage1_decoder_block0_deconv: torch.Tensor
    stage1_decoder_block0_res1_conv1: torch.Tensor
    stage1_decoder_block0_res1_output: torch.Tensor
    stage1_decoder_block0_res2_conv1: torch.Tensor
    stage1_decoder_block0_res2_output: torch.Tensor
    stage1_decoder_block0_res3_conv1: torch.Tensor
    stage1_decoder_block0_res3_output: torch.Tensor
    stage1_decoder_block1_deconv: torch.Tensor
    stage1_decoder_block1_res1_conv1: torch.Tensor
    stage1_decoder_block1_res1_output: torch.Tensor
    stage1_decoder_block1_res2_conv1: torch.Tensor
    stage1_decoder_block1_res2_output: torch.Tensor
    stage1_decoder_block1_res3_conv1: torch.Tensor
    stage1_decoder_block1_res3_output: torch.Tensor
    stage1_decoder_block2_deconv: torch.Tensor
    stage1_decoder_block2_res1_conv1: torch.Tensor
    stage1_decoder_block2_res1_output: torch.Tensor
    stage1_decoder_block2_res2_conv1: torch.Tensor
    stage1_decoder_block2_res2_output: torch.Tensor
    stage1_decoder_block2_res3_conv1: torch.Tensor
    stage1_decoder_block2_res3_output: torch.Tensor
    stage1_decoder_block3_deconv: torch.Tensor
    stage1_decoder_block3_res1_conv1: torch.Tensor
    stage1_decoder_block3_res1_output: torch.Tensor
    stage1_decoder_block3_res2_conv1: torch.Tensor
    stage1_decoder_block3_res2_output: torch.Tensor
    stage1_decoder_block3_res3_conv1: torch.Tensor
    stage1_decoder_block3_res3_output: torch.Tensor
    stage1_decoder_block4_deconv: torch.Tensor
    stage1_decoder_block4_res1_conv1: torch.Tensor
    stage1_decoder_block4_res1_output: torch.Tensor
    stage1_decoder_block4_res2_conv1: torch.Tensor
    stage1_decoder_block4_res2_output: torch.Tensor
    stage1_decoder_block4_res3_conv1: torch.Tensor
    stage1_decoder_block4_res3_output: torch.Tensor
    waveform: torch.Tensor
    embedding_sum: torch.Tensor
    argmax_ids: torch.Tensor
    fingerprint: str

    @classmethod
    def from_case(
        cls,
        *,
        spec: OmniVoiceSpec,
        case: OmniVoiceDebugCase,
        model_dir: str | Path | None = None,
    ) -> _OmniVoiceDebugFixture:
        hidden = spec.qwen3.hidden_size
        steps = case.target_steps
        codebooks = spec.num_audio_codebook
        vocab_per_codebook = spec.audio_vocab_size
        vocab = codebooks * vocab_per_codebook
        seed = int.from_bytes(
            hashlib.sha256(f"{case.text}\0{case.seed}".encode()).digest()[:8],
            "little",
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        audio_ids = torch.randint(
            low=0,
            high=vocab_per_codebook,
            size=(1, codebooks, steps),
            dtype=torch.int32,
            generator=generator,
        )
        codebook_offsets = torch.arange(0, vocab, vocab_per_codebook, dtype=torch.int64)
        (
            audio_embeddings,
            audio_heads,
            stage1_codebook_embeds,
            stage1_project_out_weights,
            stage1_project_out_biases,
            stage1_fc2_weight,
            stage1_fc2_bias,
            stage1_decoder_conv1_weight,
            stage1_decoder_conv1_bias,
            stage1_decoder_block0_snake_alpha,
            stage1_decoder_block0_deconv_weight,
            stage1_decoder_block0_deconv_bias,
            stage1_decoder_block0_resunit_snake1_alpha,
            stage1_decoder_block0_resunit_conv1_weight,
            stage1_decoder_block0_resunit_conv1_bias,
            stage1_decoder_block0_resunit_snake2_alpha,
            stage1_decoder_block0_resunit_conv2_weight,
            stage1_decoder_block0_resunit_conv2_bias,
            stage1_decoder_block1_snake_alpha,
            stage1_decoder_block1_deconv_weight,
            stage1_decoder_block1_deconv_bias,
            stage1_decoder_block1_resunit_snake1_alpha,
            stage1_decoder_block1_resunit_conv1_weight,
            stage1_decoder_block1_resunit_conv1_bias,
            stage1_decoder_block1_resunit_snake2_alpha,
            stage1_decoder_block1_resunit_conv2_weight,
            stage1_decoder_block1_resunit_conv2_bias,
            stage1_decoder_block2_snake_alpha,
            stage1_decoder_block2_deconv_weight,
            stage1_decoder_block2_deconv_bias,
            stage1_decoder_block2_resunit_snake1_alpha,
            stage1_decoder_block2_resunit_conv1_weight,
            stage1_decoder_block2_resunit_conv1_bias,
            stage1_decoder_block2_resunit_snake2_alpha,
            stage1_decoder_block2_resunit_conv2_weight,
            stage1_decoder_block2_resunit_conv2_bias,
            stage1_decoder_block3_snake_alpha,
            stage1_decoder_block3_deconv_weight,
            stage1_decoder_block3_deconv_bias,
            stage1_decoder_block3_resunit_snake1_alpha,
            stage1_decoder_block3_resunit_conv1_weight,
            stage1_decoder_block3_resunit_conv1_bias,
            stage1_decoder_block3_resunit_snake2_alpha,
            stage1_decoder_block3_resunit_conv2_weight,
            stage1_decoder_block3_resunit_conv2_bias,
            stage1_decoder_block4_snake_alpha,
            stage1_decoder_block4_deconv_weight,
            stage1_decoder_block4_deconv_bias,
            stage1_decoder_block4_resunit_snake1_alpha,
            stage1_decoder_block4_resunit_conv1_weight,
            stage1_decoder_block4_resunit_conv1_bias,
            stage1_decoder_block4_resunit_snake2_alpha,
            stage1_decoder_block4_resunit_conv2_weight,
            stage1_decoder_block4_resunit_conv2_bias,
            stage1_decoder_final_snake_alpha,
            stage1_decoder_final_conv_weight,
            stage1_decoder_final_conv_bias,
        ) = _load_float_weights(
            model_dir,
            vocab=vocab,
            hidden=hidden,
        )
        audio_head_hidden = torch.randn(
            1,
            steps,
            hidden,
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.01)
        logits = torch.matmul(audio_head_hidden, audio_heads.T.contiguous())
        embedding_sum = _reference_embedding_sum(audio_ids, codebook_offsets, audio_embeddings)
        argmax_ids = _reference_codebook_argmax(logits, codebook_offsets)
        stage1_embed_sum = _reference_stage1_embed_sum(argmax_ids, stage1_codebook_embeds)
        stage1_project_out_sum = _reference_stage1_project_out_sum(
            argmax_ids,
            stage1_codebook_embeds,
            stage1_project_out_weights,
            stage1_project_out_biases,
        )
        stage1_project_out_hidden256 = torch.matmul(
            stage1_project_out_sum,
            stage1_fc2_weight.T.contiguous(),
        )
        stage1_project_out_hidden256 += stage1_fc2_bias
        stage1_decoder_conv1 = _reference_conv1d_k7(
            stage1_project_out_hidden256,
            stage1_decoder_conv1_weight,
            stage1_decoder_conv1_bias,
        )
        stage1_decoder_block0_deconv = _reference_snake_deconv_block0(
            stage1_decoder_conv1,
            stage1_decoder_block0_snake_alpha,
            stage1_decoder_block0_deconv_weight,
            stage1_decoder_block0_deconv_bias,
        )
        (
            stage1_decoder_block0_res1_conv1,
            stage1_decoder_block0_res1_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block0_deconv,
            dilation=1,
            snake1_alpha=stage1_decoder_block0_resunit_snake1_alpha[0],
            conv1_weight=stage1_decoder_block0_resunit_conv1_weight[0],
            conv1_bias=stage1_decoder_block0_resunit_conv1_bias[0],
            snake2_alpha=stage1_decoder_block0_resunit_snake2_alpha[0],
            conv2_weight=stage1_decoder_block0_resunit_conv2_weight[0],
            conv2_bias=stage1_decoder_block0_resunit_conv2_bias[0],
        )
        (
            stage1_decoder_block0_res2_conv1,
            stage1_decoder_block0_res2_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block0_res1_output,
            dilation=3,
            snake1_alpha=stage1_decoder_block0_resunit_snake1_alpha[1],
            conv1_weight=stage1_decoder_block0_resunit_conv1_weight[1],
            conv1_bias=stage1_decoder_block0_resunit_conv1_bias[1],
            snake2_alpha=stage1_decoder_block0_resunit_snake2_alpha[1],
            conv2_weight=stage1_decoder_block0_resunit_conv2_weight[1],
            conv2_bias=stage1_decoder_block0_resunit_conv2_bias[1],
        )
        (
            stage1_decoder_block0_res3_conv1,
            stage1_decoder_block0_res3_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block0_res2_output,
            dilation=9,
            snake1_alpha=stage1_decoder_block0_resunit_snake1_alpha[2],
            conv1_weight=stage1_decoder_block0_resunit_conv1_weight[2],
            conv1_bias=stage1_decoder_block0_resunit_conv1_bias[2],
            snake2_alpha=stage1_decoder_block0_resunit_snake2_alpha[2],
            conv2_weight=stage1_decoder_block0_resunit_conv2_weight[2],
            conv2_bias=stage1_decoder_block0_resunit_conv2_bias[2],
        )
        stage1_decoder_block1_deconv = _reference_snake_deconv(
            stage1_decoder_block0_res3_output,
            stage1_decoder_block1_snake_alpha,
            stage1_decoder_block1_deconv_weight,
            stage1_decoder_block1_deconv_bias,
            stride=5,
            padding=3,
            output_padding=1,
        )
        (
            stage1_decoder_block1_res1_conv1,
            stage1_decoder_block1_res1_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block1_deconv,
            dilation=1,
            snake1_alpha=stage1_decoder_block1_resunit_snake1_alpha[0],
            conv1_weight=stage1_decoder_block1_resunit_conv1_weight[0],
            conv1_bias=stage1_decoder_block1_resunit_conv1_bias[0],
            snake2_alpha=stage1_decoder_block1_resunit_snake2_alpha[0],
            conv2_weight=stage1_decoder_block1_resunit_conv2_weight[0],
            conv2_bias=stage1_decoder_block1_resunit_conv2_bias[0],
        )
        (
            stage1_decoder_block1_res2_conv1,
            stage1_decoder_block1_res2_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block1_res1_output,
            dilation=3,
            snake1_alpha=stage1_decoder_block1_resunit_snake1_alpha[1],
            conv1_weight=stage1_decoder_block1_resunit_conv1_weight[1],
            conv1_bias=stage1_decoder_block1_resunit_conv1_bias[1],
            snake2_alpha=stage1_decoder_block1_resunit_snake2_alpha[1],
            conv2_weight=stage1_decoder_block1_resunit_conv2_weight[1],
            conv2_bias=stage1_decoder_block1_resunit_conv2_bias[1],
        )
        (
            stage1_decoder_block1_res3_conv1,
            stage1_decoder_block1_res3_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block1_res2_output,
            dilation=9,
            snake1_alpha=stage1_decoder_block1_resunit_snake1_alpha[2],
            conv1_weight=stage1_decoder_block1_resunit_conv1_weight[2],
            conv1_bias=stage1_decoder_block1_resunit_conv1_bias[2],
            snake2_alpha=stage1_decoder_block1_resunit_snake2_alpha[2],
            conv2_weight=stage1_decoder_block1_resunit_conv2_weight[2],
            conv2_bias=stage1_decoder_block1_resunit_conv2_bias[2],
        )
        stage1_decoder_block2_deconv = _reference_snake_deconv(
            stage1_decoder_block1_res3_output,
            stage1_decoder_block2_snake_alpha,
            stage1_decoder_block2_deconv_weight,
            stage1_decoder_block2_deconv_bias,
            stride=4,
            padding=2,
        )
        (
            stage1_decoder_block2_res1_conv1,
            stage1_decoder_block2_res1_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block2_deconv,
            dilation=1,
            snake1_alpha=stage1_decoder_block2_resunit_snake1_alpha[0],
            conv1_weight=stage1_decoder_block2_resunit_conv1_weight[0],
            conv1_bias=stage1_decoder_block2_resunit_conv1_bias[0],
            snake2_alpha=stage1_decoder_block2_resunit_snake2_alpha[0],
            conv2_weight=stage1_decoder_block2_resunit_conv2_weight[0],
            conv2_bias=stage1_decoder_block2_resunit_conv2_bias[0],
        )
        (
            stage1_decoder_block2_res2_conv1,
            stage1_decoder_block2_res2_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block2_res1_output,
            dilation=3,
            snake1_alpha=stage1_decoder_block2_resunit_snake1_alpha[1],
            conv1_weight=stage1_decoder_block2_resunit_conv1_weight[1],
            conv1_bias=stage1_decoder_block2_resunit_conv1_bias[1],
            snake2_alpha=stage1_decoder_block2_resunit_snake2_alpha[1],
            conv2_weight=stage1_decoder_block2_resunit_conv2_weight[1],
            conv2_bias=stage1_decoder_block2_resunit_conv2_bias[1],
        )
        (
            stage1_decoder_block2_res3_conv1,
            stage1_decoder_block2_res3_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block2_res2_output,
            dilation=9,
            snake1_alpha=stage1_decoder_block2_resunit_snake1_alpha[2],
            conv1_weight=stage1_decoder_block2_resunit_conv1_weight[2],
            conv1_bias=stage1_decoder_block2_resunit_conv1_bias[2],
            snake2_alpha=stage1_decoder_block2_resunit_snake2_alpha[2],
            conv2_weight=stage1_decoder_block2_resunit_conv2_weight[2],
            conv2_bias=stage1_decoder_block2_resunit_conv2_bias[2],
        )
        stage1_decoder_block3_deconv = _reference_snake_deconv(
            stage1_decoder_block2_res3_output,
            stage1_decoder_block3_snake_alpha,
            stage1_decoder_block3_deconv_weight,
            stage1_decoder_block3_deconv_bias,
            stride=2,
            padding=1,
        )
        (
            stage1_decoder_block3_res1_conv1,
            stage1_decoder_block3_res1_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block3_deconv,
            dilation=1,
            snake1_alpha=stage1_decoder_block3_resunit_snake1_alpha[0],
            conv1_weight=stage1_decoder_block3_resunit_conv1_weight[0],
            conv1_bias=stage1_decoder_block3_resunit_conv1_bias[0],
            snake2_alpha=stage1_decoder_block3_resunit_snake2_alpha[0],
            conv2_weight=stage1_decoder_block3_resunit_conv2_weight[0],
            conv2_bias=stage1_decoder_block3_resunit_conv2_bias[0],
        )
        (
            stage1_decoder_block3_res2_conv1,
            stage1_decoder_block3_res2_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block3_res1_output,
            dilation=3,
            snake1_alpha=stage1_decoder_block3_resunit_snake1_alpha[1],
            conv1_weight=stage1_decoder_block3_resunit_conv1_weight[1],
            conv1_bias=stage1_decoder_block3_resunit_conv1_bias[1],
            snake2_alpha=stage1_decoder_block3_resunit_snake2_alpha[1],
            conv2_weight=stage1_decoder_block3_resunit_conv2_weight[1],
            conv2_bias=stage1_decoder_block3_resunit_conv2_bias[1],
        )
        (
            stage1_decoder_block3_res3_conv1,
            stage1_decoder_block3_res3_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block3_res2_output,
            dilation=9,
            snake1_alpha=stage1_decoder_block3_resunit_snake1_alpha[2],
            conv1_weight=stage1_decoder_block3_resunit_conv1_weight[2],
            conv1_bias=stage1_decoder_block3_resunit_conv1_bias[2],
            snake2_alpha=stage1_decoder_block3_resunit_snake2_alpha[2],
            conv2_weight=stage1_decoder_block3_resunit_conv2_weight[2],
            conv2_bias=stage1_decoder_block3_resunit_conv2_bias[2],
        )
        stage1_decoder_block4_deconv = _reference_snake_deconv(
            stage1_decoder_block3_res3_output,
            stage1_decoder_block4_snake_alpha,
            stage1_decoder_block4_deconv_weight,
            stage1_decoder_block4_deconv_bias,
            stride=3,
            padding=2,
            output_padding=1,
        )
        (
            stage1_decoder_block4_res1_conv1,
            stage1_decoder_block4_res1_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block4_deconv,
            dilation=1,
            snake1_alpha=stage1_decoder_block4_resunit_snake1_alpha[0],
            conv1_weight=stage1_decoder_block4_resunit_conv1_weight[0],
            conv1_bias=stage1_decoder_block4_resunit_conv1_bias[0],
            snake2_alpha=stage1_decoder_block4_resunit_snake2_alpha[0],
            conv2_weight=stage1_decoder_block4_resunit_conv2_weight[0],
            conv2_bias=stage1_decoder_block4_resunit_conv2_bias[0],
        )
        (
            stage1_decoder_block4_res2_conv1,
            stage1_decoder_block4_res2_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block4_res1_output,
            dilation=3,
            snake1_alpha=stage1_decoder_block4_resunit_snake1_alpha[1],
            conv1_weight=stage1_decoder_block4_resunit_conv1_weight[1],
            conv1_bias=stage1_decoder_block4_resunit_conv1_bias[1],
            snake2_alpha=stage1_decoder_block4_resunit_snake2_alpha[1],
            conv2_weight=stage1_decoder_block4_resunit_conv2_weight[1],
            conv2_bias=stage1_decoder_block4_resunit_conv2_bias[1],
        )
        (
            stage1_decoder_block4_res3_conv1,
            stage1_decoder_block4_res3_output,
        ) = _reference_block0_resunit(
            stage1_decoder_block4_res2_output,
            dilation=9,
            snake1_alpha=stage1_decoder_block4_resunit_snake1_alpha[2],
            conv1_weight=stage1_decoder_block4_resunit_conv1_weight[2],
            conv1_bias=stage1_decoder_block4_resunit_conv1_bias[2],
            snake2_alpha=stage1_decoder_block4_resunit_snake2_alpha[2],
            conv2_weight=stage1_decoder_block4_resunit_conv2_weight[2],
            conv2_bias=stage1_decoder_block4_resunit_conv2_bias[2],
        )
        waveform = _reference_snake_conv1d_k7(
            stage1_decoder_block4_res3_output,
            alpha=stage1_decoder_final_snake_alpha,
            weight=stage1_decoder_final_conv_weight,
            bias=stage1_decoder_final_conv_bias,
        )
        fingerprint = hashlib.sha256(
            b"".join(
                (
                    audio_ids.numpy().tobytes(),
                    codebook_offsets.numpy().tobytes(),
                    audio_embeddings.numpy().tobytes(),
                    audio_heads.numpy().tobytes(),
                    b"".join(embed.numpy().tobytes() for embed in stage1_codebook_embeds),
                    b"".join(weight.numpy().tobytes() for weight in stage1_project_out_weights),
                    b"".join(bias.numpy().tobytes() for bias in stage1_project_out_biases),
                    stage1_fc2_weight.numpy().tobytes(),
                    stage1_fc2_bias.numpy().tobytes(),
                    stage1_decoder_conv1_weight.numpy().tobytes(),
                    stage1_decoder_conv1_bias.numpy().tobytes(),
                    stage1_decoder_block0_snake_alpha.numpy().tobytes(),
                    stage1_decoder_block0_deconv_weight.numpy().tobytes(),
                    stage1_decoder_block0_deconv_bias.numpy().tobytes(),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block0_resunit_snake1_alpha
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block0_resunit_conv1_weight
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block0_resunit_conv1_bias
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block0_resunit_snake2_alpha
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block0_resunit_conv2_weight
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block0_resunit_conv2_bias
                    ),
                    stage1_decoder_block1_snake_alpha.numpy().tobytes(),
                    stage1_decoder_block1_deconv_weight.numpy().tobytes(),
                    stage1_decoder_block1_deconv_bias.numpy().tobytes(),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block1_resunit_snake1_alpha
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block1_resunit_conv1_weight
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block1_resunit_conv1_bias
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block1_resunit_snake2_alpha
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block1_resunit_conv2_weight
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block1_resunit_conv2_bias
                    ),
                    stage1_decoder_block2_snake_alpha.numpy().tobytes(),
                    stage1_decoder_block2_deconv_weight.numpy().tobytes(),
                    stage1_decoder_block2_deconv_bias.numpy().tobytes(),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block2_resunit_snake1_alpha
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block2_resunit_conv1_weight
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block2_resunit_conv1_bias
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block2_resunit_snake2_alpha
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block2_resunit_conv2_weight
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block2_resunit_conv2_bias
                    ),
                    stage1_decoder_block3_snake_alpha.numpy().tobytes(),
                    stage1_decoder_block3_deconv_weight.numpy().tobytes(),
                    stage1_decoder_block3_deconv_bias.numpy().tobytes(),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block3_resunit_snake1_alpha
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block3_resunit_conv1_weight
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block3_resunit_conv1_bias
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block3_resunit_snake2_alpha
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block3_resunit_conv2_weight
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block3_resunit_conv2_bias
                    ),
                    stage1_decoder_block4_snake_alpha.numpy().tobytes(),
                    stage1_decoder_block4_deconv_weight.numpy().tobytes(),
                    stage1_decoder_block4_deconv_bias.numpy().tobytes(),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block4_resunit_snake1_alpha
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block4_resunit_conv1_weight
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block4_resunit_conv1_bias
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block4_resunit_snake2_alpha
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block4_resunit_conv2_weight
                    ),
                    b"".join(
                        tensor.numpy().tobytes()
                        for tensor in stage1_decoder_block4_resunit_conv2_bias
                    ),
                    stage1_decoder_final_snake_alpha.numpy().tobytes(),
                    stage1_decoder_final_conv_weight.numpy().tobytes(),
                    stage1_decoder_final_conv_bias.numpy().tobytes(),
                    audio_head_hidden.numpy().tobytes(),
                    logits.numpy().tobytes(),
                    stage1_embed_sum.numpy().tobytes(),
                    stage1_project_out_sum.numpy().tobytes(),
                    stage1_project_out_hidden256.numpy().tobytes(),
                    stage1_decoder_conv1.numpy().tobytes(),
                    stage1_decoder_block0_deconv.numpy().tobytes(),
                    stage1_decoder_block0_res1_conv1.numpy().tobytes(),
                    stage1_decoder_block0_res1_output.numpy().tobytes(),
                    stage1_decoder_block0_res2_conv1.numpy().tobytes(),
                    stage1_decoder_block0_res2_output.numpy().tobytes(),
                    stage1_decoder_block0_res3_conv1.numpy().tobytes(),
                    stage1_decoder_block0_res3_output.numpy().tobytes(),
                    stage1_decoder_block1_deconv.numpy().tobytes(),
                    stage1_decoder_block1_res1_conv1.numpy().tobytes(),
                    stage1_decoder_block1_res1_output.numpy().tobytes(),
                    stage1_decoder_block1_res2_conv1.numpy().tobytes(),
                    stage1_decoder_block1_res2_output.numpy().tobytes(),
                    stage1_decoder_block1_res3_conv1.numpy().tobytes(),
                    stage1_decoder_block1_res3_output.numpy().tobytes(),
                    stage1_decoder_block2_deconv.numpy().tobytes(),
                    stage1_decoder_block2_res1_conv1.numpy().tobytes(),
                    stage1_decoder_block2_res1_output.numpy().tobytes(),
                    stage1_decoder_block2_res2_conv1.numpy().tobytes(),
                    stage1_decoder_block2_res2_output.numpy().tobytes(),
                    stage1_decoder_block2_res3_conv1.numpy().tobytes(),
                    stage1_decoder_block2_res3_output.numpy().tobytes(),
                    stage1_decoder_block3_deconv.numpy().tobytes(),
                    stage1_decoder_block3_res1_conv1.numpy().tobytes(),
                    stage1_decoder_block3_res1_output.numpy().tobytes(),
                    stage1_decoder_block3_res2_conv1.numpy().tobytes(),
                    stage1_decoder_block3_res2_output.numpy().tobytes(),
                    stage1_decoder_block3_res3_conv1.numpy().tobytes(),
                    stage1_decoder_block3_res3_output.numpy().tobytes(),
                    stage1_decoder_block4_deconv.numpy().tobytes(),
                    stage1_decoder_block4_res1_conv1.numpy().tobytes(),
                    stage1_decoder_block4_res1_output.numpy().tobytes(),
                    stage1_decoder_block4_res2_conv1.numpy().tobytes(),
                    stage1_decoder_block4_res2_output.numpy().tobytes(),
                    stage1_decoder_block4_res3_conv1.numpy().tobytes(),
                    stage1_decoder_block4_res3_output.numpy().tobytes(),
                    waveform.numpy().tobytes(),
                )
            )
        ).hexdigest()
        return cls(
            audio_ids=audio_ids,
            audio_embeddings=audio_embeddings,
            audio_heads=audio_heads,
            codebook_offsets=codebook_offsets,
            audio_head_hidden=audio_head_hidden,
            audio_head_logits=logits,
            stage1_codebook_embeds=stage1_codebook_embeds,
            stage1_project_out_weights=stage1_project_out_weights,
            stage1_project_out_biases=stage1_project_out_biases,
            stage1_fc2_weight=stage1_fc2_weight,
            stage1_fc2_bias=stage1_fc2_bias,
            stage1_decoder_conv1_weight=stage1_decoder_conv1_weight,
            stage1_decoder_conv1_bias=stage1_decoder_conv1_bias,
            stage1_decoder_block0_snake_alpha=stage1_decoder_block0_snake_alpha,
            stage1_decoder_block0_deconv_weight=stage1_decoder_block0_deconv_weight,
            stage1_decoder_block0_deconv_bias=stage1_decoder_block0_deconv_bias,
            stage1_decoder_block0_resunit_snake1_alpha=(stage1_decoder_block0_resunit_snake1_alpha),
            stage1_decoder_block0_resunit_conv1_weight=(stage1_decoder_block0_resunit_conv1_weight),
            stage1_decoder_block0_resunit_conv1_bias=(stage1_decoder_block0_resunit_conv1_bias),
            stage1_decoder_block0_resunit_snake2_alpha=(stage1_decoder_block0_resunit_snake2_alpha),
            stage1_decoder_block0_resunit_conv2_weight=(stage1_decoder_block0_resunit_conv2_weight),
            stage1_decoder_block0_resunit_conv2_bias=(stage1_decoder_block0_resunit_conv2_bias),
            stage1_decoder_block1_snake_alpha=stage1_decoder_block1_snake_alpha,
            stage1_decoder_block1_deconv_weight=stage1_decoder_block1_deconv_weight,
            stage1_decoder_block1_deconv_bias=stage1_decoder_block1_deconv_bias,
            stage1_decoder_block1_resunit_snake1_alpha=(stage1_decoder_block1_resunit_snake1_alpha),
            stage1_decoder_block1_resunit_conv1_weight=(stage1_decoder_block1_resunit_conv1_weight),
            stage1_decoder_block1_resunit_conv1_bias=(stage1_decoder_block1_resunit_conv1_bias),
            stage1_decoder_block1_resunit_snake2_alpha=(stage1_decoder_block1_resunit_snake2_alpha),
            stage1_decoder_block1_resunit_conv2_weight=(stage1_decoder_block1_resunit_conv2_weight),
            stage1_decoder_block1_resunit_conv2_bias=(stage1_decoder_block1_resunit_conv2_bias),
            stage1_decoder_block2_snake_alpha=stage1_decoder_block2_snake_alpha,
            stage1_decoder_block2_deconv_weight=stage1_decoder_block2_deconv_weight,
            stage1_decoder_block2_deconv_bias=stage1_decoder_block2_deconv_bias,
            stage1_decoder_block2_resunit_snake1_alpha=(stage1_decoder_block2_resunit_snake1_alpha),
            stage1_decoder_block2_resunit_conv1_weight=(stage1_decoder_block2_resunit_conv1_weight),
            stage1_decoder_block2_resunit_conv1_bias=(stage1_decoder_block2_resunit_conv1_bias),
            stage1_decoder_block2_resunit_snake2_alpha=(stage1_decoder_block2_resunit_snake2_alpha),
            stage1_decoder_block2_resunit_conv2_weight=(stage1_decoder_block2_resunit_conv2_weight),
            stage1_decoder_block2_resunit_conv2_bias=(stage1_decoder_block2_resunit_conv2_bias),
            stage1_decoder_block3_snake_alpha=stage1_decoder_block3_snake_alpha,
            stage1_decoder_block3_deconv_weight=stage1_decoder_block3_deconv_weight,
            stage1_decoder_block3_deconv_bias=stage1_decoder_block3_deconv_bias,
            stage1_decoder_block3_resunit_snake1_alpha=(stage1_decoder_block3_resunit_snake1_alpha),
            stage1_decoder_block3_resunit_conv1_weight=(stage1_decoder_block3_resunit_conv1_weight),
            stage1_decoder_block3_resunit_conv1_bias=(stage1_decoder_block3_resunit_conv1_bias),
            stage1_decoder_block3_resunit_snake2_alpha=(stage1_decoder_block3_resunit_snake2_alpha),
            stage1_decoder_block3_resunit_conv2_weight=(stage1_decoder_block3_resunit_conv2_weight),
            stage1_decoder_block3_resunit_conv2_bias=(stage1_decoder_block3_resunit_conv2_bias),
            stage1_decoder_block4_snake_alpha=stage1_decoder_block4_snake_alpha,
            stage1_decoder_block4_deconv_weight=stage1_decoder_block4_deconv_weight,
            stage1_decoder_block4_deconv_bias=stage1_decoder_block4_deconv_bias,
            stage1_decoder_block4_resunit_snake1_alpha=(stage1_decoder_block4_resunit_snake1_alpha),
            stage1_decoder_block4_resunit_conv1_weight=(stage1_decoder_block4_resunit_conv1_weight),
            stage1_decoder_block4_resunit_conv1_bias=(stage1_decoder_block4_resunit_conv1_bias),
            stage1_decoder_block4_resunit_snake2_alpha=(stage1_decoder_block4_resunit_snake2_alpha),
            stage1_decoder_block4_resunit_conv2_weight=(stage1_decoder_block4_resunit_conv2_weight),
            stage1_decoder_block4_resunit_conv2_bias=(stage1_decoder_block4_resunit_conv2_bias),
            stage1_decoder_final_snake_alpha=stage1_decoder_final_snake_alpha,
            stage1_decoder_final_conv_weight=stage1_decoder_final_conv_weight,
            stage1_decoder_final_conv_bias=stage1_decoder_final_conv_bias,
            stage1_embed_sum=stage1_embed_sum,
            stage1_project_out_sum=stage1_project_out_sum,
            stage1_project_out_hidden256=stage1_project_out_hidden256,
            stage1_decoder_conv1=stage1_decoder_conv1,
            stage1_decoder_block0_deconv=stage1_decoder_block0_deconv,
            stage1_decoder_block0_res1_conv1=stage1_decoder_block0_res1_conv1,
            stage1_decoder_block0_res1_output=stage1_decoder_block0_res1_output,
            stage1_decoder_block0_res2_conv1=stage1_decoder_block0_res2_conv1,
            stage1_decoder_block0_res2_output=stage1_decoder_block0_res2_output,
            stage1_decoder_block0_res3_conv1=stage1_decoder_block0_res3_conv1,
            stage1_decoder_block0_res3_output=stage1_decoder_block0_res3_output,
            stage1_decoder_block1_deconv=stage1_decoder_block1_deconv,
            stage1_decoder_block1_res1_conv1=stage1_decoder_block1_res1_conv1,
            stage1_decoder_block1_res1_output=stage1_decoder_block1_res1_output,
            stage1_decoder_block1_res2_conv1=stage1_decoder_block1_res2_conv1,
            stage1_decoder_block1_res2_output=stage1_decoder_block1_res2_output,
            stage1_decoder_block1_res3_conv1=stage1_decoder_block1_res3_conv1,
            stage1_decoder_block1_res3_output=stage1_decoder_block1_res3_output,
            stage1_decoder_block2_deconv=stage1_decoder_block2_deconv,
            stage1_decoder_block2_res1_conv1=stage1_decoder_block2_res1_conv1,
            stage1_decoder_block2_res1_output=stage1_decoder_block2_res1_output,
            stage1_decoder_block2_res2_conv1=stage1_decoder_block2_res2_conv1,
            stage1_decoder_block2_res2_output=stage1_decoder_block2_res2_output,
            stage1_decoder_block2_res3_conv1=stage1_decoder_block2_res3_conv1,
            stage1_decoder_block2_res3_output=stage1_decoder_block2_res3_output,
            stage1_decoder_block3_deconv=stage1_decoder_block3_deconv,
            stage1_decoder_block3_res1_conv1=stage1_decoder_block3_res1_conv1,
            stage1_decoder_block3_res1_output=stage1_decoder_block3_res1_output,
            stage1_decoder_block3_res2_conv1=stage1_decoder_block3_res2_conv1,
            stage1_decoder_block3_res2_output=stage1_decoder_block3_res2_output,
            stage1_decoder_block3_res3_conv1=stage1_decoder_block3_res3_conv1,
            stage1_decoder_block3_res3_output=stage1_decoder_block3_res3_output,
            stage1_decoder_block4_deconv=stage1_decoder_block4_deconv,
            stage1_decoder_block4_res1_conv1=stage1_decoder_block4_res1_conv1,
            stage1_decoder_block4_res1_output=stage1_decoder_block4_res1_output,
            stage1_decoder_block4_res2_conv1=stage1_decoder_block4_res2_conv1,
            stage1_decoder_block4_res2_output=stage1_decoder_block4_res2_output,
            stage1_decoder_block4_res3_conv1=stage1_decoder_block4_res3_conv1,
            stage1_decoder_block4_res3_output=stage1_decoder_block4_res3_output,
            waveform=waveform,
            embedding_sum=embedding_sum,
            argmax_ids=argmax_ids,
            fingerprint=fingerprint,
        )

    def reference_trace(self, *, scope: str = "") -> ReferenceTrace:
        return ReferenceTrace(
            tensors=_scope_artifacts(
                {
                    "stage0.audio_embedding.output": self.embedding_sum,
                    "stage0.audio_head.logits": self.audio_head_logits,
                    "stage1.quantizer.embed_sum": self.stage1_embed_sum,
                    "stage1.quantizer.project_out_sum.hidden1024": self.stage1_project_out_sum,
                    "stage1.quantizer.project_out_sum.hidden256": self.stage1_project_out_hidden256,
                    "stage1.decoder.conv1": self.stage1_decoder_conv1,
                    "stage1.decoder.block0.deconv": self.stage1_decoder_block0_deconv,
                    "stage1.decoder.block0.res_unit1.conv1": self.stage1_decoder_block0_res1_conv1,
                    "stage1.decoder.block0.res_unit1.output": (
                        self.stage1_decoder_block0_res1_output
                    ),
                    "stage1.decoder.block0.res_unit2.conv1": self.stage1_decoder_block0_res2_conv1,
                    "stage1.decoder.block0.res_unit2.output": (
                        self.stage1_decoder_block0_res2_output
                    ),
                    "stage1.decoder.block0.res_unit3.conv1": self.stage1_decoder_block0_res3_conv1,
                    "stage1.decoder.block0.res_unit3.output": (
                        self.stage1_decoder_block0_res3_output
                    ),
                    "stage1.decoder.block1.deconv": self.stage1_decoder_block1_deconv,
                    "stage1.decoder.block1.res_unit1.conv1": self.stage1_decoder_block1_res1_conv1,
                    "stage1.decoder.block1.res_unit1.output": (
                        self.stage1_decoder_block1_res1_output
                    ),
                    "stage1.decoder.block1.res_unit2.conv1": self.stage1_decoder_block1_res2_conv1,
                    "stage1.decoder.block1.res_unit2.output": (
                        self.stage1_decoder_block1_res2_output
                    ),
                    "stage1.decoder.block1.res_unit3.conv1": self.stage1_decoder_block1_res3_conv1,
                    "stage1.decoder.block1.res_unit3.output": (
                        self.stage1_decoder_block1_res3_output
                    ),
                    "stage1.decoder.block2.deconv": self.stage1_decoder_block2_deconv,
                    "stage1.decoder.block2.res_unit1.conv1": self.stage1_decoder_block2_res1_conv1,
                    "stage1.decoder.block2.res_unit1.output": (
                        self.stage1_decoder_block2_res1_output
                    ),
                    "stage1.decoder.block2.res_unit2.conv1": self.stage1_decoder_block2_res2_conv1,
                    "stage1.decoder.block2.res_unit2.output": (
                        self.stage1_decoder_block2_res2_output
                    ),
                    "stage1.decoder.block2.res_unit3.conv1": self.stage1_decoder_block2_res3_conv1,
                    "stage1.decoder.block2.res_unit3.output": (
                        self.stage1_decoder_block2_res3_output
                    ),
                    "stage1.decoder.block3.deconv": self.stage1_decoder_block3_deconv,
                    "stage1.decoder.block3.res_unit1.conv1": self.stage1_decoder_block3_res1_conv1,
                    "stage1.decoder.block3.res_unit1.output": (
                        self.stage1_decoder_block3_res1_output
                    ),
                    "stage1.decoder.block3.res_unit2.conv1": self.stage1_decoder_block3_res2_conv1,
                    "stage1.decoder.block3.res_unit2.output": (
                        self.stage1_decoder_block3_res2_output
                    ),
                    "stage1.decoder.block3.res_unit3.conv1": self.stage1_decoder_block3_res3_conv1,
                    "stage1.decoder.block3.res_unit3.output": (
                        self.stage1_decoder_block3_res3_output
                    ),
                    "stage1.decoder.block4.deconv": self.stage1_decoder_block4_deconv,
                    "stage1.decoder.block4.res_unit1.conv1": self.stage1_decoder_block4_res1_conv1,
                    "stage1.decoder.block4.res_unit1.output": (
                        self.stage1_decoder_block4_res1_output
                    ),
                    "stage1.decoder.block4.res_unit2.conv1": self.stage1_decoder_block4_res2_conv1,
                    "stage1.decoder.block4.res_unit2.output": (
                        self.stage1_decoder_block4_res2_output
                    ),
                    "stage1.decoder.block4.res_unit3.conv1": self.stage1_decoder_block4_res3_conv1,
                    "stage1.decoder.block4.res_unit3.output": (
                        self.stage1_decoder_block4_res3_output
                    ),
                    "stage1.decoder.waveform": self.waveform,
                },
                scope=scope,
            ),
            tokens=_scope_artifacts(
                {
                    "stage0.audio_head.tokens": self.argmax_ids,
                },
                scope=scope,
            ),
            metadata={"source": "omnivoice_debug_fixture", "fingerprint": self.fingerprint},
        )


def _scope_artifacts(
    artifacts: Mapping[str, torch.Tensor],
    *,
    scope: str,
) -> dict[str, torch.Tensor]:
    if not scope:
        return dict(artifacts)
    return {f"{scope}.{name}": value for name, value in artifacts.items()}


def _write_fixture_tensors(
    lookup: LogicalTensorLookup,
    allocations: Mapping[str, VulkanBuffer],
    initial: tuple[LogicalTensor, ...],
    *,
    tensors: OmniVoiceStage0Tensors,
    weights: OmniVoiceWeights,
    fixture: _OmniVoiceDebugFixture,
) -> None:
    payloads = {
        tensors.audio_ids.name: fixture.audio_ids.numpy().tobytes(),
        tensors.audio_head_hidden.name: fixture.audio_head_hidden.numpy().tobytes(),
        weights.stage0.audio_embeddings.name: fixture.audio_embeddings.numpy().tobytes(),
        weights.stage0.audio_heads.name: fixture.audio_heads.numpy().tobytes(),
        weights.stage0.codebook_layer_offsets.name: fixture.codebook_offsets.numpy().tobytes(),
        **{
            weight.name: embed.numpy().tobytes()
            for weight, embed in zip(
                weights.stage1.codebook_embeds,
                fixture.stage1_codebook_embeds,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.project_out_weights,
                fixture.stage1_project_out_weights,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.project_out_biases,
                fixture.stage1_project_out_biases,
                strict=True,
            )
        },
        weights.stage1.fc2_weight.name: fixture.stage1_fc2_weight.numpy().tobytes(),
        weights.stage1.fc2_bias.name: fixture.stage1_fc2_bias.numpy().tobytes(),
        weights.stage1.decoder_conv1_weight.name: (
            fixture.stage1_decoder_conv1_weight.numpy().tobytes()
        ),
        weights.stage1.decoder_conv1_bias.name: fixture.stage1_decoder_conv1_bias.numpy().tobytes(),
        weights.stage1.decoder_block0_snake_alpha.name: (
            fixture.stage1_decoder_block0_snake_alpha.numpy().tobytes()
        ),
        weights.stage1.decoder_block0_deconv_weight.name: (
            fixture.stage1_decoder_block0_deconv_weight.numpy().tobytes()
        ),
        weights.stage1.decoder_block0_deconv_bias.name: (
            fixture.stage1_decoder_block0_deconv_bias.numpy().tobytes()
        ),
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block0_resunit_snake1_alpha,
                fixture.stage1_decoder_block0_resunit_snake1_alpha,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block0_resunit_conv1_weight,
                fixture.stage1_decoder_block0_resunit_conv1_weight,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block0_resunit_conv1_bias,
                fixture.stage1_decoder_block0_resunit_conv1_bias,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block0_resunit_snake2_alpha,
                fixture.stage1_decoder_block0_resunit_snake2_alpha,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block0_resunit_conv2_weight,
                fixture.stage1_decoder_block0_resunit_conv2_weight,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block0_resunit_conv2_bias,
                fixture.stage1_decoder_block0_resunit_conv2_bias,
                strict=True,
            )
        },
        weights.stage1.decoder_block1_snake_alpha.name: (
            fixture.stage1_decoder_block1_snake_alpha.numpy().tobytes()
        ),
        weights.stage1.decoder_block1_deconv_weight.name: (
            fixture.stage1_decoder_block1_deconv_weight.numpy().tobytes()
        ),
        weights.stage1.decoder_block1_deconv_bias.name: (
            fixture.stage1_decoder_block1_deconv_bias.numpy().tobytes()
        ),
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block1_resunit_snake1_alpha,
                fixture.stage1_decoder_block1_resunit_snake1_alpha,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block1_resunit_conv1_weight,
                fixture.stage1_decoder_block1_resunit_conv1_weight,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block1_resunit_conv1_bias,
                fixture.stage1_decoder_block1_resunit_conv1_bias,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block1_resunit_snake2_alpha,
                fixture.stage1_decoder_block1_resunit_snake2_alpha,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block1_resunit_conv2_weight,
                fixture.stage1_decoder_block1_resunit_conv2_weight,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block1_resunit_conv2_bias,
                fixture.stage1_decoder_block1_resunit_conv2_bias,
                strict=True,
            )
        },
        weights.stage1.decoder_block2_snake_alpha.name: (
            fixture.stage1_decoder_block2_snake_alpha.numpy().tobytes()
        ),
        weights.stage1.decoder_block2_deconv_weight.name: (
            fixture.stage1_decoder_block2_deconv_weight.numpy().tobytes()
        ),
        weights.stage1.decoder_block2_deconv_bias.name: (
            fixture.stage1_decoder_block2_deconv_bias.numpy().tobytes()
        ),
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block2_resunit_snake1_alpha,
                fixture.stage1_decoder_block2_resunit_snake1_alpha,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block2_resunit_conv1_weight,
                fixture.stage1_decoder_block2_resunit_conv1_weight,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block2_resunit_conv1_bias,
                fixture.stage1_decoder_block2_resunit_conv1_bias,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block2_resunit_snake2_alpha,
                fixture.stage1_decoder_block2_resunit_snake2_alpha,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block2_resunit_conv2_weight,
                fixture.stage1_decoder_block2_resunit_conv2_weight,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block2_resunit_conv2_bias,
                fixture.stage1_decoder_block2_resunit_conv2_bias,
                strict=True,
            )
        },
        weights.stage1.decoder_block3_snake_alpha.name: (
            fixture.stage1_decoder_block3_snake_alpha.numpy().tobytes()
        ),
        weights.stage1.decoder_block3_deconv_weight.name: (
            fixture.stage1_decoder_block3_deconv_weight.numpy().tobytes()
        ),
        weights.stage1.decoder_block3_deconv_bias.name: (
            fixture.stage1_decoder_block3_deconv_bias.numpy().tobytes()
        ),
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block3_resunit_snake1_alpha,
                fixture.stage1_decoder_block3_resunit_snake1_alpha,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block3_resunit_conv1_weight,
                fixture.stage1_decoder_block3_resunit_conv1_weight,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block3_resunit_conv1_bias,
                fixture.stage1_decoder_block3_resunit_conv1_bias,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block3_resunit_snake2_alpha,
                fixture.stage1_decoder_block3_resunit_snake2_alpha,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block3_resunit_conv2_weight,
                fixture.stage1_decoder_block3_resunit_conv2_weight,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block3_resunit_conv2_bias,
                fixture.stage1_decoder_block3_resunit_conv2_bias,
                strict=True,
            )
        },
        weights.stage1.decoder_block4_snake_alpha.name: (
            fixture.stage1_decoder_block4_snake_alpha.numpy().tobytes()
        ),
        weights.stage1.decoder_block4_deconv_weight.name: (
            fixture.stage1_decoder_block4_deconv_weight.numpy().tobytes()
        ),
        weights.stage1.decoder_block4_deconv_bias.name: (
            fixture.stage1_decoder_block4_deconv_bias.numpy().tobytes()
        ),
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block4_resunit_snake1_alpha,
                fixture.stage1_decoder_block4_resunit_snake1_alpha,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block4_resunit_conv1_weight,
                fixture.stage1_decoder_block4_resunit_conv1_weight,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block4_resunit_conv1_bias,
                fixture.stage1_decoder_block4_resunit_conv1_bias,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block4_resunit_snake2_alpha,
                fixture.stage1_decoder_block4_resunit_snake2_alpha,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block4_resunit_conv2_weight,
                fixture.stage1_decoder_block4_resunit_conv2_weight,
                strict=True,
            )
        },
        **{
            weight.name: tensor.numpy().tobytes()
            for weight, tensor in zip(
                weights.stage1.decoder_block4_resunit_conv2_bias,
                fixture.stage1_decoder_block4_resunit_conv2_bias,
                strict=True,
            )
        },
        weights.stage1.decoder_final_snake_alpha.name: (
            fixture.stage1_decoder_final_snake_alpha.numpy().tobytes()
        ),
        weights.stage1.decoder_final_conv_weight.name: (
            fixture.stage1_decoder_final_conv_weight.numpy().tobytes()
        ),
        weights.stage1.decoder_final_conv_bias.name: (
            fixture.stage1_decoder_final_conv_bias.numpy().tobytes()
        ),
    }
    for tensor in initial:
        bound = first_tensor(lookup[tensor.name])
        payload = payloads.get(tensor.name, bytes(tensor_nbytes(bound)))
        write_bound_tensor_bytes(bound, allocations, payload)


def _load_float_weights(  # noqa: PLR0915
    model_dir: str | Path | None,
    *,
    vocab: int,
    hidden: int,
) -> tuple[Any, ...]:
    if model_dir is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
        return (
            torch.randn(vocab, hidden, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(vocab, hidden, dtype=torch.float32, generator=generator).mul_(0.01),
            tuple(
                torch.randn(1024, 64, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(8)
            ),
            tuple(
                torch.randn(1024, 64, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(8)
            ),
            tuple(
                torch.randn(1024, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(8)
            ),
            torch.randn(256, 1024, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(256, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(1024, 256, 7, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(1024, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(1, 1024, 1, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(1024, 512, 16, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(512, dtype=torch.float32, generator=generator).mul_(0.01),
            tuple(
                torch.randn(1, 512, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(512, 512, 7, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(512, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(1, 512, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(512, 512, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(512, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            torch.randn(1, 512, 1, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(512, 256, 10, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(256, dtype=torch.float32, generator=generator).mul_(0.01),
            tuple(
                torch.randn(1, 256, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(256, 256, 7, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(256, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(1, 256, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(256, 256, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(256, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            torch.randn(1, 256, 1, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(256, 128, 8, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(128, dtype=torch.float32, generator=generator).mul_(0.01),
            tuple(
                torch.randn(1, 128, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(128, 128, 7, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(128, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(1, 128, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(128, 128, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(128, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            torch.randn(1, 128, 1, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(128, 64, 4, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(64, dtype=torch.float32, generator=generator).mul_(0.01),
            tuple(
                torch.randn(1, 64, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(64, 64, 7, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(64, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(1, 64, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(64, 64, 1, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            tuple(
                torch.randn(64, dtype=torch.float32, generator=generator).mul_(0.01)
                for _ in range(3)
            ),
            torch.randn(1, 32, 1, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(1, 32, 7, dtype=torch.float32, generator=generator).mul_(0.01),
            torch.randn(1, dtype=torch.float32, generator=generator).mul_(0.01),
        )
    weights_path = Path(model_dir) / "model.safetensors"
    tokenizer_weights_path = Path(model_dir) / "audio_tokenizer" / "model.safetensors"
    handle = cast("Any", safe_open(weights_path, framework="pt", device="cpu"))
    with handle:
        audio_embeddings = cast(
            "torch.Tensor",
            handle.get_tensor("audio_embeddings.weight").contiguous(),
        )
        audio_heads = cast(
            "torch.Tensor",
            handle.get_tensor("audio_heads.weight").contiguous(),
        )
    tokenizer_handle = cast(
        "Any",
        safe_open(tokenizer_weights_path, framework="pt", device="cpu"),
    )
    with tokenizer_handle:
        stage1_codebook_embeds = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"quantizer.quantizers.{index}.codebook.embed"
                ).contiguous(),
            )
            for index in range(8)
        )
        stage1_project_out_weights = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"quantizer.quantizers.{index}.project_out.weight"
                ).contiguous(),
            )
            for index in range(8)
        )
        stage1_project_out_biases = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"quantizer.quantizers.{index}.project_out.bias"
                ).contiguous(),
            )
            for index in range(8)
        )
        stage1_fc2_weight = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("fc2.weight").contiguous(),
        )
        stage1_fc2_bias = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("fc2.bias").contiguous(),
        )
        stage1_decoder_conv1_weight = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.conv1.weight").contiguous(),
        )
        stage1_decoder_conv1_bias = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.conv1.bias").contiguous(),
        )
        stage1_decoder_block0_snake_alpha = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.0.snake1.alpha").contiguous(),
        )
        stage1_decoder_block0_deconv_weight = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.0.conv_t1.weight").contiguous(),
        )
        stage1_decoder_block0_deconv_bias = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.0.conv_t1.bias").contiguous(),
        )
        stage1_decoder_block0_resunit_snake1_alpha = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.0.res_unit{index}.snake1.alpha"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block0_resunit_conv1_weight = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.0.res_unit{index}.conv1.weight"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block0_resunit_conv1_bias = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.0.res_unit{index}.conv1.bias"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block0_resunit_snake2_alpha = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.0.res_unit{index}.snake2.alpha"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block0_resunit_conv2_weight = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.0.res_unit{index}.conv2.weight"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block0_resunit_conv2_bias = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.0.res_unit{index}.conv2.bias"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block1_snake_alpha = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.1.snake1.alpha").contiguous(),
        )
        stage1_decoder_block1_deconv_weight = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.1.conv_t1.weight").contiguous(),
        )
        stage1_decoder_block1_deconv_bias = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.1.conv_t1.bias").contiguous(),
        )
        stage1_decoder_block1_resunit_snake1_alpha = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.1.res_unit{index}.snake1.alpha"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block1_resunit_conv1_weight = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.1.res_unit{index}.conv1.weight"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block1_resunit_conv1_bias = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.1.res_unit{index}.conv1.bias"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block1_resunit_snake2_alpha = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.1.res_unit{index}.snake2.alpha"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block1_resunit_conv2_weight = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.1.res_unit{index}.conv2.weight"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block1_resunit_conv2_bias = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.1.res_unit{index}.conv2.bias"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block2_snake_alpha = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.2.snake1.alpha").contiguous(),
        )
        stage1_decoder_block2_deconv_weight = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.2.conv_t1.weight").contiguous(),
        )
        stage1_decoder_block2_deconv_bias = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.2.conv_t1.bias").contiguous(),
        )
        stage1_decoder_block2_resunit_snake1_alpha = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.2.res_unit{index}.snake1.alpha"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block2_resunit_conv1_weight = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.2.res_unit{index}.conv1.weight"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block2_resunit_conv1_bias = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.2.res_unit{index}.conv1.bias"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block2_resunit_snake2_alpha = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.2.res_unit{index}.snake2.alpha"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block2_resunit_conv2_weight = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.2.res_unit{index}.conv2.weight"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block2_resunit_conv2_bias = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.2.res_unit{index}.conv2.bias"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block3_snake_alpha = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.3.snake1.alpha").contiguous(),
        )
        stage1_decoder_block3_deconv_weight = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.3.conv_t1.weight").contiguous(),
        )
        stage1_decoder_block3_deconv_bias = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.3.conv_t1.bias").contiguous(),
        )
        stage1_decoder_block3_resunit_snake1_alpha = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.3.res_unit{index}.snake1.alpha"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block3_resunit_conv1_weight = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.3.res_unit{index}.conv1.weight"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block3_resunit_conv1_bias = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.3.res_unit{index}.conv1.bias"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block3_resunit_snake2_alpha = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.3.res_unit{index}.snake2.alpha"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block3_resunit_conv2_weight = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.3.res_unit{index}.conv2.weight"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block3_resunit_conv2_bias = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.3.res_unit{index}.conv2.bias"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block4_snake_alpha = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.4.snake1.alpha").contiguous(),
        )
        stage1_decoder_block4_deconv_weight = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.4.conv_t1.weight").contiguous(),
        )
        stage1_decoder_block4_deconv_bias = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.block.4.conv_t1.bias").contiguous(),
        )
        stage1_decoder_block4_resunit_snake1_alpha = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.4.res_unit{index}.snake1.alpha"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block4_resunit_conv1_weight = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.4.res_unit{index}.conv1.weight"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block4_resunit_conv1_bias = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.4.res_unit{index}.conv1.bias"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block4_resunit_snake2_alpha = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.4.res_unit{index}.snake2.alpha"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block4_resunit_conv2_weight = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.4.res_unit{index}.conv2.weight"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_block4_resunit_conv2_bias = tuple(
            cast(
                "torch.Tensor",
                tokenizer_handle.get_tensor(
                    f"acoustic_decoder.block.4.res_unit{index}.conv2.bias"
                ).contiguous(),
            )
            for index in range(1, 4)
        )
        stage1_decoder_final_snake_alpha = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.snake1.alpha").contiguous(),
        )
        stage1_decoder_final_conv_weight = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.conv2.weight").contiguous(),
        )
        stage1_decoder_final_conv_bias = cast(
            "torch.Tensor",
            tokenizer_handle.get_tensor("acoustic_decoder.conv2.bias").contiguous(),
        )
    _validate_float_tensor(
        "audio_embeddings.weight",
        audio_embeddings,
        expected_shape=(vocab, hidden),
    )
    _validate_float_tensor("audio_heads.weight", audio_heads, expected_shape=(vocab, hidden))
    _validate_float_tensor_sequence(
        "quantizer.quantizers.{index}.codebook.embed",
        stage1_codebook_embeds,
        expected_shape=(1024, 64),
    )
    _validate_float_tensor_sequence(
        "quantizer.quantizers.{index}.project_out.weight",
        stage1_project_out_weights,
        expected_shape=(1024, 64),
    )
    _validate_float_tensor_sequence(
        "quantizer.quantizers.{index}.project_out.bias",
        stage1_project_out_biases,
        expected_shape=(1024,),
    )
    _validate_float_tensor("fc2.weight", stage1_fc2_weight, expected_shape=(256, 1024))
    _validate_float_tensor("fc2.bias", stage1_fc2_bias, expected_shape=(256,))
    _validate_float_tensor(
        "acoustic_decoder.conv1.weight",
        stage1_decoder_conv1_weight,
        expected_shape=(1024, 256, 7),
    )
    _validate_float_tensor(
        "acoustic_decoder.conv1.bias",
        stage1_decoder_conv1_bias,
        expected_shape=(1024,),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.0.snake1.alpha",
        stage1_decoder_block0_snake_alpha,
        expected_shape=(1, 1024, 1),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.0.conv_t1.weight",
        stage1_decoder_block0_deconv_weight,
        expected_shape=(1024, 512, 16),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.0.conv_t1.bias",
        stage1_decoder_block0_deconv_bias,
        expected_shape=(512,),
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.0.res_unit{index}.snake1.alpha",
        stage1_decoder_block0_resunit_snake1_alpha,
        expected_shape=(1, 512, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.0.res_unit{index}.conv1.weight",
        stage1_decoder_block0_resunit_conv1_weight,
        expected_shape=(512, 512, 7),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.0.res_unit{index}.conv1.bias",
        stage1_decoder_block0_resunit_conv1_bias,
        expected_shape=(512,),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.0.res_unit{index}.snake2.alpha",
        stage1_decoder_block0_resunit_snake2_alpha,
        expected_shape=(1, 512, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.0.res_unit{index}.conv2.weight",
        stage1_decoder_block0_resunit_conv2_weight,
        expected_shape=(512, 512, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.0.res_unit{index}.conv2.bias",
        stage1_decoder_block0_resunit_conv2_bias,
        expected_shape=(512,),
        index_base=1,
    )
    _validate_float_tensor(
        "acoustic_decoder.block.1.snake1.alpha",
        stage1_decoder_block1_snake_alpha,
        expected_shape=(1, 512, 1),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.1.conv_t1.weight",
        stage1_decoder_block1_deconv_weight,
        expected_shape=(512, 256, 10),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.1.conv_t1.bias",
        stage1_decoder_block1_deconv_bias,
        expected_shape=(256,),
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.1.res_unit{index}.snake1.alpha",
        stage1_decoder_block1_resunit_snake1_alpha,
        expected_shape=(1, 256, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.1.res_unit{index}.conv1.weight",
        stage1_decoder_block1_resunit_conv1_weight,
        expected_shape=(256, 256, 7),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.1.res_unit{index}.conv1.bias",
        stage1_decoder_block1_resunit_conv1_bias,
        expected_shape=(256,),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.1.res_unit{index}.snake2.alpha",
        stage1_decoder_block1_resunit_snake2_alpha,
        expected_shape=(1, 256, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.1.res_unit{index}.conv2.weight",
        stage1_decoder_block1_resunit_conv2_weight,
        expected_shape=(256, 256, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.1.res_unit{index}.conv2.bias",
        stage1_decoder_block1_resunit_conv2_bias,
        expected_shape=(256,),
        index_base=1,
    )
    _validate_float_tensor(
        "acoustic_decoder.block.2.snake1.alpha",
        stage1_decoder_block2_snake_alpha,
        expected_shape=(1, 256, 1),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.2.conv_t1.weight",
        stage1_decoder_block2_deconv_weight,
        expected_shape=(256, 128, 8),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.2.conv_t1.bias",
        stage1_decoder_block2_deconv_bias,
        expected_shape=(128,),
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.2.res_unit{index}.snake1.alpha",
        stage1_decoder_block2_resunit_snake1_alpha,
        expected_shape=(1, 128, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.2.res_unit{index}.conv1.weight",
        stage1_decoder_block2_resunit_conv1_weight,
        expected_shape=(128, 128, 7),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.2.res_unit{index}.conv1.bias",
        stage1_decoder_block2_resunit_conv1_bias,
        expected_shape=(128,),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.2.res_unit{index}.snake2.alpha",
        stage1_decoder_block2_resunit_snake2_alpha,
        expected_shape=(1, 128, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.2.res_unit{index}.conv2.weight",
        stage1_decoder_block2_resunit_conv2_weight,
        expected_shape=(128, 128, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.2.res_unit{index}.conv2.bias",
        stage1_decoder_block2_resunit_conv2_bias,
        expected_shape=(128,),
        index_base=1,
    )
    _validate_float_tensor(
        "acoustic_decoder.block.3.snake1.alpha",
        stage1_decoder_block3_snake_alpha,
        expected_shape=(1, 128, 1),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.3.conv_t1.weight",
        stage1_decoder_block3_deconv_weight,
        expected_shape=(128, 64, 4),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.3.conv_t1.bias",
        stage1_decoder_block3_deconv_bias,
        expected_shape=(64,),
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.3.res_unit{index}.snake1.alpha",
        stage1_decoder_block3_resunit_snake1_alpha,
        expected_shape=(1, 64, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.3.res_unit{index}.conv1.weight",
        stage1_decoder_block3_resunit_conv1_weight,
        expected_shape=(64, 64, 7),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.3.res_unit{index}.conv1.bias",
        stage1_decoder_block3_resunit_conv1_bias,
        expected_shape=(64,),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.3.res_unit{index}.snake2.alpha",
        stage1_decoder_block3_resunit_snake2_alpha,
        expected_shape=(1, 64, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.3.res_unit{index}.conv2.weight",
        stage1_decoder_block3_resunit_conv2_weight,
        expected_shape=(64, 64, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.3.res_unit{index}.conv2.bias",
        stage1_decoder_block3_resunit_conv2_bias,
        expected_shape=(64,),
        index_base=1,
    )
    _validate_float_tensor(
        "acoustic_decoder.block.4.snake1.alpha",
        stage1_decoder_block4_snake_alpha,
        expected_shape=(1, 64, 1),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.4.conv_t1.weight",
        stage1_decoder_block4_deconv_weight,
        expected_shape=(64, 32, 6),
    )
    _validate_float_tensor(
        "acoustic_decoder.block.4.conv_t1.bias",
        stage1_decoder_block4_deconv_bias,
        expected_shape=(32,),
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.4.res_unit{index}.snake1.alpha",
        stage1_decoder_block4_resunit_snake1_alpha,
        expected_shape=(1, 32, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.4.res_unit{index}.conv1.weight",
        stage1_decoder_block4_resunit_conv1_weight,
        expected_shape=(32, 32, 7),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.4.res_unit{index}.conv1.bias",
        stage1_decoder_block4_resunit_conv1_bias,
        expected_shape=(32,),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.4.res_unit{index}.snake2.alpha",
        stage1_decoder_block4_resunit_snake2_alpha,
        expected_shape=(1, 32, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.4.res_unit{index}.conv2.weight",
        stage1_decoder_block4_resunit_conv2_weight,
        expected_shape=(32, 32, 1),
        index_base=1,
    )
    _validate_float_tensor_sequence(
        "acoustic_decoder.block.4.res_unit{index}.conv2.bias",
        stage1_decoder_block4_resunit_conv2_bias,
        expected_shape=(32,),
        index_base=1,
    )
    _validate_float_tensor(
        "acoustic_decoder.snake1.alpha",
        stage1_decoder_final_snake_alpha,
        expected_shape=(1, 32, 1),
    )
    _validate_float_tensor(
        "acoustic_decoder.conv2.weight",
        stage1_decoder_final_conv_weight,
        expected_shape=(1, 32, 7),
    )
    _validate_float_tensor(
        "acoustic_decoder.conv2.bias",
        stage1_decoder_final_conv_bias,
        expected_shape=(1,),
    )
    return (
        audio_embeddings,
        audio_heads,
        stage1_codebook_embeds,
        stage1_project_out_weights,
        stage1_project_out_biases,
        stage1_fc2_weight,
        stage1_fc2_bias,
        stage1_decoder_conv1_weight,
        stage1_decoder_conv1_bias,
        stage1_decoder_block0_snake_alpha,
        stage1_decoder_block0_deconv_weight,
        stage1_decoder_block0_deconv_bias,
        stage1_decoder_block0_resunit_snake1_alpha,
        stage1_decoder_block0_resunit_conv1_weight,
        stage1_decoder_block0_resunit_conv1_bias,
        stage1_decoder_block0_resunit_snake2_alpha,
        stage1_decoder_block0_resunit_conv2_weight,
        stage1_decoder_block0_resunit_conv2_bias,
        stage1_decoder_block1_snake_alpha,
        stage1_decoder_block1_deconv_weight,
        stage1_decoder_block1_deconv_bias,
        stage1_decoder_block1_resunit_snake1_alpha,
        stage1_decoder_block1_resunit_conv1_weight,
        stage1_decoder_block1_resunit_conv1_bias,
        stage1_decoder_block1_resunit_snake2_alpha,
        stage1_decoder_block1_resunit_conv2_weight,
        stage1_decoder_block1_resunit_conv2_bias,
        stage1_decoder_block2_snake_alpha,
        stage1_decoder_block2_deconv_weight,
        stage1_decoder_block2_deconv_bias,
        stage1_decoder_block2_resunit_snake1_alpha,
        stage1_decoder_block2_resunit_conv1_weight,
        stage1_decoder_block2_resunit_conv1_bias,
        stage1_decoder_block2_resunit_snake2_alpha,
        stage1_decoder_block2_resunit_conv2_weight,
        stage1_decoder_block2_resunit_conv2_bias,
        stage1_decoder_block3_snake_alpha,
        stage1_decoder_block3_deconv_weight,
        stage1_decoder_block3_deconv_bias,
        stage1_decoder_block3_resunit_snake1_alpha,
        stage1_decoder_block3_resunit_conv1_weight,
        stage1_decoder_block3_resunit_conv1_bias,
        stage1_decoder_block3_resunit_snake2_alpha,
        stage1_decoder_block3_resunit_conv2_weight,
        stage1_decoder_block3_resunit_conv2_bias,
        stage1_decoder_block4_snake_alpha,
        stage1_decoder_block4_deconv_weight,
        stage1_decoder_block4_deconv_bias,
        stage1_decoder_block4_resunit_snake1_alpha,
        stage1_decoder_block4_resunit_conv1_weight,
        stage1_decoder_block4_resunit_conv1_bias,
        stage1_decoder_block4_resunit_snake2_alpha,
        stage1_decoder_block4_resunit_conv2_weight,
        stage1_decoder_block4_resunit_conv2_bias,
        stage1_decoder_final_snake_alpha,
        stage1_decoder_final_conv_weight,
        stage1_decoder_final_conv_bias,
    )


def _validate_float_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype is not torch.float32:
        raise ValueError(f"{name} dtype mismatch: expected torch.float32, got {tensor.dtype}")


def _validate_float_tensor_sequence(
    name_template: str,
    tensors: tuple[torch.Tensor, ...],
    *,
    expected_shape: tuple[int, ...],
    index_base: int = 0,
) -> None:
    for index, tensor in enumerate(tensors):
        _validate_float_tensor(
            name_template.format(index=index + index_base),
            tensor,
            expected_shape=expected_shape,
        )


def _reference_embedding_sum(
    audio_ids: torch.Tensor,
    codebook_offsets: torch.Tensor,
    audio_embeddings: torch.Tensor,
) -> torch.Tensor:
    batch, codebooks, steps = audio_ids.shape
    hidden = audio_embeddings.shape[1]
    output = torch.empty((batch, steps, hidden), dtype=torch.float32)
    embeddings = audio_embeddings.float()
    for batch_index in range(batch):
        for step in range(steps):
            value = torch.zeros((hidden,), dtype=torch.float32)
            for codebook in range(codebooks):
                token = int(audio_ids[batch_index, codebook, step])
                value += embeddings[int(codebook_offsets[codebook]) + token]
            output[batch_index, step] = value
    return output


def _reference_codebook_argmax(
    logits: torch.Tensor,
    codebook_offsets: torch.Tensor,
) -> torch.Tensor:
    batch, steps, vocab = logits.shape
    codebooks = int(codebook_offsets.numel())
    output = torch.empty((batch, codebooks, steps), dtype=torch.int32)
    for batch_index in range(batch):
        for codebook in range(codebooks):
            start = int(codebook_offsets[codebook])
            end = int(codebook_offsets[codebook + 1]) if codebook + 1 < codebooks else vocab
            for step in range(steps):
                output[batch_index, codebook, step] = int(
                    torch.argmax(logits[batch_index, step, start:end])
                )
    return output


def _reference_stage1_embed_sum(
    audio_ids: torch.Tensor,
    codebook_embeds: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    batch, codebooks, steps = audio_ids.shape
    output = torch.empty((batch, steps, 64), dtype=torch.float32)
    for batch_index in range(batch):
        for step in range(steps):
            value = torch.zeros((64,), dtype=torch.float32)
            for codebook in range(codebooks):
                token = int(audio_ids[batch_index, codebook, step])
                token = max(0, min(token, codebook_embeds[codebook].shape[0] - 1))
                value += codebook_embeds[codebook][token]
            output[batch_index, step] = value
    return output


def _reference_stage1_project_out_sum(
    audio_ids: torch.Tensor,
    codebook_embeds: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    biases: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    batch, codebooks, steps = audio_ids.shape
    output = torch.empty((batch, steps, 1024), dtype=torch.float32)
    for batch_index in range(batch):
        for step in range(steps):
            value = torch.zeros((1024,), dtype=torch.float32)
            for codebook in range(codebooks):
                token = int(audio_ids[batch_index, codebook, step])
                token = max(0, min(token, codebook_embeds[codebook].shape[0] - 1))
                value += torch.mv(weights[codebook], codebook_embeds[codebook][token])
                value += biases[codebook]
            output[batch_index, step] = value
    return output


def _reference_conv1d_k7(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    y = functional.conv1d(
        x.transpose(1, 2).contiguous(),
        weight,
        bias=bias,
        padding=3,
    )
    return y.transpose(1, 2).contiguous()


def _reference_snake_deconv_block0(
    x: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _reference_snake_deconv(x, alpha, weight, bias, stride=8, padding=4)


def _reference_snake_deconv(
    x: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride: int,
    padding: int,
    output_padding: int = 0,
) -> torch.Tensor:
    a = alpha.reshape(-1).view(1, -1, 1)
    snake_x = x.transpose(1, 2).contiguous()
    snake_x = snake_x + torch.sin(a * snake_x).square() / (a + 1.0e-9)
    y = functional.conv_transpose1d(
        snake_x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    return y.transpose(1, 2).contiguous()


def _reference_block0_resunit(
    x: torch.Tensor,
    *,
    dilation: int,
    snake1_alpha: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    snake2_alpha: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_channels = x.transpose(1, 2).contiguous()
    conv1_input = _reference_snake_channels(x_channels, snake1_alpha)
    conv1 = functional.conv1d(
        conv1_input,
        conv1_weight,
        bias=conv1_bias,
        padding=3 * dilation,
        dilation=dilation,
    )
    conv2_input = _reference_snake_channels(conv1, snake2_alpha)
    conv2 = functional.conv1d(conv2_input, conv2_weight, bias=conv2_bias)
    output = conv2 + x_channels
    return conv1.transpose(1, 2).contiguous(), output.transpose(1, 2).contiguous()


def _reference_snake_conv1d_k7(
    x: torch.Tensor,
    *,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_channels = x.transpose(1, 2).contiguous()
    y = functional.conv1d(
        _reference_snake_channels(x_channels, alpha),
        weight,
        bias=bias,
        padding=3,
    )
    return y.transpose(1, 2).contiguous()


def _reference_snake_channels(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a = alpha.reshape(-1).view(1, -1, 1)
    return x + torch.sin(a * x).square() / (a + 1.0e-9)
