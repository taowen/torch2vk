"""OmniVoice stage1 LogicalTensor tree."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.logical import (
    ComparePolicy,
    LogicalTensor,
    PyTorchProbe,
    activation_tensor,
    input_tensor,
    output_tensor,
)


@dataclass(frozen=True, slots=True)
class OmniVoiceStage1Tensors:
    audio_ids: LogicalTensor
    quantizer_embed_sum: LogicalTensor
    project_out_sum_hidden1024: LogicalTensor
    project_out_sum_hidden256: LogicalTensor
    decoder_conv1: LogicalTensor
    decoder_block0_deconv: LogicalTensor
    decoder_block0_res1_conv1: LogicalTensor
    decoder_block0_res1_output: LogicalTensor
    decoder_block0_res2_conv1: LogicalTensor
    decoder_block0_res2_output: LogicalTensor
    decoder_block0_res3_conv1: LogicalTensor
    decoder_block0_res3_output: LogicalTensor
    decoder_block1_deconv: LogicalTensor
    decoder_block1_res1_conv1: LogicalTensor
    decoder_block1_res1_output: LogicalTensor
    decoder_block1_res2_conv1: LogicalTensor
    decoder_block1_res2_output: LogicalTensor
    decoder_block1_res3_conv1: LogicalTensor
    decoder_block1_res3_output: LogicalTensor
    decoder_block2_deconv: LogicalTensor
    decoder_block2_res1_conv1: LogicalTensor
    decoder_block2_res1_output: LogicalTensor
    decoder_block2_res2_conv1: LogicalTensor
    decoder_block2_res2_output: LogicalTensor
    decoder_block2_res3_conv1: LogicalTensor
    decoder_block2_res3_output: LogicalTensor
    decoder_block3_deconv: LogicalTensor
    decoder_block3_res1_conv1: LogicalTensor
    decoder_block3_res1_output: LogicalTensor
    decoder_block3_res2_conv1: LogicalTensor
    decoder_block3_res2_output: LogicalTensor
    decoder_block3_res3_conv1: LogicalTensor
    decoder_block3_res3_output: LogicalTensor
    decoder_block4_deconv: LogicalTensor
    decoder_block4_res1_conv1: LogicalTensor
    decoder_block4_res1_output: LogicalTensor
    decoder_block4_res2_conv1: LogicalTensor
    decoder_block4_res2_output: LogicalTensor
    decoder_block4_res3_conv1: LogicalTensor
    decoder_block4_res3_output: LogicalTensor
    decoder_hidden256: LogicalTensor
    waveform: LogicalTensor
    project_out_argmax_ids: LogicalTensor
    project_out_argmax_scores: LogicalTensor


def omnivoice_stage1_tensors(
    *,
    batch: int,
    steps: int,
) -> OmniVoiceStage1Tensors:
    return OmniVoiceStage1Tensors(
        audio_ids=input_tensor(
            "stage1.audio_ids",
            dtype="int32",
            shape=(batch, 8, steps),
        ),
        quantizer_embed_sum=activation_tensor(
            "stage1.quantizer.embed_sum",
            dtype="float32",
            shape=(batch, steps, 64),
            pytorch_probe=PyTorchProbe(kind="manual", source="stage1.quantizer.embed_sum"),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        project_out_sum_hidden1024=activation_tensor(
            "stage1.quantizer.project_out_sum.hidden1024",
            dtype="float32",
            shape=(batch, steps, 1024),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.quantizer.project_out_sum.hidden1024",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        project_out_sum_hidden256=activation_tensor(
            "stage1.quantizer.project_out_sum.hidden256",
            dtype="float32",
            shape=(batch, steps, 256),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.quantizer.project_out_sum.hidden256",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_conv1=activation_tensor(
            "stage1.decoder.conv1",
            dtype="float32",
            shape=(batch, steps, 1024),
            pytorch_probe=PyTorchProbe(kind="manual", source="stage1.decoder.conv1"),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block0_deconv=activation_tensor(
            "stage1.decoder.block0.deconv",
            dtype="float32",
            shape=(batch, steps * 8, 512),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block0.deconv",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block0_res1_conv1=activation_tensor(
            "stage1.decoder.block0.res_unit1.conv1",
            dtype="float32",
            shape=(batch, steps * 8, 512),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block0.res_unit1.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block0_res1_output=activation_tensor(
            "stage1.decoder.block0.res_unit1.output",
            dtype="float32",
            shape=(batch, steps * 8, 512),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block0.res_unit1.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block0_res2_conv1=activation_tensor(
            "stage1.decoder.block0.res_unit2.conv1",
            dtype="float32",
            shape=(batch, steps * 8, 512),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block0.res_unit2.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block0_res2_output=activation_tensor(
            "stage1.decoder.block0.res_unit2.output",
            dtype="float32",
            shape=(batch, steps * 8, 512),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block0.res_unit2.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block0_res3_conv1=activation_tensor(
            "stage1.decoder.block0.res_unit3.conv1",
            dtype="float32",
            shape=(batch, steps * 8, 512),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block0.res_unit3.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block0_res3_output=activation_tensor(
            "stage1.decoder.block0.res_unit3.output",
            dtype="float32",
            shape=(batch, steps * 8, 512),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block0.res_unit3.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block1_deconv=activation_tensor(
            "stage1.decoder.block1.deconv",
            dtype="float32",
            shape=(batch, steps * 40, 256),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block1.deconv",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block1_res1_conv1=activation_tensor(
            "stage1.decoder.block1.res_unit1.conv1",
            dtype="float32",
            shape=(batch, steps * 40, 256),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block1.res_unit1.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block1_res1_output=activation_tensor(
            "stage1.decoder.block1.res_unit1.output",
            dtype="float32",
            shape=(batch, steps * 40, 256),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block1.res_unit1.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block1_res2_conv1=activation_tensor(
            "stage1.decoder.block1.res_unit2.conv1",
            dtype="float32",
            shape=(batch, steps * 40, 256),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block1.res_unit2.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block1_res2_output=activation_tensor(
            "stage1.decoder.block1.res_unit2.output",
            dtype="float32",
            shape=(batch, steps * 40, 256),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block1.res_unit2.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block1_res3_conv1=activation_tensor(
            "stage1.decoder.block1.res_unit3.conv1",
            dtype="float32",
            shape=(batch, steps * 40, 256),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block1.res_unit3.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block1_res3_output=activation_tensor(
            "stage1.decoder.block1.res_unit3.output",
            dtype="float32",
            shape=(batch, steps * 40, 256),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block1.res_unit3.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block2_deconv=activation_tensor(
            "stage1.decoder.block2.deconv",
            dtype="float32",
            shape=(batch, steps * 160, 128),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block2.deconv",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block2_res1_conv1=activation_tensor(
            "stage1.decoder.block2.res_unit1.conv1",
            dtype="float32",
            shape=(batch, steps * 160, 128),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block2.res_unit1.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block2_res1_output=activation_tensor(
            "stage1.decoder.block2.res_unit1.output",
            dtype="float32",
            shape=(batch, steps * 160, 128),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block2.res_unit1.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block2_res2_conv1=activation_tensor(
            "stage1.decoder.block2.res_unit2.conv1",
            dtype="float32",
            shape=(batch, steps * 160, 128),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block2.res_unit2.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block2_res2_output=activation_tensor(
            "stage1.decoder.block2.res_unit2.output",
            dtype="float32",
            shape=(batch, steps * 160, 128),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block2.res_unit2.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block2_res3_conv1=activation_tensor(
            "stage1.decoder.block2.res_unit3.conv1",
            dtype="float32",
            shape=(batch, steps * 160, 128),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block2.res_unit3.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block2_res3_output=activation_tensor(
            "stage1.decoder.block2.res_unit3.output",
            dtype="float32",
            shape=(batch, steps * 160, 128),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block2.res_unit3.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block3_deconv=activation_tensor(
            "stage1.decoder.block3.deconv",
            dtype="float32",
            shape=(batch, steps * 320, 64),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block3.deconv",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block3_res1_conv1=activation_tensor(
            "stage1.decoder.block3.res_unit1.conv1",
            dtype="float32",
            shape=(batch, steps * 320, 64),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block3.res_unit1.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block3_res1_output=activation_tensor(
            "stage1.decoder.block3.res_unit1.output",
            dtype="float32",
            shape=(batch, steps * 320, 64),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block3.res_unit1.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block3_res2_conv1=activation_tensor(
            "stage1.decoder.block3.res_unit2.conv1",
            dtype="float32",
            shape=(batch, steps * 320, 64),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block3.res_unit2.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block3_res2_output=activation_tensor(
            "stage1.decoder.block3.res_unit2.output",
            dtype="float32",
            shape=(batch, steps * 320, 64),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block3.res_unit2.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block3_res3_conv1=activation_tensor(
            "stage1.decoder.block3.res_unit3.conv1",
            dtype="float32",
            shape=(batch, steps * 320, 64),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block3.res_unit3.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block3_res3_output=activation_tensor(
            "stage1.decoder.block3.res_unit3.output",
            dtype="float32",
            shape=(batch, steps * 320, 64),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block3.res_unit3.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block4_deconv=activation_tensor(
            "stage1.decoder.block4.deconv",
            dtype="float32",
            shape=(batch, steps * 960, 32),
            pytorch_probe=PyTorchProbe(kind="manual", source="stage1.decoder.block4.deconv"),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block4_res1_conv1=activation_tensor(
            "stage1.decoder.block4.res_unit1.conv1",
            dtype="float32",
            shape=(batch, steps * 960, 32),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block4.res_unit1.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block4_res1_output=activation_tensor(
            "stage1.decoder.block4.res_unit1.output",
            dtype="float32",
            shape=(batch, steps * 960, 32),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block4.res_unit1.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block4_res2_conv1=activation_tensor(
            "stage1.decoder.block4.res_unit2.conv1",
            dtype="float32",
            shape=(batch, steps * 960, 32),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block4.res_unit2.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block4_res2_output=activation_tensor(
            "stage1.decoder.block4.res_unit2.output",
            dtype="float32",
            shape=(batch, steps * 960, 32),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block4.res_unit2.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block4_res3_conv1=activation_tensor(
            "stage1.decoder.block4.res_unit3.conv1",
            dtype="float32",
            shape=(batch, steps * 960, 32),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block4.res_unit3.conv1",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_block4_res3_output=activation_tensor(
            "stage1.decoder.block4.res_unit3.output",
            dtype="float32",
            shape=(batch, steps * 960, 32),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage1.decoder.block4.res_unit3.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        decoder_hidden256=input_tensor(
            "stage1.decoder.hidden256",
            dtype="float32",
            shape=(batch, steps, 256),
        ),
        waveform=output_tensor(
            "stage1.decoder.waveform",
            dtype="float32",
            shape=(batch, steps * 960, 1),
            pytorch_probe=PyTorchProbe(kind="manual", source="stage1.decoder.waveform"),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        project_out_argmax_ids=output_tensor(
            "stage1.quantizer.project_out.argmax_ids",
            dtype="int32",
            shape=(batch, 8, steps),
        ),
        project_out_argmax_scores=output_tensor(
            "stage1.quantizer.project_out.argmax_scores",
            dtype="float32",
            shape=(batch, 8, steps),
        ),
    )
