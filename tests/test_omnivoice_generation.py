"""OmniVoice generation-state helper contracts."""

from __future__ import annotations

import unittest

import torch

from torch2vk.models.omnivoice_safetensor.generation import (
    omnivoice_unmask_schedule,
    select_omnivoice_audio_tokens,
)
from torch2vk.models.omnivoice_safetensor.tensors.case import default_omnivoice_debug_case


class OmniVoiceGenerationTest(unittest.TestCase):
    def test_default_case_carries_generation_parameters(self) -> None:
        case = default_omnivoice_debug_case()

        self.assertEqual(case.guidance_scale, 2.0)
        self.assertEqual(case.layer_penalty_factor, 5.0)
        self.assertEqual(case.t_shift, 0.1)
        self.assertEqual(case.class_temperature, 0.0)

    def test_unmask_schedule_matches_shifted_cumulative_count(self) -> None:
        schedule = omnivoice_unmask_schedule(total=64, num_steps=8, t_shift=0.1)

        self.assertEqual(schedule, (1, 2, 2, 3, 4, 6, 12, 34))
        self.assertEqual(sum(schedule), 64)
        self.assertEqual(len(schedule), 8)

    def test_select_tokens_skips_mask_id_and_filled_positions(self) -> None:
        mask_id = 4
        tokens_before = torch.tensor([[mask_id, 2], [mask_id, mask_id]], dtype=torch.int32)
        cond_logits = torch.zeros((2, 2, 5), dtype=torch.float32)
        uncond_logits = torch.zeros_like(cond_logits)
        cond_logits[0, 0, mask_id] = 100.0
        cond_logits[0, 0, 1] = 120.0
        cond_logits[0, 1, 3] = 99.0
        cond_logits[1, 0, 2] = 8.0
        cond_logits[1, 1, 3] = 9.0

        result = select_omnivoice_audio_tokens(
            tokens_before=tokens_before,
            cond_logits=cond_logits,
            uncond_logits=uncond_logits,
            unmask_count=2,
            mask_id=mask_id,
            guidance_scale=1.0,
            layer_penalty_factor=0.0,
        )

        self.assertEqual(result.pred_tokens[0, 0].item(), 1)
        self.assertEqual(result.update_mask[0, 1].item(), 0)
        self.assertEqual(result.tokens_after[0, 1].item(), 2)
        self.assertEqual(int(result.update_mask.sum().item()), 2)
        self.assertNotEqual(result.tokens_after[0, 0].item(), mask_id)


if __name__ == "__main__":
    unittest.main()
