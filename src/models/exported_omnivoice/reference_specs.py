"""Generated PyTorch reference specs."""

from __future__ import annotations

from torch2vk.runtime.reference import ReferenceSpec

AUDIO_HEAD_SPEC = ReferenceSpec(
    program='reference_programs/audio_head.pt2',
    input_bindings={'input': 'audio_head.input'},
    output_bindings={'linear': 'audio_head.linear'},
)

INPUT_EMBED_SPEC = ReferenceSpec(
    program=None,
    input_bindings={'input_ids': 'batch_input_ids', 'audio_mask': 'batch_audio_mask'},
    output_bindings={'hidden_states': 'llm_forward.hidden_states'},
)

LLM_FORWARD_SPEC = ReferenceSpec(
    program=None,
    input_bindings={'hidden_states': 'llm_forward.hidden_states', 'attention_mask': 'attention_mask'},
    output_bindings={'mul_365': 'llm_forward.mul_365'},
)

TOKEN_SCORE_SPEC = ReferenceSpec(
    program=None,
    input_bindings={'logits': 'audio_head.linear', 'tokens': 'tokens', 'audio_mask_id': 'audio_mask_id', 'rng_seed': 'rng_seed', 'step_index': 'step_index'},
    output_bindings={'candidate_tokens': 'candidate_tokens', 'candidate_scores': 'candidate_scores'},
)

TOKEN_UPDATE_SPEC = ReferenceSpec(
    program=None,
    input_bindings={'tokens': 'tokens', 'batch_input_ids': 'batch_input_ids', 'candidate_tokens': 'candidate_tokens', 'candidate_scores': 'candidate_scores', 'unmask_count': 'unmask_count'},
    output_bindings={'tokens': 'tokens', 'batch_input_ids': 'batch_input_ids'},
)
