"""Generated PyTorch reference specs."""

from __future__ import annotations

from torch2vk.runtime.reference import ReferenceSpec

AUDIO_ENCODER_SPEC = ReferenceSpec(
    program=None,
    tensors='model_tensors().audio_encoder',
    name='spike.audio.encoder',
    policy='tensor',
    input_bindings={'x': 'x', 'position_embedding': 'position_embedding', 'compact_index': 'compact_index', 'attention_mask': 'attention_mask'},
    output_bindings={'linear_110': 'linear_110'},
)

AUDIO_INJECT_SPEC = ReferenceSpec(
    program=None,
    tensors='model_tensors().audio_inject',
    name='spike.text.audio_inject',
    policy='tensor',
    input_bindings={'inputs_embeds': 'index_copy', 'audio_positions': 'audio_positions', 'audio_features': 'audio_features'},
    output_bindings={'embedding': 'index_copy'},
)

DECODE_EMBED_SPEC = ReferenceSpec(
    program='reference_programs/decode_embed.pt2',
    tensors='model_tensors().decode_embed',
    name='spike.decode.{step:04d}.embed',
    policy='tensor',
    input_bindings={'input': 'input'},
    output_bindings={'embedding': 'embedding'},
)

DECODE_LAYER_SPEC = ReferenceSpec(
    program=None,
    tensors='model_tensors().decode_layers[layer_idx]',
    name='spike.decode.{step:04d}.layer.{layer_idx}',
    policy='tensor',
    input_bindings={'hidden_states': 'hidden_states', 'position_embeddings_0': 'position_embeddings_0', 'position_embeddings_1': 'position_embeddings_1', 'cache_position': 'cache_position'},
    output_bindings={'add_7': 'add_7'},
)

DECODE_LM_HEAD_SPEC = ReferenceSpec(
    program='reference_programs/decode_lm_head.pt2',
    tensors='model_tensors().decode_lm_head',
    name='spike.decode.{step:04d}.lm_head',
    policy='tensor',
    input_bindings={'input': 'input'},
    output_bindings={'linear': 'linear'},
)

DECODE_NORM_SPEC = ReferenceSpec(
    program='reference_programs/decode_norm.pt2',
    tensors='model_tensors().decode_norm',
    name='spike.decode.{step:04d}.norm',
    policy='tensor',
    input_bindings={'hidden_states': 'hidden_states'},
    output_bindings={'mul_1': 'mul_1'},
)

EMBED_TOKENS_SPEC = ReferenceSpec(
    program='reference_programs/embed_tokens.pt2',
    tensors='model_tensors().embed_tokens',
    name='spike.text.embed',
    policy='tensor',
    input_bindings={'input': 'input'},
    output_bindings={'embedding': 'embedding'},
)

LM_HEAD_SPEC = ReferenceSpec(
    program='reference_programs/lm_head.pt2',
    tensors='model_tensors().lm_head',
    name='spike.text.lm_head',
    policy='tensor',
    input_bindings={'input': 'input'},
    output_bindings={'linear': 'linear'},
)

TEXT_LAYER_SPEC = ReferenceSpec(
    program=None,
    tensors='model_tensors().text_layers[layer_idx]',
    name='spike.text.layer.{layer_idx}',
    policy='tensor',
    input_bindings={'hidden_states': 'hidden_states', 'position_embeddings_0': 'position_embeddings_0', 'position_embeddings_1': 'position_embeddings_1', 'cache_position': 'cache_position'},
    output_bindings={'add_7': 'add_7'},
)

TEXT_NORM_SPEC = ReferenceSpec(
    program='reference_programs/text_norm.pt2',
    tensors='model_tensors().text_norm',
    name='spike.text.norm',
    policy='tensor',
    input_bindings={'hidden_states': 'hidden_states'},
    output_bindings={'mul_1': 'mul_1'},
)

TOKEN_SELECT_SPEC = ReferenceSpec(
    program=None,
    tensors='model_tensors()',
    name='{name}',
    policy='token',
    input_bindings={'logits': 'decode_lm_head.linear', 'eos_token_ids': 'eos_token_ids'},
    output_bindings={'next_token': 'next_token', 'done': 'done'},
)

TOKEN_STORE_SPEC = ReferenceSpec(
    program=None,
    tensors='model_tensors()',
    name='{name}',
    policy='token',
    input_bindings={'next_token': 'next_token', 'token_index': 'token_index', 'done': 'done'},
    output_bindings={'generated_tokens': 'generated_tokens', 'generated_length': 'generated_length', 'stopped': 'stopped'},
)
