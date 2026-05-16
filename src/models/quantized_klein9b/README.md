# quantized_klein9b

FLUX.2 Klein 9B text-to-image pipeline wiring:

- Qwen3-8B text encoder produces FLUX text context.
- FLUX denoiser runs through generated torch2vk Vulkan dispatch using Q4_K_M/Q6_K/Q8_0 GGUF weights.
- AutoEncoder decodes the denoised latent to a PNG.
- `export_gguf.py` writes GGUF files for all three weight modules:
  `flux/model.gguf`, `text_encoder/model.gguf`, and `ae/model.gguf`.
- By default the weights are downloaded from ModelScope. Pass local directories
  only when you already have the checkpoints on disk.

```bash
uv run python -m models.quantized_klein9b.export_gguf

uv run python -m models.quantized_klein9b.run \
  --prompt "a quiet mountain lake at sunrise" \
  --output klein9b.png
```
