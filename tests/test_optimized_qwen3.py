"""Optimized Qwen3 Q4_K_M Vulkan integration coverage."""

from models.optimized_qwen3.run import main


def test_optimized_qwen3_generates_vulkan_text() -> None:
    result = main(max_new_tokens=8)
    assert "Vulkan" in result.text
