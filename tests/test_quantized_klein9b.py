"""FLUX.2 Klein 9B Q4_K_M Vulkan integration tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from models.quantized_klein9b.run import main


def test_quantized_klein9b_writes_128px_image(tmp_path: Path) -> None:
    output = tmp_path / "quantized_klein9b.png"
    result = main(
        prompt="a small red sailboat on calm blue water",
        output=output,
        width=128,
        height=128,
        seed=7,
    )

    image_path = Path(result.image_path)
    assert image_path == output.resolve()
    assert image_path.exists()
    assert image_path.stat().st_size > 0
    assert result.width == 128
    assert result.height == 128
    assert result.num_steps == 4

    with Image.open(image_path) as image:
        assert image.size == (128, 128)
        assert image.mode == "RGB"
