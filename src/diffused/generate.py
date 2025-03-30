from diffusers import AutoPipelineForText2Image
from PIL import Image


def generate(
    model: str, prompt: str, width: int | None = None, height: int | None = None
) -> Image.Image:
    """
    Generate image with diffusion model.

    Args:
        model (str): Diffusion model.
        prompt (str): Text prompt.
        width (int): Image width.
        height (int): Image height.

    Returns:
        image (PIL.Image.Image): Pillow image.
    """
    pipeline = AutoPipelineForText2Image.from_pretrained(model)
    return pipeline(prompt=prompt, width=width, height=height).images[0]
