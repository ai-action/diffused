from diffusers import AutoPipelineForText2Image
from PIL import Image


def generate(model: str, prompt: str) -> Image.Image:
    """
    Generate image with diffusion model.

    Args:
        model (str): Diffusion model.
        prompt (str): Text prompt.

    Returns:
        image (PIL.Image.Image): Pillow image.
    """
    pipeline = AutoPipelineForText2Image.from_pretrained(model)
    return pipeline(prompt).images[0]
