from diffusers import AutoPipelineForText2Image
from PIL import Image


def generate(
    model: str,
    prompt: str,
    width: int | None = None,
    height: int | None = None,
    device: str | None = None,
    negative_prompt: str | None = None,
    guidance_scale: float | None = None,
) -> Image.Image:
    """
    Generate image with diffusion model.

    Args:
        model (str): Diffusion model.
        prompt (str): Text prompt.
        width (int): Generated image width in pixels.
        height (int): Generated image height in pixels.
        device (str): Device to accelerate computation (cpu, cuda, mps).
        negative_prompt (str): What to exclude from the generated image.
        guidance_scale (float): How much the prompt influences image generation.

    Returns:
        image (PIL.Image.Image): Pillow image.
    """
    pipeline = AutoPipelineForText2Image.from_pretrained(model)
    pipeline.to(device)

    if guidance_scale:
        images = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
        ).images
    else:
        images = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
        ).images

    return images[0]
