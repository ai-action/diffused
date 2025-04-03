from typing import NotRequired, TypedDict, Unpack

from diffusers import AutoPipelineForText2Image
from PIL import Image


class Generate(TypedDict):
    model: str
    prompt: str
    width: NotRequired[int]
    height: NotRequired[int]
    device: NotRequired[str]
    negative_prompt: NotRequired[str]
    guidance_scale: NotRequired[float]


def generate(**kwargs: Unpack[Generate]) -> Image.Image:
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
    pipeline = AutoPipelineForText2Image.from_pretrained(kwargs.get("model"))

    device = kwargs.get("device")
    if device:
        pipeline.to(device)

    pipeline_args = {
        "prompt": kwargs.get("prompt"),
        "width": kwargs.get("width"),
        "height": kwargs.get("height"),
        "negative_prompt": kwargs.get("negative_prompt"),
    }

    guidance_scale = kwargs.get("guidance_scale")
    if guidance_scale:
        pipeline_args["guidance_scale"] = guidance_scale

    images = pipeline(**pipeline_args).images
    return images[0]
