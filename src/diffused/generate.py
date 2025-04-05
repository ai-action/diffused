from typing import NotRequired, TypedDict, Unpack

import diffusers
from PIL import Image


class Generate(TypedDict):
    model: str
    prompt: str
    image: NotRequired[str]
    mask_image: NotRequired[str]
    width: NotRequired[int]
    height: NotRequired[int]
    device: NotRequired[str]
    negative_prompt: NotRequired[str]
    guidance_scale: NotRequired[float]
    num_inference_steps: NotRequired[int]
    use_safetensors: NotRequired[bool]


def generate(**kwargs: Unpack[Generate]) -> Image.Image:
    """
    Generate image with diffusion model.

    Args:
        model (str): Diffusion model.
        prompt (str): Text prompt.
        image (str): Input image path or URL.
        mask_image (str): Mask image path or URL.
        width (int): Generated image width in pixels.
        height (int): Generated image height in pixels.
        device (str): Device to accelerate computation (cpu, cuda, mps).
        negative_prompt (str): What to exclude from the generated image.
        guidance_scale (float): How much the prompt influences image generation.
        num_inference_steps (int): Number of diffusion steps used for generation.
        use_safetensors (bool): Whether to load safetensors.

    Returns:
        image (PIL.Image.Image): Pillow image.
    """
    pipeline_args = {
        "prompt": kwargs.get("prompt"),
        "width": kwargs.get("width"),
        "height": kwargs.get("height"),
        "negative_prompt": kwargs.get("negative_prompt"),
        "use_safetensors": kwargs.get("use_safetensors", True),
    }

    if kwargs.get("guidance_scale"):
        pipeline_args["guidance_scale"] = kwargs.get("guidance_scale")

    if kwargs.get("num_inference_steps"):
        pipeline_args["num_inference_steps"] = kwargs.get("num_inference_steps")

    Pipeline = diffusers.AutoPipelineForText2Image

    if kwargs.get("image"):
        pipeline_args["image"] = diffusers.utils.load_image(kwargs.get("image"))
        Pipeline = diffusers.AutoPipelineForImage2Image

    if kwargs.get("mask_image"):
        pipeline_args["mask_image"] = diffusers.utils.load_image(
            kwargs.get("mask_image")
        )
        Pipeline = diffusers.AutoPipelineForInpainting

    pipeline = Pipeline.from_pretrained(kwargs.get("model"))

    device = kwargs.get("device")
    if device:
        pipeline.to(device)

    images = pipeline(**pipeline_args).images
    return images[0]
