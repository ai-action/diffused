import argparse
import math
import os
from uuid import uuid1

from diffused import __version__, generate


def main(argv: list[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate image with diffusion model")

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=__version__,
    )

    parser.add_argument(
        "model",
        help="diffusion model",
    )

    parser.add_argument(
        "prompt",
        help="text prompt",
    )

    parser.add_argument(
        "--negative-prompt",
        "-np",
        help="what to exclude from the output image",
    )

    parser.add_argument(
        "--image",
        "-i",
        help="input image path/url",
    )

    parser.add_argument(
        "--mask-image",
        "-mi",
        help="mask image path/url",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="output image filename",
    )

    parser.add_argument(
        "--width",
        "-W",
        help="output image width (pixels)",
        type=int,
    )

    parser.add_argument(
        "--height",
        "-H",
        help="output image height (pixels)",
        type=int,
    )

    parser.add_argument(
        "--number",
        "-n",
        default=1,
        help="number of output images [default: 1]",
        type=int,
    )

    parser.add_argument(
        "--guidance-scale",
        "-gs",
        help="how much the prompt influences output image",
        type=float,
    )

    parser.add_argument(
        "--inference-steps",
        "-is",
        help="number of diffusion steps",
        type=int,
    )

    parser.add_argument(
        "--strength",
        "-s",
        help="how much noise is added to the input image",
        type=float,
    )

    parser.add_argument(
        "--seed",
        "-S",
        help="seed for generating reproducible images",
        type=int,
    )

    parser.add_argument(
        "--device",
        "-d",
        help="device to accelerate computation (cpu, cuda, mps)",
    )

    parser.add_argument(
        "--safetensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use safetensors [default: True]",
    )

    args = parser.parse_args(argv)
    generate_args = {
        "device": args.device,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "image": args.image,
        "mask_image": args.mask_image,
        "model": args.model,
        "negative_prompt": args.negative_prompt,
        "num_images_per_prompt": args.number,
        "num_inference_steps": args.inference_steps,
        "prompt": args.prompt,
        "seed": args.seed,
        "strength": args.strength,
        "use_safetensors": args.safetensors,
        "width": args.width,
    }

    images = generate(**generate_args)
    images_length = len(images)

    for index, image in enumerate(images):
        if args.output:
            if images_length == 1:
                filename = args.output
            else:
                basename, extension = os.path.splitext(args.output)
                idx = str(index + 1).rjust(
                    math.floor(math.log10(images_length)) + 1, "0"
                )
                filename = f"{basename}{idx}{extension}"
        else:
            filename = f"{uuid1()}.png"

        image.save(filename)
        print(f"🤗 {filename}")


if __name__ == "__main__":  # pragma: no cover
    main()
