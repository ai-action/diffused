import argparse
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
        "--device",
        "-d",
        help="device to accelerate computation (cpu, cuda, mps)",
    )

    parser.add_argument(
        "--negative-prompt",
        "-np",
        help="what to exclude from the output image",
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
        "--safetensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use safetensors [default: True]",
    )

    args = parser.parse_args(argv)
    generate_args = {
        "model": args.model,
        "prompt": args.prompt,
        "image": args.image,
        "mask_image": args.mask_image,
        "width": args.width,
        "height": args.height,
        "device": args.device,
        "negative_prompt": args.negative_prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.inference_steps,
        "use_safetensors": args.safetensors,
    }

    filename = args.output if args.output else f"{uuid1()}.png"
    image = generate(**generate_args)
    image.save(filename)
    print(f"🤗 {filename}")


if __name__ == "__main__":  # pragma: no cover
    main()
