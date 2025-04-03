from argparse import ArgumentParser
from uuid import uuid1

from diffused import __version__, generate


def main(argv: list[str] = None) -> None:
    parser = ArgumentParser(description="Generate image with diffusion model")

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
        "--output",
        "-o",
        help="generated image filename",
    )

    parser.add_argument(
        "--width",
        "-W",
        help="generated image width in pixels",
        type=int,
    )

    parser.add_argument(
        "--height",
        "-H",
        help="generated image height in pixels",
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
        help="what to exclude from the generated image",
    )

    parser.add_argument(
        "--guidance-scale",
        "-gs",
        help="how much the prompt influences image generation",
        type=float,
    )

    parser.add_argument(
        "--inference-steps",
        "-is",
        help="number of diffusion steps",
        type=int,
    )

    args = parser.parse_args(argv)
    generate_args = {
        "model": args.model,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "device": args.device,
        "negative_prompt": args.negative_prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.inference_steps,
    }

    filename = args.output if args.output else f"{uuid1()}.png"
    image = generate(**generate_args)
    image.save(filename)
    print(f"🤗 {filename}")


if __name__ == "__main__":  # pragma: no cover
    main()
