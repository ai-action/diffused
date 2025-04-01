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
        help="output file",
    )

    parser.add_argument(
        "--width",
        "-W",
        type=int,
        help="image width",
    )

    parser.add_argument(
        "--height",
        "-H",
        type=int,
        help="image height",
    )

    parser.add_argument(
        "--device",
        "-d",
        help="device (cpu, cuda, mps)",
    )

    args = parser.parse_args(argv)

    filename = args.output if args.output else f"{uuid1()}.png"
    image = generate(
        model=args.model,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        device=args.device,
    )
    image.save(filename)
    print(f"🤗 {filename}")


if __name__ == "__main__":  # pragma: no cover
    main()
