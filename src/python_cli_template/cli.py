from argparse import ArgumentParser

from python_cli_template import __version__
from python_cli_template.template import hello


def main(argv: list[str] = None) -> None:
    parser = ArgumentParser(description="Python CLI Template")

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=__version__,
    )

    parser.add_argument(
        "--name",
        "-n",
        default="World",
        help="name to greet [default: World]",
    )

    parser.add_argument(
        "--color",
        "-c",
        action="store_true",
        help="colorize the text",
    )

    args = parser.parse_args(argv)

    color = "\033[95m" if args.color else ""
    print(f"{color}{hello(args.name)}")


if __name__ == "__main__":  # pragma: no cover
    main()
