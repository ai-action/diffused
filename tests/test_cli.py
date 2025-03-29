import re
from unittest.mock import Mock, patch

import pytest

from diffused import __version__
from diffused.cli import main


def test_version(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    captured = capsys.readouterr()
    assert captured.out == __version__ + "\n"


def test_help(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "Generate image with diffusion model" in captured.out


def test_required(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exception:
        main([])
    captured = capsys.readouterr()
    assert exception.type is SystemExit
    assert "error: the following arguments are required: model, prompt" in captured.err


def test_invalid(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exception:
        main(["model", "prompt", "--invalid"])
    captured = capsys.readouterr()
    assert exception.type is SystemExit
    assert "error: unrecognized arguments: --invalid" in captured.err


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_generate(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert (
        re.match(
            "🤗 [a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}.png\n",
            captured.out,
        )
        is not None
    )


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_generate_output(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    filename = "image.png"
    main(["model", "prompt", "--output", filename])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert captured.out == f"🤗 {filename}\n"
