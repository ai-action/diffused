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


def test_version_short(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["-v"])
    captured = capsys.readouterr()
    assert captured.out == __version__ + "\n"


def test_help(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "Generate image with diffusion model" in captured.out


def test_help_short(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["-h"])
    captured = capsys.readouterr()
    assert "Generate image with diffusion model" in captured.out


def test_no_arguments(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exception:
        main([])
    captured = capsys.readouterr()
    assert exception.type is SystemExit
    assert "error: the following arguments are required: model, prompt" in captured.err


def test_invalid_argument(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exception:
        main(["model", "prompt", "--invalid"])
    captured = capsys.readouterr()
    assert exception.type is SystemExit
    assert "error: unrecognized arguments: --invalid" in captured.err


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_generate_image(
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
def test_output(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    filename = "image.png"
    main(["model", "prompt", "--output", filename])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert captured.out == f"🤗 {filename}\n"


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_output_short(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    filename = "image.png"
    main(["model", "prompt", "-o", filename])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert captured.out == f"🤗 {filename}\n"


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_width_height(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "--width", "1024", "--height", "1024"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_width_height_short(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "-W", "1024", "-H", "1024"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_device(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "--device", "cuda"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_device_short(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "-d", "cpu"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_negative_prompt(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "--negative-prompt", "blurry"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_negative_prompt_short(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "-np", "blurry"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_guidance_scale(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "--guidance-scale", "7.5"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_guidance_scale_short(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "-gs", "7.5"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_inference_steps(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "--inference-steps", "50"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_inference_steps_short(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "-is", "15"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_safetensors(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "--safetensors"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
@patch("PIL.Image.Image.save")
def test_no_safetensors(
    mock_from_pretrained: Mock, mock_save: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    main(["model", "prompt", "--no-safetensors"])
    mock_save.assert_called_once()
    captured = capsys.readouterr()
    assert "🤗 " in captured.out
