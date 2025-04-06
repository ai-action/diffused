import re
from unittest.mock import Mock, call, patch

import pytest

from diffused import __version__
from diffused.cli import main

from .conftest import Pipeline, device, image, mask_image, model, prompt


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
        main([model, prompt, "--invalid"])
    captured = capsys.readouterr()
    assert exception.type is SystemExit
    assert "error: unrecognized arguments: --invalid" in captured.err


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_generate_image(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt])
    mock_from_pretrained.assert_called_once_with(model)
    Pipeline.mock.assert_called_once_with(
        prompt=prompt,
        negative_prompt=None,
        width=None,
        height=None,
        num_images_per_prompt=1,
        use_safetensors=True,
    )
    assert len(Pipeline.images) == 1
    Pipeline.images[0].save.assert_called_once()
    captured = capsys.readouterr()
    assert (
        re.match(
            "🤗 [a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}.png\n",
            captured.out,
        )
        is not None
    )


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_output(mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--output", image])
    mock_from_pretrained.assert_called_once_with(model)
    assert len(Pipeline.images) == 1
    Pipeline.images[0].save.assert_called_once_with(image)
    captured = capsys.readouterr()
    assert captured.out == f"🤗 {image}\n"


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_output_short(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-o", image])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert captured.out == f"🤗 {image}\n"


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_number(mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-o", "test.jpg", "-n", "2"])
    mock_from_pretrained.assert_called_once_with(model)
    assert len(Pipeline.images) == 2
    captured = capsys.readouterr()
    assert captured.out == "🤗 test1.jpg\n🤗 test2.jpg\n"


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_number_short(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-o", "test.jpg", "-n", "10"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert len(Pipeline.images) == 10
    for img in Pipeline.images:
        img.save.assert_called_once()
    assert "🤗 test01.jpg" in captured.out
    assert "🤗 test10.jpg" in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_width_height(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--width", "1024", "--height", "1024"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_width_height_short(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-W", "1024", "-H", "1024"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_device(mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--device", device])
    mock_from_pretrained.assert_called_once_with(model)
    Pipeline.to.assert_called_once_with(device)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_device_short(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-d", "cpu"])
    mock_from_pretrained.assert_called_once_with(model)
    Pipeline.to.assert_called_once_with("cpu")
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_negative_prompt(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--negative-prompt", "blurry"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_negative_prompt_short(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-np", "blurry"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_guidance_scale(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--guidance-scale", "7.5"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_guidance_scale_short(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-gs", "7.5"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_inference_steps(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--inference-steps", "50"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_inference_steps_short(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-is", "15"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_strength(mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--strength", "0.5"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_strength_short(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-s", "0.5"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_seed(mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--seed", "0"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_seed_short(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-S", "-1"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_safetensors(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--safetensors"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_no_safetensors(
    mock_from_pretrained: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--no-safetensors"])
    mock_from_pretrained.assert_called_once_with(model)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.utils.load_image")
@patch("diffusers.AutoPipelineForImage2Image.from_pretrained", return_value=Pipeline)
def test_image(
    mock_from_pretrained: Mock, mock_load_image: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--image", image])
    mock_from_pretrained.assert_called_once_with(model)
    mock_load_image.assert_called_once_with(image)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.utils.load_image")
@patch("diffusers.AutoPipelineForImage2Image.from_pretrained", return_value=Pipeline)
def test_image_short(
    mock_from_pretrained: Mock, mock_load_image: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-i", image])
    mock_from_pretrained.assert_called_once_with(model)
    mock_load_image.assert_called_once_with(image)
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.utils.load_image")
@patch("diffusers.AutoPipelineForInpainting.from_pretrained", return_value=Pipeline)
def test_mask_image(
    mock_from_pretrained: Mock, mock_load_image: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "--image", image, "--mask-image", mask_image])
    mock_from_pretrained.assert_called_once_with(model)
    mock_load_image.assert_has_calls([call(image), call(mask_image)])
    captured = capsys.readouterr()
    assert "🤗 " in captured.out


@patch("diffusers.utils.load_image")
@patch("diffusers.AutoPipelineForInpainting.from_pretrained", return_value=Pipeline)
def test_mask_image_short(
    mock_from_pretrained: Mock, mock_load_image: Mock, capsys: pytest.LogCaptureFixture
) -> None:
    Pipeline.reset_mock()
    main([model, prompt, "-i", image, "-mi", mask_image])
    mock_from_pretrained.assert_called_once_with(model)
    mock_load_image.assert_has_calls([call(image), call(mask_image)])
    captured = capsys.readouterr()
    assert "🤗 " in captured.out
