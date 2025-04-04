from unittest.mock import Mock, create_autospec, patch

from diffused import generate


def pipeline(**kwargs):
    pass  # pragma: no cover


def pipeline_to(*args):
    pass  # pragma: no cover


pipeline.to = create_autospec(pipeline_to)
mock_pipeline = create_autospec(pipeline)


@patch(
    "diffusers.AutoPipelineForText2Image.from_pretrained", return_value=mock_pipeline
)
def test_generate_text_to_image(mock_from_pretrained: Mock) -> None:
    model = "test/model"
    pipeline_args = {
        "prompt": "test prompt",
        "width": None,
        "height": None,
        "negative_prompt": None,
        "use_safetensors": True,
    }
    image = generate(model=model, prompt=pipeline_args["prompt"])
    assert isinstance(image, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_pipeline.assert_called_once_with(**pipeline_args)
    mock_pipeline.to.assert_not_called()
    mock_pipeline.reset_mock()


@patch(
    "diffusers.AutoPipelineForText2Image.from_pretrained", return_value=mock_pipeline
)
def test_generate_text_to_image_with_arguments(mock_from_pretrained: Mock) -> None:
    model = "test/model"
    device = "cuda"
    pipeline_args = {
        "prompt": "test prompt",
        "negative_prompt": "test negative prompt",
        "width": 1024,
        "height": 1024,
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "use_safetensors": False,
    }
    image = generate(model=model, device=device, **pipeline_args)
    assert isinstance(image, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_pipeline.assert_called_once_with(**pipeline_args)
    mock_pipeline.to.assert_called_once_with(device)
    mock_pipeline.reset_mock()
    mock_pipeline.to.reset_mock()


@patch("diffusers.utils.load_image")
@patch(
    "diffusers.AutoPipelineForImage2Image.from_pretrained", return_value=mock_pipeline
)
def test_generate_image_to_image(
    mock_from_pretrained: Mock, mock_load_image: Mock
) -> None:
    model = "test/model"
    image = "https://example.com/image.png"
    pipeline_args = {
        "prompt": "test prompt",
        "image": mock_load_image(),
        "width": None,
        "height": None,
        "negative_prompt": None,
        "use_safetensors": True,
    }
    output = generate(model=model, prompt=pipeline_args["prompt"], image=image)
    assert isinstance(output, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_load_image.assert_called_with(image)
    mock_pipeline.assert_called_once_with(**pipeline_args)
    mock_pipeline.to.assert_not_called()
    mock_pipeline.reset_mock()
