from unittest.mock import Mock, call, create_autospec, patch

from diffused import generate


def pipeline(**kwargs):
    pass  # pragma: no cover


def pipeline_to(*args):
    pass  # pragma: no cover


pipeline.to = create_autospec(pipeline_to)
mock_pipeline = create_autospec(pipeline)

model = "test/model"
device = "cuda"
prompt = "prompt"
image = "image.png"
mask_image = "mask.png"


@patch(
    "diffusers.AutoPipelineForText2Image.from_pretrained", return_value=mock_pipeline
)
def test_text_to_image(mock_from_pretrained: Mock) -> None:
    pipeline_args = {
        "prompt": prompt,
        "width": None,
        "height": None,
        "negative_prompt": None,
        "use_safetensors": True,
    }
    image = generate(model=model, prompt=prompt)
    assert isinstance(image, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_pipeline.assert_called_once_with(**pipeline_args)
    mock_pipeline.to.assert_not_called()
    mock_pipeline.reset_mock()


@patch(
    "diffusers.AutoPipelineForText2Image.from_pretrained", return_value=mock_pipeline
)
def test_text_to_image_with_arguments(mock_from_pretrained: Mock) -> None:
    pipeline_args = {
        "prompt": prompt,
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
def test_image_to_image(mock_from_pretrained: Mock, mock_load_image: Mock) -> None:
    pipeline_args = {
        "prompt": prompt,
        "image": mock_load_image(),
        "width": None,
        "height": None,
        "negative_prompt": None,
        "use_safetensors": True,
    }
    mock_load_image.reset_mock()
    output = generate(model=model, prompt=prompt, image=image)
    assert isinstance(output, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_load_image.assert_called_once_with(image)
    mock_pipeline.assert_called_once_with(**pipeline_args)
    mock_pipeline.to.assert_not_called()
    mock_pipeline.reset_mock()


@patch("diffusers.utils.load_image")
@patch(
    "diffusers.AutoPipelineForInpainting.from_pretrained", return_value=mock_pipeline
)
def test_inpainting(mock_from_pretrained: Mock, mock_load_image: Mock) -> None:
    pipeline_args = {
        "prompt": prompt,
        "image": mock_load_image(),
        "mask_image": mock_load_image(),
        "width": None,
        "height": None,
        "negative_prompt": None,
        "use_safetensors": True,
    }
    mock_load_image.reset_mock()
    output = generate(model=model, prompt=prompt, image=image, mask_image=mask_image)
    assert isinstance(output, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_load_image.assert_has_calls([call(image), call(mask_image)])
    mock_pipeline.assert_called_once_with(**pipeline_args)
    mock_pipeline.to.assert_not_called()
    mock_pipeline.reset_mock()
