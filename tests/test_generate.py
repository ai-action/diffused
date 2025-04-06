from unittest.mock import ANY, Mock, call, patch

from diffused import generate

from .conftest import Pipeline, device, image, mask_image, model, prompt


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_text_to_image(mock_from_pretrained: Mock) -> None:
    pipeline_args = {
        "prompt": prompt,
        "negative_prompt": None,
        "width": None,
        "height": None,
        "num_images_per_prompt": 1,
        "use_safetensors": True,
    }
    Pipeline.reset_mock()
    images = generate(model=model, prompt=prompt)
    assert len(images) == 1
    assert isinstance(images[0], Mock)
    mock_from_pretrained.assert_called_once_with(model)
    Pipeline.mock.assert_called_once_with(**pipeline_args)
    Pipeline.to.assert_not_called()


@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_text_to_image_with_arguments(mock_from_pretrained: Mock) -> None:
    pipeline_args = {
        "prompt": prompt,
        "negative_prompt": "test negative prompt",
        "width": 1024,
        "height": 1024,
        "num_images_per_prompt": 2,
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "strength": 0.5,
        "use_safetensors": False,
    }
    Pipeline.reset_mock()
    images = generate(model=model, device=device, **pipeline_args)
    assert len(images) == 2
    assert isinstance(images[0], Mock)
    mock_from_pretrained.assert_called_once_with(model)
    Pipeline.mock.assert_called_once_with(**pipeline_args)
    Pipeline.to.assert_called_once_with(device)


@patch("torch.Generator")
@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_text_to_image_with_seed(
    mock_from_pretrained: Mock, mock_generator: Mock
) -> None:
    seed = -1
    pipeline_args = {
        "prompt": prompt,
        "negative_prompt": None,
        "width": None,
        "height": None,
        "num_images_per_prompt": 1,
        "use_safetensors": True,
    }
    Pipeline.reset_mock()
    images = generate(model=model, device=device, seed=seed, **pipeline_args)
    assert len(images) == 1
    assert isinstance(images[0], Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_generator.assert_called_once_with(device=device)
    pipeline_args["generator"] = ANY
    Pipeline.mock.assert_called_once_with(**pipeline_args)
    Pipeline.to.assert_called_once_with(device)


@patch("torch.Generator")
@patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=Pipeline)
def test_arguments_with_zero_values(
    mock_from_pretrained: Mock, mock_generator: Mock
) -> None:
    seed = 0
    pipeline_args = {
        "prompt": prompt,
        "negative_prompt": None,
        "width": None,
        "height": None,
        "num_images_per_prompt": 1,
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "strength": 0,
        "use_safetensors": True,
    }
    Pipeline.reset_mock()
    images = generate(model=model, seed=seed, **pipeline_args)
    assert len(images) == 1
    assert isinstance(images[0], Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_generator.assert_called_once_with(device=None)
    pipeline_args["generator"] = ANY
    Pipeline.mock.assert_called_once_with(**pipeline_args)
    Pipeline.to.assert_not_called()


@patch("diffusers.utils.load_image")
@patch("diffusers.AutoPipelineForImage2Image.from_pretrained", return_value=Pipeline)
def test_image_to_image(mock_from_pretrained: Mock, mock_load_image: Mock) -> None:
    pipeline_args = {
        "prompt": prompt,
        "negative_prompt": None,
        "image": mock_load_image(),
        "width": None,
        "height": None,
        "num_images_per_prompt": 1,
        "use_safetensors": True,
    }
    mock_load_image.reset_mock()
    Pipeline.reset_mock()
    images = generate(model=model, prompt=prompt, image=image)
    assert len(images) == 1
    assert isinstance(images[0], Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_load_image.assert_called_once_with(image)
    Pipeline.mock.assert_called_once_with(**pipeline_args)
    Pipeline.to.assert_not_called()


@patch("diffusers.utils.load_image")
@patch("diffusers.AutoPipelineForInpainting.from_pretrained", return_value=Pipeline)
def test_inpainting(mock_from_pretrained: Mock, mock_load_image: Mock) -> None:
    pipeline_args = {
        "prompt": prompt,
        "negative_prompt": None,
        "image": mock_load_image(),
        "mask_image": mock_load_image(),
        "width": None,
        "height": None,
        "num_images_per_prompt": 1,
        "use_safetensors": True,
    }
    mock_load_image.reset_mock()
    Pipeline.reset_mock()
    images = generate(model=model, prompt=prompt, image=image, mask_image=mask_image)
    assert len(images) == 1
    assert isinstance(images[0], Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_load_image.assert_has_calls([call(image), call(mask_image)])
    Pipeline.mock.assert_called_once_with(**pipeline_args)
    Pipeline.to.assert_not_called()
