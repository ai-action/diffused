from unittest.mock import Mock, create_autospec, patch

from diffused import generate


def pipeline(prompt: str, width: int | None, height: int | None):
    pass  # pragma: no cover


mock_pipeline = create_autospec(pipeline)


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
def test_generate(mock_from_pretrained: Mock) -> None:
    mock_from_pretrained.return_value = mock_pipeline
    model = "model/test"
    prompt = "test prompt"
    image = generate(model=model, prompt=prompt)
    assert isinstance(image, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_pipeline.assert_called_once_with(prompt=prompt, width=None, height=None)
    mock_pipeline.reset_mock()


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
def test_generate_width_height(mock_from_pretrained: Mock) -> None:
    mock_from_pretrained.return_value = mock_pipeline
    model = "model/test"
    prompt = "test prompt"
    width = 1024
    height = 1024
    image = generate(model=model, prompt=prompt, width=width, height=height)
    assert isinstance(image, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_pipeline.assert_called_once_with(prompt=prompt, width=width, height=height)
    mock_pipeline.reset_mock()
