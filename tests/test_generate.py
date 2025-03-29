from unittest.mock import Mock, create_autospec, patch

from diffused import generate

mock_image = Mock()


def pipeline(prompt: str):
    return {"images": [mock_image]}  # pragma: no cover


mock_pipeline = create_autospec(pipeline)


@patch("diffusers.AutoPipelineForText2Image.from_pretrained")
def test_generate(mock_from_pretrained: Mock) -> None:
    mock_from_pretrained.return_value = mock_pipeline
    model = "model/test"
    prompt = "test prompt"
    image = generate(model=model, prompt=prompt)
    assert isinstance(image, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_pipeline.assert_called_once_with(prompt)
