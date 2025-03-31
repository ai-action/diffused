from unittest.mock import Mock, create_autospec, patch

from diffused import generate


def pipeline(prompt: str, width: int | None, height: int | None) -> None:
    pass  # pragma: no cover


def pipeline_to(device: str) -> None:
    pass  # pragma: no cover


pipeline.to = create_autospec(pipeline_to)
mock_pipeline = create_autospec(pipeline)


@patch(
    "diffusers.AutoPipelineForText2Image.from_pretrained", return_value=mock_pipeline
)
def test_generate(mock_from_pretrained: Mock) -> None:
    model = "model/test"
    prompt = "test prompt"
    image = generate(model, prompt)
    assert isinstance(image, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_pipeline.assert_called_once_with(prompt=prompt, width=None, height=None)
    mock_pipeline.to.assert_called_once_with(None)
    mock_pipeline.reset_mock()
    mock_pipeline.to.reset_mock()


@patch(
    "diffusers.AutoPipelineForText2Image.from_pretrained", return_value=mock_pipeline
)
def test_generate_arguments(mock_from_pretrained: Mock) -> None:
    model = "model/test"
    prompt = "test prompt"
    width = 1024
    height = 1024
    device = "cuda"
    image = generate(
        model=model, prompt=prompt, width=width, height=height, device=device
    )
    assert isinstance(image, Mock)
    mock_from_pretrained.assert_called_once_with(model)
    mock_pipeline.assert_called_once_with(prompt=prompt, width=width, height=height)
    mock_pipeline.to.assert_called_once_with(device)
    mock_pipeline.reset_mock()
    mock_pipeline.to.reset_mock()
