from unittest.mock import Mock

model = "test/model"
device = "cuda"
prompt = "prompt"
image = "image.png"
mask_image = "mask.png"


class Pipeline:
    to = Mock()
    mock = Mock()

    def __init__(self, **kwargs):
        Pipeline.mock(**kwargs)
        Pipeline.images = [Mock()] * kwargs.get("num_images_per_prompt")

    @staticmethod
    def reset_mock():
        Pipeline.mock.reset_mock()
        Pipeline.to.reset_mock()
