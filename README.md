# diffused

[![PyPI version](https://badgen.net/pypi/v/diffused)](https://pypi.org/project/diffused/)
[![codecov](https://codecov.io/gh/ai-action/diffused/graph/badge.svg?token=fObC6rYkAJ)](https://codecov.io/gh/ai-action/diffused)
[![lint](https://github.com/ai-action/diffused/actions/workflows/lint.yml/badge.svg)](https://github.com/ai-action/diffused/actions/workflows/lint.yml)

🤗 Generate images with diffusion [models](https://huggingface.co/models):

```sh
diffused <model> <prompt>
```

## Quick Start

[Text-to-image](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation):

```sh
pipx run diffused segmind/tiny-sd "red apple"
```

[Image-to-image](https://huggingface.co/docs/diffusers/using-diffusers/img2img):

```sh
pipx run diffused OFA-Sys/small-stable-diffusion-v0 "cat wizard" --image=https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
```

[Inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint):

```sh
pipx run diffused kandinsky-community/kandinsky-2-2-decoder-inpaint "black cat" --image=https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png --mask-image=https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png
```

## Prerequisites

- [Python](https://www.python.org/)
- [pipx](https://pipx.pypa.io/)

## CLI

Install the CLI:

```sh
pipx install diffused
```

### `model`

**Required** (*str*): The diffusion [model](https://huggingface.co/models).

```sh
diffused segmind/SSD-1B "An astronaut riding a green horse"
```

See [segmind/SSD-1B](https://huggingface.co/segmind/SSD-1B).

### `prompt`

**Required** (*str*): The text prompt.

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cinematic photo of Godzilla eating sushi with a cat in a izakaya, 35mm photograph, film, professional, 4k, highly detailed"
```

### `--image`

**Optional** (*str*): The input image path or URL. The initial image is used as a starting point for an [image-to-image](https://huggingface.co/docs/diffusers/using-diffusers/img2img) diffusion process.

```sh
diffused stabilityai/stable-diffusion-xl-refiner-1.0 "astronaut in a desert" --image=https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png
```

With the short option:

```sh
diffused stabilityai/stable-diffusion-xl-refiner-1.0 "astronaut in a desert" -i=./local/image.png
```

### `--mask-image`

**Optional** (*str*): The mask image path or URL. [Inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint) replaces or edits specific areas of an image. [Create a mask image](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint#create-a-mask-image) to inpaint images.

```sh
diffused kandinsky-community/kandinsky-2-2-decoder-inpaint "black cat" --image=https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png --mask-image=https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png
```

With the short option:

```sh
diffused kandinsky-community/kandinsky-2-2-decoder-inpaint "black cat" -i=inpaint.png -mi=inpaint_mask.png
```

### `--output`

**Optional** (*str*): The output image filename.

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cat eating sushi" --output=cat.jpg
```

With the short option:

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cat eating sushi" -o=cat.jpg
```

### `--width`

**Optional** (*int*): The output image width in pixels.

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" --width=1024
```

With the short option:

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" -W=1024
```

### `--height`

**Optional** (*int*): The output image height in pixels.

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" --height=1024
```

With the short option:

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" -H=1024
```

### `--device`

**Optional** (*str*): The [device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) to accelerate the computation (`cpu`, `cuda`, `mps`, `xpu`, `xla`, or `meta`).

```sh
diffused stable-diffusion-v1-5/stable-diffusion-v1-5 "astronaut on earth, 8k" --device=cuda
```

With the short option:

```sh
diffused stable-diffusion-v1-5/stable-diffusion-v1-5 "astronaut on earth, 8k" -d=cuda
```

### `--negative-prompt`

**Optional** (*str*): What to exclude from the output image.

```sh
diffused stabilityai/stable-diffusion-2 "photo of an apple" --negative-prompt="blurry, bright photo, red"
```

With the short option:

```sh
diffused stabilityai/stable-diffusion-2 "photo of an apple" -np="blurry, bright photo, red"
```

### `--guidance-scale`

**Optional** (*int*): How much the prompt influences the output image. A lower value leads to more deviation and creativity, whereas a higher value follows the prompt to a tee.

```sh
diffused stable-diffusion-v1-5/stable-diffusion-v1-5 "astronaut in a jungle" --guidance-scale=7.5
```

With the short option:

```sh
diffused stable-diffusion-v1-5/stable-diffusion-v1-5 "astronaut in a jungle" -gs=7.5
```

### `--inference-steps`

**Optional** (*int*): The number of diffusion steps used during image generation. The more steps you use, the higher the quality, but the generation time will increase.

```sh
diffused CompVis/stable-diffusion-v1-4 "astronaut rides horse" --inference-steps=50
```

With the short option:

```sh
diffused CompVis/stable-diffusion-v1-4 "astronaut rides horse" -is=50
```

### `--no-safetensors`

**Optional** (*bool*): Whether to disable [safetensors](https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors).

```sh
diffused runwayml/stable-diffusion-v1-5 "astronaut on mars" --no-safetensors
```

### `--version`

Show the program's version number and exit:

```sh
diffused --version # diffused -v
```

### `--help`

Show the help message and exit:

```sh
diffused --help # diffused -h
```

## Script

Create a virtual environment:

```sh
python3 -m venv .venv
```

Activate the virtual environment:

```sh
source .venv/bin/activate
```

Install the package:

```sh
pip install diffused
```

Generate an image with a [model](https://huggingface.co/segmind/tiny-sd) and a prompt:

```py
# script.py
from diffused import generate

image = generate(model="segmind/tiny-sd", prompt="apple")
image.save("apple.png")
```

Run the script:

```sh
python script.py
```

Open the image:

```sh
open apple.png
```

See the [API documentation](https://ai-action.github.io/diffused/diffused/generate.html).

## License

[MIT](https://github.com/ai-action/diffused/blob/master/LICENSE)
