# diffused

[![PyPI version](https://badgen.net/pypi/v/diffused)](https://pypi.org/project/diffused/)
[![codecov](https://codecov.io/gh/ai-action/diffused/graph/badge.svg?token=fObC6rYkAJ)](https://codecov.io/gh/ai-action/diffused)
[![lint](https://github.com/ai-action/diffused/actions/workflows/lint.yml/badge.svg)](https://github.com/ai-action/diffused/actions/workflows/lint.yml)

🤗 Generate images with diffusion [models](https://huggingface.co/models?pipeline_tag=text-to-image):

```sh
diffused <model> <prompt>
```

## Quick Start

Generate an image with [model](https://huggingface.co/segmind/tiny-sd) and prompt:

```sh
pipx run diffused segmind/tiny-sd "portrait of a cat"
```

Generate an image with [model](https://huggingface.co/OFA-Sys/small-stable-diffusion-v0), prompt, and filename:

```sh
pipx run diffused OFA-Sys/small-stable-diffusion-v0 "cartoon of a cat" --output cat.png
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

**Required** (_str_): Text-to-image diffusion [model](https://huggingface.co/models?pipeline_tag=text-to-image).

```sh
diffused segmind/SSD-1B "An astronaut riding a green horse"
```

### `prompt`

**Required** (_str_): Text prompt.

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cinematic photo of Godzilla eating sushi with a cat in a izakaya, 35mm photograph, film, professional, 4k, highly detailed"
```

### `--output`

**Optional** (_str_): Generated image filename.

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cat eating sushi" --output=cat.jpg
```

With the short option:

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cat eating sushi" -o=cat.jpg
```

### `--width`

**Optional** (_int_): Generated image width in pixels.

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" --width=1024
```

With the short option:

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" -W=1024
```

### `--height`

**Optional** (_int_): Generated image height in pixels.

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" --height=1024
```

With the short option:

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" -H=1024
```

### `--device`

**Optional** (_str_): [Device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) to accelerate the computation (`cpu`, `cuda`, `mps`, `xpu`, `xla`, or `meta`).

```sh
diffused stable-diffusion-v1-5/stable-diffusion-v1-5 "astronaut in the ocean, 8k" --device=cuda
```

With the short option:

```sh
diffused stable-diffusion-v1-5/stable-diffusion-v1-5 "astronaut in the ocean, 8k" -d=cuda
```

### `--negative-prompt`

**Optional** (_str_): What to exclude from the generated image.

```sh
diffused stabilityai/stable-diffusion-2 "photo of an apple" --negative-prompt="blurry, bright photo, red"
```

With the short option:

```sh
diffused stabilityai/stable-diffusion-2 "photo of an apple" -np="blurry, bright photo, red"
```

### `--guidance-scale`

**Optional** (_int_): How much the prompt influences image generation. A lower value leads to more deviation and creativity, whereas a higher value follows the prompt to a tee.

```sh
diffused stable-diffusion-v1-5/stable-diffusion-v1-5 "astronaut in a jungle" --guidance-scale=7.5
```

With the short option:

```sh
diffused stable-diffusion-v1-5/stable-diffusion-v1-5 "astronaut in a jungle" -gs=7.5
```

### `--inference-steps`

**Optional** (_int_): Number of diffusion steps used during image generation. The more steps you use, the higher the quality, but the generation time will increase.

```sh
diffused CompVis/stable-diffusion-v1-4 "astronaut rides horse" --inference-steps=50
```

With the short option:

```sh
diffused CompVis/stable-diffusion-v1-4 "astronaut rides horse" -is=50
```

### `--no-safetensors`

**Optional** (_bool_): Disable [safetensors](https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors).

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

Generate an image with [model](https://huggingface.co/segmind/tiny-sd) and prompt:

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

See the [API documentation](https://ai-action.github.io/diffused/diffused/generate.html).

## License

[MIT](https://github.com/ai-action/diffused/blob/master/LICENSE)
