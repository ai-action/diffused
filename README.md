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

**Required**: Text-to-image diffusion [model](https://huggingface.co/models?pipeline_tag=text-to-image).

```sh
diffused segmind/SSD-1B "An astronaut riding a green horse"
```

### `prompt`

**Required**: Text prompt.

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cinematic photo of Godzilla eating sushi with a cat in a izakaya, 35mm photograph, film, professional, 4k, highly detailed"
```

### `--output`

**Optional**: Generated image filename.

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cat eating sushi" --output=cat.jpg
```

With short option:

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cat eating sushi" -o=cat.jpg
```

### `--width`

**Optional**: Generated image width in pixels.

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" --width=1024
```

With short option:

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" -W=1024
```

### `--height`

**Optional**: Generated image height in pixels.

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" --height=1024
```

With short option:

```sh
diffused stabilityai/stable-diffusion-xl-base-1.0 "dog in space" -H=1024
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

## License

[MIT](https://github.com/ai-action/diffused/blob/master/LICENSE)
