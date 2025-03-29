# diffused

[![PyPI version](https://badgen.net/pypi/v/diffused)](https://pypi.org/project/diffused/)
[![codecov](https://codecov.io/gh/ai-action/diffused/graph/badge.svg?token=fObC6rYkAJ)](https://codecov.io/gh/ai-action/diffused)
[![lint](https://github.com/ai-action/diffused/actions/workflows/lint.yml/badge.svg)](https://github.com/ai-action/diffused/actions/workflows/lint.yml)

🤗 Generate images with diffusion [models](https://huggingface.co/models?pipeline_tag=text-to-image):

```sh
diffused <model> <prompt>
```

## Quick Start

```sh
pipx run diffused segmind/tiny-sd "an apple"
```

## Prerequisites

- [Python](https://www.python.org/)
- [pipx](https://pipx.pypa.io/)

## Install

[Python](https://pypi.org/project/diffused/):

```sh
pipx install diffused
```

## Usage

Generate image with [model](https://huggingface.co/segmind/tiny-sd) and prompt:

```sh
diffused segmind/tiny-sd "portrait of a cat"
```

Generate image with [model](https://huggingface.co/OFA-Sys/small-stable-diffusion-v0), prompt, and filename:

```sh
diffused OFA-Sys/small-stable-diffusion-v0 "cartoon of a cat" --output cat.png
```

## Arguments

### model

**Required**: Text-to-image diffusion [model](https://huggingface.co/models?pipeline_tag=text-to-image).

```sh
diffused segmind/SSD-1B "An astronaut riding a green horse"
```

### prompt

**Required**: Text prompt.

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cinematic photo of Godzilla eating sushi with a cat in a izakaya, 35mm photograph, film, professional, 4k, highly detailed"
```

### --output

**Optional**: Generated image filename.

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cat eating sushi" --output cat.jpg
```

```sh
diffused dreamlike-art/dreamlike-photoreal-2.0 "cat eating sushi" -o cat.jpg
```

### --version

```sh
diffused --version # diffused -v
```

### --help

```sh
diffused --help # diffused -h
```

## License

[MIT](https://github.com/ai-action/diffused/blob/master/LICENSE)
