# Contributing

Pull requests are welcome! By participating in this project, you agree to abide by our [code of conduct](https://github.com/ai-action/.github/blob/master/CODE_OF_CONDUCT.md).

## Fork

[Fork](https://github.com/ai-action/diffused/fork) and then clone the repository:

```sh
# replace <USER> with your username
git clone git@github.com:<USER>/diffused.git
```

```sh
cd diffused
```

## Install

Install [Python](https://www.python.org/):

```sh
brew install python
```

Install [uv](https://docs.astral.sh/uv/):

```sh
brew install uv
```

Install the dependencies:

```sh
uv sync --all-groups
```

Install pre-commit into your git hooks:

```sh
uv run pre-commit install
```

## Develop

Make your changes, add tests/documentation, and ensure [tests](#test) pass.

Write a commit message that follows the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Add missing tests or correct existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Updates configuration files and scripts for continuous integration
- **docs**: Documentation only changes

Push to your fork and create a [pull request](https://github.com/ai-action/diffused/compare/).

At this point, wait for us to review your pull request. We'll try to review pull requests within 1-3 business days. We may suggest changes, improvements, and/or alternatives.

Things that will improve the chance that your pull request will be accepted:

- [ ] Write tests that pass [CI](https://github.com/ai-action/diffused/actions/workflows/test.yml).
- [ ] Write solid documentation.
- [ ] Write a good [commit message](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit).

## Test

If you didn't install the dependencies earlier, you can install only the test dependencies:

```sh
uv sync --only-group test
```

Run the tests:

```sh
uv run pytest
```

Run the tests with [coverage](https://coverage.readthedocs.io/):

```sh
uv run coverage run -m pytest
```

Generate a coverage report:

```sh
uv run coverage report
```

```sh
uv run coverage html
```

Install the package with [uv](https://docs.astral.sh/uv/):

```sh
uv tool install . --force
```

Test the command:

```sh
diffused --help
```

## Lint

If you didn't install the dependencies earlier, you can install only the lint dependencies:

```sh
uv sync --only-group lint
```

Update pre-commit hooks to the latest version:

```sh
uv run pre-commit autoupdate
```

Run all pre-commit hooks:

```sh
uv run pre-commit run --all-files
```

Lint all files in the current directory:

```sh
uv run ruff check
```

Format all files in the current directory:

```sh
uv run ruff format
```

## Build

If you didn't install the dependencies earlier, you can install only the build dependencies:

```sh
uv sync --only-group build
```

Generate the distribution packages:

```sh
uv build
```

Upload all of the archives under `dist` to TestPyPI:

```sh
uv publish --publish-url https://test.pypi.org/legacy/
```

Install the package:

```sh
uv tool install diffused --index-url https://test.pypi.org/simple/
```

Bundle the package with [PyInstaller](https://pyinstaller.org/):

```sh
uv run pyinstaller src/diffused/cli.py --name diffused
```

## Docs

If you didn't install the dependencies earlier, you can install only the docs dependencies:

```sh
uv sync --only-group docs
```

Generate the docs with [pdoc](https://pdoc.dev/):

```sh
uv run pdoc src/diffused/
```

## Release

Release and publish are automated with [Release Please](https://github.com/googleapis/release-please).

[Add a new pending publisher to PyPI.](https://pypi.org/manage/account/publishing/)
