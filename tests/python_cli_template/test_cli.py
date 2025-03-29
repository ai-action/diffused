import pytest

from python_cli_template import __version__
from python_cli_template.cli import main


def test_version(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    captured = capsys.readouterr()
    assert captured.out == __version__ + "\n"


def test_help(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "Python CLI Template" in captured.out


def test_hello(capsys: pytest.LogCaptureFixture) -> None:
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"


def test_hello_world(capsys: pytest.LogCaptureFixture) -> None:
    main(["--name", "world"])
    captured = capsys.readouterr()
    assert captured.out == "Hello, world!\n"


def test_invalid(capsys: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exception:
        main(["--invalid"])
    captured = capsys.readouterr()
    assert exception.type is SystemExit
    assert "error: unrecognized arguments: --invalid" in captured.err
