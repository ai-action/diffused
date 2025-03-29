from python_cli_template import hello


def test_hello():
    assert hello() == "Hello, World!"


def test_hello_world():
    assert hello("world") == "Hello, world!"
