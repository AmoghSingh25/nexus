import pytest
from jaxtyping import install_import_hook

install_import_hook("nexus", "beartype.beartype")


def pytest_addoption(parser):
    parser.addoption(
        "--memtest", action="store_true", default=False, help="run memory test"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "memtest: mark memory test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--memtest"):
        return
    skip_test = pytest.mark.skip(reason="need --memtest option to run")
    for item in items:
        if item.get_closest_marker("memtest"):
            item.add_marker(skip_test)
