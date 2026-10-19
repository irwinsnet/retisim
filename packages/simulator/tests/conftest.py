"""Test fixtures."""

import pathlib
import shutil
import tomllib

import pytest

TESTS_FOLDER = pathlib.Path(__file__).parent
DATA_FOLDER = TESTS_FOLDER / "data"
OUTPUT_FOLDER = TESTS_FOLDER / "output"


@pytest.fixture
def empty_output_folder() -> pathlib.Path:
    """Create an empty folder for output files before each test."""
    if not OUTPUT_FOLDER.exists():
        OUTPUT_FOLDER.mkdir()
        return OUTPUT_FOLDER
    for item in OUTPUT_FOLDER.iterdir():
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(item)
    return OUTPUT_FOLDER


@pytest.fixture
def data_paths() -> dict[str, pathlib.Path]:
    """Get filesystem paths to portfolio data."""
    with open(DATA_FOLDER / "data.toml", "rb") as tfile:
        pdata = tomllib.load(tfile)
    return {key: pathlib.Path(val) for key, val in pdata.items()}
