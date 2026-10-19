"""Test data preprocessing features."""

import pathlib

import rich.console

from simulator import preprocessor

PACKAGE_DATA_FOLDER = pathlib.Path(__file__).parents[1] / "src" / "simulator" / "_data"
cons = rich.console.Console()


def test_read_asset_classes(data_paths: dict[str, pathlib.Path]) -> None:
    """Read asset classes."""
    # Action
    aclasses = preprocessor.read_asset_classes(data_paths["asset-classes-path"])
    # Assert
    assert len(aclasses) > 1
    for col in [
        "name",
        "ticker",
        "lcap",
        "scap",
        "idev",
        "iemerg",
        "hyieldbonds",
        "ibonds",
        "tbill",
        "reit",
        "gold",
        "total",
    ]:
        assert col in aclasses.columns
