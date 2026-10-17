"""Test data preprocessing features."""

import pathlib

import numpy as np
import rich.console

from simulator import model, preprocessor


PACKAGE_DATA_FOLDER = pathlib.Path(__file__).parents[1] / "src" / "simulator" / "_data"
cons = rich.console.Console()


def test_cholesky(empty_output_folder) -> None:
    """Generate a cholesky matrix."""
    cholesky = preprocessor.generate_cholesky_matrx(
        PACKAGE_DATA_FOLDER / "total-returns.csv"
    )
    np.savetxt(
        empty_output_folder / "cholesky.csv",
        cholesky,
        delimiter=",",
        fmt="%3.8f"
    )


def test_read_asset_classes(data_paths: dict[str, pathlib.Path]) -> None:
    """Read asset classes."""
    # Action
    aclasses = preprocessor.read_asset_classes(data_paths["asset-classes-path"])
    # Assert
    assert len(aclasses) > 1
    for col in [
            "name", "ticker",
            "lcap", "scap", "idev", "iemerg", "hyieldbonds", "ibonds",
            "tbill", "reit", "gold", "total"
    ]:
        assert col in aclasses.columns


