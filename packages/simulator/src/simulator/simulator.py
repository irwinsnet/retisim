
import pathlib

import polars as pl

from simulator import model, preprocessor


class Simulator:
    """Run Monte Carlo simulations."""
    securities: pl.DataFrame
    holdings: list[model.Investment]
    total_returns: pl.DataFrame

    def __init__(
        self,
        asset_classes_path: pathlib.Path,
        holdings_path: pathlib.Path
    ) -> None:
        """Read asset classes and holdings from filesystem."""
        self.securities = preprocessor.read_asset_classes(asset_classes_path)
        self.holdings = preprocessor.read_investment_files(holdings_path)
        self.total_returns = preprocessor.read_total_returns()


    @property
    def asset_classes(self) -> list[str]:
        """Asset classes, in column order."""
        return [
            col for col in self.total_returns.columns if col in model.asset_classes
        ]