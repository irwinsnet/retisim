"""Run Monte Carlo simulations on a retirement portfolio."""

import functools
import pathlib

import numpy as np
import polars as pl
from numpy import linalg

from simulator import model, preprocessor


class RandomGenerator:
    """Generate random rates and asset class returns."""

    """Asset class returns by year."""
    historical_returns: pl.DataFrame
    """Federal funds rate and inflaction rate by year."""
    historical_rates: pl.DataFrame
    """Mean rates of return and standard deviations for various asset classes."""
    asset_class_stats: pl.DataFrame
    """Inflation rate and federal funds rate mean and standard deviation."""
    rate_stats: pl.DataFrame
    """Covariance matrix for rates of return by asset class."""
    returns_covariance: pl.DataFrame
    """The lower Cholesky decomposition matrix of the returns covariance matrix."""
    returns_cholsky_array: np.ndarray
    """Random number generator."""
    _rng: np.random.Generator

    def __init__(self) -> None:
        """Read asset classes and holdings from filesystem."""
        historical_data = preprocessor.read_returns_and_rates()
        self.historical_returns, self.historical_rates = historical_data
        self.asset_class_stats = self._calc_stats(self.historical_returns)
        self.rate_stats = self._calc_stats(self.historical_rates)
        self.returns_covariance = self._calc_covariance()
        self.returns_cholsky_array = self._calc_asset_cholesky()
        self._rng = np.random.default_rng()

    @functools.cached_property
    def asset_classes(self) -> list[str]:
        """Asset classes, in column order."""
        return [
            col for col in self.historical_returns.columns if col in model.asset_classes
        ]

    @functools.cached_property
    def rates(self) -> list[str]:
        """Financial rates"""
        return [col for col in self.historical_returns.columns if col in model.rates]

    @functools.cached_property
    def covariance_array(self) -> np.ndarray:
        """Get covariance matrix as a numpy array."""
        return self.returns_covariance.select(*[self.asset_classes]).to_numpy()

    @functools.cached_property
    def asset_means_array(self) -> np.ndarray:
        """Get mean of asset class returns as a numpy array."""
        return (
            self.asset_class_stats.filter(pl.col("stat") == "mean")
            .drop("stat")
            .to_numpy()
        )

    @functools.cached_property
    def asset_sdevs_array(self) -> np.ndarray:
        """Get standard deviation of asset class returns as a numpy array."""
        return (
            self.asset_class_stats.filter(pl.col("stat") == "sdev")
            .drop("stat")
            .to_numpy()
        )

    def _calc_stats(self, source: pl.DataFrame) -> pl.DataFrame:
        """Calculate means and standard deviations for each asset class."""
        means = source.drop("year").mean()
        sdevs = source.drop("year").std()
        return pl.concat([means, sdevs]).insert_column(
            0, pl.Series("stat", ["mean", "sdev"])
        )

    def _calc_covariance(self) -> pl.DataFrame:
        """Calculate the covariance matrix from the total returns."""
        cov = np.cov(
            self.historical_returns.select(*self.asset_classes).to_numpy(), rowvar=False
        )
        cov_df = pl.from_numpy(cov)
        cov_df.columns = self.asset_classes
        cov_df.insert_column(0, pl.Series("asset", self.asset_classes))
        return cov_df

    def _calc_asset_cholesky(self) -> np.ndarray:
        """Calculate the Cholesky decomposition of the covariance matrix."""
        return linalg.cholesky(self.covariance_array)

    def get_random_returns(self, num_samples: int) -> np.ndarray:
        """Get correlated random returns."""
        standard_normals = self._rng.normal(size=(num_samples, len(self.asset_classes)))
        correlated_normals = (self.returns_cholsky_array @ standard_normals.T).T
        # Cholesky array comes from a covariance matrix, so generated values already
        #   have desired standard deviation.
        return self.asset_means_array + correlated_normals


class PortfolioSimulator:
    """Simulate future returns for an investment portfolio."""

    """Individual securities with asset classes."""
    securities: pl.DataFrame
    """List of investments held in portfolio."""
    holdings: list[model.Investment]
    """Generator of correllated asset class returns."""
    _generator: RandomGenerator

    def __init__(
        self, asset_classes_path: pathlib.Path, holdings_path: pathlib.Path
    ) -> None:
        """Read asset classes and holdings from filesystem."""
        self.securities = preprocessor.read_asset_classes(asset_classes_path)
        self.holdings = preprocessor.read_investment_files(holdings_path)
        self._generator = RandomGenerator()

    
