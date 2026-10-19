"""Test core data structures."""

import datetime
import decimal

import numpy as np
import rich.console
from scipy import linalg

from simulator import model, simulator

cons = rich.console.Console()



class TestGenerator:
    """Test generation of correlated random values."""

    def test_mean_returns(self) -> None:
        """Test mean calculations."""
        # Arrange
        gen = simulator.RandomGenerator()
        # Act
        means = gen.asset_means_array
        # Assert
        assert means.shape[0] == 1
        assert means.shape[1] == len(gen.asset_classes)
        assert np.all(means > 0)

    def test_sdev_returns(self) -> None:
        """Test standard deviation calculations."""
        # Arrange
        gen = simulator.RandomGenerator()
        # Act
        means = gen.asset_sdevs_array
        # Assert
        assert means.shape[0] == 1
        assert means.shape[1] == len(gen.asset_classes)
        assert np.all(means > 0)

    def test_covariance_array(self) -> None:
        """Generate the covariance array."""
        # Arrange
        gen = simulator.RandomGenerator()
        # Act
        cov = gen.covariance_array
        # Assert
        num_classes = len(gen.asset_classes)
        assert cov.shape == (num_classes, num_classes)
        assert linalg.issymmetric(cov)

    def test_random_returns(self) -> None:
        """Generate random returns"""
        # Arrange
        gen = simulator.RandomGenerator()
        # Act
        returns = gen.get_random_returns(1000)
        # Assert
        means = returns.mean(axis=0)
        exp_means = gen.asset_means_array
        mean_diffs = np.absolute(means - exp_means)
        # print("Max mean diff (proportion):", np.max(mean_diffs / exp_means))
        assert np.all(mean_diffs < exp_means * 0.5)
        sdevs = returns.std(axis=0)
        exp_sdevs = gen.asset_sdevs_array
        sdev_diffs = np.absolute(sdevs - exp_sdevs)
        # print("Max sdef diff (proportion):", np.max(sdev_diffs / exp_sdevs))
        assert np.all(sdev_diffs < exp_sdevs * 0.5)


def test_investment_value() -> None:
    """Calculate investment value."""
    # Arrange
    investment = model.Investment(
        account="Brokerage",
        security="VTI",
        date=datetime.date(2026, 1, 2),
        shares=123.456,
        price=decimal.Decimal("45.67"),
    )
    # Act
    value = investment.value
    # Assert
    assert isinstance(value, decimal.Decimal)
    assert value == decimal.Decimal("5638.24")  # 5638.23552 rounded
    assert value.as_tuple().exponent == -2


def test_simulator() -> None:
    """Create a Simulator object."""
    # Action
    sim = simulator.RandomGenerator()
    # Assert
    cons.print(sim.rate_stats)
    cons.print(sim.asset_class_stats)


def test_random_returns() -> None:
    """Create a Simulator object."""
    # Arrange
    sim = simulator.RandomGenerator()
    # Act
    returns = sim.get_random_returns(3)
    # Assert
    print()
    cons.print(returns)
    cons.print(returns.shape)
    cons.print(sim.asset_means_array)
    cons.print(sim.asset_means_array.shape)
