"""Test core data structures."""

import datetime
import decimal
import pathlib

import rich.console

from simulator import model, simulator


cons = rich.console.Console()


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


def test_simulator(data_paths: dict[str, pathlib.Path]) -> None:
    """Create a Simulator object."""
    # Action
    sim = simulator.Simulator(
        data_paths["asset-classes-path"],
        data_paths["holdings-path"]
    )
    # Assert
    cons.print(sim.total_returns)
    cons.print(sim.asset_classes)

