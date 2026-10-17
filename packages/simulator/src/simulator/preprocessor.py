"""Generate a lower Cholesky matrix from a dataset of total returns."""

import csv
import datetime
import decimal
import enum
import pathlib
import re

import numpy as np
from numpy import linalg
import polars as pl

import simulator
from simulator import model


def generate_cholesky_matrx() -> np.array:
    """Generate a lower Cholesky matrix from a dataset of total returns."""
    total_returns = np.loadtxt(
        simulator.data_path / "total-returns.csv",
        delimiter=",",
        skiprows=1,
        usecols=range(1, 15)
    )
    covariance_matrix = np.cov(total_returns, rowvar=False)
    lower_cholesky = linalg.cholesky(covariance_matrix)
    return lower_cholesky


def read_total_returns() -> pl.DataFrame:
    """Build a dataframe of total returns data."""
    schema = {
        "year": pl.Int64,
        **{col: pl.Float64 for col in model.asset_classes.keys()},
        **{col: pl.Float64 for col in model.rates},
        **{f"{col}_dif": pl.Float64 for col in model.rates}
    }
    return pl.read_csv(simulator.data_path / "total-returns.csv", schema=schema)


def read_investment_files(
    folder: pathlib.Path,
    as_of: datetime.date | None = None
) -> list[model.Investment]:
    """Extract investments from all CSV files in a folder and it's subfolders.

    Files without a .csv extension are ignored. Files are read in sorted path
    order so the results are repeatable.
    """
    investments = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".csv":
            investments.extend(read_investment_file(path, as_of))
    return investments


def read_vanguard(
    csv_path: pathlib.Path,
    as_of: datetime.date | None = None,
) -> list[model.Investment]:
    """Read Vanguard CSV file and convert contents to Investments.

    The Vanguard download starts with a holdings section, followed by blank
    rows and a transaction history section. Only the holdings section is read.
    Funds with zero shares are skipped. The price is the total value divided
    by the shares. The file does not record the date of the holdings, so ``as_of`` is used.
    If ``as_of`` is not provided, the file's creation date is used (falls back
    to the modification date on platforms that don't track creation time).
    """
    if as_of is None:
        stat = csv_path.stat()
        timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
        as_of = datetime.date.fromtimestamp(timestamp)
    investments = []
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row["Plan Number"]:
                break  # Blank row marks the end of the holdings section.
            shares = float(row["Shares"])
            if shares == 0:
                continue
            # The Price column is rounded to cents, so derive the price from
            # the total value to keep the investment value exact.
            total_value = decimal.Decimal(row["Total Value"])
            investments.append(model.Investment(
                account=row["Plan Name"],
                security=row["Fund Name"],
                date=as_of,
                shares=shares,
                price=(
                    (total_value / decimal.Decimal(row["Shares"]))
                    .quantize(
                        decimal.Decimal("0.00000001"),
                        rounding=decimal.ROUND_HALF_UP
                    )
                )
            ))
    return investments


def read_tsp(
    csv_path: pathlib.Path,
    as_of: datetime.date | None = None,
) -> list[model.Investment]:
    """Read TSP CSV files and return a list of Investments.

    Funds with zero units are skipped. The price is the closing balance
    divided by the units. If ``as_of`` is not provided, the ending date of
    each row's Date Range column is used.
    """
    investments = []
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            shares = float(row["Units"])
            if shares == 0:
                continue
            if as_of is None:
                end_date = row["Date Range"].split(" - ")[1]
                date = datetime.datetime.strptime(end_date, "%B %d, %Y").date()
            else:
                date = as_of
            # The Fund Price column is rounded to cents, so derive the price
            # from the closing balance to keep the investment value exact.
            closing_balance = _parse_dollars(row["Closing Balance"])
            investments.append(model.Investment(
                account=row["Plan"],
                security=row["Fund Name"],
                date=date,
                shares=shares,
                price=(
                    (closing_balance / decimal.Decimal(row["Units"]))
                    .quantize(
                     decimal.Decimal("0.00000001"),
                     rounding=decimal.ROUND_HALF_UP
                    )
                )
            ))
    return investments


def read_schwab(
    csv_path: pathlib.Path,
    as_of: datetime.date | None = None,
) -> list[model.Investment]:
    """Read Schwab CSV files and return a list of Investments.

    The first line names the account and the date of the positions, e.g.
    "Positions for account IRA Trad. Stacy ...600 as of 11:36 AM ET,
    2026/09/27". The date is used if ``as_of`` is not provided. Cash is
    returned as a "Cash" security priced at $1. Positions with zero shares
    and the Positions Total row are skipped. The price is the market value
    divided by the quantity.
    """
    investments = []
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        title = next(csv.reader(csv_file))[0]
        match = re.fullmatch(
            r"Positions for account (.+?) \.\.\.\d+ as of .*, "
            r"(\d{4}/\d{2}/\d{2})",
            title,
        )
        if match is None:
            raise ValueError(f"Unrecognized Schwab title line: {title!r}")
        account = match.group(1).strip()
        if as_of is None:
            as_of = datetime.datetime.strptime(match.group(2), "%Y/%m/%d").date()
        next(csv_file)  # Skip the blank line before the column headers.
        reader = csv.DictReader(csv_file)
        for row in reader:
            symbol = row["Symbol"]
            if symbol == "Positions Total":
                continue
            market_value = _parse_dollars(row["Mkt Val (Market Value)"])
            if symbol == "Cash & Cash Investments":
                if market_value == 0:
                    continue
                investments.append(model.Investment(
                    account=account,
                    security="Cash",
                    date=as_of,
                    shares=float(market_value),
                    price=decimal.Decimal("1.00"),
                ))
                continue
            quantity = decimal.Decimal(row["Qty (Quantity)"].replace(",", ""))
            if quantity == 0:
                continue
            investments.append(model.Investment(
                account=account,
                security=symbol,
                date=as_of,
                shares=float(quantity),
                price=(
                    (market_value / quantity)
                    .quantize(
                        decimal.Decimal("0.00000001"),
                        rounding=decimal.ROUND_HALF_UP
                    )
                )
            ))
    return investments


def _parse_dollars(text: str) -> decimal.Decimal:
    """Convert a dollar string such as "$1,234.56 " or "($12.34)" to Decimal."""
    text = text.strip()
    negative = text.startswith("(") and text.endswith(")")
    value = decimal.Decimal(text.strip("()").lstrip("$").replace(",", ""))
    return -value if negative else value


class FileType(enum.Enum):
    """Type of CSV file."""
    SCHWAB = enum.auto()
    TSP = enum.auto()
    VANGUARD = enum.auto()


def categorize_file(path: pathlib.Path) -> FileType:
    """Determine the type of on investment CSV file.

    The type is identified from the first line of the file. Raises
    ValueError if the file is not a recognized type.
    """
    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        first_row = next(csv.reader(csv_file), [])
    if first_row[:3] == ["Plan Number", "Plan Name", "Fund Name"]:
        return FileType.VANGUARD
    if first_row[:3] == ["Plan", "Asset Class", "Fund Name"]:
        return FileType.TSP
    if first_row and first_row[0].startswith("Positions for account "):
        return FileType.SCHWAB
    raise ValueError(f"Unrecognized investment CSV file: {path}")


def read_investment_file(
        path: pathlib.Path, as_of: datetime.date | None = None,) -> list[model.Investment]:
    """Determine the type then read an investment CSV file."""
    ftype = categorize_file(path)
    match ftype:
        case FileType.SCHWAB:
            return read_schwab(path, as_of)
        case FileType.TSP:
            return read_tsp(path, as_of)
        case FileType.VANGUARD:
            return read_vanguard(path, as_of)


def read_asset_classes(path: pathlib.Path) -> pl.DataFrame:
    """Generate a dataframe of securities and their asset classes."""
    schema = {
        col: pl.Float64 for col in [
            "lcap", "scap", "idev", "iemerg", "hyieldbonds", "ibonds",
            "tbill", "reit", "gold", "total"
        ]
    }
    return pl.read_excel(path, schema_overrides=schema).fill_null(0)