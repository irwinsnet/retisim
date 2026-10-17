"""Core data structures for the retirement simulation."""
import dataclasses
import datetime
import decimal


asset_classes = {
    "lcap": "Large Cap US Stocks",
    "scap": "Small Cap US Stocks",
    "idev": "International Developed Markets Stocks",
    "imerg": "International Emerging Markets Stocks",
    "bonds": "US Bonds",
    "hyieldbonds": "US High Yield Bonds",
    "ibonds": "International Bonds",
    "tbill": "US Treasury Securities",
    "reit": "US Real Estate Investment Trusts",
    "gold": "Gold"
}

rates = {
    "ffrate": "Federal Funds Rate",
    "infrate": "US Inflation Rate",
}


@dataclasses.dataclass
class Investment:
    """An individual security held in an investment account."""
    account: str
    security: str
    date: datetime
    shares: float
    price: decimal.Decimal

    @property
    def value(self) -> decimal.Decimal:
        """Shares times price."""
        return round(decimal.Decimal.from_float(self.shares) * self.price, 2)



    