import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Asset Class Covariance
    The _total-returns-asset.xlsx_ datset was generated from a dataset at https://themeasureofaplan.com/investment-returns-by-asset-class/. It contains total returns by year since 1985 for several asset classes.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from numpy import linalg
    import plotly.express as px
    import polars as pl
    import polars.selectors as cs
    from scipy import stats
    import seaborn as sns

    return cs, linalg, mo, np, pl, sns, stats


@app.cell
def _(mo, pl):
    data_path = mo.notebook_dir() / "data"

    returns_df = pl.read_excel(
        data_path / "total-returns-asset.xlsx",
        read_options={
            "skip_rows": 2,
            "column_names": [
                "year",
                "lcap",         # US Large Cap Stocks
                "scap",         # US Small Cap Stocks
                "idev",         # International Stockes, Developed Markets
                "iemerg",       # Intenational Stocks, Emerging Markets
                "can",          # Canadian Stocks
                "bonds",        # All US Bonds
                "hyieldbonds",  # Hi-Yield US Bonds
                "ibonds",       # International Bonds
                "canbonds",     # Candadien Bonds 
                "tbill",        # Treasury Bills
                "reit",         # Real Estate Investment Trusts
                "gold"          # Gold
            ]
        }
    )
    returns_df
    return (returns_df,)


@app.cell
def _(returns_df):
    returns_df.describe()
    return


@app.cell
def _(asset_input, returns_df, sns):
    def plot_returns():
        plot = sns.displot(
            data=returns_df[asset_input.value].to_list(),
            kind="kde"
        )
        return plot

    plot_returns()
    return


@app.cell(hide_code=True)
def _(mo, returns_df):
    asset_input = mo.ui.dropdown(
        returns_df.columns[1:],
        value="lcap",
        allow_select_none=False
    )
    asset_input
    return (asset_input,)


@app.cell
def _():
    return


@app.cell
def _(pl, returns_df, stats):
    def _ad_test(asset_class, dist):
        return stats.anderson(
            returns_df[asset_class].to_list(),
            dist=dist,
            method="interpolate"
        )

    def _test_dist(dist):
        results = {}
        for asset_class in returns_df.columns[1:]:
            results[asset_class] = stats.anderson(
                returns_df[asset_class].to_list(),
                dist=dist,
                method="interpolate"
            )
        return pl.DataFrame([
            {
                "asset": aclass,
                "stat": adr.statistic,
                "p_value": adr.pvalue
            }
            for aclass, adr in results.items()
        ])

    _test_dist("logistic")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Best Fit
    According to the Anderson Darling test, the logistic distribution is a better fit for the dataset than the normal distribution. This makes sense because logistic distributions have bigger tails. Logistic distributions have two parameters:
    * location: The mean $\mu$
    * scale: $s$, proportional to the standard deviation, $s = \frac{\sqrt{3}}{\pi}\sigma$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation Matrix
    We'll need a covariance matrix for the retirement simulation project. But we'll make a correlation matrix first so we can see if the results look plausible.
    """)
    return


@app.cell
def _(cs, np, pl, returns_df):
    returns_np = (
        returns_df
        .select(
            cs.by_index(range(1, returns_df.shape[1]))
        )
        .to_numpy()
    )
    cor = np.corrcoef(returns_np, rowvar=False)
    cor_df = pl.DataFrame(cor)
    cor_df.insert_column(0, pl.Series(returns_df.columns[1:]))
    cor_df.columns = ["asset"] + returns_df.columns[1:]
    cor_df.select("asset", pl.all().exclude("asset").round(3))
    return (returns_np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Covariance Matrix
    """)
    return


@app.cell
def _(np, returns_np):
    cov = np.cov(returns_np, rowvar=False)
    cov.shape
    return (cov,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cholesky Decomposition
    For Generating correlated random numbers.
    """)
    return


@app.cell
def _(cov, linalg):
    cholesky_l = linalg.cholesky(cov)
    cholesky_u = linalg.cholesky(cov, upper=True)
    cholesky_u
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
