import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correlated Normal Random Values
    Use the Cholesky decomposition and a covariance matrix to generate correlated random values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test Run
    Generate four sequences of correlated values.

    ### Step 1 - Generate uncorrelated random values
    The random values are from a standard normal distribution. Each row of the numpy array is one series. Each column is one set of measurements.
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng()
    return (rng,)


@app.cell
def _(rng):
    uncor_vals = rng.normal(loc=0, scale=1, size=(4, 100))
    uncor_vals.shape
    return (uncor_vals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, plot the kernal density estimates to verify all four series come from the same probability distribution.
    """)
    return


@app.cell
def _(pl, sns, string, uncor_vals):
    def plot_kde(vals, kind="kde"):
        df = (
            pl.DataFrame(uncor_vals.T)
            .pipe(lambda df: df.rename({col: string.ascii_uppercase[i] for i, col in enumerate(df.columns)}))
            .unpivot(variable_name="series", value_name="value")
        )
        return sns.displot(df, x="value", hue="series", kind=kind)

    plot_kde(uncor_vals)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2 - Generate a Valid Correlation Matrix
    Generating a valid correlation matrix by hand is not easy. The matrix must be positive definite, so we use Scipy to generate one for us.
    """)
    return


@app.cell
def _(rng, stats):
    cor = stats.random_correlation.rvs([0.2, 0.8, 0.9, 2.1], random_state=rng)
    cor
    return (cor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3 - Choose Standard Deviations for Each Variable
    """)
    return


@app.cell
def _(rng):
    sdevs = rng.uniform(0, 5, size=4)
    sdevs
    return (sdevs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4 - Convert the Correlation Matrix to a Covariance Matrix
    """)
    return


@app.cell
def cor_to_cov(cor, sdevs):
    def cor_to_cov(cor_matrix, sdevs):
        sdevs_row = sdevs[None, :]
        sdevs_col = sdevs_row.reshape(-1, 1)
        sdevs_matrix = sdevs_col @ sdevs_row
        return sdevs_matrix * cor_matrix

    cov = cor_to_cov(cor, sdevs)
    cov
    return (cov,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 5 - Take the Cholesky Decomposition of the Covariance Matrix
    """)
    return


@app.cell
def _(cov, np):
    chol = np.linalg.cholesky(cov)
    chol
    return (chol,)


@app.cell
def _(chol, uncor_vals):
    correlated_vals = chol @ uncor_vals
    correlated_vals.shape
    return (correlated_vals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scatter Plots
    """)
    return


@app.cell
def _(correlated_vals, pl, sns):
    def plot_scatter_grid(arr):
        df = pl.DataFrame(arr.T)
        grid = sns.PairGrid(df)
        grid.map_diag(sns.histplot)
        return grid.map_offdiag(sns.scatterplot)

    plot_scatter_grid(correlated_vals)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Correlation Matrix
    """)
    return


@app.cell
def _(cor):
    cor
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Covariance Matrix
    """)
    return


@app.cell
def _(cov):
    cov
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Histogram
    The histogram demonstrates that each series has a different standard deviation, as intended.
    """)
    return


@app.cell
def _(correlated_vals, pl, sns):
    sns.histplot(pl.DataFrame(correlated_vals.T), multiple="dodge")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adjusting the Means
    Adjusting the mean of each series is easy. Just add the mean to each element.
    """)
    return


@app.cell
def _():
    import string

    import marimo as mo
    import numpy as np
    import marimo as mo
    import polars as pl
    from scipy import stats
    import seaborn as sns

    return mo, np, pl, sns, stats, string


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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
