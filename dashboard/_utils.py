from typing import Dict, Iterable, Tuple
import pandas as pd
import numpy as np


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Convert a returns series (log or simple) into cumulative return series.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns (either log-returns or simple returns). If DataFrame,
        the operation is applied column-wise.

    Returns
    -------
    pd.Series or pd.DataFrame
        Cumulative growth series where the first value is the period starting
        level (1.0 = start).

    Notes
    -----
    Detects log-returns heuristically: if absolute values are small (<0.5),
    treats as log-returns and uses exp(cumsum); otherwise uses (1 + r).cumprod().
    """
    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            return returns.copy()
        # heuristic: if all absolute values < 0.5 treat as log-returns
        if (returns.abs().max() < 0.5).all():
            return np.exp(returns).cumprod()
        return (1 + returns).cumprod()

    # Series
    if returns.empty:
        return returns.copy()
    if abs(returns).max() < 0.5:
        return np.exp(returns).cumprod()
    return (1 + returns).cumprod()


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Convert a returns series (log or simple) into cumulative return series.

    Returns index is preserved. The result is a decimal cumulative growth (1.0 = no change).
    """
    # If values look like log returns (small), convert via exp
    if (returns.abs().max() < 0.5).all() if isinstance(returns, pd.DataFrame) else abs(returns).max() < 0.5:
        cum = np.exp(returns).cumprod()
    else:
        cum = (1 + returns).cumprod()
    # If DataFrame, return DataFrame, else Series
    return cum


def weights_to_series(weights: Iterable[float], tickers: Iterable[str]) -> pd.Series:
    """Return a pandas Series of weights indexed by tickers.

    Ensures alignment and sorts by weight descending for plotting convenience.

    Parameters
    ----------
    weights : Iterable[float]
        Sequence of weight values.
    tickers : Iterable[str]
        Sequence of ticker strings to use as index.

    Returns
    -------
    pd.Series
        Series indexed by tickers with weights, sorted descending.
    """
    s = pd.Series(data=list(weights), index=list(tickers), dtype=float)
    s.index.name = "ticker"
    return s.sort_values(ascending=False)

