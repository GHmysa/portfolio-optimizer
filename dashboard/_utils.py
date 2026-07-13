from typing import Dict, Iterable, Tuple
import pandas as pd
import numpy as np


def compute_cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Convert a series of daily log-returns into a cumulative growth index.

    Formula
    -------
        cumulative_t = exp( sum_{i=1}^{t} r_i )
                     = prod_{i=1}^{t} exp(r_i)

    A value of 1.0 means no change; 1.10 means +10 % since the start.

    Parameters
    ----------
    returns : daily log-returns from data_core.compute_log_returns().
              Series (single asset) or DataFrame (multiple assets, column-wise).

    Returns
    -------
    Same shape as input — cumulative growth factor at each date.
    """
    return np.exp(returns).cumprod()


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

