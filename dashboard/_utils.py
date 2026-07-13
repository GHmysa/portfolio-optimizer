from typing import Dict, Iterable, Tuple
import pandas as pd
import numpy as np

from data_core import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


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


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.Series:
    """
    Rolling annualised Sharpe ratio of a daily log-return series.

    Formula (for each date t, over the preceding `window` trading days):
        rolling_sharpe_t = (mean(r_{t-window:t}) * 252 - r_f)
                           / (std(r_{t-window:t}) * sqrt(252))

    The first `window - 1` values are NaN because the rolling window is not
    yet full. With the default window of 252, the chart starts approximately
    one year after the series begins — this is intentional and honest.

    Parameters
    ----------
    returns       : daily log-returns (pd.Series).
    window        : rolling window size in trading days (default 252 ≈ 1 year).
    risk_free_rate: annualised risk-free rate (default data_core.RISK_FREE_RATE).

    Returns
    -------
    pd.Series with the same index as `returns`. First window-1 values are NaN.
    """
    mean_ann = returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    vol_ann = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (mean_ann - risk_free_rate) / vol_ann


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

