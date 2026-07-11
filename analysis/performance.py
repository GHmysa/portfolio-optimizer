"""
analysis/performance.py — Module D: performance comparison.

Owner: Emin
Computes realised (annualised return, volatility, Sharpe ratio) for the
optimised portfolio with cash allocation and for the CAC40 benchmark.
"""

import numpy as np
import pandas as pd

from data_core import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def performance_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.Series:
    """
    Realised performance metrics of one daily return series.

    Formulas
    --------
        annualized return:      mu      = mean(r_daily) * 252
        annualized volatility:  sigma   = std(r_daily) * sqrt(252)
        Sharpe ratio:           (mu - r_f) / sigma

    Daily log returns are time-additive, so the annual expected return is
    the daily mean scaled by TRADING_DAYS_PER_YEAR. Variance (not
    volatility) grows linearly with time, hence the square root on 252.

    Parameters
    ----------
    daily_returns  : pd.Series of daily log returns
                     (e.g. from data_core.compute_log_returns).
    risk_free_rate : annual risk-free rate, default data_core.RISK_FREE_RATE.

    Returns
    -------
    pd.Series with index
    ["annualized_return", "annualized_volatility", "sharpe_ratio"].
    """
    mu = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    sigma = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (mu - risk_free_rate) / sigma

    return pd.Series(
        {
            "annualized_return": mu,
            "annualized_volatility": sigma,
            "sharpe_ratio": sharpe,
        }
    )