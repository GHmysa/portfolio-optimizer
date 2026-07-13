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


def portfolio_daily_returns(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Daily returns of a portfolio with fixed weights.

    Formula
    -------
        r_p,t = sum_i( w_i * r_i,t )        (one dot product per day)

    The portfolio return on each day is the weighted average of that
    day's asset returns. Keeping the weights fixed corresponds to daily
    rebalancing back to the target allocation — the standard
    simplification, consistent with Modules A and B.

    Weights are aligned to the columns of `asset_returns` by ticker, so
    column order does not matter. Raises if any ticker has no weight.
    """
    aligned_weights = weights.reindex(asset_returns.columns)
    if aligned_weights.isna().any():
        missing = list(aligned_weights[aligned_weights.isna()].index)
        raise ValueError(f"No weight provided for: {missing}")

    portfolio_returns = asset_returns.dot(aligned_weights)
    portfolio_returns.name = "portfolio"
    return portfolio_returns


def apply_cash_allocation(
    portfolio_returns: pd.Series,
    risky_weight: float,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.Series:
    """
    Blend the risky portfolio with cash (the risk-free asset).

    Formula
    -------
        r_c,t = w_risky * r_p,t + (1 - w_risky) * (r_f / 252)

    Cash earns the same small return every day and has zero volatility,
    so mixing in cash scales both the excess return and the risk by
    w_risky — which is exactly why the Sharpe ratio stays constant
    along the Capital Allocation Line.

    risky_weight must lie in [0, 1]: 1 = fully invested, 0 = all cash
    (long-only, no leverage — consistent with Module B).
    """
    if not 0.0 <= risky_weight <= 1.0:
        raise ValueError(f"risky_weight must be in [0, 1], got {risky_weight}")

    daily_risk_free = risk_free_rate / TRADING_DAYS_PER_YEAR
    return risky_weight * portfolio_returns + (1.0 - risky_weight) * daily_risk_free

def comparison_table(
    named_returns: dict[str, pd.Series],
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Side-by-side performance metrics for several return series.

    One row per series, computed with performance_metrics(). This
    produces the Module D deliverable per CLAUDE.md: the optimised
    portfolio (with cash) next to the CAC40 benchmark.

    Returns
    -------
    pd.DataFrame — rows: series names; columns:
    annualized_return, annualized_volatility, sharpe_ratio.
    """
    rows = {
        name: performance_metrics(series, risk_free_rate)
        for name, series in named_returns.items()
    }
    return pd.DataFrame(rows).T

