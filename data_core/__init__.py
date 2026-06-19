"""
data_core — Module A

Owner: Maxime
Responsibility: data ingestion, return computation, and descriptive statistics.

Public API (import from here in other modules):
    from data_core import fetch_prices, fetch_benchmark
    from data_core import compute_log_returns, annualized_return
    from data_core import annualized_volatility, covariance_matrix
    from data_core import statistical_moments
    from data_core import get_cac40_tickers, get_cac40_weights
    from data_core import BENCHMARK_TICKER, RISK_FREE_RATE
"""

from data_core.constituents import (
    BENCHMARK_TICKER,
    RISK_FREE_RATE,
    get_cac40_names,
    get_cac40_tickers,
    get_cac40_weights,
    get_estimated_tickers,
)
from data_core.fetcher import fetch_benchmark, fetch_prices
from data_core.returns import (
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    covariance_matrix,
    compute_log_returns,
    statistical_moments,
)

__all__ = [
    "BENCHMARK_TICKER",
    "RISK_FREE_RATE",
    "TRADING_DAYS_PER_YEAR",
    "get_cac40_names",
    "get_cac40_tickers",
    "get_cac40_weights",
    "get_estimated_tickers",
    "fetch_prices",
    "fetch_benchmark",
    "compute_log_returns",
    "annualized_return",
    "annualized_volatility",
    "covariance_matrix",
    "statistical_moments",
]
