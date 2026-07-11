"""
optimisation — Module B

Owner: Yusuf
Responsibility: Markowitz efficient frontier, minimum-variance portfolio,
maximum-Sharpe portfolio, and Capital Allocation Line with a risk-free asset.

Public API (import from here in other modules):
    from optimisation import minimum_variance_portfolio, max_sharpe_portfolio
    from optimisation import efficient_frontier, capital_allocation_line
    from optimisation import portfolio_return, portfolio_volatility, sharpe_ratio

Inputs come from data_core:
    mu             = data_core.annualized_return(returns)
    cov_matrix     = data_core.covariance_matrix(returns)
    risk_free_rate = data_core.RISK_FREE_RATE
"""

from optimisation.cal import capital_allocation_line
from optimisation.markowitz import (
    efficient_frontier,
    max_sharpe_portfolio,
    minimum_variance_portfolio,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)

__all__ = [
    "portfolio_return",
    "portfolio_volatility",
    "sharpe_ratio",
    "minimum_variance_portfolio",
    "max_sharpe_portfolio",
    "efficient_frontier",
    "capital_allocation_line",
]
