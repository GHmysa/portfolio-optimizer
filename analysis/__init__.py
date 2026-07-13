"""
analysis — Module D

Owner: Emin
Responsibility: quantitative comparison of the optimised portfolio (with cash
allocation via the CAL) against the actual CAC40 index return. Outputs:
return, volatility, and Sharpe side by side; discussion of the cash
allocation's contribution to risk-adjusted performance.

Public API (import from here in other modules):
    from analysis import performance_metrics, portfolio_daily_returns
    from analysis import apply_cash_allocation, comparison_table
"""

from analysis.performance import (
    apply_cash_allocation,
    comparison_table,
    performance_metrics,
    portfolio_daily_returns,
)

__all__ = [
    "performance_metrics",
    "portfolio_daily_returns",
    "apply_cash_allocation",
    "comparison_table",
]
