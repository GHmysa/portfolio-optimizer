"""
optimisation/cal.py — Capital Allocation Line (CAL).

Owner: Yusuf
Responsibility: split an investment between the risk-free asset (cash)
and the max-Sharpe ("tangency") risky portfolio to hit a target volatility.

The CAL is the straight line, in (volatility, return) space, that starts
at the risk-free rate (sigma = 0) and passes through the tangency
portfolio found by max_sharpe_portfolio(). It dominates every other
combination of risky assets + cash because the tangency portfolio has
the highest Sharpe ratio available.
"""


def capital_allocation_line(
    target_volatility: float,
    tangency_return: float,
    tangency_volatility: float,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    """
    Split cash and the tangency (max-Sharpe) portfolio to hit a target risk.

    Formula
    -------
    The CAL is a straight line: R_c = r_f + Sharpe * sigma_c, where
    Sharpe = (R_tangency - r_f) / sigma_tangency is the tangency
    portfolio's Sharpe ratio (the CAL's slope).

    Because volatility scales linearly with the share invested in the
    risky portfolio (cash has zero volatility and is uncorrelated with
    itself), reaching a target volatility only requires:

        w_risky = target_volatility / sigma_tangency
        w_cash  = 1 - w_risky

    Long-only note: w_risky is capped at 1 (no borrowing to leverage the
    risky portfolio beyond 100 %), consistent with the long-only
    constraint used everywhere else in this module.

    Parameters
    ----------
    target_volatility    — desired portfolio volatility (sigma_c), same
                            units as tangency_volatility (annualised).
    tangency_return       — R_p of the max-Sharpe portfolio.
    tangency_volatility   — sigma_p of the max-Sharpe portfolio.
    risk_free_rate        — r_f, e.g. data_core.RISK_FREE_RATE.

    Returns
    -------
    (risky_weight, cash_weight, expected_return)
        risky_weight    — fraction of the portfolio in the tangency
                          portfolio, in [0, 1]
        cash_weight     — fraction of the portfolio in cash, in [0, 1]
        expected_return — R_c at the achieved (possibly capped) risk
    """
    risky_weight = target_volatility / tangency_volatility
    risky_weight = min(risky_weight, 1.0)
    cash_weight = 1.0 - risky_weight

    expected_return = risk_free_rate + risky_weight * (tangency_return - risk_free_rate)

    return risky_weight, cash_weight, expected_return
