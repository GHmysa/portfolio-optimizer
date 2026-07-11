"""
optimisation/markowitz.py — Module B core maths.

Owner: Yusuf
Responsibility: mean-variance (Markowitz) portfolio optimisation.

Inputs are the annualised outputs of data_core:
    mu          — pd.Series, annualised expected return per asset
                  (from data_core.annualized_return)
    cov_matrix  — pd.DataFrame, annualised covariance matrix Sigma
                  (from data_core.covariance_matrix)

Constraints used throughout (per CLAUDE.md):
    * long-only: every weight w_i >= 0 (no short-selling)
    * fully invested: sum(w_i) = 1
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Basic portfolio statistics
# ---------------------------------------------------------------------------

def portfolio_return(weights: np.ndarray, mu: pd.Series) -> float:
    """
    Expected annualised return of a portfolio.

    Formula
    -------
        R_p = w^T . mu = sum_i( w_i * mu_i )
    """
    return float(np.dot(weights, mu))


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """
    Expected annualised volatility (standard deviation) of a portfolio.

    Formula
    -------
        sigma_p = sqrt( w^T . Sigma . w )

    where Sigma is the annualised covariance matrix. The w^T . Sigma . w
    term is the portfolio variance; it accounts for both individual asset
    variances (diagonal of Sigma) and pairwise covariances (off-diagonal),
    which is exactly where diversification benefit comes from.
    """
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return float(np.sqrt(variance))


def sharpe_ratio(
    weights: np.ndarray,
    mu: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """
    Sharpe ratio of a portfolio: excess return per unit of risk.

    Formula
    -------
        Sharpe = (R_p - r_f) / sigma_p
    """
    r_p = portfolio_return(weights, mu)
    sigma_p = portfolio_volatility(weights, cov_matrix)
    return (r_p - risk_free_rate) / sigma_p


# ---------------------------------------------------------------------------
# Shared optimiser setup
# ---------------------------------------------------------------------------

def _long_only_fully_invested_constraints(n_assets: int) -> tuple[dict, tuple]:
    """
    Build the constraint set shared by every optimisation in this module.

    * bounds: 0 <= w_i <= 1 for every asset            (long-only)
    * constraint: sum(w_i) - 1 = 0                      (fully invested)
    """
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    sum_to_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    return sum_to_one, bounds


def _equal_weights_starting_point(n_assets: int) -> np.ndarray:
    """Equal-weight portfolio (1/n each) used as the SLSQP starting guess."""
    return np.full(n_assets, 1.0 / n_assets)


# ---------------------------------------------------------------------------
# Minimum-variance portfolio
# ---------------------------------------------------------------------------

def minimum_variance_portfolio(
    mu: pd.Series, cov_matrix: pd.DataFrame
) -> tuple[pd.Series, float, float]:
    """
    Find the long-only portfolio with the lowest possible volatility.

    This is the left-most point of the efficient frontier: no other
    long-only portfolio has a smaller sigma_p, regardless of its return.

    Solves
    ------
        minimise    w^T . Sigma . w
        subject to  sum(w_i) = 1,  w_i >= 0

    Returns
    -------
    (weights, expected_return, volatility)
        weights          — pd.Series indexed by ticker, sums to 1
        expected_return  — R_p at the found weights
        volatility       — sigma_p at the found weights
    """
    n_assets = len(mu)
    constraints, bounds = _long_only_fully_invested_constraints(n_assets)
    initial_weights = _equal_weights_starting_point(n_assets)

    result = minimize(
        fun=lambda w: portfolio_volatility(w, cov_matrix),
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance optimisation failed: {result.message}")

    weights = pd.Series(result.x, index=mu.index)
    r_p = portfolio_return(result.x, mu)
    sigma_p = portfolio_volatility(result.x, cov_matrix)
    return weights, r_p, sigma_p


# ---------------------------------------------------------------------------
# Maximum-Sharpe (tangency) portfolio
# ---------------------------------------------------------------------------

def max_sharpe_portfolio(
    mu: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float
) -> tuple[pd.Series, float, float, float]:
    """
    Find the long-only portfolio that maximises the Sharpe ratio.

    This is the "tangency portfolio": the point on the efficient frontier
    where a line drawn from (0, risk_free_rate) is tangent to the curve.
    It is the single best risky portfolio to combine with cash — see
    capital_allocation_line().

    Solves
    ------
        maximise    (w^T . mu - r_f) / sqrt(w^T . Sigma . w)
        subject to  sum(w_i) = 1,  w_i >= 0

    scipy.optimize.minimize only minimises, so we minimise the negative
    Sharpe ratio instead.

    Returns
    -------
    (weights, expected_return, volatility, sharpe)
    """
    n_assets = len(mu)
    constraints, bounds = _long_only_fully_invested_constraints(n_assets)
    initial_weights = _equal_weights_starting_point(n_assets)

    def negative_sharpe(w: np.ndarray) -> float:
        return -sharpe_ratio(w, mu, cov_matrix, risk_free_rate)

    result = minimize(
        fun=negative_sharpe,
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimisation failed: {result.message}")

    weights = pd.Series(result.x, index=mu.index)
    r_p = portfolio_return(result.x, mu)
    sigma_p = portfolio_volatility(result.x, cov_matrix)
    sharpe = sharpe_ratio(result.x, mu, cov_matrix, risk_free_rate)
    return weights, r_p, sigma_p, sharpe


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------

def efficient_frontier(
    mu: pd.Series, cov_matrix: pd.DataFrame, n_points: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trace the long-only efficient frontier.

    For each target return between the minimum-variance portfolio's return
    and the highest achievable single-asset return, find the portfolio
    with the lowest possible volatility that achieves at least that return.

    Solves, for each target return R_target
    -----------------------------------------
        minimise    w^T . Sigma . w
        subject to  sum(w_i) = 1,  w_i >= 0,  w^T . mu = R_target

    Returns
    -------
    (returns, volatilities, weights)
        returns       — np.ndarray, shape (n_points,)
        volatilities  — np.ndarray, shape (n_points,)
        weights       — np.ndarray, shape (n_points, n_assets)
    """
    n_assets = len(mu)
    _, min_var_return, _ = minimum_variance_portfolio(mu, cov_matrix)

    # The frontier spans from the min-variance portfolio's return up to the
    # highest expected return among individual assets (going higher than
    # that is infeasible under long-only + fully-invested constraints).
    max_return = mu.max()
    target_returns = np.linspace(min_var_return, max_return, n_points)

    returns = np.zeros(n_points)
    volatilities = np.zeros(n_points)
    weights_grid = np.zeros((n_points, n_assets))

    initial_weights = _equal_weights_starting_point(n_assets)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))

    for i, target in enumerate(target_returns):
        sum_to_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        hits_target_return = {
            "type": "eq",
            "fun": lambda w, target=target: portfolio_return(w, mu) - target,
        }

        result = minimize(
            fun=lambda w: portfolio_volatility(w, cov_matrix),
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[sum_to_one, hits_target_return],
        )
        if not result.success:
            raise RuntimeError(
                f"Efficient frontier optimisation failed at target return "
                f"{target:.4f}: {result.message}"
            )

        returns[i] = portfolio_return(result.x, mu)
        volatilities[i] = portfolio_volatility(result.x, cov_matrix)
        weights_grid[i] = result.x

    return returns, volatilities, weights_grid
