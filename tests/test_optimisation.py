"""
Tests for Module B (optimisation).

All tests use small synthetic mu / covariance inputs — no network calls,
no dependency on data_core's live fetchers. Run with:

    pytest tests/test_optimisation.py -v
"""

import numpy as np
import pandas as pd
import pytest

from optimisation import (
    capital_allocation_line,
    efficient_frontier,
    max_sharpe_portfolio,
    minimum_variance_portfolio,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)


from data_core import RISK_FREE_RATE


@pytest.fixture
def two_asset_uncorrelated():
    """
    Two uncorrelated assets, asset B twice as volatile as asset A.

    For the uncorrelated minimum-variance case, the analytical optimum is
    inverse-variance weighting: w_i = (1/var_i) / sum(1/var_j).
    With var_A = 0.04 (sigma=20%) and var_B = 0.16 (sigma=40%):
        w_A = (1/0.04) / (1/0.04 + 1/0.16) = 25 / 31.25 = 0.8
        w_B = 0.2
    """
    mu = pd.Series({"A": 0.08, "B": 0.12})
    cov_matrix = pd.DataFrame(
        {"A": [0.04, 0.0], "B": [0.0, 0.16]}, index=["A", "B"]
    )
    return mu, cov_matrix


@pytest.fixture
def three_asset_correlated():
    """Three assets with a non-trivial correlation structure."""
    mu = pd.Series({"A": 0.06, "B": 0.10, "C": 0.14})
    cov_matrix = pd.DataFrame(
        {
            "A": [0.010, 0.002, 0.001],
            "B": [0.002, 0.020, 0.004],
            "C": [0.001, 0.004, 0.040],
        },
        index=["A", "B", "C"],
    )
    return mu, cov_matrix


# ---------------------------------------------------------------------------
# portfolio_return / portfolio_volatility / sharpe_ratio
# ---------------------------------------------------------------------------

class TestPortfolioStatistics:
    def test_portfolio_return_is_weighted_average(self, two_asset_uncorrelated):
        mu, _ = two_asset_uncorrelated
        weights = np.array([0.5, 0.5])
        expected = 0.5 * 0.08 + 0.5 * 0.12
        assert portfolio_return(weights, mu) == pytest.approx(expected)

    def test_portfolio_volatility_uncorrelated_assets(self, two_asset_uncorrelated):
        _, cov_matrix = two_asset_uncorrelated
        weights = np.array([0.8, 0.2])
        # var_p = w_A^2 * var_A + w_B^2 * var_B  (no covariance term, cov=0)
        expected_var = (0.8**2) * 0.04 + (0.2**2) * 0.16
        assert portfolio_volatility(weights, cov_matrix) == pytest.approx(
            np.sqrt(expected_var)
        )

    def test_single_asset_volatility_equals_its_own_sigma(self, two_asset_uncorrelated):
        _, cov_matrix = two_asset_uncorrelated
        weights = np.array([1.0, 0.0])
        assert portfolio_volatility(weights, cov_matrix) == pytest.approx(0.20)

    def test_sharpe_ratio_formula(self, two_asset_uncorrelated):
        mu, cov_matrix = two_asset_uncorrelated
        weights = np.array([0.8, 0.2])
        r_p = portfolio_return(weights, mu)
        sigma_p = portfolio_volatility(weights, cov_matrix)
        expected = (r_p - RISK_FREE_RATE) / sigma_p
        assert sharpe_ratio(weights, mu, cov_matrix, RISK_FREE_RATE) == pytest.approx(
            expected
        )


# ---------------------------------------------------------------------------
# minimum_variance_portfolio
# ---------------------------------------------------------------------------

class TestMinimumVariancePortfolio:
    def test_weights_sum_to_one(self, three_asset_correlated):
        mu, cov_matrix = three_asset_correlated
        weights, _, _ = minimum_variance_portfolio(mu, cov_matrix)
        assert weights.sum() == pytest.approx(1.0)

    def test_weights_are_long_only(self, three_asset_correlated):
        mu, cov_matrix = three_asset_correlated
        weights, _, _ = minimum_variance_portfolio(mu, cov_matrix)
        assert (weights >= -1e-8).all()

    def test_matches_analytical_inverse_variance_solution(self, two_asset_uncorrelated):
        """For uncorrelated assets, MVP weights are exactly inverse-variance."""
        mu, cov_matrix = two_asset_uncorrelated
        weights, _, _ = minimum_variance_portfolio(mu, cov_matrix)
        assert weights["A"] == pytest.approx(0.8, abs=1e-3)
        assert weights["B"] == pytest.approx(0.2, abs=1e-3)

    def test_volatility_is_no_larger_than_any_single_asset(self, three_asset_correlated):
        """The whole point of diversification: MVP sigma <= min individual sigma."""
        mu, cov_matrix = three_asset_correlated
        _, _, mvp_volatility = minimum_variance_portfolio(mu, cov_matrix)
        individual_sigmas = np.sqrt(np.diag(cov_matrix))
        assert mvp_volatility <= individual_sigmas.min() + 1e-8


# ---------------------------------------------------------------------------
# max_sharpe_portfolio
# ---------------------------------------------------------------------------

class TestMaxSharpePortfolio:
    def test_weights_sum_to_one(self, three_asset_correlated):
        mu, cov_matrix = three_asset_correlated
        weights, _, _, _ = max_sharpe_portfolio(mu, cov_matrix, RISK_FREE_RATE)
        assert weights.sum() == pytest.approx(1.0)

    def test_weights_are_long_only(self, three_asset_correlated):
        mu, cov_matrix = three_asset_correlated
        weights, _, _, _ = max_sharpe_portfolio(mu, cov_matrix, RISK_FREE_RATE)
        assert (weights >= -1e-8).all()

    def test_sharpe_is_at_least_as_good_as_min_variance_portfolio(
        self, three_asset_correlated
    ):
        mu, cov_matrix = three_asset_correlated
        _, mvp_return, mvp_vol = minimum_variance_portfolio(mu, cov_matrix)
        mvp_sharpe = (mvp_return - RISK_FREE_RATE) / mvp_vol

        _, _, _, max_sharpe = max_sharpe_portfolio(mu, cov_matrix, RISK_FREE_RATE)
        assert max_sharpe >= mvp_sharpe - 1e-6

    def test_sharpe_value_matches_reported_return_and_volatility(
        self, three_asset_correlated
    ):
        mu, cov_matrix = three_asset_correlated
        _, r_p, sigma_p, sharpe = max_sharpe_portfolio(mu, cov_matrix, RISK_FREE_RATE)
        assert sharpe == pytest.approx((r_p - RISK_FREE_RATE) / sigma_p)


# ---------------------------------------------------------------------------
# efficient_frontier
# ---------------------------------------------------------------------------

class TestEfficientFrontier:
    def test_output_shapes(self, three_asset_correlated):
        mu, cov_matrix = three_asset_correlated
        n_points = 10
        returns, volatilities, weights = efficient_frontier(mu, cov_matrix, n_points)
        assert returns.shape == (n_points,)
        assert volatilities.shape == (n_points,)
        assert weights.shape == (n_points, len(mu))

    def test_every_frontier_point_is_long_only_and_fully_invested(
        self, three_asset_correlated
    ):
        mu, cov_matrix = three_asset_correlated
        _, _, weights = efficient_frontier(mu, cov_matrix, n_points=10)
        assert (weights >= -1e-8).all()
        assert weights.sum(axis=1) == pytest.approx(np.ones(10))

    def test_returns_are_non_decreasing_along_the_frontier(self, three_asset_correlated):
        mu, cov_matrix = three_asset_correlated
        returns, _, _ = efficient_frontier(mu, cov_matrix, n_points=10)
        assert np.all(np.diff(returns) >= -1e-8)

    def test_first_point_matches_minimum_variance_portfolio(self, three_asset_correlated):
        mu, cov_matrix = three_asset_correlated
        _, mvp_return, mvp_vol = minimum_variance_portfolio(mu, cov_matrix)
        returns, volatilities, _ = efficient_frontier(mu, cov_matrix, n_points=10)
        assert returns[0] == pytest.approx(mvp_return, abs=1e-3)
        assert volatilities[0] == pytest.approx(mvp_vol, abs=1e-3)


# ---------------------------------------------------------------------------
# capital_allocation_line
# ---------------------------------------------------------------------------

class TestCapitalAllocationLine:
    def test_zero_target_volatility_is_all_cash(self):
        risky_w, cash_w, expected_return = capital_allocation_line(
            target_volatility=0.0,
            tangency_return=0.15,
            tangency_volatility=0.25,
            risk_free_rate=RISK_FREE_RATE,
        )
        assert risky_w == pytest.approx(0.0)
        assert cash_w == pytest.approx(1.0)
        assert expected_return == pytest.approx(RISK_FREE_RATE)

    def test_target_volatility_equal_to_tangency_is_all_risky(self):
        risky_w, cash_w, expected_return = capital_allocation_line(
            target_volatility=0.25,
            tangency_return=0.15,
            tangency_volatility=0.25,
            risk_free_rate=RISK_FREE_RATE,
        )
        assert risky_w == pytest.approx(1.0)
        assert cash_w == pytest.approx(0.0)
        assert expected_return == pytest.approx(0.15)

    def test_target_volatility_above_tangency_is_capped_long_only(self):
        """No leverage: exceeding tangency sigma still caps risky weight at 1."""
        risky_w, cash_w, _ = capital_allocation_line(
            target_volatility=0.50,
            tangency_return=0.15,
            tangency_volatility=0.25,
            risk_free_rate=RISK_FREE_RATE,
        )
        assert risky_w == pytest.approx(1.0)
        assert cash_w == pytest.approx(0.0)

    def test_weights_sum_to_one(self):
        risky_w, cash_w, _ = capital_allocation_line(
            target_volatility=0.10,
            tangency_return=0.15,
            tangency_volatility=0.25,
            risk_free_rate=RISK_FREE_RATE,
        )
        assert risky_w + cash_w == pytest.approx(1.0)

    def test_expected_return_is_on_the_cal_line(self):
        """R_c = r_f + Sharpe * sigma_c must hold exactly at the target sigma."""
        tangency_return, tangency_vol = 0.15, 0.25
        target_vol = 0.10
        _, _, expected_return = capital_allocation_line(
            target_volatility=target_vol,
            tangency_return=tangency_return,
            tangency_volatility=tangency_vol,
            risk_free_rate=RISK_FREE_RATE,
        )
        slope = (tangency_return - RISK_FREE_RATE) / tangency_vol
        assert expected_return == pytest.approx(RISK_FREE_RATE + slope * target_vol)
