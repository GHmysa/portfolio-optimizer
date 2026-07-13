"""
Tests for Module D (analysis/performance.py).

Synthetic data only — no network, fast. Run with:
    pytest tests/test_analysis.py -v
"""

import numpy as np
import pandas as pd
import pytest

from data_core import TRADING_DAYS_PER_YEAR
from analysis.performance import (
    apply_cash_allocation,
    comparison_table,
    performance_metrics,
    portfolio_daily_returns,
)


def sample_returns() -> pd.Series:
    """Small deterministic daily-return series used across tests."""
    return pd.Series([0.01, -0.02, 0.015, 0.0, 0.005])


class TestPerformanceMetrics:
    def test_annualized_return_scales_daily_mean(self):
        """Two days at +2 % and 0 % -> mean 1 % -> 1 % x 252 per year."""
        returns = pd.Series([0.02, 0.00])
        metrics = performance_metrics(returns)
        assert np.isclose(metrics["annualized_return"], 0.01 * TRADING_DAYS_PER_YEAR)

    def test_annualized_volatility_scales_with_sqrt_time(self):
        """Same series: sample std is 0.01*sqrt(2), annualised via sqrt(252)."""
        returns = pd.Series([0.02, 0.00])
        expected = 0.01 * np.sqrt(2) * np.sqrt(TRADING_DAYS_PER_YEAR)
        metrics = performance_metrics(returns)
        assert np.isclose(metrics["annualized_volatility"], expected)

    def test_sharpe_is_excess_return_over_volatility(self):
        """With r_f = 0 the Sharpe ratio must equal mu / sigma."""
        metrics = performance_metrics(sample_returns(), risk_free_rate=0.0)
        expected = metrics["annualized_return"] / metrics["annualized_volatility"]
        assert np.isclose(metrics["sharpe_ratio"], expected)


class TestPortfolioDailyReturns:
    def test_weighted_average_two_assets(self):
        asset_returns = pd.DataFrame({"AAA": [0.01, 0.02], "BBB": [0.03, -0.02]})
        weights = pd.Series({"AAA": 0.6, "BBB": 0.4})
        rp = portfolio_daily_returns(asset_returns, weights)
        assert np.isclose(rp.iloc[0], 0.6 * 0.01 + 0.4 * 0.03)
        assert np.isclose(rp.iloc[1], 0.6 * 0.02 + 0.4 * (-0.02))

    def test_weight_order_does_not_matter(self):
        """Weights are matched by ticker name, not by column position."""
        asset_returns = pd.DataFrame({"AAA": [0.01], "BBB": [0.03]})
        weights_reversed = pd.Series({"BBB": 0.4, "AAA": 0.6})
        rp = portfolio_daily_returns(asset_returns, weights_reversed)
        assert np.isclose(rp.iloc[0], 0.6 * 0.01 + 0.4 * 0.03)

    def test_missing_weight_raises(self):
        asset_returns = pd.DataFrame({"AAA": [0.01], "BBB": [0.03]})
        with pytest.raises(ValueError):
            portfolio_daily_returns(asset_returns, pd.Series({"AAA": 1.0}))


class TestApplyCashAllocation:
    def test_fully_invested_is_unchanged(self):
        rp = sample_returns()
        rc = apply_cash_allocation(rp, risky_weight=1.0)
        assert np.allclose(rc, rp)

    def test_all_cash_earns_exactly_the_daily_risk_free_rate(self):
        """r_f = 2.52 % p.a. -> exactly 0.01 % on each of the 252 days."""
        rc = apply_cash_allocation(sample_returns(), 0.0, risk_free_rate=0.0252)
        assert np.allclose(rc, 0.0001)

    def test_volatility_scales_linearly_with_risky_weight(self):
        rp = sample_returns()
        rc = apply_cash_allocation(rp, risky_weight=0.5)
        assert np.isclose(rc.std(), 0.5 * rp.std())

    def test_sharpe_ratio_is_constant_along_the_cal(self):
        """The core CAL property Module D's discussion is built on."""
        rp = sample_returns()
        rc = apply_cash_allocation(rp, 0.5, risk_free_rate=0.02)
        sharpe_full = performance_metrics(rp, risk_free_rate=0.02)["sharpe_ratio"]
        sharpe_cash = performance_metrics(rc, risk_free_rate=0.02)["sharpe_ratio"]
        assert np.isclose(sharpe_full, sharpe_cash)

    def test_risky_weight_outside_unit_interval_raises(self):
        with pytest.raises(ValueError):
            apply_cash_allocation(sample_returns(), risky_weight=1.2)


class TestComparisonTable:
    def test_one_row_per_series_with_metric_columns(self):
        table = comparison_table({"a": sample_returns(), "b": sample_returns()})
        assert list(table.index) == ["a", "b"]
        assert list(table.columns) == [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
        ]

    def test_rows_match_performance_metrics(self):
        rp = sample_returns()
        table = comparison_table({"a": rp})
        expected = performance_metrics(rp)
        assert np.allclose(table.loc["a"], expected)