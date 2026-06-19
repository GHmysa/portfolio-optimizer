"""
Tests for Module A (data_core).

Run with:  pytest tests/test_data_core.py -v

These tests use a tiny set of tickers and a short date range to avoid slow
network calls in CI. The full dataset fetch is an integration test that
should be run manually when working on Module A.
"""

import numpy as np
import pandas as pd
import pytest

from data_core.constituents import (
    get_cac40_tickers,
    get_cac40_weights,
    get_cac40_names,
    get_estimated_tickers,
    RISK_FREE_RATE,
)
from data_core.returns import (
    compute_log_returns,
    annualized_return,
    annualized_volatility,
    covariance_matrix,
    statistical_moments,
    TRADING_DAYS_PER_YEAR,
)


# ---------------------------------------------------------------------------
# constituents.py tests
# ---------------------------------------------------------------------------

class TestConstituents:
    def test_ticker_count(self):
        """We have 39 confirmed tickers (40th is unresolved — see TODO)."""
        assert len(get_cac40_tickers()) == 39

    def test_weights_sum_to_100(self):
        """Weights must sum to exactly 100 % (within floating-point tolerance)."""
        total = sum(get_cac40_weights().values())
        assert abs(total - 100.0) < 0.01, f"Weights sum to {total:.4f}, expected 100.0"

    def test_every_ticker_has_a_name(self):
        names = get_cac40_names()
        for ticker in get_cac40_tickers():
            assert ticker in names, f"{ticker} has no name mapping"

    def test_every_ticker_has_a_weight(self):
        weights = get_cac40_weights()
        for ticker in get_cac40_tickers():
            assert ticker in weights, f"{ticker} has no weight"

    def test_all_weights_positive(self):
        for ticker, w in get_cac40_weights().items():
            assert w > 0, f"{ticker} has non-positive weight {w}"

    def test_risk_free_rate_reasonable(self):
        """Sanity check: rate should be between 0 % and 10 %."""
        assert 0.0 < RISK_FREE_RATE < 0.10

    def test_estimated_tickers_are_subset(self):
        all_tickers = set(get_cac40_tickers())
        assert get_estimated_tickers().issubset(all_tickers)


# ---------------------------------------------------------------------------
# returns.py tests  (use synthetic prices, no network required)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """
    Two fake price series that let us verify the maths by hand.
    Stock A: constant +1 % daily (deterministic).
    Stock B: constant -0.5 % daily.
    """
    dates = pd.date_range("2020-01-02", periods=253, freq="B")
    price_a = 100 * (1.01 ** np.arange(253))
    price_b = 100 * (0.995 ** np.arange(253))
    return pd.DataFrame({"A": price_a, "B": price_b}, index=dates)


class TestReturns:
    def test_log_returns_shape(self, synthetic_prices):
        """Log returns drop the first row (no previous price for day 0)."""
        returns = compute_log_returns(synthetic_prices)
        assert returns.shape == (252, 2)

    def test_log_returns_values(self, synthetic_prices):
        """
        For price_a = 100 * 1.01^t, the daily log return is ln(1.01) ≈ 0.00995.
        """
        returns = compute_log_returns(synthetic_prices)
        expected = np.log(1.01)
        assert abs(returns["A"].mean() - expected) < 1e-10

    def test_annualized_return(self, synthetic_prices):
        """
        Daily log return of ln(1.01) annualises to ln(1.01) × 252 ≈ 2.507.
        (A stock gaining 1 % every trading day more than doubles in a year.)
        """
        returns = compute_log_returns(synthetic_prices)
        mu = annualized_return(returns)
        assert abs(mu["A"] - np.log(1.01) * TRADING_DAYS_PER_YEAR) < 1e-8

    def test_covariance_matrix_is_symmetric(self, synthetic_prices):
        returns = compute_log_returns(synthetic_prices)
        cov = covariance_matrix(returns)
        assert cov.shape == (2, 2)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_covariance_diagonal_equals_variance(self, synthetic_prices):
        """
        The diagonal of the covariance matrix should equal σ² (annualised).
        Since our synthetic prices are deterministic there is zero variance —
        this just checks the formula structure holds (Cov(i,i) = Var(i)).
        """
        returns = compute_log_returns(synthetic_prices)
        cov = covariance_matrix(returns)
        vol = annualized_volatility(returns)
        for col in returns.columns:
            # σ² = Cov(i,i)
            assert abs(cov.loc[col, col] - vol[col] ** 2) < 1e-10

    def test_statistical_moments_shape(self, synthetic_prices):
        returns = compute_log_returns(synthetic_prices)
        moments = statistical_moments(returns)
        assert moments.shape == (4, 2)
        assert list(moments.index) == [
            "mean_annual", "volatility_annual", "skewness", "excess_kurtosis"
        ]


# ---------------------------------------------------------------------------
# Integration test — skipped by default, run manually with -m integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_fetch_prices_live():
    """Download a tiny 3-ticker, 5-day slice to confirm yfinance works."""
    from data_core.fetcher import fetch_prices
    prices = fetch_prices(
        tickers=["TTE.PA", "MC.PA", "AI.PA"],
        start="2024-01-02",
        end="2024-01-10",
    )
    assert not prices.empty
    assert prices.shape[1] <= 3   # may be fewer if one ticker is unavailable
    assert prices.isna().sum().sum() == 0


@pytest.mark.integration
def test_fetch_benchmark_live():
    """Download ^FCHI for one week."""
    from data_core.fetcher import fetch_benchmark
    bench = fetch_benchmark(start="2024-01-02", end="2024-01-10")
    assert not bench.empty
    assert bench.name == "CAC40"
