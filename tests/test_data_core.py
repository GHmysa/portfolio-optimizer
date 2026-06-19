"""
Tests for Module A (data_core).

Run with:  pytest tests/test_data_core.py -v
Network tests require:  pytest tests/test_data_core.py -v -m integration
"""

import numpy as np
import pandas as pd
import pytest

from data_core.constituents import (
    BENCHMARK_TICKER,
    RISK_FREE_RATE,
    get_cac40_names,
    get_cac40_sectors,
    get_cac40_tickers,
    load_reference_weights,
)
from data_core.returns import (
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    covariance_matrix,
    compute_log_returns,
    statistical_moments,
)


# ---------------------------------------------------------------------------
# constituents.py tests
# ---------------------------------------------------------------------------

class TestConstituents:
    def test_ticker_count(self):
        """Must be exactly 40 — catches any future copy-paste omission."""
        assert len(get_cac40_tickers()) == 40

    def test_no_duplicate_tickers(self):
        tickers = get_cac40_tickers()
        assert len(tickers) == len(set(tickers)), "Duplicate tickers found"

    def test_every_ticker_has_a_name(self):
        names = get_cac40_names()
        for ticker in get_cac40_tickers():
            assert ticker in names, f"{ticker} has no name mapping"

    def test_every_ticker_has_a_sector(self):
        sectors = get_cac40_sectors()
        for ticker in get_cac40_tickers():
            assert ticker in sectors, f"{ticker} has no sector"

    def test_axa_present(self):
        """AXA (CS.PA) was the historically missing 40th entry — pin it."""
        assert "CS.PA" in get_cac40_tickers()

    def test_arcelormittal_uses_amsterdam_ticker(self):
        """ArcelorMittal must use MT.AS, not MT.PA (Amsterdam primary on Yahoo)."""
        assert "MT.AS" in get_cac40_tickers()
        assert "MT.PA" not in get_cac40_tickers()

    def test_risk_free_rate_reasonable(self):
        assert 0.0 < RISK_FREE_RATE < 0.10

    def test_benchmark_ticker(self):
        assert BENCHMARK_TICKER == "^FCHI"

    def test_reference_weights_has_25_rows(self):
        """The public Euronext PDF disclosed exactly 25 of 40 constituents."""
        ref = load_reference_weights()
        assert len(ref) == 25

    def test_reference_weights_columns(self):
        ref = load_reference_weights()
        assert set(ref.columns) == {"ticker", "mnemo", "company_name", "weight_pct"}

    def test_reference_weights_top25_sum(self):
        """25 official Euronext weights must sum to ~90.57% (from PDF)."""
        ref = load_reference_weights()
        total = ref["weight_pct"].sum()
        assert abs(total - 90.57) < 0.05, f"Top-25 sum = {total:.2f}%, expected ~90.57%"

    def test_reference_weights_all_positive(self):
        ref = load_reference_weights()
        assert (ref["weight_pct"] > 0).all()

    def test_reference_weights_tickers_in_full_list(self):
        """Every ticker with a reference weight must also be in the 40-stock list."""
        all_tickers = set(get_cac40_tickers())
        for ticker in load_reference_weights()["ticker"]:
            assert ticker in all_tickers, f"{ticker} in weights CSV but not in tickers CSV"


# ---------------------------------------------------------------------------
# returns.py tests  (synthetic prices — no network required)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """
    Two deterministic price series for formula verification.
    Stock A: +1 % per day. Stock B: -0.5 % per day.
    """
    dates = pd.date_range("2020-01-02", periods=253, freq="B")
    return pd.DataFrame(
        {
            "A": 100 * (1.01 ** np.arange(253)),
            "B": 100 * (0.995 ** np.arange(253)),
        },
        index=dates,
    )


class TestReturns:
    def test_log_returns_shape(self, synthetic_prices):
        """Log returns drop the first row (no previous price for day 0)."""
        returns = compute_log_returns(synthetic_prices)
        assert returns.shape == (252, 2)

    def test_log_returns_values(self, synthetic_prices):
        """For price = 100 * 1.01^t, daily log return = ln(1.01) exactly."""
        returns = compute_log_returns(synthetic_prices)
        assert abs(returns["A"].mean() - np.log(1.01)) < 1e-10

    def test_annualized_return(self, synthetic_prices):
        """Daily log(1.01) annualises to ln(1.01) * 252."""
        returns = compute_log_returns(synthetic_prices)
        mu = annualized_return(returns)
        assert abs(mu["A"] - np.log(1.01) * TRADING_DAYS_PER_YEAR) < 1e-8

    def test_covariance_matrix_is_symmetric(self, synthetic_prices):
        returns = compute_log_returns(synthetic_prices)
        cov = covariance_matrix(returns)
        assert cov.shape == (2, 2)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_covariance_diagonal_equals_variance(self, synthetic_prices):
        """Cov(i, i) must equal σ_i² (annualised)."""
        returns = compute_log_returns(synthetic_prices)
        cov = covariance_matrix(returns)
        vol = annualized_volatility(returns)
        for col in returns.columns:
            assert abs(cov.loc[col, col] - vol[col] ** 2) < 1e-10

    def test_statistical_moments_shape(self, synthetic_prices):
        returns = compute_log_returns(synthetic_prices)
        moments = statistical_moments(returns)
        assert moments.shape == (4, 2)
        assert list(moments.index) == [
            "mean_annual", "volatility_annual", "skewness", "excess_kurtosis"
        ]


# ---------------------------------------------------------------------------
# Integration tests — live network, run manually with -m integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_fetch_prices_live():
    """Download a 3-ticker, 5-day slice to confirm yfinance works."""
    from data_core.fetcher import fetch_prices
    prices = fetch_prices(
        tickers=["TTE.PA", "MC.PA", "AI.PA"],
        start="2024-01-02",
        end="2024-01-10",
    )
    assert not prices.empty
    assert prices.shape[1] <= 3
    assert prices.isna().sum().sum() == 0


@pytest.mark.integration
def test_fetch_benchmark_live():
    """Download ^FCHI for one week."""
    from data_core.fetcher import fetch_benchmark
    bench = fetch_benchmark(start="2024-01-02", end="2024-01-10")
    assert not bench.empty
    assert bench.name == "CAC40"
