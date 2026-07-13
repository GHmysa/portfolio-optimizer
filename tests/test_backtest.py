"""
Tests for analysis/backtest.py.

Synthetic data only — no network calls. Run with:
    pytest tests/test_backtest.py -v
"""

import numpy as np
import pandas as pd
import pytest

from analysis.backtest import out_of_sample_backtest


def make_prices(n_days: int, tickers: list[str], seed: int = 0) -> pd.DataFrame:
    """Deterministic log-normal price series for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    log_returns = rng.normal(0.0003, 0.01, size=(n_days, len(tickers)))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def make_benchmark(n_days: int, seed: int = 99) -> pd.Series:
    """Deterministic benchmark price Series for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    log_returns = rng.normal(0.0002, 0.008, size=n_days)
    prices = 100 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=dates, name="CAC40")


@pytest.fixture
def standard_inputs():
    """
    Ten-ticker case where estimation and evaluation share all tickers.

    Ten tickers are required because the default max_weight=0.15 needs at
    least ceil(1/0.15) = 7 assets to be feasible (7 × 0.15 = 1.05 ≥ 1.0).
    Using 10 gives a comfortable margin and mirrors a realistic mini-universe.
    """
    tickers = list("ABCDEFGHIJ")
    est = make_prices(300, tickers, seed=1)
    evl = make_prices(252, tickers, seed=2)
    bench = make_benchmark(252, seed=3)
    return est, evl, bench


class TestOutOfSampleBacktest:
    def test_returns_required_keys(self, standard_inputs):
        est, evl, bench = standard_inputs
        result = out_of_sample_backtest(est, evl, bench)
        for key in ("weights", "weights_eval", "table", "port_cum", "bench_cum"):
            assert key in result

    def test_weights_eval_sum_to_one(self, standard_inputs):
        est, evl, bench = standard_inputs
        result = out_of_sample_backtest(est, evl, bench)
        assert abs(result["weights_eval"].sum() - 1.0) < 1e-6

    def test_weights_eval_are_long_only(self, standard_inputs):
        est, evl, bench = standard_inputs
        result = out_of_sample_backtest(est, evl, bench)
        assert (result["weights_eval"] >= -1e-8).all()

    def test_weights_eval_respect_max_weight_when_no_tickers_dropped(self, standard_inputs):
        """When estimation and evaluation share all tickers, max_weight must hold."""
        est, evl, bench = standard_inputs
        result = out_of_sample_backtest(est, evl, bench, max_weight=0.4)
        assert (result["weights_eval"] <= 0.4 + 1e-8).all()

    def test_table_has_two_rows_with_correct_columns(self, standard_inputs):
        est, evl, bench = standard_inputs
        result = out_of_sample_backtest(est, evl, bench)
        assert result["table"].shape[0] == 2
        assert list(result["table"].columns) == [
            "annualized_return", "annualized_volatility", "sharpe_ratio"
        ]

    def test_cumulative_series_nonempty_and_start_near_one(self, standard_inputs):
        """Both cumulative series must be non-empty and normalised to start at 1.0."""
        est, evl, bench = standard_inputs
        result = out_of_sample_backtest(est, evl, bench)
        assert not result["port_cum"].empty
        assert not result["bench_cum"].empty
        assert abs(result["port_cum"].iloc[0] - 1.0) < 1e-10
        assert abs(result["bench_cum"].iloc[0] - 1.0) < 1e-10

    def test_missing_evaluation_ticker_redistributes_proportionally(self):
        """
        Ticker J is in estimation (10 tickers) but absent from evaluation (9 tickers).
        Weights for A–I must be renormalised so they still sum to 1.

        10 tickers are needed so max_weight=0.15 is feasible in estimation
        (10 × 0.15 = 1.5 ≥ 1.0); 9 tickers are enough in evaluation
        (9 × 0.15 = 1.35 ≥ 1.0).
        """
        tickers_est = list("ABCDEFGHIJ")   # 10 tickers
        tickers_evl = list("ABCDEFGHI")    # 9 tickers — J missing
        est = make_prices(300, tickers_est, seed=10)
        evl = make_prices(252, tickers_evl, seed=11)
        bench = make_benchmark(252, seed=12)

        result = out_of_sample_backtest(est, evl, bench)

        assert "J" not in result["weights_eval"].index
        assert set(result["weights_eval"].index) == set(tickers_evl)
        assert abs(result["weights_eval"].sum() - 1.0) < 1e-6
        assert (result["weights_eval"] >= -1e-8).all()

    def test_original_weights_unchanged_after_redistribution(self):
        """
        The 'weights' key must contain the raw estimation weights (all 10 tickers),
        while 'weights_eval' contains only the renormalised 9-ticker subset.
        """
        tickers_est = list("ABCDEFGHIJ")
        tickers_evl = list("ABCDEFGHI")
        est = make_prices(300, tickers_est, seed=20)
        evl = make_prices(252, tickers_evl, seed=21)
        bench = make_benchmark(252, seed=22)

        result = out_of_sample_backtest(est, evl, bench)

        assert set(result["weights"].index) == set(tickers_est)
        assert abs(result["weights"].sum() - 1.0) < 1e-6
