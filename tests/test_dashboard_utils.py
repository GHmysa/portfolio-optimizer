import pandas as pd
import numpy as np
from dashboard._utils import compute_cumulative_returns, rolling_sharpe, weights_to_series


def test_compute_cumulative_preserves_index():
    """Index (e.g. dates) must pass through unchanged."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    r = pd.Series([0.01, -0.005, 0.02], index=idx)
    cum = compute_cumulative_returns(r)
    assert list(cum.index) == list(idx)


def test_compute_cumulative_log():
    r = pd.Series([0.001, -0.002, 0.003])
    cum = compute_cumulative_returns(r)
    expected = np.exp(r).cumprod()
    pd.testing.assert_series_equal(cum, expected)


def test_weights_to_series():
    w = [0.1, 0.2, 0.7]
    t = ["A", "B", "C"]
    s = weights_to_series(w, t)
    assert list(s.index) == ["C", "B", "A"]
    assert abs(s.sum() - 1.0) < 1e-8


def test_rolling_sharpe_nan_for_initial_period():
    """First window-1 observations must be NaN because the window is not yet full."""
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.001, 0.01, size=300))
    rs = rolling_sharpe(r, window=100)
    assert rs.iloc[:99].isna().all()
    assert pd.notna(rs.iloc[99])


def test_rolling_sharpe_same_index_as_input():
    """Output index must match the input Series index exactly."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 300), index=idx)
    rs = rolling_sharpe(r, window=50)
    assert list(rs.index) == list(idx)


def test_rolling_sharpe_zero_risk_free_is_return_over_vol():
    """With r_f = 0, rolling Sharpe = annualised_mean / annualised_std."""
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.002, 0.01, size=300))
    window = 100
    rs = rolling_sharpe(r, window=window, risk_free_rate=0.0)
    # Verify the last valid point manually
    last_window = r.iloc[-window:]
    expected = (last_window.mean() * 252) / (last_window.std() * np.sqrt(252))
    assert abs(rs.iloc[-1] - expected) < 1e-10
