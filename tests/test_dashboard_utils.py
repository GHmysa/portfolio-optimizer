import pandas as pd
import numpy as np
from dashboard._utils import compute_cumulative_returns, weights_to_series


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
