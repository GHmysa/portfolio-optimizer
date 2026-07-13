import pandas as pd
import numpy as np
from dashboard._utils import compute_cumulative_returns, weights_to_series


def test_compute_cumulative_simple():
    r = pd.Series([0.01, -0.005, 0.02])
    cum = compute_cumulative_returns(r)
    expected = (1 + r).cumprod()
    pd.testing.assert_series_equal(cum, expected)


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
