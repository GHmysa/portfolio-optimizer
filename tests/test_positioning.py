"""
Tests for analysis/positioning.py.

Synthetic data only — no network calls. Run with:
    pytest tests/test_positioning.py -v
"""

import numpy as np
import pandas as pd
import pytest

from analysis.positioning import (
    DELTA_THRESHOLD_PCT,
    build_positioning_report,
    compute_overweight_table,
    compute_stock_stats,
    generate_commentary,
)


def make_returns(n_days: int, tickers: list[str], seed: int = 0) -> pd.DataFrame:
    """Deterministic daily log-return DataFrame, with a different mean/vol
    per ticker so Sharpe/volatility percentiles spread out meaningfully."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        mean = 0.0002 + 0.0001 * i
        vol = 0.005 + 0.001 * i
        data[t] = rng.normal(mean, vol, size=n_days)
    return pd.DataFrame(data, index=dates)


def make_reference_weights(rows: dict[str, tuple[str, float]]) -> pd.DataFrame:
    """rows: {ticker: (company_name, weight_pct)}"""
    return pd.DataFrame(
        [{"ticker": t, "company_name": n, "weight_pct": w} for t, (n, w) in rows.items()]
    )


class TestComputeOverweightTable:
    def test_ticker_missing_from_reference_is_flagged_not_dropped(self):
        weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
        ref = make_reference_weights({"AAA": ("Alpha", 40.0)})  # BBB has no reference
        table = compute_overweight_table(weights, ref)
        assert "BBB" in table.index
        assert pd.isna(table.loc["BBB", "delta_pct"])
        assert table.loc["BBB", "classification"] == "No index reference available"

    def test_delta_unit_conversion(self):
        """optimized_weight is a 0-1 fraction; weight_pct is 0-100. delta_pct
        must be in percentage points, not raw fraction units."""
        weights = pd.Series({"AAA": 0.30, "BBB": 0.70})
        ref = make_reference_weights({"AAA": ("Alpha", 25.0), "BBB": ("Beta", 75.0)})
        table = compute_overweight_table(weights, ref)
        assert np.isclose(table.loc["AAA", "delta_pct"], 30.0 - 25.0)
        assert np.isclose(table.loc["BBB", "delta_pct"], 70.0 - 75.0)

    def test_classification_boundary_exactly_one_point_is_neutral(self):
        """delta_pct == +-DELTA_THRESHOLD_PCT must be Neutral (strict >, <)."""
        weights = pd.Series({"AAA": 0.26, "BBB": 0.74})
        ref = make_reference_weights({"AAA": ("Alpha", 25.0), "BBB": ("Beta", 75.0)})
        table = compute_overweight_table(weights, ref)
        assert np.isclose(table.loc["AAA", "delta_pct"], DELTA_THRESHOLD_PCT)
        assert table.loc["AAA", "classification"] == "Neutral"
        assert np.isclose(table.loc["BBB", "delta_pct"], -DELTA_THRESHOLD_PCT)
        assert table.loc["BBB", "classification"] == "Neutral"

    def test_classification_just_beyond_boundary(self):
        weights = pd.Series({"AAA": 0.2601, "BBB": 0.7399})
        ref = make_reference_weights({"AAA": ("Alpha", 25.0), "BBB": ("Beta", 75.0)})
        table = compute_overweight_table(weights, ref)
        assert table.loc["AAA", "classification"] == "Overweight"
        assert table.loc["BBB", "classification"] == "Underweight"

    def test_sorted_most_overweight_to_most_underweight_nan_last(self):
        weights = pd.Series({"AAA": 0.50, "BBB": 0.30, "CCC": 0.20})
        ref = make_reference_weights({"AAA": ("Alpha", 20.0), "BBB": ("Beta", 40.0)})
        table = compute_overweight_table(weights, ref)
        # AAA: +30 (most overweight), BBB: -10 (underweight), CCC: NaN (last)
        assert list(table.index) == ["AAA", "BBB", "CCC"]


class TestComputeStockStats:
    def test_columns_and_percentile_range(self):
        tickers = list("ABCDE")
        returns = make_returns(300, tickers, seed=1)
        weights = pd.Series({t: 1 / len(tickers) for t in tickers})
        stats_df = compute_stock_stats(returns, weights)
        for col in [
            "mean_annual", "volatility_annual", "sharpe", "sharpe_percentile",
            "volatility_percentile", "skewness", "correlation_to_rest",
            "correlation_percentile",
        ]:
            assert col in stats_df.columns
        assert stats_df["sharpe_percentile"].between(0, 1).all()
        assert stats_df["volatility_percentile"].between(0, 1).all()

    def test_single_asset_portfolio_correlation_is_nan_not_crash(self):
        """Edge case: a stock IS the whole portfolio (weight ~1.0) -- there
        is no 'rest' to correlate against. Must not crash."""
        tickers = ["AAA", "BBB"]
        returns = make_returns(100, tickers, seed=2)
        weights = pd.Series({"AAA": 1.0, "BBB": 0.0})
        stats_df = compute_stock_stats(returns, weights)
        assert pd.isna(stats_df.loc["AAA", "correlation_to_rest"])
        # BBB (weight 0) has a well-defined "rest of portfolio" (= AAA) and
        # must NOT be NaN.
        assert pd.notna(stats_df.loc["BBB", "correlation_to_rest"])


class TestGenerateCommentary:
    def test_no_reference_weight_returns_dedicated_sentence(self):
        msg = generate_commentary("XXX", float("nan"), {})
        assert "No index reference weight" in msg

    def test_neutral_delta_returns_dedicated_sentence(self):
        msg = generate_commentary("XXX", 0.5, {"sharpe_percentile": 0.99})
        assert "no significant tilt" in msg

    def test_neutral_at_exact_boundary(self):
        """|delta| == DELTA_THRESHOLD_PCT must be Neutral, not Over/Underweight."""
        assert "no significant tilt" in generate_commentary("XXX", DELTA_THRESHOLD_PCT, {})
        assert "no significant tilt" in generate_commentary("XXX", -DELTA_THRESHOLD_PCT, {})

    def test_overweight_high_sharpe_only(self):
        stats = {"sharpe_percentile": 0.9, "correlation_percentile": 0.9}
        msg = generate_commentary("XXX", 5.0, stats)
        assert "strong historical risk-adjusted return" in msg
        assert "diversification benefit" not in msg

    def test_overweight_combines_both_rules_in_priority_order(self):
        stats = {"sharpe_percentile": 0.9, "correlation_percentile": 0.1}
        msg = generate_commentary("XXX", 5.0, stats)
        assert "strong historical risk-adjusted return" in msg
        assert "diversification benefit" in msg
        assert "also shows" in msg
        # rule 1 (Sharpe) must appear before rule 2 (correlation) in the sentence
        assert msg.index("strong historical risk-adjusted return") < msg.index("diversification benefit")

    def test_overweight_no_rule_fires_uses_fallback(self):
        stats = {"sharpe_percentile": 0.5, "correlation_percentile": 0.5}
        msg = generate_commentary("XXX", 5.0, stats)
        assert "Overweighted relative to the index" in msg
        assert "no single dominant statistical driver" in msg

    def test_underweight_high_vol_low_sharpe_only(self):
        stats = {"volatility_percentile": 0.9, "sharpe_percentile": 0.3, "skewness": 0.0}
        msg = generate_commentary("XXX", -5.0, stats)
        assert "high historical volatility relative to its return" in msg
        assert "sharp downside moves" not in msg

    def test_underweight_high_vol_requires_below_median_sharpe(self):
        """High volatility ALONE (with above-median Sharpe) must NOT trigger
        rule 3 -- 'relative to its return' is an efficiency statement, not
        a pure risk statement."""
        stats = {"volatility_percentile": 0.9, "sharpe_percentile": 0.9, "skewness": 0.0}
        msg = generate_commentary("XXX", -5.0, stats)
        assert "high historical volatility relative to its return" not in msg

    def test_underweight_negative_skew_only(self):
        stats = {"volatility_percentile": 0.2, "sharpe_percentile": 0.9, "skewness": -0.8}
        msg = generate_commentary("XXX", -5.0, stats)
        assert "sharp downside moves" in msg

    def test_underweight_combines_both_rules_in_priority_order(self):
        stats = {"volatility_percentile": 0.9, "sharpe_percentile": 0.1, "skewness": -0.8}
        msg = generate_commentary("XXX", -5.0, stats)
        assert "high historical volatility relative to its return" in msg
        assert "sharp downside moves" in msg
        assert "also shows" in msg
        # rule 3 (volatility) must appear before rule 4 (skewness) in the sentence
        assert msg.index("high historical volatility") < msg.index("sharp downside moves")

    def test_underweight_no_rule_fires_uses_fallback(self):
        stats = {"volatility_percentile": 0.5, "sharpe_percentile": 0.5, "skewness": 0.0}
        msg = generate_commentary("XXX", -5.0, stats)
        assert "Underweighted relative to the index" in msg
        assert "no single dominant statistical driver" in msg

    def test_rules_never_mix_across_directions(self):
        """An overweight stock must never surface underweight-only rule text
        (volatility/skew), even if those stats would technically qualify."""
        stats = {
            "sharpe_percentile": 0.1,          # would fail overweight rule 1
            "correlation_percentile": 0.9,     # would fail overweight rule 2
            "volatility_percentile": 0.9,      # would satisfy underweight rule 3
            "skewness": -0.9,                  # would satisfy underweight rule 4
        }
        msg = generate_commentary("XXX", 5.0, stats)  # overweight direction
        assert "high historical volatility" not in msg
        assert "sharp downside moves" not in msg
        assert "Overweighted relative to the index" in msg  # falls back, doesn't borrow rule 3/4

    def test_missing_correlation_key_does_not_crash(self):
        """Correlation key entirely absent (not just NaN) -- rule 2 must be
        skipped silently, not raise a KeyError."""
        stats = {"sharpe_percentile": 0.9}
        msg = generate_commentary("XXX", 5.0, stats)
        assert "strong historical risk-adjusted return" in msg

    def test_nan_correlation_value_does_not_crash(self):
        stats = {"sharpe_percentile": 0.9, "correlation_percentile": float("nan")}
        msg = generate_commentary("XXX", 5.0, stats)
        assert "strong historical risk-adjusted return" in msg

    def test_ticker_not_in_reference_weights_exact_text(self):
        """Mirrors the Part D scenario: a stock not in the reference weights
        has delta=NaN and must not crash even with a completely empty stats dict."""
        msg = generate_commentary("ZZZ", float("nan"), {})
        assert msg == (
            "No index reference weight is available for this stock (outside the 25 "
            "officially published Euronext weights), so an overweight/underweight "
            "comparison cannot be computed."
        )


class TestBuildPositioningReport:
    def test_end_to_end_shape_and_sort_order(self):
        tickers = list("ABCDEFGH")
        returns = make_returns(300, tickers, seed=5)
        weights = pd.Series({t: 1 / len(tickers) for t in tickers})
        ref = make_reference_weights({
            "A": ("Alpha", 5.0), "B": ("Beta", 30.0), "C": ("Gamma", 10.0),
        })  # D..H have no reference weight

        report = build_positioning_report(returns, weights, ref)

        assert list(report.columns) == [
            "company_name", "optimized_weight_pct", "index_weight_pct",
            "delta_pct", "abs_delta_pct", "classification", "commentary",
        ]
        assert set(report.index) == set(tickers)  # nothing dropped

        abs_deltas = report["abs_delta_pct"].dropna().tolist()
        assert abs_deltas == sorted(abs_deltas, reverse=True)
        assert report["commentary"].apply(lambda s: isinstance(s, str) and len(s) > 0).all()

        for t in "DEFGH":
            assert report.loc[t, "classification"] == "No index reference available"
            assert "No index reference weight" in report.loc[t, "commentary"]
