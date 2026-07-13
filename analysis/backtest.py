"""
analysis/backtest.py — Module D: out-of-sample back-test.

Tests whether the max-Sharpe portfolio, fitted entirely on an estimation
window, continues to outperform the CAC40 on an evaluation window it has
never seen. This is the honest version of run_comparison.py — the
optimiser never touches the evaluation data.
"""

import warnings

import numpy as np
import pandas as pd

from data_core import (
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    compute_log_returns,
    covariance_matrix,
)
from optimisation import max_sharpe_portfolio
from analysis.performance import comparison_table


def out_of_sample_backtest(
    estimation_prices: pd.DataFrame,
    evaluation_prices: pd.DataFrame,
    evaluation_benchmark: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    max_weight: float = 0.15,
) -> dict:
    """
    Fit a max-Sharpe portfolio on an estimation window; evaluate it on an
    unseen evaluation window alongside the CAC40 benchmark.

    Ticker alignment
    ----------------
    Some tickers may appear in one window but not the other (e.g. a stock
    dropped by fetch_prices() because it lacks history in that window).
    Only the intersection of tickers is used for evaluation.

    Estimation weights for dropped tickers are redistributed proportionally
    across the surviving common tickers and then renormalised so the weights
    still sum to 1:

        w_i_new = w_i_old / sum( w_j  for j in common_tickers )

    Example: estimation weights {A: 0.50, B: 0.30, C: 0.20} and C is absent
    from evaluation_prices.  Surviving weight sum = 0.50 + 0.30 = 0.80.
    Renormalised: {A: 0.50/0.80 = 0.625, B: 0.30/0.80 = 0.375}.
    The weights still sum to 1, and the per-asset max_weight constraint may
    technically be exceeded for the remaining assets — this is documented and
    acceptable when only a small fraction of the portfolio is redistributed.

    Parameters
    ----------
    estimation_prices   : price DataFrame covering the estimation window.
                          Columns = ticker symbols; index = DatetimeIndex.
    evaluation_prices   : price DataFrame covering the evaluation window.
    evaluation_benchmark: benchmark price Series for the evaluation window.
                          Name should be "CAC40" (set by fetch_benchmark).
    risk_free_rate      : annualised risk-free rate (default RISK_FREE_RATE).
    max_weight          : per-asset weight cap passed to max_sharpe_portfolio
                          (default 0.15 = 15 %).

    Returns
    -------
    dict with keys:
        "weights"      — pd.Series: fitted weights from the estimation window.
        "weights_eval" — pd.Series: renormalised weights for evaluation tickers.
        "table"        — pd.DataFrame: comparison_table over the evaluation window.
        "port_cum"     — pd.Series: cumulative growth index of the portfolio
                         over the evaluation period, normalised to 1.0 at start.
        "bench_cum"    — pd.Series: same for the benchmark.
    """
    # --- 1. Fit on estimation window ---
    est_returns = compute_log_returns(estimation_prices)
    mu = annualized_return(est_returns)
    cov = covariance_matrix(est_returns)
    weights, _, _, _ = max_sharpe_portfolio(mu, cov, risk_free_rate, max_weight=max_weight)

    # --- 2. Align weights to evaluation tickers (intersection only) ---
    eval_returns = compute_log_returns(evaluation_prices)
    common_tickers = weights.index.intersection(eval_returns.columns)

    dropped = set(weights.index) - set(common_tickers)
    if dropped:
        warnings.warn(
            f"Tickers in estimation but absent from evaluation window — "
            f"redistributing their weights proportionally: {sorted(dropped)}",
            stacklevel=2,
        )

    weights_eval = weights.reindex(common_tickers)
    total = weights_eval.sum()
    if total <= 1e-10:
        raise RuntimeError(
            "No common tickers between estimation and evaluation windows. "
            "Check that both price DataFrames share at least one column."
        )
    weights_eval = weights_eval / total  # renormalise so sum = 1

    # --- 3. Compute daily portfolio returns on the evaluation window ---
    eval_port_returns = eval_returns[common_tickers].dot(weights_eval)
    eval_port_returns.name = "portfolio"

    bench_returns = compute_log_returns(evaluation_benchmark)

    # --- 4. Performance table ---
    label = f"Max-Sharpe <={max_weight:.0%}/stock (out-of-sample)"
    table = comparison_table(
        {label: eval_port_returns, "CAC40 index (^FCHI)": bench_returns},
        risk_free_rate=risk_free_rate,
    )

    # --- 5. Cumulative growth index (both normalised to 1.0 at first observation) ---
    port_cum = np.exp(eval_port_returns).cumprod()
    port_cum = port_cum / port_cum.iloc[0]

    bench_cum = np.exp(bench_returns).cumprod()
    bench_cum = bench_cum / bench_cum.iloc[0]

    return {
        "weights": weights,
        "weights_eval": weights_eval,
        "table": table,
        "port_cum": port_cum,
        "bench_cum": bench_cum,
    }
