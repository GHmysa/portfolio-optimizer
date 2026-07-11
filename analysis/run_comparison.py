"""
analysis/run_comparison.py — Module D entry point.

Builds the full pipeline (data -> optimisation -> analysis) and prints
the Module D deliverable: performance of the optimised portfolio, with
and without a cash allocation, next to the CAC40 benchmark.

Run from the project root:
    python -m analysis.run_comparison
"""

from data_core import (
    RISK_FREE_RATE,
    annualized_return,
    compute_log_returns,
    covariance_matrix,
    fetch_benchmark,
    fetch_prices,
)
from optimisation import capital_allocation_line, max_sharpe_portfolio
from analysis.performance import (
    apply_cash_allocation,
    comparison_table,
    performance_metrics,
    portfolio_daily_returns,
)

# A conservative investor's risk target. The tangency portfolio's own
# volatility is ~12 % p.a., below the index's ~20 %: matching the index
# risk would require leverage (capped by Module B). Targeting 10 %
# instead forces a genuine cash position — the case Module D discusses.
TARGET_VOLATILITY = 0.10

# --- 1. Data (Module A) -----------------------------------------------
prices = fetch_prices()
returns = compute_log_returns(prices)
bench_returns = compute_log_returns(fetch_benchmark())

# --- 2. Optimisation (Module B) ---------------------------------------
mu = annualized_return(returns)
cov = covariance_matrix(returns)
weights, r_tan, vol_tan, sharpe_tan = max_sharpe_portfolio(mu, cov, RISK_FREE_RATE)

w_risky, w_cash, _ = capital_allocation_line(
    TARGET_VOLATILITY, r_tan, vol_tan, RISK_FREE_RATE
)

# --- 3. Analysis (Module D) -------------------------------------------
rp = portfolio_daily_returns(returns, weights)
rc = apply_cash_allocation(rp, w_risky)

table = comparison_table(
    {
        "Max-Sharpe portfolio (fully invested)": rp,
        f"Max-Sharpe + cash ({w_cash:.0%} cash)": rc,
        "CAC40 index (^FCHI)": bench_returns,
    }
)

print("\nTangency portfolio - weights above 1 %:")
print(weights[weights > 0.01].round(3).sort_values(ascending=False))

print(f"\nCash split for target volatility {TARGET_VOLATILITY:.0%}: "
      f"{w_risky:.1%} risky / {w_cash:.1%} cash")

print("\nCross-check Module B vs Module D (same data, must be ~equal):")
realised = performance_metrics(rp)
print(f"  return: optimiser {r_tan:.4f} vs realised "
      f"{realised['annualized_return']:.4f}")
print(f"  vol   : optimiser {vol_tan:.4f} vs realised "
      f"{realised['annualized_volatility']:.4f}")

print("\nPerformance comparison (2019-2024, in-sample):")
print(table.round(4))

print("\nNote: ^FCHI is a price-return index (no dividends); the true "
      "total-return benchmark would be ~2 pp p.a. higher.")