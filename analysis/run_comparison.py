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
# volatility is above 20 % p.a.; targeting 10 % instead forces a genuine
# cash position — the case Module D discusses.
TARGET_VOLATILITY = 0.10

# Maximum weight any single stock can hold in the constrained portfolio.
# 0.15 = 15 %, a common real-world limit (e.g. UCITS regulation caps at 10 %).
MAX_WEIGHT = 0.15

# --- 1. Data (Module A) -----------------------------------------------
prices = fetch_prices()
returns = compute_log_returns(prices)
bench_returns = compute_log_returns(fetch_benchmark())

# --- 2. Optimisation (Module B) ---------------------------------------
mu = annualized_return(returns)
cov = covariance_matrix(returns)

# Unconstrained tangency portfolio
weights_unc, r_unc, vol_unc, sharpe_unc = max_sharpe_portfolio(
    mu, cov, RISK_FREE_RATE
)
w_risky_unc, w_cash_unc, _ = capital_allocation_line(
    TARGET_VOLATILITY, r_unc, vol_unc, RISK_FREE_RATE
)

# Constrained tangency portfolio (max MAX_WEIGHT per stock)
weights_con, r_con, vol_con, sharpe_con = max_sharpe_portfolio(
    mu, cov, RISK_FREE_RATE, max_weight=MAX_WEIGHT
)
w_risky_con, w_cash_con, _ = capital_allocation_line(
    TARGET_VOLATILITY, r_con, vol_con, RISK_FREE_RATE
)

# --- 3. Analysis (Module D) -------------------------------------------
rp_unc = portfolio_daily_returns(returns, weights_unc)
rc_unc = apply_cash_allocation(rp_unc, w_risky_unc)

rp_con = portfolio_daily_returns(returns, weights_con)
rc_con = apply_cash_allocation(rp_con, w_risky_con)

table = comparison_table(
    {
        f"Max-Sharpe unconstrained (fully invested)": rp_unc,
        f"Max-Sharpe unconstrained + cash ({w_cash_unc:.0%} cash)": rc_unc,
        f"Max-Sharpe <={MAX_WEIGHT:.0%}/stock (fully invested)": rp_con,
        f"Max-Sharpe <={MAX_WEIGHT:.0%}/stock + cash ({w_cash_con:.0%} cash)": rc_con,
        "CAC40 index (^FCHI)": bench_returns,
    }
)

# --- Output -----------------------------------------------------------
print("\nUnconstrained tangency portfolio — weights above 1 %:")
print(weights_unc[weights_unc > 0.01].round(3).sort_values(ascending=False))

print(f"\nConstrained tangency portfolio (max {MAX_WEIGHT:.0%}/stock) — all weights:")
print(weights_con[weights_con > 0.01].round(3).sort_values(ascending=False))

print(f"\nCash split (target vol {TARGET_VOLATILITY:.0%}):")
print(f"  Unconstrained : {w_risky_unc:.1%} risky / {w_cash_unc:.1%} cash")
print(f"  Constrained   : {w_risky_con:.1%} risky / {w_cash_con:.1%} cash")

print("\nCross-check Module B vs Module D (unconstrained, must be ~equal):")
realised = performance_metrics(rp_unc)
print(f"  return : optimiser {r_unc:.4f} vs realised {realised['annualized_return']:.4f}")
print(f"  vol    : optimiser {vol_unc:.4f} vs realised {realised['annualized_volatility']:.4f}")

print("\nPerformance comparison (2019-2024, in-sample):")
print(table.to_string())

print(
    "\nNote: ^FCHI is a price-return index (no dividends); the true "
    "total-return benchmark would be ~2 pp p.a. higher."
)
