"""
optimisation/run_summary.py — Module B entry point.

Prints all optimisation results: MVP, tangency portfolio (unconstrained and
constrained at 15%), Capital Allocation Line, and efficient frontier bounds.

Run from the project root:
    python -m optimisation.run_summary
"""

from data_core import (
    RISK_FREE_RATE,
    annualized_return,
    covariance_matrix,
    compute_log_returns,
    fetch_prices,
    get_cac40_names,
)
from optimisation import (
    capital_allocation_line,
    efficient_frontier,
    max_sharpe_portfolio,
    minimum_variance_portfolio,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)

TARGET_VOLATILITY = 0.10   # conservative investor target (matches run_comparison)
MAX_WEIGHT        = 0.15   # 15 % per-stock cap

# --- Data (Module A) ------------------------------------------------------
prices  = fetch_prices()
returns = compute_log_returns(prices)
names   = get_cac40_names()
mu      = annualized_return(returns)
cov     = covariance_matrix(returns)

# --- Optimisation (Module B) ----------------------------------------------
mvp_w,  mvp_r,  mvp_v                    = minimum_variance_portfolio(mu, cov)
unc_w,  unc_r,  unc_v,  unc_s            = max_sharpe_portfolio(mu, cov, RISK_FREE_RATE)
con_w,  con_r,  con_v,  con_s            = max_sharpe_portfolio(mu, cov, RISK_FREE_RATE, max_weight=MAX_WEIGHT)

w_risky_unc, w_cash_unc, cal_r_unc       = capital_allocation_line(TARGET_VOLATILITY, unc_r, unc_v, RISK_FREE_RATE)
w_risky_con, w_cash_con, cal_r_con       = capital_allocation_line(TARGET_VOLATILITY, con_r, con_v, RISK_FREE_RATE)

ef_returns, ef_vols, _                   = efficient_frontier(mu, cov, n_points=50)
ef_returns_con, ef_vols_con, _           = efficient_frontier(mu, cov, n_points=50, max_weight=MAX_WEIGHT)

# ==========================================================================

print("=" * 62)
print("MODULE B — OPTIMISATION RESULTS")
print(f"Risk-free rate : {RISK_FREE_RATE:.2%}    Target vol : {TARGET_VOLATILITY:.0%}")
print("=" * 62)

# --- MVP ------------------------------------------------------------------
print("\n--- MINIMUM VARIANCE PORTFOLIO ---")
print(f"Expected return : {mvp_r:.2%}")
print(f"Volatility      : {mvp_v:.2%}")
print(f"Sharpe ratio    : {(mvp_r - RISK_FREE_RATE) / mvp_v:.3f}")
print("Weights (> 1%) :")
for t, w in mvp_w[mvp_w > 0.01].sort_values(ascending=False).items():
    print(f"  {t:<12} {names.get(t,''):<28} {w:.1%}")

# --- Unconstrained tangency -----------------------------------------------
print("\n--- MAX-SHARPE PORTFOLIO (unconstrained) ---")
print(f"Expected return : {unc_r:.2%}")
print(f"Volatility      : {unc_v:.2%}")
print(f"Sharpe ratio    : {unc_s:.3f}")
print("Weights (> 1%) :")
for t, w in unc_w[unc_w > 0.01].sort_values(ascending=False).items():
    print(f"  {t:<12} {names.get(t,''):<28} {w:.1%}")

print(f"\nCAL at target vol {TARGET_VOLATILITY:.0%} :")
print(f"  {w_risky_unc:.1%} risky + {w_cash_unc:.1%} cash  ->  expected return {cal_r_unc:.2%}")

# --- Constrained tangency -------------------------------------------------
print(f"\n--- MAX-SHARPE PORTFOLIO (max {MAX_WEIGHT:.0%}/stock) ---")
print(f"Expected return : {con_r:.2%}")
print(f"Volatility      : {con_v:.2%}")
print(f"Sharpe ratio    : {con_s:.3f}")
print("Weights (all) :")
for t, w in con_w[con_w > 0.001].sort_values(ascending=False).items():
    print(f"  {t:<12} {names.get(t,''):<28} {w:.1%}")

print(f"\nCAL at target vol {TARGET_VOLATILITY:.0%} :")
print(f"  {w_risky_con:.1%} risky + {w_cash_con:.1%} cash  ->  expected return {cal_r_con:.2%}")

# --- Efficient frontier summary -------------------------------------------
print("\n--- EFFICIENT FRONTIER ---")
print(f"{'':30} {'Return':>8} {'Vol':>8}")
print(f"  Unconstrained — left end (MVP)  : {ef_returns[0]:>7.2%}   {ef_vols[0]:>7.2%}")
print(f"  Unconstrained — right end       : {ef_returns[-1]:>7.2%}   {ef_vols[-1]:>7.2%}")
print(f"  Constrained   — left end (MVP)  : {ef_returns_con[0]:>7.2%}   {ef_vols_con[0]:>7.2%}")
print(f"  Constrained   — right end       : {ef_returns_con[-1]:>7.2%}   {ef_vols_con[-1]:>7.2%}")

# --- Comparison table -----------------------------------------------------
print("\n--- SUMMARY TABLE ---")
print(f"{'Portfolio':<42} {'Return':>7} {'Vol':>7} {'Sharpe':>7}")
print("-" * 62)
rows = [
    ("MVP (min variance)",           mvp_r, mvp_v, (mvp_r - RISK_FREE_RATE) / mvp_v),
    ("Max-Sharpe unconstrained",     unc_r, unc_v, unc_s),
    (f"Max-Sharpe <={MAX_WEIGHT:.0%}/stock",  con_r, con_v, con_s),
    (f"CAL unc. @ {TARGET_VOLATILITY:.0%} vol",  cal_r_unc, TARGET_VOLATILITY, unc_s),
    (f"CAL con. @ {TARGET_VOLATILITY:.0%} vol",  cal_r_con, TARGET_VOLATILITY, con_s),
]
for label, r, v, s in rows:
    print(f"  {label:<40} {r:>6.2%} {v:>6.2%} {s:>7.3f}")

print("\nNote: all figures are annualised, in-sample 2019-2024.")
