"""
data_core/run_summary.py — Module A entry point.

Prints a complete statistical summary of the CAC40 price data:
tickers, date range, per-asset return/vol/Sharpe, normality check,
covariance diagnostics, and benchmark stats.

Run from the project root:
    python -m data_core.run_summary
"""

import numpy as np

from data_core import (
    BENCHMARK_TICKER,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    covariance_matrix,
    compute_log_returns,
    fetch_benchmark,
    fetch_prices,
    get_cac40_names,
    get_cac40_tickers,
    load_reference_weights,
    statistical_moments,
)

# --- 1. Download ----------------------------------------------------------
tickers = get_cac40_tickers()
names   = get_cac40_names()
prices  = fetch_prices()
returns = compute_log_returns(prices)
bench   = fetch_benchmark()
bench_r = compute_log_returns(bench)

# --- 2. Derived statistics ------------------------------------------------
mu      = annualized_return(returns)
vol     = annualized_volatility(returns)
cov     = covariance_matrix(returns)
moments = statistical_moments(returns)
sharpe_per_stock = (mu - RISK_FREE_RATE) / vol

# --- Output ---------------------------------------------------------------

print("=" * 62)
print("MODULE A — DATA SUMMARY")
print("=" * 62)

print(f"\nTickers in CSV     : {len(tickers)} (full CAC40 list)")
print(f"Tickers downloaded : {len(prices.columns)} (after dropping bad data)")
dropped = set(tickers) - set(prices.columns)
if dropped:
    print(f"Dropped            : {sorted(dropped)}")
print(f"Date range         : {prices.index[0].date()} -> {prices.index[-1].date()}")
print(f"Price rows         : {len(prices)}")
print(f"Return rows        : {len(returns)}  (prices - 1, first row dropped)")

print(f"\nConstants")
print(f"  Risk-free rate        : {RISK_FREE_RATE:.2%}  (ECB deposit facility, mid-2026)")
print(f"  Trading days per year : {TRADING_DAYS_PER_YEAR}")
print(f"  Benchmark ticker      : {BENCHMARK_TICKER}")

# --- Per-asset table -------------------------------------------------------
print("\n" + "=" * 62)
print("PER-ASSET STATISTICS  (annualised, sorted by Sharpe)")
print("=" * 62)
print(f"{'Ticker':<12} {'Name':<28} {'Return':>7} {'Vol':>7} {'Sharpe':>7}")
print("-" * 62)
for ticker in sharpe_per_stock.sort_values(ascending=False).index:
    short_name = names.get(ticker, ticker)[:27]
    print(
        f"{ticker:<12} {short_name:<28} "
        f"{mu[ticker]:>6.1%} {vol[ticker]:>6.1%} {sharpe_per_stock[ticker]:>7.2f}"
    )

# --- Summary stats --------------------------------------------------------
print(f"\nBest return  : {mu.idxmax()}  {mu.max():.1%}")
print(f"Worst return : {mu.idxmin()}  {mu.min():.1%}")
print(f"Lowest vol   : {vol.idxmin()}  {vol.min():.1%}")
print(f"Highest vol  : {vol.idxmax()}  {vol.max():.1%}")
print(f"Best Sharpe  : {sharpe_per_stock.idxmax()}  {sharpe_per_stock.max():.2f}")

# --- Normality check ------------------------------------------------------
print("\n" + "=" * 62)
print("NORMALITY CHECK  (Markowitz assumes normal returns)")
print("=" * 62)
neg_skew = (moments.loc["skewness"] < 0).sum()
fat_tails = (moments.loc["excess_kurtosis"] > 0).sum()
n = len(prices.columns)
print(f"Negative skewness  (left tail heavier) : {neg_skew}/{n} stocks")
print(f"Positive excess kurtosis (fat tails)   : {fat_tails}/{n} stocks")
print(f"Mean skewness      : {moments.loc['skewness'].mean():.3f}  (0 = normal)")
print(f"Mean excess kurt.  : {moments.loc['excess_kurtosis'].mean():.3f}  (0 = normal)")
print("\nConclusion: returns are non-normal (fat tails, left skew).")
print("Markowitz is a simplification — still practically useful.")

# --- Covariance diagnostics -----------------------------------------------
print("\n" + "=" * 62)
print("COVARIANCE MATRIX DIAGNOSTICS")
print("=" * 62)
print(f"Shape          : {cov.shape[0]} x {cov.shape[1]}")
print(f"Symmetric      : {np.allclose(cov.values, cov.values.T)}")
eigenvalues = np.linalg.eigvalsh(cov.values)
print(f"Positive semi-definite : {(eigenvalues >= -1e-10).all()}  "
      f"(min eigenvalue = {eigenvalues.min():.2e})")

# Most correlated pair
corr = returns.corr()
# Stack upper triangle only (exclude self-pairs where row == col)
flat = corr.stack()
flat = flat[flat.index.get_level_values(0) < flat.index.get_level_values(1)]
top_pair = flat.idxmax()
print(f"Highest correlation    : {top_pair[0]} / {top_pair[1]}  "
      f"rho = {flat.max():.3f}")
low_pair = flat.idxmin()
print(f"Lowest correlation     : {low_pair[0]} / {low_pair[1]}  "
      f"rho = {flat.min():.3f}")

# --- Benchmark ------------------------------------------------------------
print("\n" + "=" * 62)
print("BENCHMARK  (^FCHI — price return, no dividends)")
print("=" * 62)
bm     = annualized_return(bench_r)
bv     = annualized_volatility(bench_r)
bs     = (bm - RISK_FREE_RATE) / bv
print(f"Annualised return : {bm:.2%}")
print(f"Annualised vol    : {bv:.2%}")
print(f"Sharpe ratio      : {bs:.3f}")
print("Note: true total-return CAC40 would be ~2 pp/year higher.")

# --- Reference weights ----------------------------------------------------
print("\n" + "=" * 62)
print("OFFICIAL EURONEXT WEIGHTS  (25 of 40, March 2026 PDF)")
print("=" * 62)
ref = load_reference_weights().sort_values("weight_pct", ascending=False)
print(f"{'Company':<30} {'Ticker':<10} {'Weight':>7}")
print("-" * 50)
for _, row in ref.iterrows():
    print(f"{row['company_name']:<30} {row['ticker']:<10} {row['weight_pct']:>6.2f}%")
print(f"\nTop-25 total : {ref['weight_pct'].sum():.2f}%  "
      f"(remaining 15 stocks require a paid licence)")
