"""
analysis/run_backtest.py — Module D out-of-sample validation entry point.

Fits the max-Sharpe portfolio on 2019-2021 and evaluates it on 2022-2024.
The evaluation window is entirely unseen by the optimiser.

Run from the project root:
    python -m analysis.run_backtest
"""

from data_core import RISK_FREE_RATE, fetch_benchmark, fetch_prices, get_cac40_tickers
from analysis.backtest import out_of_sample_backtest

ESTIMATION_START = "2019-01-01"
ESTIMATION_END = "2021-12-31"
EVALUATION_START = "2022-01-01"
EVALUATION_END = "2024-12-31"
MAX_WEIGHT = 0.15

tickers = get_cac40_tickers()

print(f"Fetching estimation window ({ESTIMATION_START} to {ESTIMATION_END})...")
estimation_prices = fetch_prices(tickers, start=ESTIMATION_START, end=ESTIMATION_END)

print(f"Fetching evaluation window ({EVALUATION_START} to {EVALUATION_END})...")
evaluation_prices = fetch_prices(tickers, start=EVALUATION_START, end=EVALUATION_END)
evaluation_benchmark = fetch_benchmark(start=EVALUATION_START, end=EVALUATION_END)

result = out_of_sample_backtest(
    estimation_prices=estimation_prices,
    evaluation_prices=evaluation_prices,
    evaluation_benchmark=evaluation_benchmark,
    risk_free_rate=RISK_FREE_RATE,
    max_weight=MAX_WEIGHT,
)

print(f"\nPortfolio fitted on {ESTIMATION_START} to {ESTIMATION_END} (max {MAX_WEIGHT:.0%}/stock):")
w = result["weights_eval"]
print(w[w > 0.01].round(3).sort_values(ascending=False).to_string())

print(f"\nOut-of-sample performance ({EVALUATION_START} to {EVALUATION_END}):")
print(result["table"].to_string())

print(
    "\nNote: ^FCHI is a price-return index (no dividends); the true "
    "total-return benchmark would be ~2 pp p.a. higher."
)
