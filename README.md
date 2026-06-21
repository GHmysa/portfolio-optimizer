# Portfolio Optimizer

Applied Quantitative Methods project — Technische Hochshule Mittelhessen.

**Goal:** Apply Markowitz portfolio theory to the CAC40: find the optimal
allocation across its 40 constituent stocks plus a cash position (risk-free
asset) to maximise risk-adjusted return, then measure whether the result
beats the actual CAC40 index.

---

## Module Architecture

```
portfolio-optimizer/
├── data_core/       Module A — data ingestion & statistics   (Maxime)
├── optimisation/    Module B — Markowitz + Capital Allocation Line (Yusuf)
├── dashboard/       Module C — Streamlit UI                  (Kris)
├── analysis/        Module D — performance comparison        (Emin)
├── ai_advisor/      Module E — DEFERRED (see ai_advisor/README.md)
└── tests/           pytest test suite
```

### How modules depend on each other

```
data_core  ──►  optimisation  ──►  dashboard
    │                │                 │
    └────────────────┴────────►  analysis
```

`data_core` is the only module that touches the network (Yahoo Finance).
All other modules receive clean Python objects from it — no raw HTTP calls.

---

## Data Contract (what Module A provides)

Import everything from `data_core` directly:

```python
from data_core import (
    fetch_prices,          # pd.DataFrame of adjusted daily close prices
    fetch_benchmark,       # pd.Series of ^FCHI (CAC40 index) prices
    compute_log_returns,   # pd.DataFrame of daily log returns
    annualized_return,     # pd.Series — μ × 252 per stock
    annualized_volatility, # pd.Series — σ × √252 per stock
    covariance_matrix,     # pd.DataFrame — annualised covariance matrix Σ
    statistical_moments,   # pd.DataFrame — mean / vol / skew / kurtosis
    get_cac40_tickers,     # list[str] — 40 Yahoo Finance ticker strings
    get_cac40_names,       # dict[str, str] — {ticker: company_name}
    get_cac40_sectors,     # dict[str, str] — {ticker: ICB sector}
    load_reference_weights,# pd.DataFrame — 25 official Euronext weights (reference only)
    BENCHMARK_TICKER,      # str  — "^FCHI"
    RISK_FREE_RATE,        # float — 0.0225  (2.25 % p.a., EUR cash proxy)
    TRADING_DAYS_PER_YEAR, # int  — 252
)
```

**Default date range:** 2019-01-01 → 2024-12-31 (6 years, ~1,500 trading days).
Pass `start=` and `end=` keyword args to override.

---

## Known Data Limitations (read before writing your report)

| # | Limitation | Impact | Mitigation |
|---|-----------|--------|------------|
| 1 | Euronext only publishes top-25 weights publicly | Weights for the remaining 15 constituents (~9.43 % of index) are unavailable without a paid data licence | `load_reference_weights()` provides the 25 confirmed weights for the dashboard concentration chart; all return calculations use `^FCHI` directly — no per-stock weights needed |
| 2 | `^FCHI` is a **price-return** index on Yahoo Finance | Understates the true index total return by ~2 % p.a. (dividends excluded) | Note in report; for rigour, compare against CAC 40 GR (gross return) if available |
| 3 | Constituent list is a static snapshot (Euronext stocks page, verified 2026-06-21) | CAC40 rebalances quarterly; one or two names may change | Historical analysis is run over a fixed past window where the composition was stable; rebalancing risk is minor for a 5-year backtest |
| 4 | ArcelorMittal uses the Amsterdam listing `MT.AS` | Only non-`.PA` ticker; Euronext Amsterdam and Paris may have different public holidays | `fetch_prices()` forward-fills across non-trading days, aligning all series to the same daily grid |
| 5 | Stellantis `STLAP.PA` is dual-listed (Paris + Milan + NYSE) | Yahoo Finance coverage of the Paris listing can be inconsistent | `fetch_prices()` drops failed tickers with a warning; run the integration test to verify before the full download |

---

## Setup Instructions (for Yusuf, Kris, Emin)

### 1. Clone and enter the repo

```bash
git clone <remote-url>
cd portfolio-optimizer
```

### 2. Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the test suite

```bash
# Unit tests only (no network, fast):
pytest tests/ -v -m "not integration"

# Include live yfinance tests:
pytest tests/ -v
```

### 5. Verify Module A works end-to-end

```python
from data_core import fetch_prices, fetch_benchmark, compute_log_returns, covariance_matrix

prices    = fetch_prices()          # downloads ~1500 rows × up to 40 columns (~15 s)
benchmark = fetch_benchmark()
returns   = compute_log_returns(prices)
cov       = covariance_matrix(returns)

print(prices.shape)       # (≈1500, ≤40) — fewer if any tickers fail to download
print(cov.shape)          # (≤40, ≤40)
print(benchmark.head())
```

### Branch policy

Work on your own module branch: `module-b`, `module-c`, `module-d`.
Open a PR into `main` once your module passes tests and has been reviewed
by at least one other team member.

---

## Running the Dashboard (Module C — once built)

```bash
streamlit run dashboard/app.py
```

---

## Project Timeline

| Milestone | Module | Owner |
|-----------|--------|-------|
| Data pipeline complete | A | Maxime |
| Efficient frontier + CAL | B | Yusuf |
| Streamlit UI | C | Kris |
| Performance analysis | D | Emin |
| ai_advisor scope (post-professor meeting) | E | TBD |
