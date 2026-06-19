# Portfolio Optimizer

Applied Quantitative Methods project — Technische Hochshule Mitelhessen.

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
| 1 | Euronext only publishes top-25 weights publicly | Bottom-14 weights are derived from *total* market cap (yfinance), not official free-float weights | Use ^FCHI for all benchmark comparisons; treat bottom-14 weights as approximate |
| 2 | Crédit Agricole (ACA.PA) free-float is ~41 % of total market cap | Our weight (2.23 %) is likely an overestimate by 1–2 pp | Document in report; sensitivity-test by removing ACA.PA |
| 3 | Composition is a static September 2024 snapshot | CAC40 rebalances quarterly | Keep historical analysis within the same period; note in report |
| 4 | ^FCHI is a price-return index (Yahoo Finance) | Slightly understates the actual index total return (dividends excluded) | Note in report; effect is ~2 % p.a. |
| 5 | 40th constituent is unidentified | Optimizer runs on 39 stocks | Weight is <0.3 %; effect on results is negligible |

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

prices    = fetch_prices()          # downloads ~1500 rows × 39 columns (~10 s)
benchmark = fetch_benchmark()
returns   = compute_log_returns(prices)
cov       = covariance_matrix(returns)

print(prices.shape)       # (≈1500, 39)
print(cov.shape)          # (39, 39)
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
