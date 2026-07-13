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
├── data_core/       Module A — data ingestion & statistics     (Maxime)   ✅ complete
├── optimisation/    Module B — Markowitz + Capital Allocation Line (Yusuf) ✅ complete
├── dashboard/       Module C — Streamlit UI                    (Kris)     ✅ complete
├── analysis/        Module D — performance comparison          (Emin)     ✅ complete
├── ai_advisor/      Module E — DEFERRED (see ai_advisor/README.md)
└── tests/           pytest test suite (56 unit tests, 2 integration tests)
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

---

## Optimisation Contract (what Module B provides)

```python
from optimisation import (
    portfolio_return,           # float — w^T · mu
    portfolio_volatility,       # float — sqrt(w^T · Sigma · w)
    sharpe_ratio,               # float — (R_p - r_f) / sigma_p
    minimum_variance_portfolio, # (weights, return, volatility)
    max_sharpe_portfolio,       # (weights, return, volatility, sharpe)
    efficient_frontier,         # (returns array, vols array, weights grid)
    capital_allocation_line,    # (risky_weight, cash_weight, expected_return)
)
```

**End-to-end usage example:**
```python
from data_core import fetch_prices, compute_log_returns, annualized_return
from data_core import covariance_matrix, RISK_FREE_RATE
from optimisation import max_sharpe_portfolio, capital_allocation_line

prices  = fetch_prices()
returns = compute_log_returns(prices)
mu      = annualized_return(returns)
sigma   = covariance_matrix(returns)

weights, r_p, sigma_p, sharpe = max_sharpe_portfolio(mu, sigma, RISK_FREE_RATE)
risky_w, cash_w, r_c = capital_allocation_line(
    target_volatility=0.10,
    tangency_return=r_p,
    tangency_volatility=sigma_p,
    risk_free_rate=RISK_FREE_RATE,
)
```

---

## Analysis Contract (what Module D provides)

```python
from analysis import (
    performance_metrics,      # pd.Series — (ann. return, ann. vol, sharpe)
    portfolio_daily_returns,  # pd.Series — weighted sum of asset log-returns
    apply_cash_allocation,    # pd.Series — blended risky + cash daily returns
    comparison_table,         # pd.DataFrame — side-by-side metrics for N series
)
```

Run the full comparison script:
```bash
python -m analysis.run_comparison
```

---

## Known Data Limitations (read before writing your report)

| # | Limitation | Impact | Mitigation |
|---|-----------|--------|------------|
| 1 | Euronext only publishes top-25 weights publicly | Weights for the remaining 15 constituents (~9.43 % of index) are unavailable without a paid data licence | `load_reference_weights()` provides the 25 confirmed weights for the dashboard concentration chart; all return calculations use `^FCHI` directly |
| 2 | `^FCHI` is a **price-return** index on Yahoo Finance | Understates true index total return by ~2 % p.a. (dividends excluded) | Note in report; the true total-return benchmark would show ~2 pp p.a. higher performance |
| 3 | Constituent list is a static snapshot (verified 2026-06-21) | CAC40 rebalances quarterly; one or two names may change | Historical analysis is run over a fixed past window where the composition was stable |
| 4 | ArcelorMittal uses the Amsterdam listing `MT.AS` | Only non-`.PA` ticker; different public holidays possible | `fetch_prices()` forward-fills across non-trading days, aligning all series |
| 5 | Stellantis `STLAP.PA` is dual-listed (Paris + Milan + NYSE) | Yahoo Finance coverage of the Paris listing can be inconsistent | `fetch_prices()` drops failed tickers with a warning |

---

## Setup Instructions

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

### 3. Install dependencies and the project itself

```bash
pip install -r requirements.txt
pip install -e .
```

The second command (`pip install -e .`) registers all project modules
(`data_core`, `optimisation`, `dashboard`, `analysis`) in the virtual
environment. Without it, Streamlit and other entry points cannot find
the modules. You only need to run it once per machine.

### 4. Verify everything works

```bash
# Unit tests — fast, no network required
pytest tests/ -v -m "not integration"
# Expected: 56 passed

# Integration tests — downloads live data (~30 s)
pytest tests/ -v

# Performance comparison script
python -m analysis.run_comparison

# Streamlit dashboard
streamlit run dashboard/app.py
```

---

## Project Status

| Module | Owner | Status | Tests |
|--------|-------|--------|-------|
| A — data_core | Maxime | ✅ Complete | 13 unit + 6 returns + 2 integration |
| B — optimisation | Yusuf | ✅ Complete | 21 unit |
| C — dashboard | Kris | ✅ Complete | 3 unit |
| D — analysis | Emin | ✅ Complete | 13 unit |
| E — ai_advisor | TBD | ⏸ Deferred pending professor meeting | — |

---

## Branch Policy

Work on your own module branch: `module-a`, `module-b`, `module-c`, `module-d`.
Open a PR into `master` once your module passes tests and has been reviewed
by at least one other team member.

Commit message format: `module-X: short description of what changed`
