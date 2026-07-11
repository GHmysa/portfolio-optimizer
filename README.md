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
├── data_core/       Module A — data ingestion & statistics   (Maxime)   ✅ complete
├── optimisation/    Module B — Markowitz + Capital Allocation Line (Yusuf) ✅ complete
├── dashboard/       Module C — Streamlit UI                  (Kris)     🔲 not started
├── analysis/        Module D — performance comparison        (Emin)     🔲 not started
├── ai_advisor/      Module E — DEFERRED (see ai_advisor/README.md)
└── tests/           pytest test suite (40 unit tests, 2 integration tests)
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

## Optimisation Contract (what Module B provides)

Import from `optimisation` directly:

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

**How to call from real data:**
```python
from data_core import (
    fetch_prices, compute_log_returns, annualized_return,
    covariance_matrix, RISK_FREE_RATE
)
from optimisation import max_sharpe_portfolio, capital_allocation_line

prices  = fetch_prices()
returns = compute_log_returns(prices)
mu      = annualized_return(returns)
sigma   = covariance_matrix(returns)

weights, r_p, sigma_p, sharpe = max_sharpe_portfolio(mu, sigma, RISK_FREE_RATE)
risky_w, cash_w, r_c = capital_allocation_line(
    target_volatility=0.12,
    tangency_return=r_p,
    tangency_volatility=sigma_p,
    risk_free_rate=RISK_FREE_RATE,
)
```

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

# Include live yfinance tests (requires internet):
pytest tests/ -v
```

You should see **40 tests pass** with no failures before touching any code.

### 5. Verify Modules A + B work end-to-end

```python
from data_core import fetch_prices, fetch_benchmark, compute_log_returns
from data_core import annualized_return, covariance_matrix, RISK_FREE_RATE
from optimisation import minimum_variance_portfolio, max_sharpe_portfolio, efficient_frontier

prices  = fetch_prices()                    # ~15 s, downloads 40 tickers
returns = compute_log_returns(prices)
mu      = annualized_return(returns)
sigma   = covariance_matrix(returns)

mvp_weights, mvp_return, mvp_vol = minimum_variance_portfolio(mu, sigma)
tan_weights, tan_return, tan_vol, sharpe = max_sharpe_portfolio(mu, sigma, RISK_FREE_RATE)
ef_returns, ef_vols, ef_weights = efficient_frontier(mu, sigma, n_points=50)

print(f"MVP:        return={mvp_return:.1%}  vol={mvp_vol:.1%}")
print(f"Tangency:   return={tan_return:.1%}  vol={tan_vol:.1%}  Sharpe={sharpe:.2f}")
```

### Branch policy

Each module lives on its own branch: `module-a`, `module-b`, `module-c`, `module-d`.
Open a PR into `master` once your module passes tests and has been reviewed
by at least one other team member.

---

## Running the Dashboard (Module C — not yet built)

```bash
streamlit run dashboard/app.py
```

Kris: implement `dashboard/app.py`. See `CLAUDE.md` for the required charts.

---

## Project Status

| Module | Owner | Status | Branch |
|--------|-------|--------|--------|
| A — data_core | Maxime | ✅ Complete, 19 unit tests + 2 integration | `module-a` → merged |
| B — optimisation | Yusuf | ✅ Complete, 21 unit tests | `module-b` → merged |
| C — dashboard | Kris | 🔲 Not started | `module-c` |
| D — analysis | Emin | 🔲 Not started | `module-d` |
| E — ai_advisor | TBD | ⏸ Deferred pending professor meeting | — |
