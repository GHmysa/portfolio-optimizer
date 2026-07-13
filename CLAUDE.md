# CLAUDE.md — Project Guide for Claude Code

This file documents conventions, module contracts, and known data issues for
the Portfolio Optimizer project. Read it before making any change to any module.

---

## Guiding Principle

**Clarity over cleverness.** Maxime must be able to explain every line of code
in an oral presentation to the professor. Prefer one explicit step over a
clever one-liner. Name variables to match the maths (e.g. `mu`, `sigma`,
`cov_matrix`, `sharpe`). When in doubt, add a comment explaining the formula.

---

## Project Layout

```
portfolio-optimizer/
├── pyproject.toml        makes the project pip-installable (run pip install -e . once)
├── requirements.txt      third-party dependencies
├── pytest.ini            registers the "integration" mark
├── data_core/            Module A — data ingestion & statistics  (Maxime) ✅
├── optimisation/         Module B — Markowitz + CAL              (Yusuf)  ✅
├── dashboard/            Module C — Streamlit UI                 (Kris)   ✅
│   ├── app.py            Streamlit entry point
│   └── _utils.py         compute_cumulative_returns, weights_to_series
├── analysis/             Module D — performance comparison       (Emin)   ✅
│   ├── performance.py    performance_metrics, portfolio_daily_returns,
│   │                     apply_cash_allocation, comparison_table
│   └── run_comparison.py python -m analysis.run_comparison
├── ai_advisor/           Module E — DEFERRED (see ai_advisor/README.md)
└── tests/                56 unit tests + 2 integration tests
```

---

## Module Contracts

### Module A — data_core (Maxime) ✅ Implemented

**Files:** `data_core/constituents.py`, `data_core/fetcher.py`, `data_core/returns.py`,
`data_core/__init__.py`, `data_core/data/cac40_tickers.csv`,
`data_core/data/cac40_weights_ref.csv`

**Inputs:** Yahoo Finance (network), CSV files — no inline data in Python files.

**Outputs (public API — import from `data_core`):**

| Symbol | Type | Description |
|--------|------|-------------|
| `fetch_prices(tickers, start, end)` | `pd.DataFrame` | Adj. close prices. Index = DatetimeIndex, columns = ticker strings. NaN rows dropped. |
| `fetch_benchmark(start, end)` | `pd.Series` | ^FCHI adj. close. Name = "CAC40". |
| `compute_log_returns(prices)` | same shape | Daily log returns r_t = ln(P_t/P_{t-1}). First row dropped. |
| `annualized_return(returns)` | `pd.Series` | μ × 252 per asset. |
| `annualized_volatility(returns)` | `pd.Series` | σ × √252 per asset. |
| `covariance_matrix(returns)` | `pd.DataFrame` | Σ × 252. Shape (n_assets × n_assets). |
| `statistical_moments(returns)` | `pd.DataFrame` | Rows: mean_annual, volatility_annual, skewness, excess_kurtosis. |
| `get_cac40_tickers()` | `list[str]` | 40 Yahoo Finance tickers. Source: Euronext stocks page, verified 2026-06-21. |
| `get_cac40_names()` | `dict[str, str]` | {ticker: company_name}. |
| `get_cac40_sectors()` | `dict[str, str]` | {ticker: ICB sector}. |
| `load_reference_weights()` | `pd.DataFrame` | 25 official Euronext weights (March 2026). Reference only — do not use in return calculations. |
| `BENCHMARK_TICKER` | `str` | `"^FCHI"` |
| `RISK_FREE_RATE` | `float` | `0.0225` (2.25 % p.a., ECB deposit facility proxy, mid-2026) |
| `TRADING_DAYS_PER_YEAR` | `int` | `252` |

**Do not call yfinance directly from any module other than `data_core/fetcher.py`.**

---

### Module B — optimisation (Yusuf) ✅ Implemented

**Files:** `optimisation/markowitz.py`, `optimisation/cal.py`, `optimisation/__init__.py`

**Inputs (passed as parameters — Module B does not import data_core directly):**
- `mu: pd.Series` — from `data_core.annualized_return()`
- `cov_matrix: pd.DataFrame` — from `data_core.covariance_matrix()`
- `risk_free_rate: float` — from `data_core.RISK_FREE_RATE`

**Outputs (public API — import from `optimisation`):**

| Symbol | Returns | Description |
|--------|---------|-------------|
| `portfolio_return(weights, mu)` | `float` | R_p = w^T · mu |
| `portfolio_volatility(weights, cov_matrix)` | `float` | sigma_p = sqrt(w^T · Sigma · w) |
| `sharpe_ratio(weights, mu, cov_matrix, risk_free_rate)` | `float` | (R_p - r_f) / sigma_p |
| `minimum_variance_portfolio(mu, cov_matrix)` | `(pd.Series, float, float)` | (weights, return, volatility) |
| `max_sharpe_portfolio(mu, cov_matrix, risk_free_rate)` | `(pd.Series, float, float, float)` | (weights, return, volatility, sharpe) |
| `efficient_frontier(mu, cov_matrix, n_points=50)` | `(np.ndarray, np.ndarray, np.ndarray)` | (returns, volatilities, weights_grid). Shapes: (n,), (n,), (n, n_assets). |
| `capital_allocation_line(target_volatility, tangency_return, tangency_volatility, risk_free_rate)` | `(float, float, float)` | (risky_weight, cash_weight, expected_return). risky_weight capped at 1.0. |

**Constraints:** long-only (w_i ≥ 0), fully-invested (sum = 1). Method: SciPy SLSQP.

---

### Module C — dashboard (Kris) ✅ Implemented

**Files:** `dashboard/app.py`, `dashboard/_utils.py`

**Entry point:** `streamlit run dashboard/app.py` (requires `pip install -e .` first)

**Sidebar controls:**
- Demo mode checkbox (bypasses live data with synthetic values)
- Start / End date pickers (default: last 365 days)
- Efficient frontier points slider (20–200, default 50)
- Cash allocation slider (0.0–1.0, sets the CAL allocation point)

**Charts rendered:**
1. Cumulative return comparison — optimised portfolio vs. ^FCHI (centrepiece)
2. Efficient frontier + CAL — with MVP (green), tangency (red), CAL dashed line, and live CAL allocation dot
3. Tangency portfolio weights — bar chart with % labels
4. Index concentration — bar chart of the 25 official Euronext weights

**Key implementation notes:**
- Uses `compute_log_returns()` + `compute_cumulative_returns()` throughout — consistent with Modules A and D.
- Uses `@st.cache_data(ttl=3600)` so repeated sidebar interactions do not re-download prices.
- Falls back to `safe_run_demo()` if live data fails, so the UI never crashes.

**`dashboard/_utils.py` public helpers:**

| Symbol | Description |
|--------|-------------|
| `compute_cumulative_returns(returns)` | `exp(r).cumprod()` — always treats input as log-returns |
| `weights_to_series(weights, tickers)` | Returns pd.Series sorted descending for plotting |

---

### Module D — analysis (Emin) ✅ Implemented

**Files:** `analysis/performance.py`, `analysis/run_comparison.py`, `analysis/__init__.py`

**Inputs:** Module A log-returns, Module B optimal portfolio weights.

**Outputs (public API — import from `analysis`):**

| Symbol | Returns | Description |
|--------|---------|-------------|
| `performance_metrics(daily_returns, risk_free_rate)` | `pd.Series` | (annualized_return, annualized_volatility, sharpe_ratio) |
| `portfolio_daily_returns(asset_returns, weights)` | `pd.Series` | r_p,t = w^T · r_t (fixed-weight daily rebalancing) |
| `apply_cash_allocation(portfolio_returns, risky_weight, risk_free_rate)` | `pd.Series` | r_c,t = w_risky · r_p,t + (1 − w_risky) · r_f/252 |
| `comparison_table(named_returns, risk_free_rate)` | `pd.DataFrame` | Side-by-side metrics; one row per series |

**Run the comparison pipeline:**
```bash
python -m analysis.run_comparison
```

Sample output (2019–2024, in-sample):
```
Max-Sharpe portfolio (fully invested)   23.8 %   12.0 %   Sharpe 1.81
Max-Sharpe + cash (16 % cash)           20.3 %   10.0 %   Sharpe 1.81
CAC40 index (^FCHI)                      7.3 %   17.8 %   Sharpe 0.26
```

**Note:** ^FCHI is price-return only — the true total-return index would show ~2 pp p.a. higher, which should be acknowledged in the written report.

---

### Module E — ai_advisor (DEFERRED)

See `ai_advisor/README.md`. Do not implement until the scoping questions
are resolved with the professor.

---

## Coding Conventions

### Python style
- **Type hints on all function signatures.** Return types included.
- **Docstrings on every public function.** Include the formula when the function implements one.
- **No magic numbers.** Reference `TRADING_DAYS_PER_YEAR` and import `RISK_FREE_RATE` from `data_core`.
- **Prefer pandas / numpy over raw Python loops** for numerical operations.
- **Short functions.** If a function is longer than ~30 lines, consider splitting.

### Testing
- Unit tests use synthetic data — no network calls, fast.
- Integration tests are marked `@pytest.mark.integration` and use real yfinance.
- Run `pytest -m "not integration"` before every commit.
- Every new public function in `data_core/` needs at least one test.

### Git
- One module per branch. Branch names: `module-a`, `module-b`, `module-c`, `module-d`.
- Commit message format: `module-X: short description of what changed`
- Do not commit `venv/` or `__pycache__/` — both are in `.gitignore`.

---

## Known Data Issues (summary for quick reference)

### Weights — 25 official, 15 not publicly available
The Euronext public PDF discloses only 25 of 40 weights (90.57 % of the index).
The remaining 15 require a paid licence. These are in `data_core/data/cac40_weights_ref.csv`
for the dashboard concentration chart only. No per-stock weights are used in return
calculations — the benchmark is always `^FCHI` via `fetch_benchmark()`.

### Benchmark — ^FCHI is price-return, not total-return
Yahoo Finance's ^FCHI excludes dividend reinvestment (~2 % p.a. lower than true total return).
Acknowledge this in the analysis when comparing against the optimised portfolio.

### Constituent list — static snapshot
Verified from Euronext CAC 40 Stocks page on 2026-06-21. AXA (CS.PA) confirmed via
the March 2026 composition PDF. CAC40 rebalances quarterly; the list may drift over time.

### ArcelorMittal — MT.AS (Amsterdam), not MT.PA
The only non-Paris ticker. Yahoo Finance does not consistently carry MT.PA.

### Stellantis — STLAP.PA (verify on first run)
Dual-listed (Paris, Milan, NYSE). Yahoo Finance coverage of the Paris listing
can be inconsistent. `fetch_prices()` will warn and drop it if it fails.
