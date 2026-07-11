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

## Module Contracts

### Module A — data_core (Maxime) ✅ Implemented

**Files:** `data_core/constituents.py`, `data_core/fetcher.py`, `data_core/returns.py`,
`data_core/__init__.py`, `data_core/data/cac40_tickers.csv`,
`data_core/data/cac40_weights_ref.csv`

**Inputs:** Yahoo Finance (network), `data_core/data/cac40_tickers.csv` (CSV, not inline data)

**Outputs (public API — import from `data_core`):**

| Symbol | Type | Description |
|--------|------|-------------|
| `fetch_prices(tickers, start, end)` | `pd.DataFrame` | Adj. close prices. Index = DatetimeIndex, columns = ticker strings. All NaN rows dropped. |
| `fetch_benchmark(start, end)` | `pd.Series` | ^FCHI adj. close. Name = "CAC40". |
| `compute_log_returns(prices)` | same shape as input | Daily log returns. First row dropped. |
| `annualized_return(returns)` | `pd.Series` | μ × 252 per asset. |
| `annualized_volatility(returns)` | `pd.Series` | σ × √252 per asset. |
| `covariance_matrix(returns)` | `pd.DataFrame` | Σ × 252. Shape (n_assets × n_assets). |
| `statistical_moments(returns)` | `pd.DataFrame` | Rows: mean_annual, volatility_annual, skewness, excess_kurtosis. |
| `get_cac40_tickers()` | `list[str]` | 40 Yahoo Finance tickers. Source: Euronext stocks page, verified 2026-06-21. |
| `get_cac40_names()` | `dict[str, str]` | {ticker: company_name}. |
| `get_cac40_sectors()` | `dict[str, str]` | {ticker: ICB sector}. Official for PDF-25; general ICB for remaining 15. |
| `load_reference_weights()` | `pd.DataFrame` | 25 official Euronext weights (March 2026). Columns: ticker, mnemo, company_name, weight_pct. Reference only — do not use in return calculations. |
| `BENCHMARK_TICKER` | `str` | `"^FCHI"` |
| `RISK_FREE_RATE` | `float` | `0.0225` (2.25 % p.a., ECB deposit facility proxy, mid-2026) |
| `TRADING_DAYS_PER_YEAR` | `int` | `252` |

**Do not call yfinance directly from any module other than `data_core/fetcher.py`.**

**Known quality issue:** The `statistical_moments()` docstring incorrectly lists the
return index as `["mean", "volatility", "skewness", "kurtosis"]`. The actual index is
`["mean_annual", "volatility_annual", "skewness", "excess_kurtosis"]`. Code and tests
are correct; only the docstring is wrong.

---

### Module B — optimisation (Yusuf) ✅ Implemented

**Files:** `optimisation/markowitz.py`, `optimisation/cal.py`, `optimisation/__init__.py`

**Inputs (passed as parameters — Module B does not import data_core directly):**
- `mu: pd.Series` — annualised return vector from `data_core.annualized_return()`
- `cov_matrix: pd.DataFrame` — annualised covariance matrix from `data_core.covariance_matrix()`
- `risk_free_rate: float` — use `data_core.RISK_FREE_RATE`

**Outputs (public API — import from `optimisation`):**

| Symbol | Returns | Description |
|--------|---------|-------------|
| `portfolio_return(weights, mu)` | `float` | R_p = w^T · mu |
| `portfolio_volatility(weights, cov_matrix)` | `float` | sigma_p = sqrt(w^T · Sigma · w) |
| `sharpe_ratio(weights, mu, cov_matrix, risk_free_rate)` | `float` | (R_p - r_f) / sigma_p |
| `minimum_variance_portfolio(mu, cov_matrix)` | `(pd.Series, float, float)` | (weights, return, volatility). Weights indexed by ticker. |
| `max_sharpe_portfolio(mu, cov_matrix, risk_free_rate)` | `(pd.Series, float, float, float)` | (weights, return, volatility, sharpe). |
| `efficient_frontier(mu, cov_matrix, n_points=50)` | `(np.ndarray, np.ndarray, np.ndarray)` | (returns, volatilities, weights_grid). Shapes: (n,), (n,), (n, n_assets). |
| `capital_allocation_line(target_volatility, tangency_return, tangency_volatility, risk_free_rate)` | `(float, float, float)` | (risky_weight, cash_weight, expected_return). risky_weight capped at 1.0 — no leverage. |

**Constraints (enforced everywhere):** long-only (w_i ≥ 0), fully-invested (sum = 1).
**Optimisation method:** SciPy SLSQP. Initial guess: equal-weight 1/n.

**Known quality issue:** `tests/test_optimisation.py:25` hardcodes `RISK_FREE_RATE = 0.0225`
instead of `from data_core import RISK_FREE_RATE`. If the rate ever changes in data_core,
the optimisation tests won't pick it up automatically.

---

### Module C — dashboard (Kris) 🔲 Not yet implemented

**Files to create:** `dashboard/app.py` (Streamlit entry point)

**Inputs from optimisation:** efficient frontier arrays, MVP weights, max-Sharpe weights, CAL data.
**Inputs from data_core:** `fetch_prices`, `fetch_benchmark`, `get_cac40_names`.

**Required charts (centrepiece first):**
1. **Cumulative return comparison:** optimised portfolio (with cash via CAL) vs. ^FCHI over
   the same historical period. This is the most important visual for the presentation.
2. **Efficient frontier:** risk/return scatter with MVP and tangency portfolio marked.
3. **Capital Allocation Line:** overlaid on the frontier plot.
4. **Portfolio weights:** pie or bar chart of the tangency portfolio weights.
5. (Optional) **Index concentration:** bar chart of the 25 official Euronext weights.

**Framework:** Streamlit + Plotly.

---

### Module D — analysis (Emin) 🔲 Not yet implemented

**Inputs:** Module A price/return data, Module B optimal portfolio weights.
**Output:** table of (annualised return, annualised volatility, Sharpe ratio)
for: (a) optimised portfolio with cash, (b) CAC40 index, side by side.
Plus a written discussion of the cash allocation's role.

**Note:** ^FCHI is price-return only — acknowledge this limitation explicitly
when comparing against the optimised portfolio.

---

### Module E — ai_advisor (DEFERRED)

See `ai_advisor/README.md`. Do not implement until the scoping questions
are resolved with the professor.

---

## Coding Conventions

### Python style
- **Type hints on all function signatures.** Return types included.
- **Docstrings on every public function.** Include the formula when the function
  implements one (not just what it does — write the maths).
- **No magic numbers.** If you use 252, reference `TRADING_DAYS_PER_YEAR`.
  If you use 0.0225, import `RISK_FREE_RATE` from `data_core`.
- **Prefer pandas / numpy over raw Python loops** for numerical operations.
- **Short functions.** If a function is longer than ~30 lines, consider splitting.

### Testing
- Unit tests use synthetic data — no network calls, fast.
- Integration tests are marked `@pytest.mark.integration` and use real yfinance.
- Run `pytest -m "not integration"` in CI / before every commit.
- Every new public function in `data_core/` needs at least one test.

### Git
- One module per branch. Branch names: `module-a`, `module-b`, `module-c`, `module-d`.
- Commit message format: `module-X: short description of what changed`
- Do not commit `venv/` or `__pycache__/` — both are in `.gitignore`.

---

## Known Data Issues (summary for quick reference)

### Weights — 25 official, 15 not publicly available
The Euronext public PDF discloses only 25 of 40 weights (90.57 % of the index).
The remaining 15 require a paid licence. These weights are stored in
`data_core/data/cac40_weights_ref.csv` for the dashboard concentration chart only.
**No per-stock weights are used in return calculations** — the benchmark
is always `^FCHI` via `fetch_benchmark()`.

### Benchmark — ^FCHI is price-return, not total-return
Yahoo Finance's ^FCHI excludes dividend reinvestment. The true CAC40
total-return index would show ~2 % higher annual performance. Note this
in the analysis when comparing against the optimised portfolio.

### Constituent list — static snapshot (40 tickers confirmed)
The list in `data_core/data/cac40_tickers.csv` was verified from the Euronext
CAC 40 Stocks page on 2026-06-21. AXA (CS.PA) was confirmed via the March 2026
composition PDF. CAC40 rebalances quarterly; the list may drift slightly over time.

### ArcelorMittal — MT.AS (Amsterdam), not MT.PA
The only non-Paris ticker. Yahoo Finance does not consistently carry MT.PA.

### Stellantis — STLAP.PA (verify on first run)
Dual-listed (Paris, Milan, NYSE). Yahoo Finance coverage of the Paris listing
can be inconsistent. `fetch_prices()` will warn and drop it if it fails.
