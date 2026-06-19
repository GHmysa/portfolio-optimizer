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

### Module A — data_core (Maxime)

**Inputs:** Yahoo Finance (network), `data_core/constituents.py` (hardcoded table)

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
| `get_cac40_tickers()` | `list[str]` | 39 Yahoo Finance tickers. |
| `get_cac40_weights()` | `dict[str, float]` | {ticker: weight_%}. Sum = 100.00 %. |
| `get_cac40_names()` | `dict[str, str]` | {ticker: company_name}. |
| `get_estimated_tickers()` | `set[str]` | Tickers whose weights came from market cap (not Euronext). |
| `BENCHMARK_TICKER` | `str` | `"^FCHI"` |
| `RISK_FREE_RATE` | `float` | `0.0225` (2.25 % p.a.) |
| `TRADING_DAYS_PER_YEAR` | `int` | `252` |

**Do not call yfinance directly from any module other than `data_core/fetcher.py`.**

---

### Module B — optimisation (Yusuf)

**Inputs from data_core:**
- `covariance_matrix(returns)` → Σ (annualised, shape n×n)
- `annualized_return(returns)` → μ vector (shape n,)
- `RISK_FREE_RATE` → r_f scalar

**Outputs (to be defined by Yusuf):**
- `minimum_variance_portfolio()` → weights array + (return, volatility)
- `max_sharpe_portfolio()` → weights array + Sharpe ratio
- `efficient_frontier(n_points)` → arrays of (returns, volatilities, weights)
- `capital_allocation_line(target_volatility)` → (risky weight, cash weight)

**Constraints:** all weights ≥ 0 (long-only), weights sum to 1.

---

### Module C — dashboard (Kris)

**Inputs from optimisation:** efficient frontier arrays, MVP weights, max-Sharpe weights, CAL data.
**Inputs from data_core:** `fetch_prices`, `fetch_benchmark`, `get_cac40_names`.

**Key chart:** optimised portfolio (with cash) cumulative return vs. ^FCHI over
the same historical period. This is the centrepiece visual.

**Framework:** Streamlit + Plotly.

---

### Module D — analysis (Emin)

**Inputs:** Module A price/return data, Module B optimal portfolio weights.
**Output:** table of (annualised return, annualised volatility, Sharpe ratio)
for: (a) optimised portfolio with cash, (b) CAC40 index, side by side.
Plus a written discussion of the cash allocation's role.

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

### Weights — bottom 14 constituents
The Euronext PDF only shows the top 25. The bottom 14 use *total* market cap
from yfinance (2026-06-19 snapshot), normalised to the residual 9.43 %.
**Crédit Agricole (ACA.PA) is flagged:** its free-float is ~41 % of total,
so our 2.23 % weight is likely 1–2 pp too high vs. the official index weight.
Use `get_estimated_tickers()` to flag these in the dashboard if needed.

### Benchmark — ^FCHI is price-return, not total-return
Yahoo Finance's ^FCHI excludes dividend reinvestment. The true CAC40
total-return index would show ~2 % higher annual performance. Note this
in the analysis when comparing against the optimised portfolio.

### Constituent list — 39, not 40
Wikipedia returned 39 companies; the 40th is unidentified. Its weight is
~0.3 % and will not materially affect optimisation.

### ArcelorMittal — MT.AS (Amsterdam), not MT.PA
The only non-Paris ticker. Yahoo Finance does not consistently carry MT.PA.
