"""
scripts/health_check.py — one-time manual verification script.

Run this before sharing the dashboard with the professor:

    python -m scripts.health_check

This is NOT part of the automated test suite (pytest). It hits live
Yahoo Finance and Wikipedia, takes a few minutes, and prints a
human-readable report covering:

    1. Ticker verification — all 40 tickers resolve on yfinance, and
       their company names are cross-checked against an independently
       scraped source (Wikipedia's CAC 40 constituent table).
    2. Edge-case stress test — the actual fetch -> returns -> covariance
       -> max_sharpe_portfolio pipeline run against scenarios that are
       easy to trigger by accident in the dashboard (tiny max_weight,
       very short date ranges, cash-allocation boundaries, the
       URW.PA history cutoff, and a large efficient frontier).

Optional dependencies used only by this script (not requirements.txt):
    pip install beautifulsoup4
(requests is already pulled in transitively by yfinance.)
"""
from __future__ import annotations

import sys
import time
import warnings
import unicodedata
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Force UTF-8 stdout so accented company names (Crédit Agricole, L'Oréal,
# Société Générale, ...) never crash this script on a Windows cp1252 console.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data_core import (
    fetch_prices,
    get_cac40_tickers,
    get_cac40_names,
    compute_log_returns,
    annualized_return,
    covariance_matrix,
    RISK_FREE_RATE,
)
from optimisation import max_sharpe_portfolio, efficient_frontier
from analysis import apply_cash_allocation, portfolio_daily_returns

WIKIPEDIA_CAC40_URL = "https://en.wikipedia.org/wiki/CAC_40"


def hr(title: str = "") -> None:
    print()
    print("=" * 78)
    if title:
        print(title)
        print("=" * 78)


def normalize_name(name: str) -> str:
    """Lowercase and strip accents/punctuation, for fuzzy company-name matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return "".join(c for c in ascii_name.lower() if c.isalnum())


# ---------------------------------------------------------------------------
# Section 1 — ticker verification
# ---------------------------------------------------------------------------

def fetch_wikipedia_cac40() -> dict[str, str]:
    """Return {ticker: company_name} scraped from Wikipedia's CAC 40 page.

    Used as an independent cross-check for data_core/data/cac40_tickers.csv.
    Raises on network/parse failure — caller decides how to report that.
    """
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(WIKIPEDIA_CAC40_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Ticker" in headers and "Company" in headers:
            result: dict[str, str] = {}
            for row in table.find_all("tr")[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if len(cells) >= 4:
                    company, ticker = cells[0], cells[3]
                    result[ticker] = company
            return result
    raise RuntimeError("Could not find a Company/Ticker table on the Wikipedia CAC 40 page.")


def check_tickers() -> list[str]:
    """Section 1: validate all 40 tickers via yfinance, cross-check names vs Wikipedia."""
    hr("SECTION 1 -- TICKER VERIFICATION")
    tickers = get_cac40_tickers()
    names = get_cac40_names()
    print(f"Loaded {len(tickers)} tickers from data_core/data/cac40_tickers.csv")

    invalid: list[str] = []
    print(f"\n{'Ticker':<10} {'Status':<6} {'Our name':<28}")
    print("-" * 78)
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            ok = info.get("quoteType") == "EQUITY" and bool(
                info.get("longName") or info.get("shortName")
            )
        except Exception:
            ok = False
        status = "OK" if ok else "FAIL"
        if not ok:
            invalid.append(t)
        print(f"{t:<10} {status:<6} {names.get(t, ''):<28}")
        time.sleep(0.3)  # be polite to Yahoo's quoteSummary endpoint

    print(f"\n{len(tickers) - len(invalid)}/{len(tickers)} tickers verified live on yfinance.")
    if invalid:
        print(f"FAILED tickers: {invalid}")

    print("\n--- Cross-check against Wikipedia's CAC 40 constituent table ---")
    try:
        wiki = fetch_wikipedia_cac40()
        print(f"Fetched {len(wiki)} constituents from Wikipedia.\n")

        mismatches = []
        for t in tickers:
            our_name = names.get(t, "")
            wiki_name = wiki.get(t)
            if wiki_name is None:
                mismatches.append((t, our_name, None))
                continue
            a, b = normalize_name(our_name), normalize_name(wiki_name)
            if a not in b and b not in a:
                mismatches.append((t, our_name, wiki_name))

        drifted = [m for m in mismatches if m[2] is None]
        renamed = [m for m in mismatches if m[2] is not None]
        extra_in_wiki = sorted(set(wiki) - set(tickers))

        if renamed:
            print("Name mismatches (same ticker, different name):")
            for t, ours, wiki_name in renamed:
                print(f"  {t:<10} ours='{ours}'  wikipedia='{wiki_name}'")
        if drifted:
            print("\nTickers in our CSV but NOT on Wikipedia's current CAC40 list "
                  "(possible composition drift since the 2026-06-19 snapshot):")
            for t, ours, _ in drifted:
                print(f"  {t:<10} ours='{ours}'")
        if not renamed and not drifted:
            print("No name mismatches for tickers present in both lists.")

        if extra_in_wiki:
            print("\nTickers on Wikipedia's current CAC40 list but NOT in our CSV "
                  "(possible new entrants we are missing):")
            for t in extra_in_wiki:
                print(f"  {t:<10} '{wiki[t]}'")
    except Exception as e:
        print(f"Could not cross-check against Wikipedia: {type(e).__name__}: {e}")
        print("(Non-fatal -- the yfinance ticker validity above still stands.)")

    return invalid


# ---------------------------------------------------------------------------
# Section 2 — edge-case stress test
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    name: str
    status: str  # "PASS", "FAIL", or "SKIP"
    detail: str
    elapsed: float = 0.0


def sanity_check_portfolio(weights: pd.Series, r: float, sigma: float, sharpe: float) -> Optional[str]:
    """Return a description of the problem, or None if the result looks sane."""
    if weights.isna().any():
        return "weights contain NaN"
    if not np.isclose(weights.sum(), 1.0, atol=1e-4):
        return f"weights sum to {weights.sum():.6f}, not 1.0"
    if (weights < -1e-6).any():
        return f"negative weight(s) found: {weights[weights < -1e-6].to_dict()}"
    if not np.isfinite(r):
        return f"non-finite expected return: {r}"
    if not np.isfinite(sigma) or sigma < 0:
        return f"invalid volatility: {sigma}"
    if not np.isfinite(sharpe):
        return f"non-finite Sharpe ratio: {sharpe}"
    return None


def scenario_infeasible_max_weight(prices: pd.DataFrame) -> ScenarioResult:
    name = "2a. Infeasible max_weight (0.02)"
    returns = compute_log_returns(prices)
    mu = annualized_return(returns)
    cov = covariance_matrix(returns)
    max_weight = 0.02
    t0 = time.perf_counter()
    try:
        max_sharpe_portfolio(mu, cov, RISK_FREE_RATE, max_weight=max_weight)
        elapsed = time.perf_counter() - t0
        return ScenarioResult(
            name, "FAIL",
            f"Expected a ValueError (max_weight={max_weight} too small for "
            f"{len(mu)} assets), but it returned a result instead. That means "
            "an infeasible constraint is silently producing a portfolio.",
            elapsed,
        )
    except ValueError as e:
        elapsed = time.perf_counter() - t0
        return ScenarioResult(name, "PASS", f"Raised ValueError as expected: {e}", elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return ScenarioResult(
            name, "FAIL",
            f"Raised {type(e).__name__} instead of the expected ValueError: {e}. "
            f"Inputs: max_weight={max_weight}, n_assets={len(mu)}.",
            elapsed,
        )


def scenario_short_date_range(label: str, start: str, end: str, max_weight: float = 0.15) -> ScenarioResult:
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prices = fetch_prices(start=start, end=end)
        returns = compute_log_returns(prices)
        mu = annualized_return(returns)
        cov = covariance_matrix(returns)
        weights, r, sigma, sharpe = max_sharpe_portfolio(mu, cov, RISK_FREE_RATE, max_weight=max_weight)
        elapsed = time.perf_counter() - t0
        problem = sanity_check_portfolio(weights, r, sigma, sharpe)
        drop_note = f" [{len(caught)} ticker-drop warning(s)]" if caught else ""
        if problem:
            return ScenarioResult(
                label, "FAIL",
                f"{problem}. Inputs: start={start}, end={end}, max_weight={max_weight}, "
                f"n_assets={len(mu)}, n_trading_days={len(returns)}.",
                elapsed,
            )
        return ScenarioResult(
            label, "PASS",
            f"return={r:.2%}, vol={sigma:.2%}, sharpe={sharpe:.2f}, "
            f"n_assets={len(mu)}, n_trading_days={len(returns)}{drop_note}.",
            elapsed,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return ScenarioResult(
            label, "FAIL",
            f"{type(e).__name__}: {e}. Inputs: start={start}, end={end}, max_weight={max_weight}.",
            elapsed,
        )


def scenario_cash_allocation_boundaries(prices: pd.DataFrame, max_weight: float = 0.15) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    returns = compute_log_returns(prices)
    mu = annualized_return(returns)
    cov = covariance_matrix(returns)
    weights, r, sigma, sharpe = max_sharpe_portfolio(mu, cov, RISK_FREE_RATE, max_weight=max_weight)
    port_returns = portfolio_daily_returns(returns, weights)

    for risky_w in (0.0, 1.0):
        label = f"2d. Cash allocation boundary (risky_weight={risky_w})"
        t0 = time.perf_counter()
        try:
            blended = apply_cash_allocation(port_returns, risky_w, RISK_FREE_RATE)
            elapsed = time.perf_counter() - t0
            if blended.isna().any():
                results.append(ScenarioResult(label, "FAIL", "Blended return series contains NaN.", elapsed))
                continue
            ann_return = blended.mean() * 252
            ann_vol = blended.std() * np.sqrt(252)
            if risky_w == 0.0 and not np.isclose(ann_vol, 0.0, atol=1e-6):
                results.append(ScenarioResult(
                    label, "FAIL",
                    f"Expected ~0 volatility at risky_weight=0.0 (all cash), got {ann_vol:.6f}.",
                    elapsed,
                ))
                continue
            if risky_w == 1.0 and not np.isclose(ann_vol, sigma, atol=1e-6):
                results.append(ScenarioResult(
                    label, "FAIL",
                    f"Expected volatility to match the tangency portfolio "
                    f"({sigma:.4f}) at risky_weight=1.0, got {ann_vol:.6f}.",
                    elapsed,
                ))
                continue
            results.append(ScenarioResult(
                label, "PASS", f"annualised return={ann_return:.2%}, vol={ann_vol:.2%}.", elapsed
            ))
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append(ScenarioResult(label, "FAIL", f"{type(e).__name__}: {e}. Input: risky_weight={risky_w}.", elapsed))
    return results


def scenario_urw_present() -> ScenarioResult:
    label = "2e. URW.PA should NOT be dropped (2023-06-01 to 2024-12-31)"
    start, end = "2023-06-01", "2024-12-31"
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prices = fetch_prices(start=start, end=end)
        elapsed = time.perf_counter() - t0
        messages = [str(w.message) for w in caught]
        urw_in_warning = [m for m in messages if "URW.PA" in m]
        urw_in_columns = "URW.PA" in prices.columns
        if urw_in_warning or not urw_in_columns:
            reason = (
                f"was flagged in a drop warning: {'; '.join(urw_in_warning)}"
                if urw_in_warning else "is missing from the output columns"
            )
            return ScenarioResult(
                label, "FAIL",
                f"URW.PA {reason}, even though this window (start={start}) is well "
                "after its Yahoo Finance listing date. Inputs: "
                f"start={start}, end={end}.",
                elapsed,
            )
        return ScenarioResult(
            label, "PASS",
            f"URW.PA present in output columns ({len(prices)} trading days), "
            "no drop-warning mentioned it.",
            elapsed,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return ScenarioResult(label, "FAIL", f"{type(e).__name__}: {e}. Inputs: start={start}, end={end}.", elapsed)


def scenario_frontier_timing(prices: pd.DataFrame, max_weight: float = 0.15, n_points: int = 200) -> ScenarioResult:
    label = f"2f. Efficient frontier timing (n_points={n_points})"
    t0 = time.perf_counter()
    try:
        returns = compute_log_returns(prices)
        mu = annualized_return(returns)
        cov = covariance_matrix(returns)
        t0 = time.perf_counter()
        rets, vols, _ = efficient_frontier(mu, cov, n_points=n_points, max_weight=max_weight)
        elapsed = time.perf_counter() - t0
        if np.isnan(rets).any() or np.isnan(vols).any():
            return ScenarioResult(label, "FAIL", f"NaN values in frontier output after {elapsed:.2f}s.", elapsed)
        note = " -- may feel slow on the interactive dashboard slider" if elapsed > 15 else ""
        return ScenarioResult(label, "PASS", f"Computed {n_points} frontier points in {elapsed:.2f}s{note}.", elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return ScenarioResult(label, "FAIL", f"{type(e).__name__}: {e}. Inputs: n_points={n_points}, max_weight={max_weight}.", elapsed)


def run_stress_tests() -> list[ScenarioResult]:
    hr("SECTION 2 -- EDGE CASE STRESS TEST")
    results: list[ScenarioResult] = []

    print("Fetching the main 2019-2024 window once, reused by several scenarios...")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            main_prices = fetch_prices(start="2019-01-01", end="2024-12-31")
        for w in caught:
            print(f"  (drop warning during main fetch: {w.message})")
        print(f"Main window: {main_prices.shape[1]} tickers, {len(main_prices)} trading days.\n")
    except Exception as e:
        print(f"Could not fetch the main window: {type(e).__name__}: {e}")
        print("Scenarios that depend on it (2a, 2d, 2f) will be skipped.\n")
        main_prices = None

    if main_prices is not None:
        results.append(scenario_infeasible_max_weight(main_prices))
    else:
        results.append(ScenarioResult("2a. Infeasible max_weight (0.02)", "SKIP", "Main window fetch failed."))

    results.append(scenario_short_date_range(
        "2b. 3-month date range (2024-10-01 to 2024-12-31)", "2024-10-01", "2024-12-31"
    ))
    results.append(scenario_short_date_range(
        "2c. 6-month date range (2024-07-01 to 2024-12-31)", "2024-07-01", "2024-12-31"
    ))

    if main_prices is not None:
        results.extend(scenario_cash_allocation_boundaries(main_prices))
    else:
        results.append(ScenarioResult("2d. Cash allocation boundary (risky_weight=0.0)", "SKIP", "Main window fetch failed."))
        results.append(ScenarioResult("2d. Cash allocation boundary (risky_weight=1.0)", "SKIP", "Main window fetch failed."))

    results.append(scenario_urw_present())

    if main_prices is not None:
        results.append(scenario_frontier_timing(main_prices))
    else:
        results.append(ScenarioResult("2f. Efficient frontier timing (n_points=200)", "SKIP", "Main window fetch failed."))

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main() -> None:
    print("PORTFOLIO OPTIMIZER -- MANUAL HEALTH CHECK")
    print(f"Run at: {pd.Timestamp.now()}")
    print("(Live Yahoo Finance + Wikipedia calls -- this takes a few minutes.)")

    invalid_tickers = check_tickers()
    results = run_stress_tests()

    hr("SUMMARY")
    print(f"{'Scenario':<58} {'Status':<6} {'Time':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r.name:<58} {r.status:<6} {r.elapsed:>7.2f}s")

    problems = [r for r in results if r.status != "PASS"]
    if problems:
        print()
        print("--- DETAILS FOR NON-PASSING SCENARIOS ---")
        for r in problems:
            print(f"\n[{r.status}] {r.name}")
            print(f"    {r.detail}")

    print()
    print("-" * 78)
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    if n_fail == 0 and n_skip == 0 and not invalid_tickers:
        print("ALL CHECKS PASSED.")
    else:
        print(
            f"{n_fail} scenario(s) FAILED, {n_skip} SKIPPED, "
            f"{len(invalid_tickers)} ticker(s) failed yfinance validation. "
            "See details above."
        )


if __name__ == "__main__":
    main()
