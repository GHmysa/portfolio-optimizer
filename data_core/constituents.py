"""
data_core/constituents.py — project constants and CSV loaders.

All constituent data lives in data_core/data/ as CSV files.
This module only reads those files; no ticker strings or weights
are hardcoded here.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Project-wide constants
# ---------------------------------------------------------------------------

# Yahoo Finance ticker for the CAC40 index benchmark series.
BENCHMARK_TICKER: str = "^FCHI"

# Annualised risk-free rate used in Sharpe ratio and Capital Allocation Line.
# Source: ECB deposit facility rate, mid-2026 trajectory. 2.25 % is a
# conservative proxy for short-term EUR cash returns.
RISK_FREE_RATE: float = 0.0225

# ---------------------------------------------------------------------------
# Internal path — resolves relative to this file, not the working directory.
# This means the CSVs load correctly regardless of where you run Python from.
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Ticker and name loaders
# ---------------------------------------------------------------------------

def get_cac40_tickers() -> list[str]:
    """
    Return the 40 Yahoo Finance ticker strings for CAC40 constituents.

    Source: Euronext CAC 40 Stocks page, verified 2026-06-19.
    See data/cac40_tickers.csv for full source notes and per-ticker caveats.
    """
    df = pd.read_csv(_DATA_DIR / "cac40_tickers.csv", comment="#", encoding="utf-8")
    return df["ticker"].tolist()


def get_cac40_names() -> dict[str, str]:
    """Return {yahoo_ticker: company_name} for all 40 constituents."""
    df = pd.read_csv(_DATA_DIR / "cac40_tickers.csv", comment="#", encoding="utf-8")
    return dict(zip(df["ticker"], df["company_name"]))


def get_cac40_sectors() -> dict[str, str]:
    """
    Return {yahoo_ticker: sector_icb} for all 40 constituents.

    Sectors for the 25 PDF-confirmed stocks follow the Euronext composition
    sheet (ICB Level 1). Sectors for the remaining 15 are assigned from
    general ICB Level-1 classification and are not official Euronext data.
    """
    df = pd.read_csv(_DATA_DIR / "cac40_tickers.csv", comment="#", encoding="utf-8")
    return dict(zip(df["ticker"], df["sector_icb"]))


# ---------------------------------------------------------------------------
# Reference weights loader
# ---------------------------------------------------------------------------

def load_reference_weights() -> pd.DataFrame:
    """
    Load the 25 official Euronext index weights (March 31, 2026).

    Returns a DataFrame with columns:
        ticker (str), mnemo (str), company_name (str), weight_pct (float)

    FOR QUALITATIVE USE ONLY — index concentration chart in the dashboard.
    Do not use these weights in return calculations. The CAC40 benchmark
    return is always the ^FCHI price series from fetch_benchmark().

    Only 25 of 40 weights are publicly available; the remaining 15 require
    a licensed Euronext data subscription (see data/cac40_weights_ref.csv).
    """
    return pd.read_csv(
        _DATA_DIR / "cac40_weights_ref.csv", comment="#", encoding="utf-8"
    )
