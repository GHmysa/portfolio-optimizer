"""
Fetches price data from Yahoo Finance via yfinance.

Design decisions
----------------
* We download "Adj Close" prices. Yahoo Finance adjusts for splits and
  dividends, so the price series reflects total return — essential for a
  fair comparison with the ^FCHI index.
* Failed tickers are dropped with a warning rather than crashing, because
  some tickers may have been delisted or have different Yahoo Finance symbols
  than the Euronext mnemonic (e.g. Stellantis: STLAP.PA may redirect).
* Data is downloaded in a single yfinance batch call, which is much faster
  than 40 separate requests.
"""

import warnings
import pandas as pd
import yfinance as yf

from data_core.constituents import BENCHMARK_TICKER, get_cac40_tickers


def fetch_prices(
    tickers: list[str] | None = None,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Download daily adjusted-close prices for a list of tickers.

    Parameters
    ----------
    tickers : list of Yahoo Finance ticker strings.
              Defaults to the full CAC40 constituent list.
    start, end : date strings in "YYYY-MM-DD" format (inclusive).

    Returns
    -------
    pd.DataFrame  — index: trading dates (DatetimeIndex),
                    columns: ticker symbols,
                    values: adjusted close price in EUR (or local currency).
    Tickers that yfinance cannot resolve are silently dropped after a warning.
    """
    if tickers is None:
        tickers = get_cac40_tickers()

    raw: pd.DataFrame = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # gives us adjusted prices directly in "Close"
        progress=False,
    )

    # yfinance returns a MultiIndex when multiple tickers are requested.
    # We only need the "Close" level.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw  # single ticker edge case

    # Drop any column that is entirely NaN — those tickers failed to download.
    failed = prices.columns[prices.isna().all()].tolist()
    if failed:
        warnings.warn(
            f"The following tickers returned no data and were dropped: {failed}\n"
            "Possible causes: wrong Yahoo Finance ticker, delisted stock, or "
            "regional data unavailability. Check data_core/constituents.py.",
            stacklevel=2,
        )
        prices = prices.drop(columns=failed)

    # Forward-fill then drop any remaining leading NaNs.
    # Forward-fill handles non-trading days (public holidays that differ by
    # country — e.g. ArcelorMittal on Euronext Amsterdam may have different
    # closure days than Paris stocks).
    prices = prices.ffill().dropna()

    return prices


def fetch_benchmark(
    start: str = "2019-01-01",
    end: str = "2024-12-31",
) -> pd.Series:
    """
    Download the CAC40 index level (^FCHI) adjusted-close series.

    ^FCHI is the price-return index (not total-return). Euronext also
    publishes a total-return variant (^FCHI-GR) but Yahoo Finance does not
    carry it reliably. For a fair comparison, note this in your report and
    optionally add dividend income to the constituent portfolio returns.

    Returns
    -------
    pd.Series — index: trading dates, name: "CAC40", values: index level.
    """
    raw = yf.download(
        tickers=BENCHMARK_TICKER,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    series = raw["Close"].squeeze()
    series.name = "CAC40"
    return series.dropna()
