"""
analysis/positioning.py — Module D: overweight/underweight analysis and
rule-based commentary ("AI advice").

Compares the optimised portfolio's weights (Module B) against the CAC40's
official index weights (Module A's 25-ticker Euronext reference) to show
which stocks the optimiser is betting for or against relative to the
index, and explains WHY using simple, explicit if/then rules over
statistics we already compute (Sharpe ratio, volatility, skewness,
correlation to the rest of the portfolio).

This is rule-based commentary, NOT machine learning: every threshold is a
named module-level constant, documented below and in generate_commentary(),
so it can be defended individually rather than treated as a black box.
"""
import pandas as pd
from scipy import stats

from data_core import RISK_FREE_RATE, annualized_return, annualized_volatility
from analysis.performance import portfolio_daily_returns

# Part A: delta threshold, in percentage points, above/below which a stock
# is classified Overweight/Underweight rather than Neutral. 1.0 point
# mirrors the granularity Euronext itself reports at (weights to 2 decimal
# places) while staying well above the noise floor of quarterly rebalancing.
DELTA_THRESHOLD_PCT = 1.0

# Part B, rules 1-3: "high"/"below-median" Sharpe and volatility are
# defined as PERCENTILE RANKS within the current fetch's universe (however
# many tickers survived data_core.fetch_prices() for the selected date
# range) rather than fixed numbers. A fixed cutoff (e.g. "Sharpe > 1.0")
# would stop making sense across different date ranges -- a 3-month window
# and a 6-year window produce very different Sharpe magnitudes for the same
# stock. Ranking within the same period's universe is self-normalising.
SHARPE_HIGH_PERCENTILE = 0.75      # top quartile of the universe's Sharpe ratios
SHARPE_MEDIAN_PERCENTILE = 0.50    # median of the universe's Sharpe ratios
VOLATILITY_HIGH_PERCENTILE = 0.75  # top quartile of the universe's volatilities
CORRELATION_LOW_PERCENTILE = 0.25  # bottom quartile of correlation-to-rest-of-portfolio

# Part B, rule 4: skewness uses an ABSOLUTE threshold, unlike the others,
# because skewness has a standard, dataset-independent interpretation
# (Bulmer, 1979): distributions with skewness in [-0.5, 0.5] are "fairly
# symmetric"; beyond that they are "moderately to highly skewed". This
# convention does not depend on comparing stocks to each other, so ranking
# it within the universe would add noise rather than remove it.
SKEWNESS_NEGATIVE_THRESHOLD = -0.5


# ---------------------------------------------------------------------------
# Per-ticker statistics feeding the commentary rules
# ---------------------------------------------------------------------------

def compute_stock_stats(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    Per-ticker statistics used by generate_commentary(), for every ticker
    in `returns` (the full fetched universe for the selected date range,
    not just the 25 tickers with an official index weight).

    `weights` must be indexed by the same tickers as `returns.columns`
    (as Module B's max_sharpe_portfolio() output already is) -- this is
    the same contract portfolio_daily_returns() already enforces, and it
    is reused here rather than re-implemented.

    Returns
    -------
    pd.DataFrame indexed by ticker, columns:
        mean_annual              — annualised return (data_core.annualized_return)
        volatility_annual        — annualised volatility (data_core.annualized_volatility)
        sharpe                   — (mean_annual - RISK_FREE_RATE) / volatility_annual
        sharpe_percentile        — percentile rank of `sharpe` (0-1, pandas rank(pct=True))
        volatility_percentile    — percentile rank of `volatility_annual` (0-1)
        skewness                 — third standardised moment of daily returns (scipy.stats.skew)
        correlation_to_rest      — correlation of this stock's daily returns to the
                                    REST of the portfolio, i.e. the portfolio with this
                                    stock's own contribution removed and the remaining
                                    weights renormalised:

                                        r_rest,t = (r_p,t - w_i * r_i,t) / (1 - w_i)

                                    Correlating a stock against a portfolio that already
                                    includes itself would inflate the number (a stock is
                                    trivially correlated with its own contribution); this
                                    measures actual diversification benefit against
                                    everything else being held. NaN if w_i is
                                    (numerically) 1.0 -- there is no "rest" to compare
                                    against a single-stock portfolio.
        correlation_percentile   — percentile rank of `correlation_to_rest` (0-1);
                                    NaN propagates (pandas rank() default: keep NaN as NaN).

    Percentiles are computed across whichever tickers are present in
    `returns` for the CURRENT fetch, so they automatically re-normalise to
    whatever date range the user selects.
    """
    mu = annualized_return(returns)
    vol = annualized_volatility(returns)
    sharpe = (mu - RISK_FREE_RATE) / vol
    skew = returns.apply(stats.skew)

    portfolio_returns = portfolio_daily_returns(returns, weights)

    correlation_to_rest = pd.Series(index=returns.columns, dtype=float)
    for ticker in returns.columns:
        w_i = weights.get(ticker, 0.0)
        if w_i >= 1.0 - 1e-9:
            correlation_to_rest[ticker] = float("nan")
            continue
        rest_returns = (portfolio_returns - w_i * returns[ticker]) / (1.0 - w_i)
        correlation_to_rest[ticker] = returns[ticker].corr(rest_returns)

    stats_df = pd.DataFrame({
        "mean_annual": mu,
        "volatility_annual": vol,
        "sharpe": sharpe,
        "skewness": skew,
        "correlation_to_rest": correlation_to_rest,
    })
    stats_df["sharpe_percentile"] = stats_df["sharpe"].rank(pct=True)
    stats_df["volatility_percentile"] = stats_df["volatility_annual"].rank(pct=True)
    stats_df["correlation_percentile"] = stats_df["correlation_to_rest"].rank(pct=True)
    return stats_df


# ---------------------------------------------------------------------------
# Part A — overweight / underweight table
# ---------------------------------------------------------------------------

def compute_overweight_table(weights: pd.Series, reference_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Compare optimised portfolio weights to the CAC40's official index
    weights.

    Every ticker the portfolio holds is included, not just the ones with
    an official weight. Only 25 of the 40 CAC40 constituents have a
    publicly available Euronext weight (data_core.load_reference_weights());
    for the rest, index_weight_pct and delta_pct are NaN and classification
    is "No index reference available" rather than silently dropping them
    from the table -- otherwise the table would hide part of the portfolio.

    Formula
    -------
        delta_pct = optimized_weight * 100 - index_weight_pct

    `weights` values are fractions (0-1, sum to 1); `reference_weights`'s
    `weight_pct` column is already on a 0-100 scale (Euronext's published
    format). The *100 conversion is required -- comparing a 0-1 number to
    a 0-100 number directly would make almost every stock look wrongly
    "underweight" by two orders of magnitude.

    Classification
    --------------
        delta_pct >  +DELTA_THRESHOLD_PCT               -> "Overweight"
        delta_pct <  -DELTA_THRESHOLD_PCT               -> "Underweight"
        otherwise (including exactly +-DELTA_THRESHOLD_PCT) -> "Neutral"
        index_weight_pct is NaN (no reference weight)   -> "No index reference available"

    Returns
    -------
    pd.DataFrame indexed by ticker (index name "ticker"), columns:
        company_name, optimized_weight_pct, index_weight_pct, delta_pct,
        abs_delta_pct, classification
    Sorted by delta_pct descending (most overweight -> most underweight);
    rows with no reference weight (NaN delta_pct) sort last.
    """
    ref = reference_weights.set_index("ticker")

    table = pd.DataFrame({"optimized_weight_pct": weights * 100.0})
    table.index.name = "ticker"
    table["company_name"] = ref["company_name"].reindex(table.index)
    table["index_weight_pct"] = ref["weight_pct"].reindex(table.index)
    table["delta_pct"] = table["optimized_weight_pct"] - table["index_weight_pct"]
    table["abs_delta_pct"] = table["delta_pct"].abs()

    def classify(delta: float) -> str:
        if pd.isna(delta):
            return "No index reference available"
        if delta > DELTA_THRESHOLD_PCT:
            return "Overweight"
        if delta < -DELTA_THRESHOLD_PCT:
            return "Underweight"
        return "Neutral"

    table["classification"] = table["delta_pct"].apply(classify)
    table = table.sort_values("delta_pct", ascending=False, na_position="last")
    return table[[
        "company_name", "optimized_weight_pct", "index_weight_pct",
        "delta_pct", "abs_delta_pct", "classification",
    ]]


# ---------------------------------------------------------------------------
# Part B — rule-based commentary ("AI advice")
# ---------------------------------------------------------------------------

_NO_REFERENCE_TEXT = (
    "No index reference weight is available for this stock (outside the 25 "
    "officially published Euronext weights), so an overweight/underweight "
    "comparison cannot be computed."
)
_NEUTRAL_TEXT = "Weight closely matches its index allocation; no significant tilt."
_NO_DRIVER_TEXT = (
    "{direction} relative to the index; no single dominant statistical driver "
    "(return, volatility, correlation, or skew) explains this -- likely a "
    "joint covariance effect across the portfolio."
)


def generate_commentary(ticker: str, delta: float, stock_stats: dict) -> str:
    """
    One-sentence, rule-based explanation for why `ticker` is over- or
    underweighted relative to the index ("AI advice"). This is a fixed set
    of if/then rules over statistics we already compute -- NOT a machine
    learning model -- so every threshold is a named constant at the top of
    this module (SHARPE_HIGH_PERCENTILE, SHARPE_MEDIAN_PERCENTILE,
    VOLATILITY_HIGH_PERCENTILE, CORRELATION_LOW_PERCENTILE,
    SKEWNESS_NEGATIVE_THRESHOLD) with its own justification.

    Rules, evaluated ONLY in the direction matching the stock's actual
    tilt -- an overweight stock is NEVER checked against the underweight
    rules and vice versa; the two directions never mix within one sentence:

        Overweight (delta > +DELTA_THRESHOLD_PCT):
          1. sharpe_percentile >= SHARPE_HIGH_PERCENTILE
             (top quartile of the current universe's Sharpe ratios)
             -> "strong historical risk-adjusted return"
          2. correlation_percentile <= CORRELATION_LOW_PERCENTILE
             (bottom quartile of correlation to the REST of the portfolio)
             -> "diversification benefit (low correlation to the rest of
                the portfolio)"

        Underweight (delta < -DELTA_THRESHOLD_PCT):
          3. volatility_percentile >= VOLATILITY_HIGH_PERCENTILE
             (top quartile of volatility)
             AND sharpe_percentile <= SHARPE_MEDIAN_PERCENTILE
             (below-median Sharpe) -- BOTH conditions are required, not
             high volatility alone, because "high volatility relative to
             its return" is a statement about EFFICIENCY, not risk in
             isolation: a high-vol, high-return stock must not be
             penalised by this rule.
             -> "high historical volatility relative to its return"
          4. skewness <= SKEWNESS_NEGATIVE_THRESHOLD
             (Bulmer's threshold for "moderately to highly negatively skewed")
             -> "a history of sharp downside moves"

    Priority and combination -- deliberate, not incidental:
        Rules are listed above in PRIORITY ORDER within each direction:
        rule 1 outranks rule 2 for overweight stocks; rule 3 outranks
        rule 4 for underweight stocks. At most the first TWO applicable
        rules, in that order, are combined into one sentence. This order
        reflects which explanation is treated as the stronger/more
        fundamental driver for that direction: a Sharpe-driven overweight
        is considered a more direct explanation than a secondary
        diversification note, and a volatility/efficiency-driven
        underweight is considered a more fundamental risk-control reason
        than a raw skewness observation. Rules NEVER combine across
        directions -- an overweight stock is only ever evaluated against
        rules 1-2, never 3-4, and symmetrically for underweight stocks.

    Special cases (checked before any rule above, no rule evaluation happens):
        - delta is NaN (ticker outside the 25 officially published Euronext
          weights, per compute_overweight_table): returns a dedicated
          "no reference" sentence.
        - |delta| <= DELTA_THRESHOLD_PCT (Neutral): returns a dedicated
          neutral sentence.
        - Over/underweighted but NO rule applies (can genuinely happen --
          Markowitz optimises jointly over the whole covariance structure,
          not any single statistic): returns an honest fallback sentence
          rather than fabricating a reason that isn't actually supported
          by the data.

    Parameters
    ----------
    ticker      : ticker symbol. Not currently interpolated into the
                  sentence (the dashboard table already has a separate
                  ticker column) but kept in the signature as the agreed
                  public interface.
    delta       : delta_pct from compute_overweight_table (NaN allowed).
    stock_stats : one row of compute_stock_stats(), as a dict or pd.Series,
                  expected to provide sharpe_percentile,
                  volatility_percentile, correlation_percentile, skewness.
                  Any of these may be missing or NaN (e.g. too few return
                  observations for a very short date range, or a
                  single-stock "rest of portfolio") -- the corresponding
                  rule is simply skipped, never raises.

    Returns
    -------
    str — one sentence, or two clauses joined into one sentence.
    """
    if pd.isna(delta):
        return _NO_REFERENCE_TEXT
    if abs(delta) <= DELTA_THRESHOLD_PCT:
        return _NEUTRAL_TEXT

    sharpe_pct = stock_stats.get("sharpe_percentile")
    vol_pct = stock_stats.get("volatility_percentile")
    corr_pct = stock_stats.get("correlation_percentile")
    skew = stock_stats.get("skewness")

    clauses: list[str] = []
    if delta > DELTA_THRESHOLD_PCT:
        direction = "Overweighted"
        if sharpe_pct is not None and pd.notna(sharpe_pct) and sharpe_pct >= SHARPE_HIGH_PERCENTILE:
            clauses.append("strong historical risk-adjusted return")
        if corr_pct is not None and pd.notna(corr_pct) and corr_pct <= CORRELATION_LOW_PERCENTILE:
            clauses.append("diversification benefit (low correlation to the rest of the portfolio)")
    else:
        direction = "Underweighted"
        if (
            vol_pct is not None and pd.notna(vol_pct) and vol_pct >= VOLATILITY_HIGH_PERCENTILE
            and sharpe_pct is not None and pd.notna(sharpe_pct) and sharpe_pct <= SHARPE_MEDIAN_PERCENTILE
        ):
            clauses.append("high historical volatility relative to its return")
        if skew is not None and pd.notna(skew) and skew <= SKEWNESS_NEGATIVE_THRESHOLD:
            clauses.append("a history of sharp downside moves")

    clauses = clauses[:2]
    if not clauses:
        return _NO_DRIVER_TEXT.format(direction=direction)
    if len(clauses) == 1:
        return f"{direction} due to {clauses[0]}."
    return f"{direction} due to {clauses[0]}; also shows {clauses[1]}."


# ---------------------------------------------------------------------------
# Part C support — one-call report for the dashboard
# ---------------------------------------------------------------------------

def build_positioning_report(
    returns: pd.DataFrame, weights: pd.Series, reference_weights: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine Parts A and B into the table the dashboard renders directly.

    Returns
    -------
    pd.DataFrame indexed by ticker (index name "ticker"), columns:
        company_name, optimized_weight_pct, index_weight_pct, delta_pct,
        abs_delta_pct, classification, commentary
    Sorted by abs_delta_pct descending (most significant tilt first, in
    either direction) -- this is the dashboard's display order, and is
    deliberately different from compute_overweight_table()'s own default
    sort (delta_pct descending, most overweight -> most underweight),
    which is why the re-sort happens here rather than in that function.
    """
    stock_stats = compute_stock_stats(returns, weights)
    table = compute_overweight_table(weights, reference_weights)

    commentary = [
        generate_commentary(
            ticker,
            row["delta_pct"],
            stock_stats.loc[ticker].to_dict() if ticker in stock_stats.index else {},
        )
        for ticker, row in table.iterrows()
    ]
    table["commentary"] = commentary

    return table.sort_values("abs_delta_pct", ascending=False, na_position="last")
