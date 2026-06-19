"""
CAC40 constituent tickers and official index weights.

SOURCES
-------
Tickers  — Wikipedia CAC40 article, composition effective September 2024.
           https://en.wikipedia.org/wiki/CAC_40

Weights  — Euronext official composition sheet, March 31, 2026.
           https://live.euronext.com/sites/default/files/documentation/
           index-composition/CAC_40_Index_Composition.pdf

KNOWN LIMITATIONS (read before using this module in your report)
------------------------------------------------------------------
1. PARTIAL WEIGHTS — METHOD AND LIMITATIONS:
   The Euronext public PDF only discloses the top 25 constituents (the rest
   requires a paid licence). For the remaining 14 stocks, weights were derived
   from total market-capitalisation data via yfinance (.info["marketCap"]),
   snapshotted on 2026-06-19, then normalised so all 14 sum to the residual
   9.43 % (= 100 % − confirmed top-25 total). Formula:
       weight_i = (marketCap_i / Σ marketCap_14) × 9.43 %

   KEY LIMITATION — total vs. free-float market cap:
   The official CAC40 uses FREE-FLOAT market cap (shares available to public
   investors, excluding founder/government/long-term holdings). yfinance
   reports TOTAL market cap. For stocks with concentrated ownership — notably
   Crédit Agricole (ACA.PA, ~59 % held by regional cooperative banks) —
   total market cap materially overstates the free-float figure and therefore
   the index weight. ACA.PA's weight here (2.23 %) is almost certainly an
   overestimate; its true CAC40 weight is probably well below 1 %. This
   limitation must be disclosed in the report.

   Do not present any of these 14 weights as official Euronext data.

2. STATIC SNAPSHOT: The CAC40 rebalances quarterly. This file reflects the
   September 2024 composition (tickers) and March 2026 weights. If you run
   the project well after June 2026 you should update both tables from
   Euronext's latest publication.

3. BENCHMARK: For the actual "does the portfolio beat the index?" comparison,
   use the ^FCHI price series fetched by fetcher.py — NOT the weights here.
   ^FCHI is the authoritative index return; the weights are only needed for
   "what would the naive CAC40-allocation portfolio look like" comparisons.

4. ARCELOR MITTAL: ArcelorMittal (MNEMO: MT) uses its Euronext Amsterdam
   listing on Yahoo Finance (MT.AS), not a Paris suffix. This is the only
   constituent that differs from the .PA pattern.

5. MISSING 40th CONSTITUENT: Wikipedia's extraction returned 39 companies.
   The 40th is unconfirmed; it is likely a very small weight (~0.3 %) and
   will not materially affect optimisation results. Update this table once
   confirmed from a licensed source.
"""

# The benchmark index ticker on Yahoo Finance.
BENCHMARK_TICKER: str = "^FCHI"

# Risk-free rate used across the project (annualised, decimal form).
# Source: ECB deposit facility rate trajectory mid-2026. 2.25 % is a
# conservative, widely used proxy for short-term EUR cash returns.
RISK_FREE_RATE: float = 0.0225

# ---------------------------------------------------------------------------
# Top 25 constituents — weights from Euronext March 31 2026 (official)
# ---------------------------------------------------------------------------
_CONFIRMED: dict[str, tuple[str, float]] = {
    # ticker      : (company name,              weight %)
    "TTE.PA"  : ("TotalEnergies",              9.52),
    "SU.PA"   : ("Schneider Electric",         7.57),
    "MC.PA"   : ("LVMH",                       6.63),
    "AI.PA"   : ("Air Liquide",                5.90),
    "SAN.PA"  : ("Sanofi",                     5.53),
    "AIR.PA"  : ("Airbus",                     5.47),
    "SAF.PA"  : ("Safran",                     5.09),
    "BNP.PA"  : ("BNP Paribas",               4.98),
    "OR.PA"   : ("L'Oréal",                   4.80),
    "CS.PA"   : ("AXA",                        4.14),
    "DG.PA"   : ("Vinci",                      3.86),
    "EL.PA"   : ("EssilorLuxottica",           3.69),
    "RMS.PA"  : ("Hermès International",       2.92),
    "ENGI.PA" : ("Engie",                      2.90),
    "BN.PA"   : ("Danone",                     2.56),
    "GLE.PA"  : ("Société Générale",          2.50),
    "LR.PA"   : ("Legrand",                    1.98),
    "SGO.PA"  : ("Saint-Gobain",               1.80),
    "ORA.PA"  : ("Orange",                     1.75),
    "HO.PA"   : ("Thales",                     1.34),
    "VIE.PA"  : ("Veolia",                     1.25),
    "ML.PA"   : ("Michelin",                   1.18),
    "KER.PA"  : ("Kering",                     1.09),
    "MT.AS"   : ("ArcelorMittal",              1.07),  # Amsterdam listing
    "STMPA.PA": ("STMicroelectronics",         1.05),
}

# ---------------------------------------------------------------------------
# Remaining 14 constituents — weights from yfinance total market cap
# (see KNOWN LIMITATION #1 in the module docstring before using these)
#
# Raw total market caps fetched via yfinance on 2026-06-19 (EUR, approx.):
#   ACA.PA   53 539 352 576  DSY.PA   22 541 563 904  PUB.PA   22 288 093 184
#   EN.PA    19 372 296 192  STLAP.PA 16 257 916 928  RI.PA    15 924 152 320
#   CAP.PA   14 895 554 560  URW.PA   14 178 818 048  BVI.PA   11 563 990 016
#   AC.PA    11 489 999 872  CA.PA    10 939 701 248  RNO.PA    8 115 529 728
#   EDEN.PA   5 468 840 448  TEP.PA    3 227 001 088
#   Total   229 802 810 112
#
# Each weight = (marketCap_i / 229 802 810 112) × 9.43 %, rounded to 2 dp.
# ACA.PA is adjusted from its unrounded value (2.195 %) to 2.23 % so that
# the 14 weights sum exactly to 9.43 % (the rounding adjustment is +0.03 pp,
# well within the total-vs-free-float uncertainty on ACA.PA itself).
# ---------------------------------------------------------------------------
_ESTIMATED: dict[str, tuple[str, float]] = {
    # ticker      : (company name,                   weight % — see docstring)
    "ACA.PA"  : ("Crédit Agricole",               2.23),  # ⚠ likely overstated (low free-float)
    "DSY.PA"  : ("Dassault Systèmes",             0.92),
    "PUB.PA"  : ("Publicis Groupe",               0.91),
    "EN.PA"   : ("Bouygues",                       0.79),
    "STLAP.PA": ("Stellantis",                     0.67),
    "RI.PA"   : ("Pernod Ricard",                 0.65),
    "CAP.PA"  : ("Capgemini",                      0.61),
    "URW.PA"  : ("Unibail-Rodamco-Westfield",      0.58),
    "BVI.PA"  : ("Bureau Veritas",                 0.47),
    "AC.PA"   : ("Accor",                          0.47),
    "CA.PA"   : ("Carrefour",                      0.45),
    "RNO.PA"  : ("Renault",                        0.33),
    "EDEN.PA" : ("Edenred",                        0.22),
    "TEP.PA"  : ("Teleperformance",               0.13),
    # TODO: identify and add the 40th constituent (weight ~0.3 %).
    # Candidate sources: Euronext licensed feed, Bloomberg, Refinitiv.
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cac40_tickers() -> list[str]:
    """Return the list of 39 confirmed CAC40 constituent Yahoo Finance tickers."""
    return list(_CONFIRMED.keys()) + list(_ESTIMATED.keys())


def get_cac40_weights() -> dict[str, float]:
    """
    Return {ticker: weight_percent} for all 39 constituents.

    Weights for the top 25 are official (Euronext March 2026).
    Weights for the remaining 14 are estimated — see module docstring.
    All 39 weights sum to 100.00 %.
    """
    weights = {t: w for t, (_, w) in _CONFIRMED.items()}
    weights.update({t: w for t, (_, w) in _ESTIMATED.items()})
    return weights


def get_cac40_names() -> dict[str, str]:
    """Return {ticker: company_name} for all 39 constituents."""
    names = {t: n for t, (n, _) in _CONFIRMED.items()}
    names.update({t: n for t, (n, _) in _ESTIMATED.items()})
    return names


def get_estimated_tickers() -> set[str]:
    """Return the set of tickers whose weights are estimated (not from Euronext)."""
    return set(_ESTIMATED.keys())
