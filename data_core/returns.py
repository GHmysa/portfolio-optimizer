"""
Statistical computations on price series.

All maths is explained inline because you need to be able to present it
in your report. Each function has the formula in its docstring.

Convention used throughout the project
---------------------------------------
* DAILY log-returns are the base unit.
* ANNUALISED figures multiply/scale by 252 (standard trading-days-per-year
  count for European exchanges).
* The covariance matrix is annualised — it is the quantity directly used by
  the Markowitz optimiser in optimisation/.
"""

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Compute daily log (continuously compounded) returns from a price series.

    Formula
    -------
        r_t = ln(P_t) − ln(P_{t-1})   ≡   ln(P_t / P_{t-1})

    Why log returns instead of simple returns?
      1. Time-additivity: the log return over k days is the sum of k daily
         log returns — makes multi-period aggregation trivial.
      2. Symmetry: a +10 % and -10 % log return cancel out exactly; with
         simple returns they do not (-1 % net).
      3. Better approximation to normality: important for Markowitz theory,
         which assumes normally distributed returns.
      4. Standard in quantitative finance literature.

    The first row is always NaN (no previous price) and is dropped.
    """
    return np.log(prices / prices.shift(1)).dropna()


def annualized_return(daily_returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """
    Annualise the mean daily log-return.

    Formula
    -------
        μ_annual = mean(r_daily) × 252

    Interpretation: if a stock earns 0.04 % per day on average, it earns
    approximately 0.04 % × 252 ≈ 10.1 % per year.

    Note: this is the arithmetic annualisation of log-returns (i.e. the
    "expected log return" per year). For the portfolio optimiser, this is
    the μ vector input to the Markowitz mean-variance problem.
    """
    return daily_returns.mean() * TRADING_DAYS_PER_YEAR


def annualized_volatility(daily_returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """
    Annualise the standard deviation of daily log-returns.

    Formula
    -------
        σ_annual = std(r_daily) × √252

    Why √252?  Variance scales linearly with time (under the i.i.d. returns
    assumption), so Var_annual = Var_daily × 252, and taking the square root
    gives σ_annual = σ_daily × √252.

    This is the standard risk measure used in Markowitz and Sharpe ratio
    calculations throughout this project.
    """
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def covariance_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the annualised covariance matrix of log-returns.

    Formula
    -------
        Σ_annual = Σ_daily × 252

    where Σ_daily[i,j] = Cov(r_i, r_j) = E[(r_i − μ_i)(r_j − μ_j)]

    The covariance matrix is the key input to the Markowitz optimiser. Its
    diagonal entries are the individual variances (σ_i²); off-diagonal
    entries capture how stocks co-move. A large positive Cov(i,j) means
    stocks i and j tend to rise and fall together — holding both gives less
    diversification benefit than holding two negatively correlated stocks.

    The matrix is always symmetric (Cov(i,j) = Cov(j,i)) and positive
    semi-definite, which guarantees the optimisation problem has a solution.
    """
    return daily_returns.cov() * TRADING_DAYS_PER_YEAR


def statistical_moments(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the four statistical moments for each asset's daily return series.

    Returns
    -------
    pd.DataFrame with index = ["mean", "volatility", "skewness", "kurtosis"]
    and columns = ticker symbols. All values are annualised where applicable.

    The four moments
    ----------------
    1. Mean (μ): expected daily return. Positive = stock tends to go up.

    2. Volatility (σ): std of daily returns × √252. The primary risk measure
       in Markowitz theory. A stock with σ = 30 % annually is much riskier
       than one with σ = 15 %.

    3. Skewness (γ₁): measures asymmetry of the return distribution.
       Formula: γ₁ = E[(r − μ)³] / σ³
       - γ₁ = 0  → symmetric (normal)
       - γ₁ < 0  → left tail is fatter (more large negative returns than
                   large positive ones) — common in equity markets
       - γ₁ > 0  → right tail is fatter

    4. Excess kurtosis (γ₂): measures tail heaviness relative to a normal.
       Formula: γ₂ = E[(r − μ)⁴] / σ⁴ − 3  (the −3 centres it at 0 for normal)
       - γ₂ > 0  → "leptokurtic" / fat tails: more extreme returns than a
                   normal distribution predicts — empirically true for stocks
       - γ₂ = 0  → normal distribution

    Report tip: showing that skewness ≠ 0 and kurtosis > 0 is the standard
    empirical argument for why Markowitz (which assumes normality) is a
    simplification, yet still practically useful.
    """
    mu   = annualized_return(daily_returns)
    vol  = annualized_volatility(daily_returns)
    skew = daily_returns.apply(stats.skew)
    kurt = daily_returns.apply(stats.kurtosis)  # excess kurtosis (Fisher)

    return pd.DataFrame(
        [mu, vol, skew, kurt],
        index=["mean_annual", "volatility_annual", "skewness", "excess_kurtosis"],
    )
