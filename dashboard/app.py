"""Streamlit dashboard for Portfolio Optimizer.

Minimal interactive app showing:
- cumulative return comparison (portfolio vs CAC40)
- efficient frontier with MVP and tangency portfolio
- capital allocation line
- tangency portfolio weights

The app provides a demo fallback if live data or optimisation inputs are not available.
"""
from typing import Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from optimisation.markowitz import (
    efficient_frontier,
    minimum_variance_portfolio,
    max_sharpe_portfolio,
)
from optimisation.cal import capital_allocation_line

from data_core import (
    fetch_prices,
    fetch_benchmark,
    get_cac40_tickers,
    get_cac40_names,
    load_reference_weights,
    compute_log_returns,
    TRADING_DAYS_PER_YEAR,
    RISK_FREE_RATE,
)

try:
    # When imported as a package
    from ._utils import compute_cumulative_returns, weights_to_series
except Exception:
    # When run as a script (streamlit run dashboard/app.py)
    from dashboard._utils import compute_cumulative_returns, weights_to_series


def safe_run_demo() -> dict:
    """Generate demo optimisation outputs when real data is unavailable."""
    rng = np.random.default_rng(42)
    n = 10
    tickers = [f"T{i}" for i in range(n)]
    mu = pd.Series(rng.uniform(0.03, 0.25, size=n), index=tickers)
    A = rng.normal(size=(n, n))
    cov = pd.DataFrame(np.dot(A, A.T) * 0.05, index=tickers, columns=tickers)

    rets, vols, weights_grid = efficient_frontier(mu, cov, n_points=50)
    mvp_w, mvp_r, mvp_v = minimum_variance_portfolio(mu, cov)
    tang_w, tang_r, tang_v, tang_sh = max_sharpe_portfolio(mu, cov, RISK_FREE_RATE)

    # create fake prices to demonstrate cumulative plot
    dates = pd.date_range(end=pd.Timestamp.today(), periods=252)
    prices = pd.DataFrame(rng.lognormal(mean=0.0002, sigma=0.02, size=(len(dates), n)), index=dates, columns=tickers)

    return dict(
        tickers=tickers,
        prices=prices.cumprod(),
        benchmark=pd.Series(np.cumprod(1 + rng.normal(0.0002, 0.01, size=len(dates))), index=dates, name="CAC40"),
        frontier_returns=rets,
        frontier_vols=vols,
        mvp=(mvp_w, mvp_r, mvp_v),
        tangency=(tang_w, tang_r, tang_v, tang_sh),
    )


@st.cache_data(ttl=60 * 60)
def cached_fetch_prices(tickers, start, end):
    return fetch_prices(tickers=tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))


@st.cache_data(ttl=60 * 60)
def cached_fetch_benchmark(start, end):
    return fetch_benchmark(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))



def plot_cumulative(portfolio_cum: pd.Series, benchmark_cum: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_cum.index, y=portfolio_cum.values, name="Optimised portfolio"))
    fig.add_trace(go.Scatter(x=benchmark_cum.index, y=benchmark_cum.values, name="CAC40"))
    fig.update_layout(title="Cumulative return comparison", xaxis_title="Date", yaxis_title="Growth (1 = start)")
    return fig


def plot_frontier(returns: np.ndarray, vols: np.ndarray, mvp_point, tangency_point, risk_free: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols, y=returns, mode="lines", name="Efficient frontier", line=dict(color="blue")))
    mvp_w, mvp_r, mvp_v = mvp_point
    tang_w, tang_r, tang_v, tang_sh = tangency_point
    fig.add_trace(go.Scatter(x=[mvp_v], y=[mvp_r], mode="markers", name="Min-variance portfolio", marker=dict(size=12, color="green")))
    fig.add_trace(go.Scatter(x=[tang_v], y=[tang_r], mode="markers", name="Max-Sharpe (tangency)", marker=dict(size=12, color="red")))

    # CAL: y = r_f + slope * vol
    slope = (tang_r - risk_free) / tang_v if tang_v > 0 else 0.0
    vols_line = np.linspace(0.0, vols.max() * 1.1, 50)
    cal_returns = risk_free + slope * vols_line
    fig.add_trace(go.Scatter(x=vols_line, y=cal_returns, mode="lines", name="Capital Allocation Line", line=dict(dash="dash", color="black")))

    fig.update_layout(title="Efficient Frontier & Capital Allocation Line", xaxis_title="Volatility (annualised)", yaxis_title="Expected return (annualised)")
    return fig


def plot_weights(weights: pd.Series) -> go.Figure:
    # weights: pd.Series indexed by ticker
    df = weights.reset_index()
    df.columns = ["ticker", "weight"]
    fig = px.bar(df, x="ticker", y="weight", title="Tangency portfolio weights", text="weight")
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%")
    return fig


def plot_concentration(ref_weights: pd.DataFrame) -> go.Figure:
    """Plot the reference 25-stock index concentration as a bar chart.

    Expects `ref_weights` with columns: ticker, company_name, weight_pct (percentage, 0-100).
    """
    df = ref_weights.copy()
    if "weight_pct" in df.columns:
        df = df.sort_values("weight_pct", ascending=False)
        fig = px.bar(df, x="company_name", y="weight_pct", title="Index concentration (25 official Euronext weights)", labels={"company_name": "Company", "weight_pct": "Weight (%)"})
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    else:
        # fallback: try columns 'weight' or similar
        numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
        if numeric_cols:
            wc = numeric_cols[-1]
            df = df.sort_values(wc, ascending=False)
            fig = px.bar(df, x=df.columns[0], y=wc, title="Index concentration (reference)")
            return fig
        return go.Figure()


def main() -> None:
    st.set_page_config(page_title="Portfolio Optimizer Dashboard", layout="wide")
    st.title("Portfolio Optimizer — Dashboard")

    sidebar = st.sidebar
    sidebar.title("Settings")
    demo_mode = sidebar.checkbox("Demo mode", value=False)
    # Date range selector
    today = pd.Timestamp.today().normalize()
    default_start = today - pd.Timedelta(days=365)
    start_date = sidebar.date_input("Start date", value=default_start)
    end_date = sidebar.date_input("End date", value=today)
    if isinstance(start_date, pd.Timestamp):
        start = start_date
    else:
        start = pd.Timestamp(start_date)
    if isinstance(end_date, pd.Timestamp):
        end = end_date
    else:
        end = pd.Timestamp(end_date)

    n_points = sidebar.slider("Efficient frontier points", min_value=20, max_value=200, value=50)
    allow_leverage = sidebar.checkbox("Allow leverage on CAL (use with caution)", value=False)
    cash_slider = sidebar.slider("Cash allocation on CAL (0 = all cash, 1 = all risky)", min_value=0.0, max_value=1.0, value=1.0)

    if demo_mode:
        data = safe_run_demo()
    else:
        with st.spinner("Loading data from data_core and optimisation..."):
            try:
                tickers = get_cac40_tickers()
                names = get_cac40_names()
                prices = cached_fetch_prices(tickers, start, end)
                benchmark = cached_fetch_benchmark(start, end)

                # compute returns and statistics for optimisation inputs
                returns = compute_log_returns(prices)
                mu = returns.mean() * TRADING_DAYS_PER_YEAR
                cov = returns.cov() * TRADING_DAYS_PER_YEAR

                frontier_returns, frontier_vols, weights_grid = efficient_frontier(mu, cov, n_points=n_points)
                mvp = minimum_variance_portfolio(mu, cov)
                tangency = max_sharpe_portfolio(mu, cov, RISK_FREE_RATE)

                data = dict(tickers=tickers, prices=prices, benchmark=benchmark, frontier_returns=frontier_returns, frontier_vols=frontier_vols, mvp=mvp, tangency=tangency)
            except Exception as e:
                st.error(f"Live data failed: {e}. Switching to demo mode.")
                data = safe_run_demo()

    # Load reference weights for concentration chart if available
    try:
        ref_weights = load_reference_weights()
        data["ref_weights"] = ref_weights
    except Exception:
        data["ref_weights"] = None

    # build cumulative return comparison
    tang_w = data["tangency"][0]
    # ensure weights are a pandas Series
    if not isinstance(tang_w, pd.Series):
        tang_w = pd.Series(tang_w, index=data["tickers"])

    # derive simple returns from prices if possible
    if isinstance(data["prices"], pd.DataFrame) and not data["prices"].empty:
        simple_returns = data["prices"].pct_change().dropna()
        common = set(tang_w.index).intersection(simple_returns.columns)
        if len(common) == 0:
            st.warning("No common tickers between tangency weights and price columns — portfolio series cannot be constructed.")
            port_cum = pd.Series(dtype=float)
        else:
            aligned_w = tang_w.reindex(simple_returns.columns).fillna(0)
            port_returns = simple_returns.dot(aligned_w)
            port_cum = (1 + port_returns).cumprod()
    else:
        port_cum = pd.Series(dtype=float)

    benchmark_cum = data.get("benchmark")
    if isinstance(benchmark_cum, pd.Series):
        benchmark_cum = benchmark_cum / benchmark_cum.iloc[0]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(plot_cumulative(port_cum, benchmark_cum), use_container_width=True)

        # Export CSV for the displayed series
        if not port_cum.empty and isinstance(benchmark_cum, pd.Series):
            df_export = pd.DataFrame({"portfolio": port_cum, "benchmark": benchmark_cum}).dropna()
            csv_bytes = df_export.to_csv(index=True).encode("utf-8")
            st.download_button("Download displayed series (CSV)", csv_bytes, file_name="series_export.csv", mime="text/csv")

    with col2:
        fig_front = plot_frontier(data["frontier_returns"], data["frontier_vols"], data["mvp"], data["tangency"], RISK_FREE_RATE)

        # highlight CAL allocation point corresponding to cash_slider (risky weight)
        try:
            tang_v = float(data["tangency"][2])
            tang_r = float(data["tangency"][1])
            slope = (tang_r - RISK_FREE_RATE) / tang_v if tang_v > 0 else 0.0
            risky_w = float(cash_slider)
            vol_point = tang_v * risky_w
            ret_point = RISK_FREE_RATE + slope * vol_point
            fig_front.add_trace(go.Scatter(x=[vol_point], y=[ret_point], mode="markers+text", name="CAL allocation", marker=dict(size=10, color="black"), text=[f"w_risky={risky_w:.2f}"], textposition="top center"))
        except Exception:
            pass

        st.plotly_chart(fig_front, use_container_width=True)

    st.header("Tangency portfolio weights")
    weights_series = weights_to_series(tang_w, data["tickers"]) if not isinstance(tang_w, pd.Series) else tang_w.sort_values(ascending=False)
    st.plotly_chart(plot_weights(weights_series), use_container_width=True)

    if data.get("ref_weights") is not None:
        st.header("Index concentration (25 weights)")
        st.plotly_chart(plot_concentration(data["ref_weights"]), use_container_width=True)


if __name__ == "__main__":
    main()
