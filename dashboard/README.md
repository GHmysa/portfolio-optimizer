Dashboard for Portfolio Optimizer

Run the Streamlit app locally:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

What it shows

- Cumulative return comparison: optimised portfolio vs CAC40 benchmark
- Efficient frontier with MVP and tangency portfolio and Capital Allocation Line
- Tangency portfolio weights (bar chart)
- Index concentration (25 official Euronext weights) — uses `data_core.load_reference_weights()`

Notes

- The app has a "Demo mode" toggle in the sidebar for offline testing.
- Live mode fetches prices via `data_core.fetch_prices()` and computes optimisation inputs automatically.
