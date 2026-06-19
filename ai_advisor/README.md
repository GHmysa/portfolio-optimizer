# ai_advisor — Scoping Questions (Deferred Module)

This module is intentionally empty. Before writing any code, the team needs
answers to the questions below from the professor. Building without them risks
implementing the wrong thing entirely.

---

## What kind of "AI advice" is expected?

There are three fundamentally different interpretations, each implying
different data, models, and complexity:

### Option A — Natural-language commentary (LLM-based)
An LLM (e.g. GPT-4, Claude) reads the optimisation output (weights, Sharpe,
returns) and generates a plain-English explanation of the portfolio and its
risks. No prediction, just narration.
- **Data needed**: the Module B/D output fed as context to the LLM API.
- **Cost**: API call per report generation (~€0.01–0.10 per request).
- **Complexity**: low — mostly prompt engineering.
- **Key question**: is the professor expecting this to be "AI-powered" in
  substance, or just in presentation?

### Option B — Scoring / ranking model (supervised ML)
A trained model scores or ranks the 40 stocks (e.g. by predicted quality or
risk-adjusted attractiveness), and those scores influence the optimisation
weights as a prior or constraint.
- **Data needed**: fundamental data (P/E, earnings, debt) + historical returns
  labels. Not available in free yfinance data.
- **Cost**: significant data sourcing effort (Compustat, Bloomberg, or manual
  scraping).
- **Complexity**: high — model training, validation, leakage prevention.

### Option C — Return prediction model (time-series / ML)
A model predicts future returns for each stock, and those predictions feed
into the Markowitz optimiser as the μ vector instead of historical means.
- **Data needed**: macro signals, technical indicators, earnings calendars.
- **Cost**: high; prediction models in finance are notoriously hard to beat.
- **Complexity**: very high — and raises ethical/academic-integrity questions
  about overfitting to the test period.
- **Risk**: if you present predictions as signals without proper walk-forward
  validation, the professor will likely ask hard questions.

---

## Questions to ask the professor

1. Which of the three options above (A/B/C) are you expecting?
2. If Option A: which LLM API can we use? Is there a budget/credit limit?
3. If Option B or C: what data sources are approved? Is Bloomberg/Refinitiv
   available through the university?
4. Should the AI module influence the optimisation weights, or only comment
   on results produced by Module B?
5. Is there a word/slide limit on the AI component in the report?
6. Are there any academic integrity constraints on using cloud AI APIs?

---

## Implementation plan (fill in after professor meeting)

- [ ] Confirm option (A / B / C)
- [ ] List required data sources and confirm access
- [ ] Define the module's input/output contract with Module B and Module D
- [ ] Agree on evaluation metric (how do we know the AI advice is "good"?)
- [ ] Schedule implementation sprint
