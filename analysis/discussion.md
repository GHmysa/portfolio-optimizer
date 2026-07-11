<!-- Lernhinweis: Die Kommentare sind Leitfragen. Jeder Absatz beantwortet

die Frage direkt darüber. Im gerenderten Dokument (GitHub) sind diese

Kommentare unsichtbar - nur der englische Text zählt. -->



\# Module D — Performance Analysis: Discussion



\## 1. Results



<!-- Frage 1: Was wurde verglichen, über welchen Zeitraum, mit welchen

Daten und Kennzahlen? -->



We compare three daily return series over the six-year period from January

2019 to December 2024 (about 1,500 trading days): the maximum-Sharpe

("tangency") portfolio produced by the Markowitz optimiser (Module B), the

same portfolio combined with a cash position via the Capital Allocation

Line, and the CAC40 index (^FCHI) as the benchmark. All price data come

from Yahoo Finance through Module A; performance is measured as annualised

return, annualised volatility and Sharpe ratio, using a risk-free rate of

2.25 % p.a.



| Series                                 | Annualised return | Annualised volatility | Sharpe ratio |

|----------------------------------------|-------------------|-----------------------|--------------|

| Max-Sharpe portfolio (fully invested)  | 23.80 %           | 11.93 %               | 1.81         |

| Max-Sharpe + cash (16.2 % cash)        | 20.31 %           | 10.00 %               | 1.81         |

| CAC40 index (^FCHI)                    | 7.29 %            | 19.56 %               | 0.26         |



<!-- Frage 2: Was kam heraus? Welche Zahlen sind entscheidend, und wie

sieht das optimale Portfolio aus (Konzentration, groesste Position)? -->



In this in-sample comparison the optimised portfolio dominates the index

on every metric: it earned more than three times the index return at

roughly half the risk. The optimiser achieves this with a concentrated

allocation: only about ten of the forty constituents receive a weight

above 1 %, and the largest single position is Euronext (ENX.PA) with

25.8 %, followed by Accor (16.6 %) and Safran (12.2 %).



\## 2. The role of the cash allocation



<!-- Frage 3: Was ist die Capital Allocation Line - und was passiert mit

Rendite, Risiko und Sharpe Ratio, wenn Cash beigemischt wird? Beleg:

die beiden Portfolio-Zeilen der Tabelle. -->



The Capital Allocation Line (CAL) describes all combinations of the

risk-free asset (cash) and the tangency portfolio. Because cash earns a

constant return and has zero volatility, blending in cash scales both the

excess return and the volatility by the same factor — the share invested

in the risky portfolio. As a consequence, the Sharpe ratio is identical at

every point on the CAL. Our results confirm this exactly: the fully

invested portfolio and the 16.2 %-cash portfolio show the same Sharpe

ratio of 1.81, while return and volatility shrink proportionally.



<!-- Frage 4: Wenn Cash die Sharpe Ratio nicht verbessert - wozu ist es

dann gut? Beleg: Risikoziel 10 % -> 16,2 % Cash -> Volatilitaet exakt

10,00 %. -->



Cash therefore does not improve the quality of the portfolio; it lets the

investor choose the level of risk without giving up risk-adjusted

efficiency. In our application we targeted a conservative volatility of

10 % p.a., below the tangency portfolio's own 11.9 %. The CAL prescribes

83.8 % in the risky portfolio and 16.2 % in cash, and the realised

volatility hits the target exactly (10.00 %).



<!-- Frage 5: Warum kam beim Ziel "gleiches Risiko wie der Index" genau

0 % Cash heraus - und was hat das mit dem Kreditverbot (long-only,

kein Leverage) zu tun? -->



The reverse direction, however, is not available in this project. Because

the tangency portfolio is less volatile than the index itself (11.9 % vs.

19.6 %), matching the index's risk level would require investing more than

100 % in the risky portfolio — that is, borrowing at the risk-free rate.

Under the project's long-only, no-leverage constraint the risky weight is

capped at 100 %, so the cash mechanism only works "downwards", towards

lower risk. (When we initially targeted the index volatility, the CAL

function returned exactly this corner solution: 100 % risky, 0 % cash.)



\## 3. Limitations of the analysis



<!-- Frage 6: Warum ist der Vergleich gegen ^FCHI leicht unfair, zu

wessen Gunsten und um wie viel - und kippt das das Gesamtergebnis? -->



Two biases affect the comparison, and both favour the optimised portfolio.



First, a dividend bias. The benchmark ^FCHI on Yahoo Finance is a

price-return index: it ignores dividends. Our individual stock prices, in

contrast, are adjusted closing prices that do include dividends. The

benchmark return is therefore understated by roughly 2 percentage points

per year; a fair total-return benchmark would show approximately 9.3 %

annual return and a Sharpe ratio of about 0.36. This correction narrows

the gap but does not change the overall picture — the measured

outperformance is far larger than the bias.



<!-- Frage 7: Warum darf man der Sharpe Ratio von 1,81 nicht blind

trauen? Was bedeutet "in-sample" (der Optimierer kannte die Antworten

schon), und wie saehe der ehrliche Test aus? -->



Second, and more importantly, an in-sample bias. The optimiser was given

the mean returns and covariances of 2019–2024 and is then evaluated on

exactly the same period. In effect it selected the winners of these six

years with hindsight; concentrated bets such as the 25.8 % position in

Euronext are a symptom of this. The Sharpe ratio of 1.81 is therefore a

backward-looking description, not a forecast. An honest test would split

the sample — for example, optimise on 2019–2022 and evaluate the resulting

weights out-of-sample on 2023–2024 — and we would expect a considerably

more modest result.



\## 4. Conclusion



<!-- Frage 8: Was zeigt das Projekt - und was zeigt es nicht? Stichwoerter:

Cross-Check der Module, keine Prognose kuenftiger Outperformance. -->



The project demonstrates that the Markowitz/CAL machinery works end to

end: the optimiser identifies a portfolio that, in-sample, dominates the

CAC40, and the cash allocation moves the investor along the CAL exactly as

theory predicts, leaving the Sharpe ratio unchanged. The consistency of

the modules is verified by a cross-check: the optimiser's analytical

return and volatility (23.80 %, 11.93 %) match the realised time-series

values computed independently in Module D to four decimal places. What the

analysis does not show is a strategy with guaranteed future

outperformance: after correcting for the missing dividends in the

benchmark and, above all, for the in-sample nature of the optimisation,

the headline numbers should be read as an illustration of the method

rather than as an investment recommendation. A natural next step would be

an out-of-sample backtest with a train/test split.

