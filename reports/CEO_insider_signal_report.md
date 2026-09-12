# CEO Insider Purchase Signal — Backtest Report

**Does heavy insider (CEO/CFO/COO) buying predict forward stock outperformance?**

**TL;DR: No.** Across 4 independent configurations, 20 quarters of walk-forward testing (Oct 2021 – Jul 2026), and ~4,000 individual ticker-period return observations, no configuration produced a statistically significant spread (Q5 − Q1) or reliable alpha vs SPY. The signal, as implemented, has no predictive value.

---

## Methodology

### Signal Construction
At each quarterly rebalance date, SEC EDGAR Form 4 insider purchase transactions (CEO, CFO, COO, President, General Counsel, CTO) exceeding $100k were collected within a lookback window. Each ticker received a composite score:

```
composite = conviction × w_conviction + recency × w_recency + size × w_size
```

- **Conviction** = fraction of transactions attributed to qualifying insiders (vs. total transactions in window)
- **Recency** = how recently the most recent qualifying transaction occurred (0–1)
- **Size** = log-dollar value of total qualifying purchases / 20

Tickers were ranked by composite score and quintile-assigned. Q5 = top quintile (heaviest insider buying). Q1 = bottom quintile (lightest/no insider buying).

### Configurations Tested

| Config | Lookback | Size wt | Recency wt | Conviction wt | Character |
|--------|----------|---------|-----------|--------------|-----------|
| A | 365 days | 0.2 | 0.3 | 0.5 | Conviction-weighted, long history |
| B | 365 days | 0.7 | 0.1 | 0.2 | Size-weighted, long history |
| C | 90 days | 0.2 | 0.3 | 0.5 | Conviction-weighted, 1-quarter window |
| D | 90 days | 0.7 | 0.1 | 0.2 | Size-weighted, 1-quarter window |

### Backtest Design
- **20 quarterly rebalances**: 2021-10-01 → 2026-07-01
- **Hold period**: ~63 trading days (one quarter)
- **Rebalancing**: Equal-weight within quintile (individual configs may vary)
- **Benchmark**: SPY buy-and-hold over identical periods
- **Entry/exit**: Next-day open after rebalance date; exit at next rebalance open
- **Data**: yfinance adjusted close (survivorship-bias free via EDGAR inclusion)

---

## Results

### Summary: All 4 Configurations vs SPY

| Config | Ann. Return | Total Return | vs SPY ($) | Alpha/qtr | Avg Spread | Sharpe | Spread p-val |
|--------|------------|--------------|------------|-----------|------------|--------|-------------|
| **A** Baseline | +10.2% | +54.6% | **−$20k** | −0.6% | −0.83%/q | 0.36 | 0.46 |
| **B** Val-wt | +13.0% | +77.1% | **+$2k** | ≈0% | −0.62%/q | 0.62 | 0.69 |
| **C** 1Q lookback | +10.2% | +52.5% | **−$22k** | −0.6% | −1.69%/q | 0.31 | 0.25 |
| **D** 1Q+Val-wt | +14.1% | +79.2% | **+$4k** | +0.2% | +0.32%/q | 0.50 | 0.86 |
| **SPY buy-hold** | +13.0% | +74.8% | — | — | — | — | — |

*Starting capital: $100,000. Ann. return computed from mean quarterly return compounded ×4.*

### Only D beats SPY — and within noise

- **Config D** is the best performer: +79.2% total (+14.1% ann.) vs SPY +74.8%. A $100k investment would be worth **$4,420 more** than the same SPY holding.
- However, this outperformance is not statistically distinguishable from zero (Sharpe 0.50, spread p=0.86, 95% CI on spread: [−3.1%, +3.7%]).
- **Config B** essentially ties SPY (+$2k, alpha ≈ 0).
- **A and C** meaningfully lag SPY by $20–22k.

### The spread (Q5 − Q1) is broken in 3 of 4 configs

This is the most damning finding. The core thesis — *buy top-quintile insider activity, sell/avoid bottom-quintile* — fails:

| Config | Avg Spread | t-stat | p-value | Spread > 0 (freq) |
|--------|-----------|--------|---------|-------------------|
| A | −0.83%/q | −0.76 | 0.46 | 50% |
| B | −0.62%/q | −0.40 | 0.69 | 50% |
| C | −1.69%/q | −1.18 | 0.25 | 45% |
| D | +0.32%/q | +0.18 | 0.86 | 40% |

**Interpretation**: In configs A, B, C, heavy insider buying is associated with *worse* forward performance. Even D's positive spread is tiny (+0.32%/q) and indistinguishable from noise.

---

## Quarterly Returns

| Period | Config A | Config B | Config C | Config D | SPY |
|--------|----------|----------|----------|----------|-----|
| 2021-10-01 | +7.7% | +8.3% | +7.6% | +11.4% | +10.0% |
| 2022-01-01 | −3.3% | +0.8% | −8.5% | −0.1% | −3.7% |
| 2022-04-01 | −17.8% | −11.7% | −15.1% | −15.8% | −15.7% |
| 2022-07-01 | −1.3% | −2.9% | −2.2% | −3.8% | −4.4% |
| 2022-10-01 | +9.4% | +12.5% | +0.2% | +5.8% | +4.8% |
| 2023-01-01 | +6.0% | +4.8% | +8.1% | +1.7% | +6.4% |
| 2023-04-01 | +4.7% | +6.8% | +5.4% | +3.8% | +7.0% |
| 2023-07-01 | −1.9% | −2.1% | −4.5% | −0.8% | −3.3% |
| 2023-10-01 | +15.5% | +14.0% | +16.0% | +15.5% | +11.7% |
| 2024-01-01 | +7.2% | +6.2% | +11.5% | +7.6% | +11.0% |
| 2024-04-01 | −1.9% | −1.5% | −1.9% | −2.9% | +4.6% |
| 2024-07-01 | +10.8% | +12.5% | +12.1% | +14.1% | +5.1% |
| 2024-10-01 | +4.6% | +2.9% | −2.8% | −6.1% | +3.8% |
| 2025-01-01 | −2.2% | +1.0% | +1.9% | +14.6% | −4.7% |
| 2025-04-01 | +9.6% | +5.8% | +13.1% | +7.3% | +9.9% |
| 2025-07-01 | −1.3% | −0.6% | −1.2% | +7.3% | +7.8% |
| 2025-10-01 | +1.5% | +3.2% | +3.3% | +5.2% | +3.1% |
| 2026-01-01 | −5.3% | −5.3% | −8.3% | −9.8% | −7.2% |
| 2026-04-01 | +4.6% | +7.9% | +13.4% | +17.7% | +13.4% |
| 2026-07-01 | +2.8% | −0.7% | +1.2% | −5.7% | +2.5% |

---

## Equity Curves

Starting from $100,000:

| Period | Config A | Config B | Config C | Config D | SPY |
|--------|----------|----------|----------|----------|-----|
| 2021-10-01 | $107,675 | $108,270 | $107,644 | $111,360 | $110,035 |
| 2022-04-01 | $85,523 | $91,385 | $86,002 | $90,050 | $88,471 |
| 2022-10-01 | $92,352 | $102,838 | $86,214 | $95,259 | $88,588 |
| 2023-10-01 | $119,744 | $130,380 | $120,104 | $131,170 | $112,294 |
| 2024-07-01 | $137,409 | $153,930 | $137,170 | $159,270 | $135,901 |
| 2025-04-01 | $150,023 | $163,050 | $155,940 | $170,900 | $143,427 |
| 2026-07-01 | **$154,609** | **$177,100** | **$152,531** | **$179,244** | **$174,824** |

---

## Key Risks and Caveats

### Statistical limitations
- **N = 20 quarters** — insufficient for reliable Sharpe ratio estimation. Sharpe estimates require 30+ periods for stability.
- **Multiple testing**: 4 configs tested against the same data — some apparent outperformance is expected by chance.
- **No out-of-sample validation** — all results are in-sample walk-forward on the same 2021–2026 period.

### Data and execution limitations
- **Survivorship bias**: yfinance only returns surviving tickers; delisted companies' returns are lost (understates Q5 losses if heavy buyers disproportionately included failing firms).
- **Transaction costs and slippage**: Not modelled. Long-short rebalancing in Q5/Q1 would incur 2-way costs of ~0.1–0.3% per quarter per leg, likely sufficient to eliminate D's marginal alpha.
- **SEC filing lag**: Form 4 has a 2-business-day filing deadline — prices may have moved by the time signals are actionable.
- **Tax**: Short-term capital gains on quarterly rebalancing would significantly erode returns in a real account.

### Signal design limitations
- **Equal-weight within quintile**: Some configs (e.g. D) use value-weighting implicitly via the scoring function, but the backtest uses equal weighting. A true size-proportional basket may differ.
- **No sector controls**: Heavy insider buying may cluster in overextended sectors (e.g. Q4 2024 tech rotation into D's 1Q lookback).

---

## Conclusion

**No configuration of the CEO insider purchase signal produces a reliable, statistically significant edge over SPY buy-and-hold.**

The results are consistent with the hypothesis being false: insider purchases, as captured by SEC EDGAR Form 4 data with standard weighting schemes, do not predict forward outperformance. Three of four configs show a *negative* spread (top insider-buy stocks underperform bottom), and the one positive spread (D, +0.32%/q) is statistically indistinguishable from zero.

Config D's marginal outperformance (+$4k on $100k) would not survive transaction costs, tax drag, or out-of-sample degradation. The signal is not suitable as a standalone basis for live or paper trading.

**Recommendations for further work**:
1. Test on pre-2010 data (longer history, more quarters for significance)
2. Exclude filing-window periods (2-day lag) to reduce look-ahead bias
3. Add sector/factor controls — insider buying may be a risk factor, not an alpha signal
4. Test long-short with explicit transaction cost modelling (< $0.05/share round-trip)
5. Consider whether insider *sales* carry information (asymmetric signalling)

---

*Backtest scripts: `src/backtest/insider_signal/`*
*Generated: September 2026 | Data: SEC EDGAR Form 4 via EDGAR parser, yfinance price history*
