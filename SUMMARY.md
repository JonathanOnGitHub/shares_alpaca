# Shares Alpaca — Experiment Summary

A data-driven share trader using the Alpaca API, built with Python.

## Architecture

```
config/config.yaml               → configuration
src/data/alpaca_client.py        → Alpaca API wrapper (IEX daily bars)
src/data/earnings.py             → yfinance earnings data
src/features/technical.py        → SMA, EMA, RSI, MACD, Bollinger, ATR
src/models/                      → LSTM, XGBoost, Ridge, weighted ensemble
src/backtest/engine.py           → multi-symbol walk-forward with slippage
src/trading/executor.py          → Alpaca paper trading execution
src/trading/pead_overlay.py      → post-earnings drift overlay
src/trading/momentum.py          → cross-sectional momentum strategy
src/trading/momentum_forward.py  → forward momentum trading (cron-ready)
src/pipeline.py                  → end-to-end data → train → backtest
```

## Experiment 1: ML Ensemble (next-day return prediction)

Predicts next-day returns using an ensemble of LSTM, XGBoost, and Ridge models trained on 28 technical features. Walk-forward backtest over 9 windows (~360 days).

| Variant | Return | Sharpe | Win Rate |
|---|---|---|---|
| Baseline (regression, 28 features) | **4.65%** | **0.82** | 68% |
| +SPY market features | 0.50% | 0.22 | 53% |
| +Threshold 0.2% filter | 0.03% | 0.05 | 67% |
| +Classification (up/down) | -0.01% | 0.04 | 65% |
| +PEAD overlay | 0.22% | 0.11 | 67% |
| +Momentum rank features | 0.28% | 0.13 | 90% |

**Result:** The baseline regression ensemble was the best ML configuration but still only returned 4.65%. None of the augmentations (market features, PEAD, momentum ranks, classification, thresholds) improved it.

### Diligence Suite Score: 4/9

When run through the full 9-check diligence suite, the ensemble scores only 4/9:

| Check | Ensemble ML | Momentum LS | Trend (30yr) |
|---|---|---|---|---|
| vs B&H | **FAIL** (5.3% vs 13.6%) | **FAIL** (6.4% vs 26.1%) | **FAIL** (142% vs 676%) |
| Best Month | PASS (82% from 1m) | PASS (57% from 1m) | PASS (39% from 1m) |
| Win Rate | PASS (59%) | **FAIL** (47%) | PASS (55%) |
| Max DD | PASS (-1.1%) | PASS (-41%) | **FAIL** (-57%, 7yr) |
| Monthly Consistency | **FAIL** | PASS (80%) | PASS (59%) |
| Concentration | PASS | PASS | PASS |
| Sub-Period | **FAIL** | **FAIL** | PASS |
| Sharpe Significance | **FAIL** | PASS | PASS |
| Permutation Test | **FAIL** (31st) | **FAIL** (35th) | **FAIL** (19th) |

All three strategies fail the permutation test, and none beats buy-and-hold. **No strategy tested is ready for live capital.**

## Experiment 2: Cross-Sectional Momentum

Ranks stocks by trailing returns and goes long winners / short losers. Rebalanced monthly.

### Walk-Forward Results (19 monthly test periods, 0.1% slippage)

| Universe | Lookback | Skip | TopN | Return | Sharpe | Max DD | Win% |
|---|---|---|---|---|---|---|---|
| Large-cap (50) | 6m | 0m | 5 | **+12%** | 0.39 | -30% | 68% |
| Large-cap (50) | 6m | 1m | 1 | -29% | 0.16 | -71% | 63% |
| Small-cap (85) | **12m** | **1m** | **5** | **+69%** | **0.92** | -34% | 53% |
| Small-cap (85) | 6m | 0m | 5 | +8% | 0.40 | -61% | 53% |
| Small-cap (85) | 6m | 1m | 1 | -93% | -1.13 | -92% | 37% |

### Key Findings

- **Single-split backtests are misleading:** The large-cap 6m momentum returned +165% in a single 80/20 split but **-29% in walk-forward**. Same pattern for small-cap 12m momentum: +69% in single split but **-15% without one lucky month**.
- **The DiligenceSuite tells the real story:** The momentum strategy passes basic metrics (win rate, drawdown) but **fails the critical checks** — it can't survive best-month removal, lags buy-and-hold, and its return is indistinguishable from random noise (permutation test: 66th percentile).
- **Walk-forward was negative across all momentum configs** on both large and small caps. Nothing survived out-of-sample testing.
- **Drawdowns are severe:** Even the best-looking config has -34% max DD in a bull market.
- **Momentum is retained as a validation harness** — `src/validation/run_diligence.py` uses it as the first candidate any new strategy must beat before being considered for live deployment.

### Paper Trading Status

The momentum paper trading (`src/trading/momentum_forward.py`) has been left running for observation but should **not** be sized up with real capital as-is. The diligence suite flags it as failing the permutation test and being dependent on a single month for all its returns.

Rebalances monthly via cron: `.venv/bin/python -m src.trading.momentum_forward --universe small`

## Files Created

## Experiment 3: Swing Trading (RSI + MA + ATR)

Swing trading strategy using RSI oversold/overbought crossovers with trend confirmation (price vs SMA), volume confirmation, and ATR-based risk management (stop loss, take profit, max holding days).

### Strategy Logic

- **Entry (long):** RSI crosses below oversold then back above, price > SMA_trend, volume > SMA_volume × multiplier
- **Entry (short):** RSI crosses above overbought then back below, price < SMA_trend, volume > SMA_volume × multiplier
- **Exit:** ATR-based take profit (3×), ATR-based stop loss (2×), max holding days (10–15), or RSI reversal

### Results

| Universe | Trades | Win Rate | Sharpe | Total Return | Max DD | Score |
|---|---|---|---|---|---|---|
| Mega-cap (5 tickers) | 27 | 59% | 0.35 | 2.6% | -3.5% | **7/9** |
| **Small-cap (15 tickers)** | **41** | **49%** | **1.39** | **44.1%** | **-8.1%** | **6/9** |

### Key Findings

- **Small-cap swing is the best-performing strategy in the repo.** Sharpe 1.39 drastically beats B&H 0.84. Max drawdown (-8.1%) is the lowest of any tested strategy. 84% of rolling 6-month windows are positive. The Sharpe is highly statistically significant (1.18 vs 0.29 threshold).
- **Mega-cap swing** passed 7/9 checks (Sharpe now significant at 0.32 vs 0.29) but returns are modest — mega-caps don't pull back enough for swing entries during bull markets.
- **Low win rate (~49%) is normal** for swing trading. Winners are larger than losers due to the 3× ATR target vs 2× ATR stop structure.
- **Permutation test still fails** (23rd percentile) — the strong small-cap bull market means random portfolios perform well. The strategy's return beats B&H on a risk-adjusted basis but can't match raw 155% buy-and-hold returns.
- **What makes this different:** Unlike the ML ensemble (4/9) and momentum (5–8/9), swing trading is the first strategy with a *statistically significant Sharpe ratio*, passing the Sharpe significance check with room to spare.

| File | Purpose |
|---|---|
| `config/config.yaml` | Base configuration |
| `config/config_smallcap.yaml` | Small-cap ML configuration |
| `src/data/alpaca_client.py` | Alpaca API data client |
| `src/data/earnings.py` | yfinance earnings data fetcher |
| `src/features/technical.py` | Technical indicator feature engineering |
| `src/models/base.py` | Abstract model interface |
| `src/models/lstm_model.py` | LSTM sequence model |
| `src/models/xgb_model.py` | XGBoost model (regression + classification) |
| `src/models/ensemble.py` | Weighted ensemble + Ridge baseline |
| `src/backtest/engine.py` | Multi-symbol backtest with walk-forward, slippage |
| `src/trading/executor.py` | Alpaca order execution |
| `src/trading/pead_overlay.py` | Post-earnings drift overlay strategy |
| `src/trading/momentum.py` | Cross-sectional momentum backtester |
| `src/trading/momentum_forward.py` | Live forward momentum trading (cron-ready) |
| `src/trading/trend.py` | Multi-asset trend-following (CTA-style) |
| `src/trading/swing.py` | Swing trading strategy (RSI + MA + ATR) |
| `src/trading/swing_forward.py` | Forward paper-trading for swing (daily cron) |
| `src/trading/merger_arb.py` | Merger arbitrage strategy (in progress) |
| `src/data/edgar.py` | SEC EDGAR 8-K filing fetcher |
| `src/data/lm_dictionary.csv` | Loughran-McDonald financial sentiment dictionary |
| `src/features/sentiment.py` | L-M sentiment analyzer |
| `src/pipeline.py` | End-to-end pipeline |
| `src/validation/diligence.py` | Reusable diligence/validation suite (9 checks) |
| `SUMMARY.md` | This file |

## DiligenceSuite — Reusable Validation Harness

Any proposed strategy — including the ML ensemble below — must pass the diligence suite before being considered for live deployment. The momentum strategy serves as the baseline: if a new idea can't beat this, it's not worth pursuing.

`src/validation/diligence.py` provides 8 standard diligence checks for any strategy backtest.

### Usage

```python
from src.validation.diligence import DiligenceSuite

suite = DiligenceSuite(
    equity_curve=result.equity_curve,
    trades=result.trades,
    prices=price_data,
    strategy_name="My Strategy",
)
report = suite.run_all()
print(report.summary())
```

### The 8 Checks

| # | Check | What it tests | Pass Condition |
|---|---|---|---|
| 1 | vs Equal-Weight B&H | Beats holding all assets equally | Strategy return > EW return |
| 2 | Best Month Exclusion | Survives removing the single best month | Return stays positive |
| 3 | Win Rate | Wins more than it loses (trade or monthly) | Win rate > 50% |
| 4 | Max Drawdown Duration | Drawdown severity and recovery | DD > -50%, recovery < 2yr |
| 5 | Monthly Consistency | Fraction of rolling 6-month windows positive | > 50% positive + Sharpe > 0 |
| 6 | Trade Concentration | Any single trade dominates PnL | Largest trade < 50% of total |
| 7 | Sub-Period Consistency | First half vs second half of test period | Both halves positive or 2nd not terrible |
| 8 | Sharpe Significance | Sharpe ratio vs noise threshold | Sharpe > 2/√(N) |
| 9 | Permutation Test | Strategy return vs 500 random return shuffles | Beats ≥ 95% of random trials |

### Strategy Scores

All strategies in the registry have been run through the full suite:

| Strategy | Score | Fails |
|---|---|---|
| Momentum small-cap (long-only) | 8/9 | Permutation (66th percentile) |
| Momentum small-cap (long/short) | 5/9 | vs B&H, Win Rate, Sub-Period, Permutation |
| ML Ensemble (mega-cap) | 4/9 | vs B&H, Monthly, Sub-Period, Sharpe, Permutation |
| Trend-following (30yr multi-asset) | 6/9 | vs B&H, Max DD, Permutation (19th) |
| Swing mega-cap | 6/9 | vs B&H, Sharpe Significance, Permutation (7th) |
| **Swing small-cap** | **6/9** | **vs B&H, Win Rate, Permutation (23rd)** |
| Swing mega-cap | 7/9 | vs B&H, Permutation (8th) |

**No strategy beats buy-and-hold** or passes the permutation test. However, swing small-cap is the first strategy to achieve a **statistically significant Sharpe ratio** (1.18 vs 0.29 threshold) with the lowest max drawdown (-8.1%) of any strategy tested. Its 23rd percentile on the permutation test reflects the strong bull market (random portfolios return 215% median) rather than strategy weakness.

### Paper Trading

A forward paper-trading script is wired in (`src/trading/swing_forward.py`) that runs daily. It checks open positions for exits (stop loss, take profit, max holding days, RSI reversal) and scans for new entry signals on the latest bar. Entry dates are persisted to `src/trading/swing_positions.json` so max holding days work correctly across runs.

```bash
# Run daily for small-cap universe (best backtest results)
.venv/bin/python -m src.trading.swing_forward --universe small

# Run daily for mega-cap universe
.venv/bin/python -m src.trading.swing_forward --universe large
```

The swing module provides an instant reality check for any proposed strategy before live deployment.

## Merger Arbitrage — In Progress

Building a merger arbitrage strategy using SEC EDGAR 8-K filings and Loughran-McDonald sentiment analysis.

### Current State (2026-07-20)

- ✅ **EDGAR fetcher** (`src/data/edgar.py`): Downloads and parses 8-K filings. Rate-limited to comply with SEC rules. CIK lookup working.
- ✅ **LM sentiment analyzer** (`src/features/sentiment.py`): Dictionary downloaded (~170 positive, ~560 negative financial words). Analyzes filing tone.
- ✅ **Pipeline works end-to-end**: Fetches 8-Ks for a ticker, detects merger-related content, scores sentiment.
- ❌ **Deal detection too broad**: "MERGER" keyword catches routine filings (debt programs, buybacks). Need precise M&A identification.
- ❌ **No structured deal extraction**: Can't yet extract target company ticker, offer price per share, or expected close date.
- ❌ **No outcome tracking**: Need to determine if deals completed or failed.
- ❌ **No merger arb strategy yet**: The backtest logic hasn't been built.

### Next Steps

1. **Refine merger detection**: Use specific patterns ("Agreement and Plan of Merger", "Definitive Agreement" + "Merger") rather than broad keyword search. Cross-reference with company names to identify actual M&A.
2. **Extract structured deal data**: Parse target ticker, offer price ($X per share), and expected close date from filing text using regex patterns.
3. **Track deal outcomes**: Cross-reference subsequent filings to determine if deals closed or failed.
4. **Build merger arb strategy**: Trade the spread between current price and offer price, with position sizing based on deal confidence (LM sentiment as risk signal).
5. **Run through diligence suite**: Add to `run_diligence.py` registry and test against the 9 checks.

### Data Sources

| Source | Access | Coverage |
|---|---|---|
| SEC EDGAR | Free, unlimited (rate-limited) | All US exchange filings since 1994 |
| L-M Dictionary | Free (publicly available) | Financial sentiment word lists |
| FMP (explored, not used) | Free tier too limited | Only 5 most recent M&A deals |
