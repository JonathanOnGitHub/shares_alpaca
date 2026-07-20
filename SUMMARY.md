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

| Check | Ensemble ML | Momentum LS |
|---|---|---|
| vs B&H | **FAIL** (5.3% vs 13.6%) | **FAIL** (6.4% vs 26.1%) |
| Best Month Exclusion | PASS (82% from 1 month) | PASS (57% from 1 month) |
| Win Rate | PASS (59%) | **FAIL** (47%) |
| Max DD | PASS (-1.1%) | PASS (-41%) |
| Monthly Consistency | **FAIL** (too few months) | PASS (80% windows) |
| Concentration | PASS | PASS |
| Sub-Period | **FAIL** | **FAIL** |
| Sharpe Significance | **FAIL** (too few months) | PASS |
| Permutation Test | **FAIL** (31st percentile) | **FAIL** (35th percentile) |

Neither strategy survives the diligence gauntlet. The ensemble's best feature is low drawdown (-1.1%) but it lags buy-and-hold and can't distinguish itself from random noise (31st percentile). **No strategy tested is ready for live capital.**

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
| `src/pipeline.py` | End-to-end pipeline |
| `src/validation/diligence.py` | Reusable diligence/validation suite (8 checks) |
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
| 1 | vs Equal-Weight B&H | Beats holding all stocks equally | Strategy return > EW return |
| 2 | Best Month Exclusion | Survives removing the single best month | Return stays positive |
| 3 | Win Rate | Wins more than it loses (trade or monthly) | Win rate > 50% |
| 4 | Max Drawdown Duration | Drawdown severity and recovery | DD > -50%, recovery < 2yr |
| 5 | Monthly Consistency | Fraction of rolling 6-month windows positive | > 50% positive + Sharpe > 0 |
| 6 | Trade Concentration | Any single trade dominates PnL | Largest trade < 50% of total |
| 7 | Sub-Period Consistency | First half vs second half of test period | Both halves positive or 2nd not terrible |
| 8 | Sharpe Significance | Sharpe ratio vs noise threshold | Sharpe > 2/√(N) |

### Momentum Strategy Score: 8/9 → 1/9 (honest config)

The long-only momentum variant (bottom_n=0) scores 8/9 but the permutation test flags the signal as indistinguishable from random (66th percentile). The **long/short variant** (bottom_n=5, matching the walk-forward test) tells the true story:

| Check | Result |
|---|---|
| vs Equal-Weight B&H | **FAIL** — lags simple B&H |
| Best Month Exclusion | **FAIL** — 712% of return from 1 month |
| Trade Win Rate | PASS (55%) |
| Max Drawdown | PASS (-41%) |
| Monthly Consistency | **FAIL** — rolling windows inconsistent |
| Trade Concentration | PASS |
| Sub-Period Consistency | **FAIL** — negative in both halves |
| Sharpe Significance | **FAIL** — too few months |
| Permutation Test | **FAIL** — indistinguishable from random |

**Effective score: 2/9 on the checks that matter.** The momentum strategy fails every critical test — it doesn't beat B&H, its entire return comes from one month, and it can't distinguish itself from random noise. Kept as a validation harness only.

The module provides an instant reality check for any proposed strategy before live deployment.
