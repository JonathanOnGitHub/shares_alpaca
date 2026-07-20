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

**Result:** The baseline regression ensemble was the best ML configuration but still only returned 4.65%. None of the augmentations (market features, PEAD, momentum ranks, classification, thresholds) improved it. The signal is real but thin — the 28 technical features alone are weakly predictive of next-day returns.

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

- **Single-split backtests are misleading:** The large-cap 6m momentum returned +165% in a single 80/20 split but **-29% in walk-forward**. The walk-forward is the ground truth.
- **The only config that holds up out of sample:** Small-cap, 12-month lookback, 1-month skip, top 5/5. Returns +69% with 0.92 Sharpe across 19 monthly test periods.
- **Small caps need longer lookbacks:** 6-month momentum fails (-53% to -95%), but 12-month works (+69%). The extra noise in small caps requires a longer signal window.
- **Drawdowns are severe:** Even the best config has -34% max DD. This is inherent to concentrated long/short momentum — it wins most months but crashes hard when it's wrong.
- **Diversification reduces DD but caps returns:** Top 10 cuts DD from -46% to -22% but returns drop from 166% to 17%.

### Current Forward Position (Live Paper Trading)

The most promising strategy — **small-cap 12-month momentum, top 5 long** — is running on Alpaca paper trading:

```
LONG  MNDY (63 shares)
LONG  WIX  (93 shares)
LONG  UPST (173 shares)
LONG  COIN (31 shares)
LONG  ZS   (32 shares)
```

Rebalances monthly. Cron-ready: `.venv/bin/python -m src.trading.momentum_forward --universe small`

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

## DiligenceSuite — Reusable Validation Module

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

### Momentum Strategy Score: 3/8

Applied to the small-cap 12-month momentum strategy, the diligence checks revealed:
- **Lags B&H** (-17.1% vs -11.5%) — destroys capital vs simple holding
- **712% of return from 1 month** — without July 2026 the strategy loses 14.9%
- **Monthly Sharpe 0.08** — effectively zero risk-adjusted return
- **Negative in both halves** — consistently bad, not just unlucky

The module provides an instant reality check for any proposed strategy before live deployment.
