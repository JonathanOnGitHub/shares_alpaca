# SMA100 vs SMA200 Strategy Backtest Report

**Period**: 2015-01-01 to 2025-01-01 (10 years)  
**Universe**: 10 large-cap symbols (SPY, QQQ, IWM, AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA)  
**Capital**: $100,000 initial  
**Position sizing**: 10% per position, max 10 positions  
**Costs**: 0.1% slippage, no commission  

---

## Performance Summary

| Strategy | Total Ret | Ann Ret | Sharpe | Max DD | Win Rate | Trades |
|----------|-----------|---------|--------|--------|----------|--------|
| **SMA100** | 267% | 13.9% | 1.09 | -22.9% | 21.9% | 1,142 |
| **SMA200** | 511% | 19.9% | 1.22 | -23.0% | 20.2% | 623 |
| **Buy & Hold** | 1,339% | 30.6% | 1.21 | -43.1% | N/A | N/A |

---

## Key Findings

### SMA200 Outperforms SMA100
- **Total return: 511% vs 267%** — SMA200 earns almost double
- **Annual return: 19.9% vs 13.9%** — 6% more per year
- **Sharpe: 1.22 vs 1.09** — Better risk-adjusted returns
- **Fewer trades: 623 vs 1,142** — Lower transaction costs/turnover

### SMA100 Has Better Diligence Score
- **SMA100: 6/9 passed** vs **SMA200: 5/9 passed**
- SMA100 passes Max Drawdown Analysis (SMA200 fails due to 588-day drawdown streak)
- Both fail: vs B&H, Trade Win Rate, Permutation Test

### Both Beat Buy & Hold on Risk Metrics
- **Max DD: -23% vs -43%** — Both strategies cut drawdowns in half
- **Sharpe: 1.09-1.22 vs 1.21 (B&H)** — Similar or better risk-adjusted returns
- **Trade-off**: Accept ~10-17% lower annual returns for ~20% lower max drawdown

---

## Diligence Check Comparison

| Check | SMA100 | SMA200 |
|-------|--------|--------|
| vs Equal-Weight B&H | FAIL | FAIL |
| Best Month Exclusion | PASS | PASS |
| Trade Win Rate | FAIL | FAIL |
| Max Drawdown Analysis | **PASS** | FAIL |
| Monthly Consistency | PASS | PASS |
| Trade Concentration | PASS | PASS |
| Sub-Period Consistency | PASS | PASS |
| Sharpe Significance | PASS | PASS |
| Permutation Test | FAIL | FAIL |

---

## Analysis

### Why Does SMA200 Beat SMA100?
1. **Longer lookback = fewer false signals** — Less whipsaw in choppy markets
2. **Lower turnover** — 623 trades vs 1,142 (saves on slippage)
3. **Better trend capture** — 200-day captures bigger multi-month trends

### Why Does SMA100 Pass More Checks?
- SMA100's max drawdown duration is **474 days** vs SMA200's **588 days**
- SMA100's trade concentration is better (top trade = 10% vs 20% of PnL)
- Both still fail the permutation test — returns are not statistically significant vs random

### The Core Problem
Both strategies **underperform buy-and-hold on absolute returns** while delivering similar or slightly better Sharpe ratios. The Permutation Test failure (17-30th percentile) suggests these returns could be noise.

---

## Conclusion

**SMA200 is the better strategy** between the two:
- Nearly 2x the total return
- Higher Sharpe ratio
- Fewer trades (lower costs)

**But neither beats buy-and-hold** on absolute returns in this 10-year backtest. The 2015-2025 period was exceptionally bullish — the Nasdaq rose ~500% and a simple buy-hold on many of these stocks would have vastly outperformed both SMA strategies.

**Use case**: These strategies work as **drawdown reduction overlays** rather than return maximization tools. They shine in volatile or declining markets but give up significant upside in strong bull markets.

---

## Files

- `test_sma200.py` — Backtest script comparing SMA100, SMA200, and B&H (runs with `python3 test_sma200.py`)
- Uses existing `src/backtest/engine.py` and `src/validation/diligence.py`