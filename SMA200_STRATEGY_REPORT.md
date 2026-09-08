# SMA Crossover Strategy Backtest Report

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
| **SMA250** | 467% | 19.0% | 1.18 | -25.6% | 20.0% | 440 |
| **SMA300** | 482% | 19.3% | 1.16 | -26.9% | 17.8% | 414 |
| **Buy & Hold** | 1,339% | 30.6% | 1.21 | -43.1% | N/A | N/A |

---

## Key Findings

### SMA200 is the Best Performer
- **Total return: 511%** — Highest among all SMA strategies
- **Sharpe: 1.22** — Best risk-adjusted returns
- **Annual return: 19.9%** — Nearly tied with SMA300 (19.3%)

### Longer SMAs = Less Trades, More Concentration
| Period | Trades | Top Trade % | Top 3 % |
|--------|--------|-------------|---------|
| SMA100 | 1,142 | 10% | 22% |
| SMA200 | 623 | 20% | 37% |
| SMA250 | 440 | 20% | 38% |
| SMA300 | 414 | 25% | 46% |

Longer SMAs reduce trades but increase PnL concentration in fewer winners.

### Longer SMAs = Worse Max Drawdown
- **SMA100**: -22.9% max DD, 474-day streak → passes Max Drawdown check
- **SMA200+**: -23% to -27% max DD, 500+ day streaks → fail Max Drawdown check

### The Sweet Spot: SMA200
- Highest total and risk-adjusted returns
- Half the trades of SMA100 (less friction)
- Best balance between signal quality and turnover

---

## Diligence Check Comparison

| Check | SMA100 | SMA200 | SMA250 | SMA300 |
|-------|--------|--------|--------|--------|
| vs Equal-Weight B&H | FAIL | FAIL | FAIL | FAIL |
| Best Month Exclusion | PASS | PASS | PASS | PASS |
| Trade Win Rate | FAIL | FAIL | FAIL | FAIL |
| Max Drawdown Analysis | **PASS** | FAIL | FAIL | FAIL |
| Monthly Consistency | PASS | PASS | PASS | PASS |
| Trade Concentration | PASS | PASS | PASS | PASS |
| Sub-Period Consistency | PASS | PASS | PASS | PASS |
| Sharpe Significance | PASS | PASS | PASS | PASS |
| Permutation Test | FAIL | FAIL | FAIL | FAIL |

**SMA100 passes 6/9; SMA200/250/300 pass 5/9**

All strategies fail: vs B&H (absolute returns), Trade Win Rate (low ~20%), and Permutation Test (not statistically significant).

---

## Analysis

### Why Does SMA200 Beat SMA300 Despite Similar Logic?
- 200-day is ~40 weeks — captures roughly 2 quarterly earnings cycles
- 300-day is ~60 weeks — starts to lose signal relevance (1+ year)
- The additional 100 days of lag means SMA300 enters trends later and exits later

### The Trend-Following Paradox
All strategies show:
- **Low win rate (17-22%)** — Most trades are small losses
- **Good Sharpe (1.09-1.22)** — Winners are large enough to offset losers
- **High trade concentration** — Few trades drive most of the PnL

This is the classic trend-following profile: "lose small, win big."

### Why All Fail Permutation Test
All strategies rank 17-30th percentile vs random — their returns are not statistically distinguishable from noise given the 500 random permutations.

---

## Conclusion

**SMA200 is the winner** among tested moving average strategies:
- Highest total return (511%)
- Best Sharpe (1.22)
- Moderate turnover (623 trades)
- Lowest PnL concentration among longer SMAs

**But none beat buy-and-hold** on absolute returns in this 10-year bull market. The SMA strategies work as **drawdown reduction tools** (max DD -23% vs -43%) but give up significant upside.

**Use case**: Risk-averse investors who prioritize capital preservation over maximal returns. The ~10% annual return gap vs B&H is the "insurance premium" for the -20% lower max drawdown.

---

## Files

- `test_sma200.py` — Backtest script comparing SMA100, SMA200, SMA250, SMA300, and B&H
- Uses existing `src/backtest/engine.py` and `src/validation/diligence.py`