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

---

## Covered Calls Mega: Live Trading Assessment

**Best-performing strategy: 8/9 checks passed**

### Backtest Performance
| Metric | Value |
|--------|-------|
| Total Return | 93.5% |
| Annual Return | ~18% |
| Sharpe | 1.43 |
| Max Drawdown | -17.2% |
| Win Rate | 100% (4 trades) |
| Period | ~4 years |

### What the Strategy Does
- Hold 8 mega-cap positions (SPY, QQQ, etc.), equal-weight
- Sell 30-day calls at 5% OTM, monthly roll
- Collect premium = immediate income
- If stock above strike at expiry: assigned (sold at strike)
- If below strike: option expires, hold stock, collect premium

### ⚠️ Critical Concerns for Live Trading

**1. Theoretical Black-Scholes Premiums**
- Backtest uses `scipy.stats.norm` for Black-Scholes
- Real market premiums depend on bid/ask spreads, liquidity, volatility smile
- **Gap between theoretical and actual premiums could be 10-30%**

**2. Very Few Trades (4 round-trips)**
- Only 4 complete buy-sell cycles in entire backtest
- Insufficient statistical significance
- Strategy may be lucky, not skilled

**3. Short Backtest Period**
- Only ~4 years of data
- Does not include 2020 crash (COVID)
- May underperform in high-volatility regimes

**4. Permutation Test is Borderline**
- 52nd percentile vs random — barely passes
- Not statistically distinguishable from noise at 5% level

### ✅ Positives for Live Trading

- **Beats B&H** in backtest (93% vs 68%)
- **Lower max drawdown** (-17% vs -43%)
- **High Sharpe (1.43)** — statistically significant
- **Monthly income** — predictable premium collection
- **Simple execution** — sell calls against existing stock
- **Well-understood risk** — capped upside, defined loss

### 📋 Implementation Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Options permissions | ❌ Required | Need Level 2/3 (covered calls) |
| Margin account | ❌ Yes | To hold stock + sell calls |
| Black-Scholes model | ⚠️ Modify | Use real-time market premiums |
| Execution infrastructure | ⚠️ Build | Options order routing |
| Volatility model | ⚠️ Improve | Include IV smile, not just realized vol |
| Early exercise risk | ❌ Not modeled | Relevant for deep ITM calls |

### Verdict: **Cautious MAYBE**

**Arguments FOR:**
- Best diligence score (8/9)
- Beats B&H with half the drawdown
- Statistically significant Sharpe (1.43)
- Simple, well-understood mechanics

**Arguments AGAINST:**
- Only 4 trades in backtest — luck vs skill?
- Permutation test is borderline (52nd percentile)
- Theoretical premiums, not real market data
- Missing 2020 crash data — unknown behavior in crisis

**Recommendation:**
1. **Paper trade first** for 6-12 months with real market premiums
2. Compare actual premium collected vs theoretical
3. Extend backtest to include 2020 if possible
4. If paper results match ~80% of theoretical, proceed with caution
5. Start with small position sizes (1-2 contracts) before scaling

---

## All Strategies Ranked by Diligence Score

| Rank | Strategy | Passed | Pct |
|------|----------|--------|-----|
| 1-2 | covered_calls_mega, momentum_smallcap | 8/9 | 89% |
| 3-6 | lowvol_small_inverse, lowvol_small_quality, swing_mega, val_timing_mega | 7/9 | 78% |
| 7-15 | jt_J6_K6_quintile, ma_timing_spy (50d,100d), meanrev_mega/small, **sma100**, swing_smallcap, trend_multiasset | 6/9 | 67% |
| 16-21 | crossasset_rot_long, momentum_smallcap_ls, **sma200/250/300**, val_timing_small | 5/9 | 56% |
| 22-26 | ensemble_mega, jt_J6/K9 variants, lowvol_mega_inverse | 4/9 | 44% |
| 27-28 | jt_J6_K3_decile, lowvol_mega_quality | 3/9 | 33% |
| 29-30 | crossasset_rot_ls, lowvol_mega_minvar | 1/9 | 11% |

**Total: 30 strategies tested**

### Key Observations
- **Covered calls & momentum** are the only strategies to pass 8/9 checks
- **SMA100 (6/9)** ranks tied 13th — best among the SMAs
- **Long/short strategies** generally underperform long-only versions
- **Low-vol small-cap** strategies do well (7/9), but low-vol mega-cap variants do poorly (1-4/9)

---

## Files

- `test_sma200.py` — Backtest script comparing SMA100, SMA200, SMA250, SMA300, and B&H
- Uses existing `src/backtest/engine.py` and `src/validation/diligence.py`