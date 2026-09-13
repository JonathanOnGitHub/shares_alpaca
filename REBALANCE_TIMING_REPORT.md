# Rebalance Timing & Survivorship Bias Analysis

**Date:** September 2026  
**Universe:** 104 stocks (large/mega-cap)  
**Period:** January 2015 – December 2025 (~10.5 years)

---

## Executive Summary

We tested whether **rebalance month** affects HighVol strategy returns when using:
- Volatility measured using **previous 12 months** of data
- **12-month hold** period for each rebalance

**Result:** Rebalance month matters enormously — a range of **3,858 percentage points** between best (October: +5,907%) and worst (March: +2,049%).

---

## Month-by-Month Rebalance Test Results

| Rank | Month | Avg Ann Ret | 9yr Compound | vs Best |
|------|-------|------------|--------------|---------|
| 1 | **Oct** | +62.5% | **+5,907%** | — |
| 2 | May | +58.6% | +5,537% | -370 |
| 3 | Jan | +62.1% | +4,800% | -1,107 |
| 4 | Nov | +52.8% | +4,119% | -1,788 |
| 5 | Feb | +57.3% | +4,067% | -1,840 |
| 6 | Dec | +55.9% | +3,807% | -2,100 |
| 7 | Jul | +49.5% | +3,667% | -2,240 |
| 8 | Aug | +52.1% | +3,461% | -2,446 |
| 9 | Sep | +47.6% | +3,340% | -2,567 |
| 10 | Jun | +47.8% | +2,822% | -3,085 |
| 11 | Apr | +45.2% | +2,206% | -3,701 |
| 12 | **Mar** | +51.2% | **+2,049%** | **-3,858** |

### Statistical Summary

| Metric | Value |
|--------|-------|
| Range of compounds | 3,858 pp |
| Std deviation | 1,138% |
| Best month | October (+5,907%) |
| Worst month | March (+2,049%) |

---

## Annual vs Monthly Rebalancing Comparison

| Strategy | Rebalance | Avg Return | 9yr Compound |
|----------|-----------|------------|--------------|
| HighVol 10 | Annual | +66.0% | +4,068% |
| HighVol 10 | Monthly | +36.9% | +1,409% |
| LowVol 10 | Monthly | +13.7% | +125% |

**Key finding:** Annual rebalancing crushes monthly for HighVol because:
- Monthly "chops off winners" — you rotate out before stocks compound
- Annual gives winners time to run

---

## Survivorship Bias Summary

| Strategy | Original | Debiased | Verdict |
|----------|----------|----------|---------|
| covered_calls_mega | 2,170% | ~250% | **DEBUNKED** (~9x inflation) |
| lowvol_small_inverse | 952% | ~159% | **DEBUNKED** (~6x inflation) |
| momentum_smallcap | 1,512% | ~952% | **VALID** (~1.6x inflation) |
| HighVol 10 | N/A | ~3,000-5,000% | **VALID but concentration-biased** |

---

## Key Findings

1. **Rebalance month has massive effect** — October beats March by 3,858pp over 9 years
2. **Annual > Monthly for HighVol** — Monthly rebalancing chops winners
3. **Survivorship bias varies by strategy** — LowVol and Covered Calls are heavily inflated; Momentum is robust
4. **HighVol concentration risk is real** — A few stocks (CVNA, MRNA, TSLA) dominate returns

---

## Files

- `reports/momentum_survivorship_test.json` — Top/Bottom 15 rolling annual test
- `reports/lowvol_survivorship_test.json` — LowVol rolling annual test
- `reports/all_strategies_survivorship_test.json` — Complete comparison
- `reports/annual_basket_comparison.json` — All basket strategies compared
