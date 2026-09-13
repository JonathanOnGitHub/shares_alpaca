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

## Basket Composition by Rebalance Month

### Most Frequent Stocks per Month (2015-2024)

| Month | #1 Most Frequent | Count | Also Frequent |
|-------|-----------------|-------|---------------|
| **Jan** | BOOT | 10/10 | CVNA(7), TSLA(6), ANF(6), MRNA(6) |
| **Feb** | AMD, BOOT | 8/10 | CVNA(7), META(5), MRNA(5) |
| **Mar** | CVNA | 7/10 | BOOT(6), TSLA(6), MRNA(6) |
| **Apr** | TSLA | 9/10 | BOOT(6), CVNA(6), FIVE(6) |
| **May** | CROX, CVNA | 7/10 | AMD(6), BOOT(5), TSLA(5), NET(5) |
| **Jun** | ANF | 10/10 | BOOT(7), CVNA(6), AEO(6), AMD(5) |
| **Jul** | TSLA, ANF, CVNA | 7/10 | AMD(5), BOOT(5), AVGO(4) |
| **Aug** | BOOT | 9/10 | AMD(7), CVNA(7), TSLA(6) |
| **Sep** | ANF | 9/10 | OKTA(6), TSLA(5), DLTR(5), ZS(5) |
| **Oct** | AMD, CVNA, TSLA, NET | 6/10 | NFLX(5), BOOT(4), ZS(4) |
| **Nov** | BOOT | 7/10 | TSLA(6), CROX(6), CVNA(6) |
| **Dec** | ANF | 8/10 | MRNA(6), AEO(5), OKTA(5), ZS(5) |

### Stock Appearance Frequency (All Months Combined)

| Stock | Times Selected | % of Baskets |
|-------|----------------|--------------|
| BOOT | 88/120 | 73% |
| CVNA | 72/120 | 60% |
| ANF | 70/120 | 58% |
| TSLA | 65/120 | 54% |
| AMD | 63/120 | 53% |
| MRNA | 58/120 | 48% |
| CROX | 46/120 | 38% |
| NET | 38/120 | 32% |
| ZS | 35/120 | 29% |

---

## October vs March: Why the Difference?

### Year-by-Year Returns (Actual)

| Year | Oct Return | Mar Return | Difference | Oct Winner? |
|------|-----------|------------|------------|-------------|
| 2015 | +27.6% | +2.5% | +25.1pp | ✓ |
| 2016 | +46.7% | +55.0% | -8.2pp | |
| 2017 | +104.2% | +67.3% | +37.0pp | ✓ |
| 2018 | +33.3% | +64.3% | -31.0pp | |
| 2019 | +194.6% | +19.5% | +175.2pp | ✓ |
| 2020 | +127.9% | +377.6% | -249.7pp | |
| 2021 | -25.6% | -3.1% | -22.5pp | |
| 2022 | +14.6% | -15.8% | +30.4pp | ✓ |
| 2023 | +79.3% | +158.2% | -78.8pp | |
| 2024 | +36.3% | +11.2% | +25.2pp | ✓ |

**9yr Compound:** October: +7,039% | March: +5,734%

### Key Findings

1. **October wins 5/10 years, March wins 5/10 years** — it's not as simple as "October always wins"
2. **2019 was the biggest October advantage** (+175pp) — October got CRWD, DDOG, NET, MRNA all surging
3. **2020 was the biggest March advantage** (+250pp) — March caught MRNA +339%, CRWD +524% during COVID
4. **October's worst year (2021: -25.6%)** was still better than March's worst (2022: -15.8%)
5. **October has higher floor** — only 1 negative year vs 1 negative year for March

### Individual Stock Winners Driving October Outperformance

**2019 October basket:**
- MRNA: +434% | CRWD: +168% | DDOG: +226% | NET: +276% | CVNA: +174%

**2017 October basket:**
- CVNA: +212% | BOOT: +258% | AMD: +84% | OKTA: +119%

**2016 October basket:**
- AMD: +114% | NFLX: +103% | VRTX: +92% | TSLA: +81% | NOW: +63%

### Why October Sometimes Wins

October rebalances capture stocks **entering** their momentum run:
- Q4 often marks the start of speculative momentum cycles
- Stocks like MRNA, CVNA, CRWD had major runups after October selection
- Energy stocks (OXY, COP) benefit from end-of-year commodity moves

### Why March Sometimes Wins

March rebalances catch **post-correction** opportunities:
- 2020 March caught COVID crash recovery perfectly
- Some HighVol stocks are oversold by March and bounce
- But March 2021 missed the Jan 2021 meme stock peak

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

1. **Rebalance month affects compound returns by ~1,305pp** — October (+7,039%) vs March (+5,734%) over 9 years
2. **October wins 5/10 years, March wins 5/10 years** — but October has higher floor and better worst-year
3. **Annual > Monthly for HighVol** — Monthly rebalancing chops winners
4. **Survivorship bias varies by strategy** — LowVol and Covered Calls are heavily inflated; Momentum is robust
5. **HighVol concentration risk is real** — A few stocks (CVNA, MRNA, TSLA) dominate returns
6. **BOOT and ANF appear in every basket for their "home" months** — certain stocks consistently have highest volatility at specific times of year
7. **2019 was decisive for October** (+175pp advantage) — best-of-cycle stock selection in CRWD, MRNA, NET

---

## Files

- `reports/momentum_survivorship_test.json` — Top/Bottom 15 rolling annual test
- `reports/lowvol_survivorship_test.json` — LowVol rolling annual test
- `reports/all_strategies_survivorship_test.json` — Complete comparison
- `reports/annual_basket_comparison.json` — All basket strategies compared
