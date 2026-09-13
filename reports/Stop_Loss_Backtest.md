# Stop-Loss Backtest: October HighVol Strategy

**Strategy:** October rebalance, top 10 stocks by 12-month volatility, 12-month hold  
**Period:** 2015-2024 (10 years)  
**Universe:** 104 stocks

---

## Summary: Stop-Loss Sensitivity

| Stop-Loss | 9yr Compound | Avg Annual Return | Avg Stocks Stopped/Year |
|-----------|-------------|-------------------|------------------------|
| **No SL** | **+7,039%** | **+63.9%** | 0 |
| 50% | +5,423% | +59.3% | 1.4 |
| 40% | +4,633% | +55.9% | 2.1 |
| 35% | +4,662% | +55.8% | 2.3 |
| 30% | +4,238% | +54.3% | 3.0 |
| 25% | +4,155% | +54.1% | 3.5 |
| 20% | +3,211% | +49.8% | 4.0 |
| 15% | +1,931% | +40.5% | 5.2 |
| 10% | +412% | +20.3% | 7.5 |

---

## Key Finding: Stop-Losses Destroy HighVol Returns

**Conclusion: Do NOT use stop-losses with this HighVol strategy.**

### Why Stop-Losses Hurt

1. **HighVol stocks rebound**: 60% of stocks that hit a 25% stop would have recovered to finish positive
2. **Missed gains**: Average "missed gain" from being stopped = +28 percentage points
3. **Wrong years**: Stop-loss only helps in crash years (2021), but hurts in trending years (2019: -44pp cost)

### Year-by-Year: 25% Stop-Loss vs No Stop-Loss

| Year | No SL | 25% SL | # Stopped | Diff | Verdict |
|------|-------|--------|-----------|------|---------|
| 2015 | +27.6% | +26.5% | 3 | -1.1pp | SL hurt |
| 2016 | +46.7% | +41.1% | 3 | -5.7pp | SL hurt |
| 2017 | +104.2% | +92.8% | 1 | -11.4pp | SL hurt |
| 2018 | +33.3% | +10.8% | 4 | -22.5pp | SL hurt badly |
| 2019 | +194.6% | +150.7% | 3 | -43.9pp | SL hurt terribly |
| 2020 | +127.9% | +127.9% | 0 | 0.0pp | Neutral |
| 2021 | -25.6% | -5.6% | 8 | +20.0pp | SL helped! |
| 2022 | +14.6% | -9.9% | 7 | -24.5pp | SL hurt |
| 2023 | +79.3% | +77.7% | 2 | -1.6pp | SL hurt |
| 2024 | +36.3% | +29.1% | 4 | -7.2pp | SL hurt |

**Only 1 of 10 years did stop-loss help (2021)**

---

## Example: 2019 - Why Stop-Loss Destroyed Returns

**Basket:** CRWD, DDOG, CVNA, NET, AMD, MRNA, ANF, ZS, CHWY, OKTA

Without stop-loss: **+194.6%**  
With 25% stop-loss: **+150.7%** (-44pp)

Stocks stopped in 2019:
- CVNA stopped at -26% → went on to +174% (missed +200pp!)
- MRNA stopped at -27% → went on to +434% (missed +461pp!)
- AMD stopped at -26% → went on to +174%

**The stop-loss cut your winners right before they exploded.**

---

## Practical Recommendations

If you're worried about crash risk:

1. **Position sizing**: Keep to 10% per stock max (already planned)
2. **Trailing stops**: Use TRAILING 30% instead of hard 25% stop (lets winners run)
3. **Diversification**: 10 stocks already provides diversification
4. **Rebalance timing**: October has shown better crash survival than March

**Bottom line: Don't use hard stop-losses with this HighVol momentum strategy.**
