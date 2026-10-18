# Technical Backtests — do patterns, pullbacks and breakouts actually pay?

A systematic attempt to find an edge in classical technical entry rules, at three
resolutions (1-minute, 5-minute, daily), long and short. **Result: no rule, at any
resolution, produced an edge that survives transaction costs.**

The one statistically significant effect found is *negative* — buying daily
breakouts underperforms random entry — and its profitable mirror is smaller than
the cost of being short. Both are documented below.

---

## TL;DR

| Study | Resolution | Signals | Verdict |
|---|---|---|---|
| 1. Reclaim vs limit entry | 5m | 1,659 | Reclaim is **insurance, not alpha** |
| 2. Candlestick patterns | 5m | 5,450 | All six rules −0.09% to −0.41% |
| 3. Exit schemes | 5m | 8,760 | Every exit within 2bp of every other |
| 4. Candlestick patterns | daily | 41,650 | Raw means positive, **excess ≈ 0** |
| 5. Fading the breakout | daily | 17,054 | Edge real (+0.11%, t=3.57), **costs 3–5× it** |

The unifying finding: **at both resolutions the residual mispricing is smaller than
the friction.** When your measured edge and your measured cost are the same order
of magnitude, you are not looking at an opportunity.

---

## Two methodological traps this study had to fix

Both of these would have manufactured a false positive. They are the more
transferable result here than the tables.

**1. The unconditional baseline.** This sample covers a rising market. *Any* long
entry has positive drift, so a raw mean of +0.2% is not evidence of edge. The
correct control is the same stop and target applied to **every bar**. Excess over
that baseline is the only number that means anything.

This is the single biggest error in retail backtesting. In study 4 above, four of
six rules show a *positive* raw mean and every one of them is worthless once the
baseline (+0.147%) is subtracted.

**2. Entry timing.** The 5m studies enter at the **close of the signal bar**, i.e.
they assume you can act on a bar at the instant it completes. The daily study
enters at the **next bar's open**. Where the two disagree, trust the daily
method — it costs something to be honest about timing.

A third, subtler one: **stops anchored to the trigger price rather than the entry
price** silently inflates results. The reclaim studies corrected this in v2; it
moved the stop-out rate from 1.6% to 47.5%.

---

## Study 1 — Reclaim vs limit entry (5m)

Motivation: a resting limit order catches falling knives (the TCRT trade filled at
$2.47 and was $2.31 six minutes later, −6.5% MAE). Does waiting for a reclaim fix
that, and what does the insurance cost?

23 US symbols, 60 days of 5m bars, **1,659 trades**, equal risk per trade from
entry, 0.10% cost:

| Mechanism | n | Win% | Mean% | MAE% | Stopped% |
|---|---|---|---|---|---|
| LIMIT (buy the dip) | 1,283 | 15.4 | −3.05 | 5.41 | 47.5 |
| RECLAIM (wait for confirmation) | 359 | 42.3 | −0.39 | 2.00 | 2.9 |

Paired within session: mean diff **+0.14%**, median **−0.17%**, reclaim better in
only **8.4%** of sessions.

**Reading: reclaim cuts the tail dramatically** (stop-outs 47.5% → 2.9%, MAE
5.41% → 2.00%) **but neither mechanism makes money.** The positive mean diff with
a negative median is the signature of insurance, not edge — you give up return in
the typical session to avoid the catastrophic one.

Three confounds were found and removed across `backtest.py` → `backtest4.py`:
stop anchored to the trigger; unequal risk between arms; and reclaim signals
arriving later in the day (59.5 vs 77.4 bars remaining), which was tested by
restricting both arms to entries before 15:00 ET — the asymmetry held.

---

## Study 2 — Candlestick patterns (5m)

Six long-entry rules, same 23 symbols and 60 days, **identical exits** so the only
variable is the entry predicate. 1 ATR stop, 1.5 ATR target, 0.10% cost,
**5,450 trades**:

| Rule | n | Win% | Mean% |
|---|---|---|---|
| Bullish engulfing | 1,336 | 44.3 | −0.22 |
| Hammer | 1,208 | 44.3 | −0.22 |
| Inside-bar breakout | 813 | 42.8 | −0.14 |
| NR7 breakout | 1,313 | 45.5 | −0.17 |
| Trend pullback | 437 | **48.3** | **−0.09** |
| Reclaim (baseline) | 343 | 42.0 | −0.41 |
| **ALL** | **5,450** | **44.6** | **−0.20** |

**No rule is positive.** The best (trend pullback) still loses, and the spread
between best and worst is 0.32pp — noise around a constant.

---

## Study 3 — Exit schemes (5m)

If the entry doesn't matter, does the exit? Same **8,760 signals**, six schemes:

| Exit | Win% | Mean% | Stopped% |
|---|---|---|---|
| End of day only | 44.0 | −0.22 | — |
| 0.5 / 1.0 ATR | 43.3 | −0.21 | 11.2 |
| 1.0 / 1.5 ATR | 43.9 | −0.22 | 1.1 |
| 1.0 / 3.0 ATR | 43.9 | −0.21 | — |
| 2.0 / 3.0 ATR | 44.0 | −0.22 | 0.1 |
| Trailing 0.75 ATR | 9.8 | −0.23 | — |

**Every scheme within 2bp of every other.** That is the diagnostic: when neither
entry nor exit moves the result, the signal has no content and you are paying
costs plus bid-ask bounce. (5m lag-1 autocorrelation is ≈ −0.157; buying at a bar
close after a bullish pattern means buying the offer and marking against the bid.)

---

## Study 4 — Candlestick patterns (daily)

The 5m result was plausibly a cost artefact — spread on a 0.15% five-minute range.
On daily bars the range is ~2%, so the drag should vanish. It does. **51 symbols,
2017–2026, 41,650 signals**, entered at the next bar's open.

Unconditional baseline (same exit on all 112,771 bars): **+0.147% mean, 46.4% win,
median −0.865%**.

| Rule | n | Win% | Mean% | Excess% | t |
|---|---|---|---|---|---|
| Breakout | 17,054 | 45.3 | −0.00 | **−0.15** | **−5.09** |
| Engulfing | 3,964 | 46.7 | +0.09 | −0.05 | −0.77 |
| Hammer | 4,951 | 47.1 | +0.21 | +0.06 | +0.96 |
| Inside bar | 2,934 | 46.0 | +0.03 | −0.12 | −1.61 |
| NR7 | 6,048 | 46.4 | +0.09 | −0.05 | −0.97 |
| Pullback | 6,699 | 46.8 | **+0.23** | +0.08 | +1.52 |

**The raw means turn positive and that is entirely beta.** Rule win rates are
within 1.8pp of the baseline's 46.4% — the patterns do not change the probability
of winning at all. Pullback's headline +0.23% is +0.08% of excess at t=1.52.

Exit sweep on the same signals: every scheme's excess lands between −0.08% and
+0.03%. Same verdict as 5m — the exit is never the driver.

At the real **0.30% round-trip** cost for US shares on Trading 212: pullback
+0.03%, hammer +0.01%, everything else negative (breakout −0.20%).

**The one significant result is negative.** Buying daily breakouts is worse than
random entry at t=−5.09. Do not buy strength on daily bars.

---

## Study 5 — Fading the breakout

A t of −5.09 on 17,000 trades is worth chasing. Shorting the identical signals,
gross (no cost either side):

| Scheme | Win% | Mean% | Base% | Excess% | t |
|---|---|---|---|---|---|
| 1.0/1.5 ATR, 5d | 40.4 | −0.08 | −0.18 | **+0.11** | **+3.57** |
| 1.0/3.0 ATR, 10d | 29.6 | −0.18 | −0.29 | +0.11 | +2.91 |
| 2.0/3.0 ATR, 10d | 41.4 | −0.43 | −0.54 | +0.11 | +2.20 |
| 2.0/6.0 ATR, 20d | 33.8 | −0.67 | −0.90 | +0.23 | +3.52 |
| No stop, 20d | 43.9 | −2.05 | −2.05 | +0.00 | +0.04 |

**The fade is statistically real** — the long result was not an artefact. But the
edge is **3–5× smaller than the cost of being short**:

| Short cost | Net excess |
|---|---|
| 0.30% (FX only) | −0.19% |
| 0.42% (FX + spread + low borrow) | −0.31% |
| 0.60% (FX + spread + hard-to-borrow) | −0.49% |

Three further disqualifiers:

- **Sign unstable by year.** Positive 2018, 2020, 2021, 2022, 2026; negative 2017,
  2019, 2023, 2024, 2025. Choppy years work, trending years fail — and the last
  three years are all negative.
- **Negative median (−1.21%) with a positive mean.** 59.6% of trades lose; P&L
  rides a tail (p99 +11.3%, best +27.3%, worst −16.6%). This is a low-hit-rate,
  tail-driven trade — *not* the premium-selling profile it superficially resembles.
- **Not executable long-only.** Trading 212's Invest/ISA accounts cannot short.
  A CFD account adds wider spreads plus daily financing, so the real cost is worse
  than modelled.

---

## Conclusions

1. **No classical technical entry rule shows positive expectancy** at 1m, 5m or
   daily resolution, long or short, once costs and the unconditional baseline are
   accounted for.
2. **The exit is not the problem.** Fourteen exit schemes across two timeframes
   move results by less than 0.1pp.
3. **The only significant effect is negative** — daily breakouts (t=−5.09) — and
   its mirror is smaller than the cost of trading it.
4. **Confirmation-based entry (reclaim) is worth having as risk control**, cutting
   MAE from 5.41% to 2.00%, but it buys safety, not return.
5. Where this project *has* found structure is **event-driven** — merger closes,
   reverse-merger announcements, after-hours catalysts, sector rotation. That is
   information a candle cannot see.

## Limitations

- **Survivorship bias.** The universe is named today, so it is today's survivors.
  Large caps with long history were added deliberately to dilute this, but it is
  not eliminated.
- **The 5m studies cover one 60-day window** in a single regime. Treat as
  directional, not conclusive. Study 4 spans 2017–2026 and is far stronger.
- **Costs are modelled, not measured.** 0.10% base; 0.30% for real US-share
  round-trip FX on Trading 212. No slippage, no queue position, no partial fills.
- **No shorting costs in the gross fade table** by design — they are applied
  explicitly afterwards.
- Data ends 2026-10-15.

## Files

| File | Purpose |
|---|---|
| `backtest.py` → `backtest4.py` | Reclaim-vs-limit iteration history (study 1) |
| `backtest_patterns.py` | Six pattern rules, 5m (study 2) |
| `backtest_exits.py` | Six exit schemes, 5m (study 3) |
| `backtest_daily.py` | Six pattern rules, daily, with baseline + t-stats (study 4) |
| `backtest_fade.py` | Short side of the breakout (study 5) |
| `daily_signals.csv` | 41,650 daily signals with per-trade returns |
| `fade_signals.csv` | 17,054 faded signals with per-trade returns |

## Running

```bash
pip install yfinance pandas numpy

python backtest_daily.py     # study 4 — start here
python backtest_fade.py      # study 5
python backtest_patterns.py  # study 2 (downloads 5m bars, slower)
python backtest_exits.py     # study 3
```

Bars are cached as pickles in `cache/` and `cache_daily/` (gitignored) so repeat
runs don't re-hit Yahoo. Override with `BACKTEST_CACHE=/path/to/cache`. The first
run of each study downloads 51 symbols and takes a few minutes; subsequent runs
take seconds.
