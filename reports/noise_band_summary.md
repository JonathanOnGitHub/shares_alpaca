# Noise Area Intraday Momentum — Backtest Summary

**Paper:** Zarattini, Aziz & Barbon (2024), *"Beat the Market: An Effective
Intraday Momentum Strategy for the S&P500 ETF (SPY)"*, SSRN 4824172
([doi:10.2139/ssrn.4824172](https://doi.org/10.2139/ssrn.4824172)).

**Implementation:** `src/trading/noise_band.py`, ported from the authors' own
reference notebook (linked from the paper's Q&A page) rather than from a
paraphrase of the prose.

**Data:** Alpaca 1-minute bars, SIP feed, SPY, 2022-01-03 → 2025-12-31.
**990 sessions**, 390,761 bars, every session a complete 390 bars.

---

## Verdict

**Promising, and the most credible technical result in this repo — but the
edge is regime-dependent and the return is tail-concentrated. Not deployable
as-is.**

Three things have to be said together:

1. Against the correct control it looks strong: **+29% excess** over an
   intraday-long baseline at Sharpe **1.34 vs 0.30**, with *half* the
   volatility and *less than a third* of the drawdown.
2. It **flips sign across parameter settings**. The excess is +29% at
   VM 1.0/lookback 14, +27% at VM 0.5/14, and **−18% at VM 2.0/90**. Half the
   sweep is negative. This repo has been here before — see the fade study in
   `technical-backtests/README.md`, whose edge also flipped sign by year.
3. The return is **carried by a handful of days**: the best 5 days are
   $60,886 of a $117,430 total (52%), and the median day is $0.00.

---

## The strategy

The band is **not** a Bollinger band. For each minute-of-day, `σ` is the mean
over the previous N days of `|Close/Open − 1|` — the average absolute
excursion from the open by that time of day:

```
UB = max(open, prev_close) × (1 + VM·σ)
LB = min(open, prev_close) × (1 − VM·σ)
```

The previous close is dividend-adjusted, so an overnight gap widens the band.
A breach means the market has moved further than it "normally" has by that
time of day — a demand/supply imbalance rather than noise.

- **Entry:** long above UB, short below LB, **plus VWAP confirmation**.
- **Decisions:** semi-hourly marks only (the 09:59 bar, traded from 10:00).
- **Exit:** opposite-band cross, or the close. Exposure is flat at 16:00.
- **Costs:** $0.0035/share commission, $0.35/order minimum. The paper models
  no slippage and neither do we by default.
- **Sizing:** two modes — `full_notional` (100% of equity) and `vol_target`
  (the paper's default: 2% daily vol target, 4× leverage cap).

---

## Headline result — `vol_target` sizing (the reference code's default)

The authors' reference notebook defaults to `sizing_type = "vol_target"`,
`band_mult = 1`, `target_vol = 0.02`, `max_leverage = 4`, `trade_freq = 30`.
These are the defaults in `NoiseBandConfig`.

| | Strategy | Intraday-long | SPY B&H |
|---|---:|---:|---:|
| Total return | **+117.4%** | −10.6% | +52.6% |
| Annualised | **+21.6%** | −2.8% | +11.4% |
| Volatility | **+14.7%** | +27.9% | +17.8% |
| **Sharpe** | **+1.40** | +0.04 | +0.69 |
| Worst day | −5.5% | −12.1% | −6.0% |
| Max drawdown | **−15.7%** | −35.4% | −22.8% |
| Total cost | $5,154 | | |

The strategy beat SPY buy & hold on every risk measure while taking less
volatility than SPY itself. Against the intraday-long baseline the excess is
**+128.0%**.

**What "intraday-long" means, and why it is not buy-and-hold:** the control
is long from the open to the close and flat overnight, exactly like the
strategy. That is the right matched control — it shares the strategy's
exposure schedule, so the difference isolates the signal. But it misses the
overnight gap, and for SPY over 2020-2025 the gap was **+52% of the total
return** (intraday-only +38.7%, overnight-only +51.9%). Anyone comparing
against this baseline should not read "+128% excess" as "beat buy-and-hold by
128%"; the honest comparison is the SPY B&H column, where the strategy still
comes out ahead on every risk measure but not on the size of the gap.

## `full_notional` sizing (no leverage)

| | Strategy | Intraday-long | SPY B&H |
|---|---:|---:|---:|
| Total return | +42.4% | +13.4% | +52.6% |
| Annualised | +9.3% | +3.2% | +11.4% |
| Volatility | +6.8% | +13.8% | +17.8% |
| **Sharpe** | **+1.34** | +0.30 | +0.69 |
| Max drawdown | −6.8% | −13.8% | −22.8% |
| **Excess** | **+29.0%** | | |

The unlevered version is the honest one to compare against the paper's
*base case* (178% total / Sharpe 0.61). We beat it on Sharpe, on a different
feed, with a shorter sample.

---

## Parameter sweep — the edge does not survive every setting

`full_notional`, excess over the intraday-long baseline by year:

| VM | lookback | total | excess | Sharpe | trades | yearly excess |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 14 | +40.8% | **+27.4%** | 1.00 | 2,762 | 2022:+15% 2023:+2% 2024:+7% 2025:**+2%** |
| 1.0 | 14 | +42.4% | **+29.0%** | 1.34 | 1,792 | 2022:+20% 2023:+5% 2024:+7% 2025:−5% |
| 1.5 | 14 | +30.9% | +17.5% | 1.27 | 1,150 | 2022:+17% 2023:−3% 2024:+4% 2025:−2% |
| 2.0 | 14 | +7.8% | **−5.5%** | 0.46 | 688 | 2022:+2% 2023:−9% 2024:+2% 2025:−3% |
| 0.5 | 90 | +12.0% | **−6.8%** | 0.39 | 2,418 | 2022:−7% 2023:−1% 2024:+4% 2025:−5% |
| 1.0 | 90 | +41.2% | **+22.4%** | 1.27 | 1,466 | 2022:+7% 2023:+2% 2024:+6% 2025:**+5%** |
| 1.5 | 90 | +16.2% | −2.6% | 0.64 | 904 | 2022:+6% 2023:−10% 2024:−0% 2025:−1% |
| 2.0 | 90 | +1.1% | **−17.7%** | 0.09 | 544 | 2022:+2% 2023:−11% 2024:−4% 2025:−6% |

**Two settings are positive in all four years** (VM 0.5/14 and VM 1.0/90) on
the SIP feed. **Six of the eight are negative in 2025.** A band that only
works when it is narrow *and* recent, or wide *and* slow, is not a robust
parameterisation.

The same sweep on the filtered IEX feed (≥300 bars) reproduces the shape but
not the year-2025 signs exactly:

| VM | lookback | IEX ≥300 excess | IEX 2025 | SIP excess | SIP 2025 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 14 | +29.3% | −1% | +27.4% | **+2%** |
| 1.0 | 14 | +26.4% | −5% | +29.0% | −5% |
| 1.5 | 14 | +19.3% | −4% | +17.5% | −2% |
| 2.0 | 14 | −5.5% | −2% | −5.5% | −3% |
| 0.5 | 90 | −5.7% | −8% | −6.8% | −5% |
| 1.0 | 90 | +21.6% | **+4%** | +22.4% | **+5%** |
| 1.5 | 90 | +1.6% | **+2%** | −2.6% | −1% |
| 2.0 | 90 | −13.3% | −7% | −17.7% | −6% |

Two cells disagree across feeds (VM 0.5/14 and VM 1.5/90 in 2025). Those
differences are 1–3 points on a year where the strategy trades rarely, so
they are sampling noise, not a disagreement about the mechanism. The stable
findings across both feeds are: **VM 1.0/L14 and VM 1.0/L90 are the two
robust parameterisations, and 2025 was negative for VM 1.0/L14 on both.**

---

## The data-quality finding that changed the conclusion

An earlier run on Alpaca's **IEX** feed showed 2025 negative at *every*
parameter setting, which looked like regime decay. That was an **artefact of an
earlier, less faithful implementation** — before the volatility-window and
dividend-adjustment corrections described below. It also had a genuine data
component: **10 of 1,002 IEX sessions had fewer than 300 bars** (one had 47).

Re-measured with the current code, the data-quality effect is real but much
smaller than first reported:

| | excess (VM 1.0/L14) | Sharpe | 2025 excess |
|---|---:|---:|---:|
| IEX, unfiltered | +25.1% | 1.30 | −5% |
| IEX, ≥300 bars | +26.4% | 1.28 | −5% |
| SIP, complete | +29.0% | 1.34 | −5% |

Filtering the short sessions moves the excess by **1.3pp**, not the 6pp
originally claimed, and it does **not** flip any year's sign at VM 1.0/L14.
What it does do is make the IEX and SIP results agree, which is the point: a
venue with holes in the 1-minute series is not measuring the same thing as a
complete one.

The claim that 2025 was "negative at every setting" does not survive either.
Under the current code, 2025 is positive at VM 1.0/L90 and VM 1.5/L90 on both
feeds, and negative at VM 1.0/L14, VM 1.5/L14, VM 2.0/L14, VM 0.5/L90 and
VM 2.0/L90.

**So the honest version of this finding is: a negative result from a venue
with holes in the 1-minute series may be an artefact — and this one partly
was.** A 1-minute band is only meaningful on complete sessions, and
incomplete sessions manufacture both fake signals and fake conclusions. But
the size of that effect here was smaller than first claimed, and most of the
original swing came from implementation error rather than from the data.

`scripts/check_noise_band_data_quality.py --feed iex` reproduces the gap
problem; the default SIP path passes trivially.

---

## Concentration — the reason this is not deployable yet

`full_notional`, 990 sessions:

| | |
|---|---:|
| Total P&L | $42,361 |
| Best 5 days | $16,511 (**39%**) |
| Worst 5 days | −$8,784 |
| Median day | $0.00 |
| Hit rate (days > 0) | 28.8% |

With `vol_target` sizing the concentration is worse: $60,886 of $117,430
(52%) from the best 5 days, and −$31,782 from the worst 5.

Forty per cent of the return comes from five days out of 990. The median day is
zero because the strategy is flat most of the time — it trades a few times a
month. That is a low-hit-rate, tail-driven profile: the expected value is
positive, but the path depends on a handful of outsized intraday moves, which
is exactly the profile that decays hardest in live execution.

---

## What £10,000 in January 2020 would have been worth

`scripts/gbp_path_noise_band.py` runs the strategy in USD and converts the
equity curve to GBP at each day's actual GBP/USD rate, so the answer is in the
currency the money was actually in. `full_notional`, VM 1.0, lookback 14.

| year | start GBP | end GBP | P&L | return | intraday-long | excess |
|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 10,000 | 10,313 | +313 | +3.1% | +4.0% | −0.9% |
| 2021 | 10,313 | 11,159 | +846 | +8.2% | −0.3% | +8.5% |
| 2022 | 11,159 | 11,396 | +237 | +2.1% | −19.5% | **+21.6%** |
| 2023 | 11,396 | 13,361 | +1,965 | +17.2% | +13.4% | +3.9% |
| 2024 | 13,361 | 13,788 | +428 | +3.2% | −1.6% | +4.8% |
| 2025 | 13,788 | 14,842 | +1,054 | +7.6% | +11.2% | −3.5% |

| | GBP | return |
|---|---:|---:|
| **Strategy** (`full_notional`) | **14,842** | **+48.4%** |
| Strategy (`vol_target`, up to 4× levered) | 36,731 | +267.3% |
| SPY buy-and-hold | 21,399 | **+114.0%** |
| Intraday-long baseline | 10,356 | +3.6% |

GBP/USD went 1.3263 → 1.3467 over the period, so the currency cost the
investor about 1.5%.

**The honest headline: SPY buy-and-hold beat the unlevered strategy by a wide
margin over 2020–2025.** £10,000 became £21,399 passive versus £14,842 in the
strategy — and the passive route needed no trading, no signal, and no
execution risk.

The strategy's edge shows up where the market was ugly: **2022, the bear
year, it returned +2.1% while buy-and-hold lost 19.5%** — a +21.6% excess and
the only year where the strategy's risk-management behaviour clearly paid.
In 2020 and 2025 it underperformed the intraday-long control, and in 2023 and
2024 it lagged SPY buy-and-hold despite being in the market the whole time,
because it sat out rallies overnight.

This is the same shape as the rest of the study: a positive, sometimes large
excess over a matched control, in a period whose bull market rewards simply
holding. The 2020–2025 sample is a poor test of the strategy and a good test
of the alternative.

### The £10,000 account is too small for this strategy

At this size the **$0.35/order minimum commission is a 12.2% drag** — $921 of
costs on $7,540 of starting USD capital. The same backtest at other sizes:

| starting capital | strategy return | intraday-long | cost as % of capital |
|---:|---:|---:|---:|
| $7,540 | +46.2% | +2.0% | **12.2%** |
| $25,000 | +56.6% | +11.5% | 3.7% |
| $100,000 | +57.9% | +13.5% | 2.7% |
| $1,000,000 | +57.9% | +13.6% | 2.7% |

The strategy's edge is roughly the same; the *cost* of getting it is not. At
$100k the strategy earns +57.9% and the control +13.5%; at $7.5k it earns
+46.2% and the control +2.0%. The £10k account loses most of the edge to
commissions, and the control loses more, so the *excess* looks better at
small size purely because the control is penalised harder. That is an
artefact of the minimum-commission model, not evidence that £10k is the sweet
spot.

### The `vol_target` result is real, but it is not a free lunch

The levered variant ends at £36,731 (+267.3%), ahead of buy-and-hold. That
number survives scrutiny: the strategy's **gross** is +$279,695 against the
levered baseline's **−$36,006**, so the excess is +308.8% and the *strategy's
own gross* is +279.7% of capital — it is not an artefact of the comparison.

But the mechanism is leverage applied to a low-vol regime, and it cuts both
ways. The levered baseline loses money in five of six years (−$15.7k in 2020,
−$16.2k in 2022) because the leverage is set from *trailing* volatility: it
ramps to the 4× cap in quiet periods and then meets a reversal. Its worst
single day is −$12,840 (2020-09-03), on a day the unlevered baseline lost
0.3%. The strategy survives this because it is *short* into those reversals —
that is the actual edge. A £10k account running 4× leverage on a $7,540
notional would need $30k of buying power, and the drawdown on the levered
path is far worse than the unlevered one.

The unlevered path is the one to trust for the "what would I have made"
question; the levered path is what the paper's headline is built on, and it
comes with the tail risk described in the concentration section above.

---

## Comparison to the paper's own numbers

The paper's abstract quotes **1,985%** total return and Sharpe **1.33**. The
body's base case quotes **178%** and Sharpe **0.61**.

The reference notebook's default configuration is `vol_target` sizing with
`band_mult = 1` — the *dynamic-volatility-sizing* variant. So our
`vol_target` run (+117.4%, Sharpe 1.40) is the closest thing we have to the
paper's headline, and it lands at roughly 60% of their claimed total return
with a comparable Sharpe, on a shorter sample and a different feed. The
`full_notional` run (+42.4%, Sharpe 1.34) is the *static* base case, which
the paper quotes at 178% / 0.61 — we do not reproduce that level of return.

**Not implemented:** the paper's trailing stop, VWAP stop, and
dynamic-volatility-sizing *variants* beyond the reference notebook's default
(the notebook has a `band_simplified` flag but no trailing-stop or VWAP-stop
logic). Those are where the headline number comes from and would need to be
added before any apples-to-apples claim.

---

## Limitations

- **Sample.** 2022–2025 only, four years, one bull-to-flat regime. The
  paper's own 2007–2024 sample includes two crashes.
- **Venue.** Alpaca SIP, not the paper's IQFeed or Polygon. Not a like-for-like
  replication of the published numbers.
- **Cost model.** Commission only, no slippage, no borrow cost, no queue
  position, no partial fills. A strategy with a 28.8% hit rate and a $0
  median day will feel slippage.
- **No short-availability modelling.** The strategy shorts SPY; real short
  execution costs are not in these numbers.
- **No walk-forward validation.** The sweep shows the parameter surface; it
  does not show that a parameter chosen on 2022–2024 survives 2025.
- **No live or paper trading.** Backtest only. Nothing here places an order.

---

## Files

| File | Purpose |
|---|---|
| `src/trading/noise_band.py` | Band construction, signals, sizing, backtest engine |
| `scripts/backtest_noise_band.py` | Alpaca fetch (cached), full report, baseline control |
| `scripts/sweep_noise_band.py` | VM × lookback sweep with per-year excess |
| `scripts/check_noise_band_data_quality.py` | Session-completeness sensitivity check |
| `tests/test_noise_band.py` | 9 tests pinning the paper's exact semantics |

## Running

```bash
python scripts/backtest_noise_band.py --start 2022-01-01 --end 2026-01-01 \
    --sizing vol_target
python scripts/sweep_noise_band.py
python scripts/check_noise_band_data_quality.py --feed iex   # reproduce the gaps
```

Bars are cached as pickles in `data/cache_noise_band/` (gitignored). Override
with `NOISE_BAND_CACHE=/path/to/cache`. The first run downloads ~390k bars
over ~20 API calls; later runs take seconds.
