# Engulfing `bull_overnight_tl10` — Whole-Share vs Fractional Sizing

Simulated on the **live execution path**: market entry at the next
session's open, one-session overnight hold, exit at the next open,
live simple-mean ATR14, live trend filter, live top-20 universe.

Capital base, risk per trade (1%) and the 2x notional cap match the
live executor (`PAPER_CAPITAL_US = $10,000`).

Metrics are built from the **dollar** equity curve, because the
per-trade return stream is independent of position size and cannot
distinguish the two sizing rules.

## Sizing modes

- `frac` — fractional shares (what the published backtest assumes)
- `whole` — round DOWN to whole shares (what the executor does now, so a
  broker-side GTC stop can be placed)
- `whole_capped` — as `whole`, but additionally SKIP a setup when one
  share alone would breach the 2x notional cap

## Results

| Universe | Capital | Cost bps | Sizing | N | Skipped | Sharpe$ | Total P&L $ | Ann Ret% | Max DD% | Avg risk $ | Max notional $ |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all60 | $10000 | 0 | fractional | 7722 | 0 | 0.09 | 201.21 | 0.18 | -4.65 | 4.07 | 232.62 |
| all60 | $10000 | 0 | whole | 7722 | 0 | 0.03 | 78.97 | 0.07 | -5.73 | 4.04 | 1,064.92 |
| all60 | $10000 | 0 | whole (capped) | 7722 | 1287 | 0.03 | 78.97 | 0.07 | -5.73 | 4.04 | 1,064.92 |
| all60 | $10000 | 10 | fractional | 7722 | 0 | -0.59 | -1,260.29 | -1.22 | -13.72 | 3.76 | 217.01 |
| all60 | $10000 | 10 | whole | 7722 | 0 | -0.62 | -1,406.51 | -1.37 | -15.31 | 3.87 | 1,064.92 |
| all60 | $10000 | 10 | whole (capped) | 7722 | 1456 | -0.62 | -1,406.51 | -1.37 | -15.31 | 3.87 | 1,064.92 |
| all60 | $103000 | 0 | fractional | 7722 | 0 | 0.09 | 2,072.47 | 0.18 | -4.65 | 41.97 | 2,395.96 |
| all60 | $103000 | 0 | whole | 7722 | 0 | 0.08 | 1,834.35 | 0.16 | -4.54 | 40.71 | 2,373.50 |
| all60 | $103000 | 0 | whole (capped) | 7722 | 0 | 0.08 | 1,834.35 | 0.16 | -4.54 | 40.71 | 2,373.50 |
| all60 | $103000 | 10 | fractional | 7722 | 0 | -0.59 | -12,980.94 | -1.22 | -13.72 | 38.74 | 2,235.24 |
| all60 | $103000 | 10 | whole | 7722 | 0 | -0.59 | -12,505.54 | -1.17 | -13.19 | 37.57 | 2,222.13 |
| all60 | $103000 | 10 | whole (capped) | 7722 | 0 | -0.59 | -12,505.54 | -1.17 | -13.19 | 37.57 | 2,222.13 |
| all60 | $250000 | 0 | fractional | 7722 | 0 | 0.09 | 5,030.28 | 0.18 | -4.65 | 101.86 | 5,815.45 |
| all60 | $250000 | 0 | whole | 7722 | 0 | 0.09 | 5,016.29 | 0.18 | -4.60 | 100.61 | 5,804.50 |
| all60 | $250000 | 0 | whole (capped) | 7722 | 0 | 0.09 | 5,016.29 | 0.18 | -4.60 | 100.61 | 5,804.50 |
| all60 | $250000 | 10 | fractional | 7722 | 0 | -0.59 | -31,507.13 | -1.22 | -13.72 | 94.03 | 5,425.34 |
| all60 | $250000 | 10 | whole | 7722 | 0 | -0.59 | -31,097.44 | -1.20 | -13.54 | 92.86 | 5,410.39 |
| all60 | $250000 | 10 | whole (capped) | 7722 | 0 | -0.59 | -31,097.44 | -1.20 | -13.54 | 92.86 | 5,410.39 |
| all60 | $50000 | 0 | fractional | 7722 | 0 | 0.09 | 1,006.06 | 0.18 | -4.65 | 20.37 | 1,163.09 |
| all60 | $50000 | 0 | whole | 7722 | 0 | 0.09 | 987.92 | 0.18 | -4.41 | 19.16 | 1,151.50 |
| all60 | $50000 | 0 | whole (capped) | 7722 | 4 | 0.09 | 987.92 | 0.18 | -4.41 | 19.16 | 1,151.50 |
| all60 | $50000 | 10 | fractional | 7722 | 0 | -0.59 | -6,301.43 | -1.22 | -13.72 | 18.81 | 1,085.07 |
| all60 | $50000 | 10 | whole | 7722 | 0 | -0.58 | -5,805.93 | -1.12 | -12.66 | 17.66 | 1,064.92 |
| all60 | $50000 | 10 | whole (capped) | 7722 | 27 | -0.58 | -5,805.93 | -1.12 | -12.66 | 17.66 | 1,064.92 |
| live_top20 | $10000 | 0 | fractional | 2522 | 0 | 0.04 | 38.04 | 0.03 | -2.52 | 4.83 | 228.06 |
| live_top20 | $10000 | 0 | whole | 2522 | 0 | 0.04 | 46.26 | 0.04 | -3.44 | 4.95 | 1,042.83 |
| live_top20 | $10000 | 0 | whole (capped) | 2522 | 542 | 0.04 | 46.26 | 0.04 | -3.44 | 4.95 | 1,042.83 |
| live_top20 | $10000 | 10 | fractional | 2522 | 0 | -0.43 | -456.67 | -0.43 | -6.19 | 4.70 | 219.99 |
| live_top20 | $10000 | 10 | whole | 2522 | 0 | -0.40 | -470.65 | -0.44 | -6.81 | 4.89 | 1,042.83 |
| live_top20 | $10000 | 10 | whole (capped) | 2522 | 563 | -0.40 | -470.65 | -0.44 | -6.81 | 4.89 | 1,042.83 |
| live_top20 | $103000 | 0 | fractional | 2522 | 0 | 0.04 | 391.81 | 0.03 | -2.52 | 49.71 | 2,349.03 |
| live_top20 | $103000 | 0 | whole | 2522 | 0 | 0.02 | 213.69 | 0.02 | -2.45 | 48.13 | 2,326.50 |
| live_top20 | $103000 | 0 | whole (capped) | 2522 | 0 | 0.02 | 213.69 | 0.02 | -2.45 | 48.13 | 2,326.50 |
| live_top20 | $103000 | 10 | fractional | 2522 | 0 | -0.43 | -4,703.65 | -0.43 | -6.19 | 48.41 | 2,265.91 |
| live_top20 | $103000 | 10 | whole | 2522 | 0 | -0.43 | -4,556.28 | -0.41 | -6.00 | 46.85 | 2,261.68 |
| live_top20 | $103000 | 10 | whole (capped) | 2522 | 0 | -0.43 | -4,556.28 | -0.41 | -6.00 | 46.85 | 2,261.68 |
| live_top20 | $250000 | 0 | fractional | 2522 | 0 | 0.04 | 950.99 | 0.03 | -2.52 | 120.65 | 5,701.53 |
| live_top20 | $250000 | 0 | whole | 2522 | 0 | 0.03 | 845.27 | 0.03 | -2.48 | 119.06 | 5,687.00 |
| live_top20 | $250000 | 0 | whole (capped) | 2522 | 0 | 0.03 | 845.27 | 0.03 | -2.48 | 119.06 | 5,687.00 |
| live_top20 | $250000 | 10 | fractional | 2522 | 0 | -0.43 | -11,416.63 | -0.43 | -6.19 | 117.51 | 5,499.79 |
| live_top20 | $250000 | 10 | whole | 2522 | 0 | -0.43 | -11,205.81 | -0.42 | -6.09 | 115.98 | 5,484.57 |
| live_top20 | $250000 | 10 | whole (capped) | 2522 | 0 | -0.43 | -11,205.81 | -0.42 | -6.09 | 115.98 | 5,484.57 |
| live_top20 | $50000 | 0 | fractional | 2522 | 0 | 0.04 | 190.20 | 0.03 | -2.52 | 24.13 | 1,140.31 |
| live_top20 | $50000 | 0 | whole | 2522 | 0 | 0.03 | 136.72 | 0.02 | -2.30 | 22.66 | 1,130.84 |
| live_top20 | $50000 | 0 | whole (capped) | 2522 | 2 | 0.03 | 136.72 | 0.02 | -2.30 | 22.66 | 1,130.84 |
| live_top20 | $50000 | 10 | fractional | 2522 | 0 | -0.43 | -2,283.33 | -0.43 | -6.19 | 23.50 | 1,099.96 |
| live_top20 | $50000 | 10 | whole | 2522 | 0 | -0.40 | -1,980.10 | -0.37 | -5.48 | 22.08 | 1,081.00 |
| live_top20 | $50000 | 10 | whole (capped) | 2522 | 6 | -0.40 | -1,980.10 | -0.37 | -5.48 | 22.08 | 1,081.00 |

## Sizing diagnostics — notional-cap breach by capital base

| Universe | Capital | Risk $ | Max notional $ | Risk-skipped | Notional breaches | Breach % | Max notional seen $ |
|---|---:|---:|---:|---:|---:|---:|---:|
| all60 | $10,000 | 100.00 | 200.00 | 0 | 1317 | 17.1% | 1,064.92 |
| all60 | $50,000 | 500.00 | 1,000.00 | 0 | 4 | 0.1% | 1,128.00 |
| all60 | $103,000 | 1,030.00 | 2,060.00 | 0 | 0 | 0.0% | 2,326.50 |
| all60 | $250,000 | 2,500.00 | 5,000.00 | 0 | 0 | 0.0% | 5,663.50 |
| live_top20 | $10,000 | 100.00 | 200.00 | 0 | 549 | 21.8% | 1,042.83 |
| live_top20 | $50,000 | 500.00 | 1,000.00 | 0 | 2 | 0.1% | 1,128.00 |
| live_top20 | $103,000 | 1,030.00 | 2,060.00 | 0 | 0 | 0.0% | 2,326.50 |
| live_top20 | $250,000 | 2,500.00 | 5,000.00 | 0 | 0 | 0.0% | 5,663.50 |

## Exit rule — live vs published

| Exit rule | N | Sharpe$ | Total P&L $ | Exit reasons |
|---|---:|---:|---:|---|
| live 1-session hold | 2522 | 0.04 | 46.26 | overnight_exit=1875, stop=491, target=156 |
| published unbounded hold | 2516 | 1.50 | 3,107.85 | stop=1400, target=1116 |

The published backtest holds until the stop or target with NO time
limit. The live executor force-closes at the NEXT session's open.
They are different strategies, so the published Sharpe does not
describe live behaviour.
