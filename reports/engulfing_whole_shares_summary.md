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
| all60 | $10000 | 0 | fractional | 7861 | 0 | -0.02 | -74.96 | -0.07 | -9.99 | 9.11 | 234.58 |
| all60 | $10000 | 0 | whole | 7861 | 0 | -0.07 | -210.51 | -0.19 | -9.10 | 9.06 | 1,064.92 |
| all60 | $10000 | 0 | whole (capped) | 7861 | 1333 | -0.07 | -210.51 | -0.19 | -9.10 | 9.06 | 1,064.92 |
| all60 | $10000 | 10 | fractional | 7861 | 0 | -0.51 | -1,519.41 | -1.49 | -16.62 | 8.40 | 210.18 |
| all60 | $10000 | 10 | whole | 7861 | 0 | -0.57 | -1,735.77 | -1.72 | -18.87 | 8.69 | 1,064.92 |
| all60 | $10000 | 10 | whole (capped) | 7861 | 1517 | -0.57 | -1,735.77 | -1.72 | -18.87 | 8.69 | 1,064.92 |
| all60 | $103000 | 0 | fractional | 7861 | 0 | -0.02 | -772.08 | -0.07 | -9.99 | 93.82 | 2,416.21 |
| all60 | $103000 | 0 | whole | 7861 | 0 | -0.02 | -753.36 | -0.07 | -9.71 | 91.03 | 2,395.27 |
| all60 | $103000 | 0 | whole (capped) | 7861 | 0 | -0.02 | -753.36 | -0.07 | -9.71 | 91.03 | 2,395.27 |
| all60 | $103000 | 10 | fractional | 7861 | 0 | -0.51 | -15,649.90 | -1.49 | -16.62 | 86.51 | 2,164.89 |
| all60 | $103000 | 10 | whole | 7861 | 0 | -0.50 | -14,875.15 | -1.41 | -15.85 | 84.01 | 2,156.48 |
| all60 | $103000 | 10 | whole (capped) | 7861 | 0 | -0.50 | -14,875.15 | -1.41 | -15.85 | 84.01 | 2,156.48 |
| all60 | $250000 | 0 | fractional | 7861 | 0 | -0.02 | -1,873.99 | -0.07 | -9.99 | 227.72 | 5,864.59 |
| all60 | $250000 | 0 | whole | 7861 | 0 | -0.02 | -1,796.33 | -0.07 | -9.88 | 224.93 | 5,788.57 |
| all60 | $250000 | 0 | whole (capped) | 7861 | 0 | -0.02 | -1,796.33 | -0.07 | -9.88 | 224.93 | 5,788.57 |
| all60 | $250000 | 10 | fractional | 7861 | 0 | -0.51 | -37,985.19 | -1.49 | -16.62 | 209.98 | 5,254.58 |
| all60 | $250000 | 10 | whole | 7861 | 0 | -0.51 | -37,315.29 | -1.46 | -16.34 | 207.33 | 5,225.36 |
| all60 | $250000 | 10 | whole (capped) | 7861 | 0 | -0.51 | -37,315.29 | -1.46 | -16.34 | 207.33 | 5,225.36 |
| all60 | $50000 | 0 | fractional | 7861 | 0 | -0.02 | -374.80 | -0.07 | -9.99 | 45.54 | 1,172.92 |
| all60 | $50000 | 0 | whole | 7861 | 0 | -0.02 | -260.10 | -0.05 | -9.41 | 42.82 | 1,128.00 |
| all60 | $50000 | 0 | whole (capped) | 7861 | 5 | -0.02 | -260.10 | -0.05 | -9.41 | 42.82 | 1,128.00 |
| all60 | $50000 | 10 | fractional | 7861 | 0 | -0.51 | -7,597.04 | -1.49 | -16.62 | 42.00 | 1,050.92 |
| all60 | $50000 | 10 | whole | 7861 | 0 | -0.50 | -6,983.88 | -1.36 | -15.30 | 39.42 | 1,064.92 |
| all60 | $50000 | 10 | whole (capped) | 7861 | 27 | -0.50 | -6,983.88 | -1.36 | -15.30 | 39.42 | 1,064.92 |
| live_top20 | $10000 | 0 | fractional | 2569 | 0 | -0.02 | -26.86 | -0.02 | -4.27 | 10.82 | 232.19 |
| live_top20 | $10000 | 0 | whole | 2569 | 0 | -0.04 | -56.21 | -0.05 | -5.50 | 11.11 | 1,042.83 |
| live_top20 | $10000 | 0 | whole (capped) | 2569 | 555 | -0.04 | -56.21 | -0.05 | -5.50 | 11.11 | 1,042.83 |
| live_top20 | $10000 | 10 | fractional | 2569 | 0 | -0.41 | -526.75 | -0.49 | -7.28 | 10.54 | 221.56 |
| live_top20 | $10000 | 10 | whole | 2569 | 0 | -0.41 | -569.98 | -0.53 | -7.93 | 10.96 | 1,042.83 |
| live_top20 | $10000 | 10 | whole (capped) | 2569 | 580 | -0.41 | -569.98 | -0.53 | -7.93 | 10.96 | 1,042.83 |
| live_top20 | $103000 | 0 | fractional | 2569 | 0 | -0.02 | -276.69 | -0.02 | -4.27 | 111.47 | 2,391.56 |
| live_top20 | $103000 | 0 | whole | 2569 | 0 | -0.03 | -329.47 | -0.03 | -4.10 | 107.91 | 2,326.50 |
| live_top20 | $103000 | 0 | whole (capped) | 2569 | 0 | -0.03 | -329.47 | -0.03 | -4.10 | 107.91 | 2,326.50 |
| live_top20 | $103000 | 10 | fractional | 2569 | 0 | -0.41 | -5,425.55 | -0.49 | -7.28 | 108.51 | 2,282.05 |
| live_top20 | $103000 | 10 | whole | 2569 | 0 | -0.40 | -5,181.93 | -0.47 | -6.98 | 105.04 | 2,261.68 |
| live_top20 | $103000 | 10 | whole (capped) | 2569 | 0 | -0.40 | -5,181.93 | -0.47 | -6.98 | 105.04 | 2,261.68 |
| live_top20 | $250000 | 0 | fractional | 2569 | 0 | -0.02 | -671.58 | -0.02 | -4.27 | 270.56 | 5,804.76 |
| live_top20 | $250000 | 0 | whole | 2569 | 0 | -0.02 | -616.02 | -0.02 | -4.18 | 267.01 | 5,788.57 |
| live_top20 | $250000 | 0 | whole (capped) | 2569 | 0 | -0.02 | -616.02 | -0.02 | -4.18 | 267.01 | 5,788.57 |
| live_top20 | $250000 | 10 | fractional | 2569 | 0 | -0.41 | -13,168.80 | -0.49 | -7.28 | 263.38 | 5,538.96 |
| live_top20 | $250000 | 10 | whole | 2569 | 0 | -0.40 | -12,864.24 | -0.48 | -7.14 | 259.97 | 5,489.17 |
| live_top20 | $250000 | 10 | whole (capped) | 2569 | 0 | -0.40 | -12,864.24 | -0.48 | -7.14 | 259.97 | 5,489.17 |
| live_top20 | $50000 | 0 | fractional | 2569 | 0 | -0.02 | -134.32 | -0.02 | -4.27 | 54.11 | 1,160.95 |
| live_top20 | $50000 | 0 | whole | 2569 | 0 | -0.01 | -69.71 | -0.01 | -3.86 | 50.79 | 1,130.84 |
| live_top20 | $50000 | 0 | whole (capped) | 2569 | 3 | -0.01 | -69.71 | -0.01 | -3.86 | 50.79 | 1,130.84 |
| live_top20 | $50000 | 10 | fractional | 2569 | 0 | -0.41 | -2,633.76 | -0.49 | -7.28 | 52.68 | 1,107.79 |
| live_top20 | $50000 | 10 | whole | 2569 | 0 | -0.38 | -2,270.94 | -0.42 | -6.47 | 49.48 | 1,097.83 |
| live_top20 | $50000 | 10 | whole (capped) | 2569 | 5 | -0.38 | -2,270.94 | -0.42 | -6.47 | 49.48 | 1,097.83 |

## Sizing diagnostics — notional-cap breach by capital base

| Universe | Capital | Risk $ | Max notional $ | Risk-skipped | Notional breaches | Breach % | Max notional seen $ |
|---|---:|---:|---:|---:|---:|---:|---:|
| all60 | $10,000 | 100.00 | 200.00 | 0 | 1339 | 17.0% | 1,064.92 |
| all60 | $50,000 | 500.00 | 1,000.00 | 0 | 4 | 0.1% | 1,128.00 |
| all60 | $103,000 | 1,030.00 | 2,060.00 | 0 | 0 | 0.0% | 2,326.50 |
| all60 | $250,000 | 2,500.00 | 5,000.00 | 0 | 0 | 0.0% | 5,688.77 |
| live_top20 | $10,000 | 100.00 | 200.00 | 0 | 564 | 22.0% | 1,042.83 |
| live_top20 | $50,000 | 500.00 | 1,000.00 | 0 | 2 | 0.1% | 1,128.00 |
| live_top20 | $103,000 | 1,030.00 | 2,060.00 | 0 | 0 | 0.0% | 2,326.50 |
| live_top20 | $250,000 | 2,500.00 | 5,000.00 | 0 | 0 | 0.0% | 5,688.77 |

## Hold period — does a longer hold recover the edge?

Live top-20 universe, $10,000 capital, whole shares. The live rule
is **1 session** (force-close at the next open); `unbounded` is the
published rule (hold until stop or target).

| Hold | N | Sharpe$ @0bps | P&L $ @0bps | Sharpe$ @10bps | P&L $ @10bps | Ann ret% @0 | Max DD% @0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 session | 2569 | -0.04 | -56.21 | -0.41 | -569.98 | -0.05 | -5.50 |
| 2 sessions | 2376 | 0.45 | 797.26 | 0.16 | 289.44 | 0.70 | -4.48 |
| 3 sessions | 2231 | 0.60 | 1,146.95 | 0.34 | 645.53 | 0.99 | -2.99 |
| 5 sessions | 2038 | 0.68 | 1,513.27 | 0.48 | 1,076.93 | 1.29 | -3.79 |
| 10 sessions | 1932 | 1.04 | 2,885.86 | 0.89 | 2,450.17 | 2.35 | -4.76 |
| unbounded | 1454 | 1.61 | 5,619.98 | 1.49 | 5,207.69 | 4.15 | -6.12 |

### Exit reasons

| Hold | Exit reasons |
|---|---|
| 1 session | hold_1_exit=2468, stop=89, target=12 |
| 2 sessions | hold_2_exit=2118, stop=220, target=38 |
| 3 sessions | hold_3_exit=1842, stop=335, target=54 |
| 5 sessions | hold_5_exit=1478, stop=446, target=114 |
| 10 sessions | hold_10_exit=981, stop=673, target=278 |
| unbounded | stop=814, target=640 |

### Robustness — Sharpe$ per calendar period @10bps

| Period | H1 | H2 | H3 | H5 | H10 |
|---|---:|---:|---:|---:|---:|
| 2015-01-07 .. 2017-01-12 | -0.53 | 0.05 | -0.03 | 0.25 | 0.67 |
| 2017-01-18 .. 2019-04-23 | 0.07 | 0.76 | 0.90 | 1.23 | 1.46 |
| 2019-04-24 .. 2021-08-06 | -0.25 | -0.05 | 0.50 | 0.77 | 1.44 |
| 2021-08-16 .. 2023-11-06 | -0.50 | 0.70 | 0.48 | 0.49 | 0.13 |
| 2023-11-22 .. 2025-12-24 | -0.74 | -0.41 | -0.02 | -0.09 | 0.75 |

Note: overlapping setups on the same ticker are dropped, matching the
live executor's "already have an active position" skip — so a longer
hold trades fewer distinct setups, not more.
