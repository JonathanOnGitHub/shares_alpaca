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
| all60 | $10000 | 0 | fractional | 7582 | 0 | 0.07 | 159.09 | 0.14 | -4.65 | 4.07 | 231.73 |
| all60 | $10000 | 0 | whole | 7582 | 0 | 0.01 | 31.36 | 0.03 | -5.66 | 4.04 | 1,064.92 |
| all60 | $10000 | 0 | whole (capped) | 7582 | 1284 | 0.01 | 31.36 | 0.03 | -5.66 | 4.04 | 1,064.92 |
| all60 | $10000 | 10 | fractional | 7582 | 0 | -0.62 | -1,271.95 | -1.23 | -13.83 | 3.76 | 216.46 |
| all60 | $10000 | 10 | whole | 7582 | 0 | -0.65 | -1,413.37 | -1.38 | -15.38 | 3.88 | 1,064.92 |
| all60 | $10000 | 10 | whole (capped) | 7582 | 1445 | -0.65 | -1,413.37 | -1.38 | -15.38 | 3.88 | 1,064.92 |
| all60 | $103000 | 0 | fractional | 7582 | 0 | 0.07 | 1,638.67 | 0.14 | -4.65 | 41.90 | 2,386.84 |
| all60 | $103000 | 0 | whole | 7582 | 0 | 0.07 | 1,441.32 | 0.13 | -4.55 | 40.63 | 2,373.50 |
| all60 | $103000 | 0 | whole (capped) | 7582 | 0 | 0.07 | 1,441.32 | 0.13 | -4.55 | 40.63 | 2,373.50 |
| all60 | $103000 | 10 | fractional | 7582 | 0 | -0.62 | -13,101.04 | -1.23 | -13.83 | 38.73 | 2,229.51 |
| all60 | $103000 | 10 | whole | 7582 | 0 | -0.62 | -12,618.73 | -1.18 | -13.29 | 37.56 | 2,222.13 |
| all60 | $103000 | 10 | whole (capped) | 7582 | 0 | -0.62 | -12,618.73 | -1.18 | -13.29 | 37.56 | 2,222.13 |
| all60 | $250000 | 0 | fractional | 7582 | 0 | 0.07 | 3,977.36 | 0.14 | -4.65 | 101.69 | 5,793.29 |
| all60 | $250000 | 0 | whole | 7582 | 0 | 0.07 | 3,998.54 | 0.14 | -4.61 | 100.42 | 5,781.00 |
| all60 | $250000 | 0 | whole (capped) | 7582 | 0 | 0.07 | 3,998.54 | 0.14 | -4.61 | 100.42 | 5,781.00 |
| all60 | $250000 | 10 | fractional | 7582 | 0 | -0.62 | -31,798.64 | -1.23 | -13.83 | 94.01 | 5,411.44 |
| all60 | $250000 | 10 | whole | 7582 | 0 | -0.62 | -31,377.23 | -1.21 | -13.65 | 92.83 | 5,410.39 |
| all60 | $250000 | 10 | whole (capped) | 7582 | 0 | -0.62 | -31,377.23 | -1.21 | -13.65 | 92.83 | 5,410.39 |
| all60 | $50000 | 0 | fractional | 7582 | 0 | 0.07 | 795.47 | 0.14 | -4.65 | 20.34 | 1,158.66 |
| all60 | $50000 | 0 | whole | 7582 | 0 | 0.08 | 782.54 | 0.14 | -4.44 | 19.11 | 1,151.50 |
| all60 | $50000 | 0 | whole (capped) | 7582 | 4 | 0.08 | 782.54 | 0.14 | -4.44 | 19.11 | 1,151.50 |
| all60 | $50000 | 10 | fractional | 7582 | 0 | -0.62 | -6,359.73 | -1.23 | -13.83 | 18.80 | 1,082.29 |
| all60 | $50000 | 10 | whole | 7582 | 0 | -0.61 | -5,872.14 | -1.13 | -12.80 | 17.65 | 1,064.92 |
| all60 | $50000 | 10 | whole (capped) | 7582 | 27 | -0.61 | -5,872.14 | -1.13 | -12.80 | 17.65 | 1,064.92 |
| live_top20 | $10000 | 0 | fractional | 2470 | 0 | 0.03 | 26.87 | 0.02 | -2.45 | 4.83 | 227.81 |
| live_top20 | $10000 | 0 | whole | 2470 | 0 | 0.03 | 38.50 | 0.04 | -3.37 | 4.97 | 1,042.83 |
| live_top20 | $10000 | 0 | whole (capped) | 2470 | 539 | 0.03 | 38.50 | 0.04 | -3.37 | 4.97 | 1,042.83 |
| live_top20 | $10000 | 10 | fractional | 2470 | 0 | -0.45 | -457.36 | -0.43 | -6.22 | 4.71 | 219.94 |
| live_top20 | $10000 | 10 | whole | 2470 | 0 | -0.42 | -469.08 | -0.44 | -6.81 | 4.91 | 1,042.83 |
| live_top20 | $10000 | 10 | whole (capped) | 2470 | 561 | -0.42 | -469.08 | -0.44 | -6.81 | 4.91 | 1,042.83 |
| live_top20 | $103000 | 0 | fractional | 2470 | 0 | 0.03 | 276.75 | 0.02 | -2.45 | 49.74 | 2,346.42 |
| live_top20 | $103000 | 0 | whole | 2470 | 0 | 0.01 | 93.47 | 0.01 | -2.38 | 48.15 | 2,326.50 |
| live_top20 | $103000 | 0 | whole (capped) | 2470 | 0 | 0.01 | 93.47 | 0.01 | -2.38 | 48.15 | 2,326.50 |
| live_top20 | $103000 | 10 | fractional | 2470 | 0 | -0.45 | -4,710.81 | -0.43 | -6.22 | 48.47 | 2,265.36 |
| live_top20 | $103000 | 10 | whole | 2470 | 0 | -0.45 | -4,578.84 | -0.41 | -6.04 | 46.90 | 2,261.68 |
| live_top20 | $103000 | 10 | whole (capped) | 2470 | 0 | -0.45 | -4,578.84 | -0.41 | -6.04 | 46.90 | 2,261.68 |
| live_top20 | $250000 | 0 | fractional | 2470 | 0 | 0.03 | 671.73 | 0.02 | -2.45 | 120.73 | 5,695.18 |
| live_top20 | $250000 | 0 | whole | 2470 | 0 | 0.02 | 609.30 | 0.02 | -2.41 | 119.13 | 5,687.00 |
| live_top20 | $250000 | 0 | whole (capped) | 2470 | 0 | 0.02 | 609.30 | 0.02 | -2.41 | 119.13 | 5,687.00 |
| live_top20 | $250000 | 10 | fractional | 2470 | 0 | -0.45 | -11,434.01 | -0.43 | -6.22 | 117.65 | 5,498.44 |
| live_top20 | $250000 | 10 | whole | 2470 | 0 | -0.44 | -11,209.53 | -0.42 | -6.11 | 116.12 | 5,484.57 |
| live_top20 | $250000 | 10 | whole (capped) | 2470 | 0 | -0.44 | -11,209.53 | -0.42 | -6.11 | 116.12 | 5,484.57 |
| live_top20 | $50000 | 0 | fractional | 2470 | 0 | 0.03 | 134.35 | 0.02 | -2.45 | 24.15 | 1,139.04 |
| live_top20 | $50000 | 0 | whole | 2470 | 0 | 0.02 | 106.65 | 0.02 | -2.18 | 22.65 | 1,128.00 |
| live_top20 | $50000 | 0 | whole (capped) | 2470 | 2 | 0.02 | 106.65 | 0.02 | -2.18 | 22.65 | 1,128.00 |
| live_top20 | $50000 | 10 | fractional | 2470 | 0 | -0.45 | -2,286.80 | -0.43 | -6.22 | 23.53 | 1,099.69 |
| live_top20 | $50000 | 10 | whole | 2470 | 0 | -0.41 | -1,982.94 | -0.37 | -5.51 | 22.09 | 1,081.00 |
| live_top20 | $50000 | 10 | whole (capped) | 2470 | 6 | -0.41 | -1,982.94 | -0.37 | -5.51 | 22.09 | 1,081.00 |

## Sizing diagnostics — notional-cap breach by capital base

| Universe | Capital | Risk $ | Max notional $ | Risk-skipped | Notional breaches | Breach % | Max notional seen $ |
|---|---:|---:|---:|---:|---:|---:|---:|
| all60 | $10,000 | 100.00 | 200.00 | 0 | 1304 | 17.2% | 1,064.92 |
| all60 | $50,000 | 500.00 | 1,000.00 | 0 | 4 | 0.1% | 1,128.00 |
| all60 | $103,000 | 1,030.00 | 2,060.00 | 0 | 0 | 0.0% | 2,326.50 |
| all60 | $250,000 | 2,500.00 | 5,000.00 | 0 | 0 | 0.0% | 5,663.50 |
| live_top20 | $10,000 | 100.00 | 200.00 | 0 | 546 | 22.1% | 1,042.83 |
| live_top20 | $50,000 | 500.00 | 1,000.00 | 0 | 2 | 0.1% | 1,128.00 |
| live_top20 | $103,000 | 1,030.00 | 2,060.00 | 0 | 0 | 0.0% | 2,326.50 |
| live_top20 | $250,000 | 2,500.00 | 5,000.00 | 0 | 0 | 0.0% | 5,663.50 |

## Hold period — does a longer hold recover the edge?

Live top-20 universe, $10,000 capital, whole shares. The live rule
is **1 session** (force-close at the next open); `unbounded` is the
published rule (hold until stop or target).

| Hold | N | Sharpe$ @0bps | P&L $ @0bps | Sharpe$ @10bps | P&L $ @10bps | Ann ret% @0 | Max DD% @0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 session | 2470 | 0.03 | 38.50 | -0.42 | -469.08 | 0.04 | -3.37 |
| 2 sessions | 2350 | 0.22 | 307.60 | -0.13 | -181.91 | 0.28 | -3.27 |
| 3 sessions | 2287 | 0.38 | 554.76 | 0.05 | 70.15 | 0.49 | -2.80 |
| 5 sessions | 2211 | 0.27 | 424.39 | -0.03 | -46.03 | 0.38 | -3.61 |
| 10 sessions | 2188 | 0.61 | 1,061.48 | 0.35 | 597.73 | 0.92 | -3.27 |
| unbounded | 2148 | 0.68 | 1,196.50 | 0.40 | 706.84 | 1.04 | -3.57 |

### Exit reasons

| Hold | Exit reasons |
|---|---|
| 1 session | hold_1_exit=1833, stop=482, target=155 |
| 2 sessions | hold_2_exit=1256, stop=791, target=303 |
| 3 sessions | hold_3_exit=915, stop=925, target=447 |
| 5 sessions | hold_5_exit=521, stop=1080, target=610 |
| 10 sessions | hold_10_exit=186, stop=1198, target=804 |
| unbounded | stop=1256, target=892 |

### Robustness — Sharpe$ per calendar period @10bps

| Period | H1 | H2 | H3 | H5 | H10 |
|---|---:|---:|---:|---:|---:|
| 2015-01-07 .. 2017-01-26 | -0.60 | -0.07 | -0.13 | -0.24 | 0.21 |
| 2017-01-30 .. 2019-04-17 | -0.24 | 0.25 | 0.08 | 0.47 | 0.96 |
| 2019-04-23 .. 2021-08-16 | -0.48 | -0.49 | 0.06 | -0.28 | 0.43 |
| 2021-08-19 .. 2023-12-11 | -0.36 | 0.45 | 0.42 | 0.10 | 0.18 |
| 2023-12-13 .. 2025-12-24 | -0.51 | -0.74 | -0.23 | -0.21 | 0.16 |

Note: overlapping setups on the same ticker are dropped, matching the
live executor's "already have an active position" skip — so a longer
hold trades fewer distinct setups, not more.
