# Engulfing vs Buy & Hold — like-for-like portfolio comparison

- Universe: live top-20 (TSLA, NVDA, AMD, AAPL, META, MSFT, AMZN, GOOGL, NFLX, INTC, LLY, JPM, UNH, COST, V, CRM, WMT, BAC, QCOM, GS)
- Capital: $10,000 (PAPER_CAPITAL_US)
- Costs: 10bps round trip

## Headline

| | Strategy (reoptimised) | Buy & Hold |
|---|---:|---:|
| Sharpe | 0.34 | 1.19 |
| Max drawdown | -3.29% | -51.93% |
| Ann. return | 0.57% | 36.98% |
| Total return | 6.46% | 3078.02% |

## Capital utilisation (why the dollar rows differ)

| | Strategy | Buy & Hold |
|---|---:|---:|
| Avg notional deployed | $214 | $10,000 |
| % of capital at risk | 2.1% | 100% |

The strategy risks 1% of capital per trade with a 2x notional cap,
so it holds a fraction of the account while Buy & Hold holds all of
it. The dollar-return rows are therefore NOT comparable; Sharpe and
drawdown are.

## Composite — strategy + Buy & Hold on the idle cash

The fairest comparison: the strategy keeps its P&L and the capital
it is NOT using earns the same equal-weight B&H return.

| | Ann. return | Sharpe | MaxDD |
|---|---:|---:|---:|
| Buy & Hold (100% deployed) | 36.98% | 1.19 | -51.93% |
| Composite: baseline (stop1.0, hold1) | 34.85% | 1.16 | -51.98% |
| Composite: reoptimised (stop2.0, hold3) | 34.09% | 1.19 | -51.28% |

## Strategy detail

| Config | Trades | P&L | Sharpe | MaxDD | Win% | Avg bps | Util% |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (stop1.0, hold1) | 2470 | $-469.08 | -0.42 | -6.81% | 46.9% | -9.2 | 2.1% |
| reoptimised (stop2.0, hold3) | 2231 | $645.53 | 0.34 | -3.29% | 53.2% | +18.8 | 2.1% |

## Per-period Sharpe

| Period | B&H | baseline (stop1.0, hold1) | reoptimised (stop2.0, hold3) |
|---|---:|---:|---:|
| 2015-01-07 .. 2017-01-12 | 1.68 | -0.71 | -0.03 |
| 2017-01-18 .. 2019-04-23 | 1.30 | -0.14 | 0.90 |
| 2019-04-24 .. 2021-08-06 | 1.04 | -0.51 | 0.50 |
| 2021-08-16 .. 2023-11-06 | 0.39 | -0.31 | 0.48 |
| 2023-11-22 .. 2025-12-24 | 0.54 | -0.53 | -0.02 |

## Caveats

- The universe is the live top-20 by ADV, a STATIC cached list, so it
  carries survivorship and look-ahead bias: NVDA alone contributes
  12x of the 32x average multiple, and B&H annual return falls from
  ~37% to ~27% with NVDA removed.
- B&H is equal-weight buy-and-hold with no rebalancing, bought at the
  first open and sold at the last close.
- The strategy's capital is 98% idle by construction (1% risk, 2x
  notional cap), so it is compared on a composite with the idle cash
  in B&H, not on dollar return.
