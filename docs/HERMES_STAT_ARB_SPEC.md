# Crypto Statistical-Arbitrage Research System — Specification

> Version 1.0 | Saved for agent context | Do not confuse with live trading

---

## Objective

Build a Python research and backtesting system for statistical arbitrage / pairwise relative-value trading across **BTC, XRP, LINK, SOL**.

**Initial objective: research and backtesting only. No live trades. No real orders.**

The system must determine whether statistically exploitable mean-reverting relationships exist and whether they remain profitable after realistic transaction costs, slippage, and funding costs. **Do not assume pair trading works.** The system must be capable of concluding there is no robust edge.

---

## Core Research Questions

Investigate all 6 pairs:

1. BTC/XRP
2. BTC/LINK
3. BTC/SOL
4. XRP/LINK
5. XRP/SOL
6. LINK/SOL

For every pair determine:

- Pearson correlation
- Spearman correlation
- Rolling correlation (30/60/90-day)
- Engle-Granger cointegration
- Rolling cointegration stability
- Hedge ratio (static + rolling OLS)
- Spread: `log(P_A) - beta * log(P_B)`
- Spread mean and standard deviation
- Z-score
- Half-life of mean reversion
- Stationarity (ADF test on spread)
- Historical maximum z-score
- % of excursions that subsequently mean-revert
- Average time to mean reversion
- Trade frequency
- Gross return
- Net return
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate
- Profit factor
- Average trade
- Exposure
- Turnover

**Do not equate high correlation with a valid pairs-trading relationship.**

---

## Data

### Design Principle
The data layer must allow exchange/data providers to be changed without modifying strategy code.

### Supported Intervals
- 5 minute
- 15 minute
- 1 hour
- 4 hour
- 1 day

All timestamps in **UTC** internally.

### Minimum Fields
`timestamp, open, high, low, close, volume`

### Storage
Preserve raw downloaded data locally in `data/raw/` so experiments are reproducible.

### Look-Ahead Bias Prevention
Clearly tag every DataFrame column with the timestamp it was available. Never use future data in signal generation.

### Documentation
For every dataset record:
- exchange
- trading pair
- quote currency
- timezone
- candle interval
- data start/end dates
- missing observations count/periods
- API limitations

---

## Architecture

```
crypto-stat-arb/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── src/
│   ├── data/           # downloaders, validators, cleaners
│   ├── statistics/     # correlations, cointegration, half-life
│   ├── strategies/     # pair-strategy logic
│   ├── backtest/       # event-driven backtester
│   ├── execution/      # (placeholder — not required for research)
│   ├── risk/           # risk limits, position sizing
│   └── reporting/      # equity curves, perf reports
├── tests/              # unit + integration tests
├── notebooks/          # exploration only — not production
├── reports/            # saved experiment results
└── scripts/            # entry points
```

**Keep research logic strictly separate from execution logic.** The execution layer is not required for the backtesting system to function.

---

## Pair Construction

For each pair A/B compute the log-price spread:

```
spread = log(P_A) - beta * log(P_B)
```

Estimate beta using **rolling OLS** — do not assume beta = 1.

Test these beta lookbacks as research parameters:
- Static (full historical)
- 30-day rolling
- 60-day rolling
- 90-day rolling
- 180-day rolling

**Do not hard-code the optimal lookback.** Treat it as a research parameter evaluated out-of-sample.

---

## Cointegration

Implement at minimum:

1. **Engle-Granger two-step test** — record test statistic, p-value, critical values
2. **Augmented Dickey-Fuller (ADF) test** on the spread
3. **Rolling-window cointegration** — stability through time

For every test record:
- test statistic
- p-value
- critical values (1%, 5%, 10%)
- window size
- timestamp

**A pair is not tradeable merely because one historical test produces p < 0.05.** Investigate stability through time. One lucky test is not a strategy.

---

## Mean-Reversion Model

Spread z-score:

```
z = (spread - rolling_mean) / rolling_std
```

### Entry Thresholds
Test: ±1.0, ±1.5, ±2.0, ±2.5, ±3.0

### Exit Types
- z = 0 (centerline)
- z = ±0.25
- z = ±0.5
- Time-based exit
- Stop-loss

### Walk-Forward Validation
**Do not select parameters because they maximise historical return.** Use walk-forward training/test splits. Parameter selection occurs inside the training period only.

---

## Trade Construction

For pair A/B:

**If z > entry_threshold:**
→ A is relatively expensive vs B
→ **SHORT A, LONG B** (hedge ratio = beta)

**If z < -entry_threshold:**
→ A is relatively cheap vs B
→ **LONG A, SHORT B**

### Hedging Variants
Compare:
1. Equal-dollar hedge
2. OLS hedge ratio
3. Volatility-adjusted hedge
4. Beta-neutral hedge

---

## Costs (Critical)

The backtester **must** model:

- Exchange trading fees
- Bid/ask spread
- Slippage
- Market impact (where estimable)
- Perpetual futures funding costs
- Borrow/short costs
- Latency assumptions

All costs are **configurable**. Run every experiment under at minimum:

| Scenario | Fees | Slippage | Funding | Short Cost |
|----------|------|----------|---------|------------|
| Optimistic | 0.02% | 1 bps | 0.001%/hr | 0.005%/hr |
| Base | 0.04% | 2 bps | 0.003%/hr | 0.01%/hr |
| Pessimistic | 0.06% | 4 bps | 0.006%/hr | 0.02%/hr |

Report **gross and net performance separately.** A strategy profitable only under optimistic costs is **fragile**.

---

## Backtesting Methodology

### Avoid Look-Ahead Bias
Signal generation uses only data available at that point in time.

### Walk-Forward Splits
Test at minimum:

| Train | Test | Roll |
|-------|------|------|
| 180 days | 30 days | 30 days forward |
| 365 days | 90 days | 90 days forward |

### Rule
Parameter selection inside training period only. Report out-of-sample test-period performance.

---

## Regime Analysis

Divide results by:
- High / low volatility periods
- BTC bull / bear markets
- Broad crypto drawdowns
- Strong altcoin periods
- High / low correlation regimes

Calculate rolling:
- Correlation
- Beta
- Half-life
- Sharpe
- Drawdown

**Flag structural breaks.**

---

## Portfolio-Level Model

After pairwise analysis, build a multi-asset factor model:

```
SOL_return = beta_BTC * BTC_return
           + beta_XRP * XRP_return
           + beta_LINK * LINK_return
           + residual
```

Investigate whether the residual is stationary and mean-reverting. Repeat for each asset. Compare multi-factor approach vs simple pair trading.

---

## Signal Quality

For every potential trade calculate:
- Current z-score
- Rolling z-score percentile
- Cointegration status
- Half-life
- Estimated probability of mean reversion
- Expected holding period
- Current volatility
- Estimated transaction costs
- Expected net edge

**The system must output `NO_TRADE` when statistical conditions are insufficient.**

---

## Risk Management

Implement:
- Maximum position size
- Maximum gross exposure
- Maximum net exposure
- Maximum pair exposure
- Maximum portfolio exposure
- Stop-loss
- Maximum holding period
- Volatility scaling
- Drawdown-based risk reduction
- Correlation/concentration limits

**Do not assume two legs make a position risk-free.**

Monitor:
- gross exposure
- net exposure
- BTC beta
- altcoin beta
- pair concentration
- portfolio volatility
- drawdown

---

## Performance Reporting

### Machine-Readable
JSON for every pair experiment:
```
pair, timeframe, train_period, test_period,
cointegration_pvalue, half_life, entry_threshold, exit_threshold,
n_trades, win_rate, avg_trade,
gross_return, fees, slippage, funding, net_return,
annualised_sharpe, sortino, max_drawdown, calmar, turnover,
avg_holding_period
```

### Human-Readable
Charts:
- equity curve
- drawdown curve
- z-score chart
- spread chart
- rolling beta
- rolling correlation
- rolling Sharpe
- trade distribution
- monthly returns
- regime performance

---

## Statistical Robustness

Explicitly test for:
- Multiple-testing bias
- Parameter overfitting
- Survivorship bias
- Look-ahead bias
- Data snooping
- Non-stationarity
- Structural breaks

Where practical use:
- Bootstrap tests
- Permutation tests
- Monte Carlo simulations
- White's Reality Check
- Deflated Sharpe Ratio

**Do not report a high Sharpe without investigating whether it could arise from data mining.**

---

## Benchmarks

Compare against:
1. BTC buy-and-hold
2. Equal-weight BTC/XRP/LINK/SOL (rebalanced monthly)
3. 25/25/25/25 crypto portfolio
4. BTC market-neutral (if applicable)
5. Randomised-entry control strategy

**The goal is to determine whether stat-arb adds value, not just whether it beats a rising market.**

---

## Output Classification

Classify each pair research result as:

| Classification | Criteria |
|----------------|---------|
| **ROBUST** | Out-of-sample Sharpe > 1.0, net of all costs, stable across regimes, survives permutation test, survives all three cost scenarios |
| **PROMISING BUT INCONCLUSIVE** | Directionally correct, statistically significant in-sample, but fragile to costs or regime changes |
| **FRAGILE** | Profitable only under optimistic cost assumptions, or p > 0.10 in permutation test, or unstable across walk-forward windows |
| **NO EVIDENCE OF EDGE** | Not profitable net of base costs, or p > 0.20 in permutation test, or cointegration unstable |

Classification criteria must be explicitly defined in `config/default.yaml`.

---

## Current-Market Analysis Script

A separate script outputs the live state of all 6 pairs:

```
BTC/XRP
  z-score: +1.82
  half-life: 31 hours
  cointegration: PASS
  rolling stability: PASS
  signal: SHORT XRP / LONG BTC
  confidence: ...

BTC/LINK
  z-score: ...
  ...
```

**Do not generate a live trade recommendation** unless all configured statistical and risk conditions are satisfied.

Clearly distinguish:
- `RESEARCH_SIGNAL` — statistical edge detected in backtest
- `PAPER_SIGNAL` — passes live statistical filters, for paper trading only
- `LIVE_SIGNAL` — for live execution (not implemented in v1)

**Default mode: RESEARCH only.**

---

## Reproducibility

Every experiment saves:
- Configuration (YAML)
- Data period and source
- Git commit hash
- Random seed
- All parameters
- Results

A backtest must be reproducible from the saved configuration. Use YAML/TOML — not hard-coded parameters.

---

## Testing

### Unit Tests
- Return calculations
- Log-price calculations
- Hedge-ratio estimation (OLS vs rolling)
- Spread calculation
- Z-score calculation
- Cointegration test wrapper
- Half-life estimation
- Position sizing (all 4 variants)
- Transaction-cost calculations
- P&L attribution
- Funding cost calculations
- Drawdown computation
- Walk-forward train/test splitting
- Look-ahead bias prevention

### Integration Tests
- Complete historical backtest on one pair
- Full walk-forward run with reporting

---

## Code Quality

- Python 3.11+
- pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, pytest
- Type hints on all public functions
- Docstrings on all public functions and classes
- `logging` (not `print`) for application output
- **Notebooks for exploration only — production logic lives in `src/`**

---

## Explicit Constraints

**Do NOT:**
- Assume correlation implies cointegration
- Optimise parameters on the full dataset then report that same dataset
- Use future information in signal generation
- Ignore fees, slippage, or funding
- Assume shorting is free
- Assume relationships remain stationary
- Claim profitability from backtesting alone
- Automatically place live trades

**The system's primary purpose is to discover whether a statistically robust edge exists. If none exists, report that clearly.**

---

## Implementation Order

### Phase 1 — Foundation
1. Project structure + `pyproject.toml` + `requirements.txt`
2. Configuration system (`config/default.yaml`)
3. Data downloader (Binance initially — free API, reliable data)
4. Data validation and preservation to `data/raw/`
5. Pairwise statistics engine (correlation, spread, z-score)

### Phase 2 — Statistical Core
6. Cointegration analysis (Engle-Granger + ADF + rolling)
7. Rolling hedge-ratio calculation (OLS)
8. Half-life estimation
9. Stationarity testing

### Phase 3 — Backtesting
10. Event-driven backtester
11. Transaction-cost model (all three cost scenarios)
12. Walk-forward validation framework
13. Risk management module

### Phase 4 — Reporting
14. Performance report generator (JSON + markdown)
15. Charts (equity curve, drawdown, z-score, spread, rolling stats)
16. Current-market analysis script

### Phase 5 — Research Run
17. Run complete analysis on all 6 pairs across all timeframes
18. Produce research report with explicit classifications
19. Sensitivity analysis (cost scenarios, parameter robustness)

---

## Research Report — Required Output

After the full run, produce a concise report identifying:

- Which pairs appear statistically mean-reverting
- Which relationships are unstable or spurious
- Estimated half-lives
- Historical trading frequency
- Gross vs net profitability
- Out-of-sample performance
- Sensitivity to transaction costs
- Evidence of overfitting
- Whether evidence justifies proceeding to paper trading

**Conclusion must be evidence-based — not assumed profitable.**
