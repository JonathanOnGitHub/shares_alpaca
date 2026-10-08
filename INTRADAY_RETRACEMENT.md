# Intraday Retracement Trading — Paper Trading System

Intraday mean-reversion strategy based on the **opening drive** (front side) and the **50% retracement level**. Buy when price pulls back 50% of the opening drive; exit at a static stop beyond the session extreme or at a 1× ATR target.

**Strategy is proven on FTSE 100. S&P 500 edge is unproven — paper trade to validate.**

---

## Strategy: CONFIRM_TIGHT

### Rules (FTSE 100, 5-min bars)

| Parameter | Value |
|---|---|
| Lookback | 2 bars (opening drive = first 10 minutes) |
| Min gap | 0.3% (gap must exceed this to qualify) |
| Stop | Session high × 1.002 (short) / Session low × 0.998 (long) |
| Target | Entry ± 1× ATR |
| Confirm filter | True (front-side must align with session direction) |
| Time exit | 78 bars (6.5 hours → close of session) |

**Edge is real**: +16.21 bps avg on H1 2026 out-of-sample data, walk-forward validated, 95% CI [7.3, 8.6] bps.

### Why It Works on FTSE 100

- Opening drive on LSE is ~0.35% ATR wide (~70 bps)
- Static stop at 0.2% beyond session extreme = ~35 bps from entry
- Target = ~65 bps → R:R ≈ 1.9:1
- 48% of trades exit at stop for a small loss (−35 bps avg)
- 33% hit target for +65 bps avg
- Net edge: **+7.98 bps per trade**

### Why It Fails on S&P 500 (Hourly Bars)

- With hourly bars (lowest resolution yfinance provides for 2-year backtests), the opening drive concept is destroyed
- Session H/L ≈ whole day range ≈ 1.0% ATR (~210 bps wide)
- Static stop at 0.2% = ~105 bps from entry → R:R ≈ 0.48:1
- **Zero stop exits** on 15,949 US trades — price reverses before reaching the stop
- Only TARGET (39%) and TIME (60%) exits fire, both negative
- **Cannot fairly test without 5-min bars**, which yfinance won't provide

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│                                                                     │
│  IBKR API (5-sec bars)  ──►  data_feeder_uk.py  ──►  live.jsonl   │
│  Alpaca WebSocket (1-min) ──►  data_feeder_us.py ──►  live.jsonl   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SIGNAL GENERATION                              │
│                                                                     │
│  signal_gen_service.py  ◄── live.jsonl                              │
│    └─► IntradayRetracement (CONFIRM_TIGHT params)                  │
│    └─► signals.json (UK: confirm=True, US: confirm=False)           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                                │
│                                                                     │
│  executor_live.py  ◄── signals.json                                 │
│    └─► AlpacaBackend (US / S&P 500, paper trading, free)            │
│    └─► IBKRBackend  (UK / FTSE 100, paper trading, ~£3-5/trade)    │
│    └─► paper_trades.csv                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Install dependencies

```bash
cd /path/to/shares_alpaca
uv venv
source .venv/bin/activate
uv pip install alpaca-py ib_insync python-dotenv pandas numpy
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys:
ALPACA_API_KEY=PKXXXXXXXX
ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXX
```

### 3. FTSE 100 paper trading (UK)

**Requires**: IBKR account + TWS or IB Gateway running on localhost

```bash
# Terminal 1 — Start IBKR TWS/Gateway on port 7497 (paper)
# Then start the data feeder:
python -m src.trading.data_feeder_uk \
    --output live_ftse100.jsonl \
    --universe majors

# Terminal 2 — Start signal generator:
python -m src.trading.signal_gen_service \
    --market UK \
    --live-file live_ftse100.jsonl \
    --signals-dir signals \
    --poll-seconds 5

# Terminal 3 — Start executor:
python -m src.trading.executor_live \
    --broker ibkr \
    --market UK \
    --signals signals/signals_uk.json \
    --log-file paper_trades_ftse100.csv
```

### 4. S&P 500 paper trading (US)

**Requires**: Alpaca paper trading account (free)

```bash
# Terminal 1 — Start data feeder:
python -m src.trading.data_feeder_us \
    --symbols AAPL,MSFT,GOOGL,AMZN,META,NVDA \
    --output live_us.jsonl

# Terminal 2 — Start signal generator:
python -m src.trading.signal_gen_service \
    --market US \
    --live-file live_us.jsonl \
    --signals-dir signals \
    --poll-seconds 5

# Terminal 3 — Start executor:
python -m src.trading.executor_live \
    --broker alpaca \
    --market US \
    --signals signals/signals_us.json \
    --log-file paper_trades_sp500.csv
```

---

## Key Files

| File | Purpose |
|---|---|
| `src/trading/intraday_retracement.py` | Strategy logic — CONFIRM_TIGHT, bar-resolution agnostic |
| `src/trading/signal_gen_service.py` | Daemon: reads `live.jsonl` → writes `signals.json` |
| `src/trading/executor_live.py` | Daemon: reads `signals.json` → executes via broker |
| `src/trading/data_feeder_uk.py` | IBKR → 5-sec bars for FTSE 100 |
| `src/trading/data_feeder_us.py` | Alpaca WebSocket → 1-min bars for S&P 500 |

---

## Paper Trading Results

### FTSE 100 (CONFIRM_TIGHT, proven)

| Metric | Value |
|---|---|
| Avg pnl | **+7.98 bps** (95% CI [7.3, 8.6]) |
| Win rate | 47.5% |
| Out-of-sample (H1 2026) | **+16.21 bps** avg |
| Exit: STOP | 48.4% (−35 bps avg) |
| Exit: TARGET | 33.2% (+65 bps avg) |
| Exit: TIME | 18.4% (+28 bps avg) |

### S&P 500 (unconfirmed — paper trade to validate)

| Metric | Value |
|---|---|
| Avg pnl | **−67.42 bps** (hourly bars, invalid test) |
| Win rate | 7.1% |
| Status | **UNPROVEN — edge unknown at 5-min resolution** |

**The S&P 500 result cannot be trusted** — hourly bars destroy the opening-drive concept. Paper trade with live 5-min data before drawing conclusions.

---

## Honest Caveats

1. **Execution gap**: backtest assumes entry at the exact 50% retracement price. Live execution may experience slippage, especially in fast markets.
2. **Capital inefficiency**: strategy holds one position per symbol per session. With 20 FTSE symbols, average 1–2 positions active at once → capital utilisation ≈ 10%.
3. **Regime dependence**: edge confirmed on 2025–H1 2026 FTSE 100. May degrade in low-volatility or strong-trending regimes.
4. **Commission**: Alpaca is commission-free (US). IBKR charges ~£3–5/trade for UK stocks. At 2 trades/symbol/week, FTSE paper trading costs ~£200–300/month.
5. **Data latency**: IBKR 5-sec bars may have 1–2 second latency. For HFT-style strategies this matters; for 5-min bars it does not.
6. **S&P 500**: no positive-expectancy variant found at hourly resolution. The edge may exist at 5-min but cannot be tested with current data sources. **Paper trade to find out.**

---

## Backtest Scripts

For reproducing the FTSE and US results:

```bash
# FTSE 100 CONFIRM_TIGHT backtest
cd /home/burley/Personal/Trading/day_trading/
uv run python retracement_backtest.py

# Parameter exploration
uv run python retracement_explore.py

# Walk-forward validation
uv run python retracement_walkforward.py
```
