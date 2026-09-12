# CEO Insider Purchase Signal — Backtest Package

Backtest of a CEO/CFO/CLO/COO/President/General Counsel/CTO insider purchase signal using SEC EDGAR Form 4 data.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy yfinance scipy

# Run all 4 configurations (fetch fresh prices)
python run_4backtests.py

# Full comparative analysis vs SPY
python analyse_all_configs.py
```

## Architecture

```
scrape_edgar_5yr.py     — fetch raw Form 4 transactions from SEC EDGAR
run_4configs.py         — generate signal_history files for 4 scoring configs
score_basket.py         — scoring logic (conviction, recency, size)
walkforward_scorer.py   — walk-forward signal generation
backtest_walkforward.py — original single-config backtest
run_4backtests.py       — batch backtest across all 4 configs (threaded price fetch)
analyse_all_configs.py  — equity curves, SPY comparison, statistical tests
```

## Data Required

Before running, you need `ceo_purchases_edgar_5yr.csv` in the working directory. 
Fetch it first with:

```python
# From scrape_edgar_5yr.py — connects to SEC EDGAR API
# Requires: pip install requests beautifulsoup4
```

## Results

See: `../../reports/CEO_insider_signal_report.md`

**TL;DR**: No statistically significant alpha or spread across any configuration (N=20 quarters, 2021–2026).
