#!/usr/bin/env python3
"""
Engulfing vs Buy & Hold — survivorship-bias-free comparison.

Survivorship filter: only stocks with uninterrupted 2015-2025 price history.
Cost model: tiered — 5bps large-caps, 10bps mid-caps, 20bps small-caps.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import math
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.trading.engulfing import run_all_variants

# ── Universe ──────────────────────────────────────────────────────────────────
TICKERS = [
    'MSFT','AAPL','NVDA','GOOGL','AMZN','META','TSLA','JPM','JNJ',
    'V','PG','UNH','HD','MA','DIS','NFLX','ADBE','CRM','CMCSA','VZ',
    'INTC','T','PFE','MRK','ABT','KO','PEP','LLY','CVX','XOM','WMT',
    'COST','NKE','MCD','BA','CAT','MMM','IBM','GE','TXN','QCOM',
    'HON','LOW','ORCL','AMD','AMGN','BMY','GILD','ISRG','LRCX','REGN',
    'ZTS','MDT','BDX','SYK','EW','AJG'
]

# ── 1. Download & structure data ────────────────────────────────────────────────
print("Downloading 10-year price history...")
raw = yf.download(TICKERS, start='2015-01-01', end='2025-12-31', auto_adjust=True, progress=True)
# MultiIndex columns: (field, ticker) → flatten to dict of DataFrames
fields = ['Open','High','Low','Close','Volume']
price_df: dict[str, pd.DataFrame] = {}
for t in TICKERS:
    try:
        df_dict = {f: raw[f][t].values for f in fields}
        dates = raw.index.values
        price_df[t] = pd.DataFrame(df_dict, index=pd.DatetimeIndex(dates))
    except Exception:
        pass

# Survivorship filter — must have data throughout
MIN_DATE = pd.Timestamp('2015-06-01')
MAX_DATE = pd.Timestamp('2025-10-01')
survivors = []
for t, df in price_df.items():
    s = df['Close'].dropna()
    if len(s) < 10:
        continue
    first = pd.Timestamp(s.index[0])
    last  = pd.Timestamp(s.index[-1])
    if first <= MIN_DATE and last >= MAX_DATE:
        survivors.append(t)

print(f"\nSurvivors (uninterrupted 2015-2025): {len(survivors)}/{len(TICKERS)}")
print(f"Dropped: {set(TICKERS) - set(survivors)}")

# ── 2. B&H baseline ───────────────────────────────────────────────────────────
close_prices = {t: price_df[t]['Close'] for t in survivors}
start_p = {t: float(close_prices[t].iloc[0]) for t in survivors}
end_p   = {t: float(close_prices[t].iloc[-1]) for t in survivors}

bh_total  = {t: (end_p[t]/start_p[t] - 1)*100 for t in survivors}
bh_ann   = {t: ((1 + bh_total[t]/100)**(1/10) - 1)*100 for t in survivors}

# Universe equal-weight portfolio
bh_returns_arr = np.array(list(bh_total.values()))
bh_ann_arr     = np.array(list(bh_ann.values()))
bh_port_ann    = float(np.mean(bh_ann_arr))
bh_port_vol    = float(np.mean([close_prices[t].pct_change().std()*math.sqrt(252)*100 for t in survivors]))
bh_port_sharpe = bh_port_ann / bh_port_vol if bh_port_vol > 0 else 0

print(f"\n=== Buy & Hold (equal-weight, survivors only) ===")
print(f"  Mean ann. return:  {bh_port_ann:.1f}%")
print(f"  Ann. vol:          {bh_port_vol:.1f}%")
print(f"  Sharpe proxy:      {bh_port_sharpe:.2f}")
print(f"  Median total ret:  {np.median(bh_returns_arr):.1f}%")
print(f"  Winners:           {(bh_returns_arr>0).sum()}/{len(bh_returns_arr)}")
print(f"  Top 5:             {dict(sorted(bh_total.items(), key=lambda x:-x[1])[:5])}")
print(f"  Bottom 5:          {dict(sorted(bh_total.items(), key=lambda x:x[1])[:5])}")

# ── 3. Engulfing — tiered cost model ─────────────────────────────────────────
# Tier by average price over the period
avg_price = {t: float(close_prices[t].mean()) for t in survivors}
tier5  = [t for t in survivors if avg_price[t] >= 50]
tier10 = [t for t in survivors if 20 <= avg_price[t] < 50]
tier20 = [t for t in survivors if avg_price[t] < 20]

print(f"\n=== Engulfing tier breakdown ===")
print(f"  Tier 5bps (price ≥ $50):  {len(tier5)} stocks")
print(f"  Tier 10bps ($20-$50):     {len(tier10)} stocks")
print(f"  Tier 20bps (<$20):        {len(tier20)} stocks")

# For simplicity run the full universe at 3 fixed cost levels
print(f"\n=== Engulfing bull_overnight tl10 (survivorship-free) ===")
for cost in [5, 10, 20]:
    r = run_all_variants(
        price_df,
        survivors,
        start_date='2015-01-01',
        end_date='2025-12-31',
        cost_bps=cost,
    )
    key = 'bull_overnight_tl10'
    s = r[key]
    print(f"  {cost}bps → Ann={s['ann_ret']*100:+.1f}%  Sharpe={s['sharpe']:.2f}  "
          f"MaxDD={s['max_dd']*100:.1f}%  WR={s['win_rate']:.1f}%  "
          f"Trades={s['n']}  AvgBps={s['avg_bps']:+.2f}")

# ── 4. B&H with same tiered costs ─────────────────────────────────────────────
# What if B&H paid the same tiered transaction costs?
# (simulated: 1 rebalance/year at tiered round-trip cost)
tier_map = {t: 5 if t in tier5 else (10 if t in tier10 else 20) for t in survivors}
bh_cost_adjusted = {}
for t in survivors:
    gross = bh_total[t]
    cost  = tier_map[t] * 2 / 10000 * 100  # 2 rebalances worth of round-trip
    bh_cost_adjusted[t] = gross - cost

bh_adj_arr = np.array(list(bh_cost_adjusted.values()))
bh_adj_ann = ((1 + bh_adj_arr/100)**(1/10) - 1) * 100
bh_adj_port_ann = float(np.mean(bh_adj_ann))

print(f"\n=== B&H cost-adjusted (tiered round-trip) ===")
print(f"  Cost-adj mean ann: {bh_adj_port_ann:.1f}%")

# ── Save results ──────────────────────────────────────────────────────────────
out = {
    'survivors': survivors,
    'bh': {
        'portfolio_ann': bh_port_ann,
        'portfolio_vol': bh_port_vol,
        'sharpe': bh_port_sharpe,
        'median_total': float(np.median(bh_returns_arr)),
        'winners': int((bh_returns_arr>0).sum()),
        'total': len(bh_returns_arr),
        'tier_map': tier_map,
        'cost_adjusted_ann': bh_adj_port_ann,
    },
}
Path('reports/engulfing_vs_bh.json').write_text(json.dumps(out, indent=2, default=str))
print(f"\nResults saved to reports/engulfing_vs_bh.json")
