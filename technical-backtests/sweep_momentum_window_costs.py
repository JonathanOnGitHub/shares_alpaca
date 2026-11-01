#!/usr/bin/env python3
"""Sweep the momentum window NET OF COSTS.

Motivation: the 5d window flips ~25 round trips/year/name. At 30bp round trip
that is ~7.8%/yr of drag, which is most of the gross edge. A slower window
trades less and might be better net even with a weaker gross signal.

This directly tests the decision made on 2026-11-01 to align the bot to the
screen's 5-day window — the alignment was right for consistency, but the
window CHOICE deserves its own evidence.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/burley/Documents/shares_alpaca/technical-backtests')
sys.path.insert(0, '/home/burley/Documents')
import backtest_stop as bt

raw = bt.load_data()
tickers = list(bt.ALL_TICKERS)
CACHE_DATA = {}
for tkr in tickers:
    try:
        df = raw[tkr].dropna(how='all')
    except Exception:
        continue
    if len(df) < 300:
        continue
    o = df['Open'].to_numpy(float); c = df['Close'].to_numpy(float)
    if np.isnan(o).any() or np.isnan(c).any():
        continue
    CACHE_DATA[tkr] = (o, c, df.index)

print(f'{len(CACHE_DATA)} tickers loaded\n')
print(f"{'window':>7} {'cost':>7} {'CAGR':>8} {'Sharpe':>8} {'maxDD':>8} "
      f"{'trades/name/yr':>15} {'med CAGR':>9}")
print('-' * 74)

for win in (5, 10, 20, 40):
    bt.MOMENTUM_DAYS = win
    for cost in (0.0, 0.0015):
        bt.COST_PER_SIDE = cost
        port, per, ntr, nyrs = [], [], 0, 0.0
        for tkr, (o, c, idx) in CACHE_DATA.items():
            rets, n_tr, _ = bt.simulate(o, c, None)
            m = bt.metrics(rets)
            if m is None:
                continue
            per.append(m); ntr += n_tr; nyrs += m['years']
            port.append(pd.Series(rets, index=idx, name=tkr))
        s = pd.concat(port, axis=1).mean(axis=1, skipna=True).fillna(0.0)
        pm = bt.metrics(s.to_numpy())
        cagrs = np.array([p['cagr'] for p in per])
        per_name_yr = ntr / nyrs if nyrs else 0
        tag = f'{cost*100:.2f}%'
        print(f"{win:>7} {tag:>7} {pm['cagr']*100:>7.2f}% {pm['sharpe']:>8.3f} "
              f"{pm['maxdd']*100:>7.2f}% {per_name_yr:>15.1f} {np.median(cagrs)*100:>8.2f}%")
