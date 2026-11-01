#!/usr/bin/env python3
"""Diagnostic: separate gross edge from cost drag, and check baseline sanity."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/burley/Documents/shares_alpaca/technical-backtests')
sys.path.insert(0, '/home/burley/Documents')
import backtest_stop as bt

raw = bt.load_data()
tickers = list(bt.ALL_TICKERS)

for cost in (0.0, 0.0015):
    bt.COST_PER_SIDE = cost
    for lvl in (None, 0.15):
        port, per, ntr = [], [], 0
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
            rets, n_tr, _ = bt.simulate(o, c, lvl)
            m = bt.metrics(rets)
            if m is None:
                continue
            per.append(m); ntr += n_tr
            port.append(pd.Series(rets, index=df.index, name=tkr))
        s = pd.concat(port, axis=1).mean(axis=1, skipna=True).fillna(0.0)
        pm = bt.metrics(s.to_numpy())
        cagrs = np.array([p['cagr'] for p in per])
        sharpes = np.array([p['sharpe'] for p in per])
        tag = 'no stop' if lvl is None else '15% stop'
        print(f"cost {cost*100:>4.2f}%/side | {tag:<8} | portfolio CAGR {pm['cagr']*100:>6.2f}% "
              f"Sharpe {pm['sharpe']:>6.3f} maxDD {pm['maxdd']*100:>6.1f}% | "
              f"per-name median CAGR {np.median(cagrs)*100:>6.2f}% "
              f"median Sharpe {np.median(sharpes):>6.3f} | trades {ntr}")
