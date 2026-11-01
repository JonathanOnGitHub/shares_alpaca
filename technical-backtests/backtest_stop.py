#!/usr/bin/env python3
"""
Stop-loss impact backtest — Experiment 13
==========================================
Question: does a trailing stop improve the 5-day momentum on/off strategy, or
does it cut winners and leak fees?

This tests THE STRATEGY THE BOT ACTUALLY RUNS (t212_bot.py as of 2026-11-01):
  * signal: 5-day simple momentum > 0  => in market, else cash
  * decision at the CLOSE of day t, execution at the OPEN of day t+1
    (next-open execution — no soft look-ahead)
  * one name, all-or-nothing, no leverage
  * costs 0.15% per side (30bp round trip, ~the T212 US-share figure)

Stop variants: none (baseline), 10%, 15%, 20%, 25%, 30% TRAILING from the peak
close since entry — identical to evaluate_stop_loss() in t212_bot.py, including
the re-arm rule: after a stop-out the name stays flat until momentum itself
resets to cash. Without that rule a stop immediately re-buys and the result is
meaningless.

Deliberately long window (2017+) so the sample contains 2018Q4, the 2020 crash
and the 2022 bear — a stop's whole purpose is crash protection, and a 3-year
window starting 2023 contains none of that.

Known bias: the universe is today's ticker list, so delisted/bankrupt names are
missing (survivorship). This makes the no-stop baseline look better than it
really was, i.e. it is CONSERVATIVE for the stop. Stated, not corrected.

Run:  /home/burley/.hermes/hermes-agent/venv/bin/python3 backtest_stop.py
"""
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, '/home/burley/Documents')
from t212_quarterly_review import ALL_TICKERS

START = '2017-01-01'
END = pd.Timestamp.now().strftime('%Y-%m-%d')
MOMENTUM_DAYS = 5
COST_PER_SIDE = 0.0015
STOP_LEVELS = [None, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
RF_DAILY = 0.05 / 252
CACHE = os.environ.get('BACKTEST_CACHE', '/tmp/stop_bt_cache.pkl')


def load_data():
    if os.path.exists(CACHE):
        print(f'Loading cached prices: {CACHE}')
        return pd.read_pickle(CACHE)
    print(f'Downloading {len(ALL_TICKERS)} tickers {START}..{END} ...')
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True,
                      progress=False, group_by='ticker', threads=True)
    print('Downloaded. Caching.')
    pd.to_pickle(raw, CACHE)
    return raw


def simulate(opens, closes, stop_pct):
    """One ticker, one stop level. Returns (daily_returns, n_trades, stop_events).

    State machine, all decisions at close t applied at open t+1:
      momentum > 0 & flat & not blocked  -> enter
      momentum <= 0 & holding            -> exit (and clear stopped_out)
      holding & close <= peak*(1-stop)   -> exit, set stopped_out
      stopped_out                        -> stay flat until momentum <= 0
    """
    n = len(closes)
    rets = np.zeros(n)
    holding = False
    entry_px = 0.0
    peak = 0.0
    stopped_out = False
    n_trades = 0
    stop_events = []
    pending = None          # target position to apply at next open

    # momentum at close t
    mom = np.full(n, np.nan)
    if n > MOMENTUM_DAYS:
        mom[MOMENTUM_DAYS:] = closes[MOMENTUM_DAYS:] / closes[:-MOMENTUM_DAYS] - 1.0

    for t in range(MOMENTUM_DAYS, n - 1):
        # ---- apply yesterday's decision at today's open ----
        if pending is not None and pending != holding:
            if pending:                      # buy
                holding = True
                entry_px = opens[t]
                peak = closes[t]
                rets[t] -= COST_PER_SIDE
                n_trades += 1
            else:                            # sell
                holding = False
                rets[t] -= COST_PER_SIDE
                n_trades += 1
            pending = None

        # ---- earn today's open->next-open return ----
        if holding:
            rets[t] += (opens[t + 1] / opens[t]) - 1.0
            # peak ratchets on close while held
            if closes[t] > peak:
                peak = closes[t]

        # ---- decide at today's close, execute next open ----
        m = mom[t]
        if np.isnan(m):
            continue

        if holding:
            # stop first — it overrides the signal
            if stop_pct is not None and peak > 0 and \
                    closes[t] <= peak * (1.0 - stop_pct):
                pending = False
                stopped_out = True
                stop_events.append({'date': t, 'exit_px': opens[t + 1],
                                    'peak': peak, 'close': closes[t]})
                continue
            if m <= 0:
                pending = False
                stopped_out = False       # re-arm on signal reset
        else:
            if stopped_out:
                if m <= 0:
                    stopped_out = False   # re-armed, now wait for bullish
            elif m > 0:
                pending = True

    return rets, n_trades, stop_events


def metrics(rets):
    r = rets[rets != 0.0] if False else rets
    eq = np.cumprod(1.0 + r)
    if len(eq) == 0 or eq[-1] <= 0:
        return None
    years = len(r) / 252.0
    cagr = eq[-1] ** (1.0 / years) - 1.0 if years > 0 else 0.0
    vol = r.std() * np.sqrt(252)
    excess = r - RF_DAILY
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0.0
    dd = (eq / np.maximum.accumulate(eq)) - 1.0
    return {'cagr': cagr, 'vol': vol, 'sharpe': sharpe, 'maxdd': dd.min(),
            'total': eq[-1] - 1.0, 'years': years}


def main():
    raw = load_data()
    tickers = [t for t in ALL_TICKERS]
    per_name = {lvl: [] for lvl in STOP_LEVELS}
    port = {lvl: [] for lvl in STOP_LEVELS}
    trades = {lvl: 0 for lvl in STOP_LEVELS}
    all_stops = {lvl: [] for lvl in STOP_LEVELS if lvl is not None}
    used = 0

    for i, tkr in enumerate(tickers, 1):
        try:
            df = raw[tkr].dropna(how='all')
        except Exception:
            continue
        if len(df) < 300:
            continue
        o = df['Open'].to_numpy(dtype=float)
        c = df['Close'].to_numpy(dtype=float)
        if np.isnan(o).any() or np.isnan(c).any():
            continue
        idx = df.index
        used += 1
        for lvl in STOP_LEVELS:
            rets, ntr, ev = simulate(o, c, lvl)
            m = metrics(rets)
            if m is None:
                continue
            per_name[lvl].append(m)
            trades[lvl] += ntr
            port[lvl].append(pd.Series(rets, index=idx, name=tkr))
            if lvl is not None:
                for e in ev:
                    j = e['date'] + 20
                    fwd = (c[j] / e['exit_px'] - 1.0) if j < len(c) else np.nan
                    all_stops[lvl].append({'ticker': tkr, 'fwd20': fwd,
                                           'gave_back': e['close'] / e['peak'] - 1.0})
        if i % 40 == 0:
            print(f'  {i}/{len(tickers)} tickers…')

    print(f'\nSimulated {used} tickers, {START} to {END}\n')

    # ---- portfolio-level (equal weight across names, daily) ----
    print('=' * 88)
    print('PORTFOLIO — equal weight across all names, daily rebalance')
    print('=' * 88)
    hdr = f"{'stop':>6} {'CAGR':>8} {'vol':>7} {'Sharpe':>8} {'maxDD':>8} {'trades':>8}"
    print(hdr)
    print('-' * 88)
    port_metrics = {}
    for lvl in STOP_LEVELS:
        s = pd.concat(port[lvl], axis=1).mean(axis=1, skipna=True).fillna(0.0)
        m = metrics(s.to_numpy())
        port_metrics[lvl] = m
        label = 'none' if lvl is None else f'{lvl*100:.0f}%'
        print(f"{label:>6} {m['cagr']*100:>7.2f}% {m['vol']*100:>6.2f}% "
              f"{m['sharpe']:>8.3f} {m['maxdd']*100:>7.2f}% {trades[lvl]:>8}")

    # ---- paired per-name comparison ----
    print('\n' + '=' * 88)
    print('PAIRED PER-NAME vs NO-STOP — the statistically meaningful comparison')
    print('=' * 88)
    print(f"{'stop':>6} {'d Sharpe':>9} {'t-stat':>7} {'d CAGR':>8} "
          f"{'helps':>7} {'hurts':>6} {'d maxDD':>8}")
    print('-' * 88)
    base = per_name[None]
    for lvl in STOP_LEVELS:
        if lvl is None:
            continue
        cur = per_name[lvl]
        n = min(len(base), len(cur))
        ds = np.array([cur[i]['sharpe'] - base[i]['sharpe'] for i in range(n)])
        dc = np.array([cur[i]['cagr'] - base[i]['cagr'] for i in range(n)])
        dd = np.array([cur[i]['maxdd'] - base[i]['maxdd'] for i in range(n)])
        t = ds.mean() / (ds.std(ddof=1) / np.sqrt(n)) if n > 2 and ds.std(ddof=1) > 0 else 0.0
        print(f"{lvl*100:>5.0f}% {ds.mean():>+9.3f} {t:>7.2f} {dc.mean()*100:>+7.2f}% "
              f"{(ds > 0).sum():>7} {(ds < 0).sum():>6} {dd.mean()*100:>+7.2f}%")

    # ---- did the stop actually help? forward returns after each stop ----
    print('\n' + '=' * 88)
    print('WAS EACH STOP CORRECT? — forward 20-day return after the stop-out')
    print('=' * 88)
    for lvl, evs in all_stops.items():
        fwd = np.array([e['fwd20'] for e in evs if not np.isnan(e['fwd20'])])
        if len(fwd) == 0:
            continue
        print(f"  {lvl*100:.0f}%: {len(fwd)} stops | mean fwd20 {fwd.mean()*100:+.2f}% "
              f"| median {np.median(fwd)*100:+.2f}% | "
              f"declined further {(fwd < 0).mean()*100:.0f}% of the time")

    # ---- save ----
    out = pd.DataFrame([{
        'stop': 'none' if lvl is None else f'{lvl*100:.0f}%',
        'port_cagr': port_metrics[lvl]['cagr'],
        'port_vol': port_metrics[lvl]['vol'],
        'port_sharpe': port_metrics[lvl]['sharpe'],
        'port_maxdd': port_metrics[lvl]['maxdd'],
        'trades': trades[lvl],
    } for lvl in STOP_LEVELS])
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stop_backtest_results.csv')
    out.to_csv(dest, index=False)
    print(f'\nSaved: {dest}')


if __name__ == '__main__':
    main()
