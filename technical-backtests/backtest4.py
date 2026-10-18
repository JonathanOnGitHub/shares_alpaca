#!/usr/bin/env python3
"""
Backtest v4 — does HIGH VOLUME improve the reclaim entry?

Base case (v3) was: reclaim cuts the left tail but still loses money.
Hypothesis: reclaim + unusually high volume = genuine participation, not a
dead-cat bounce. Test whether that beats the base rate.

NO LOOK-AHEAD. Baselines use only PRIOR sessions.

Filters measured at the moment of entry:
  vol_ratio      = session cumulative volume at entry / median cum-vol at the same
                   time-of-day across the prior 10 sessions
  prev_vol_ratio = previous session's total volume / median of the prior 20
  prev_ret       = previous session's close-to-close move (catalyst proxy)
"""
import yfinance as yf, pandas as pd, numpy as np, time, datetime, sys

UNIVERSE = [
    "NVDA","AMD","MU","DELL","ANET","MS","GE","CRWD","KLAC",
    "SOFI","PLTR","MARA","RIOT","COIN","HOOD","IONQ","RKLB","ACHR","SOUN","LCID","RIVN","NIO",
    "SAFX",
]
K_STOP, K_TGT, COST = 1.0, 1.5, 0.10
WARMUP = 12

def atr_pct(daily):
    if len(daily) < 10: return None
    tr = pd.concat([daily["h"]-daily["l"],
                    (daily["h"]-daily["c"].shift()).abs(),
                    (daily["l"]-daily["c"].shift()).abs()], axis=1).max(axis=1)
    return float((tr.rolling(14).mean()/daily["c"]).dropna().median())

rows = []
for sym in UNIVERSE:
    try:
        df = yf.Ticker(sym).history(period="60d", interval="5m", prepost=False)
    except Exception:
        continue
    time.sleep(1.2)
    if df is None or len(df) < 400: continue
    df = df.copy(); df.index = df.index.tz_convert("America/New_York")
    dates = sorted(set(df.index.date))
    if len(dates) < WARMUP + 10: continue

    ses = {d: df[df.index.date == d] for d in dates}
    daily = pd.DataFrame({"h": df.groupby(df.index.date)["High"].max(),
                          "l": df.groupby(df.index.date)["Low"].min(),
                          "c": df.groupby(df.index.date)["Close"].last()})
    a = atr_pct(daily)
    if not a or not (0 < a <= 0.6): continue

    # cumulative-volume profile per session, keyed by time-of-day
    cum = {}
    for d, s in ses.items():
        cv = s["Volume"].fillna(0).cumsum()
        cv.index = [t.time() for t in s.index]
        cum[d] = cv
    tot = {d: float(ses[d]["Volume"].fillna(0).sum()) for d in dates}
    cls = {d: float(ses[d]["Close"].iloc[-1]) for d in dates}

    for i in range(WARMUP, len(dates)):
        d, prev = dates[i], dates[i-1]
        s = ses[d]
        if len(s) < 30: continue
        T = float(ses[prev]["High"].max())
        if T <= 0: continue
        closes, lows, highs = s["Close"].values, s["Low"].values, s["High"].values
        times = [t.time() for t in s.index]

        below = closes <= T
        if not below.any(): continue
        j0 = int(np.argmax(below))
        ri = np.where(closes[j0+1:] > T)[0]
        if not len(ri): continue
        j = j0 + 1 + int(ri[0])
        entry = float(closes[j])

        # --- volume filters, prior data only ---
        prior = dates[max(0, i-10):i]
        vals = [cum[pd][times[j]] for pd in prior if times[j] in cum[pd].index]
        base_cum = float(np.median(vals)) if vals else 0.0
        vol_ratio = (float(cum[d][times[j]]) / base_cum) if base_cum > 0 else np.nan
        pv = [tot[x] for x in dates[max(0, i-21):i-1]]
        base_tot = float(np.median(pv)) if pv else 0.0
        prev_vol_ratio = (tot[prev] / base_tot) if base_tot > 0 else np.nan
        prev_ret = ((cls[prev]/cls[dates[i-2]]-1)*100) if i >= 2 else np.nan

        stop, tgt = entry*(1-K_STOP*a), entry*(1+K_TGT*a)
        why, px = "eod", closes[-1]
        for k in range(j, len(closes)):
            if lows[k] <= stop: why, px = "stop", stop; break
            if highs[k] >= tgt: why, px = "target", tgt; break

        rows.append(dict(sym=sym, date=str(d), entry=entry, why=why,
                         ret=(px/entry-1)*100 - COST,
                         mae=(entry-lows[j:].min())/entry*100,
                         vol_ratio=vol_ratio, prev_vol_ratio=prev_vol_ratio,
                         prev_ret=prev_ret))

r = pd.DataFrame(rows).dropna(subset=["vol_ratio"])
print(f"reclaim signals: {len(r)} across {r.sym.nunique()} symbols\n")

def show(title, mask):
    g = r[mask]
    if len(g) < 10:
        print(f"  {title:28s} n={len(g):3d}  (too few)"); return
    print(f"  {title:28s} n={len(g):4d}  win={100*(g.ret>0).mean():5.1f}%  "
          f"mean={g.ret.mean():+6.2f}%  med={g.ret.median():+6.2f}%  "
          f"MAE={g.mae.mean():5.2f}%  stop={100*(g.why=='stop').mean():4.1f}%")

print("="*96); print("BASELINE"); print("="*96)
show("ALL reclaim", r.index == r.index)

print("\n" + "="*96); print("BY SESSION VOLUME AT ENTRY (vol_ratio)"); print("="*96)
show("quiet  <1.0x",  r.vol_ratio < 1.0)
show("normal 1.0-2.0x", (r.vol_ratio >= 1.0) & (r.vol_ratio < 2.0))
show("elevated 2-4x", (r.vol_ratio >= 2.0) & (r.vol_ratio < 4.0))
show("heavy >4x", r.vol_ratio >= 4.0)

print("\n" + "="*96); print("BY PREVIOUS SESSION'S VOLUME (catalyst-day proxy)"); print("="*96)
show("prev quiet <1.0x", r.prev_vol_ratio < 1.0)
show("prev normal 1-2x", (r.prev_vol_ratio >= 1.0) & (r.prev_vol_ratio < 2.0))
show("prev elevated 2-4x", (r.prev_vol_ratio >= 2.0) & (r.prev_vol_ratio < 4.0))
show("prev heavy >4x", r.prev_vol_ratio >= 4.0)

print("\n" + "="*96); print("BY PREVIOUS SESSION'S MOVE (catalyst proxy)"); print("="*96)
show("prev flat  |r|<2%", r.prev_ret.abs() < 2)
show("prev move 2-5%", (r.prev_ret.abs() >= 2) & (r.prev_ret.abs() < 5))
show("prev move 5-10%", (r.prev_ret.abs() >= 5) & (r.prev_ret.abs() < 10))
show("prev move >10%", r.prev_ret.abs() >= 10)

print("\n" + "="*96); print("COMBINED FILTERS"); print("="*96)
show("vol>2x", r.vol_ratio >= 2.0)
show("vol>2x AND prev_vol>2x", (r.vol_ratio >= 2.0) & (r.prev_vol_ratio >= 2.0))
show("vol>2x AND |prev_ret|>5%", (r.vol_ratio >= 2.0) & (r.prev_ret.abs() > 5))
show("all three", (r.vol_ratio >= 2.0) & (r.prev_vol_ratio >= 2.0) & (r.prev_ret.abs() > 5))

print("\n" + "="*96); print("SAFX"); print("="*96)
sf = r[r.sym == "SAFX"]
if len(sf): print(f"  n={len(sf)}  mean={sf.ret.mean():+.2f}%  MAE={sf.mae.mean():.2f}%  "
                  f"median vol_ratio={sf.vol_ratio.median():.2f}  median prev_vol={sf.prev_vol_ratio.median():.2f}")
