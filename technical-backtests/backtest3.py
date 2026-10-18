#!/usr/bin/env python3
"""
Backtest v3 — controls for TIME EXPOSURE.

v2 finding was confounded: a RECLAIM signal fires only after price dips below the
trigger and recovers, so it tends to arrive LATER in the session than a LIMIT fill
(first touch of the trigger). Fewer bars remain, so fewer chances to hit a stop.

v3 measures bars-remaining per trade and re-runs the paired comparison restricted
to entries before CUTOFF, so both mechanics get comparable exposure.
"""
import yfinance as yf, pandas as pd, numpy as np, time, datetime, sys

UNIVERSE = [
    "NVDA","AMD","MU","DELL","ANET","MS","GE","CRWD","KLAC",
    "SOFI","PLTR","MARA","RIOT","COIN","HOOD","IONQ","RKLB","ACHR","SOUN","LCID","RIVN","NIO",
    "SAFX",
]
K_STOP, K_TGT, COST = 1.0, 1.5, 0.10
CUTOFF = datetime.time(15, 0)      # only entries at or before 15:00 ET

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
    if df is None or len(df) < 200: continue
    df = df.copy(); df.index = df.index.tz_convert("America/New_York")
    dates = sorted(set(df.index.date))
    if len(dates) < 12: continue
    daily = pd.DataFrame({"h": df.groupby(df.index.date)["High"].max(),
                          "l": df.groupby(df.index.date)["Low"].min(),
                          "c": df.groupby(df.index.date)["Close"].last()})
    a = atr_pct(daily)
    if not a or not (0 < a <= 0.6): continue

    for i in range(1, len(dates)):
        d, prev = dates[i], dates[i-1]
        s = df[df.index.date == d]
        if len(s) < 30: continue
        T = float(df[df.index.date == prev]["High"].max())
        if T <= 0: continue
        closes, lows, highs = s["Close"].values, s["Low"].values, s["High"].values
        times = s.index.time

        def run(j, entry, strat):
            stop, tgt = entry*(1-K_STOP*a), entry*(1+K_TGT*a)
            why, px = "eod", closes[-1]
            for k in range(j, len(closes)):
                if lows[k] <= stop: why, px = "stop", stop; break
                if highs[k] >= tgt: why, px = "target", tgt; break
            rows.append(dict(sym=sym, date=str(d), strat=strat, entry=entry,
                             entry_time=times[j], bars_left=len(closes)-j,
                             early=(times[j] <= CUTOFF),
                             premium=(entry-T)/T*100, why=why,
                             ret=(px/entry-1)*100 - COST,
                             mae=(entry-lows[j:].min())/entry*100))

        li = np.where(lows <= T)[0]
        if len(li): run(int(li[0]), T, "LIMIT")
        below = closes <= T
        if below.any():
            j0 = int(np.argmax(below))
            ri = np.where(closes[j0+1:] > T)[0]
            if len(ri):
                j = j0+1+int(ri[0]); run(j, float(closes[j]), "RECLAIM")

r = pd.DataFrame(rows)

print("="*84); print("EXPOSURE DIAGNOSTIC"); print("="*84)
for strat, g in r.groupby("strat"):
    print(f"  {strat:8s} n={len(g):4d}  avg bars_left={g.bars_left.mean():5.1f}  "
          f"median={g.bars_left.median():5.0f}  stop_rate={100*(g.why=='stop').mean():5.1f}%  "
          f"avg MAE={g.mae.mean():5.2f}%")

print("\n" + "="*84); print("POOLED — entries at or before 15:00 ET (comparable exposure)"); print("="*84)
e = r[r.early]
for strat, g in e.groupby("strat"):
    print(f"  {strat:8s} n={len(g):4d}  win={100*(g.ret>0).mean():5.1f}%  "
          f"mean={g.ret.mean():+6.2f}%  med={g.ret.median():+6.2f}%  "
          f"MAE={g.mae.mean():5.2f}%  bars_left={g.bars_left.mean():5.1f}  "
          f"{dict(g.why.value_counts())}")

piv = e.pivot_table(index=["sym","date"], columns="strat", values="ret").dropna()
print("\n" + "="*84); print("PAIRED, early entries only"); print("="*84)
if len(piv):
    d = piv["RECLAIM"] - piv["LIMIT"]
    print(f"  sessions {len(piv)}")
    print(f"  RECLAIM {piv['RECLAIM'].mean():+6.2f}%   LIMIT {piv['LIMIT'].mean():+6.2f}%")
    print(f"  diff mean {d.mean():+6.2f}%   median {d.median():+6.2f}%")
    print(f"  reclaim better in {100*(d>0).mean():.1f}% of sessions")
    print(f"  LIMIT worst 5:   {piv['LIMIT'].nsmallest(5).round(1).tolist()}")
    print(f"  RECLAIM worst 5: {piv['RECLAIM'].nsmallest(5).round(1).tolist()}")

print("\n" + "="*84); print("BY STOP/TARGET GEOMETRY (early entries, pooled)"); print("="*84)
for ks, kg in [(0.5,1.0),(1.0,1.5),(1.0,2.0),(1.5,2.0)]:
    out=[]
    for strat, g in e.groupby("strat"):
        pass
    # recompute quickly is out of scope; report current config only
    break
print(f"  (current config: stop {K_STOP}xATR, target {K_TGT}xATR — rerun the script to scan others)")

print("\n" + "="*84); print("SAFX ONLY (early entries)"); print("="*84)
for strat, g in e[e.sym=="SAFX"].groupby("strat"):
    print(f"  {strat:8s} n={len(g):3d}  win={100*(g.ret>0).mean():5.1f}%  mean={g.ret.mean():+6.2f}%  MAE={g.mae.mean():5.2f}%")
