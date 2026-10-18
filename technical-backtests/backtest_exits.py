#!/usr/bin/env python3
"""
If every entry rule gives the same result, the EXIT must be what drives P&L.
Takes all pattern signals from the cached dataset and applies several exit schemes
to the SAME entries.
"""
import pandas as pd, numpy as np, os, pickle

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("BACKTEST_CACHE", os.path.join(HERE, "cache"))
COST = 0.10
EXITS = {
  "EOD only":        None,
  "0.5/1.0 ATR":     (0.5, 1.0),
  "1.0/1.5 ATR":     (1.0, 1.5),
  "1.0/3.0 ATR":     (1.0, 3.0),
  "2.0/3.0 ATR":     (2.0, 3.0),
  "trail 0.75 ATR":  ("TRAIL", 0.75),
}

def atr_pct(daily):
    if len(daily) < 10: return None
    tr = pd.concat([daily["h"]-daily["l"],
                    (daily["h"]-daily["c"].shift()).abs(),
                    (daily["l"]-daily["c"].shift()).abs()], axis=1).max(axis=1)
    return float((tr.rolling(14).mean()/daily["c"]).dropna().median())

def body(o,c): return abs(c-o)
def green(o,c): return c>o

results = {k: [] for k in EXITS}

for f in sorted(os.listdir(CACHE)):
    if not f.endswith(".pkl"): continue
    sym = f[:-4]
    df = pickle.load(open(os.path.join(CACHE,f),"rb"))
    dates = sorted(set(df.index.date))
    if len(dates) < 12: continue
    daily = pd.DataFrame({"h": df.groupby(df.index.date)["High"].max(),
                          "l": df.groupby(df.index.date)["Low"].min(),
                          "c": df.groupby(df.index.date)["Close"].last()})
    a = atr_pct(daily)
    if not a or not (0 < a <= 0.6): continue

    for i in range(1,len(dates)):
        s = df[df.index.date==dates[i]]
        if len(s) < 40: continue
        o=s["Open"].values; c=s["Close"].values; h=s["High"].values; l=s["Low"].values
        n=len(c); rng=h-l
        sma50=pd.Series(c).rolling(50).mean().values
        prev_high=float(df[df.index.date==dates[i-1]]["High"].max())
        jmax = int(n*0.75)

        sigs=set()
        for j in range(5,jmax):
            if "ENGULF" not in sigs and (not green(o[j-1],c[j-1]) and green(o[j],c[j])
                and body(o[j],c[j])>body(o[j-1],c[j-1]) and c[j]>=o[j-1] and o[j]<=c[j-1]):
                sigs.add(j)
            if "HAMMER" not in sigs and rng[j]>0:
                lo=min(o[j],c[j])-l[j]; up=h[j]-max(o[j],c[j])
                if lo>=2*body(o[j],c[j]) and up<=body(o[j],c[j]) and (c[j]-l[j])>=0.6*rng[j]:
                    sigs.add(j)
            if "PULLBACK" not in sigs and j>=51 and not np.isnan(sma50[j]):
                if c[j-3]>c[j-2]>c[j-1]<c[j] and c[j]>c[j-1] and c[j]>sma50[j]:
                    sigs.add(j)

        for j in sigs:
            entry=float(c[j])
            if entry<=0: continue
            for name,cfg in EXITS.items():
                if cfg is None:
                    px=c[-1]; why="eod"
                elif cfg[0]=="TRAIL":
                    k=cfg[1]; trail=entry-k*a; px=c[-1]; why="eod"
                    for m in range(j,n):
                        if l[m]<=trail: px,why=trail,"trail"; break
                        trail=max(trail,h[m]-k*a)
                else:
                    ks,kg=cfg; stop,tgt=entry*(1-ks*a),entry*(1+kg*a)
                    px,why=c[-1],"eod"
                    for m in range(j,n):
                        if l[m]<=stop: px,why=stop,"stop"; break
                        if h[m]>=tgt: px,why=tgt,"target"; break
                results[name].append(dict(sym=sym,ret=(px/entry-1)*100-COST,why=why))

print(f"{'exit':16s} {'n':>6} {'win%':>7} {'mean%':>8} {'med%':>8} {'stops%':>8}")
print("-"*60)
for name,rows in results.items():
    if not rows: continue
    r=pd.DataFrame(rows)
    print(f"{name:16s} {len(r):>6} {100*(r.ret>0).mean():>7.1f} {r.ret.mean():>+8.2f} "
          f"{r.ret.median():>+8.2f} {100*(r.why=='stop').mean():>8.1f}")
