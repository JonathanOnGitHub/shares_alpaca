#!/usr/bin/env python3
"""
Backtest v2 — equAL RISK PER TRADE.

v1 flaw: stop anchored to the trigger level T, not the entry. The reclaim entry
sits above T, so it got a wider stop and stopped out 1.6% of the time vs 47% for
the limit. That measured stop distance, not entry quality.

v2: stop and target are both measured from the ENTRY price, so every trade risks
the same amount. Costs of COST% round trip are deducted. What remains is a clean
test of the entry mechanic itself.

  Trigger T = previous session's high
  LIMIT   : low <= T                    -> fill at T
  RECLAIM : close > T after close <= T  -> fill at that close
  Stop    = entry * (1 - K_STOP*ATR%)
  Target  = entry * (1 + K_TGT*ATR%)
  Exit    : stop, target, or session close
"""
import yfinance as yf, pandas as pd, numpy as np, time, sys

UNIVERSE = [
    "NVDA","AMD","MU","DELL","ANET","MS","GE","CRWD","KLAC",
    "SOFI","PLTR","MARA","RIOT","COIN","HOOD","IONQ","RKLB","ACHR","SOUN","LCID","RIVN","NIO",
    "SAFX",
]
K_STOP, K_TGT, COST = 1.0, 1.5, 0.10   # cost in % round trip

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

        def run(j, entry, strat):
            stop, tgt = entry*(1-K_STOP*a), entry*(1+K_TGT*a)
            why, px = "eod", closes[-1]
            for k in range(j, len(closes)):
                if lows[k] <= stop: why, px = "stop", stop; break
                if highs[k] >= tgt: why, px = "target", tgt; break
            rows.append(dict(sym=sym, date=str(d), strat=strat, entry=entry,
                             premium=(entry-T)/T*100, why=why,
                             ret=(px/entry-1)*100 - COST,
                             mae=(entry-lows[j:].min())/entry*100))

        li = np.where(lows <= T)[0]
        if len(li):
            run(int(li[0]), T, "LIMIT")

        below = closes <= T
        if below.any():
            j0 = int(np.argmax(below))
            ri = np.where(closes[j0+1:] > T)[0]
            if len(ri):
                run(j0+1+int(ri[0]), float(closes[j0+1+int(ri[0])]), "RECLAIM")

r = pd.DataFrame(rows)
print(f"symbols {r.sym.nunique()}  trades {len(r)}  |  stop {K_STOP}xATR  target {K_TGT}xATR  cost {COST}%\n")
print("="*82); print("POOLED — equal risk per trade"); print("="*82)
for strat, g in r.groupby("strat"):
    print(f"  {strat:8s} n={len(g):4d}  win={100*(g.ret>0).mean():5.1f}%  "
          f"mean={g.ret.mean():+6.2f}%  med={g.ret.median():+6.2f}%  "
          f"MAE={g.mae.mean():5.2f}%  entry_prem={g.premium.mean():+5.2f}%  "
          f"{dict(g.why.value_counts())}")

piv = r.pivot_table(index=["sym","date"], columns="strat", values="ret").dropna()
print("\n" + "="*82); print("PAIRED — same session, both mechanics"); print("="*82)
print(f"  sessions {len(piv)}")
d = piv["RECLAIM"] - piv["LIMIT"]
print(f"  RECLAIM {piv['RECLAIM'].mean():+6.2f}%   LIMIT {piv['LIMIT'].mean():+6.2f}%")
print(f"  diff mean {d.mean():+6.2f}%   diff median {d.median():+6.2f}%")
print(f"  reclaim better in {100*(d>0).mean():.1f}% of sessions")
print(f"  reclaim WORSE by >5% in {100*(d<-5).mean():.1f}% | better by >5% in {100*(d>5).mean():.1f}%")
print(f"  worst 5 sessions for LIMIT: {piv['LIMIT'].nsmallest(5).round(1).tolist()}")
print(f"  worst 5 for RECLAIM:        {piv['RECLAIM'].nsmallest(5).round(1).tolist()}")

print("\n" + "="*82); print("SAFX ONLY"); print("="*82)
for strat, g in r[r.sym=="SAFX"].groupby("strat"):
    print(f"  {strat:8s} n={len(g):3d}  win={100*(g.ret>0).mean():5.1f}%  mean={g.ret.mean():+6.2f}%  MAE={g.mae.mean():5.2f}%")
