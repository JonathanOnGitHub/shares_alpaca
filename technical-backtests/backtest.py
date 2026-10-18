#!/usr/bin/env python3
"""
Backtest: does a RECLAIM entry beat a passive LIMIT entry?

Both use the same trigger level and identical stop/target, so the only
difference is the entry mechanic.

  Trigger  T   = previous session's high
  LIMIT    : first bar in the session where low <= T  -> filled at T
  RECLAIM  : first close > T after price has closed <= T -> filled at that close
  Stop     : T * (1 - 1.0 * ATR%)      (ATR% from the symbol's own daily range)
  Target   : T * (1 + 2.0 * ATR%)
  Exit     : stop, target, or session close

Data: Yahoo 5m bars, 60-day limit. That is the hard ceiling — treat the result
as suggestive, not conclusive.
"""
import yfinance as yf, pandas as pd, numpy as np, time, sys, json

UNIVERSE = [
    # liquid large/mid
    "NVDA","AMD","MU","DELL","ANET","MS","GE","CRWD","KLAC",
    # high-beta / retail favourites
    "SOFI","PLTR","MARA","RIOT","COIN","HOOD","IONQ","RKLB","ACHR","SOUN","LCID","RIVN","NIO",
    # the name in question
    "SAFX",
]
K_STOP, K_TGT = 1.0, 2.0

def atr_pct(daily):
    """Daily ATR as a fraction of close, median over the window."""
    if len(daily) < 10: return None
    tr = pd.concat([
        daily["h"] - daily["l"],
        (daily["h"] - daily["c"].shift()).abs(),
        (daily["l"] - daily["c"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float((tr.rolling(14).mean() / daily["c"]).dropna().median())

rows = []
for sym in UNIVERSE:
    try:
        df = yf.Ticker(sym).history(period="60d", interval="5m", prepost=False)
    except Exception as e:
        print(f"  {sym}: fetch error {str(e)[:60]}", file=sys.stderr); continue
    time.sleep(1.2)
    if df is None or len(df) < 200:
        print(f"  {sym}: insufficient 5m data ({0 if df is None else len(df)})", file=sys.stderr); continue

    df = df.copy()
    df.index = df.index.tz_convert("America/New_York")
    dates = sorted(set(df.index.date))
    if len(dates) < 12:
        continue

    # build daily OHLC from the 5m bars
    daily = pd.DataFrame({
        "h": df.groupby(df.index.date)["High"].max(),
        "l": df.groupby(df.index.date)["Low"].min(),
        "c": df.groupby(df.index.date)["Close"].last(),
    })
    a = atr_pct(daily)
    if not a or a <= 0 or a > 0.6:
        continue

    for i in range(1, len(dates)):
        d, prev = dates[i], dates[i-1]
        s = df[df.index.date == d]
        if len(s) < 30: continue
        T = float(df[df.index.date == prev]["High"].max())
        if T <= 0: continue

        stop, tgt = T*(1-K_STOP*a), T*(1+K_TGT*a)
        closes = s["Close"].values; lows = s["Low"].values; highs = s["High"].values

        # --- LIMIT: first touch of T ---
        li = np.where(lows <= T)[0]
        if len(li):
            j = li[0]
            entry = T
            out = None
            for k in range(j, len(closes)):
                if lows[k] <= stop: out = ("stop", stop); break
                if highs[k] >= tgt: out = ("target", tgt); break
            if out is None: out = ("eod", closes[-1])
            rows.append(dict(sym=sym, date=str(d), strat="LIMIT", entry=entry,
                             exit=out[1], why=out[0],
                             ret=(out[1]/entry-1)*100,
                             mae=(entry-lows[j:].min())/entry*100))

        # --- RECLAIM: close back above T after closing below it ---
        below = closes <= T
        if below.any():
            j0 = int(np.argmax(below))
            ri = np.where(closes[j0+1:] > T)[0]
            if len(ri):
                j = j0 + 1 + int(ri[0])
                entry = closes[j]
                out = None
                for k in range(j, len(closes)):
                    if lows[k] <= stop: out = ("stop", stop); break
                    if highs[k] >= tgt: out = ("target", tgt); break
                if out is None: out = ("eod", closes[-1])
                rows.append(dict(sym=sym, date=str(d), strat="RECLAIM", entry=entry,
                                 exit=out[1], why=out[0],
                                 ret=(out[1]/entry-1)*100,
                                 mae=(entry-lows[j:].min())/entry*100))

r = pd.DataFrame(rows)
if r.empty:
    print("no trades generated"); sys.exit(0)

print(f"symbols: {r.sym.nunique()}   trades: {len(r)}   window: 60d of 5m bars\n")
print("="*78); print("POOLED"); print("="*78)
for strat, g in r.groupby("strat"):
    print(f"  {strat:8s} n={len(g):4d}  win={100*(g.ret>0).mean():5.1f}%  "
          f"mean={g.ret.mean():+6.2f}%  median={g.ret.median():+6.2f}%  "
          f"avgMAE={g.mae.mean():5.2f}%  exits={dict(g.why.value_counts())}")

print("\n" + "="*78); print("PAIRED — sessions where BOTH fired"); print("="*78)
piv = r.pivot_table(index=["sym","date"], columns="strat", values="ret")
both = piv.dropna()
if len(both):
    diff = (both["RECLAIM"] - both["LIMIT"])
    print(f"  paired sessions: {len(both)}")
    print(f"  RECLAIM mean {both['RECLAIM'].mean():+6.2f}%   LIMIT mean {both['LIMIT'].mean():+6.2f}%")
    print(f"  difference  {diff.mean():+6.2f}%  (median {diff.median():+.2f}%)")
    print(f"  reclaim better in {100*(diff>0).mean():.1f}% of paired sessions")
else:
    print("  no paired sessions")

print("\n" + "="*78); print("PER SYMBOL (n>=5)"); print("="*78)
for sym, g in r.groupby("sym"):
    if len(g) < 5: continue
    parts = []
    for strat, h in g.groupby("strat"):
        parts.append(f"{strat} n={len(h):3d} mean={h.ret.mean():+6.2f}% win={100*(h.ret>0).mean():4.0f}%")
    print(f"  {sym:6s} " + " | ".join(parts))
