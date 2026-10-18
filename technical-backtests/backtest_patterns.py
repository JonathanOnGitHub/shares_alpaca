#!/usr/bin/env python3
"""
Do candlestick / price-action patterns actually work as entry triggers?

Tests six long-entry rules on the same 23-symbol, 60-day 5m dataset used for the
reclaim study, with IDENTICAL exits (1xATR stop, 1.5xATR target, EOD, 0.1% cost),
so the only variable is the entry predicate.

Rules
  ENGULF      bullish engulfing (prev red, current green, body engulfs prev body)
  HAMMER      lower wick >= 2x body, upper wick <= body, in lower half of range
  INSIDE      inside bar, then close above the prior bar's high
  NR7         narrowest range of last 7 bars, then close above that bar's high
  PULLBACK    price > 50-bar SMA, 3 consecutive lower closes, then close > prior high
  RECLAIM     close > previous session high after having closed <= it  (baseline)

Cache: 5m bars are saved to ./cache so repeat runs don't re-hit Yahoo.
"""
import yfinance as yf, pandas as pd, numpy as np, time, os, sys, pickle

UNIVERSE = [
    "NVDA","AMD","MU","DELL","ANET","MS","GE","CRWD","KLAC",
    "SOFI","PLTR","MARA","RIOT","COIN","HOOD","IONQ","RKLB","ACHR","SOUN","LCID","RIVN","NIO",
    "SAFX",
]
K_STOP, K_TGT, COST = 1.0, 1.5, 0.10
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("BACKTEST_CACHE", os.path.join(HERE, "cache"))
os.makedirs(CACHE, exist_ok=True)

def atr_pct(daily):
    if len(daily) < 10: return None
    tr = pd.concat([daily["h"]-daily["l"],
                    (daily["h"]-daily["c"].shift()).abs(),
                    (daily["l"]-daily["c"].shift()).abs()], axis=1).max(axis=1)
    return float((tr.rolling(14).mean()/daily["c"]).dropna().median())

def load(sym):
    p = os.path.join(CACHE, f"{sym}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    try:
        df = yf.Ticker(sym).history(period="60d", interval="5m", prepost=False)
        time.sleep(1.2)
    except Exception:
        return None
    if df is None or len(df) < 200: return None
    df = df.copy(); df.index = df.index.tz_convert("America/New_York")
    pickle.dump(df, open(p, "wb"))
    return df

def body(o, c): return abs(c - o)
def is_green(o, c): return c > o

rows = []
for sym in UNIVERSE:
    df = load(sym)
    if df is None: continue
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
        if len(s) < 40: continue
        o = s["Open"].values; c = s["Close"].values
        h = s["High"].values; l = s["Low"].values
        n = len(c)
        sma50 = pd.Series(c).rolling(50).mean().values
        rng = h - l
        prev_high_session = float(df[df.index.date == prev]["High"].max())

        def run(j, rule):
            entry = float(c[j])
            if entry <= 0: return
            stop, tgt = entry*(1-K_STOP*a), entry*(1+K_TGT*a)
            why, px = "eod", c[-1]
            for k in range(j, n):
                if l[k] <= stop: why, px = "stop", stop; break
                if h[k] >= tgt: why, px = "target", tgt; break
            rows.append(dict(sym=sym, date=str(d), rule=rule, entry=entry, why=why,
                             ret=(px/entry-1)*100 - COST,
                             mae=(entry-l[j:].min())/entry*100))

        fired = set()
        for j in range(5, n-1):
            # only one signal per rule per session, and only in the first 2/3 of it
            if j > n*0.75: break

            if "ENGULF" not in fired:
                if (not is_green(o[j-1], c[j-1]) and is_green(o[j], c[j])
                        and body(o[j], c[j]) > body(o[j-1], c[j-1])
                        and c[j] >= o[j-1] and o[j] <= c[j-1]):
                    run(j, "ENGULF"); fired.add("ENGULF")

            if "HAMMER" not in fired and rng[j] > 0:
                lower = min(o[j], c[j]) - l[j]
                upper = h[j] - max(o[j], c[j])
                # hammer: long lower wick, small upper wick, CLOSE NEAR THE HIGH
                if lower >= 2*body(o[j], c[j]) and upper <= body(o[j], c[j]) \
                   and (c[j]-l[j]) >= 0.6*rng[j]:
                    run(j, "HAMMER"); fired.add("HAMMER")

            if "INSIDE" not in fired and j >= 2:
                # inside bar forms on j-1; breakout is the close above ITS parent's high
                if h[j-1] < h[j-2] and l[j-1] > l[j-2] and c[j] > h[j-2]:
                    run(j, "INSIDE"); fired.add("INSIDE")

            if "NR7" not in fired and j >= 8:
                # NR7 forms on j-1 (narrowest of the last 7); breakout close above its high
                if rng[j-1] == rng[j-7:j].min() and c[j] > h[j-1]:
                    run(j, "NR7"); fired.add("NR7")

            if "PULLBACK" not in fired and j >= 51 and not np.isnan(sma50[j]):
                if (c[j-3] > c[j-2] > c[j-1] < c[j]) and c[j] > c[j-1] \
                   and c[j] > sma50[j]:
                    run(j, "PULLBACK"); fired.add("PULLBACK")

        # RECLAIM baseline
        below = c <= prev_high_session
        if below.any():
            j0 = int(np.argmax(below))
            ri = np.where(c[j0+1:] > prev_high_session)[0]
            if len(ri):
                j = j0+1+int(ri[0])
                if j <= n*0.75: run(j, "RECLAIM")

r = pd.DataFrame(rows)
print(f"trades: {len(r)}  symbols: {r.sym.nunique()}  |  stop {K_STOP}xATR target {K_TGT}xATR cost {COST}%\n")
print(f"{'rule':10s} {'n':>5} {'win%':>7} {'mean%':>8} {'med%':>8} {'MAE%':>7} {'stop%':>7}")
print("-"*60)
for rule, g in r.groupby("rule"):
    print(f"{rule:10s} {len(g):>5} {100*(g.ret>0).mean():>7.1f} {g.ret.mean():>+8.2f} "
          f"{g.ret.median():>+8.2f} {g.mae.mean():>7.2f} {100*(g.why=='stop').mean():>7.1f}")
print("-"*60)
print(f"{'ALL':10s} {len(r):>5} {100*(r.ret>0).mean():>7.1f} {r.ret.mean():>+8.2f} "
      f"{r.ret.median():>+8.2f} {r.mae.mean():>7.2f} {100*(r.why=='stop').mean():>7.1f}")
