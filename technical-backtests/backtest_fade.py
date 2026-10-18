#!/usr/bin/env python3
"""
Does the breakout result fade?

The long study found buying daily breakouts returns -0.15% of excess vs random
entry at t=-5.09 (n=17,054). This asks whether the mirror - shorting breakouts -
carries a real, cost-surviving edge.

Three things decide that, and all three are traps:

 1. EXCESS IS NEARLY SYMMETRIC BY CONSTRUCTION. If long-breakout excess is -0.15%,
    short-breakout excess is roughly +0.15% BEFORE costs, because the short
    baseline is just the negative of the long baseline. So the question is never
    "is there a fade edge" - it is "is +0.15% bigger than the cost of being short".

 2. SHORT COSTS ARE STRUCTURALLY HIGHER than long costs on the same name:
    FX 0.30% round trip + spread paid twice + stock borrow. Long costs ~0.30%
    all-in. Short costs ~0.40%+ depending on borrow.

 3. THE PAYOFF IS INVERTED. Long breakouts are right-skewed (mean > median).
    Shorting them produces a high win rate with a mean below the median - you
    collect small premiums and pay occasional large losses. Positive expectancy
    on paper, negative skew in the tail.

Gross excess (cost 0 on both sides) is reported so the comparison is clean;
costs are then applied explicitly at the end.
"""
import yfinance as yf, pandas as pd, numpy as np, os, pickle, time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("BACKTEST_CACHE", os.path.join(HERE, "cache_daily"))
MAX_HOLD = 5

# Same 51-name universe as backtest_daily.py, so the fade is measured on the
# exact signal set the long study produced.
UNIVERSE = [
    "NVDA","AMD","MU","DELL","ANET","MS","GE","CRWD","KLAC",
    "SOFI","PLTR","MARA","RIOT","COIN","HOOD","IONQ","RKLB","ACHR","SOUN","LCID","RIVN","NIO","SAFX",
    "AAPL","MSFT","GOOGL","AMZN","META","JPM","BAC","XOM","CVX","JNJ","PFE","UNH",
    "HD","WMT","KO","PEP","PG","MCD","DIS","VZ","T","CAT","BA","HON","MMM","IBM",
    "INTC","QCOM","TXN","AVGO",
]
os.makedirs(CACHE, exist_ok=True)


def load(sym):
    p = os.path.join(CACHE, f"{sym}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    try:
        df = yf.Ticker(sym).history(period="10y", interval="1d", auto_adjust=True)
        time.sleep(0.6)
    except Exception:
        return None
    if df is None or len(df) < 300:
        return None
    pickle.dump(df, open(p, "wb"))
    return df


def atr_series(df, n=14):
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    return (tr.rolling(n).mean() / df["Close"]).values


def sim_short(o, h, l, c, j, atr, ks, kt, mh):
    """Enter short at open of j+1. Return (gross_ret%, mae%, why)."""
    e = j + 1
    if e >= len(c) or np.isnan(atr[j]):
        return None
    entry = float(o[e])
    if entry <= 0:
        return None
    a = float(atr[j])
    stop = entry * (1 + ks * a)
    tgt = entry * (1 - kt * a) if kt > 0 else None
    why, px = "time", None
    last = min(e + mh, len(c) - 1)
    for k in range(e, last + 1):
        if h[k] >= stop:
            why, px = "stop", stop
            break
        if tgt is not None and l[k] <= tgt:
            why, px = "target", tgt
            break
    if px is None:
        px = float(c[last])
    ret = (entry - px) / entry * 100          # short P&L
    mae = (float(np.max(h[e:last + 1])) - entry) / entry * 100
    return ret, mae, why


def sim_long(o, h, l, c, j, atr, ks, kt, mh):
    e = j + 1
    if e >= len(c) or np.isnan(atr[j]):
        return None
    entry = float(o[e])
    if entry <= 0:
        return None
    a = float(atr[j])
    stop = entry * (1 - ks * a)
    tgt = entry * (1 + kt * a) if kt > 0 else None
    why, px = "time", None
    last = min(e + mh, len(c) - 1)
    for k in range(e, last + 1):
        if l[k] <= stop:
            why, px = "stop", stop
            break
        if tgt is not None and h[k] >= tgt:
            why, px = "target", tgt
            break
    if px is None:
        px = float(c[last])
    return (px / entry - 1) * 100, (entry - float(np.min(l[e:last + 1]))) / entry * 100, why


data = {}
for s in UNIVERSE:
    df = load(s)
    if df is None or len(df) < 300:
        continue
    df = df.dropna()
    a = atr_series(df)
    if np.nanmedian(a) > 0.10:
        continue
    data[s] = dict(dates=df.index,
                   o=df["Open"].values.astype(float), h=df["High"].values.astype(float),
                   l=df["Low"].values.astype(float), c=df["Close"].values.astype(float),
                   atr=a)

print(f"symbols: {len(data)}")

# ---------------- breakout signal (identical to the long study) ----------------
sig = []
for s, d in data.items():
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    for j in range(60, len(c) - 1):
        if np.isnan(d["atr"][j]):
            continue
        if c[j] > h[j-5:j].max():
            sig.append((s, j, d["dates"][j]))
print(f"breakout signals: {len(sig):,}")

SCHEMES = [("1.0/1.5 ATR 5d", 1.0, 1.5, 5), ("1.0/3.0 ATR 10d", 1.0, 3.0, 10),
           ("2.0/3.0 ATR 10d", 2.0, 3.0, 10), ("2.0/6.0 ATR 20d", 2.0, 6.0, 20),
           ("no stop 5d", 99.0, 0.0, 5), ("no stop 20d", 99.0, 0.0, 20)]

print(f"\n{'scheme':18s} {'n':>6} {'win%':>6} {'mean%':>8} {'med%':>8} "
      f"{'base%':>8} {'excess%':>8} {'t':>6} {'stop%':>6} {'MAE%':>6}")
print("-" * 96)
store = {}
for label, ks, kt, mh in SCHEMES:
    fr, br = [], []
    for s, j, dt in sig:
        d = data[s]
        r = sim_short(d["o"], d["h"], d["l"], d["c"], j, d["atr"], ks, kt, mh)
        if r:
            fr.append((r[0], r[1], r[2], dt.year, s))
    for s, d in data.items():
        o, h, l, c, at = d["o"], d["h"], d["l"], d["c"], d["atr"]
        for j in range(60, len(c) - 1):
            r = sim_short(o, h, l, c, j, at, ks, kt, mh)
            if r:
                br.append(r[0])
    f = pd.DataFrame(fr, columns=["ret", "mae", "why", "yr", "sym"])
    bm = np.mean(br)
    se = f.ret.std(ddof=1) / np.sqrt(len(f))
    t = (f.ret.mean() - bm) / se if se > 0 else np.nan
    print(f"{label:18s} {len(f):>6} {100*(f.ret>0).mean():>6.1f} {f.ret.mean():>+8.2f} "
          f"{f.ret.median():>+8.2f} {bm:>+8.2f} {f.ret.mean()-bm:>+8.2f} {t:>+6.2f} "
          f"{100*(f.why=='stop').mean():>6.1f} {f.mae.mean():>6.2f}")
    store[label] = f

f = store["1.0/1.5 ATR 5d"]
_base_vals = []
for s, d in data.items():
    for j in range(60, len(d["c"]) - 1):
        _r = sim_short(d["o"], d["h"], d["l"], d["c"], j, d["atr"], 1.0, 1.5, 5)
        if _r:
            _base_vals.append(_r[0])
base_all = float(np.mean(_base_vals))

# ---------------- long side of the same signal, for the mirror check ----------
fl = []
for s, j, dt in sig:
    d = data[s]
    r = sim_long(d["o"], d["h"], d["l"], d["c"], j, d["atr"], 1.0, 1.5, 5)
    if r:
        fl.append(r[0])
fl = np.array(fl)
print(f"\nmirror check (1.0/1.5 ATR 5d):")
print(f"  LONG  breakout mean {fl.mean():+.3f}%   SHORT breakout mean {f.ret.mean():+.3f}%")
print(f"  sum = {fl.mean() + f.ret.mean():+.3f}%  (should be ~0 if the fade is a pure mirror)")

# ---------------- year stability ---------------------------------------------
print(f"\nby year (1.0/1.5 ATR 5d, gross):")
print(f"{'year':6s} {'n':>6} {'mean%':>8} {'win%':>6}")
for yr, g in f.groupby("yr"):
    print(f"{yr:6d} {len(g):>6} {g.ret.mean():>+8.2f} {100*(g.ret>0).mean():>6.1f}")

# ---------------- cost sensitivity -------------------------------------------
print(f"\ncost sensitivity on the faded breakout (gross excess {f.ret.mean()-base_all:+.2f}%):")
print("  FX 0.30% + spread 0.10% + borrow(1%/yr, 5d) 0.02%   = 0.42%")
print("  FX 0.30% + spread 0.10% + borrow(10%/yr, 5d) 0.20%  = 0.60%")
for c_ in (0.30, 0.42, 0.60):
    net = f.ret.mean() - base_all - c_
    print(f"  at {c_:.2f}% short cost -> net excess {net:+.2f}%   win {100*((f.ret-c_/2)>0).mean():.1f}%")

# ---------------- tail ---------------------------------------------------------
print(f"\ntail (gross, 1.0/1.5 ATR 5d):")
print(f"  worst trade {f.ret.min():+.1f}%   best {f.ret.max():+.1f}%")
print(f"  p1 {f.ret.quantile(.01):+.1f}%   p5 {f.ret.quantile(.05):+.1f}%   "
      f"p95 {f.ret.quantile(.95):+.1f}%   p99 {f.ret.quantile(.99):+.1f}%")
print(f"  mean {f.ret.mean():+.3f}%  median {f.ret.median():+.3f}%  -> skew "
      f"{'negative (premium selling)' if f.ret.mean() < f.ret.median() else 'positive'}")
print(f"  losers > 5%: {100*(f.ret < -5).mean():.2f}%   losers > 10%: {100*(f.ret < -10).mean():.2f}%")

f.to_csv(os.path.join(HERE, "fade_signals.csv"), index=False)
