#!/usr/bin/env python3
"""
Daily-bar version of the candlestick/price-action study.

The 5m run put every rule and every exit at ~-0.2%. That test had a flaw that
gets much worse on daily bars, so this one fixes it and adds a control:

  * ENTRY AT THE NEXT BAR'S OPEN, not the signal bar's close. The 5m test
    entered at the close of the bar that generated the signal, which quietly
    assumes you can act on information you only have after the bar completes.
    On daily bars that assumption is much more expensive.

  * UNCONDITIONAL BASELINE. This sample covers a long bull market, so ANY long
    entry has positive drift. The raw mean is therefore not evidence of edge.
    Baseline = the same exit applied to EVERY bar. Excess = rule - baseline.
    That excess is the only number that means anything.

  * t-stat on the mean. Daily return variance is roughly 10x the 5m variance,
    so a +0.3% mean on n=800 can easily be noise. Reported so we don't fool
    ourselves the way the 5m test nearly did.

Rules (evaluated on bar j, entered at open of bar j+1)
  ENGULF    prev red, current green, body engulfs prev body
  HAMMER    lower wick >= 2x body, upper wick <= body, close in upper 60% range
  INSIDE    inside bar at j-1, close(j) > high(j-2)
  NR7       range(j) narrowest of last 7, close(j) > high(j)
  PULLBACK  close > SMA50, 3 lower closes, then up close
  BREAKOUT  close(j) > max(high of j-5..j-1)      [daily analogue of RECLAIM]

Cache: ./cache_daily so repeat runs don't re-hit Yahoo.
"""
import yfinance as yf, pandas as pd, numpy as np, time, os, pickle, sys

UNIVERSE = [
    # original 23 (for comparability with the 5m study)
    "NVDA","AMD","MU","DELL","ANET","MS","GE","CRWD","KLAC",
    "SOFI","PLTR","MARA","RIOT","COIN","HOOD","IONQ","RKLB","ACHR","SOUN","LCID","RIVN","NIO","SAFX",
    # large caps with long history, to dilute today's-winners survivorship bias
    "AAPL","MSFT","GOOGL","AMZN","META","JPM","BAC","XOM","CVX","JNJ","PFE","UNH",
    "HD","WMT","KO","PEP","PG","MCD","DIS","VZ","T","CAT","BA","HON","MMM","IBM",
    "INTC","QCOM","TXN","AVGO",
]

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("BACKTEST_CACHE", os.path.join(HERE, "cache_daily"))
os.makedirs(CACHE, exist_ok=True)

COST = 0.10          # % round trip, base case
COST_REAL = 0.30     # % round trip, actual T212 US-share FX cost
ATR_MAX = 0.10       # drop names with >10% daily ATR
MAX_HOLD = 5         # bars, default time stop


def atr_series(df, n=14):
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean() / df["Close"]


def load(sym):
    p = os.path.join(CACHE, f"{sym}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    try:
        df = yf.Ticker(sym).history(period="10y", interval="1d", auto_adjust=True)
        time.sleep(0.6)
    except Exception as e:
        print(f"  {sym}: {e}")
        return None
    if df is None or len(df) < 300:
        return None
    pickle.dump(df, open(p, "wb"))
    return df


def body(o, c):
    return abs(c - o)


def is_green(o, c):
    return c > o


def rule_fires(name, o, h, l, c, j, sma50, rng):
    """True if `name` fires on bar j (using only info up to and including j)."""
    if name == "ENGULF":
        return (not is_green(o[j-1], c[j-1]) and is_green(o[j], c[j])
                and body(o[j], c[j]) > body(o[j-1], c[j-1])
                and c[j] >= o[j-1] and o[j] <= c[j-1])
    if name == "HAMMER":
        if rng[j] <= 0:
            return False
        lower = min(o[j], c[j]) - l[j]
        upper = h[j] - max(o[j], c[j])
        return (lower >= 2 * body(o[j], c[j]) and upper <= body(o[j], c[j])
                and (c[j] - l[j]) >= 0.6 * rng[j])
    if name == "INSIDE":
        return h[j-1] < h[j-2] and l[j-1] > l[j-2] and c[j] > h[j-2]
    if name == "NR7":
        # bar j-1 is the narrowest of the last 7; breakout is close(j) above ITS high.
        # (c[j] > h[j] is impossible - that bug silently killed this rule once.)
        return rng[j-1] == rng[j-7:j].min() and c[j] > h[j-1]
    if name == "PULLBACK":
        if np.isnan(sma50[j]):
            return False
        return (c[j-3] > c[j-2] > c[j-1] < c[j] and c[j] > c[j-1] and c[j] > sma50[j])
    if name == "BREAKOUT":
        return c[j] > h[j-5:j].max()
    return False


RULES = ["ENGULF", "HAMMER", "INSIDE", "NR7", "PULLBACK", "BREAKOUT"]


def simulate(o, h, l, c, j, atr, k_stop, k_tgt, max_hold, cost):
    """Enter at open of j+1. Return (ret%, mae%, why) or None."""
    e = j + 1
    if e >= len(c):
        return None
    entry = float(o[e])
    if entry <= 0 or np.isnan(atr[j]):
        return None
    a = float(atr[j])
    stop = entry * (1 - k_stop * a)
    tgt = entry * (1 + k_tgt * a) if k_tgt > 0 else None
    why, px = "time", None
    last = min(e + max_hold, len(c) - 1)
    for k in range(e, last + 1):
        if l[k] <= stop:
            why, px = "stop", stop
            break
        if tgt is not None and h[k] >= tgt:
            why, px = "target", tgt
            break
    if px is None:
        px = float(c[last])
    ret = (px / entry - 1) * 100 - cost
    mae = (entry - float(np.min(l[e:last+1]))) / entry * 100
    return ret, mae, why


# ---------------------------------------------------------------- collect
data = {}
for sym in UNIVERSE:
    df = load(sym)
    if df is None or len(df) < 300:
        continue
    df = df.dropna()
    if len(df) < 300:
        continue
    atr = atr_series(df)
    if atr.dropna().median() > ATR_MAX:
        continue
    o = df["Open"].values.astype(float); h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float);   c = df["Close"].values.astype(float)
    sma50 = pd.Series(c).rolling(50).mean().values
    rng = h - l
    data[sym] = dict(dates=df.index, o=o, h=h, l=l, c=c, sma50=sma50, rng=rng,
                     atr=atr.values)

print(f"symbols loaded: {len(data)}")
if not data:
    sys.exit("no data")

# ---------------------------------------------------------------- baseline
# Same exit applied to EVERY eligible bar = unconditional expectation.
base_rows = []
for sym, d in data.items():
    o, h, l, c, atr = d["o"], d["h"], d["l"], d["c"], d["atr"]
    for j in range(60, len(c) - 1):
        if np.isnan(atr[j]):
            continue
        r = simulate(o, h, l, c, j, atr, 1.0, 1.5, MAX_HOLD, COST)
        if r:
            base_rows.append(r)
base = pd.DataFrame(base_rows, columns=["ret", "mae", "why"])
base_mean = base.ret.mean()
print(f"baseline (all bars, 1.0/1.5 ATR, {MAX_HOLD}-bar cap): n={len(base):,} "
      f"mean={base_mean:+.3f}%  win={100*(base.ret>0).mean():.1f}%  "
      f"median={base.ret.median():+.3f}%\n")

# ---------------------------------------------------------------- rules
rows = []
for sym, d in data.items():
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    sma50, rng, atr = d["sma50"], d["rng"], d["atr"]
    for j in range(60, len(c) - 1):
        if np.isnan(atr[j]):
            continue
        for name in RULES:
            if rule_fires(name, o, h, l, c, j, sma50, rng):
                r = simulate(o, h, l, c, j, atr, 1.0, 1.5, MAX_HOLD, COST)
                if r:
                    rows.append(dict(sym=sym, date=d["dates"][j], rule=name,
                                     ret=r[0], mae=r[1], why=r[2]))

r = pd.DataFrame(rows)
print(f"signals: {len(r):,}  symbols: {r.sym.nunique()}  "
      f"span: {r.date.min().date()} -> {r.date.max().date()}\n")

hdr = f"{'rule':10s} {'n':>6} {'win%':>6} {'mean%':>8} {'excess%':>8} {'t':>6} {'MAE%':>6} {'stop%':>6}"
print(hdr); print("-" * len(hdr))
for name, g in r.groupby("rule"):
    se = g.ret.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
    t = (g.ret.mean() - base_mean) / se if se and se > 0 else np.nan
    print(f"{name:10s} {len(g):>6} {100*(g.ret>0).mean():>6.1f} {g.ret.mean():>+8.2f} "
          f"{g.ret.mean()-base_mean:>+8.2f} {t:>+6.2f} {g.mae.mean():>6.2f} "
          f"{100*(g.why=='stop').mean():>6.1f}")
print("-" * len(hdr))
se = r.ret.std(ddof=1) / np.sqrt(len(r))
print(f"{'ALL':10s} {len(r):>6} {100*(r.ret>0).mean():>6.1f} {r.ret.mean():>+8.2f} "
      f"{r.ret.mean()-base_mean:>+8.2f} {(r.ret.mean()-base_mean)/se:>+6.2f} "
      f"{r.mae.mean():>6.2f} {100*(r.why=='stop').mean():>6.1f}")

# ---------------------------------------------------------------- exit sweep
print("\nexit sweep (pooled signals)")
print(f"{'scheme':22s} {'win%':>6} {'mean%':>8} {'excess%':>8} {'stop%':>6} {'tgt%':>6}")
print("-" * 60)
schemes = [
    ("1.0/1.5 ATR, 5d", 1.0, 1.5, 5),
    ("1.0/3.0 ATR, 10d", 1.0, 3.0, 10),
    ("2.0/3.0 ATR, 10d", 2.0, 3.0, 10),
    ("2.0/6.0 ATR, 20d", 2.0, 6.0, 20),
    ("1.0 ATR, 20d cap", 1.0, 0.0, 20),
    ("no stop, 1d", 99.0, 0.0, 1),
    ("no stop, 5d", 99.0, 0.0, 5),
    ("no stop, 20d", 99.0, 0.0, 20),
]
for label, ks, kt, mh in schemes:
    sub, bsub = [], []
    for sym, d in data.items():
        o, h, l, c, atr = d["o"], d["h"], d["l"], d["c"], d["atr"]
        for j in range(60, len(c) - 1):
            if np.isnan(atr[j]):
                continue
            res = simulate(o, h, l, c, j, atr, ks, kt, mh, COST)
            if res:
                bsub.append(res[0])
        g = r[r.sym == sym]
        for _, row in g.iterrows():
            j = int(np.where(d["dates"] == row["date"])[0][0])
            res = simulate(o, h, l, c, j, atr, ks, kt, mh, COST)
            if res:
                sub.append((res[0], res[2]))
    if not sub:
        continue
    s = pd.DataFrame(sub, columns=["ret", "why"])
    bm = np.mean(bsub) if bsub else np.nan
    print(f"{label:22s} {100*(s.ret>0).mean():>6.1f} {s.ret.mean():>+8.2f} "
          f"{s.ret.mean()-bm:>+8.2f} {100*(s.why=='stop').mean():>6.1f} "
          f"{100*(s.why=='target').mean():>6.1f}")

# ---------------------------------------------------------------- real costs
print(f"\nat real T212 cost ({COST_REAL}% round trip for US shares):")
for name, g in r.groupby("rule"):
    net = g.ret - (COST_REAL - COST)
    print(f"  {name:10s} mean {net.mean():+.2f}%  win {100*(net>0).mean():.1f}%")

r.to_csv(os.path.join(HERE, "daily_signals.csv"), index=False)
print("\nsaved -> daily_signals.csv")
