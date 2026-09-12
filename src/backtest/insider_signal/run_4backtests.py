"""
Fast 4-config backtest: batch price fetch, threaded scoring.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Load all 4 signal files ───────────────────────────────────────────────────
configs = {
    "A": pd.read_csv("signal_history_A_baseline.csv"),
    "B": pd.read_csv("signal_history_B_valuewt.csv"),
    "C": pd.read_csv("signal_history_C_1qtr.csv"),
    "D": pd.read_csv("signal_history_D_both.csv"),
}
rebalances = sorted(configs["A"]["rebalance_date"].unique())
print(f"Rebalances: {len(rebalances)}, Configs: A/B/C/D")

# ── Pre-fetch all SPY returns once ────────────────────────────────────────────
def period_dates(rd):
    rd_date  = pd.to_datetime(rd)
    next_rd = rd_date + pd.DateOffset(months=3)
    hold_end = next_rd - pd.Timedelta(days=1)
    return rd_date, hold_end

spy_rets = {}
for rd in rebalances:
    rd_date, hold_end = period_dates(rd)
    try:
        d = yf.download("SPY", start=rd_date, end=hold_end, progress=False, timeout=15)
        d = d["Close"].squeeze() if d["Close"].ndim > 1 else d["Close"]
        spy_rets[rd] = float(d.iloc[-1] / d.iloc[0] - 1) if len(d) >= 5 else np.nan
    except Exception:
        spy_rets[rd] = np.nan
print(f"SPY returns cached: {sum(not np.isnan(v) for v in spy_rets.values())}/{len(rebalances)}")

# ── Collect all unique ticker × period pairs across ALL configs ───────────────
ticker_periods = {}  # ticker → [period dates]
for cfg_df in configs.values():
    for _, row in cfg_df.iterrows():
        ticker_periods.setdefault(row["ticker"], set()).add(row["rebalance_date"])

all_pairs = []
for ticker, periods in ticker_periods.items():
    for rd in periods:
        all_pairs.append((ticker, rd))
print(f"Total ticker-period pairs to fetch: {len(all_pairs)}")

# ── Batch fetch returns ────────────────────────────────────────────────────────
# Fetch up to 100 tickers at once per request (yfinance batch support)
returns = {}  # (ticker, rd) → return

def fetch_pair(ticker_rd):
    ticker, rd = ticker_rd
    rd_date, hold_end = period_dates(rd)
    try:
        d = yf.download(ticker, start=rd_date, end=hold_end, progress=False, timeout=15)
        if len(d) < 5:
            return ticker, rd, np.nan
        col = d["Close"].squeeze() if d["Close"].ndim > 1 else d["Close"]
        return ticker, rd, float(col.iloc[-1] / col.iloc[0] - 1)
    except Exception:
        return ticker, rd, np.nan

BATCH = 50
with ThreadPoolExecutor(max_workers=20) as ex:
    futures = {ex.submit(fetch_pair, pr): pr for pr in all_pairs}
    done = 0
    for fut in as_completed(futures):
        ticker, rd, ret = fut.result()
        returns[(ticker, rd)] = ret
        done += 1
        if done % 200 == 0:
            print(f"  Downloaded {done}/{len(all_pairs)} ({done/len(all_pairs)*100:.0f}%)")

print(f"Price cache built: {len(returns)} entries, {sum(np.isnan(v) for v in returns.values())} NaN")

# ── Run backtest for each config ─────────────────────────────────────────────
def run_backtest(label, df):
    rows = []
    for rd in rebalances:
        q5 = df[(df["rebalance_date"] == rd) & (df["quintile"] == 5)]
        q1 = df[(df["rebalance_date"] == rd) & (df["quintile"] == 1)]
        if q5.empty or q1.empty:
            continue

        r5 = np.array([returns.get((t, rd), np.nan) for t in q5["ticker"]])
        r1 = np.array([returns.get((t, rd), np.nan) for t in q1["ticker"]])

        q5_mean = float(np.nanmean(r5))
        q1_mean = float(np.nanmean(r1))
        spread  = q5_mean - q1_mean
        spy_ret = spy_rets.get(rd, np.nan)
        alpha   = q5_mean - spy_ret if not np.isnan(spy_ret) else np.nan
        pos_rate = float(np.mean(r5 > 0))

        rows.append({
            "period": rd, "n_q5": len(r5), "n_q1": len(r1),
            "q5_mean": q5_mean, "q1_mean": q1_mean,
            "spread": spread, "spy": spy_ret, "alpha": alpha,
        })

    res = pd.DataFrame(rows)
    ann_mult = 4
    total_ret = float((1 + res["q5_mean"]).prod() - 1)
    ann_ret   = float((1 + total_ret) ** ann_mult - 1)
    ann_vol   = float(res["q5_mean"].std() * np.sqrt(ann_mult))
    sharpe    = (ann_ret - 0.05) / ann_vol if ann_vol > 0 else 0.0
    win_rate  = float(res["q5_mean"].gt(0).mean())
    avg_spread = float(res["spread"].mean())

    return {
        "label": label, "n_periods": len(res),
        "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
        "win_rate": win_rate, "avg_spread": avg_spread,
        "total_ret": total_ret, "df": res,
    }

print("\n" + "=" * 75)
print(" BACKTEST RESULTS — 4 CONFIGURATIONS")
print("=" * 75)

results = {}
for cfg_name, df in configs.items():
    label_map = {
        "A": "A: Baseline (365d, sz=0.2, rec=0.3, conv=0.5)",
        "B": "B: Value-weighted (365d, sz=0.7, rec=0.1, conv=0.2)",
        "C": "C: 1Q lookback (90d, sz=0.2, rec=0.3, conv=0.5)",
        "D": "D: 1Q+Value-wt (90d, sz=0.7, rec=0.1, conv=0.2)",
    }
    stats = run_backtest(label_map[cfg_name], df)
    results[cfg_name] = stats
    print(f"\n[{stats['label']}]")
    print(f"  Periods:     {stats['n_periods']}")
    print(f"  Ann. return: {stats['ann_ret']*100:+.1f}%")
    print(f"  Ann. vol:    {stats['ann_vol']*100:+.1f}%")
    print(f"  Sharpe:      {stats['sharpe']:+.2f}")
    print(f"  Win rate:    {stats['win_rate']*100:.0f}%")
    print(f"  Avg spread:  {stats['avg_spread']*100:+.2f}%/qtr")
    print(f"  Total ret:   {stats['total_ret']*100:+.1f}%")

# Comparison table
print("\n" + "=" * 75)
print(" COMPARISON TABLE")
print("=" * 75)
print(f"{'Config':<50} {'Ann Ret':>8} {'Ann Vol':>8} {'Sharpe':>7} {'Win%':>6} {'AvgSprd':>9}")
print("-" * 75)
for cfg_name, stats in results.items():
    print(f"{stats['label']:<50} {stats['ann_ret']*100:>+7.1f}% {stats['ann_vol']*100:>7.1f}% {stats['sharpe']:>+7.2f} {stats['win_rate']*100:>5.0f}% {stats['avg_spread']*100:>+8.2f}%")

# Period-by-period table
print("\n Quarterly Q5 returns by period:")
print("-" * 75)
header = f"{'Period':<12}" + "".join([f"{c:>9}" for c in results.keys()])
print(header)
for rd in rebalances:
    row = f"{rd:<12}"
    for cfg_name in results.keys():
        df = results[cfg_name]["df"]
        m = df[df["period"] == rd]
        if not m.empty and not np.isnan(m["q5_mean"].iloc[0]):
            row += f"{m['q5_mean'].iloc[0]*100:>9.1f}%"
        else:
            row += f"{'—':>9}"
    print(row)
