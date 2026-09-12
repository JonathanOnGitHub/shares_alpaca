"""
Proper comparative analysis of all 4 CEO insider purchase signal configs.
Computes: equity curves vs SPY, proper annualised metrics, spread stats, t-tests.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

BASE = "/home/burley/Personal/Trading/insider_basket"
REBALANCES = [
    "2021-10-01","2022-01-01","2022-04-01","2022-07-01","2022-10-01",
    "2023-01-01","2023-04-01","2023-07-01","2023-10-01","2024-01-01",
    "2024-04-01","2024-07-01","2024-10-01","2025-01-01","2025-04-01",
    "2025-07-01","2025-10-01","2026-01-01","2026-04-01","2026-07-01",
]

# ── Load all 4 signal configs ──────────────────────────────────────────────────
configs = {
    "A_baseline":  pd.read_csv(f"{BASE}/signal_history_A_baseline.csv"),
    "B_valuewt":   pd.read_csv(f"{BASE}/signal_history_B_valuewt.csv"),
    "C_1qtr":      pd.read_csv(f"{BASE}/signal_history_C_1qtr.csv"),
    "D_both":      pd.read_csv(f"{BASE}/signal_history_D_both.csv"),
}

# ── Fetch SPY benchmark ────────────────────────────────────────────────────────
def period_bounds(rd):
    rd_date  = pd.to_datetime(rd)
    next_rd  = rd_date + pd.DateOffset(months=3)
    end      = next_rd - pd.Timedelta(days=1)
    return rd_date, end

print("Fetching SPY benchmark…")
spy_returns = {}
for rd in REBALANCES:
    d, e = period_bounds(rd)
    try:
        prices_raw = yf.download("SPY", start=d, end=e, progress=False, timeout=15)
        if isinstance(prices_raw.columns, pd.MultiIndex):
            col = prices_raw[("Close", "SPY")]
        else:
            col = prices_raw["Close"]
        col = col.squeeze()
        ret = float(col.iloc[-1] / col.iloc[0] - 1)   # decimal
        spy_returns[rd] = ret
    except Exception as ex:
        spy_returns[rd] = np.nan
        print(f"  SPY failed for {rd}: {ex}")

# ── Build per-config returns ────────────────────────────────────────────────────
def compute_config_returns(df, label):
    rows = []
    for rd in REBALANCES:
        q5 = df[(df["rebalance_date"] == rd) & (df["quintile"] == 5)]
        q1 = df[(df["rebalance_date"] == rd) & (df["quintile"] == 1)]
        if q5.empty or q1.empty:
            continue
        rows.append({
            "period":   rd,
            "n_q5":     len(q5),
            "n_q1":     len(q1),
            "q5_ret":   np.nan,   # placeholder – filled below
            "q1_ret":   np.nan,
            "spread":   np.nan,
            "spy_ret":  spy_returns.get(rd, np.nan),
            "alpha":    np.nan,
        })
    res = pd.DataFrame(rows)
    return res

# Re-run the price fetch logic from run_4backtests.py to get ticker returns
# (We rebuild the same return dictionary to ensure consistency)
print("Fetching ticker prices…")
ticker_periods = {}
for cfg_df in configs.values():
    for _, row in cfg_df.iterrows():
        ticker_periods.setdefault(row["ticker"], set()).add(row["rebalance_date"])

all_pairs = [(t, rd) for t, rds in ticker_periods.items() for rd in rds]

from concurrent.futures import ThreadPoolExecutor
def fetch_pair(ticker_rd):
    ticker, rd = ticker_rd
    d, e = period_bounds(rd)
    try:
        data = yf.download(ticker, start=d, end=e, progress=False, timeout=15)
        if len(data) < 5:
            return ticker, rd, np.nan
        col = data["Close"].squeeze()
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return ticker, rd, float(col.iloc[-1] / col.iloc[0] - 1)
    except Exception:
        return ticker, rd, np.nan

returns = {}
with ThreadPoolExecutor(max_workers=20) as ex:
    futures = {ex.submit(fetch_pair, pr): pr for pr in all_pairs}
    done = 0
    for fut in futures:
        t, rd, ret = fut.result()
        returns[(t, rd)] = ret
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(all_pairs)}")

print(f"  Price cache: {len(returns)} entries")

# ── Compute returns for each config ───────────────────────────────────────────
def fill_returns(cfg_df):
    rows = []
    for rd in REBALANCES:
        q5 = cfg_df[(cfg_df["rebalance_date"] == rd) & (cfg_df["quintile"] == 5)]
        q1 = cfg_df[(cfg_df["rebalance_date"] == rd) & (cfg_df["quintile"] == 1)]
        if q5.empty or q1.empty:
            continue
        r5 = np.array([returns.get((t, rd), np.nan) for t in q5["ticker"]])
        r1 = np.array([returns.get((t, rd), np.nan) for t in q1["ticker"]])
        q5_mean = float(np.nanmean(r5))
        q1_mean = float(np.nanmean(r1))
        spy_ret = spy_returns.get(rd, np.nan)
        rows.append({
            "period":   rd,
            "n_q5":     len(r5),
            "n_q1":     len(r1),
            "q5_ret":   q5_mean,      # decimal
            "q1_ret":   q1_mean,
            "spread":   q5_mean - q1_mean,
            "spy_ret":  spy_ret,      # decimal
            "alpha":    q5_mean - spy_ret if not np.isnan(spy_ret) else np.nan,
        })
    return pd.DataFrame(rows)

results = {}
for name, df in configs.items():
    results[name] = fill_returns(df)

# ── Equity curves ────────────────────────────────────────────────────────────────
def equity_curve(returns_series, start=100_000):
    """Compound returns starting from $100k. returns_series in decimal form."""
    curve = [start]
    for r in returns_series:
        curve.append(curve[-1] * (1 + r))
    return curve

spy_curve = equity_curve([spy_returns.get(rd, np.nan) for rd in REBALANCES])

# ── Print results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(" CEO INSIDER PURCHASE SIGNALS — 4-CONFIG COMPARATIVE ANALYSIS")
print(" 20 quarters: 2021-10-01 → 2026-07-01 | ~63-day hold | SPY benchmark")
print("=" * 78)

cfg_labels = {
    "A_baseline": "A: Baseline  (365d, sz=0.2, rec=0.3, conv=0.5)",
    "B_valuewt":  "B: Val-wt    (365d, sz=0.7, rec=0.1, conv=0.2)",
    "C_1qtr":     "C: 1Q lookbk  (90d,  sz=0.2, rec=0.3, conv=0.5)",
    "D_both":     "D: 1Q+Val-wt  (90d,  sz=0.7, rec=0.1, conv=0.2)",
}

# Summary table
print(f"\n{'Config':<40} {'Q5 ann.ret':>10} {'Q5 tot':>8} {'SPY tot':>8} {'Q5-SP Alpha':>11} {'AvgSprd':>9} {'Sharpe':>7}")
print("-" * 100)
for name, res in results.items():
    # Average quarterly return (decimal → %)
    avg_q = res["q5_ret"].mean() * 100
    # Total return: compound from $100k
    ec = equity_curve(res["q5_ret"].tolist())
    total_ret = (ec[-1] / ec[0] - 1) * 100
    # Annualised: compound avg quarterly return × 4
    ann_ret = ((1 + res["q5_ret"].mean()) ** 4 - 1) * 100
    # SPY total
    spy_ec = equity_curve([spy_returns.get(rd, np.nan) for rd in res["period"]])
    spy_tot = (spy_ec[-1] / spy_ec[0] - 1) * 100
    # Alpha
    avg_alpha = res["alpha"].mean() * 100
    # Spread
    avg_spread = res["spread"].mean() * 100
    # Sharpe (ann. return - 5% / ann. vol); ann.vol = std(q_ret) * sqrt(4)
    ann_vol = res["q5_ret"].std() * np.sqrt(4) * 100
    sharpe = (ann_ret - 5) / ann_vol if ann_vol > 0 else 0.0

    print(f"{cfg_labels[name]:<40} {ann_ret:>+9.1f}% {total_ret:>+7.1f}% {spy_tot:>+7.1f}% {avg_alpha:>+10.1f}% {avg_spread:>+8.2f}% {sharpe:>+7.2f}")

print("-" * 100)

# SPY row
spy_ec = equity_curve([spy_returns.get(rd, np.nan) for rd in REBALANCES])
spy_tot = (spy_ec[-1] / spy_ec[0] - 1) * 100
spy_avg_q = np.mean([spy_returns.get(rd, 0) for rd in REBALANCES]) * 100
spy_ann_ret = ((1 + spy_avg_q/100) ** 4 - 1) * 100
print(f"{'SPY buy-hold':<40} {spy_ann_ret:>+9.1f}% {spy_tot:>+7.1f}% {'—':>7} {'—':>10} {'—':>9} {'—':>7}")

# ── Per-quarter table ──────────────────────────────────────────────────────────
print("\n QUARTERLY Q5 RETURNS (%)\n")
print(f"{'Period':<12} {'A':>7} {'B':>7} {'C':>7} {'D':>7} {'SPY':>7}")
print("-" * 50)
for i, rd in enumerate(REBALANCES):
    row = [rd]
    for name in ["A_baseline", "B_valuewt", "C_1qtr", "D_both"]:
        r = results[name]
        m = r[r["period"] == rd]
        row.append(f"{m['q5_ret'].iloc[0]*100:>+6.1f}%" if not m.empty else "     —")
    spy_r = spy_returns.get(rd, np.nan)
    row.append(f"{spy_r*100:>+6.1f}%" if not np.isnan(spy_r) else "     —")
    print(f"{row[0]:<12} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]}")

# ── Statistical tests ──────────────────────────────────────────────────────────
print("\n STATISTICAL TESTS (spread = Q5 − Q1, decimal form)\n")
from scipy import stats as scipy_stats
for name in ["A_baseline", "B_valuewt", "C_1qtr", "D_both"]:
    r = results[name]["spread"].dropna() * 100   # convert to %
    cfg_short = name.split("_")[0]
    mean_s = r.mean()
    std_s  = r.std()
    t, p   = scipy_stats.ttest_1samp(r, 0)
    ci_lo  = mean_s - 1.96 * std_s / np.sqrt(len(r))
    ci_hi  = mean_s + 1.96 * std_s / np.sqrt(len(r))
    pos_pct = (r > 0).mean() * 100
    print(f"  {cfg_short}: mean={mean_s:+.2f}%/q  std={std_s:.2f}%  t={t:+.2f}  p={p:.3f}  "
          f"95%CI=[{ci_lo:+.2f}%, {ci_hi:+.2f}%]  spread>0: {pos_pct:.0f}%")

# ── Equity curve final values ──────────────────────────────────────────────────
print("\n EQUITY CURVE FINAL VALUES ($100k start)\n")
spy_ec = equity_curve([spy_returns.get(rd, np.nan) for rd in REBALANCES])
spy_f = spy_ec[-1]
print(f"  SPY buy-hold:  ${spy_f:,.0f}  {(spy_f/100_000-1)*100:+.1f}%")
for name in ["A_baseline", "B_valuewt", "C_1qtr", "D_both"]:
    r = results[name]
    ec = equity_curve(r["q5_ret"].tolist())
    diff = ec[-1] - spy_f
    cfg_short = name.split("_")[0]
    print(f"  Config {cfg_short}:        ${ec[-1]:,.0f}  {(ec[-1]/100_000-1)*100:+.1f}%  vs SPY {diff:+.0f}")
