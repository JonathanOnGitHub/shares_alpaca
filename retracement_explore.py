"""
50% Retracement — Multi-Parameter Exploration Engine
======================================================
Tests 6 distinct strategy variants across a full grid.

Variants tested:
  1. BASE      : entry at 50% level, stop at session extreme, target = N×ATR
  2. CONFIRM   : require bar close BEYOND 50% level before entering (confirmation)
  3. TIGHT_STOP : stop = 0.5× ATR (vs session extreme) — smaller, faster stop
  4. WIDE_TARGET: target = 2× or 3× ATR — let winners run
  5. GAP_FILTER : min_gap ≥ 1.0% only (strong momentum setups)
  6. ATR_GAP    : entry only when gap > 0.5× ATR (gap dwarfs daily noise)
  7. MEANDRIFT  : target = session open (close-to-open mean reversion)
  8. HYBRID     : CONFIRM + TIGHT_STOP combined

Exit logic: bar-by-bar walk from entry to session end.
  STOP   hit first → loss = risk
  TARGET hit first → win  = reward - cost
  TIME (neither) → PnL = (close - entry) direction × notional - cost
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings("ignore")

DATA_DIR  = Path("data/raw")
OUT_DIR   = Path("reports/retracement_explore")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed parameters ─────────────────────────────────────────────────────────
ROUND_TRIP_BPS = 10    # 10 bps round-trip cost
LOOKBACK_BARS  = 2     # front side lookback (first 2 bars)

# ── Variant definitions ──────────────────────────────────────────────────────
# Each variant is a dict of overrides to BASE
VARIANTS = {
    "BASE": {
        "min_gap_pct":    0.3,
        "stop_type":      "session_extreme",  # stop just beyond session H/L
        "stop_atr_mult":  None,
        "target_type":    "atr",               # target = entry ± N×ATR
        "target_atr_mult": 1.0,
        "confirm_entry":  False,               # enter as soon as 50% touched
        "atr_gap_filter": False,              # require gap > 0.5× ATR
    },
    "CONFIRM": {
        "min_gap_pct":    0.3,
        "stop_type":      "session_extreme",
        "stop_atr_mult":  None,
        "target_type":    "atr",
        "target_atr_mult": 1.0,
        "confirm_entry":  True,                # ← KEY DIFFERENCE
        "atr_gap_filter": False,
    },
    "TIGHT_STOP_05": {
        "min_gap_pct":    0.3,
        "stop_type":      "atr",
        "stop_atr_mult":  0.5,                # ← KEY DIFFERENCE
        "target_type":    "atr",
        "target_atr_mult": 1.0,
        "confirm_entry":  False,
        "atr_gap_filter": False,
    },
    "TIGHT_STOP_10": {
        "min_gap_pct":    0.3,
        "stop_type":      "atr",
        "stop_atr_mult":  1.0,
        "target_type":    "atr",
        "target_atr_mult": 1.0,
        "confirm_entry":  False,
        "atr_gap_filter": False,
    },
    "WIDE_TARGET_20": {
        "min_gap_pct":    0.3,
        "stop_type":      "session_extreme",
        "stop_atr_mult":  None,
        "target_type":    "atr",
        "target_atr_mult": 2.0,               # ← KEY DIFFERENCE
        "confirm_entry":  False,
        "atr_gap_filter": False,
    },
    "WIDE_TARGET_30": {
        "min_gap_pct":    0.3,
        "stop_type":      "session_extreme",
        "stop_atr_mult":  None,
        "target_type":    "atr",
        "target_atr_mult": 3.0,               # ← KEY DIFFERENCE
        "confirm_entry":  False,
        "atr_gap_filter": False,
    },
    "GAP_100": {
        "min_gap_pct":    1.0,                # ← KEY DIFFERENCE
        "stop_type":      "session_extreme",
        "stop_atr_mult":  None,
        "target_type":    "atr",
        "target_atr_mult": 1.0,
        "confirm_entry":  False,
        "atr_gap_filter": False,
    },
    "GAP_150": {
        "min_gap_pct":    1.5,                # ← KEY DIFFERENCE
        "stop_type":      "session_extreme",
        "stop_atr_mult":  None,
        "target_type":    "atr",
        "target_atr_mult": 1.0,
        "confirm_entry":  False,
        "atr_gap_filter": False,
    },
    "ATR_GAP": {
        "min_gap_pct":    0.3,
        "stop_type":      "session_extreme",
        "stop_atr_mult":  None,
        "target_type":    "atr",
        "target_atr_mult": 1.0,
        "confirm_entry":  False,
        "atr_gap_filter": True,               # ← KEY DIFFERENCE
    },
    "MEANDRIFT": {
        "min_gap_pct":    0.3,
        "stop_type":      "session_extreme",
        "stop_atr_mult":  None,
        "target_type":    "open",             # ← KEY DIFFERENCE: target = session open
        "target_atr_mult": 0.0,
        "confirm_entry":  False,
        "atr_gap_filter": False,
    },
    "CONFIRM_TIGHT": {
        "min_gap_pct":    0.3,
        "stop_type":      "atr",
        "stop_atr_mult":  0.5,
        "target_type":    "atr",
        "target_atr_mult": 2.0,
        "confirm_entry":  True,
        "atr_gap_filter": False,
    },
    "CONFIRM_WIDE": {
        "min_gap_pct":    0.3,
        "stop_type":      "session_extreme",
        "stop_atr_mult":  None,
        "target_type":    "atr",
        "target_atr_mult": 3.0,
        "confirm_entry":  True,
        "atr_gap_filter": False,
    },
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = (bars["high"] - bars["low"]).fillna(0)
    return tr.rolling(n, min_periods=n).mean()


def load_ticker(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pq.read_table(str(path)).to_pandas()
    df = df.sort_index()
    df = df[df["volume"] > 0]
    return df


def daily_sessions(df: pd.DataFrame):
    sessions = []
    for _, grp in df.groupby(df.index.normalize()):
        grp = grp.sort_index()
        if len(grp) < 4:
            continue
        sessions.append(grp)
    return sessions


def simulate_exit(bars, entry_bar, entry_price, stop_price, target_price, direction):
    """Walk forward; return (exit_type, exit_price)."""
    n = len(bars)
    for i in range(entry_bar + 1, n):
        h, lo = bars["high"].iloc[i], bars["low"].iloc[i]
        if direction == "SHORT":
            if h >= stop_price:
                return ("STOP", stop_price, i)
            if lo <= target_price:
                return ("TARGET", target_price, i)
        else:
            if lo <= stop_price:
                return ("STOP", stop_price, i)
            if h >= target_price:
                return ("TARGET", target_price, i)
    return ("TIME", bars["close"].iloc[-1], n - 1)


def run_ticker(ticker: str, params: dict) -> pd.DataFrame:
    df = load_ticker(ticker)
    if df.empty:
        return pd.DataFrame()

    sessions = daily_sessions(df)
    trades = []

    for sess in sessions:
        bars = sess.sort_index().copy()
        session_date = pd.Timestamp(bars.index[0].date())
        n = len(bars)
        if n < LOOKBACK_BARS + 2:
            continue

        open_price   = bars["open"].iloc[0]
        front        = bars.iloc[:LOOKBACK_BARS]
        session_high = front["high"].max()
        session_low  = front["low"].min()
        atr_now      = compute_atr(bars.iloc[:LOOKBACK_BARS], 14).iloc[-1]
        atr_now      = atr_now if atr_now > 0 else open_price * 0.005

        gap_up   = (session_high - open_price) / open_price * 100
        gap_down = (open_price - session_low)  / open_price * 100

        # ATR-gap filter: require gap ≥ 0.5× ATR (gap must dominate daily noise)
        if params["atr_gap_filter"]:
            gap_size = max(gap_up, gap_down)
            if gap_size * open_price / 100 < 0.5 * atr_now:
                continue

        # ── SHORT setup ─────────────────────────────────────────────────────
        if gap_up >= params["min_gap_pct"]:
            retracement = open_price + 0.50 * (session_high - open_price)

            # CONFIRM variant: need bar to close beyond 50% level first
            entry_bar = None
            if params["confirm_entry"]:
                for i in range(LOOKBACK_BARS, n):
                    lo, hi, cl = bars["low"].iloc[i], bars["high"].iloc[i], bars["close"].iloc[i]
                    if lo <= retracement <= hi:
                        # bar must close BELOW 50% level to confirm SHORT
                        if cl < retracement:
                            entry_bar = i
                            break
            else:
                for i in range(LOOKBACK_BARS, n):
                    if bars["low"].iloc[i] <= retracement <= bars["high"].iloc[i]:
                        entry_bar = i
                        break

            if entry_bar is None:
                continue

            # Stop
            if params["stop_type"] == "atr":
                stop_price = retracement + params["stop_atr_mult"] * atr_now
            else:
                stop_price = session_high * 1.002

            # Target
            if params["target_type"] == "atr":
                target_price = retracement - params["target_atr_mult"] * atr_now
            else:  # "open"
                target_price = open_price

            result, exit_price, _ = simulate_exit(
                bars, entry_bar, retracement, stop_price, target_price, "SHORT"
            )
            risk = abs(retracement - stop_price)
            if result == "TARGET":
                reward = abs(retracement - exit_price)
                pnl = (reward - risk) / retracement * 10_000 - ROUND_TRIP_BPS
            elif result == "STOP":
                pnl = (-risk / retracement * 10_000) - ROUND_TRIP_BPS
            else:  # TIME
                pnl = (retracement - exit_price) / retracement * 10_000 - ROUND_TRIP_BPS

            trades.append({
                "ticker":       ticker,
                "date":         session_date,
                "direction":    "SHORT",
                "gap_pct":      gap_up,
                "atr":          atr_now,
                "exit_type":    result,
                "pnl_bps":      pnl,
                "result":       "WIN" if pnl > 0 else "LOSS",
            })

        # ── LONG setup ─────────────────────────────────────────────────────
        if gap_down >= params["min_gap_pct"]:
            retracement = open_price - 0.50 * (open_price - session_low)

            entry_bar = None
            if params["confirm_entry"]:
                for i in range(LOOKBACK_BARS, n):
                    lo, hi, cl = bars["low"].iloc[i], bars["high"].iloc[i], bars["close"].iloc[i]
                    if lo <= retracement <= hi:
                        if cl > retracement:   # bar closes ABOVE 50% → confirm LONG
                            entry_bar = i
                            break
            else:
                for i in range(LOOKBACK_BARS, n):
                    if bars["low"].iloc[i] <= retracement <= bars["high"].iloc[i]:
                        entry_bar = i
                        break

            if entry_bar is None:
                continue

            if params["stop_type"] == "atr":
                stop_price = retracement - params["stop_atr_mult"] * atr_now
            else:
                stop_price = session_low * 0.998

            if params["target_type"] == "atr":
                target_price = retracement + params["target_atr_mult"] * atr_now
            else:
                target_price = open_price

            result, exit_price, _ = simulate_exit(
                bars, entry_bar, retracement, stop_price, target_price, "LONG"
            )
            risk = abs(retracement - stop_price)
            if result == "TARGET":
                reward = abs(exit_price - retracement)
                pnl = (reward - risk) / retracement * 10_000 - ROUND_TRIP_BPS
            elif result == "STOP":
                pnl = (-risk / retracement * 10_000) - ROUND_TRIP_BPS
            else:
                pnl = (exit_price - retracement) / retracement * 10_000 - ROUND_TRIP_BPS

            trades.append({
                "ticker":       ticker,
                "date":         session_date,
                "direction":    "LONG",
                "gap_pct":      gap_down,
                "atr":          atr_now,
                "exit_type":    result,
                "pnl_bps":      pnl,
                "result":       "WIN" if pnl > 0 else "LOSS",
            })

    return pd.DataFrame(trades)


def aggregate(trades: pd.DataFrame) -> dict:
    """Return a flat dict of summary stats for a trades DataFrame."""
    if trades.empty:
        return {}
    n   = len(trades)
    wr  = (trades["result"] == "WIN").mean()
    avg = trades["pnl_bps"].mean()
    med = trades["pnl_bps"].median()
    std = trades["pnl_bps"].std()
    sharpe = avg / (std + 1e-9) * np.sqrt(n) if std > 0 else 0

    exit_tab = trades.groupby("exit_type")["pnl_bps"].agg(["count","mean"]).to_dict("index")

    return {
        "n_trades":    n,
        "win_rate":    wr,
        "avg_pnl_bps": avg,
        "median_pnl":  med,
        "std_pnl":     std,
        "sharpe_like": sharpe,
        "exit_stop_n":    exit_tab.get("STOP",   {}).get("count", 0),
        "exit_stop_avg":  exit_tab.get("STOP",   {}).get("mean",  0),
        "exit_target_n":  exit_tab.get("TARGET", {}).get("count", 0),
        "exit_target_avg":exit_tab.get("TARGET",{}).get("mean",  0),
        "exit_time_n":    exit_tab.get("TIME",   {}).get("count", 0),
        "exit_time_avg":  exit_tab.get("TIME",   {}).get("mean",  0),
    }


def main():
    print("Loading tickers...")
    tickers = [p.stem for p in DATA_DIR.glob("*.parquet")]
    print(f"  {len(tickers)} tickers found")

    all_results = []
    all_trades  = []

    total = len(VARIANTS) * len(tickers)
    done  = 0

    print(f"\nRunning: {len(VARIANTS)} variants × {len(tickers)} tickers")
    print("=" * 70)

    for vname, vparams in VARIANTS.items():
        var_trades = []
        for ticker in tickers:
            tdf = run_ticker(ticker, vparams)
            if not tdf.empty:
                tdf["variant"] = vname
                var_trades.append(tdf)
            done += 1

        if var_trades:
            var_df = pd.concat(var_trades, ignore_index=True)
            all_trades.append(var_df)
            stats = aggregate(var_df)
            stats["variant"] = vname
            all_results.append(stats)

        pct = 100 * done / total
        if done % 87 == 0:
            print(f"  {done}/{total} ({pct:.0f}%) — {vname} done")

    results_df = pd.DataFrame(all_results)
    all_trades = pd.concat(all_trades, ignore_index=True)

    # Sort by avg_pnl_bps descending
    results_df = results_df.sort_values("avg_pnl_bps", ascending=False)

    # Print results table
    print("\n" + "=" * 80)
    print("ALL VARIANTS — sorted by avg_pnl_bps")
    print("=" * 80)
    cols = ["variant","direction","n_trades","win_rate","avg_pnl_bps","median_pnl",
            "sharpe_like","exit_stop_avg","exit_target_avg","exit_time_avg"]
    # direction is NaN in aggregate — it's overall; add it
    results_df["direction"] = "BOTH"
    print(results_df[["variant","n_trades","win_rate","avg_pnl_bps","median_pnl",
                      "sharpe_like","exit_stop_avg","exit_target_avg","exit_time_avg"]].to_string(index=False))

    # Per-direction breakdown for top variants
    print("\n" + "=" * 80)
    print("PER-DIRECTION BREAKDOWN (top 5 variants)")
    print("=" * 80)
    top5 = results_df["variant"].head(5).tolist()
    dir_results = []
    for vname in VARIANTS:
        vdf = all_trades[all_trades["variant"] == vname]
        for direction in ["SHORT", "LONG"]:
            s = vdf[vdf["direction"] == direction]
            if s.empty:
                continue
            dir_results.append({
                "variant":    vname,
                "direction": direction,
                "n_trades":  len(s),
                "win_rate":  (s["result"]=="WIN").mean(),
                "avg_pnl":   s["pnl_bps"].mean(),
                "median":    s["pnl_bps"].median(),
                "sharpe":    s["pnl_bps"].mean()/(s["pnl_bps"].std()+1e-9)*np.sqrt(len(s)),
                "stop_avg":  s[s["exit_type"]=="STOP"]["pnl_bps"].mean() if "STOP" in s["exit_type"].values else float("nan"),
                "target_avg":s[s["exit_type"]=="TARGET"]["pnl_bps"].mean() if "TARGET" in s["exit_type"].values else float("nan"),
                "time_avg":  s[s["exit_type"]=="TIME"]["pnl_bps"].mean() if "TIME" in s["exit_type"].values else float("nan"),
            })
    dir_df = pd.DataFrame(dir_results)
    for vname in top5:
        vdir = dir_df[dir_df["variant"] == vname]
        print(f"\n  {vname}")
        print(vdir[["direction","n_trades","win_rate","avg_pnl","median","sharpe",
                    "stop_avg","target_avg","time_avg"]].to_string(index=False))

    # Best tickers for best variant
    best_variant = results_df.iloc[0]["variant"]
    print("\n" + "=" * 80)
    print(f"BEST VARIANT: {best_variant}")
    print("=" * 80)
    best_df = all_trades[all_trades["variant"] == best_variant]
    ticker_best = best_df.groupby(["ticker","direction"]).agg(
        n=("pnl_bps","count"),
        wr=("result", lambda x: (x=="WIN").mean()),
        avg=("pnl_bps","mean"),
        med=("pnl_bps","median"),
    ).reset_index()
    ticker_best = ticker_best[ticker_best["n"] >= 20].sort_values("avg", ascending=False)
    print(ticker_best.head(20).to_string(index=False))

    # Sensitivity: min_gap sweep for BASE variant
    print("\n" + "=" * 80)
    print("GAP SIZE SENSITIVITY (BASE variant, both directions)")
    print("=" * 80)
    base_df = all_trades[all_trades["variant"] == "BASE"]
    for gap in [0.3, 0.5, 0.75, 1.0, 1.5]:
        g = base_df[base_df["gap_pct"] >= gap]
        if g.empty:
            continue
        print(f"  gap ≥ {gap}%: n={len(g)}, WR={(g['result']=='WIN').mean():.1%}, avg={g['pnl_bps'].mean():.2f} bps, median={g['pnl_bps'].median():.2f}")

    # Save
    results_df.to_csv(OUT_DIR / "variant_summary.csv", index=False)
    all_trades.to_csv(OUT_DIR / "all_trades.csv", index=False)
    dir_df.to_csv(OUT_DIR / "direction_summary.csv", index=False)

    # Write markdown report
    best_row = results_df.iloc[0]
    report = f"""# 50% Retracement — Strategy Exploration Report

Generated: {pd.Timestamp.now().date()}
Data: FTSE 100, hourly bars (07:00–15:00), ~May 2025–Sep 2026, {len(tickers)} tickers

## Variants Tested

| Variant | Key Change | Stop | Target |
|---|---|---|---|
| BASE | Baseline: enter at 50% touch, session-extreme stop | Session H/L ×1.002 | 1× ATR |
| CONFIRM | Require bar close beyond 50% level before entry | Session H/L ×1.002 | 1× ATR |
| TIGHT_STOP_05 | Stop = 0.5× ATR | 0.5× ATR | 1× ATR |
| TIGHT_STOP_10 | Stop = 1.0× ATR | 1.0× ATR | 1× ATR |
| WIDE_TARGET_20 | Target = 2× ATR | Session H/L ×1.002 | 2× ATR |
| WIDE_TARGET_30 | Target = 3× ATR | Session H/L ×1.002 | 3× ATR |
| GAP_100 | min_gap = 1.0% | Session H/L ×1.002 | 1× ATR |
| GAP_150 | min_gap = 1.5% | Session H/L ×1.002 | 1× ATR |
| ATR_GAP | Require gap > 0.5× ATR | Session H/L ×1.002 | 1× ATR |
| MEANDRIFT | Target = session open | Session H/L ×1.002 | Session open |
| CONFIRM_TIGHT | CONFIRM + TIGHT_STOP_05 + WIDE_TARGET_20 | 0.5× ATR | 2× ATR |
| CONFIRM_WIDE | CONFIRM + WIDE_TARGET_30 | Session H/L ×1.002 | 3× ATR |

## Overall Results (both directions)

| Variant | n_trades | win_rate | avg_pnl_bps | median_pnl | sharpe_like | stop_avg | target_avg | time_avg |
|---|---|---|---|---|---|---|---|---|
"""
    for _, row in results_df.iterrows():
        report += f"| {row['variant']} | {row['n_trades']} | {row['win_rate']:.1%} | **{row['avg_pnl_bps']:.2f}** | {row['median_pnl']:.2f} | {row['sharpe_like']:.1f} | {row['exit_stop_avg']:.1f} | {row['exit_target_avg']:.1f} | {row['exit_time_avg']:.2f} |\n"

    report += f"""
## Best Variant: {best_variant}
- **avg_pnl_bps:** {best_row['avg_pnl_bps']:.2f}
- **win_rate:** {best_row['win_rate']:.1%}
- **n_trades:** {best_row['n_trades']}
- **median_pnl:** {best_row['median_pnl']:.2f}
- **sharpe_like:** {best_row['sharpe_like']:.1f}

## Gap Size Sensitivity (BASE)
| min_gap | n_trades | win_rate | avg_pnl_bps | median_pnl |
|---|---|---|---|---|
"""
    base_df = all_trades[all_trades["variant"] == "BASE"]
    for gap in [0.3, 0.5, 0.75, 1.0, 1.5]:
        g = base_df[base_df["gap_pct"] >= gap]
        if g.empty:
            continue
        report += f"| ≥{gap}% | {len(g)} | {(g['result']=='WIN').mean():.1%} | {g['pnl_bps'].mean():.2f} | {g['pnl_bps'].median():.2f} |\n"

    report += """
## Interpretation Guide
- **stop_avg**: average PnL when the stop was hit first — how much you lose on momentumContinuation
- **target_avg**: average PnL when the reversal target was hit — how much you gain on successful mean-reversion
- **time_avg**: average PnL when neither hit by session close — residual drift P&L
- **sharpe_like**: mean(std) × √n — useful for ranking but not a true Sharpe ratio
"""
    (OUT_DIR / "EXPLORATION_REPORT.md").write_text(report)
    print(f"\n✓ Saved to {OUT_DIR}/")
    print(f"  variant_summary.csv")
    print(f"  direction_summary.csv")
    print(f"  all_trades.csv ({len(all_trades)} trades)")
    print(f"  EXPLORATION_REPORT.md")


if __name__ == "__main__":
    main()
