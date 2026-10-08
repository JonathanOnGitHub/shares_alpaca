"""
Walk-Forward Validation: 50% Retracement Strategy
==================================================
Train: 2025 (full year)
Test:  H1 2026 (Jan–Jun 2026)

Procedure:
  1. Train period: sweep stop_atr_mult × target_atr_mult grid
  2. Select top-3 parameter combos by Sharpe-like on train
  3. Freeze those params and evaluate out-of-sample on H1 2026
  4. Compare train vs test to detect overfitting
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/raw")
OUT_DIR  = Path("reports/walkforward")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUND_TRIP_BPS = 10
LOOKBACK_BARS  = 2

# ── Parameter grid to sweep ───────────────────────────────────────────────────
STOP_ATR_GRID     = [0.3, 0.5, 0.75, 1.0, 1.5]
TARGET_ATR_GRID    = [1.0, 1.5, 2.0, 2.5, 3.0]
CONFIRM_ENTRY_GRID = [True, False]   # include both for comparison

# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_atr(bars, n=14):
    tr = (bars["high"] - bars["low"]).fillna(0)
    return tr.rolling(n, min_periods=n).mean()


def load_ticker(ticker):
    path = DATA_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pq.read_table(str(path)).to_pandas()
    df = df.sort_index()
    df = df[df["volume"] > 0]
    return df


def daily_sessions(df):
    sessions = []
    for _, grp in df.groupby(df.index.normalize()):
        grp = grp.sort_index()
        if len(grp) < 4:
            continue
        sessions.append(grp)
    return sessions


def simulate_exit(bars, entry_bar, entry_price, stop_price, target_price, direction):
    n = len(bars)
    for i in range(entry_bar + 1, n):
        h, lo = bars["high"].iloc[i], bars["low"].iloc[i]
        if direction == "SHORT":
            if h >= stop_price:
                return "STOP", stop_price
            if lo <= target_price:
                return "TARGET", target_price
        else:
            if lo <= stop_price:
                return "STOP", stop_price
            if h >= target_price:
                return "TARGET", target_price
    return "TIME", bars["close"].iloc[-1]


def run_backtest(ticker, params, train=True):
    """
    params: dict with keys
      confirm_entry, stop_atr_mult, target_atr_mult, start_date, end_date
    Returns DataFrame of trades.
    """
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

        # Date filter
        if params.get("start_date") and session_date < params["start_date"]:
            continue
        if params.get("end_date") and session_date >= params["end_date"]:
            continue

        open_price    = bars["open"].iloc[0]
        front         = bars.iloc[:LOOKBACK_BARS]
        session_high  = front["high"].max()
        session_low   = front["low"].min()
        atr_now       = compute_atr(bars.iloc[:LOOKBACK_BARS], 14).iloc[-1]
        atr_now       = atr_now if atr_now > 0 else open_price * 0.005

        gap_up   = (session_high - open_price) / open_price * 100
        gap_down = (open_price - session_low)  / open_price * 100
        min_gap  = params.get("min_gap_pct", 0.3)

        # ── SHORT ────────────────────────────────────────────────────────────
        if gap_up >= min_gap:
            retracement = open_price + 0.50 * (session_high - open_price)
            entry_bar = None
            for i in range(LOOKBACK_BARS, n):
                lo, hi, cl = bars["low"].iloc[i], bars["high"].iloc[i], bars["close"].iloc[i]
                if params["confirm_entry"]:
                    if lo <= retracement <= hi and cl < retracement:
                        entry_bar = i
                        break
                else:
                    if lo <= retracement <= hi:
                        entry_bar = i
                        break
            if entry_bar is None:
                continue

            stop_price  = retracement + params["stop_atr_mult"] * atr_now
            target_price = retracement - params["target_atr_mult"] * atr_now

            result, exit_price = simulate_exit(
                bars, entry_bar, retracement, stop_price, target_price, "SHORT"
            )
            risk = abs(retracement - stop_price)
            if result == "TARGET":
                reward = abs(retracement - exit_price)
                pnl = (reward - risk) / retracement * 10_000 - ROUND_TRIP_BPS
            elif result == "STOP":
                pnl = (-risk / retracement * 10_000) - ROUND_TRIP_BPS
            else:
                pnl = (retracement - exit_price) / retracement * 10_000 - ROUND_TRIP_BPS

            trades.append({
                "ticker": ticker, "date": session_date, "direction": "SHORT",
                "gap_pct": gap_up, "atr": atr_now, "exit_type": result,
                "pnl_bps": pnl, "result": "WIN" if pnl > 0 else "LOSS",
            })

        # ── LONG ─────────────────────────────────────────────────────────────
        if gap_down >= min_gap:
            retracement = open_price - 0.50 * (open_price - session_low)
            entry_bar = None
            for i in range(LOOKBACK_BARS, n):
                lo, hi, cl = bars["low"].iloc[i], bars["high"].iloc[i], bars["close"].iloc[i]
                if params["confirm_entry"]:
                    if lo <= retracement <= hi and cl > retracement:
                        entry_bar = i
                        break
                else:
                    if lo <= retracement <= hi:
                        entry_bar = i
                        break
            if entry_bar is None:
                continue

            stop_price   = retracement - params["stop_atr_mult"] * atr_now
            target_price = retracement + params["target_atr_mult"] * atr_now

            result, exit_price = simulate_exit(
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
                "ticker": ticker, "date": session_date, "direction": "LONG",
                "gap_pct": gap_down, "atr": atr_now, "exit_type": result,
                "pnl_bps": pnl, "result": "WIN" if pnl > 0 else "LOSS",
            })

    return pd.DataFrame(trades)


def aggregate(tdf):
    if tdf.empty:
        return {}
    n   = len(tdf)
    wr  = (tdf["result"] == "WIN").mean()
    avg = tdf["pnl_bps"].mean()
    med = tdf["pnl_bps"].median()
    std = tdf["pnl_bps"].std()
    sharpe_like = avg / (std + 1e-9) * np.sqrt(n) if std > 0 else 0
    exits = tdf.groupby("exit_type")["pnl_bps"].agg(["count","mean"])
    return {
        "n_trades": n, "win_rate": wr, "avg_pnl_bps": avg,
        "median_pnl": med, "std_pnl": std, "sharpe_like": sharpe_like,
        "stop_n":  exits.get("STOP",   pd.Series([0])).iloc[0] if "STOP"   in exits.index else 0,
        "stop_avg": exits.get("STOP",   pd.Series([np.nan])).iloc[0] if "STOP"   in exits.index else np.nan,
        "tgt_n":   exits.get("TARGET", pd.Series([0])).iloc[0] if "TARGET" in exits.index else 0,
        "tgt_avg": exits.get("TARGET", pd.Series([np.nan])).iloc[0] if "TARGET" in exits.index else np.nan,
        "time_n":   exits.get("TIME",   pd.Series([0])).iloc[0] if "TIME"   in exits.index else 0,
        "time_avg": exits.get("TIME",   pd.Series([np.nan])).iloc[0] if "TIME"   in exits.index else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────────────────────

print("Loading tickers...")
tickers = [p.stem for p in DATA_DIR.glob("*.parquet")]
print(f"  {len(tickers)} tickers found")

# Train: full 2025. Test: H1 2026 (Jan–Jun 2026)
TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END   = pd.Timestamp("2026-01-01")
TEST_START  = pd.Timestamp("2026-01-01")
TEST_END    = pd.Timestamp("2026-07-01")

# ── STEP 1: Train period — full parameter sweep ───────────────────────────────
print(f"\nStep 1: Training on 2025")
print(f"  Period: {TRAIN_START.date()} → {TRAIN_END.date()}")
print("=" * 70)

param_grid = list(product(
    CONFIRM_ENTRY_GRID,
    STOP_ATR_GRID,
    TARGET_ATR_GRID,
))
print(f"  {len(param_grid)} parameter combos to evaluate...")

train_results = []
ticker_train_cache = {}

for min_gap in [0.3, 0.5]:
    for confirm, stop_atr, tgt_atr in param_grid:
        p = dict(
            confirm_entry=confirm, stop_atr_mult=stop_atr,
            target_atr_mult=tgt_atr, min_gap_pct=min_gap,
            start_date=TRAIN_START, end_date=TRAIN_END,
        )
        combo_trades = []
        for ticker in tickers:
            key = (ticker, confirm, stop_atr, tgt_atr, min_gap)
            if key not in ticker_train_cache:
                ticker_train_cache[key] = run_backtest(ticker, p)
            tdf = ticker_train_cache[key]
            if not tdf.empty:
                combo_trades.append(tdf)
        if combo_trades:
            cdf = pd.concat(combo_trades, ignore_index=True)
            agg = aggregate(cdf)
            agg.update({
                "confirm": confirm, "stop_atr_mult": stop_atr,
                "target_atr_mult": tgt_atr, "min_gap_pct": min_gap,
                "variant_name": f"confirm={confirm}_stop={stop_atr}_target={tgt_atr}_gap={min_gap}",
            })
            train_results.append(agg)

train_df = pd.DataFrame(train_results)

print(f"\nTrain period results (top 20 by sharpe_like):")
cols = ["confirm","stop_atr_mult","target_atr_mult","min_gap_pct",
        "n_trades","win_rate","avg_pnl_bps","median_pnl","sharpe_like"]
top_train = train_df.sort_values("sharpe_like", ascending=False).head(20)
print(top_train[cols].to_string(index=False))

# ── STEP 2: Select top-N params, run on test period ──────────────────────────
TOP_N = 5
print(f"\nStep 2: Evaluating top {TOP_N} parameter combos on H1 2026 test period")
print(f"  Period: {TEST_START.date()} → {TEST_END.date()}")
print("=" * 70)

top_params = train_df.sort_values("sharpe_like", ascending=False).head(TOP_N)
test_results = []

for _, row in top_params.iterrows():
    p = dict(
        confirm_entry=row["confirm"],
        stop_atr_mult=row["stop_atr_mult"],
        target_atr_mult=row["target_atr_mult"],
        min_gap_pct=row["min_gap_pct"],
        start_date=TEST_START, end_date=TEST_END,
    )
    combo_trades = []
    for ticker in tickers:
        tdf = run_backtest(ticker, p)
        if not tdf.empty:
            combo_trades.append(tdf)
    if combo_trades:
        cdf = pd.concat(combo_trades, ignore_index=True)
        agg = aggregate(cdf)
        agg.update({
            "confirm": row["confirm"],
            "stop_atr_mult": row["stop_atr_mult"],
            "target_atr_mult": row["target_atr_mult"],
            "min_gap_pct": row["min_gap_pct"],
        })
        test_results.append(agg)

test_df = pd.DataFrame(test_results)

print(f"\nTest period results (top {TOP_N} train combos):")
print(test_df[cols].to_string(index=False))

# ── STEP 3: Compare train vs test ───────────────────────────────────────────
print(f"\n" + "=" * 70)
print("TRAIN vs TEST COMPARISON (top 5 from train)")
print("=" * 70)

comparison = top_params.merge(
    test_df, on=["confirm","stop_atr_mult","target_atr_mult","min_gap_pct"],
    suffixes=("_train","_test")
)
comp_cols = ["confirm","stop_atr_mult","target_atr_mult","min_gap_pct",
             "n_trades_train","avg_pnl_bps_train","sharpe_like_train",
             "n_trades_test","avg_pnl_bps_test","sharpe_like_test"]
print(comparison[comp_cols].to_string(index=False))

# ── STEP 4: Also test the CONFIRM_TIGHT params (fixed, no selection bias) ───
print(f"\n" + "=" * 70)
print("FIXED PARAMETER SET: confirm=True, stop=0.5, target=2.0 (CONFIRM_TIGHT)")
print("=" * 70)

fixed_params = dict(
    confirm_entry=True, stop_atr_mult=0.5,
    target_atr_mult=2.0, min_gap_pct=0.3,
)

train_fixed = []
for ticker in tickers:
    p = {**fixed_params, "start_date": TRAIN_START, "end_date": TRAIN_END}
    tdf = run_backtest(ticker, p)
    if not tdf.empty:
        train_fixed.append(tdf)
train_fixed_df = pd.concat(train_fixed, ignore_index=True) if train_fixed else pd.DataFrame()
train_agg = aggregate(train_fixed_df)

test_fixed = []
for ticker in tickers:
    p = {**fixed_params, "start_date": TEST_START, "end_date": TEST_END}
    tdf = run_backtest(ticker, p)
    if not tdf.empty:
        test_fixed.append(tdf)
test_fixed_df = pd.concat(test_fixed, ignore_index=True) if test_fixed else pd.DataFrame()
test_agg = aggregate(test_fixed_df)

print(f"\n  TRAIN (2025):")
print(f"    n_trades:    {train_agg.get('n_trades',0)}")
print(f"    win_rate:    {train_agg.get('win_rate',0):.1%}")
print(f"    avg_pnl_bps: {train_agg.get('avg_pnl_bps',0):.2f}")
print(f"    median_pnl:   {train_agg.get('median_pnl',0):.2f}")
print(f"    sharpe_like: {train_agg.get('sharpe_like',0):.2f}")
print(f"    exit STOP:   {train_agg.get('stop_n',0)} trades @ {train_agg.get('stop_avg',0):.1f} bps")
print(f"    exit TARGET: {train_agg.get('tgt_n',0)} trades @ {train_agg.get('tgt_avg',0):.1f} bps")
print(f"    exit TIME:   {train_agg.get('time_n',0)} trades @ {train_agg.get('time_avg',0):.1f} bps")

print(f"\n  TEST (H1 2026):")
print(f"    n_trades:    {test_agg.get('n_trades',0)}")
print(f"    win_rate:    {test_agg.get('win_rate',0):.1%}")
print(f"    avg_pnl_bps: {test_agg.get('avg_pnl_bps',0):.2f}")
print(f"    median_pnl:   {test_agg.get('median_pnl',0):.2f}")
print(f"    sharpe_like: {test_agg.get('sharpe_like',0):.2f}")
print(f"    exit STOP:   {test_agg.get('stop_n',0)} trades @ {test_agg.get('stop_avg',0):.1f} bps")
print(f"    exit TARGET: {test_agg.get('tgt_n',0)} trades @ {test_agg.get('tgt_avg',0):.1f} bps")
print(f"    exit TIME:   {test_agg.get('time_n',0)} trades @ {test_agg.get('time_avg',0):.1f} bps")

# ── Step 5: Monthly breakdown for test period ─────────────────────────────────
print(f"\n" + "=" * 70)
print("TEST PERIOD MONTHLY BREAKDOWN (CONFIRM_TIGHT)")
print("=" * 70)
if not test_fixed_df.empty:
    test_fixed_df["month"] = pd.to_datetime(test_fixed_df["date"]).dt.to_period("M")
    monthly = test_fixed_df.groupby(["month","direction"]).agg(
        n=("pnl_bps","count"),
        wr=("result", lambda x: (x=="WIN").mean()),
        avg=("pnl_bps","mean"),
    ).reset_index()
    monthly_pivot = monthly.pivot(index="month", columns="direction", values=["n","wr","avg"])
    print(monthly_pivot.to_string())

    # Per-ticker test performance
    print(f"\n  Ticker breakdown (H1 2026, CONFIRM_TIGHT, top 15 by avg_pnl):")
    tk = test_fixed_df.groupby(["ticker","direction"]).agg(
        n=("pnl_bps","count"),
        wr=("result", lambda x: (x=="WIN").mean()),
        avg=("pnl_bps","mean"),
    ).reset_index()
    tk = tk[tk["n"] >= 10].sort_values("avg", ascending=False)
    print(tk.head(15).to_string(index=False))

# ── Save ─────────────────────────────────────────────────────────────────────
train_df.to_csv(OUT_DIR / "train_sweep_results.csv", index=False)
comparison.to_csv(OUT_DIR / "train_test_comparison.csv", index=False)
if not test_fixed_df.empty:
    test_fixed_df.to_csv(OUT_DIR / "test_trades_CONFIRM_TIGHT.csv", index=False)

# ── Write report ─────────────────────────────────────────────────────────────
decay = (test_agg.get("avg_pnl_bps", 0) - train_agg.get("avg_pnl_bps", 0)) \
        / abs(train_agg.get("avg_pnl_bps", 1)) * 100 if train_agg.get("avg_pnl_bps") else float("nan")

report = f"""# Walk-Forward Validation Report

## Setup
- **Train period:** 2025-01-01 → 2025-12-31
- **Test period:**  2026-01-01 → 2026-06-30 (H1 2026)
- **Universe:** FTSE 100, {len(tickers)} tickers, hourly bars (07:00–15:00)
- **Strategy:** 50% retracement with confirm entry + ATR-based stop/target
- **Train params swept:** confirm={{True/False}} × stop_atr_mult={{0.3,0.5,0.75,1.0,1.5}} × target_atr_mult={{1.0,1.5,2.0,2.5,3.0}} × min_gap_pct={{0.3,0.5}}
- **Top-N selected:** {TOP_N} parameter combos by sharpe_like on train

## Train Period: Top 20 Parameter Combos (by sharpe_like)

| confirm | stop_atr | target_atr | min_gap | n_trades | win_rate | avg_pnl_bps | median_pnl | sharpe_like |
|---|---|---|---|---|---|---|---|---|
"""
for _, row in train_df.sort_values("sharpe_like", ascending=False).head(20).iterrows():
    report += f"| {row['confirm']} | {row['stop_atr_mult']} | {row['target_atr_mult']} | {row['min_gap_pct']} | {row['n_trades']} | {row['win_rate']:.1%} | **{row['avg_pnl_bps']:.2f}** | {row['median_pnl']:.2f} | {row['sharpe_like']:.1f} |\n"

report += f"""
## Train vs Test Comparison (top {TOP_N} train combos applied to H1 2026)

| confirm | stop | target | gap | n_train | train_avg | train_sharpe | n_test | test_avg | test_sharpe | decay% |
|---|---|---|---|---|---|---|---|---|---|---|
"""
for _, row in comparison.iterrows():
    decay_pct = (row['avg_pnl_bps_test'] - row['avg_pnl_bps_train']) / abs(row['avg_pnl_bps_train']) * 100 \
                if row['avg_pnl_bps_train'] != 0 else float('nan')
    report += f"| {row['confirm']} | {row['stop_atr_mult']} | {row['target_atr_mult']} | {row['min_gap_pct']} | {row['n_trades_train']} | {row['avg_pnl_bps_train']:.2f} | {row['sharpe_like_train']:.1f} | {row['n_trades_test']} | **{row['avg_pnl_bps_test']:.2f}** | {row['sharpe_like_test']:.1f} | {decay_pct:.0f}% |\n"

report += f"""
## Fixed Parameter Set: CONFIRM_TIGHT (confirm=True, stop=0.5× ATR, target=2.0× ATR)

No selection bias — params chosen a priori from exploratory analysis.

### Train (2025)
| Metric | Value |
|---|---|
| n_trades | {train_agg.get('n_trades', 0)} |
| win_rate | {train_agg.get('win_rate', 0):.1%} |
| avg_pnl_bps | **{train_agg.get('avg_pnl_bps', 0):.2f}** |
| median_pnl | {train_agg.get('median_pnl', 0):.2f} |
| sharpe_like | {train_agg.get('sharpe_like', 0):.2f} |
| STOP exits | {train_agg.get('stop_n', 0)} trades @ {train_agg.get('stop_avg', 0):.1f} bps |
| TARGET exits | {train_agg.get('tgt_n', 0)} trades @ {train_agg.get('tgt_avg', 0):.1f} bps |
| TIME exits | {train_agg.get('time_n', 0)} trades @ {train_agg.get('time_avg', 0):.1f} bps |

### Test (H1 2026)
| Metric | Value |
|---|---|
| n_trades | {test_agg.get('n_trades', 0)} |
| win_rate | {test_agg.get('win_rate', 0):.1%} |
| avg_pnl_bps | **{test_agg.get('avg_pnl_bps', 0):.2f}** |
| median_pnl | {test_agg.get('median_pnl', 0):.2f} |
| sharpe_like | {test_agg.get('sharpe_like', 0):.2f} |
| STOP exits | {test_agg.get('stop_n', 0)} trades @ {test_agg.get('stop_avg', 0):.1f} bps |
| TARGET exits | {test_agg.get('tgt_n', 0)} trades @ {test_agg.get('tgt_avg', 0):.1f} bps |
| TIME exits | {test_agg.get('time_n', 0)} trades @ {test_agg.get('time_avg', 0):.1f} bps |

**Edge decay (train→test): {decay:+.1f}%**

## Interpretation

- **Positive edge decay** (test < train): expected — parameters fitted to train data.
  **{decay:+.1f}%** decay means the strategy's performance in H1 2026 is *{abs(decay):.0f}%* weaker than in 2025.
- **Negative decay** (test > train): out-of-sample bonus — parameters generalise well.
- **Statistical significance**: compare test_avg vs its standard error
  (approx SE = std_pnl / sqrt(n_test)).
"""

(OUT_DIR / "WALKFORWARD_REPORT.md").write_text(report)
print(f"\n✓ Saved to {OUT_DIR}/")
print(f"  train_sweep_results.csv ({len(train_df)} combos)")
print(f"  train_test_comparison.csv")
print(f"  test_trades_CONFIRM_TIGHT.csv")
print(f"  WALKFORWARD_REPORT.md")
