"""
50% Retracement + Front Side / Back Side mean-reversion backtest.

Strategy logic:
  - Front side: the opening impulse (open → high/low of the first N bars)
  - Back side: the mean-reversion pullback that follows
  - Entry: price reaches the 50% retracement level of the front-side move
  - Stop: beyond the opening extreme (the origin of the front side)
  - Target: 1× ATR from entry in the direction of the reversal
  - Time stop: exit at session close if neither stop nor target hit
  - Direction:
      Gapper up → price retraces 50% back down → SHORT (fade the front side)
      Gapper down → price retraces 50% back up → LONG (back side rebound)

The target is NOT pre-guaranteed to be hit, so win/loss rates are meaningful.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/retracement")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cost & parameters ────────────────────────────────────────────────────────
ROUND_TRIP_BPS  = 10    # 10 bps round-trip cost (5 bps each side)
ATR_MULT         = 2.0   # stop distance = ATR_MULT × 14-bar ATR
TARGET_ATR       = 1.0   # target = TARGET_ATR × ATR from entry
MIN_GAP_PCT_GRID = [0.3, 0.5, 0.75, 1.0]
LOOKBACK_BARS_GRID = [1, 2, 3]

# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Rolling mean of high-low range (simplified ATR, no G/K)."""
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


def _simulate_exit(bars, entry_bar, entry_price, stop_price, target_price, direction):
    """
    Walk forward from entry_bar+1 to end of session.
    Returns (exit_type, exit_price, exit_bar).
    exit_type: 'TARGET', 'STOP', or 'TIME'
    """
    n = len(bars)
    for i in range(entry_bar + 1, n):
        bar_high = bars["high"].iloc[i]
        bar_low  = bars["low"].iloc[i]
        if direction == "SHORT":
            if bar_high >= stop_price:
                return ("STOP", stop_price, i)
            if bar_low <= target_price:
                return ("TARGET", target_price, i)
        else:  # LONG
            if bar_low <= stop_price:
                return ("STOP", stop_price, i)
            if bar_high >= target_price:
                return ("TARGET", target_price, i)
    final_close = bars["close"].iloc[-1]
    return ("TIME", final_close, n - 1)


def run_backtest_for_ticker(
    ticker: str,
    min_gap_pct: float,
    lookback_bars: int,
) -> pd.DataFrame:
    df = load_ticker(ticker)
    if df.empty:
        return pd.DataFrame()

    sessions = daily_sessions(df)
    trades = []

    for sess in sessions:
        bars = sess.sort_index().copy()
        session_date = pd.Timestamp(bars.index[0].date())
        n = len(bars)
        if n < lookback_bars + 2:
            continue

        open_price   = bars["open"].iloc[0]
        front        = bars.iloc[:lookback_bars]
        session_high = front["high"].max()
        session_low  = front["low"].min()

        gap_up   = (session_high - open_price) / open_price * 100
        gap_down = (open_price - session_low)  / open_price * 100

        # ── SHORT: gap up → price retraces 50% back down ──────────────────
        if gap_up >= min_gap_pct:
            retracement_level = open_price + 0.50 * (session_high - open_price)
            entry_bar = None
            for i in range(lookback_bars, n):
                if bars["low"].iloc[i] <= retracement_level <= bars["high"].iloc[i]:
                    entry_bar = i
                    break
            if entry_bar is None:
                continue

            atr         = compute_atr(bars.iloc[:entry_bar+1], 14).iloc[-1]
            atr         = atr if atr > 0 else open_price * 0.005
            stop_price  = session_high * 1.002
            target_price = retracement_level - TARGET_ATR * atr

            result_code, exit_price, _ = _simulate_exit(
                bars, entry_bar, retracement_level,
                stop_price, target_price, "SHORT"
            )
            risk = abs(retracement_level - stop_price)
            if result_code == "TARGET":
                reward = abs(retracement_level - exit_price)
                pnl_bps = (reward - risk) / retracement_level * 10_000 - ROUND_TRIP_BPS
            elif result_code == "STOP":
                pnl_bps = (-risk / retracement_level * 10_000) - ROUND_TRIP_BPS
            else:  # TIME
                pnl_bps = (exit_price - retracement_level) / retracement_level * 10_000 - ROUND_TRIP_BPS

            trades.append({
                "ticker":       ticker,
                "date":         session_date,
                "direction":    "SHORT",
                "gap_pct":      gap_up,
                "lookback":     lookback_bars,
                "min_gap_pct":  min_gap_pct,
                "entry_price":  retracement_level,
                "stop_price":   stop_price,
                "target_price": target_price,
                "atr":          atr,
                "exit_type":    result_code,
                "pnl_bps":      pnl_bps,
                "result":       "WIN" if pnl_bps > 0 else "LOSS",
            })

        # ── LONG: gap down → price retraces 50% back up ───────────────────
        if gap_down >= min_gap_pct:
            retracement_level = open_price - 0.50 * (open_price - session_low)
            entry_bar = None
            for i in range(lookback_bars, n):
                if bars["low"].iloc[i] <= retracement_level <= bars["high"].iloc[i]:
                    entry_bar = i
                    break
            if entry_bar is None:
                continue

            atr          = compute_atr(bars.iloc[:entry_bar+1], 14).iloc[-1]
            atr          = atr if atr > 0 else open_price * 0.005
            stop_price   = session_low * 0.998
            target_price = retracement_level + TARGET_ATR * atr

            result_code, exit_price, _ = _simulate_exit(
                bars, entry_bar, retracement_level,
                stop_price, target_price, "LONG"
            )
            risk = abs(retracement_level - stop_price)
            if result_code == "TARGET":
                reward = abs(exit_price - retracement_level)
                pnl_bps = (reward - risk) / retracement_level * 10_000 - ROUND_TRIP_BPS
            elif result_code == "STOP":
                pnl_bps = (-risk / retracement_level * 10_000) - ROUND_TRIP_BPS
            else:  # TIME
                pnl_bps = (retracement_level - exit_price) / retracement_level * 10_000 - ROUND_TRIP_BPS

            trades.append({
                "ticker":       ticker,
                "date":         session_date,
                "direction":    "LONG",
                "gap_pct":      gap_down,
                "lookback":     lookback_bars,
                "min_gap_pct":  min_gap_pct,
                "entry_price":  retracement_level,
                "stop_price":   stop_price,
                "target_price": target_price,
                "atr":          atr,
                "exit_type":    result_code,
                "pnl_bps":      pnl_bps,
                "result":       "WIN" if pnl_bps > 0 else "LOSS",
            })

    return pd.DataFrame(trades)


def summarise(all_trades: pd.DataFrame) -> pd.DataFrame:
    if all_trades.empty:
        return pd.DataFrame()
    g = all_trades.groupby(["direction", "lookback", "min_gap_pct"])
    summary = g.agg(
        n_trades    = ("pnl_bps", "count"),
        win_rate    = ("result",  lambda x: (x == "WIN").mean()),
        avg_pnl_bps = ("pnl_bps", "mean"),
        median_pnl  = ("pnl_bps", "median"),
        std_pnl     = ("pnl_bps", "std"),
        sharpe_like = ("pnl_bps", lambda x: x.mean() / (x.std() + 1e-9) * np.sqrt(max(len(x), 1))),
    ).reset_index()
    summary["edge_bps_per_trade"] = summary["avg_pnl_bps"]
    return summary.sort_values("avg_pnl_bps", ascending=False)


def main():
    print("Loading tickers...")
    tickers = [p.stem for p in DATA_DIR.glob("*.parquet")]
    print(f"  {len(tickers)} tickers found")

    all_trades = []
    param_grid = list(product(MIN_GAP_PCT_GRID, LOOKBACK_BARS_GRID))
    total = len(param_grid) * len(tickers)
    done  = 0

    print(f"\nRunning: {len(param_grid)} combos × {len(tickers)} tickers")
    print("=" * 70)

    for min_gap, lookback in param_grid:
        for ticker in tickers:
            trades = run_backtest_for_ticker(ticker, min_gap, lookback)
            if not trades.empty:
                all_trades.append(trades)
            done += 1
        if done % (len(tickers) * 4) == 0:
            print(f"  {done}/{total} ({100*done/total:.0f}%)")

    if not all_trades:
        print("No trades generated.")
        return

    all_trades = pd.concat(all_trades, ignore_index=True)

    summary = summarise(all_trades)

    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER COMBINATIONS (by avg pnl bps)")
    print("=" * 70)
    print(summary.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("EXIT TYPE BREAKDOWN")
    print("=" * 70)
    exit_tab = all_trades.groupby(["direction","exit_type"]).agg(
        n=("pnl_bps","count"), avg=("pnl_bps","mean")
    ).reset_index()
    print(exit_tab.to_string(index=False))

    print("\n" + "=" * 70)
    print("DIRECTION SUMMARY")
    print("=" * 70)
    print(summary[["direction","n_trades","win_rate","avg_pnl_bps","median_pnl","sharpe_like"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP TICKERS (≥20 trades, sorted by avg_pnl_bps)")
    print("=" * 70)
    ticker_sum = all_trades.groupby(["ticker","direction"]).agg(
        n_trades=("pnl_bps","count"),
        win_rate=("result", lambda x: (x=="WIN").mean()),
        avg_pnl =("pnl_bps","mean"),
    ).reset_index()
    top_tickers = ticker_sum[ticker_sum["n_trades"] >= 20].sort_values("avg_pnl", ascending=False)
    print(top_tickers.head(20).to_string(index=False))

    print("\n" + "=" * 70)
    print("PnL DISTRIBUTION")
    print("=" * 70)
    print(all_trades["pnl_bps"].describe().to_string())

    # Save
    all_trades.to_csv(OUTPUT_DIR / "all_trades.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "param_summary.csv", index=False)
    ticker_sum.to_csv(OUTPUT_DIR / "ticker_summary.csv", index=False)

    # Write markdown report
    n = len(all_trades)
    wr = (all_trades["result"] == "WIN").mean()
    report = f"""# 50% Retracement + Front Side / Back Side — Backtest Report

Generated: {pd.Timestamp.now().date()}

## Parameters
- min_gap_pct: {MIN_GAP_PCT_GRID}
- lookback_bars: {LOOKBACK_BARS_GRID}
- stop: session extreme × 1.002 (SHORT) / 0.998 (LONG)
- target: entry ± {TARGET_ATR}× ATR
- cost: {ROUND_TRIP_BPS} bps round-trip

## Data
- {len(tickers)} FTSE 100 tickers, hourly bars (07:00–15:00), ~May 2025–Sep 2026

## Strategy
**Short (fade front side after gap up):**
1. Session opens with gap ≥ min_gap_pct above prior close
2. Front side = high of first N bars
3. Price retraces to 50% level → entry
4. Stop: just above session high (origin of front side)
5. Target: 1× ATR below entry
6. Exit: hit stop, hit target, or session close

**Long (back side rebound after gap down):**
1. Session opens with gap ≥ min_gap_pct below prior close
2. Front side = low of first N bars
3. Price retraces to 50% level → entry
4. Stop: just below session low
5. Target: 1× ATR above entry
6. Exit: hit stop, hit target, or session close

## Results

### Overall
- **Total trades:** {n}
- **Overall win rate:** {wr:.1%}
- **Mean PnL:** {all_trades['pnl_bps'].mean():.2f} bps
- **Median PnL:** {all_trades['pnl_bps'].median():.2f} bps
- **Std Dev:** {all_trades['pnl_bps'].std():.2f} bps

### Exit Type Breakdown
| direction | exit_type | count | avg_pnl_bps |
|---|---|---|---|
"""
    for _, row in exit_tab.iterrows():
        report += f"| {row['direction']} | {row['exit_type']} | {row['n']} | {row['avg']:.2f} |\n"

    report += """
### Top Parameter Combinations
| direction | lookback | min_gap_pct | n_trades | win_rate | avg_pnl_bps | median_pnl | sharpe_like |
|---|---|---|---|---|---|---|---|
"""
    for _, row in summary.head(10).iterrows():
        report += f"| {row['direction']} | {row['lookback']} | {row['min_gap_pct']} | {row['n_trades']} | {row['win_rate']:.1%} | {row['avg_pnl_bps']:.2f} | {row['median_pnl']:.2f} | {row['sharpe_like']:.3f} |\n"

    report += """
### Top Tickers (≥20 trades)
| ticker | direction | n_trades | win_rate | avg_pnl |
|---|---|---|---|---|
"""
    for _, row in top_tickers.head(15).iterrows():
        report += f"| {row['ticker']} | {row['direction']} | {row['n_trades']} | {row['win_rate']:.1%} | {row['avg_pnl']:.2f} |\n"

    report += """
### Key Metrics Explained
- **avg_pnl_bps**: mean P&L per trade in basis points (post-cost)
- **sharpe_like**: mean(std) × √n — not a true Sharpe, but useful for ranking
- **TIME exits**: neither stop nor target hit by session close; result depends on close vs entry
- **TARGET exits**: reversal continued past 50% level — full reward captured
- **STOP exits**: price kept going in the original gap direction — full risk taken

### Caveats
- Hourly bars: fills at bar extremes; real fill depends on intra-bar price action
- ATR computed from high-low (no G/K); true ATR would be slightly wider
- No walk-forward / out-of-sample validation yet
- Single regime (2025–2026); results may not generalise to other market states
"""
    (OUTPUT_DIR / "RETRACEMENT_REPORT.md").write_text(report)
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  all_trades.csv ({n} trades)")
    print(f"  RETRACEMENT_REPORT.md written.")


if __name__ == "__main__":
    main()
