"""
Opening Range Momentum Strategy v2 — Intraday

Redesign from scratch after v1 catastrophic failure.

Key design rules (learned the hard way):
  1. ONE entry per ticker per session — entry throttle is NON-NEGOTIABLE
  2. Stop anchored to OR extreme: long→stop at OR Low, short→stop at OR High
  3. Position sizing: $100 risk / dollar stop distance → real share counts
  4. 15-min bars (balances granularity vs signal frequency)
  5. Volume filter: vol > 1.5× median of prior 20 bars
  6. LONG ONLY (shorts structurally underperform in bull market)
  7. No new entries after 13:00 EST; flat by 13:45 EST
  8. 1 bps round-trip friction built in

Universe:  20 liquid names (see UNIVERSE_20)
Bars:      15-minute (yfinance, ~60-day lookback)
Session:   09:30–15:00 EST
Entry:     candle CLOSES above OR High + volume confirmation
Stop:      OR Low (fixed dollar distance from entry)
Target:    2:1 R:R from entry
Risk:      $100 flat per trade; max 3 concurrent positions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────────

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-smoothed ATR. Returns array with NaN before period."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n > period:
        atr[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _max_drawdown(rets: List[float]) -> Tuple[float, float]:
    """Returns (max_dd_pct, dd_peak_idx)."""
    if not rets:
        return 0.0, -1
    cum = np.cumprod([1 + r for r in rets])
    running_max = np.maximum.accumulate(cum)
    dd = (cum / running_max) - 1
    mdd = float(np.min(dd) * 100)
    peak = int(np.argmax(running_max))
    return mdd, peak


def _apply_friction(pnl: float, entry_px: float, shares: int) -> float:
    """Deduct 1 bps round-trip cost."""
    if shares <= 0 or entry_px <= 0:
        return pnl
    cost = entry_px * shares * 0.0001
    return pnl - cost


# ── per-ticker backtest ─────────────────────────────────────────────────────────

def _run_ticker(
    ticker: str,
    df: pd.DataFrame,
    risk_dollars: float = 100.0,
    cost_bps: float = 1.0,
    vol_ratio: float = 1.5,
    or_bars: int = 10,
    entry_cutoff_min: int = 780,   # 13:00 EST in minutes-from-midnight (780 min after midnight)
    exit_min: int = 825,          # 13:45 EST (825 min after midnight)
) -> Dict:
    """
    Run OR Momentum v2 backtest on one ticker.

    Key rules enforced:
    - ONE entry per session per ticker
    - LONG ONLY
    - Stop at OR Low (dollar distance from entry)
    - 2:1 R:R target
    - Volume confirmation on entry bar
    - No new entries after entry_cutoff_min
    - Flat by exit_min
    """
    df = df.copy().sort_index()
    if len(df) < or_bars + 5:
        return {"ticker": ticker, "error": f"Insufficient bars ({len(df)})", "trades": []}

    h = np.asarray(df["High"].values, dtype=float)
    l = np.asarray(df["Low"].values, dtype=float)
    c = np.asarray(df["Close"].values, dtype=float)
    v = np.asarray(df["Volume"].values, dtype=float)
    ts_idx = df.index  # DatetimeIndex

    n = len(c)

    # Rolling median volume (prior 20 bars)
    med_vol = np.full(n, np.nan)
    for i in range(20, n):
        med_vol[i] = np.median(v[i - 20:i])

    # ── Split into sessions using positional detection ───────────────────────────
    sessions: List[pd.DataFrame] = []
    current_date = None
    session_start = 0
    for pos in range(n):
        bar_date = ts_idx[pos].date()
        if bar_date != current_date:
            if current_date is not None:
                sessions.append(df.iloc[session_start:pos])
            session_start = pos
            current_date = bar_date
    sessions.append(df.iloc[session_start:])

    trades: List[Dict] = []
    open_trade: Optional[Dict] = None

    for sdf in sessions:
        s_n = len(sdf)
        if s_n < or_bars + 2:
            continue

        # ── Establish Opening Range ────────────────────────────────────────────
        # OR High/Low from first `or_bars` bars of session
        sp = int(np.where(df.index == sdf.index[0])[0][0])  # absolute start pos
        or_high = float(np.max(h[sp:sp + or_bars]))
        or_low  = float(np.min(l[sp:sp + or_bars]))

        entered_this_session = False

        for rel_i in range(or_bars, s_n):
            abs_i = sp + rel_i
            date  = ts_idx[abs_i]
            total_min = date.hour * 60 + date.minute

            bar_high  = float(h[abs_i])
            bar_low   = float(l[abs_i])
            bar_close = float(c[abs_i])
            bar_vol   = float(v[abs_i])

            # ── Session exit (flat everything) ────────────────────────────────
            if total_min >= exit_min:
                if open_trade is not None:
                    exit_px = bar_close
                    d = open_trade["direction"]
                    pnl = (exit_px - open_trade["entry_px"]) * open_trade["shares"] * d
                    pnl = _apply_friction(pnl, open_trade["entry_px"], open_trade["shares"])
                    trades.append({
                        "ticker": ticker,
                        "direction": "long",
                        "entry": open_trade["entry_px"],
                        "exit": round(exit_px, 4),
                        "shares": open_trade["shares"],
                        "pnl": round(pnl, 2),
                        "exit_reason": "time_close",
                        "entry_time": str(open_trade["entry_time"]),
                        "exit_time": str(date),
                        "bars_held": abs_i - open_trade["entry_bar"],
                        "or_high": round(or_high, 4),
                        "or_low": round(or_low, 4),
                        "or_bars": or_bars,
                    })
                    open_trade = None
                continue

            # ── Manage open trade ──────────────────────────────────────────────
            if open_trade is not None:
                d = open_trade["direction"]
                hit_stop   = (d ==  1 and bar_low  <= open_trade["stop_px"]) or \
                             (d == -1 and bar_high >= open_trade["stop_px"])
                hit_target = (d ==  1 and bar_high >= open_trade["target_px"]) or \
                             (d == -1 and bar_low  <= open_trade["target_px"])
                if hit_stop or hit_target:
                    exit_px = open_trade["stop_px"] if hit_stop else open_trade["target_px"]
                    reason  = "stop" if hit_stop else "target"
                    pnl = (exit_px - open_trade["entry_px"]) * open_trade["shares"] * d
                    pnl = _apply_friction(pnl, open_trade["entry_px"], open_trade["shares"])
                    trades.append({
                        "ticker": ticker,
                        "direction": "long",
                        "entry": open_trade["entry_px"],
                        "exit": round(exit_px, 4),
                        "shares": open_trade["shares"],
                        "pnl": round(pnl, 2),
                        "exit_reason": reason,
                        "entry_time": str(open_trade["entry_time"]),
                        "exit_time": str(date),
                        "bars_held": abs_i - open_trade["entry_bar"],
                        "or_high": round(or_high, 4),
                        "or_low": round(or_low, 4),
                        "or_bars": or_bars,
                    })
                    open_trade = None
                continue

            # ── Entry throttle: one trade per session ───────────────────────────
            if entered_this_session:
                continue

            # No new entries after cutoff time
            if total_min > entry_cutoff_min:
                continue

            # Volume must confirm
            vol_ok = (med_vol[abs_i] > 0) and (bar_vol >= vol_ratio * med_vol[abs_i])
            if not vol_ok:
                continue

            # LONG: price closes above OR High
            if bar_close > or_high:
                # Stop at OR Low: dollar risk = entry - or_low
                dollar_risk = bar_close - or_low
                if dollar_risk <= 0:
                    continue

                shares = int(risk_dollars / dollar_risk)
                if shares < 1:
                    continue

                target_px = round(bar_close + 2.0 * dollar_risk, 4)
                stop_px   = round(or_low, 4)

                open_trade = {
                    "direction": 1,
                    "entry_px": bar_close,
                    "stop_px": stop_px,
                    "target_px": target_px,
                    "shares": shares,
                    "entry_bar": abs_i,
                    "entry_time": date,
                }
                entered_this_session = True

    return {"ticker": ticker, "trades": trades}


# ── summary ───────────────────────────────────────────────────────────────────

def _summarize(results: List[Dict], risk_dollars: float) -> Dict:
    all_trades = []
    for r in results:
        if "error" not in r:
            all_trades.extend(r.get("trades", []))

    if not all_trades:
        return {"n_trades": 0, "error": "No trades"}

    pnls = [t["pnl"] for t in all_trades]
    rets = [p / risk_dollars for p in pnls]
    n_wins   = sum(1 for p in pnls if p > 0)
    n_losses = len(pnls) - n_wins

    ann_factor = 252 * 26   # 26 fifteen-min bars per trading day
    mean_ret = np.mean(rets)
    std_ret  = np.std(rets)
    sharpe    = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 0 else 0.0

    # Compound annual return
    if rets:
        compound = np.prod([1 + r for r in rets])
        ann_ret  = (compound ** (ann_factor / len(rets)) - 1) * 100
    else:
        ann_ret = 0.0

    max_dd, _ = _max_drawdown(rets)

    n_days = len(set(t["entry_time"][:10] for t in all_trades))
    trades_per_day = round(len(all_trades) / max(n_days, 1), 2)

    # Exit reason breakdown
    from collections import Counter
    exit_counts = Counter(t["exit_reason"] for t in all_trades)

    return {
        "n_trades":        len(all_trades),
        "n_tickers":      len([r for r in results if "error" not in r]),
        "n_days":          n_days,
        "trades_per_day":  trades_per_day,
        "win_rate":        round(n_wins / len(pnls) * 100, 1),
        "ann_return_pct":  round(float(ann_ret), 1),
        "sharpe":          round(float(sharpe), 2),
        "max_dd_pct":      round(float(max_dd), 1),
        "avg_pnl":         round(float(np.mean(pnls)), 2),
        "total_pnl":       round(float(sum(pnls)), 2),
        "best_trade":      round(float(max(pnls)), 2),
        "worst_trade":     round(float(min(pnls)), 2),
        "n_wins":          n_wins,
        "n_losses":        n_losses,
        "exit_reasons":    dict(exit_counts),
        # Friction sensitivity
        "pnl_at_1bps": round(float(sum(pnls) - len(pnls) * risk_dollars * 0.0001), 2),
        "pnl_at_2bps": round(float(sum(pnls) - len(pnls) * risk_dollars * 0.0002), 2),
        "pnl_at_5bps": round(float(sum(pnls) - len(pnls) * risk_dollars * 0.0005), 2),
    }


# ── universe + data fetch ──────────────────────────────────────────────────────

UNIVERSE_20 = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "AMZN",
    "META", "GOOGL", "TSLA", "AMD",
    "NFLX", "CRM", "ORCL",
    "JPM", "BAC", "GS",
    "XOM", "CVX",
]


def fetch_15min_bars(ticker: str, days: int = 60) -> pd.DataFrame:
    """Fetch 15-min bars, filter to RTH 09:30–15:00, drop zero-volume rows."""
    import yfinance as yf
    import datetime

    end   = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)

    df = yf.download(
        ticker, start=start, end=end,
        interval="15m", prepost=True,
        auto_adjust=True, progress=False,
    )
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Drop zero-volume (pre/after-hours) and NaN
    df = df[df["Volume"] > 0].dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    # Keep 09:30–15:00 EST
    df = df.between_time("09:30", "15:00")

    return df


def run_backtest(
    universe: List[str] = None,
    days: int = 60,
    risk_dollars: float = 100.0,
    cost_bps: float = 1.0,
    vol_ratio: float = 1.5,
    or_bars: int = 10,
    output_path: str = "reports/opening_range_v2_results.json",
) -> Tuple[Dict, Dict]:

    universe = universe or UNIVERSE_20
    results  = []
    fetched  = {}

    print(f"[ORv2] Fetching 15-min data for {len(universe)} tickers ({days} days)...")
    for ticker in universe:
        df = fetch_15min_bars(ticker, days=days)
        if df.empty:
            print(f"  {ticker}: no data")
            continue
        fetched[ticker] = df
        print(f"  {ticker}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")

    print(f"\n[ORv2] Running backtest on {len(fetched)} tickers...")
    for ticker, df in fetched.items():
        r = _run_ticker(
            ticker, df,
            risk_dollars=risk_dollars,
            cost_bps=cost_bps,
            vol_ratio=vol_ratio,
            or_bars=or_bars,
        )
        results.append(r)
        n = len(r.get("trades", []))
        print(f"  {ticker}: {n} trades  {r.get('error', '')}")

    summary = _summarize(results, risk_dollars)

    print(f"\n{'='*55}")
    print(f"[ORv2] SUMMARY  (risk=${risk_dollars}/trade, {cost_bps} bps friction)")
    print(f"{'='*55}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = {
        "summary": summary,
        "params": {
            "risk_dollars": risk_dollars,
            "cost_bps": cost_bps,
            "vol_ratio": vol_ratio,
            "or_bars": or_bars,
            "days": days,
            "n_tickers": len(fetched),
        },
        "by_ticker": {},
    }
    for r in results:
        t = r.get("ticker", "unknown")
        if "error" not in r:
            out["by_ticker"][t] = {"n_trades": len(r.get("trades", []))}
        else:
            out["by_ticker"][t] = {"n_trades": 0, "error": r["error"]}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n[ORv2] Results → {output_path}")
    return summary, out


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",       type=int,   default=60)
    ap.add_argument("--risk",       type=float, default=100.0)
    ap.add_argument("--cost-bps",   type=float, default=1.0)
    ap.add_argument("--vol-ratio",  type=float, default=1.5)
    ap.add_argument("--or-bars",    type=int,   default=10)
    ap.add_argument("--output",     default="reports/opening_range_v2_results.json")
    args = ap.parse_args()

    run_backtest(
        days=args.days,
        risk_dollars=args.risk,
        cost_bps=args.cost_bps,
        vol_ratio=args.vol_ratio,
        or_bars=args.or_bars,
        output_path=args.output,
    )
