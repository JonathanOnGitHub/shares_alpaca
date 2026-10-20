"""
Fair Value Gap (FVG) Strategy — Backtest Module

FVG definition (ICT/SMC):
  Bullish FVG:  prior candle body (i-1) is entirely ABOVE prior-prior candle body (i-2)
                → gap between the two bodies: low[i-1] > high[i-2]
                → FVG zone: [high[i-2],  low[i-1]]

  Bearish FVG:  prior candle body (i-1) is entirely BELOW prior-prior candle body (i-2)
                → gap between the two bodies: high[i-1] < low[i-2]
                → FVG zone: [low[i-2],   high[i-1]]

Three strategy approaches:
  A — Mean Reversion:  price re-enters FVG zone → fade the original direction
  B — Continuation:    price displaces THROUGH FVG zone → trade the direction
  C — Combined:        price re-enters FVG zone AND displaces back through → confirm

Exits:     intraday  → close same day
           overnight  → hold to next close (stop/target walk-forward)

Entry:     Mean Reversion: open of bar AFTER re-entry bar
           Continuation:    open of bar AFTER displacement confirmed
           Combined:        open of bar AFTER both conditions met

Stop:      1× ATR beyond the FVG extreme (the far edge of the gap)
Target:    2× risk (R:R = 1:2)  — evaluated bar by bar until hit
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-smoothed ATR.  Returns array of same length as close."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    if n > period:
        atr[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _fvg_zones(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect FVGs and return arrays marking the top and bottom of each gap zone.

    Returns:
        bull_top   — upper edge of bullish FVG at each index (NaN if no FVG)
        bull_bot   — lower edge of bullish FVG at each index (NaN if no FVG)
        bear_top   — upper edge of bearish FVG at each index (NaN if no FVG)
        bear_bot   — lower edge of bearish FVG at each index (NaN if no FVG)

    FVG forms on bar i-1 (the gap bar); zone belongs to bar i-1.
    Bars 0 and 1 are always NaN (insufficient history).
    """
    n = len(close)
    bull_top   = np.full(n, np.nan)   # upper = prior candle's high (i-2 high)
    bull_bot   = np.full(n, np.nan)   # lower = gap candle's low (i-1 low)
    bear_top   = np.full(n, np.nan)
    bear_bot   = np.full(n, np.nan)

    for i in range(2, n):
        p_o, p_h, p_l, p_c = open_[i - 2], high[i - 2], low[i - 2], close[i - 2]  # bar i-2
        g_o, g_h, g_l, g_c = open_[i - 1], high[i - 1], low[i - 1], close[i - 1]  # bar i-1 (gap)

        # Bullish FVG: gap candle body entirely above prior candle body
        #   i-2 body: [p_l, p_h];  i-1 body: [g_l, g_h]
        #   FVG exists if g_l > p_h  (no body overlap; low of gap candle above high of prior)
        if g_l > p_h:
            bull_top[i - 1] = p_h    # zone top = prior candle's high
            bull_bot[i - 1] = g_l    # zone bottom = gap candle's low

        # Bearish FVG: gap candle body entirely below prior candle body
        #   FVG exists if g_h < p_l  (high of gap candle below low of prior candle)
        if g_h < p_l:
            bear_top[i - 1] = g_h    # zone top = gap candle's high
            bear_bot[i - 1] = p_l    # zone bottom = prior candle's low

    return bull_top, bull_bot, bear_top, bear_bot


def _max_drawdown(rets: List[float]) -> float:
    """Worst peak-to-trough drawdown from a list of period returns."""
    if not rets:
        return 0.0
    cum = np.cumprod([1 + r for r in rets])
    running_max = np.maximum.accumulate(cum)
    dd = (cum / running_max) - 1
    return float(np.min(dd) * 100)


def _run_single(
    price_df: Dict[str, pd.DataFrame],
    ticker: str,
    approach: str,            # "mean_rev", "continuation", "combined"
    direction: str,          # "bull", "bear", "both"
    exit_mode: str,          # "intraday" or "overnight"
    min_gap_bps: float,      # minimum gap size in basis points (0 = no filter)
    start_date: str,
    end_date: str,
    cost_bps: float = 0.0,
) -> Dict:
    """
    Run a single FVG strategy variant on one ticker.
    Returns per-trade results and summary metrics.
    """
    df = price_df[ticker].dropna().copy()
    df = df.sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    if len(df) < 30:
        return {"error": f"Insufficient data ({len(df)} rows)", "trades": []}

    o = np.asarray(df["Open"].values,   dtype=float)
    h = np.asarray(df["High"].values,   dtype=float)
    l = np.asarray(df["Low"].values,    dtype=float)
    c = np.asarray(df["Close"].values, dtype=float)
    idx = np.asarray(df.index.values)
    n = len(c)

    atr14 = _atr(h, l, c, period=14)
    bull_top, bull_bot, bear_top, bear_bot = _fvg_zones(o, h, l, c)

    # Pre-compute prior-day bars for displacement check
    # displacement = close of bar j closes beyond the FVG zone boundary
    # For bull FVG: displacement = close[j] > bull_top[j]  (above the zone)
    # For bear FVG: displacement = close[j] < bear_bot[j]  (below the zone)

    trades: List[Dict] = []

    # ── Pre-compute FVG validity mask and zone arrays ────────────────────────
    bull_valid = ~(np.isnan(bull_top) | np.isnan(bull_bot))
    bear_valid = ~(np.isnan(bear_top) | np.isnan(bear_bot))

    # ── For each bar, record which FVG types are "open" (not yet traded) ────
    # We scan forward: when an FVG fires, we lock and walk the trade
    # ── Main signal + trade loop ─────────────────────────────────────────────
    for i in range(2, n - 2):
        date = idx[i]

        # Build active FVG list at bar i
        # FVG forms on bar i-1, zone belongs to bar i-1
        # At bar i we check if entry conditions are met
        bull_zone_active = bull_valid[i - 1]
        bear_zone_active = bear_valid[i - 1]
        if not (bull_zone_active or bear_zone_active):
            continue

        # FVG zone values from formation bar (i-1)
        bft = bull_top[i - 1]
        bfb = bull_bot[i - 1]
        brt = bear_top[i - 1]
        brb = bear_bot[i - 1]

        # ── Minimum gap filter ─────────────────────────────────────────────────
        gap_ok_bull = False
        gap_ok_bear = False
        if bull_zone_active:
            gap_bps = (bfb - bft) / bft * 10_000
            gap_ok_bull = (min_gap_bps <= 0) or (gap_bps >= min_gap_bps)
        if bear_zone_active:
            gap_bps = (brt - brb) / brb * 10_000
            gap_ok_bear = (min_gap_bps <= 0) or (gap_bps >= min_gap_bps)

        # ── Scan bars i+1 onward for entry ──────────────────────────────────
        # Build arrays of future bars for vectorized checks
        # Entry scan: look for re-entry or displacement within next 10 bars
        scan_start = i + 1
        scan_end   = min(i + 10, n - 1)   # re-entry must occur within 10 bars

        for j in range(scan_start, scan_end):
            bar_high  = float(h[j])
            bar_low   = float(l[j])
            bar_open  = float(o[j])
            bar_close = float(c[j])

            # ── Bull FVG entry logic ──────────────────────────────────────────
            if bull_zone_active and gap_ok_bull:
                is_long_entry = False
                entry_px: float = np.nan
                stop_px: float = np.nan
                tgt_px: float = np.nan
                trade_approach = ""

                # Mean Reversion: bar low touches zone bottom, bar within zone
                mr_reentry = (bar_low <= bfb) and (bar_high >= bft)
                # Displacement: bar closes above zone top
                disp = bar_close > bft

                if approach == "mean_rev" and mr_reentry:
                    is_long_entry = True
                    entry_px = max(bar_open, bft)
                    entry_px = min(entry_px, bar_close)
                    entry_px = max(entry_px, bfb)
                    trade_approach = "mean_rev"
                elif approach == "continuation" and disp:
                    is_long_entry = True
                    entry_px = bar_open
                    trade_approach = "continuation"
                elif approach == "combined":
                    # Combined: re-entry first, then displacement confirmation within 3 bars
                    if mr_reentry:
                        is_long_entry = True
                        entry_px = bar_close
                        trade_approach = "combined_reentry"
                        # Walk forward up to 3 more bars for displacement confirmation
                        confirmed = False
                        for k in range(j + 1, min(j + 4, n - 1)):
                            if c[k] > bft:   # displaced above zone
                                entry_px = o[k]
                                trade_approach = "combined"
                                confirmed = True
                                j = k         # update j to confirmed bar
                                break
                        if not confirmed:
                            # No displacement within 3 bars — skip this FVG entirely
                            is_long_entry = False
                            entry_px = np.nan
                            continue   # move to next scan bar, don't record trade

                if is_long_entry and not np.isnan(entry_px) and entry_px > 0:
                    stop_px = bfb - 1.0 * atr14[j + 1]
                    risk    = entry_px - stop_px
                    if risk <= 0:
                        continue
                    tgt_px  = entry_px + 2.0 * risk
                    fvg_entry_bar = j

                    # ── Walk forward for exit ──────────────────────────────────
                    exit_price: float = np.nan
                    exit_bar   = -1
                    exit_type  = "expired"

                    if exit_mode == "intraday":
                        exit_price = float(c[j + 1])
                        exit_bar   = j + 1
                        exit_type  = "intraday"
                    else:
                        for k in range(j + 1, n):
                            kh = float(h[k]); kl = float(l[k])
                            if kl <= stop_px:
                                exit_price = float(stop_px)
                                exit_bar   = k
                                exit_type  = "stop"
                                break
                            if kh >= tgt_px:
                                exit_price = float(tgt_px)
                                exit_bar   = k
                                exit_type  = "target"
                                break

                    if exit_bar > 0 and not np.isnan(exit_price):
                        pnl_r = (exit_price - entry_px) / entry_px
                        pnl_r -= cost_bps / 10_000
                        pnl_bps = pnl_r * 10_000
                        trades.append({
                            "ticker":      ticker,
                            "entry_date":  str(idx[fvg_entry_bar])[:10],
                            "exit_date":   str(idx[exit_bar])[:10],
                            "direction":   "long",
                            "approach":    trade_approach,
                            "entry_price": entry_px,
                            "exit_price":  exit_price,
                            "stop_price":  stop_px,
                            "target_price":tgt_px,
                            "pnl_r":       pnl_r,
                            "pnl_bps":     pnl_bps,
                            "ret_pct":     pnl_r * 100,
                            "exit_type":   exit_type,
                            "atr_entry":   atr14[fvg_entry_bar],
                        })
                    bull_zone_active = False   # FVG consumed

            # ── Bear FVG entry logic ──────────────────────────────────────────
            if bear_zone_active and gap_ok_bear:
                is_short_entry = False
                entry_px = np.nan
                stop_px  = np.nan
                tgt_px   = np.nan
                trade_approach = ""

                mr_reentry = (bar_high >= brt) and (bar_low <= brb)
                disp = bar_close < brb

                if approach == "mean_rev" and mr_reentry:
                    is_short_entry = True
                    entry_px = min(bar_open, brb)
                    entry_px = max(entry_px, bar_close)
                    entry_px = min(entry_px, brt)
                    trade_approach = "mean_rev"
                elif approach == "continuation" and disp:
                    is_short_entry = True
                    entry_px = bar_open
                    trade_approach = "continuation"
                elif approach == "combined":
                    if mr_reentry:
                        is_short_entry = True
                        entry_px = bar_close
                        trade_approach = "combined_reentry"
                        confirmed = False
                        for k in range(j + 1, min(j + 4, n - 1)):
                            if c[k] < brb:
                                entry_px = o[k]
                                trade_approach = "combined"
                                confirmed = True
                                j = k
                                break
                        if not confirmed:
                            is_short_entry = False
                            entry_px = np.nan
                            continue

                if is_short_entry and not np.isnan(entry_px) and entry_px > 0:
                    stop_px = brt + 1.0 * atr14[j + 1]
                    risk    = stop_px - entry_px
                    if risk <= 0:
                        continue
                    tgt_px  = entry_px - 2.0 * risk
                    fvg_entry_bar = j

                    if exit_mode == "intraday":
                        exit_price = float(c[j + 1])
                        exit_bar   = j + 1
                        exit_type  = "intraday"
                    else:
                        exit_price: float = np.nan
                        exit_bar   = -1
                        exit_type  = "expired"
                        for k in range(j + 1, n):
                            kh = float(h[k]); kl = float(l[k])
                            if kh >= stop_px:
                                exit_price = float(stop_px)
                                exit_bar   = k
                                exit_type  = "stop"
                                break
                            if kl <= tgt_px:
                                exit_price = float(tgt_px)
                                exit_bar   = k
                                exit_type  = "target"
                                break

                    if exit_bar > 0 and not np.isnan(exit_price):
                        pnl_r = (entry_px - exit_price) / entry_px
                        pnl_r -= cost_bps / 10_000
                        pnl_bps = pnl_r * 10_000
                        trades.append({
                            "ticker":      ticker,
                            "entry_date":  str(idx[fvg_entry_bar])[:10],
                            "exit_date":   str(idx[exit_bar])[:10],
                            "direction":   "short",
                            "approach":    trade_approach,
                            "entry_price": entry_px,
                            "exit_price":  exit_price,
                            "stop_price":  stop_px,
                            "target_price":tgt_px,
                            "pnl_r":       pnl_r,
                            "pnl_bps":     pnl_bps,
                            "ret_pct":     pnl_r * 100,
                            "exit_type":   exit_type,
                            "atr_entry":   atr14[fvg_entry_bar],
                        })
                    bear_zone_active = False

    # ── summary metrics ────────────────────────────────────────────────────
    if not trades:
        return {"ticker": ticker, "approach": approach, "direction": direction,
                "exit_mode": exit_mode, "trend_lookback": 0, "trades": [],
                "n": 0, "error": None}

    rets      = [t["pnl_r"]  for t in trades]
    bps_list  = [t["pnl_bps"] for t in trades]
    wins_list = [t for t in trades if t["pnl_r"] > 0]
    loss_list = [t for t in trades if t["pnl_r"] < 0]
    stops     = [t for t in trades if t["exit_type"] == "stop"]
    targets   = [t for t in trades if t["exit_type"] == "target"]
    intraday  = [t for t in trades if t["exit_type"] == "intraday"]

    cum           = np.cumprod([1 + r for r in rets])
    total_return  = float(cum[-1]) - 1
    ann_vol       = float(np.std(rets) * math.sqrt(252))
    ann_ret_agg   = (1 + total_return) ** (252 / max(len(rets), 1)) - 1
    max_dd        = _max_drawdown(rets)

    return {
        "ticker":        ticker,
        "approach":      approach,
        "direction":     direction,
        "exit_mode":     exit_mode,
        "trend_lookback": 0,
        "trades":        trades,
        "n":             len(trades),
        "error":         None,
        "total_ret":     total_return,
        "ann_ret":       ann_ret_agg,
        "ann_vol":       ann_vol,
        "sharpe":        (ann_ret_agg / ann_vol) if ann_vol > 0 else 0.0,
        "max_dd":        max_dd,
        "win_rate":      len(wins_list)  / len(trades) * 100,
        "avg_bps":       float(np.mean(bps_list)),
        "avg_win_bps":   float(np.mean([t["pnl_bps"] for t in wins_list])) if wins_list else 0.0,
        "avg_loss_bps":  float(np.mean([t["pnl_bps"] for t in loss_list])) if loss_list else 0.0,
        "stop_rate":     len(stops)    / len(trades) * 100,
        "target_rate":   len(targets)  / len(trades) * 100,
        "intraday_rate": len(intraday) / len(trades) * 100,
    }


def run_all_variants(
    price_df: Dict[str, pd.DataFrame],
    tickers: List[str],
    start_date: str = "2015-01-01",
    end_date:   str = "2025-12-31",
    min_gap_bps: float = 0.0,
    cost_bps: float = 0.0,
) -> Dict:
    """
    Run every combination of:
      approach     = mean_rev / continuation / combined
      direction    = bull / bear / both
      exit_mode   = intraday / overnight
    Returns dict keyed by "{approach}_{direction}_{exit_mode}".
    """
    approaches  = ["mean_rev", "continuation", "combined"]
    directions  = ["bull", "bear", "both"]
    exit_modes  = ["intraday", "overnight"]

    results: Dict[str, Dict] = {}

    for approach in approaches:
        for direction in directions:
            for exit_mode in exit_modes:
                key = f"{approach}_{direction}_{exit_mode}"
                trades_pool: List[Dict] = []
                n_total = 0
                sum_bps = 0.0
                sum_win_rate = 0.0
                sum_stop_rate = 0.0
                sum_target_rate = 0.0
                sum_intraday_rate = 0.0
                sum_avg_win_bps = 0.0
                sum_avg_loss_bps = 0.0

                for ticker in tickers:
                    r = _run_single(
                        price_df, ticker, approach, direction, exit_mode,
                        min_gap_bps, start_date, end_date, cost_bps=cost_bps,
                    )
                    n_r = r.get("n", 0)
                    if n_r > 0:
                        trades_pool.extend(r["trades"])
                        n_total              += n_r
                        sum_bps              += r.get("avg_bps", 0.0) * n_r
                        sum_win_rate         += r.get("win_rate", 0.0) * n_r
                        sum_stop_rate        += r.get("stop_rate", 0.0) * n_r
                        sum_target_rate      += r.get("target_rate", 0.0) * n_r
                        sum_intraday_rate    += r.get("intraday_rate", 0.0) * n_r
                        sum_avg_win_bps      += r.get("avg_win_bps", 0.0) * n_r
                        sum_avg_loss_bps     += r.get("avg_loss_bps", 0.0) * n_r

                if n_total > 0:
                    rets     = [t["pnl_r"]  for t in trades_pool]
                    bps_all  = [t["pnl_bps"] for t in trades_pool]
                    wins_p   = [t for t in trades_pool if t["pnl_r"] > 0]
                    loss_p   = [t for t in trades_pool if t["pnl_r"] < 0]

                    cum            = np.cumprod([1 + r for r in rets])
                    total_return   = float(cum[-1]) - 1
                    ann_vol        = float(np.std(rets) * math.sqrt(252))
                    ann_ret_agg    = (1 + total_return) ** (252 / max(len(rets), 1)) - 1

                    n_all = n_total
                    results[key] = {
                        "n":              n_total,
                        "approach":       approach,
                        "direction":      direction,
                        "exit_mode":      exit_mode,
                        "total_ret":      total_return,
                        "ann_ret":        ann_ret_agg,
                        "ann_vol":        ann_vol,
                        "sharpe":         (ann_ret_agg / ann_vol) if ann_vol > 0 else 0.0,
                        "max_dd":         _max_drawdown(rets),
                        "avg_bps":        float(np.mean(bps_all)),
                        "win_rate":       len(wins_p) / n_all * 100,
                        "avg_win_bps":    float(np.mean([t["pnl_bps"] for t in wins_p])) if wins_p else 0.0,
                        "avg_loss_bps":   float(np.mean([t["pnl_bps"] for t in loss_p])) if loss_p else 0.0,
                        "stop_rate":      len([t for t in trades_pool if t["exit_type"] == "stop"]) / n_all * 100,
                        "target_rate":    len([t for t in trades_pool if t["exit_type"] == "target"]) / n_all * 100,
                        "intraday_rate":  len([t for t in trades_pool if t["exit_type"] == "intraday"]) / n_all * 100,
                        "all_trades":     trades_pool,
                    }
                else:
                    results[key] = {"n": 0, "approach": approach,
                                    "direction": direction, "exit_mode": exit_mode}

    results["_meta"] = {"tickers": tickers, "start": start_date, "end": end_date,
                         "cost_bps": cost_bps, "min_gap_bps": min_gap_bps}
    return results


def print_results(results: Dict, title: str = "") -> None:
    meta = results.get("_meta", {})
    print(f"\n{'═'*72}")
    print(f"  FVG BACKTEST  |  {meta.get('start','?')} – {meta.get('end','?')}"
          f"  |  cost={meta.get('cost_bps', 0)} bps")
    if title:
        print(f"  {title}")
    print(f"{'═'*72}")

    variants = [(k, v) for k, v in results.items() if k != "_meta"]
    variants.sort(key=lambda x: x[1].get("sharpe", -999), reverse=True)

    header = (f"{'Variant':<42} {'N':>5} {'Ann%':>7} "
              f"{'Sharpe':>7} {'MaxDD%':>8} {'WR%':>6} "
              f"{'AvgBps':>7} {'Stop%':>6} {'Tgt%':>6}")
    print(header)
    print("─" * 88)
    for name, r in variants:
        n = r.get("n", 0)
        if n == 0:
            continue
        ann   = r.get("ann_ret", 0.0) * 100
        shp   = r.get("sharpe", 0.0)
        dd    = r.get("max_dd", 0.0)
        wr    = r.get("win_rate", 0.0)
        bps   = r.get("avg_bps", 0.0)
        stp   = r.get("stop_rate", 0.0)
        tgt   = r.get("target_rate", 0.0)
        print(f"{name:<42} {n:>5} {ann:>7.1f} {shp:>7.2f} "
              f"{dd:>8.1f} {wr:>6.1f} {bps:>7.1f} {stp:>6.1f} {tgt:>6.1f}")
    print("─" * 88)


def save_results(results: Dict, path: str) -> None:
    out: Dict = {}
    for k, v in results.items():
        if k == "_meta":
            out[k] = v
        elif isinstance(v, dict) and "all_trades" in v:
            trades_only = [{kk: vv for kk, vv in t.items()
                           if kk != "all_trades"}
                          for t in v.get("all_trades", [])]
            summary = {kk: vv for kk, vv in v.items() if kk != "all_trades"}
            out[k] = {"summary": summary, "trades": trades_only}
        else:
            out[k] = v
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
