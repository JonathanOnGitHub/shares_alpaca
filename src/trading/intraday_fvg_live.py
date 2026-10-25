"""
Intraday FVG Strategy — LIVE signal generator.

This mirrors `src/trading/intraday_fvg.py` (the backtest that produced the
reported edge) EXACTLY. The previous live path used `src/trading/fvg_live.py`,
which was a different strategy entirely:

  * daily bars, not 5-minute bars
  * a different FVG detector (body gap, re-entry trigger) instead of the
    backtest's displacement-confirmation trigger
  * 1.0 x ATR stop instead of the backtest's 0.5 x ATR
  * no volume filter, no 09:30-11:00 formation window, no 14:00 flat time

Running the backtested strategy and trading a different one is how a
Sharpe 2.95 backtest turns into an unvalidated live trade. Everything
below follows the backtest's rules.

Backtest spec (src/trading/intraday_fvg.py):
  Universe     : UNIVERSE_20 (22 liquid US names)
  Bars         : 5-minute, yfinance, prepost=True, RTH 09:30-16:00 ET
  FVG forms    : bar i-1 gaps bar i-2; must form 09:30-11:00 ET
  Entry        : bar i closes beyond the zone boundary (displacement)
  Volume       : confirmation bar volume >= median of prior 20 bars
  Stop         : 0.5 x ATR(14) from entry
  Target       : 2 x risk (2:1 R:R)
  Sizing       : risk_dollars / stop_dist, notional capped at 2 x risk
  Exit         : stop / target / 14:00 ET hard close
  Max concurrent: 3 positions
  Friction     : 1.0 bps round-trip

Live behaviour notes:
  * Signals are emitted for the CURRENT session only, and the executor
    enters them immediately (market order), matching the backtest's
    "enter at the signal bar's close".
  * The 14:00 ET flat time is carried on each signal so the executor can
    enforce it.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.trading.intraday_fvg import (  # noqa: E402
    UNIVERSE_20,
    _atr,
    _fvg_zones_5m,
    fetch_5min_bars,
)

OUT = ROOT / "signals.json"

# Live execution parameters (mirroring the backtest defaults).
RISK_DOLLARS = 100.0
MAX_RISK_MULT = 2.0
MIN_VOLUME_RATIO = 1.0
MAX_CONCURRENT = 3
# Formation window 09:30-11:00 ET; entries allowed 09:30-14:00 ET.
FORMATION_START_MIN = 570   # 09:30
FORMATION_END_MIN = 660     # 11:00
ENTRY_END_MIN = 840         # 14:00
FLAT_TIME_ET = "14:00"


def _med_volume(v: np.ndarray, n: int = 20) -> np.ndarray:
    """Rolling median of the prior `n` bars' volume."""
    out = np.full(len(v), np.nan)
    for i in range(n, len(v)):
        out[i] = np.median(v[i - n:i])
    return out


def detect_live_signals(
    ticker: str,
    days: int = 5,
    risk_dollars: float = RISK_DOLLARS,
) -> List[dict]:
    """Generate intraday FVG signals for `ticker` from recent 5-min bars.

    Only signals from TODAY's session (ET) are returned — the strategy is
    intraday and the executor must enter them during that session.
    """
    df = fetch_5min_bars(ticker, days=days)
    if df is None or df.empty or len(df) < 25:
        return []

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    v = df["Volume"].to_numpy(dtype=float)
    ts_idx = df.index.to_numpy()
    n = len(c)

    atr14 = _atr(h, l, c, period=14)
    med_vol = _med_volume(v, n=20)
    bull_top, bull_bot, bear_top, bear_bot = _fvg_zones_5m(o, h, l, c, ts_idx)
    bull_valid = ~(np.isnan(bull_top) | np.isnan(bull_bot))
    bear_valid = ~(np.isnan(bear_top) | np.isnan(bear_bot))

    now_et = pd.Timestamp.now("America/New_York")
    today_et = now_et.date()

    signals: List[dict] = []
    open_trade = False  # one position per ticker at a time

    for i in range(2, n):
        ts = pd.Timestamp(ts_idx[i])
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/New_York")
        else:
            ts = ts.tz_convert("America/New_York")
        if ts.date() != today_et:
            continue

        total_min = ts.hour * 60 + ts.minute
        if total_min > ENTRY_END_MIN:
            continue

        if open_trade:
            continue

        bar_close = float(c[i])
        bar_vol = float(v[i])

        # Volume filter on the confirmation bar.
        vol_ok = (not np.isnan(med_vol[i])) and med_vol[i] > 0 and \
            bar_vol >= MIN_VOLUME_RATIO * med_vol[i]
        if not vol_ok:
            continue

        entry_atr = float(atr14[i]) if not np.isnan(atr14[i]) else \
            (float(h[i]) - float(l[i])) * 0.7
        if entry_atr <= 0:
            continue

        for direction, valid, zt_arr, zb_arr in (
            (1, bull_valid, bull_top, bull_bot),
            (-1, bear_valid, bear_top, bear_bot),
        ):
            if not valid[i - 1]:
                continue
            zone_top = zt_arr[i - 1]
            zone_bot = zb_arr[i - 1]
            if np.isnan(zone_top) or np.isnan(zone_bot):
                continue

            # Entry: bar closes beyond the zone boundary (displacement).
            if direction == 1:
                triggered = bar_close > zone_top
            else:
                triggered = bar_close < zone_bot
            if not triggered:
                continue

            stop_dist = entry_atr * 0.5
            if direction == 1:
                stop_px = round(bar_close - stop_dist, 4)
                target_px = round(bar_close + stop_dist * 2.0, 4)
            else:
                stop_px = round(bar_close + stop_dist, 4)
                target_px = round(bar_close - stop_dist * 2.0, 4)

            shares = int(risk_dollars / stop_dist)
            if shares < 1:
                continue
            notional = bar_close * shares
            if notional > risk_dollars * MAX_RISK_MULT:
                shares = int((risk_dollars * MAX_RISK_MULT) / bar_close)
                if shares < 1:
                    continue

            signals.append({
                "symbol": ticker,
                "direction": "long" if direction == 1 else "short",
                "entry_price": round(bar_close, 4),
                "stop_price": stop_px,
                "target_price": target_px,
                "atr": round(entry_atr, 4),
                "shares_hint": shares,
                # Risk sizing, so the live executor sizes positions exactly
                # as the backtest did (fixed risk per trade, notional
                # capped at 2x risk).
                "risk_dollars": risk_dollars,
                "max_risk_mult": MAX_RISK_MULT,
                # The signal bar's own session — the executor enters this
                # during TODAY's session, not the next one.
                "bar_date": str(today_et),
                "bar_time_et": ts.strftime("%H:%M"),
                "timestamp": datetime.now().isoformat(),
                "flat_time_et": FLAT_TIME_ET,
                "variant": "intraday_fvg_5m_displacement",
            })
            open_trade = True
            break

    return signals


def generate_signals(
    universe: Optional[List[str]] = None,
    days: int = 5,
    risk_dollars: float = RISK_DOLLARS,
    max_concurrent: int = MAX_CONCURRENT,
) -> List[dict]:
    """Scan the universe for intraday FVG signals (bull and bear)."""
    universe = universe or UNIVERSE_20
    signals: List[dict] = []
    for ticker in universe:
        try:
            sigs = detect_live_signals(ticker, days=days, risk_dollars=risk_dollars)
        except Exception as e:
            print(f"[ifvg_live] {ticker}: {e}")
            continue
        signals.extend(sigs)
        time.sleep(0.05)  # rate-limit yfinance

    # Cap concurrent positions to match the backtest.
    return signals[:max_concurrent]


def write_signals(signals: List[dict], path: Path = OUT) -> Path:
    """Persist signals to disk. The executor polls this file."""
    payload = {"signals": signals, "generated_at": datetime.now().isoformat()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


if __name__ == "__main__":
    sigs = generate_signals()
    write_signals(sigs)
    print(f"[ifvg_live] {len(sigs)} signals written to {OUT}")
    for s in sigs:
        print(f"  {s['symbol']} {s['direction']} entry={s['entry_price']} "
              f"stop={s['stop_price']} target={s['target_price']} @{s['bar_time_et']} ET")
