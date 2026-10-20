"""
Engulfing Live Signal Generator
================================
Variant : bull_overnight_tl10  (bullish engulfing after 10-day downtrend)
Entry   : open of next day
Exit    : next day close  OR  stop/target hit (whichever comes first)
Stop    : H[i] - 1 × ATR14
Target  : entry + 2 × risk

Shares_alpaca layout — writes signals_engulfing.json.
Cron: weekdays ~17:00 BST (12:00 ET) to catch today's engulfing.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.alpaca_client import AlpacaClient

OUT = ROOT / "signals_engulfing.json"
RANKINGS = ROOT / "data" / "universe_rankings.pkl"

# ── helpers ──────────────────────────────────────────────────────────────────

def _atr14(high, low, close) -> float:
    """Wilder-smoothed ATR14 — last value only. Accepts array-like input."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i],
                    abs(h[i] - c[i - 1]),
                    abs(l[i] - c[i - 1]))
    if n < 15:
        return float(np.mean(tr))
    atr = np.zeros(n)
    atr[14] = np.mean(tr[1:15])
    for i in range(15, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    return float(atr[-1])


def _detect_engulfing(df: pd.DataFrame, trend_lookback: int = 10) -> list[dict]:
    """
    Scan daily bars for bullish engulfing + 10-day downtrend.
    Returns list of signal dicts (one per engulfing bar found today).
    """
    df = df.dropna().copy()
    df = df.sort_index()
    if len(df) < trend_lookback + 5:
        return []

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    dates = df.index  # DatetimeIndex

    today = pd.Timestamp.now().date()
    signals = []

    for i in range(1, len(c) - 1):
        bar_ts = pd.Timestamp(dates[i])
        bar_date = bar_ts.date()

        # Only emit signal if engulfing formed today
        if bar_date != today:
            continue

        p_o, p_c = o[i - 1], c[i - 1]   # prior candle
        c_o, c_c = o[i],     c[i]        # current (engulfing) candle

        # Bullish engulfing: prior red, current body fully engulfs prior body
        if not (p_c < p_o and c_c > p_o and c_c > p_c):
            continue

        # Trend filter: 10-day downtrend (price lower now than 10 days ago)
        if trend_lookback > 0 and i >= trend_lookback:
            if c[i] >= c[i - trend_lookback]:
                continue  # not in downtrend

        # ATR for stop sizing
        atr_val = _atr14(
            h[max(0, i - 14):i + 1],
            l[max(0, i - 14):i + 1],
            c[max(0, i - 14):i + 1],
        )

        # Entry: next day open (backtest uses next bar open; live proxy = today's close)
        entry_price = float(c[i])
        stop_price  = float(h[i]) - atr_val
        risk        = abs(entry_price - stop_price)
        target      = entry_price + 2.0 * risk

        signals.append({
            "direction":     "long",
            "entry_price":   round(entry_price, 2),
            "stop_price":    round(stop_price, 2),
            "target_price":  round(target, 2),
            "atr":           round(atr_val, 4),
            "bar_date":      str(bar_date),
            "variant":       "bull_overnight_tl10",
        })

    return signals


def fetch_bars(ticker: str, lookback: int = 252) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from yfinance."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="1y", auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def generate_signals(top_n: int = 20, lookback: int = 252,
                     check_date: Optional[str] = None) -> list[dict]:
    """
    Scan top_n universe for today's bullish engulfing patterns.
    Emits one signal per ticker (most recent engulfing bar today).
    """
    import pickle
    ranked = pickle.load(open(RANKINGS, "rb"))
    universe = [t for t in ranked[:top_n]]

    signals = []
    today = pd.Timestamp(check_date).date() if check_date else pd.Timestamp.now().date()

    for ticker in universe:
        df = fetch_bars(ticker, lookback)
        if df is None:
            continue

        # Inject a check_date into the index for backtesting
        if check_date:
            # Slice data up to check_date
            df = df[df.index.date <= pd.Timestamp(check_date).date()]
            if len(df) < 30:
                continue

        last_date = df.index[-1].date()

        # For live: engulfing must have fired today
        # For backtest/check_date: engulfing must have fired on check_date
        if check_date is None and last_date != today:
            # No fresh data today — skip (weekend/holiday)
            continue

        sigs = _detect_engulfing(df, trend_lookback=10)
        if not sigs:
            continue

        # Attach ticker to each signal (most recent engulfing only)
        sig = sigs[-1].copy()
        sig["symbol"] = ticker
        sig["timestamp"] = datetime.now().isoformat()
        # For overnight: force timestamp to today's midnight so executor's staleness
        # check (which uses bars_old = elapsed_seconds/poll_interval) treats the
        # signal as fresh even when executor runs next morning (~12 h later).
        # The executor uses signal timestamp only for staleness; the actual entry
        # is a market order at next open.
        today_midnight = pd.Timestamp.now().normalize()
        sig["timestamp"] = today_midnight.isoformat()
        signals.append(sig)

        time.sleep(0.05)   # rate-limit yfinance

    return signals


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Engulfing bull_overnight_tl10 signal generator")
    parser.add_argument("--top-n",     type=int, default=20)
    parser.add_argument("--lookback",  type=int, default=252)
    parser.add_argument("--date",      type=str, default=None,
                        help="Emit signals as-if on this date (YYYY-MM-DD, requires cached data)")
    args = parser.parse_args()

    signals = generate_signals(
        top_n=args.top_n,
        lookback=args.lookback,
        check_date=args.date,
    )

    payload = {
        "signals": signals,
        "generated_at": datetime.now().isoformat(),
        "variant": "bull_overnight_tl10",
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[engulfing_live] {len(signals)} signal(s) written to {OUT}")
    for s in signals:
        rr = abs(s["target_price"] - s["entry_price"]) / abs(s["entry_price"] - s["stop_price"])
        print(f"  {s['symbol']}: entry={s['entry_price']}  stop={s['stop_price']}  "
              f"target={s['target_price']}  R:R={rr:.1f}:1  atr={s['atr']}")
