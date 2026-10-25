"""
FVG Live Signal Generator
========================
Produces signals.json for the PaperExecutor each trading day.

Universe : top-20 by ADV (cached in data/universe_rankings.pkl)
Variant  : combined_bull_intraday  (gap=0 bps, bull FVG only)
Stop     : 1 × ATR14 below entry
Target   : 2 × ATR14 above entry  (2:1 R:R)
Exit     : Same bar (intraday) — executor closes at 4PM EST

Live mode   : fetches fresh OHLCV from yfinance each run (no stale-cache risk)
Backtest mode : use --cache to fall back to pickled price data
"""

import argparse, json, pickle, numpy as np, yfinance as yf
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import sys, time

# ── Project paths ────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))   # allow `from src` when run as script
CACHE     = ROOT / "data" / "engulfing_universe.pkl"
RANKINGS  = ROOT / "data" / "universe_rankings.pkl"
OUT       = ROOT / "signals.json"

# ── ATR helper ────────────────────────────────────────────────────────────────
def atr14(high, low, close) -> float:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    return float(np.nanmean(tr[-14:]))

# ── FVG detection ─────────────────────────────────────────────────────────────
def detect_bull_fvg(df):
    """
    Returns list of dicts with entry_price and stop_price for each bull FVG.
    Bull FVG: bar i-1 body entirely above bar i-2 body (gap).
              Bar i   fully retraces bar i-1 body back into the gap zone.
    Entry   : price closes above bar i-1 high.
    """
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    dates = df.index  # DatetimeIndex for bar dates

    zone = []  # active FVG zone stack (top, bot)
    for i in range(2, len(c)):
        bull_gap = (l[i-1] > h[i-2])   # body entirely above prior bar's high

        if bull_gap:
            zone.append((h[i-1], l[i-1]))     # push FVG zone

        # Bar i: up bar filling the zone — close above bar i-1 high
        elif zone and c[i] > h[i-1] and c[i] > o[i]:
            zone.pop(0)   # consume the oldest zone on entry
            entry = float(c[i])
            _atr  = atr14(h[max(0,i-14):i+1], l[max(0,i-14):i+1], c[max(0,i-14):i+1])
            stop  = float(entry - _atr)
            yield {
                "entry_price": entry,
                "stop_price":  stop,
                "atr":         float(_atr),
                "bar_date":    str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])[:10],
            }

def fetch_live_bars(ticker: str, lookback: int = 252) -> pd.DataFrame | None:
    """Fetch recent daily OHLCV from yfinance."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="1y", auto_adjust=True)  # ~252 trading days
        if df.empty or len(df) < 30:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def generate_signals(top_n: int = 20, lookback: int = 252, use_cache: bool = False,
                     check_date: str | None = None) -> list[dict]:
    """
    Scan top_n universe tickers for bull FVGs.

    Live mode (default)  : fetches fresh bars from yfinance.
    Cache mode --cache   : falls back to pickled price data for backtesting.
    --date YYYY-MM-DD    : run as-if on that date (checks bars up to that date).
    """
    cache = pickle.load(open(CACHE, "rb")) if use_cache else {}
    ranked = pickle.load(open(RANKINGS, "rb"))
    universe = [t for t in ranked[:top_n]]

    signals = []
    if check_date:
        target_date = pd.Timestamp(check_date).date()
    else:
        target_date = None

    for ticker in universe:
        # ── fetch data ──────────────────────────────────────────────────────────
        if use_cache:
            if ticker not in cache:
                continue
            df = cache[ticker].dropna()
        else:
            df = fetch_live_bars(ticker, lookback)
            if df is None:
                continue

        # Use last `lookback` bars to keep things fast
        df = df.iloc[-lookback:]
        if len(df) < 30:
            continue

        last_bar_date = df.index[-1].date()

        # Determine the reference date for signal emission.
        #
        # The reference date must be the last COMPLETED daily bar, not
        # simply df.index[-1]. yfinance includes today's partial bar
        # during the trading session, so the last row is an unfinished
        # candle. Emitting a signal off an unfinished bar means the
        # FVG "entry" is a live intraday price that can still move, and
        # the bar_date is today rather than a settled session.
        #
        # Use the last bar that closed on or before the previous session
        # when today is still in progress.
        if target_date:
            ref_date = target_date
        else:
            now_et = pd.Timestamp.now("America/New_York")
            today_et = now_et.date()
            ref_date = last_bar_date
            if last_bar_date == today_et and now_et.hour < 16:
                # Today's bar is still forming — step back one session.
                closed = [d.date() for d in df.index
                          if d is not None and pd.notna(d) and d.date() < today_et]
                if not closed:
                    continue
                ref_date = max(closed)

        for sig in detect_bull_fvg(df):
            bar_date = pd.Timestamp(sig["bar_date"]).date()
            if bar_date != ref_date:
                continue

            atr_val = sig["atr"]
            entry   = round(sig["entry_price"], 2)
            stop    = round(sig["stop_price"], 2)
            target  = entry + 2.0 * (entry - stop)   # 2:1 R:R

            signals.append({
                "symbol":        ticker,
                "direction":     "long",
                "entry_price":  entry,
                "stop_price":   stop,
                "target_price": round(target, 2),
                "atr":          round(atr_val, 4),
                "timestamp":     datetime.now().isoformat(),
                # The bar the FVG fired on. The executor needs this to work
                # out which session the trade is entered in (the open of the
                # session AFTER the signal bar).
                "bar_date":      str(sig["bar_date"]),
                "variant":       "combined_bull_intraday",
            })

        # Rate-limit yfinance to avoid getting blocked
        if not use_cache:
            time.sleep(0.05)

    return signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FVG Live Signal Generator")
    parser.add_argument("--top-n",    type=int, default=20)
    parser.add_argument("--cache",     action="store_true",
                        help="Use cached price data instead of fetching live yfinance data")
    parser.add_argument("--date",      type=str, default=None,
                        help="Run as-if on this date (YYYY-MM-DD). Requires --cache.")
    parser.add_argument("--lookback",  type=int, default=252)
    args = parser.parse_args()

    if args.date and not args.cache:
        print("[FVG] --date requires --cache (yfinance only provides current/recent data)")
        exit(1)

    signals = generate_signals(
        top_n=args.top_n,
        lookback=args.lookback,
        use_cache=args.cache,
        check_date=args.date,
    )

    payload = {"signals": signals, "generated_at": datetime.now().isoformat()}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[FVG Live] {len(signals)} signal(s) written to {OUT}")
    for s in signals:
        r  = s["entry_price"]
        sl = s["stop_price"]
        tp = s["target_price"]
        print(f"  {s['symbol']}: entry={r:.4f}  stop={sl:.4f}  target={tp:.4f}  "
              f"R:R={abs(tp-r)/abs(r-sl):.1f}:1  atr={s['atr']:.4f}")
