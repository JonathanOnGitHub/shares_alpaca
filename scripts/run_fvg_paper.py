#!/usr/bin/env python3
"""
run_fvg_paper.py — FVG signal generator + Alpaca paper executor daemon.

Usage:
    python scripts/run_fvg_paper.py                  # generate signals + run executor (daemon)
    python scripts/run_fvg_paper.py --signals-only    # signals only, no executor
    python scripts/run_fvg_paper.py --date 2026-10-20 # backtest a specific date (requires --cache)
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.trading.fvg_live import generate_signals
from src.trading.executor_live import PaperExecutor

def run_signals(args):
    """Generate and save signals.json."""
    print("[run_fvg_paper] Generating signals...")
    signals = generate_signals(
        top_n=args.top_n,
        lookback=args.lookback,
        use_cache=args.cache,
        check_date=args.date,
    )
    print(f"[run_fvg_paper] {len(signals)} signals written to signals.json")
    for s in signals:
        rr = abs(s["target_price"] - s["entry_price"]) / abs(s["entry_price"] - s["stop_price"])
        print(f"  {s['symbol']}: entry={s['entry_price']}  stop={s['stop_price']}  target={s['target_price']}  R:R={rr:.2f}")
    return signals

def run_executor(args, signals):
    """Start the PaperExecutor daemon."""
    print("[run_fvg_paper] Starting PaperExecutor (FVG intraday strategy)...")
    executor = PaperExecutor(
        signals_file=str(ROOT / "signals.json"),
        broker="alpaca",
        market="US",
        poll_seconds=60,
        log_file=str(ROOT / "paper_trades.csv"),
        strategy="fvg",
        close_mode="intraday",
    )
    try:
        executor.run()
    except KeyboardInterrupt:
        executor.stop()
        print("[run_fvg_paper] Stopped.")

def main():
    parser = argparse.ArgumentParser(description="FVG signal generator + Alpaca paper executor")
    parser.add_argument("--signals-only", action="store_true", help="Only generate signals, don't run executor")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N tickers by ADV (default 20)")
    parser.add_argument("--cache", action="store_true", help="Use cached universe_rankings.pkl instead of live yfinance")
    parser.add_argument("--date", type=str, default=None, help="As-if date YYYY-MM-DD (requires --cache)")
    parser.add_argument("--lookback", type=int, default=252, help="yfinance lookback bars (default 252)")
    args = parser.parse_args()

    if args.date and not args.cache:
        print("[run_fvg_paper] --date requires --cache; falling back to live mode")
        args.cache = False
        args.date = None

    signals = run_signals(args)

    if not args.signals_only:
        run_executor(args, signals)

if __name__ == "__main__":
    main()
