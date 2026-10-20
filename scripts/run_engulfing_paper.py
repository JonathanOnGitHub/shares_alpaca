#!/usr/bin/env python3
"""
run_engulfing_paper.py — Engulfing bull_overnight_tl10 Alpaca paper runner.

Usage:
    python scripts/run_engulfing_paper.py                  # signals + executor daemon
    python scripts/run_engulfing_paper.py --signals-only   # signals only
    python scripts/run_engulfing_paper.py --date 2026-10-19 # replay a specific date
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.trading.engulfing_live import generate_signals
from src.trading.executor_live import PaperExecutor

def run_signals(args):
    print("[run_engulfing] Generating engulfing signals...")
    signals = generate_signals(
        top_n=args.top_n,
        lookback=args.lookback,
        check_date=args.date,
    )
    print(f"[run_engulfing] {len(signals)} signal(s)")
    for s in signals:
        rr = abs(s["target_price"] - s["entry_price"]) / abs(s["entry_price"] - s["stop_price"])
        print(f"  {s['symbol']}: entry={s['entry_price']}  stop={s['stop_price']}  "
              f"target={s['target_price']}  R:R={rr:.1f}:1  atr={s['atr']}")
    return signals

def run_executor(args, signals):
    print("[run_engulfing] Starting PaperExecutor (engulfing bull_overnight_tl10)...")
    executor = PaperExecutor(
        signals_file=str(ROOT / "signals_engulfing.json"),
        broker="alpaca",
        market="US",
        poll_seconds=60,
        log_file=str(ROOT / "paper_trades_engulfing.csv"),
        strategy="engulfing",
        close_mode="overnight",
    )
    try:
        executor.run()
    except KeyboardInterrupt:
        executor.stop()
        print("[run_engulfing] Stopped.")

def main():
    parser = argparse.ArgumentParser(description="Engulfing bull_overnight_tl10 paper trader")
    parser.add_argument("--signals-only", action="store_true", help="Signals only, no executor")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--date", type=str, default=None,
                        help="Emit signals as-if on this date (YYYY-MM-DD)")
    args = parser.parse_args()

    signals = run_signals(args)

    if not args.signals_only:
        run_executor(args, signals)

if __name__ == "__main__":
    main()
