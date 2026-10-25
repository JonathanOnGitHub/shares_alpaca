#!/usr/bin/env python3
"""
run_fvg_paper.py — Intraday FVG signal generator + Alpaca paper executor.

This drives the INTRADAY FVG strategy defined in
`src/trading/intraday_fvg.py` (the backtest that produced the reported
edge: 5-minute bars, displacement-confirmation entry, 0.5x ATR stop,
2:1 R:R, volume filter, flat by 14:00 ET).

It previously drove `src/trading/fvg_live.py`, which was a DIFFERENT
strategy — daily bars, a different FVG detector, 1.0x ATR stop, no
volume filter, no formation window and no 14:00 flat time. Trading a
different strategy than the one that was backtested is how a Sharpe
2.95 backtest becomes an unvalidated live trade.

Usage:
    python scripts/run_fvg_paper.py                  # generate signals + run executor (daemon)
    python scripts/run_fvg_paper.py --signals-only    # signals only, no executor
    python scripts/run_fvg_paper.py --once            # single scan, then exit
    python scripts/run_fvg_paper.py --poll            # scan continuously during the session
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.trading.intraday_fvg_live import generate_signals, write_signals, OUT as SIGNALS_OUT  # noqa: E402
from src.trading.executor_live import PaperExecutor  # noqa: E402


def run_signals(args):
    """Generate and save signals.json."""
    print("[run_fvg_paper] Generating intraday FVG signals...")
    signals = generate_signals(days=args.days)
    # The executor polls signals.json, so it MUST be written here.
    # Previously only the __main__ block inside fvg_live.py wrote the
    # file, which meant the cron job printed "1 signals written to
    # signals.json" while the executor kept polling a stale file and
    # never saw the signal.
    write_signals(signals, SIGNALS_OUT)
    print(f"[run_fvg_paper] {len(signals)} signals written to {SIGNALS_OUT}")
    for s in signals:
        rr = abs(s["target_price"] - s["entry_price"]) / abs(s["entry_price"] - s["stop_price"])
        print(f"  {s['symbol']} {s['direction']}: entry={s['entry_price']}  "
              f"stop={s['stop_price']}  target={s['target_price']}  "
              f"R:R={rr:.2f}  @{s['bar_time_et']} ET")
    return signals


def run_executor(args, signals):
    """Start the PaperExecutor daemon."""
    print("[run_fvg_paper] Starting PaperExecutor (intraday FVG)...")
    # The executor's logger must be configured HERE, not only in the
    # executor_live __main__ block. The daemon is launched via this
    # function, so basicConfig() never ran and every INFO line - the
    # order submissions, fills, skips and closes - was silently discarded.
    # That is why the logs showed no activity while the daemon was
    # quietly skipping signals.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    executor = PaperExecutor(
        signals_file=str(SIGNALS_OUT),
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


def poll_loop(args):
    """Rescan for signals on a fixed interval.

    The intraday FVG strategy is a 5-minute-bar strategy, so the signal
    set changes throughout the session. A single daily scan would miss
    most of the day's opportunities (and trade yesterday's setup).
    """
    print(f"[run_fvg_paper] Polling every {args.poll_seconds}s "
          f"until {args.until_et} ET")
    while True:
        now_et = pd.Timestamp.now("America/New_York")
        if now_et.hour * 60 + now_et.minute > args.until_et:
            print("[run_fvg_paper] Trading window over — exiting poll loop")
            return
        if now_et.hour * 60 + now_et.minute < 570:   # before 09:30 ET
            print("[run_fvg_paper] Pre-market — sleeping 60s")
            time.sleep(60)
            continue
        try:
            run_signals(args)
        except Exception as e:
            print(f"[run_fvg_paper] scan failed: {e}")
        time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Intraday FVG signal generator + Alpaca paper executor")
    parser.add_argument("--signals-only", action="store_true", help="Only generate signals, don't run executor")
    parser.add_argument("--once", action="store_true", help="Single signal scan, then exit")
    parser.add_argument("--poll", action="store_true", help="Rescan for signals throughout the session")
    parser.add_argument("--poll-seconds", type=int, default=300, help="Seconds between rescans (default 300)")
    parser.add_argument("--until-et", type=int, default=840, help="Stop scanning after this minute-of-day ET (default 840 = 14:00)")
    parser.add_argument("--days", type=int, default=5, help="Days of 5-min history to fetch (default 5)")
    args = parser.parse_args()

    if args.poll:
        poll_loop(args)
        return

    signals = run_signals(args)

    if args.once or args.signals_only:
        return

    run_executor(args, signals)


if __name__ == "__main__":
    main()
