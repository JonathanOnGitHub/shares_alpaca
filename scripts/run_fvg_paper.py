#!/usr/bin/env python3
"""
FVG Alpaca Paper Trading Runner
===============================
1. Fetch live FVG signals for top-20 universe
2. Write signals.json
3. Start Alpaca PaperExecutor to execute signals

Usage:
    python scripts/run_fvg_paper.py                  # generate signals + run executor
    python scripts/run_fvg_paper.py --signals-only   # generate signals only (no executor)
    python scripts/run_fvg_paper.py --date 2025-12-30  # backtest mode (requires --cache)
"""

import argparse, json, sys, time
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))   # allow `from src` when run as script

def run():
    parser = argparse.ArgumentParser(description="FVG Alpaca Paper Trading Runner")
    parser.add_argument("--signals-only", action="store_true",
                        help="Only generate signals, don't start executor")
    parser.add_argument("--cache", action="store_true",
                        help="Use cached price data (backtest / non-trading-day mode)")
    parser.add_argument("--date", type=str, default=None,
                        help="Run as-if on date YYYY-MM-DD (requires --cache)")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    print(f"[{datetime.now():%H:%M:%S}] === FVG Paper Trading Runner ===")
    print(f"  Mode     : {'Signals only' if args.signals_only else 'Signals + Executor'}")
    print(f"  Universe : top-{args.top_n}")
    print(f"  Date     : {args.date or 'live'}")
    print()

    # ── Step 1: Generate signals ────────────────────────────────────────────
    from src.trading.fvg_live import generate_signals

    t0 = time.time()
    signals = generate_signals(
        top_n=args.top_n,
        use_cache=args.cache,
        check_date=args.date,
    )
    elapsed = time.time() - t0

    signals_file = ROOT / "signals.json"
    payload = {
        "signals": signals,
        "generated_at": datetime.now().isoformat(),
        "mode": "backtest" if args.cache else "live",
        "date": args.date or str(date.today()),
    }
    with open(signals_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[{datetime.now():%H:%M:%S}] Generated {len(signals)} signal(s) in {elapsed:.1f}s")
    for s in signals:
        print(f"  {s['symbol']}: entry={s['entry_price']:.4f}  "
              f"stop={s['stop_price']:.4f}  target={s['target_price']:.4f}  "
              f"R:R={abs(s['target_price']-s['entry_price'])/abs(s['entry_price']-s['stop_price']):.1f}:1")
    print()

    if args.signals_only:
        print("Signals written. Exiting (--signals-only).")
        return

    if not signals:
        print("No signals today. Executor would run but has nothing to do — exiting.")
        return

    # ── Step 2: Start executor ──────────────────────────────────────────────
    print(f"[{datetime.now():%H:%M:%S}] Starting Alpaca PaperExecutor ...")
    print(f"  Signals  : {signals_file}")
    print(f"  Polling : 60s")
    print()

    import os
    os.environ.setdefault("ALPACA_PAPER", "true")

    from src.trading.executor_live import PaperExecutor

    executor = PaperExecutor(
        signals_file=str(signals_file),
        broker="alpaca",
        market="US",
        poll_seconds=60,
        log_file=str(ROOT / "paper_trades.csv"),
    )

    try:
        executor.run()
    except KeyboardInterrupt:
        print(f"\n[{datetime.now():%H:%M:%S}] Interrupted — stopping executor.")
        executor.stop()


if __name__ == "__main__":
    run()
