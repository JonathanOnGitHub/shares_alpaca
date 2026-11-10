#!/usr/bin/env python3
"""
run_engulfing_paper.py — Engulfing bull_overnight_tl10 Alpaca paper runner.

Usage:
    python scripts/run_engulfing_paper.py                  # signals + executor daemon
    python scripts/run_engulfing_paper.py --signals-only   # signals only
    python scripts/run_engulfing_paper.py --date 2026-10-19 # replay a specific date
    python scripts/run_engulfing_paper.py --executor-only  # executor daemon only

--executor-only exists because the systemd unit
shares-alpaca-engulfing-paper.service passes it. The unit previously
crash-looped with status=2/INVALIDARGUMENT because argparse rejected
the unknown flag and exited — every restart re-read the same flag, so
the daemon never ran and the overnight positions it should have
managed were left orphaned at the broker.

The daily wrapper owns signal generation; the executor only consumes
signals_engulfing.json. Regenerating signals from the executor would
overwrite the file mid-session and reload the whole yfinance universe
into the executor's own memory.
"""
import argparse
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.trading.engulfing_live import generate_signals
from src.trading.executor_live import PaperExecutor
from src.trading.engulfing_live import OUT as SIGNALS_OUT
from src.trading.process_lock import acquire_lock, lock_holder_pid

# Advisory flock for the engulfing executor. On 2026-11-07 TWO
# engulfing daemons were running at once (PIDs 113311 and 537265),
# because the cron wrapper's PID file is overwritten on every run and
# the wrapper's own reaper only looks at that PID file. Two daemons for
# one strategy each keep their own `_acted_signals` set and their own
# position file, so they can double-enter a setup and overwrite each
# other's view of an open position, leaving it unmanaged at the broker.
# An flock is released by the kernel even on SIGKILL, so there is no
# stale lock to clean up after an OOM kill.
EXECUTOR_LOCK = ROOT / ".engulfing_executor.lock"

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

    # Write the signals file the PaperExecutor polls. Without this the
    # executor starts but never sees a signal, because only the
    # engulfing_live __main__ block wrote signals_engulfing.json.
    payload = {
        "signals": signals,
        "generated_at": datetime.now().isoformat(),
        "variant": "bull_overnight_tl10",
    }
    with open(SIGNALS_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[run_engulfing] {len(signals)} signal(s) written to {SIGNALS_OUT}")

    return signals

def run_executor(args, signals=None):
    print("[run_engulfing] Starting PaperExecutor (engulfing bull_overnight_tl10)...")
    # basicConfig() lives in the executor_live __main__ block, which never
    # runs for this daemon, so every INFO line was silently dropped. The
    # log file therefore showed no orders, fills or skips — which is how a
    # daemon that was skipping every signal on a stale-timestamp rule
    # looked identical to an idle one.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    executor = PaperExecutor(
        signals_file=str(ROOT / "signals_engulfing.json"),
        broker="alpaca",
        market="US",
        poll_seconds=60,
        log_file=str(ROOT / "paper_trades_engulfing.csv"),
        strategy="engulfing",
        close_mode="overnight",
    )

    # systemd stops the service with SIGTERM, so the daemon MUST handle
    # it. Without a handler, SIGTERM kills the process outright: the
    # trade log is not written, open positions are not flattened, and
    # the broker-side protective stop is left working against a
    # position this process still believes it owns.
    #
    # This handler was dropped in commit 7aa7b4c (which rewrote this
    # file to add the --executor-only flag and the flock). Its absence
    # is what produced the 2026-10-30 and 2026-11-04 "force_close"
    # exits — the daemon was killed mid-position and the next start
    # found the position gone but the ledger never updated.
    #
    # executor.run() calls _close_all() and _write_log() on exit, so
    # stopping the executor cleanly is what records the trades.
    def _sig_handler(signum, frame):
        print(f"[run_engulfing] Signal {signum} received — shutting down "
              f"gracefully (flattening positions, writing the trade log).",
              flush=True)
        executor.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

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
    parser.add_argument("--executor-only", action="store_true",
                        help="Run ONLY the executor daemon; do not generate signals. "
                             "The daily wrapper owns signal generation.")
    args = parser.parse_args()

    if args.executor_only:
        lock = acquire_lock(EXECUTOR_LOCK)
        if lock is None:
            other = lock_holder_pid(EXECUTOR_LOCK)
            print(f"[run_engulfing] Another engulfing executor already holds "
                  f"{EXECUTOR_LOCK} (pid {other}) — refusing to start a "
                  f"second daemon for the same strategy.")
            return
        run_executor(args)
        return

    signals = run_signals(args)

    if not args.signals_only:
        run_executor(args, signals)

if __name__ == "__main__":
    main()
