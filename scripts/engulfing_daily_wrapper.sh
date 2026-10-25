#!/bin/bash
# engulfing_daily_wrapper.sh — run inside a cron no_agent job
# Generates engulfing bull_overnight_tl10 signals after market close,
# then launches the executor daemon to enter positions at next open.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PY="$REPO_DIR/.venv/bin/python"

LOGFILE="$REPO_DIR/executor_engulfing.log"
PIDFILE="$REPO_DIR/executor_engulfing.pid"

cd "$REPO_DIR"

# ── 0. Retire any daemon left over from a previous session ─────────────────
# The engulfing strategy is an OVERNIGHT strategy: its daemon is meant to
# survive the close and hold the position until the next session's exit.
# That makes it genuinely long-lived, so "is the PID alive" is NOT
# sufficient to decide whether it is still the daemon we want.
#
# A daemon that is alive but wedged is WORSE than no daemon: it holds a
# position with no stop, and the next run's wrapper sees a live PID,
# prints "executor already running" and skips the start — so the
# strategy stops trading while appearing healthy.
#
# We therefore treat a daemon as stale when it was started before
# today's 00:00 host-local time AND the account holds no position that
# it is managing. A daemon that still owns a live position is left
# alone; one that does not is killed so the next session starts clean.
if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        CMD="$(ps -o args= -p "$PID" 2>/dev/null || true)"
        case "$CMD" in
            *run_engulfing_paper.py*)
                STARTED="$(ps -o lstart= -p "$PID" 2>/dev/null | sed 's/^ *//')"
                START_EPOCH="$(date -d "$STARTED" +%s 2>/dev/null || echo 0)"
                TODAY_EPOCH="$(date -d "$(date +%Y-%m-%d) 00:00:00" +%s)"
                if [ "$START_EPOCH" -lt "$TODAY_EPOCH" ]; then
                    # The daemon is from an earlier session. Before killing
                    # it, check whether it still OWNS a live position at
                    # the broker — an overnight position is meant to be
                    # held across sessions, and killing its daemon would
                    # strand it with no stop and no exit.
                    HAS_POS="$("$VENV_PY" -c "
import os, sys
sys.path.insert(0, '$REPO_DIR')
try:
    from dotenv import load_dotenv
    load_dotenv()
    from alpaca.trading.client import TradingClient
    c = TradingClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_SECRET_KEY'], paper=True)
    ps = [p for p in c.get_all_positions() if abs(float(p.qty)) > 0]
    print(len(ps))
except Exception:
    print(0)
" 2>/dev/null || echo 0)"
                    if [ "${HAS_POS:-0}" != "0" ]; then
                        echo "$(date -Iseconds) — engulfing executor PID $PID is old but the account holds $HAS_POS live position(s) — leaving it running" >> "$LOGFILE"
                    else
                        echo "$(date -Iseconds) — engulfing executor PID $PID started '$STARTED' (before today), account flat — killing stale daemon" >> "$LOGFILE"
                        kill "$PID" 2>/dev/null || true
                        sleep 2
                        if kill -0 "$PID" 2>/dev/null; then
                            kill -9 "$PID" 2>/dev/null || true
                            sleep 1
                        fi
                        echo "$(date -Iseconds) — stale engulfing executor PID $PID reaped" >> "$LOGFILE"
                    fi
                else
                    echo "$(date -Iseconds) — engulfing executor PID $PID is from today's session — reusing" >> "$LOGFILE"
                fi
                ;;
            *)
                echo "$(date -Iseconds) — engulfing PID $PID is not the engulfing daemon ('$CMD') — clearing pidfile, not killing" >> "$LOGFILE"
                ;;
        esac
    else
        echo "$(date -Iseconds) — engulfing executor PID ${PID:-?} is not running — clearing pidfile" >> "$LOGFILE"
    fi
    rm -f "$PIDFILE"
fi

# ── 1. Generate signals ───────────────────────────────────────────────────────
# yfinance daily bars update ~16:00 ET (21:00 BST); run at 17:00 BST to be safe.
echo "=== $(date -Iseconds) — engulfing daily run ===" >> "$LOGFILE"
"$VENV_PY" "$SCRIPT_DIR/run_engulfing_paper.py" --signals-only 2>&1 | tee -a "$LOGFILE"

# ── 2. P&L report ───────────────────────────────────────────────────────────
"$VENV_PY" -c "
import sys, os; sys.path.insert(0, '$REPO_DIR')
from pathlib import Path
log = Path('$REPO_DIR/paper_trades_engulfing.csv')
if log.exists():
    import pandas as pd
    df = pd.read_csv(log)
    if len(df):
        total = df['pnl_bps'].sum()
        n = len(df)
        wins = (df['pnl_bps'] > 0).sum()
        print(f'Engulfing equity curve: {n} trades | winrate={wins/n*100:.0f}% | total_pnl_bps={total:.1f} | last_exit={df.iloc[-1][\"exit_reason\"]}')
    else:
        print('Engulfing: no trades yet')
else:
    print('Engulfing: no trades yet (no log file)')
" 2>&1 | tee -a "$LOGFILE"

# ── 3. Start executor daemon (background) ────────────────────────────────────
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "$(date -Iseconds) — executor already running (PID $(cat "$PIDFILE"))" >> "$LOGFILE"
else
    # setsid detaches the daemon into its own session/process group so it
    # survives the cron runner's exit. Plain `nohup ... &` is not enough when
    # the runner captures stdout via a pipe — the child keeps the pipe open
    # and the parent blocks until timeout.
    setsid "$VENV_PY" "$SCRIPT_DIR/run_engulfing_paper.py" >> "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "$(date -Iseconds) — executor started PID $!" >> "$LOGFILE"
fi
