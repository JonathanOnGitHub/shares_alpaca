#!/bin/bash
# engulfing_daily_wrapper.sh — run inside a cron no_agent job
# Generates engulfing bull_overnight_tl10 signals after market close,
# then launches the executor daemon to enter positions at next open.
#
# ── WHY THIS WRAPPER NOW DELEGATES TO SYSTEMD ─────────────────────────────
# This wrapper used to start the daemon itself with `setsid`. Two
# problems came out of that:
#
#   1. DUPLICATE DAEMONS. The PID file was OVERWRRITTEN on every run,
#      so the daemon started by yesterday's run became invisible to
#      today's "is it already running?" check, and a second daemon
#      started alongside it. On 2026-11-07 TWO run_engulfing_paper.py
#      daemons were running at the same time (PIDs 113311 and 537265).
#      Each daemon keeps its own _acted_signals set and its own
#      .positions_engulfing.json, so they can double-enter a setup and
#      overwrite each other's view of an open position — leaving that
#      position unmanaged at the broker with no stop and no target.
#
#   2. THE CGROUP PROBLEM. setsid detaches a process from its
#      terminal and session, but NOT from the systemd cgroup it was
#      launched in, so the daemon stayed inside hermes-gateway.service's
#      cgroup. When the kernel OOM killer hit the gateway (7.4G peak on
#      28 and 29 Oct), systemd tore the whole cgroup down with SIGKILL
#      and the daemon died with it — on both mornings — and never
#      traded.
#
# The systemd unit now owns the daemon: it runs in its own cgroup, and
# the runner's flock (src/trading/process_lock.py) guarantees that only
# one daemon per strategy can ever hold the strategy's signals file.
#
# If systemd is unavailable (no user session bus), it falls back to the
# old setsid path — the flock still prevents a duplicate daemon, so the
# fallback is safe.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PY="$REPO_DIR/.venv/bin/python"

LOGFILE="$REPO_DIR/executor_engulfing.log"
PIDFILE="$REPO_DIR/executor_engulfing.pid"

cd "$REPO_DIR"

SYSTEMD_OK=0
if systemctl --user is-system-running >/dev/null 2>&1 || \
   systemctl --user list-unit-files >/dev/null 2>&1; then
    SYSTEMD_OK=1
fi

# ── 1. Generate signals ───────────────────────────────────────────────────────
# yfinance daily bars update ~16:00 ET (21:00 BST); run at 17:00 BST to be safe.
echo "=== $(date -Iseconds) — engulfing daily run ===" >> "$LOGFILE"
"$VENV_PY" "$SCRIPT_DIR/run_engulfing_paper.py" --signals-only 2>&1 | tee -a "$LOGFILE"

# ── 2. P&L report ────────────────────────────────────────────────────────────
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

# ── 3. Start the executor daemon ─────────────────────────────────────────────
if [ "$SYSTEMD_OK" = "1" ]; then
    # ── DO NOT RESTART A DAEMON THAT IS HOLDING A POSITION ──────────────
    #
    # The engulfing strategy is an OVERNIGHT strategy: it enters at the
    # open and exits at the NEXT session's open. Its daemon is therefore
    # meant to be alive across the close, holding the position.
    #
    # `systemctl --user restart` was unconditional, so every daily run
    # killed the daemon mid-position. With no SIGTERM handler (dropped
    # in 7aa7b4c) the process died outright: no trade log, no flat, and
    # the position left behind with no stop. Even with the handler
    # restored, a restart would force-close an overnight position that
    # the backtest expects to be held — the single largest source of
    # divergence between the backtest and the live book.
    #
    # So: if the daemon is already running AND is holding a position,
    # LEAVE IT ALONE. It continues to manage that position (and its
    # broker-side protective stop) into the next session.
    HOLDING=0
    if [ -f "$REPO_DIR/.positions_engulfing.json" ]; then
        HOLDING=$("$VENV_PY" -c "
import json,sys
try:
    d=json.load(open('$REPO_DIR/.positions_engulfing.json'))
    pos=d.get('positions') or {}
    live=[s for s,p in pos.items() if p.get('status') in ('pending','entered')]
    print(len(live))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
    fi

    if [ "$HOLDING" -gt 0 ] && \
       systemctl --user is-active --quiet shares-alpaca-engulfing-paper.service; then
        PID="$(systemctl --user show -p MainPID --value shares-alpaca-engulfing-paper.service)"
        echo "$PID" > "$PIDFILE"
        echo "$(date -Iseconds) — engulfing daemon already running (PID $PID) and holding $HOLDING position(s) — NOT restarting" >> "$LOGFILE"
    else
        # `restart` (not `start`) so a daemon left over from a previous
        # session — including one wedged on a pending position — is
        # replaced rather than joined.
        systemctl --user restart shares-alpaca-engulfing-paper.service >> "$LOGFILE" 2>&1
        systemctl --user show -p MainPID --value shares-alpaca-engulfing-paper.service > "$PIDFILE" 2>/dev/null || true
        echo "$(date -Iseconds) — engulfing systemd unit restarted (PID $(cat "$PIDFILE" 2>/dev/null))" >> "$LOGFILE"
    fi
else
    setsid "$VENV_PY" "$SCRIPT_DIR/run_engulfing_paper.py" --executor-only \
        >> "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "$(date -Iseconds) — engulfing fallback daemon started PID $!" >> "$LOGFILE"
fi
