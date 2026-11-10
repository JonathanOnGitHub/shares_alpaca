#!/bin/bash
# engulfing_daily_wrapper.sh — run inside a cron no_agent job
# Generates engulfing bull_overnight_tl10 signals after market close,
# then hands the executor daemon over to systemd.
#
# WHY systemd instead of setsid + a PID file:
# On 28 and 29 Oct 2026 the kernel OOM killer hit hermes-gateway.service
# (7.4G memory peak) and systemd tore down the whole gateway cgroup with
# SIGKILL. The engulfing daemon had been started by this very cron job
# via `setsid`, and setsid detaches a process from the terminal and the
# session — but NOT from the systemd cgroup it was started in. So the
# daemon was killed on both mornings, before it could act on its signal,
# and the strategy recorded zero trades.
#
# The daemon now runs in its own unit (shares-alpaca-engulfing-paper.service),
# which is NOT in the gateway's cgroup, so an OOM kill or a gateway
# restart leaves the trader alone. This wrapper only:
#   1. generates the signals (once, after the close),
#   2. makes sure the unit is installed, enabled and running,
#   3. prints the P&L report.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PY="$REPO_DIR/.venv/bin/python"

LOGFILE="$REPO_DIR/executor_engulfing.log"
UNIT="shares-alpaca-engulfing-paper.service"
UNIT_SRC="$REPO_DIR/deploy/$UNIT"
UNIT_DST="$HOME/.config/systemd/user/$UNIT"

cd "$REPO_DIR"

# ── 0. Make sure the user systemd manager is reachable ──────────────────────
# The Hermes cron runner inherits the gateway's environment, which does
# carry XDG_RUNTIME_DIR/DBUS_SESSION_BUS_ADDRESS. But this wrapper is also
# run by hand from plain shells, where those may be missing. `systemctl
# --user` needs them to find the user manager, so set them explicitly.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

systemctl_user() {
    systemctl --user "$@"
}

# ── 1. Install / refresh the unit if the file changed ─────────────────────
mkdir -p "$HOME/.config/systemd/user"
if [ ! -f "$UNIT_DST" ] || ! cmp -s "$UNIT_SRC" "$UNIT_DST"; then
    cp "$UNIT_SRC" "$UNIT_DST"
    systemctl_user daemon-reload
    echo "$(date -Iseconds) — installed/refreshed $UNIT" >> "$LOGFILE"
fi

# ── 2. Generate signals ────────────────────────────────────────────────────
# yfinance daily bars update ~16:00 ET (21:00 BST); run at 17:00 BST to be safe.
echo "=== $(date -Iseconds) — engulfing daily run ===" >> "$LOGFILE"
"$VENV_PY" "$SCRIPT_DIR/run_engulfing_paper.py" --signals-only 2>&1 | tee -a "$LOGFILE"

# ── 3. Ensure the executor daemon is running under systemd ────────────────
# `enable` makes it survive a reboot; `restart` guarantees a fresh daemon
# that has just read the signal file written above.
#
# A daemon that is alive but wedged is WORSE than no daemon: it holds a
# position with no stop, and the next run's wrapper sees a live PID,
# prints "executor already running" and skips the start — so the strategy
# stops trading while appearing healthy. Restarting unconditionally avoids
# inheriting a stale process. The daemon re-reads its position file on
# startup, so it re-adopts any position it already owns.

# Reap any LEGACY setsid daemon first. Before the systemd unit existed the
# daemon was started by this same wrapper with setsid, which left it inside
# the gateway's cgroup. Leaving it running would give two executors watching
# the same signal file, and both would open the position.
for PID in $(pgrep -f "run_engulfing_paper\.py" 2>/dev/null || true); do
    if [ "$PID" = "$$" ]; then
        continue
    fi
    # Only kill it if it is NOT the systemd unit's own main process.
    UNIT_PID="$(systemctl_user show -p MainPID --value "$UNIT" 2>/dev/null || echo 0)"
    if [ "$UNIT_PID" = "0" ] || [ "$PID" != "$UNIT_PID" ]; then
        echo "$(date -Iseconds) — reaping legacy setsid engulfing daemon PID $PID" >> "$LOGFILE"
        kill "$PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
    fi
done
rm -f "$REPO_DIR/executor_engulfing.pid"

systemctl_user enable "$UNIT" >> "$LOGFILE" 2>&1

# ── DO NOT RESTART A DAEMON THAT IS HOLDING A POSITION ──────────────────────
#
# The engulfing strategy is an OVERNIGHT strategy: it enters at the open
# and exits at the NEXT session's open. Its daemon is therefore meant to
# be alive across the close, holding the position.
#
# An unconditional `restart` kills the daemon mid-position. The executor
# now handles SIGTERM and leaves an overnight position open (still
# protected by its broker-side stop, and re-adopted by the next daemon),
# but the position's clock keeps running and a needless restart is one
# more chance to lose it. So: if the daemon is already active and
# holding a position, leave it alone.
#
# Everything else still restarts unconditionally — a wedged daemon that
# still holds a PID is worse than no daemon at all.
HOLDING=0
if [ -f "$REPO_DIR/.positions_engulfing.json" ]; then
    HOLDING=$("$VENV_PY" -c "
import json
try:
    d = json.load(open('$REPO_DIR/.positions_engulfing.json'))
    live = [s for s, p in (d.get('positions') or {}).items()
            if p.get('status') in ('pending', 'entered')]
    print(len(live))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
fi

if [ "$HOLDING" -gt 0 ] && systemctl_user is-active --quiet "$UNIT"; then
    MAIN_PID="$(systemctl_user show -p MainPID --value "$UNIT" 2>/dev/null || echo '?')"
    echo "$(date -Iseconds) — $UNIT active (PID $MAIN_PID) holding $HOLDING position(s) — NOT restarting" >> "$LOGFILE"
else
    systemctl_user restart "$UNIT" >> "$LOGFILE" 2>&1
    sleep 2

    if systemctl_user is-active --quiet "$UNIT"; then
        MAIN_PID="$(systemctl_user show -p MainPID --value "$UNIT" 2>/dev/null || echo '?')"
        echo "$(date -Iseconds) — $UNIT active (PID $MAIN_PID)" >> "$LOGFILE"
    else
        echo "$(date -Iseconds) — WARNING: $UNIT is NOT active after restart" >> "$LOGFILE"
        systemctl_user status "$UNIT" --no-pager >> "$LOGFILE" 2>&1 || true
    fi
fi

# ── 4. P&L report ──────────────────────────────────────────────────────────
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
