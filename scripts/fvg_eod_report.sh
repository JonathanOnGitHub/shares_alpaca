#!/bin/bash
# fvg_eod_report.sh — end-of-day P&L report for the intraday FVG strategy.
#
# Runs at 21:00 BST (16:00 ET), just after the US close. The intraday
# executor exits by itself once the session is over and flat, so by this
# point every trade of the day has been closed and logged.
#
# The DAILY wrapper (fvg_daily_wrapper.sh) runs at 14:00 BST, before the
# open, and reports the account as it stood at the start of the session.
# This one reports the session's result.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PY="$REPO_DIR/.venv/bin/python"

cd "$REPO_DIR"

echo "=== $(date -Iseconds) — FVG end-of-day report ===" >> "$REPO_DIR/executor.log"

# The intraday executor exits by itself once the session is over. If it
# is still alive (wedged on a pending position, or killed mid-session),
# stop it now so the next session starts from a clean process.
#
# Since 2026-11-07 the daemon is owned by
# shares-alpaca-fvg-executor.service, so it is stopped with
# `systemctl --user stop` rather than by killing the PID directly.
# Killing the PID behind systemd's back leaves the unit in a state
# where systemd's Restart=on-failure policy can immediately respawn
# the executor mid-report.
if systemctl --user is-active --quiet shares-alpaca-fvg-executor.service 2>/dev/null; then
    echo "stopping FVG executor unit at end of session" >> "$REPO_DIR/executor.log"
    systemctl --user stop shares-alpaca-fvg-executor.service >> "$REPO_DIR/executor.log" 2>&1
    sleep 2
fi

# Also stop the scanner if it is somehow still running (it exits by
# itself at 14:00 ET, but a wedged scanner would otherwise regenerate
# signals.json against a closed market).
if systemctl --user is-active --quiet shares-alpaca-fvg-scanner.service 2>/dev/null; then
    echo "stopping FVG scanner unit at end of session" >> "$REPO_DIR/executor.log"
    systemctl --user stop shares-alpaca-fvg-scanner.service >> "$REPO_DIR/executor.log" 2>&1
fi

rm -f "$REPO_DIR/executor.pid" "$REPO_DIR/fvg_scanner.pid"

"$VENV_PY" "$SCRIPT_DIR/fvg_pnl_report.py" 2>&1 | tee -a "$REPO_DIR/executor.log"

# Make sure nothing was left resting at the broker. The executor cancels
# a still-working entry order when it closes a position, but a process
# killed mid-session can leave a DAY limit order behind that could fill
# tomorrow with no stop attached.
#
# This is a standalone script file, NOT a stdin heredoc: python-dotenv's
# find_dotenv() walks the call stack and asserts on `frame.f_back`, which
# is None when the code is piped in via `python - <<PY`.
"$VENV_PY" "$SCRIPT_DIR/fvg_cleanup_check.py" 2>&1 | tee -a "$REPO_DIR/executor.log"
