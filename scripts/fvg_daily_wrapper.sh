#!/bin/bash
# fvg_daily_wrapper.sh — run inside a cron no_agent job
# Generates signals, prints P&L report, then launches executor as a background daemon.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PY="$REPO_DIR/.venv/bin/python"

LOGFILE="$REPO_DIR/executor.log"
PIDFILE="$REPO_DIR/executor.pid"

cd "$REPO_DIR"

# ── 1. Generate signals ───────────────────────────────────────────────────────
echo "=== $(date -Iseconds) — FVG daily run ===" >> "$LOGFILE"
"$VENV_PY" "$SCRIPT_DIR/run_fvg_paper.py" --signals-only 2>&1 | tee -a "$LOGFILE"

# ── 2. P&L report ────────────────────────────────────────────────────────────
"$VENV_PY" "$SCRIPT_DIR/fvg_pnl_report.py" 2>&1 | tee -a "$LOGFILE"

# ── 3. Start executor daemon (background) ────────────────────────────────────
# If already running, skip; otherwise daemonise.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "$(date -Iseconds) — executor already running (PID $(cat "$PIDFILE"))" >> "$LOGFILE"
else
    nohup "$VENV_PY" "$SCRIPT_DIR/run_fvg_paper.py" >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    echo "$(date -Iseconds) — executor started PID $!" >> "$LOGFILE"
fi
