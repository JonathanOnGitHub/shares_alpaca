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
    nohup "$VENV_PY" "$SCRIPT_DIR/run_engulfing_paper.py" >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    echo "$(date -Iseconds) — executor started PID $!" >> "$LOGFILE"
fi
